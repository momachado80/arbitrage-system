import type { GraphOpportunity, GraphOpportunityType } from "./graphArbitrageEngine";

export interface EdgeSnapshot {
  edge: number;
  confidence: number;
  liquidity: number;
  ts: number;
}

export interface GraphEpisode {
  episodeId: string;
  key: string;
  type: GraphOpportunityType;
  clusterId: string;
  marketsInvolved: Array<{ marketId: string; question: string }>;
  startedAt: string;
  lastSeenAt: string;
  endedAt: string | null;
  durationMs: number;
  status: "active" | "closed";
  observationCount: number;
  maxEdge: number;
  minEdge: number;
  avgEdge: number;
  currentEdge: number;
  maxConfidence: number;
  avgConfidence: number;
  maxLiquidity: number;
  avgLiquidity: number;
  edgeTrend: "rising" | "stable" | "falling";
  firstSnapshot: EdgeSnapshot;
  lastSnapshot: EdgeSnapshot;
  recentSamples?: EdgeSnapshot[];
  missCount?: number;
  recoveredFromDisk?: boolean;
  persistenceVersion?: string;
}

const GRACE_PERIOD_MS = 12_000;
const MAX_SAMPLES = 20;
const PERSISTENCE_VERSION = "1";

let nextEpisodeSeq = 0;

function normalizeLabel(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 80);
}

function buildOpportunitySignature(opp: GraphOpportunity): string {
  const labels = opp.marketsInvolved
    .map((m) => normalizeLabel(m.question))
    .sort()
    .join("~");
  return `${opp.type}::${labels}`;
}

function buildFallbackSignature(opp: GraphOpportunity): string {
  const ids = opp.marketsInvolved
    .map((m) => m.marketId)
    .sort()
    .join("|");
  return `${opp.type}::${opp.clusterId}::${ids}`;
}

function makeKey(opp: GraphOpportunity): string {
  const primary = buildOpportunitySignature(opp);
  if (primary.length > 10) return primary;
  return buildFallbackSignature(opp);
}

function snap(opp: GraphOpportunity, ts: number): EdgeSnapshot {
  return { edge: opp.edge, confidence: opp.confidence, liquidity: opp.liquidity, ts };
}

function pushSample(ep: GraphEpisode, s: EdgeSnapshot): void {
  if (!ep.recentSamples) ep.recentSamples = [];
  ep.recentSamples.push(s);
  if (ep.recentSamples.length > MAX_SAMPLES) {
    ep.recentSamples.shift();
  }
}

function computeTrend(ep: GraphEpisode): GraphEpisode["edgeTrend"] {
  const samples = ep.recentSamples;
  if (!samples || samples.length < 3) return "stable";

  const recent = samples.slice(-5);
  const older = samples.slice(-10, -5);
  if (older.length === 0) {
    const diff = recent[recent.length - 1].edge - recent[0].edge;
    const threshold = Math.max(0.005, ep.avgEdge * 0.1);
    if (diff > threshold) return "rising";
    if (diff < -threshold) return "falling";
    return "stable";
  }

  const avgRecent = recent.reduce((s, x) => s + x.edge, 0) / recent.length;
  const avgOlder = older.reduce((s, x) => s + x.edge, 0) / older.length;
  const threshold = Math.max(0.005, ep.avgEdge * 0.1);
  if (avgRecent - avgOlder > threshold) return "rising";
  if (avgOlder - avgRecent > threshold) return "falling";
  return "stable";
}

export class GraphEpisodeTracker {
  private active = new Map<string, GraphEpisode>();
  private missTimestamps = new Map<string, number>();
  private onClose: ((ep: GraphEpisode) => void) | null = null;
  private gracePeriodMs: number;

  constructor(opts?: {
    onClose?: (ep: GraphEpisode) => void;
    gracePeriodMs?: number;
  }) {
    this.onClose = opts?.onClose ?? null;
    this.gracePeriodMs = opts?.gracePeriodMs ?? GRACE_PERIOD_MS;
  }

  restoreEpisodes(episodes: GraphEpisode[]): number {
    let restored = 0;
    for (const ep of episodes) {
      try {
        if (!ep.key || !ep.episodeId) continue;
        if (this.active.has(ep.key)) continue;

        ep.recoveredFromDisk = true;
        ep.persistenceVersion = PERSISTENCE_VERSION;
        ep.status = "active";
        ep.endedAt = null;
        this.active.set(ep.key, ep);
        restored++;

        const seq = parseInt(ep.episodeId.split("-")[1] || "0", 10);
        if (seq >= nextEpisodeSeq) nextEpisodeSeq = seq;
      } catch {
        // skip corrupted episode
      }
    }
    if (restored > 0) {
      console.log(`[GraphEpisode] Restored ${restored} episodes from disk`);
    }
    return restored;
  }

  update(opportunities: GraphOpportunity[]): void {
    const now = Date.now();
    const nowIso = new Date(now).toISOString();
    const seenKeys = new Set<string>();

    for (const opp of opportunities) {
      try {
        const key = makeKey(opp);
        seenKeys.add(key);
        this.missTimestamps.delete(key);

        const existing = this.active.get(key);
        if (existing) {
          if (existing.recoveredFromDisk) {
            console.log(`[GraphEpisode] RESUMED recovered ${existing.episodeId}`);
            existing.recoveredFromDisk = false;
          }
          this.updateEpisode(existing, opp, now, nowIso);
        } else {
          this.openEpisode(key, opp, now, nowIso);
        }
      } catch {
        // skip malformed
      }
    }

    const toClose: string[] = [];
    this.active.forEach((ep, key) => {
      if (seenKeys.has(key)) return;

      if (!this.missTimestamps.has(key)) {
        this.missTimestamps.set(key, now);
      }
      const missStart = this.missTimestamps.get(key)!;
      ep.missCount = (ep.missCount ?? 0) + 1;

      if (now - missStart >= this.gracePeriodMs) {
        toClose.push(key);
      }
    });

    for (const key of toClose) {
      const ep = this.active.get(key);
      if (ep) {
        this.closeEpisode(ep, now, nowIso);
        this.active.delete(key);
        this.missTimestamps.delete(key);
      }
    }
  }

  private openEpisode(
    key: string,
    opp: GraphOpportunity,
    now: number,
    nowIso: string
  ): void {
    const s = snap(opp, now);
    const ep: GraphEpisode = {
      episodeId: `gep-${++nextEpisodeSeq}-${now}`,
      key,
      type: opp.type,
      clusterId: opp.clusterId,
      marketsInvolved: opp.marketsInvolved.map((m) => ({
        marketId: m.marketId,
        question: m.question,
      })),
      startedAt: nowIso,
      lastSeenAt: nowIso,
      endedAt: null,
      durationMs: 0,
      status: "active",
      observationCount: 1,
      maxEdge: opp.edge,
      minEdge: opp.edge,
      avgEdge: opp.edge,
      currentEdge: opp.edge,
      maxConfidence: opp.confidence,
      avgConfidence: opp.confidence,
      maxLiquidity: opp.liquidity,
      avgLiquidity: opp.liquidity,
      edgeTrend: "stable",
      firstSnapshot: s,
      lastSnapshot: s,
      recentSamples: [s],
      missCount: 0,
      recoveredFromDisk: false,
      persistenceVersion: PERSISTENCE_VERSION,
    };
    this.active.set(key, ep);
    console.log(
      `[GraphEpisode] OPEN ${ep.episodeId} type=${opp.type} edge=${opp.edge.toFixed(4)} markets=${opp.marketsInvolved.length}`
    );
  }

  private updateEpisode(
    ep: GraphEpisode,
    opp: GraphOpportunity,
    now: number,
    nowIso: string
  ): void {
    const n = ep.observationCount;
    ep.observationCount = n + 1;
    ep.lastSeenAt = nowIso;
    ep.durationMs = now - new Date(ep.startedAt).getTime();
    ep.currentEdge = opp.edge;
    ep.maxEdge = Math.max(ep.maxEdge, opp.edge);
    ep.minEdge = Math.min(ep.minEdge, opp.edge);
    ep.avgEdge = (ep.avgEdge * n + opp.edge) / (n + 1);
    ep.maxConfidence = Math.max(ep.maxConfidence, opp.confidence);
    ep.avgConfidence = (ep.avgConfidence * n + opp.confidence) / (n + 1);
    ep.maxLiquidity = Math.max(ep.maxLiquidity, opp.liquidity);
    ep.avgLiquidity = (ep.avgLiquidity * n + opp.liquidity) / (n + 1);
    ep.missCount = 0;

    const s = snap(opp, now);
    pushSample(ep, s);
    ep.lastSnapshot = s;
    ep.edgeTrend = computeTrend(ep);
  }

  private closeEpisode(ep: GraphEpisode, now: number, nowIso: string): void {
    ep.status = "closed";
    ep.endedAt = nowIso;
    ep.durationMs = now - new Date(ep.startedAt).getTime();

    if (ep.recentSamples && ep.recentSamples.length > 5) {
      ep.recentSamples = ep.recentSamples.slice(-5);
    }

    const wasRecovered = ep.recoveredFromDisk;
    if (ep.durationMs > 120_000 || wasRecovered) {
      console.log(
        `[GraphEpisode] CLOSE ${wasRecovered ? "(recovered-expired) " : "(long) "}${ep.episodeId} duration=${(ep.durationMs / 1000).toFixed(0)}s avgEdge=${ep.avgEdge.toFixed(4)}`
      );
    }

    if (this.onClose) {
      try { this.onClose(ep); } catch { /* non-fatal */ }
    }
  }

  getActive(): GraphEpisode[] {
    return Array.from(this.active.values());
  }

  getActiveCount(): number {
    return this.active.size;
  }

  getActiveMap(): Map<string, GraphEpisode> {
    return this.active;
  }
}
