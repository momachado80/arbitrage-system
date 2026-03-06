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
}

const GRACE_PERIOD_SCANS = 2;

let nextEpisodeSeq = 0;

function makeKey(opp: GraphOpportunity): string {
  const marketIds = opp.marketsInvolved
    .map((m) => m.marketId)
    .sort()
    .join("|");
  return `${opp.type}::${opp.clusterId}::${marketIds}`;
}

function snapshot(opp: GraphOpportunity, ts: number): EdgeSnapshot {
  return {
    edge: opp.edge,
    confidence: opp.confidence,
    liquidity: opp.liquidity,
    ts,
  };
}

function computeTrend(ep: GraphEpisode, currentEdge: number): GraphEpisode["edgeTrend"] {
  if (ep.observationCount < 3) return "stable";
  const diff = currentEdge - ep.avgEdge;
  const threshold = Math.max(0.005, ep.avgEdge * 0.1);
  if (diff > threshold) return "rising";
  if (diff < -threshold) return "falling";
  return "stable";
}

export class GraphEpisodeTracker {
  private active = new Map<string, GraphEpisode>();
  private missCount = new Map<string, number>();
  private onClose: ((ep: GraphEpisode) => void) | null = null;

  constructor(opts?: { onClose?: (ep: GraphEpisode) => void }) {
    this.onClose = opts?.onClose ?? null;
  }

  update(opportunities: GraphOpportunity[]): void {
    const now = Date.now();
    const nowIso = new Date(now).toISOString();
    const seenKeys = new Set<string>();

    for (const opp of opportunities) {
      try {
        const key = makeKey(opp);
        seenKeys.add(key);
        this.missCount.delete(key);

        const existing = this.active.get(key);
        if (existing) {
          this.updateEpisode(existing, opp, now, nowIso);
        } else {
          this.openEpisode(key, opp, now, nowIso);
        }
      } catch {
        // skip malformed opportunity
      }
    }

    const toClose: string[] = [];
    this.active.forEach((ep, key) => {
      if (seenKeys.has(key)) return;
      const misses = (this.missCount.get(key) ?? 0) + 1;
      this.missCount.set(key, misses);
      if (misses > GRACE_PERIOD_SCANS) {
        toClose.push(key);
      }
    });

    for (const key of toClose) {
      const ep = this.active.get(key);
      if (ep) {
        this.closeEpisode(ep, now, nowIso);
        this.active.delete(key);
        this.missCount.delete(key);
      }
    }
  }

  private openEpisode(
    key: string,
    opp: GraphOpportunity,
    now: number,
    nowIso: string
  ): void {
    const snap = snapshot(opp, now);
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
      firstSnapshot: snap,
      lastSnapshot: snap,
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
    ep.edgeTrend = computeTrend(ep, opp.edge);
    ep.lastSnapshot = snapshot(opp, now);
  }

  private closeEpisode(ep: GraphEpisode, now: number, nowIso: string): void {
    ep.status = "closed";
    ep.endedAt = nowIso;
    ep.durationMs = now - new Date(ep.startedAt).getTime();

    if (ep.durationMs > 120_000) {
      console.log(
        `[GraphEpisode] CLOSE (long) ${ep.episodeId} duration=${(ep.durationMs / 1000).toFixed(0)}s avgEdge=${ep.avgEdge.toFixed(4)}`
      );
    }

    if (this.onClose) {
      try {
        this.onClose(ep);
      } catch {
        // non-fatal
      }
    }
  }

  getActive(): GraphEpisode[] {
    return Array.from(this.active.values());
  }

  getActiveCount(): number {
    return this.active.size;
  }
}
