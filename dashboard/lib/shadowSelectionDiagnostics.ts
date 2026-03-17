/**
 * Shadow Selection Diagnostics — observability for entry decisions.
 * Compares exitrefine_v1 vs entrycal_v1 / entrycal_bind_v1 to see if entry calibration filters differently.
 * In-memory only; no persistence. No business logic changes.
 */

const EXITREFINE = "shadow_1000_adapt_captrade_exitrefine_v1";
const ENTRYCAL = "shadow_1000_adapt_captrade_exitrefine_entrycal_v1";
const ENTRYCAL_BIND = "shadow_1000_adapt_captrade_exitrefine_entrycal_bind_v1";
export const SELECTION_CYCLE_BUCKET_MS = 10_000;
const MAX_CYCLES_TO_KEEP = 1000;

interface DecisionRecord {
  accepted: boolean;
  rejectionReason?: string;
  capturableEdge?: number;
}

/** Per (opportunityId, cycleBucket): decisions from both profiles */
const decisionByOppCycle = new Map<string, Record<string, DecisionRecord>>();

/** Rolling window: only keep recent cycle buckets */
let oldestCycleBucket = 0;

function pruneOldCycles(currentBucket: number): void {
  if (oldestCycleBucket === 0) oldestCycleBucket = currentBucket;
  const cutoff = currentBucket - MAX_CYCLES_TO_KEEP;
  if (cutoff <= oldestCycleBucket) return;
  for (const key of Array.from(decisionByOppCycle.keys())) {
    const bucket = parseInt(key.split("|")[1] ?? "0", 10);
    if (bucket < cutoff) decisionByOppCycle.delete(key);
  }
  oldestCycleBucket = cutoff;
}

const TRACKED_PROFILES = new Set([EXITREFINE, ENTRYCAL, ENTRYCAL_BIND]);

export function recordEntryDecision(
  profileId: string,
  opportunityId: string,
  cycleBucket: number,
  accepted: boolean,
  rejectionReason?: string,
  capturableEdge?: number
): void {
  if (!TRACKED_PROFILES.has(profileId)) return;
  const key = `${opportunityId}|${cycleBucket}`;
  let rec = decisionByOppCycle.get(key);
  if (!rec) {
    rec = {};
    decisionByOppCycle.set(key, rec);
  }
  rec[profileId] = { accepted, rejectionReason, capturableEdge };
  pruneOldCycles(cycleBucket);
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const i = Math.max(0, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[i] ?? 0;
}

export interface ProfileSelectionStats {
  openedTradeCount: number;
  rejectedByEntryThresholdCount: number;
  avgCapturableEdgeOpened: number;
  minCapturableEdgeOpened: number;
  maxCapturableEdgeOpened: number;
  p10CapturableEdgeOpened: number;
  p50CapturableEdgeOpened: number;
  p90CapturableEdgeOpened: number;
  avgFillRatio: number;
}

export interface ExitrefineEntrycalComparison {
  acceptedByBoth: number;
  acceptedOnlyByExitrefine: number;
  acceptedOnlyByEntrycal: number;
  rejectedOnlyByEntrycalThreshold: number;
}

export interface SelectionDiagnosticsResult {
  byProfile: Record<string, ProfileSelectionStats>;
  exitrefineVsEntrycal: ExitrefineEntrycalComparison;
  exitrefineVsEntrycalBind: ExitrefineEntrycalComparison;
  generatedAt: string;
}

export function getSelectionDiagnostics(
  profiles: Array<{
    profileId: string;
    closedTrades: Array<{ capturableEdgeAtEntry?: number | null; fillRatio?: number | null }>;
    activeTrades: Array<{ capturableEdgeAtEntry?: number | null; fillRatio?: number | null }>;
  }>,
  rejectionCountsByProfile: Record<string, Record<string, number>>
): SelectionDiagnosticsResult {
  const byProfile: Record<string, ProfileSelectionStats> = {};
  const targetIds = [EXITREFINE, ENTRYCAL, ENTRYCAL_BIND];

  for (const profileId of targetIds) {
    const p = profiles.find((x) => x.profileId === profileId);
    const opened = [...(p?.closedTrades ?? []), ...(p?.activeTrades ?? [])];
    const edges = opened
      .map((t) => t.capturableEdgeAtEntry)
      .filter((e): e is number => typeof e === "number");
    const fillRatios = opened
      .map((t) => t.fillRatio)
      .filter((r): r is number => typeof r === "number" && r > 0);
    const rejectionCounts = rejectionCountsByProfile[profileId] ?? {};
    const rejectedByEntry = rejectionCounts["net_edge_below_threshold"] ?? 0;

    const sorted = [...edges].sort((a, b) => a - b);
    byProfile[profileId] = {
      openedTradeCount: opened.length,
      rejectedByEntryThresholdCount: rejectedByEntry,
      avgCapturableEdgeOpened: edges.length ? edges.reduce((a, b) => a + b, 0) / edges.length : 0,
      minCapturableEdgeOpened: edges.length ? Math.min(...edges) : 0,
      maxCapturableEdgeOpened: edges.length ? Math.max(...edges) : 0,
      p10CapturableEdgeOpened: percentile(sorted, 10),
      p50CapturableEdgeOpened: percentile(sorted, 50),
      p90CapturableEdgeOpened: percentile(sorted, 90),
      avgFillRatio: fillRatios.length ? fillRatios.reduce((a, b) => a + b, 0) / fillRatios.length : 0,
    };
  }

  let acceptedByBoth = 0;
  let acceptedOnlyByExitrefine = 0;
  let acceptedOnlyByEntrycal = 0;
  let rejectedOnlyByEntrycalThreshold = 0;

  let bindAcceptedByBoth = 0;
  let bindAcceptedOnlyByExitrefine = 0;
  let bindAcceptedOnlyByEntrycalBind = 0;
  let bindRejectedOnlyByEntrycalBindThreshold = 0;

  for (const rec of Array.from(decisionByOppCycle.values())) {
    const exitrec = rec[EXITREFINE];
    const entryrec = rec[ENTRYCAL];
    const bindrec = rec[ENTRYCAL_BIND];

    if (exitrec && entryrec) {
      const exitAccepted = exitrec.accepted;
      const entryAccepted = entryrec.accepted;
      const entryRejectedByThreshold =
        !entryAccepted && entryrec.rejectionReason === "net_edge_below_threshold";

      if (exitAccepted && entryAccepted) acceptedByBoth++;
      else if (exitAccepted && !entryAccepted) {
        acceptedOnlyByExitrefine++;
        if (entryRejectedByThreshold) rejectedOnlyByEntrycalThreshold++;
      } else if (!exitAccepted && entryAccepted) acceptedOnlyByEntrycal++;
    }

    if (exitrec && bindrec) {
      const exitAccepted = exitrec.accepted;
      const bindAccepted = bindrec.accepted;
      const bindRejectedByThreshold =
        !bindAccepted && bindrec.rejectionReason === "net_edge_below_threshold";

      if (exitAccepted && bindAccepted) bindAcceptedByBoth++;
      else if (exitAccepted && !bindAccepted) {
        bindAcceptedOnlyByExitrefine++;
        if (bindRejectedByThreshold) bindRejectedOnlyByEntrycalBindThreshold++;
      } else if (!exitAccepted && bindAccepted) bindAcceptedOnlyByEntrycalBind++;
    }
  }

  return {
    byProfile,
    exitrefineVsEntrycal: {
      acceptedByBoth,
      acceptedOnlyByExitrefine,
      acceptedOnlyByEntrycal,
      rejectedOnlyByEntrycalThreshold,
    },
    exitrefineVsEntrycalBind: {
      acceptedByBoth: bindAcceptedByBoth,
      acceptedOnlyByExitrefine: bindAcceptedOnlyByExitrefine,
      acceptedOnlyByEntrycal: bindAcceptedOnlyByEntrycalBind,
      rejectedOnlyByEntrycalThreshold: bindRejectedOnlyByEntrycalBindThreshold,
    },
    generatedAt: new Date().toISOString(),
  };
}
