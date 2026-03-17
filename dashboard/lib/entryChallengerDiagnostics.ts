/**
 * Entry Challenger Diagnostics — observabilidade para shadow_1000_entry_capfloor_v1 e shadow_1000_entry_degratio_v1.
 * In-memory only; no persistence.
 */

export const BASELINE_PROFILE_ID = "shadow_1000";
export const CAPFLOOR_PROFILE_ID = "shadow_1000_entry_capfloor_v1";
export const DEGRATIO_PROFILE_ID = "shadow_1000_entry_degratio_v1";

const SELECTION_CYCLE_BUCKET_MS = 10_000;
const MAX_CYCLES_TO_KEEP = 1000;

export interface EntryChallengerDecisionRecord {
  accepted: boolean;
  rejectionReason?: string;
  capturableEdge?: number;
  observedEdge?: number;
  fillRatio?: number;
  pairKey?: string;
  entryDegradationRatio?: number;
}

/** Per (opportunityId, cycleBucket): decisions from baseline + capfloor + degratio */
const decisionByOppCycle = new Map<string, Record<string, EntryChallengerDecisionRecord>>();

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

const TRACKED_PROFILES = new Set([BASELINE_PROFILE_ID, CAPFLOOR_PROFILE_ID, DEGRATIO_PROFILE_ID]);

export function recordEntryChallengerDecision(
  profileId: string,
  opportunityId: string,
  cycleBucket: number,
  accepted: boolean,
  rejectionReason?: string,
  capturableEdge?: number,
  observedEdge?: number,
  fillRatio?: number,
  pairKey?: string,
  entryDegradationRatio?: number
): void {
  if (!TRACKED_PROFILES.has(profileId)) return;
  const key = `${opportunityId}|${cycleBucket}`;
  let rec = decisionByOppCycle.get(key);
  if (!rec) {
    rec = {};
    decisionByOppCycle.set(key, rec);
  }
  rec[profileId] = {
    accepted,
    rejectionReason,
    capturableEdge,
    observedEdge,
    fillRatio,
    pairKey,
    entryDegradationRatio,
  };
  pruneOldCycles(cycleBucket);
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const i = Math.max(0, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[i] ?? 0;
}

export interface EntryChallengerDiagnosticsBlock {
  profileId: string;
  evaluatedOpportunityCount: number;
  openedTradeCount: number;
  rejectedByCapturableFloorCount: number;
  rejectedByDegRatioCount: number;
  rejectedByOtherReasonCount: number;
  avgCapturableEdgeOpened: number;
  minCapturableEdgeOpened: number;
  p10CapturableEdgeOpened: number;
  p50CapturableEdgeOpened: number;
  p90CapturableEdgeOpened: number;
  avgObservedEdgeOpened: number;
  avgDegradationRatioOpened: number;
  minDegradationRatioOpened: number;
  p10DegradationRatioOpened: number;
  p50DegradationRatioOpened: number;
  p90DegradationRatioOpened: number;
}

export interface EntryChallengerComparisonBlock {
  challengerProfileId: string;
  acceptedByBoth: number;
  acceptedOnlyByBaseline: number;
  acceptedOnlyByChallenger: number;
  rejectedOnlyByChallengerRule: number;
  sameDecisionRate: number;
  sampleRejectedByCapfloor?: Array<{
    pairKey: string;
    observedEdgeAtEntry: number;
    capturableEdgeAtEntry: number;
    entryDegradationRatio: number;
    fillRatio: number;
    baselineDecision: string;
    challengerDecision: string;
  }>;
  sampleRejectedByDegratio?: Array<{
    pairKey: string;
    observedEdgeAtEntry: number;
    capturableEdgeAtEntry: number;
    entryDegradationRatio: number;
    fillRatio: number;
    baselineDecision: string;
    challengerDecision: string;
  }>;
}

export interface FilteredOpportunityCounterfactualBlock {
  challengerProfileId: string;
  countFilteredOutThatBaselineWouldTrade: number;
  avgRealizedPnL_of_filtered_if_baseline_traded: number;
  medianRealizedPnL_of_filtered_if_baseline_traded: number;
  totalRealizedPnL_avoided: number;
  worst10FilteredAvoided: Array<{ pairKey: string; realizedPnL: number }>;
  best10FilteredAvoided: Array<{ pairKey: string; realizedPnL: number }>;
}

export interface EntryChallengerMetricsSummary {
  challengerProfileId: string;
  filterRate: number;
  lossReductionVsBaseline: number;
  lossReductionPerTradeFiltered: number;
}

export function getEntryChallengerDiagnostics(
  profiles: Array<{
    profileId: string;
    closedTrades: Array<{
      capturableEdgeAtEntry?: number | null;
      observedEdgeAtEntry?: number | null;
      fillRatio?: number | null;
      pairKey?: string | null;
    }>;
    activeTrades: Array<{
      capturableEdgeAtEntry?: number | null;
      observedEdgeAtEntry?: number | null;
      fillRatio?: number | null;
      pairKey?: string | null;
    }>;
  }>,
  rejectionCountsByProfile: Record<string, Record<string, number>>
): Record<string, EntryChallengerDiagnosticsBlock> {
  const out: Record<string, EntryChallengerDiagnosticsBlock> = {};

  for (const challengerId of [CAPFLOOR_PROFILE_ID, DEGRATIO_PROFILE_ID]) {
    const p = profiles.find((x) => x.profileId === challengerId);
    const opened = [...(p?.closedTrades ?? []), ...(p?.activeTrades ?? [])];
    const rejectionCounts = rejectionCountsByProfile[challengerId] ?? {};
    const rejectedByCapfloor = rejectionCounts["entry_capfloor"] ?? 0;
    const rejectedByDegratio = rejectionCounts["entry_degratio"] ?? 0;
    const otherReasons = Object.entries(rejectionCounts)
      .filter(([k]) => k !== "entry_capfloor" && k !== "entry_degratio")
      .reduce((s, [, v]) => s + v, 0);

    const capturableEdges = opened
      .map((t) => t.capturableEdgeAtEntry)
      .filter((e): e is number => typeof e === "number");
    const observedEdges = opened
      .map((t) => t.observedEdgeAtEntry)
      .filter((e): e is number => typeof e === "number");
    const degRatios = opened.map((t) => {
      const cap = t.capturableEdgeAtEntry ?? 0;
      const obs = t.observedEdgeAtEntry ?? 0;
      return obs > 0.0001 ? cap / obs : cap / 0.0001;
    }).filter((r): r is number => typeof r === "number" && !Number.isNaN(r));

    const sortedCap = [...capturableEdges].sort((a, b) => a - b);
    const sortedDeg = [...degRatios].sort((a, b) => a - b);

    const evaluated = opened.length + rejectedByCapfloor + rejectedByDegratio + otherReasons;
    out[challengerId] = {
      profileId: challengerId,
      evaluatedOpportunityCount: evaluated || opened.length,
      openedTradeCount: opened.length,
      rejectedByCapturableFloorCount: rejectedByCapfloor,
      rejectedByDegRatioCount: rejectedByDegratio,
      rejectedByOtherReasonCount: otherReasons,
      avgCapturableEdgeOpened: capturableEdges.length ? capturableEdges.reduce((a, b) => a + b, 0) / capturableEdges.length : 0,
      minCapturableEdgeOpened: capturableEdges.length ? Math.min(...capturableEdges) : 0,
      p10CapturableEdgeOpened: percentile(sortedCap, 10),
      p50CapturableEdgeOpened: percentile(sortedCap, 50),
      p90CapturableEdgeOpened: percentile(sortedCap, 90),
      avgObservedEdgeOpened: observedEdges.length ? observedEdges.reduce((a, b) => a + b, 0) / observedEdges.length : 0,
      avgDegradationRatioOpened: degRatios.length ? degRatios.reduce((a, b) => a + b, 0) / degRatios.length : 0,
      minDegradationRatioOpened: degRatios.length ? Math.min(...degRatios) : 0,
      p10DegradationRatioOpened: percentile(sortedDeg, 10),
      p50DegradationRatioOpened: percentile(sortedDeg, 50),
      p90DegradationRatioOpened: percentile(sortedDeg, 90),
    };
  }

  return out;
}

export function getEntryChallengerComparisonDiagnostics(): Record<string, EntryChallengerComparisonBlock> {
  const capfloor: EntryChallengerComparisonBlock = {
    challengerProfileId: CAPFLOOR_PROFILE_ID,
    acceptedByBoth: 0,
    acceptedOnlyByBaseline: 0,
    acceptedOnlyByChallenger: 0,
    rejectedOnlyByChallengerRule: 0,
    sameDecisionRate: 0,
    sampleRejectedByCapfloor: [],
  };

  const degratio: EntryChallengerComparisonBlock = {
    challengerProfileId: DEGRATIO_PROFILE_ID,
    acceptedByBoth: 0,
    acceptedOnlyByBaseline: 0,
    acceptedOnlyByChallenger: 0,
    rejectedOnlyByChallengerRule: 0,
    sameDecisionRate: 0,
    sampleRejectedByDegratio: [],
  };

  const all = Array.from(decisionByOppCycle.values());
  let capfloorTotal = 0;
  let capfloorSame = 0;
  let degratioTotal = 0;
  let degratioSame = 0;

  const capfloorSamples: EntryChallengerComparisonBlock["sampleRejectedByCapfloor"] = [];
  const degratioSamples: EntryChallengerComparisonBlock["sampleRejectedByDegratio"] = [];

  for (const rec of all) {
    const baseline = rec[BASELINE_PROFILE_ID];
    const cap = rec[CAPFLOOR_PROFILE_ID];
    const deg = rec[DEGRATIO_PROFILE_ID];

    if (baseline && cap) {
      capfloorTotal++;
      const baseAcc = baseline.accepted;
      const capAcc = cap.accepted;
      if (baseAcc === capAcc) capfloorSame++;
      if (baseAcc && !capAcc) {
        capfloor.acceptedOnlyByBaseline++;
        if (cap.rejectionReason === "entry_capfloor") {
          capfloor.rejectedOnlyByChallengerRule++;
          if (capfloorSamples.length < 10) {
            capfloorSamples.push({
              pairKey: cap.pairKey ?? "",
              observedEdgeAtEntry: cap.observedEdge ?? 0,
              capturableEdgeAtEntry: cap.capturableEdge ?? 0,
              entryDegradationRatio: cap.entryDegradationRatio ?? 0,
              fillRatio: cap.fillRatio ?? 0,
              baselineDecision: "accepted",
              challengerDecision: "rejected",
            });
          }
        }
      } else if (!baseAcc && capAcc) capfloor.acceptedOnlyByChallenger++;
      else if (baseAcc && capAcc) capfloor.acceptedByBoth++;
    }

    if (baseline && deg) {
      degratioTotal++;
      const baseAcc = baseline.accepted;
      const degAcc = deg.accepted;
      if (baseAcc === degAcc) degratioSame++;
      if (baseAcc && !degAcc) {
        degratio.acceptedOnlyByBaseline++;
        if (deg.rejectionReason === "entry_degratio") {
          degratio.rejectedOnlyByChallengerRule++;
          if (degratioSamples.length < 10) {
            degratioSamples.push({
              pairKey: deg.pairKey ?? "",
              observedEdgeAtEntry: deg.observedEdge ?? 0,
              capturableEdgeAtEntry: deg.capturableEdge ?? 0,
              entryDegradationRatio: deg.entryDegradationRatio ?? 0,
              fillRatio: deg.fillRatio ?? 0,
              baselineDecision: "accepted",
              challengerDecision: "rejected",
            });
          }
        }
      } else if (!baseAcc && degAcc) degratio.acceptedOnlyByChallenger++;
      else if (baseAcc && degAcc) degratio.acceptedByBoth++;
    }
  }

  capfloor.sameDecisionRate = capfloorTotal > 0 ? capfloorSame / capfloorTotal : 0;
  capfloor.sampleRejectedByCapfloor = capfloorSamples;
  degratio.sameDecisionRate = degratioTotal > 0 ? degratioSame / degratioTotal : 0;
  degratio.sampleRejectedByDegratio = degratioSamples;

  return { [CAPFLOOR_PROFILE_ID]: capfloor, [DEGRATIO_PROFILE_ID]: degratio };
}

export function getFilteredOpportunityCounterfactual(
  baselineClosedTrades: Array<{
    opportunityId: string;
    realizedPnL: number;
    pairKey?: string | null;
    capfloorFilteredSameCycle?: boolean;
    degratioFilteredSameCycle?: boolean;
  }>
): Record<string, FilteredOpportunityCounterfactualBlock> {
  const capfloorFiltered = baselineClosedTrades.filter((t) => t.capfloorFilteredSameCycle);
  const degratioFiltered = baselineClosedTrades.filter((t) => t.degratioFilteredSameCycle);

  function buildBlock(
    trades: typeof baselineClosedTrades,
    challengerId: string
  ): FilteredOpportunityCounterfactualBlock {
    const pnls = trades.map((t) => t.realizedPnL);
    const avg = pnls.length ? pnls.reduce((a, b) => a + b, 0) / pnls.length : 0;
    const sorted = [...pnls].sort((a, b) => a - b);
    const median = sorted.length
      ? sorted.length % 2 ? sorted[Math.floor(sorted.length / 2)]! : (sorted[sorted.length / 2 - 1]! + sorted[sorted.length / 2]!) / 2
      : 0;
    const total = pnls.reduce((a, b) => a + b, 0);
    const byPair = trades.map((t) => ({ pairKey: t.pairKey ?? "", realizedPnL: t.realizedPnL }));
    const worst10 = [...byPair].sort((a, b) => a.realizedPnL - b.realizedPnL).slice(0, 10);
    const best10 = [...byPair].sort((a, b) => b.realizedPnL - a.realizedPnL).slice(0, 10);
    return {
      challengerProfileId: challengerId,
      countFilteredOutThatBaselineWouldTrade: trades.length,
      avgRealizedPnL_of_filtered_if_baseline_traded: avg,
      medianRealizedPnL_of_filtered_if_baseline_traded: median,
      totalRealizedPnL_avoided: total,
      worst10FilteredAvoided: worst10,
      best10FilteredAvoided: best10,
    };
  }

  return {
    [CAPFLOOR_PROFILE_ID]: buildBlock(capfloorFiltered, CAPFLOOR_PROFILE_ID),
    [DEGRATIO_PROFILE_ID]: buildBlock(degratioFiltered, DEGRATIO_PROFILE_ID),
  };
}

export function getEntryChallengerMetricsSummary(
  counterfactual: Record<string, FilteredOpportunityCounterfactualBlock>,
  baselineTotalPnL: number,
  entryDiagnostics: Record<string, EntryChallengerDiagnosticsBlock>
): Record<string, EntryChallengerMetricsSummary> {
  const out: Record<string, EntryChallengerMetricsSummary> = {};
  for (const challengerId of [CAPFLOOR_PROFILE_ID, DEGRATIO_PROFILE_ID]) {
    const cf = counterfactual[challengerId];
    const diag = entryDiagnostics[challengerId];
    const evaluated = diag?.evaluatedOpportunityCount ?? 1;
    const opened = diag?.openedTradeCount ?? 0;
    const filtered = cf?.countFilteredOutThatBaselineWouldTrade ?? 0;
    const totalAvoided = cf?.totalRealizedPnL_avoided ?? 0;
    const filterRate = evaluated > 0 ? (evaluated - opened) / evaluated : 0;
    const lossReductionVsBaseline = totalAvoided;
    const lossReductionPerTradeFiltered = filtered > 0 ? totalAvoided / filtered : 0;
    out[challengerId] = {
      challengerProfileId: challengerId,
      filterRate,
      lossReductionVsBaseline,
      lossReductionPerTradeFiltered,
    };
  }
  return out;
}
