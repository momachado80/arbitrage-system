/**
 * Structural Risk-Managed Challenger Diagnostics — shadow_1000_structural_riskmanaged_v1.
 * In-memory only; no persistence.
 */

import type { ShadowProfileState } from "./shadowSimulationStore";
import type { ClosedTradeAuditEntry } from "./shadowClosedTradeAudit";

const PROFILE_ID = "shadow_1000_structural_riskmanaged_v1";

interface ProfileCounters {
  evaluatedOpportunityCount: number;
  rejectedByStructuralPairCount: number;
  rejectedByFillBucketCount: number;
  rejectedByCapfloorCount: number;
  rejectedByDegRatioCount: number;
  rejectedByOtherReasonCount: number;
}

const counters: ProfileCounters = {
  evaluatedOpportunityCount: 0,
  rejectedByStructuralPairCount: 0,
  rejectedByFillBucketCount: 0,
  rejectedByCapfloorCount: 0,
  rejectedByDegRatioCount: 0,
  rejectedByOtherReasonCount: 0,
};

export function recordStructuralRiskEvaluated(): void {
  counters.evaluatedOpportunityCount++;
}

export function recordStructuralRiskRejectedByPair(): void {
  counters.rejectedByStructuralPairCount++;
}

export function recordStructuralRiskRejectedByFillBucket(): void {
  counters.rejectedByFillBucketCount++;
}

export function recordStructuralRiskRejectedByCapfloor(): void {
  counters.rejectedByCapfloorCount++;
}

export function recordStructuralRiskRejectedByDegRatio(): void {
  counters.rejectedByDegRatioCount++;
}

export function recordStructuralRiskRejectedByOther(): void {
  counters.rejectedByOtherReasonCount++;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const i = Math.max(0, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[i] ?? 0;
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const m = Math.floor(arr.length / 2);
  return arr.length % 2 ? sorted[m]! : (sorted[m - 1]! + sorted[m]!) / 2;
}

export interface StructuralRiskManagedDiagnosticsBlock {
  profileId: string;
  evaluatedOpportunityCount: number;
  openedTradeCount: number;
  rejectedByStructuralPairCount: number;
  rejectedByFillBucketCount: number;
  rejectedByCapfloorCount: number;
  rejectedByDegRatioCount: number;
  rejectedByOtherReasonCount: number;
  avgCapturableEdgeOpened: number;
  avgObservedEdgeOpened: number;
  avgDegradationRatioOpened: number;
  avgCapitalMultiplierOpened: number;
  minCapitalMultiplierOpened: number;
  p10CapitalMultiplierOpened: number;
  p50CapitalMultiplierOpened: number;
  p90CapitalMultiplierOpened: number;
  pmaxCapitalMultiplierOpened: number;
  earlyThesisFailureExitCount: number;
  avgHoldingMsEarlyThesisFailure: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  totalRealizedPnL: number;
  avgFillRatioOpened: number;
  avgRealizedPnL_earlyExit: number;
  countEarlyExit: number;
}

export interface StructuralRiskManagedComparisonBlock {
  baselineProfileId: string;
  challengerProfileId: string;
  baselineClosed: number;
  challengerClosed: number;
  baselineAvgRealizedPnL: number;
  challengerAvgRealizedPnL: number;
  baselineMedianRealizedPnL: number;
  challengerMedianRealizedPnL: number;
  baselineTotalRealizedPnL: number;
  challengerTotalRealizedPnL: number;
  baselineAvgFillRatio: number;
  challengerAvgFillRatio: number;
  baselineAvgCapturableEdgeAtEntry: number;
  challengerAvgCapturableEdgeAtEntry: number;
  sameUniverseNote: string;
}

export function getStructuralRiskManagedDiagnostics(
  profile: ShadowProfileState | undefined,
  allAuditEntries: ClosedTradeAuditEntry[],
  rejectionCountsByProfile: Record<string, Record<string, number>>
): StructuralRiskManagedDiagnosticsBlock {
  const closed = (profile?.closedTrades ?? []).filter(
    (t) => t.status === "closed" && t.closedAt && t.structuralRiskFilterMatchAtOpen !== false
  );
  const openedCount = closed.length + (profile?.activeTrades?.length ?? 0);
  const rejectionCounts = rejectionCountsByProfile[PROFILE_ID] ?? {};

  const rejectedByPair = rejectionCounts["structural_risk_pair_mismatch"] ?? counters.rejectedByStructuralPairCount;
  const rejectedByFill = rejectionCounts["structural_risk_fill_bucket_mismatch"] ?? counters.rejectedByFillBucketCount;
  const rejectedByCap = rejectionCounts["structural_risk_capfloor"] ?? counters.rejectedByCapfloorCount;
  const rejectedByDeg = rejectionCounts["structural_risk_degratio"] ?? counters.rejectedByDegRatioCount;

  const capturableEdges = closed.map((t) => t.capturableEdgeAtEntry ?? 0);
  const observedEdges = closed.map((t) => t.observedEdgeAtEntry ?? 0);
  const degRatios = closed.map((t) => {
    const cap = t.capturableEdgeAtEntry ?? 0;
    const obs = t.observedEdgeAtEntry ?? 0;
    return obs > 0.0001 ? cap / obs : 0;
  });
  const multipliers = closed
    .map((t) => (t as { structuralRiskCapitalMultiplierAtOpen?: number }).structuralRiskCapitalMultiplierAtOpen)
    .filter((m): m is number => typeof m === "number");
  const sortedMult = [...multipliers].sort((a, b) => a - b);

  const earlyExits = closed.filter((t) => (t as { earlyThesisFailureTriggered?: boolean }).earlyThesisFailureTriggered === true);
  const holdingMsEarly = earlyExits.map((t) => t.holdingTimeMs ?? 0);
  const pnlsEarly = earlyExits.map((t) => t.realizedPnL ?? 0);
  const pnls = closed.map((t) => t.realizedPnL ?? 0);
  const fillRatios = closed.map((t) =>
    t.fillRatio != null && typeof t.fillRatio === "number"
      ? t.fillRatio
      : t.requestedCapital != null && t.requestedCapital > 0
        ? (t.filledCapital ?? 0) / t.requestedCapital
        : 0
  );
  const avgFill = fillRatios.length ? fillRatios.reduce((a, b) => a + b, 0) / fillRatios.length : 0;

  return {
    profileId: PROFILE_ID,
    evaluatedOpportunityCount: counters.evaluatedOpportunityCount,
    openedTradeCount: openedCount,
    rejectedByStructuralPairCount: rejectedByPair,
    rejectedByFillBucketCount: rejectedByFill,
    rejectedByCapfloorCount: rejectedByCap,
    rejectedByDegRatioCount: rejectedByDeg,
    rejectedByOtherReasonCount: counters.rejectedByOtherReasonCount,
    avgCapturableEdgeOpened: capturableEdges.length ? capturableEdges.reduce((a, b) => a + b, 0) / capturableEdges.length : 0,
    avgObservedEdgeOpened: observedEdges.length ? observedEdges.reduce((a, b) => a + b, 0) / observedEdges.length : 0,
    avgDegradationRatioOpened: degRatios.length ? degRatios.reduce((a, b) => a + b, 0) / degRatios.length : 0,
    avgCapitalMultiplierOpened: multipliers.length ? multipliers.reduce((a, b) => a + b, 0) / multipliers.length : 0,
    minCapitalMultiplierOpened: multipliers.length ? Math.min(...multipliers) : 0,
    p10CapitalMultiplierOpened: percentile(sortedMult, 10),
    p50CapitalMultiplierOpened: percentile(sortedMult, 50),
    p90CapitalMultiplierOpened: percentile(sortedMult, 90),
    pmaxCapitalMultiplierOpened: multipliers.length ? Math.max(...multipliers) : 0,
    earlyThesisFailureExitCount: earlyExits.length,
    avgHoldingMsEarlyThesisFailure:
      holdingMsEarly.length ? holdingMsEarly.reduce((a, b) => a + b, 0) / holdingMsEarly.length : 0,
    avgRealizedPnL: pnls.length ? pnls.reduce((a, b) => a + b, 0) / pnls.length : 0,
    medianRealizedPnL: median(pnls),
    totalRealizedPnL: pnls.reduce((a, b) => a + b, 0),
    avgFillRatioOpened: avgFill,
    avgRealizedPnL_earlyExit: pnlsEarly.length ? pnlsEarly.reduce((a, b) => a + b, 0) / pnlsEarly.length : 0,
    countEarlyExit: earlyExits.length,
  };
}

export function getStructuralRiskManagedComparison(
  profiles: ShadowProfileState[],
  diagnostics: StructuralRiskManagedDiagnosticsBlock,
  compareProfileId: string
): StructuralRiskManagedComparisonBlock {
  const baseline = profiles.find((p) => p.profileId === compareProfileId);
  const challenger = profiles.find((p) => p.profileId === PROFILE_ID);

  const baselineClosed = baseline?.closedTrades ?? [];
  const challengerClosed = challenger?.closedTrades?.filter((t) => t.structuralRiskFilterMatchAtOpen !== false) ?? [];

  const baselineAvgPnL =
    baselineClosed.length > 0
      ? baselineClosed.reduce((s, t) => s + (t.realizedPnL ?? 0), 0) / baselineClosed.length
      : 0;
  const baselineTotal = baselineClosed.reduce((s, t) => s + (t.realizedPnL ?? 0), 0);
  const baselineAvgFill =
    baselineClosed.length > 0
      ? baselineClosed.reduce((s, t) => s + (t.fillRatio ?? 0), 0) / baselineClosed.length
      : 0;
  const baselineAvgEdge =
    baselineClosed.length > 0
      ? baselineClosed.reduce((s, t) => s + (t.capturableEdgeAtEntry ?? 0), 0) / baselineClosed.length
      : 0;

  return {
    baselineProfileId: compareProfileId,
    challengerProfileId: PROFILE_ID,
    baselineClosed: baselineClosed.length,
    challengerClosed: challengerClosed.length,
    baselineAvgRealizedPnL: baselineAvgPnL,
    challengerAvgRealizedPnL: diagnostics.avgRealizedPnL,
    baselineMedianRealizedPnL: median(baselineClosed.map((t) => t.realizedPnL ?? 0)),
    challengerMedianRealizedPnL: diagnostics.medianRealizedPnL,
    baselineTotalRealizedPnL: baselineTotal,
    challengerTotalRealizedPnL: diagnostics.totalRealizedPnL,
    baselineAvgFillRatio: baselineAvgFill,
    challengerAvgFillRatio: diagnostics.avgFillRatioOpened,
    baselineAvgCapturableEdgeAtEntry: baselineAvgEdge,
    challengerAvgCapturableEdgeAtEntry: diagnostics.avgCapturableEdgeOpened,
    sameUniverseNote: `Challenger: structural pair×fill0.1-0.25 + capfloor 4.5% + degratio 0.24 + adaptive sizing. Baseline: ${compareProfileId}.`,
  };
}

export const STRUCTURAL_RISK_MANAGED_PROFILE_ID = PROFILE_ID;
