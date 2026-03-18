/**
 * Structural Late Exit Challenger Diagnostics — shadow_1000_structural_lateexit_nonreversion_v1.
 * Hipótese: saída por não reversão ou estagnação tardia, alinhada ao regime (trades holdam até late phase).
 */

import type { ShadowProfileState } from "./shadowSimulationStore";
import type { ClosedTradeAuditEntry } from "./shadowClosedTradeAudit";

const PROFILE_ID = "shadow_1000_structural_lateexit_nonreversion_v1";

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const m = Math.floor(arr.length / 2);
  return arr.length % 2 ? sorted[m]! : (sorted[m - 1]! + sorted[m]!) / 2;
}

export interface StructuralLateExitDiagnosticsBlock {
  profileId: string;
  openedTradeCount: number;
  closedTradeCount: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  totalRealizedPnL: number;
  avgHoldingTimeMs: number;
  lateExitTriggeredCount: number;
  avgHoldingMsLateExit: number;
  lateExitReasonCounts: Record<string, number>;
  avgCapitalMultiplierOpened: number;
  avgCapturableEdgeOpened: number;
  avgObservedEdgeOpened: number;
  avgDegradationRatioOpened: number;
  avgFillRatioOpened: number;
}

export interface StructuralLateExitComparisonBlock {
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

type TradeLike = {
  lateExitTriggered?: boolean;
  lateExitReason?: string | null;
  lateExitAtMsFromOpen?: number | null;
  structuralRiskCapitalMultiplierAtOpen?: number | null;
  capturableEdgeAtEntry?: number | null;
  observedEdgeAtEntry?: number | null;
  realizedPnL?: number | null;
  holdingTimeMs?: number | null;
  fillRatio?: number | null;
  filledCapital?: number | null;
  requestedCapital?: number | null;
};

export function getStructuralLateExitDiagnostics(
  profile: ShadowProfileState | undefined,
  _allAuditEntries: ClosedTradeAuditEntry[],
  _rejectionCountsByProfile: Record<string, Record<string, number>>
): StructuralLateExitDiagnosticsBlock {
  const closed = (profile?.closedTrades ?? []).filter(
    (t) => t.status === "closed" && t.closedAt && t.structuralRiskFilterMatchAtOpen !== false
  ) as TradeLike[];
  const openedCount = closed.length + (profile?.activeTrades?.length ?? 0);

  const lateExits = closed.filter((t) => t.lateExitTriggered === true);
  const holdingMsLate = lateExits.map((t) => t.lateExitAtMsFromOpen ?? t.holdingTimeMs ?? 0);
  const lateExitReasonCounts: Record<string, number> = {};
  for (const t of lateExits) {
    const r = (t.lateExitReason ?? "unknown") as string;
    lateExitReasonCounts[r] = (lateExitReasonCounts[r] ?? 0) + 1;
  }

  const multipliers = closed
    .map((t) => t.structuralRiskCapitalMultiplierAtOpen)
    .filter((m): m is number => typeof m === "number");
  const capturableEdges = closed.map((t) => t.capturableEdgeAtEntry ?? 0);
  const observedEdges = closed.map((t) => t.observedEdgeAtEntry ?? 0);
  const degRatios = closed.map((t) => {
    const cap = t.capturableEdgeAtEntry ?? 0;
    const obs = t.observedEdgeAtEntry ?? 0;
    return obs > 0.0001 ? cap / obs : 0;
  });
  const pnls = closed.map((t) => t.realizedPnL ?? 0);
  const holdingMs = closed.map((t) => t.holdingTimeMs ?? 0);
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
    openedTradeCount: openedCount,
    closedTradeCount: closed.length,
    avgRealizedPnL: pnls.length ? pnls.reduce((a, b) => a + b, 0) / pnls.length : 0,
    medianRealizedPnL: median(pnls),
    totalRealizedPnL: pnls.reduce((a, b) => a + b, 0),
    avgHoldingTimeMs: holdingMs.length ? holdingMs.reduce((a, b) => a + b, 0) / holdingMs.length : 0,
    lateExitTriggeredCount: lateExits.length,
    avgHoldingMsLateExit: holdingMsLate.length ? holdingMsLate.reduce((a, b) => a + b, 0) / holdingMsLate.length : 0,
    lateExitReasonCounts,
    avgCapitalMultiplierOpened: multipliers.length ? multipliers.reduce((a, b) => a + b, 0) / multipliers.length : 0,
    avgCapturableEdgeOpened: capturableEdges.length ? capturableEdges.reduce((a, b) => a + b, 0) / capturableEdges.length : 0,
    avgObservedEdgeOpened: observedEdges.length ? observedEdges.reduce((a, b) => a + b, 0) / observedEdges.length : 0,
    avgDegradationRatioOpened: degRatios.length ? degRatios.reduce((a, b) => a + b, 0) / degRatios.length : 0,
    avgFillRatioOpened: avgFill,
  };
}

export function getStructuralLateExitComparison(
  profiles: ShadowProfileState[],
  diagnostics: StructuralLateExitDiagnosticsBlock,
  compareProfileId: string
): StructuralLateExitComparisonBlock {
  const baseline = profiles.find((p) => p.profileId === compareProfileId);
  const challenger = profiles.find((p) => p.profileId === PROFILE_ID);

  const baselineClosed = baseline?.closedTrades ?? [];
  const challengerClosed = challenger?.closedTrades?.filter((t) => t.structuralRiskFilterMatchAtOpen !== false) ?? [];

  const baselineAvgPnL =
    baselineClosed.length > 0
      ? baselineClosed.reduce((s, t) => s + (t.realizedPnL ?? 0), 0) / baselineClosed.length
      : 0;
  const baselineTotal = baselineClosed.reduce((s, t) => s + (t.realizedPnL ?? 0), 0);
  const baselineFillRatios = baselineClosed.map((t) => {
    const fill = (t as { fillRatio?: number }).fillRatio;
    const req = (t as { requestedCapital?: number }).requestedCapital;
    const filled = (t as { filledCapital?: number }).filledCapital ?? 0;
    return fill != null ? fill : req != null && req > 0 ? filled / req : 0;
  });
  const baselineAvgFill =
    baselineFillRatios.length > 0
      ? baselineFillRatios.reduce((a, b) => a + b, 0) / baselineFillRatios.length
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
    sameUniverseNote: `Challenger: structural pair×fill×capfloor×degratio + late exit (non-reversion/stagnation). Baseline: ${compareProfileId}.`,
  };
}

export const STRUCTURAL_LATE_EXIT_PROFILE_ID = PROFILE_ID;
