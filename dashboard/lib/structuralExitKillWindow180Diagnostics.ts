/**
 * Structural Exit-Kill Window 180 Challenger Diagnostics — shadow_1000_structural_exitkill_window180_v1.
 * Hipótese: janela de 180s dá chance real à lógica de kill atuar.
 * Única diferença vs exitkill_v1: monitoringWindowMs 180_000 em vez de 90_000.
 */

import type { ShadowProfileState } from "./shadowSimulationStore";
import type { ClosedTradeAuditEntry } from "./shadowClosedTradeAudit";
import type { StructuralExitKillDiagnosticsBlock, StructuralExitKillComparisonBlock } from "./structuralExitKillDiagnostics";

const PROFILE_ID = "shadow_1000_structural_exitkill_window180_v1";
const MONITORING_WINDOW_MS = 180_000;

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const m = Math.floor(arr.length / 2);
  return arr.length % 2 ? sorted[m]! : (sorted[m - 1]! + sorted[m]!) / 2;
}

type TradeLike = {
  exitKillTriggered?: boolean;
  exitKillReason?: string | null;
  exitKillAtMsFromOpen?: number | null;
  structuralRiskCapitalMultiplierAtOpen?: number | null;
  capturableEdgeAtEntry?: number | null;
  observedEdgeAtEntry?: number | null;
  realizedPnL?: number | null;
  holdingTimeMs?: number | null;
  fillRatio?: number | null;
  filledCapital?: number | null;
  requestedCapital?: number | null;
};

export function getStructuralExitKillWindow180Diagnostics(
  profile: ShadowProfileState | undefined,
  _allAuditEntries: ClosedTradeAuditEntry[],
  _rejectionCountsByProfile: Record<string, Record<string, number>>
): StructuralExitKillDiagnosticsBlock {
  const closed = (profile?.closedTrades ?? []).filter(
    (t) => t.status === "closed" && t.closedAt && t.structuralRiskFilterMatchAtOpen !== false
  ) as TradeLike[];
  const openedCount = closed.length + (profile?.activeTrades?.length ?? 0);

  const killExits = closed.filter((t) => t.exitKillTriggered === true);
  const holdingMsKill = killExits.map((t) => t.exitKillAtMsFromOpen ?? t.holdingTimeMs ?? 0);
  const killReasonCounts: Record<string, number> = {};
  for (const t of killExits) {
    const r = (t.exitKillReason ?? "unknown") as string;
    killReasonCounts[r] = (killReasonCounts[r] ?? 0) + 1;
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
    evaluatedOpportunityCount: 0,
    openedTradeCount: openedCount,
    closedTradeCount: closed.length,
    avgRealizedPnL: pnls.length ? pnls.reduce((a, b) => a + b, 0) / pnls.length : 0,
    medianRealizedPnL: median(pnls),
    totalRealizedPnL: pnls.reduce((a, b) => a + b, 0),
    avgHoldingTimeMs: holdingMs.length ? holdingMs.reduce((a, b) => a + b, 0) / holdingMs.length : 0,
    earlyKillExitCount: killExits.length,
    avgHoldingMsEarlyKill: holdingMsKill.length ? holdingMsKill.reduce((a, b) => a + b, 0) / holdingMsKill.length : 0,
    killReasonCounts,
    avgCapitalMultiplierOpened: multipliers.length ? multipliers.reduce((a, b) => a + b, 0) / multipliers.length : 0,
    avgCapturableEdgeOpened: capturableEdges.length ? capturableEdges.reduce((a, b) => a + b, 0) / capturableEdges.length : 0,
    avgObservedEdgeOpened: observedEdges.length ? observedEdges.reduce((a, b) => a + b, 0) / observedEdges.length : 0,
    avgDegradationRatioOpened: degRatios.length ? degRatios.reduce((a, b) => a + b, 0) / degRatios.length : 0,
    avgFillRatioOpened: avgFill,
  };
}

export function getStructuralExitKillWindow180Comparison(
  profiles: ShadowProfileState[],
  diagnostics: StructuralExitKillDiagnosticsBlock,
  compareProfileId: string
): StructuralExitKillComparisonBlock {
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
    sameUniverseNote: `Challenger: structural pair×fill×capfloor×degratio + exit kill window 180s. Baseline: ${compareProfileId}.`,
  };
}

export const STRUCTURAL_EXIT_KILL_WINDOW180_PROFILE_ID = PROFILE_ID;

export interface ExitKillWindow180CausalAuditBlock {
  profileId: string;
  totalClosed: number;
  closedInKillWindow: number;
  closedOutsideKillWindow: number;
  avgHoldingTimeMs: number;
  killWindowMs: number;
  exitReasonBreakdown: Record<string, number>;
  tradesNeverEvaluatedForKill: number;
  tradesEvaluatedButNoKill: number;
}

export function getExitKillWindow180CausalAudit(
  profile: ShadowProfileState | undefined
): ExitKillWindow180CausalAuditBlock {
  const closed = (profile?.closedTrades ?? []).filter(
    (t) => t.status === "closed" && t.closedAt && t.structuralRiskFilterMatchAtOpen !== false
  );
  const inWindow = closed.filter((t) => (t.holdingTimeMs ?? 0) < MONITORING_WINDOW_MS);
  const outsideWindow = closed.filter((t) => (t.holdingTimeMs ?? 0) >= MONITORING_WINDOW_MS);

  const exitReasonBreakdown: Record<string, number> = {};
  for (const t of closed) {
    const r = t.exitReason ?? "unknown";
    exitReasonBreakdown[r] = (exitReasonBreakdown[r] ?? 0) + 1;
  }

  const avgHoldingMs =
    closed.length > 0
      ? closed.reduce((s, t) => s + (t.holdingTimeMs ?? 0), 0) / closed.length
      : 0;

  return {
    profileId: PROFILE_ID,
    totalClosed: closed.length,
    closedInKillWindow: inWindow.length,
    closedOutsideKillWindow: outsideWindow.length,
    avgHoldingTimeMs: avgHoldingMs,
    killWindowMs: MONITORING_WINDOW_MS,
    exitReasonBreakdown,
    tradesNeverEvaluatedForKill: outsideWindow.length,
    tradesEvaluatedButNoKill: inWindow.length,
  };
}
