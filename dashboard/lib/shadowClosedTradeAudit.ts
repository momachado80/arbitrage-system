/**
 * Shadow Closed Trade Audit — diagnostic module only.
 * Computes realized PnL statistics and breakdowns from closed shadow trades.
 * Does NOT change business logic, ranking, execution, or thresholds.
 */

import type { ShadowTrade, ShadowProfileState } from "./shadowSimulationStore";

export interface ClosedTradeAuditEntry {
  tradeId: string;
  profileId: string;
  opportunityId: string;
  opportunityType: string;
  sourceType: string;
  exitReason: string;
  filledCapital: number;
  realizedPnL: number;
  realizedReturn: number;
  holdingTimeMs: number;
  holdingTimeBucket: string;
  observedEdgeAtEntry: number;
  capturableEdgeAtEntry: number;
  effectiveEntryPrice: number;
  effectiveExitPrice: number;
  openedAt: string;
  closedAt: string;
}

export interface ProfileAuditSummary {
  profileId: string;
  totalClosed: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  winRate: number;
  lossRate: number;
  avgHoldingTimeMs: number;
  avgFilledCapital: number;
  avgObservedEdgeAtEntry: number;
  avgCapturableEdgeAtEntry: number;
  avgEffectiveEntryPrice: number;
  totalRealizedPnL: number;
  sumWins: number;
  sumLosses: number;
}

export interface ByBreakdown {
  byOpportunityType: Record<string, { count: number; totalPnL: number; avgPnL: number }>;
  byExitReason: Record<string, { count: number; totalPnL: number; avgPnL: number }>;
  byHoldingBucket: Record<string, { count: number; totalPnL: number; avgPnL: number }>;
}

export interface ClosedTradeAuditResult {
  timestamp: string;
  realizedPnLFormula: string;
  codePath: string;
  profileSummaries: ProfileAuditSummary[];
  byProfile: Record<string, ByBreakdown>;
  worst20: ClosedTradeAuditEntry[];
  best20: ClosedTradeAuditEntry[];
  lossDriverAnalysis: {
    badEntries: string;
    badExits: string;
    poorFillQuality: string;
    markToMarket: string;
    settlementAssumptions: string;
  };
  negativeExpectancy: boolean;
  safestNextChange: string;
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const m = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[m] : (sorted[m - 1] + sorted[m]) / 2;
}

function holdingBucket(ms: number): string {
  if (ms < 60_000) return "<1min";
  if (ms < 300_000) return "1-5min";
  if (ms < 900_000) return "5-15min";
  if (ms < 3_600_000) return "15-60min";
  if (ms < 300_000 * 5) return "60-300min";
  return ">300min";
}

function toAuditEntry(t: ShadowTrade, profileId: string): ClosedTradeAuditEntry {
  return {
    tradeId: t.tradeId,
    profileId,
    opportunityId: t.opportunityId,
    opportunityType: t.opportunityType ?? "unknown",
    sourceType: t.sourceType ?? "standard",
    exitReason: t.exitReason ?? "unknown",
    filledCapital: t.filledCapital ?? 0,
    realizedPnL: t.realizedPnL ?? 0,
    realizedReturn: t.realizedReturn ?? 0,
    holdingTimeMs: t.holdingTimeMs ?? 0,
    holdingTimeBucket: holdingBucket(t.holdingTimeMs ?? 0),
    observedEdgeAtEntry: t.observedEdgeAtEntry ?? 0,
    capturableEdgeAtEntry: t.capturableEdgeAtEntry ?? 0,
    effectiveEntryPrice: t.effectiveEntryPrice ?? 0,
    effectiveExitPrice: t.effectiveExitPrice ?? 0,
    openedAt: t.openedAt ?? "",
    closedAt: t.closedAt ?? "",
  };
}

export function computeClosedTradeAudit(profiles: ShadowProfileState[]): ClosedTradeAuditResult {
  const allEntries: ClosedTradeAuditEntry[] = [];
  const profileSummaries: ProfileAuditSummary[] = [];
  const byProfile: Record<string, ByBreakdown> = {};

  for (const p of profiles) {
    const closed = p.closedTrades.filter((t) => t.status === "closed" && t.closedAt);
    const entries = closed.map((t) => toAuditEntry(t, p.profileId));
    allEntries.push(...entries);

    const byOppType: Record<string, { count: number; totalPnL: number; pnls: number[] }> = {};
    const byExit: Record<string, { count: number; totalPnL: number; pnls: number[] }> = {};
    const byBucket: Record<string, { count: number; totalPnL: number; pnls: number[] }> = {};

    for (const e of entries) {
      const pnl = e.realizedPnL;
      byOppType[e.opportunityType] = byOppType[e.opportunityType] ?? { count: 0, totalPnL: 0, pnls: [] };
      byOppType[e.opportunityType].count++;
      byOppType[e.opportunityType].totalPnL += pnl;
      byOppType[e.opportunityType].pnls.push(pnl);

      byExit[e.exitReason] = byExit[e.exitReason] ?? { count: 0, totalPnL: 0, pnls: [] };
      byExit[e.exitReason].count++;
      byExit[e.exitReason].totalPnL += pnl;
      byExit[e.exitReason].pnls.push(pnl);

      byBucket[e.holdingTimeBucket] = byBucket[e.holdingTimeBucket] ?? { count: 0, totalPnL: 0, pnls: [] };
      byBucket[e.holdingTimeBucket].count++;
      byBucket[e.holdingTimeBucket].totalPnL += pnl;
      byBucket[e.holdingTimeBucket].pnls.push(pnl);
    }

    const wins = entries.filter((e) => e.realizedPnL > 0);
    const losses = entries.filter((e) => e.realizedPnL < 0);
    const sumWins = wins.reduce((s, e) => s + e.realizedPnL, 0);
    const sumLosses = losses.reduce((s, e) => s + e.realizedPnL, 0);
    const pnls = entries.map((e) => e.realizedPnL);

    profileSummaries.push({
      profileId: p.profileId,
      totalClosed: closed.length,
      avgRealizedPnL: closed.length ? p.realizedPnL / closed.length : 0,
      medianRealizedPnL: median(pnls),
      winRate: closed.length ? wins.length / closed.length : 0,
      lossRate: closed.length ? losses.length / closed.length : 0,
      avgHoldingTimeMs:
        closed.length ? entries.reduce((s, e) => s + e.holdingTimeMs, 0) / entries.length : 0,
      avgFilledCapital:
        closed.length ? entries.reduce((s, e) => s + e.filledCapital, 0) / entries.length : 0,
      avgObservedEdgeAtEntry:
        closed.length ? entries.reduce((s, e) => s + e.observedEdgeAtEntry, 0) / entries.length : 0,
      avgCapturableEdgeAtEntry:
        closed.length ? entries.reduce((s, e) => s + e.capturableEdgeAtEntry, 0) / entries.length : 0,
      avgEffectiveEntryPrice:
        closed.length ? entries.reduce((s, e) => s + e.effectiveEntryPrice, 0) / entries.length : 0,
      totalRealizedPnL: p.realizedPnL,
      sumWins,
      sumLosses,
    });

    const toBreakdown = (
      r: Record<string, { count: number; totalPnL: number; pnls: number[] }>
    ): ByBreakdown["byOpportunityType"] =>
      Object.fromEntries(
        Object.entries(r).map(([k, v]) => [
          k,
          {
            count: v.count,
            totalPnL: v.totalPnL,
            avgPnL: v.count ? v.totalPnL / v.count : 0,
          },
        ])
      );

    byProfile[p.profileId] = {
      byOpportunityType: toBreakdown(byOppType),
      byExitReason: toBreakdown(byExit),
      byHoldingBucket: toBreakdown(byBucket),
    };
  }

  const sortedByPnL = [...allEntries].sort((a, b) => a.realizedPnL - b.realizedPnL);
  const worst20 = sortedByPnL.slice(0, 20);
  const best20 = sortedByPnL.slice(-20).reverse();

  const totalPnL = allEntries.reduce((s, e) => s + e.realizedPnL, 0);
  const totalClosed = allEntries.length;
  const hasNegativeExpectancy = totalClosed > 0 && totalPnL / totalClosed < 0;

  return {
    timestamp: new Date().toISOString(),
    realizedPnLFormula:
      "realizedPnL = filledCapital * pnlPct where pnlPct = (effectiveExitPrice - effectiveEntryPrice) / max(0.001, effectiveEntryPrice); effectiveExitPrice = exitPrice - exitImpactResult.expectedPriceWorseningExit",
    codePath:
      "shadowSimulationService.runCycle -> simulateRealisticExit (realisticExecutionEngine.ts L326-371) -> closeShadowTrade (shadowSimulationStore.ts L155-202)",
    profileSummaries,
    byProfile,
    worst20,
    best20,
    lossDriverAnalysis: {
      badEntries:
        "Check avgObservedEdgeAtEntry vs avgCapturableEdgeAtEntry; if capturable << observed, entries are degraded by latency/impact.",
      badExits:
        "Check byExitReason: stop_loss and edge_normalization with negative PnL indicate exits at worse prices than entry.",
      poorFillQuality:
        "fillProbability not stored on trades; proxy: filledCapital/requestedCapital. Low filledCapital with high edge may indicate poor fills.",
      markToMarket:
        "Unrealized uses observedEdgeAtEntry for exitEst; does not drive realized PnL at close.",
      settlementAssumptions:
        "Exit uses latestOpportunity.edge for exitPrice; if opportunity disappeared, uses effectiveEntryPrice (flat exit).",
    },
    negativeExpectancy: hasNegativeExpectancy,
    safestNextChange:
      "Add minRealizedEdgeThreshold: only open when capturableEdgeAtEntry exceeds a floor (e.g. 0.01) to filter marginal entries. Or tighten minNetCapturableEdgeToTrade slightly. Test on shadow profile first.",
  };
}
