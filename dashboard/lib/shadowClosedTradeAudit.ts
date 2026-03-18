/**
 * Shadow Closed Trade Audit — diagnostic module only.
 * Computes realized PnL statistics and breakdowns from closed shadow trades.
 * Does NOT change business logic, ranking, execution, or thresholds.
 */

import type { ShadowTrade, ShadowProfileState } from "./shadowSimulationStore";
import { getProfileById } from "./shadowSimulationProfiles";

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
  /** Diagnostic: derived when requestedCapital available (prospective-only for historical) */
  fillRatio?: number | null;
  pairKey?: string | null;
  /** Narrow challenger: true = trade abriu com filtros narrow satisfeitos; audit usa só estes para performance oficial */
  narrowFilterMatchAtOpen?: boolean;
  /** Entry challenger counterfactual: baseline abriu mas challenger filtrou no mesmo ciclo */
  capfloorFilteredSameCycle?: boolean;
  degratioFilteredSameCycle?: boolean;
  /** Structural challenger */
  structuralFilterMatchAtOpen?: boolean;
  structuralTargetPairSetAtOpen?: string[] | null;
  structuralTargetFillBucketAtOpen?: string | null;
  structuralTargetEdgeBucketAtOpen?: string | null;
  structuralTargetVersion?: string | null;
  /** Structural risk-managed */
  structuralRiskCapitalMultiplierAtOpen?: number | null;
  structuralRiskCapfloorAtOpen?: number | null;
  structuralRiskDegRatioAtOpen?: number | null;
  structuralRiskTargetPairSetAtOpen?: string[] | null;
  structuralRiskTargetFillBucketAtOpen?: string | null;
  structuralRiskFilterMatchAtOpen?: boolean;
  structuralRiskTargetVersion?: string | null;
  earlyThesisFailureTriggered?: boolean;
  earlyThesisFailureReason?: string | null;
  earlyThesisFailureAtMsFromOpen?: number | null;
  /** Exit-kill challenger */
  exitKillTriggered?: boolean;
  exitKillReason?: string | null;
  exitKillAtMsFromOpen?: number | null;
  capturableEdgeAtKill?: number | null;
  observedEdgeAtKill?: number | null;
  degradationRatioAtKill?: number | null;
  opportunityAbsentCyclesAtKill?: number | null;
  /** Late exit challenger */
  lateExitTriggered?: boolean;
  lateExitReason?: string | null;
  lateExitAtMsFromOpen?: number | null;
  capturableEdgeAtLateExit?: number | null;
  observedEdgeAtLateExit?: number | null;
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

export interface ByCapturableEdgeDecileEntry {
  tradeCount: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  winRate: number;
  avgFilledCapital: number;
  avgFillRatio: number;
}

export interface ByFillRatioBucketEntry {
  tradeCount: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  winRate: number;
  avgCapturableEdgeAtEntry: number;
}

export interface WorstPairEntry {
  pairKey: string;
  tradeCount: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  winRate: number;
  avgCapturableEdgeAtEntry: number;
  avgFillRatio: number;
  dominantExitReason?: string;
}

export interface ExitReasonDiagnosticEntry {
  tradeCount: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  winRate: number;
  avgHoldingTimeMs: number;
  avgFillRatio: number;
}

export interface ProfileComparisonEntry {
  profileId: string;
  maxHoldingTimeMs: number;
  totalClosed: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  winRate: number;
  avgHoldingTimeMs: number;
  avgFilledCapital: number;
  avgFillRatio: number;
}

export interface CausalReadout {
  edgeMonotonicityHealthy: boolean;
  fillQualityLikelyPrimaryDriver: boolean;
  exitsLikelyPrimaryDriver: boolean;
  pairSelectionLikelyPrimaryDriver: boolean;
  enoughDataForCausalReadout: boolean;
  leadingHypothesis: string;
}

export interface ProfileAggregationConsistency {
  profileId: string;
  totalRealizedPnL: number;
  sumWins: number;
  sumLosses: number;
  sumWinsPlusSumLosses: number;
  byOpportunityTypeTotal: number;
  byExitReasonTotal: number;
  byHoldingBucketTotal: number;
  tradeCount: number;
  consistent: boolean;
  /** store.realizedPnL pode divergir quando closedTrades foi slice() sem ajustar (MAX_CLOSED_PER_PROFILE) */
  storeRealizedPnL?: number;
  storeVsDerivedDiff?: number;
  /** Diferenças absolutas vs total (para auditoria) */
  diffTotalVsSumWinsLosses: number;
  diffTotalVsByOpp: number;
  diffTotalVsByExit: number;
  diffTotalVsByBucket: number;
}

export interface AggregationConsistencyDiagnostics {
  methodology: string;
  byProfile: Record<string, ProfileAggregationConsistency>;
  allConsistent: boolean;
  universe: "closedTrades_in_memory";
}

export interface ClosedTradeAuditResult {
  timestamp: string;
  realizedPnLFormula: string;
  codePath: string;
  profileSummaries: ProfileAuditSummary[];
  byProfile: Record<string, ByBreakdown>;
  aggregationConsistencyDiagnostics: AggregationConsistencyDiagnostics;
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
  dataSufficiency: {
    totalClosed: number;
    minRequiredForTuning: number;
    sufficientForThresholdTuning: boolean;
  };
  /** Audit aggregations for loss driver identification */
  byCapturableEdgeDecile: Record<string, ByCapturableEdgeDecileEntry>;
  byFillRatioBucket: Record<string, ByFillRatioBucketEntry>;
  worstPairs: WorstPairEntry[];
  exitReasonDiagnostics: Record<string, ExitReasonDiagnosticEntry>;
  profileComparison: ProfileComparisonEntry[];
  causalReadout: CausalReadout;
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

/** Derive fillRatio only when requestedCapital is available; do not invent retroactive values */
function getFillRatio(t: ShadowTrade): number | null {
  if (t.fillRatio != null && typeof t.fillRatio === "number") return t.fillRatio;
  const rc = t.requestedCapital;
  const fc = t.filledCapital ?? 0;
  if (rc != null && rc > 0) return fc / rc;
  return null;
}

/** Derive pairKey from marketsInvolved when not stored */
function getPairKey(t: ShadowTrade): string | null {
  if (t.pairKey != null && t.pairKey !== "") return t.pairKey;
  const m = t.marketsInvolved;
  if (!m?.length) return null;
  const ids = m.map((x) => x.marketId).filter(Boolean);
  if (ids.length === 0) return null;
  return [...ids].sort().join("+");
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
    fillRatio: getFillRatio(t),
    pairKey: getPairKey(t),
    narrowFilterMatchAtOpen: t.narrowFilterMatchAtOpen,
    capfloorFilteredSameCycle: t.capfloorFilteredSameCycle,
    degratioFilteredSameCycle: t.degratioFilteredSameCycle,
    structuralFilterMatchAtOpen: t.structuralFilterMatchAtOpen,
    structuralTargetPairSetAtOpen: t.structuralTargetPairSetAtOpen,
    structuralTargetFillBucketAtOpen: t.structuralTargetFillBucketAtOpen,
    structuralTargetEdgeBucketAtOpen: t.structuralTargetEdgeBucketAtOpen,
    structuralTargetVersion: t.structuralTargetVersion,
    structuralRiskCapitalMultiplierAtOpen: t.structuralRiskCapitalMultiplierAtOpen,
    structuralRiskCapfloorAtOpen: t.structuralRiskCapfloorAtOpen,
    structuralRiskDegRatioAtOpen: t.structuralRiskDegRatioAtOpen,
    structuralRiskTargetPairSetAtOpen: t.structuralRiskTargetPairSetAtOpen,
    structuralRiskTargetFillBucketAtOpen: t.structuralRiskTargetFillBucketAtOpen,
    structuralRiskFilterMatchAtOpen: t.structuralRiskFilterMatchAtOpen,
    structuralRiskTargetVersion: t.structuralRiskTargetVersion,
    earlyThesisFailureTriggered: t.earlyThesisFailureTriggered,
    earlyThesisFailureReason: t.earlyThesisFailureReason,
    earlyThesisFailureAtMsFromOpen: t.earlyThesisFailureAtMsFromOpen,
    exitKillTriggered: t.exitKillTriggered,
    exitKillReason: t.exitKillReason,
    exitKillAtMsFromOpen: t.exitKillAtMsFromOpen,
    capturableEdgeAtKill: t.capturableEdgeAtKill,
    observedEdgeAtKill: t.observedEdgeAtKill,
    degradationRatioAtKill: t.degradationRatioAtKill,
    opportunityAbsentCyclesAtKill: t.opportunityAbsentCyclesAtKill,
    lateExitTriggered: t.lateExitTriggered,
    lateExitReason: t.lateExitReason,
    lateExitAtMsFromOpen: t.lateExitAtMsFromOpen,
    capturableEdgeAtLateExit: t.capturableEdgeAtLateExit,
    observedEdgeAtLateExit: t.observedEdgeAtLateExit,
  };
}

const FILL_RATIO_BUCKETS = ["0-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"] as const;

function bucketByFillRatio(r: number | null): string | null {
  if (r == null || r < 0) return null;
  if (r <= 0.1) return "0-0.1";
  if (r <= 0.25) return "0.1-0.25";
  if (r <= 0.5) return "0.25-0.5";
  if (r <= 0.75) return "0.5-0.75";
  if (r <= 1.0) return "0.75-1.0";
  return "0.75-1.0";
}

function decileKey(d: number): string {
  return `d${d}`;
}

export type ProfileConfigResolver = (profileId: string) => { maxHoldingTimeMs?: number } | undefined;

/** Constrói entries para segmentação local (fluxo novo vs reidratado). */
export function buildClosedTradeAuditEntries(
  profiles: ShadowProfileState[],
  _configResolver?: ProfileConfigResolver
): ClosedTradeAuditEntry[] {
  const out: ClosedTradeAuditEntry[] = [];
  for (const p of profiles) {
    const closed = p.closedTrades.filter((t) => t.status === "closed" && t.closedAt);
    for (const t of closed) {
      out.push(toAuditEntry(t, p.profileId));
    }
  }
  return out;
}

export function computeClosedTradeAudit(
  profiles: ShadowProfileState[],
  configResolver?: ProfileConfigResolver
): ClosedTradeAuditResult {
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
    /** Fonte única de verdade: soma dos realizedPnL dos trades em memória. p.realizedPnL pode divergir quando closedTrades foi slice() sem ajustar o store. */
    const totalFromEntries = entries.reduce((s, e) => s + e.realizedPnL, 0);

    profileSummaries.push({
      profileId: p.profileId,
      totalClosed: closed.length,
      avgRealizedPnL: closed.length ? totalFromEntries / closed.length : 0,
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
      totalRealizedPnL: totalFromEntries,
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

  const aggregationConsistencyDiagnostics = ((): AggregationConsistencyDiagnostics => {
    const byProfileDiag: Record<string, ProfileAggregationConsistency> = {};
    let allConsistent = true;
    for (const ps of profileSummaries) {
      const bp = byProfile[ps.profileId];
      const byOppTotal = bp
        ? Object.values(bp.byOpportunityType).reduce((s, x) => s + x.totalPnL, 0)
        : 0;
      const byExitTotal = bp
        ? Object.values(bp.byExitReason).reduce((s, x) => s + x.totalPnL, 0)
        : 0;
      const byBucketTotal = bp
        ? Object.values(bp.byHoldingBucket).reduce((s, x) => s + x.totalPnL, 0)
        : 0;
      const sumWinsPlusLosses = ps.sumWins + ps.sumLosses;
      const epsilon = 1e-9;
      const consistent =
        Math.abs(ps.totalRealizedPnL - sumWinsPlusLosses) < epsilon &&
        Math.abs(ps.totalRealizedPnL - byOppTotal) < epsilon &&
        Math.abs(ps.totalRealizedPnL - byExitTotal) < epsilon &&
        Math.abs(ps.totalRealizedPnL - byBucketTotal) < epsilon;
      if (!consistent) allConsistent = false;
      const prof = profiles.find((x) => x.profileId === ps.profileId);
      const storeRealizedPnL = prof?.realizedPnL;
      const storeVsDerivedDiff =
        storeRealizedPnL != null ? storeRealizedPnL - ps.totalRealizedPnL : undefined;
      byProfileDiag[ps.profileId] = {
        profileId: ps.profileId,
        totalRealizedPnL: ps.totalRealizedPnL,
        sumWins: ps.sumWins,
        sumLosses: ps.sumLosses,
        sumWinsPlusSumLosses: sumWinsPlusLosses,
        byOpportunityTypeTotal: byOppTotal,
        byExitReasonTotal: byExitTotal,
        byHoldingBucketTotal: byBucketTotal,
        tradeCount: ps.totalClosed,
        consistent,
        storeRealizedPnL,
        storeVsDerivedDiff,
        diffTotalVsSumWinsLosses: ps.totalRealizedPnL - sumWinsPlusLosses,
        diffTotalVsByOpp: ps.totalRealizedPnL - byOppTotal,
        diffTotalVsByExit: ps.totalRealizedPnL - byExitTotal,
        diffTotalVsByBucket: ps.totalRealizedPnL - byBucketTotal,
      };
    }
    return {
      methodology:
        "All totals derived from same universe: closedTrades in memory. totalRealizedPnL = sum(entries.realizedPnL). sumWins+sumLosses and byProfile breakdowns must reconcile.",
      byProfile: byProfileDiag,
      allConsistent,
      universe: "closedTrades_in_memory",
    };
  })();

  const sortedByPnL = [...allEntries].sort((a, b) => a.realizedPnL - b.realizedPnL);
  const worst20 = sortedByPnL.slice(0, 20);
  const best20 = sortedByPnL.slice(-20).reverse();

  const totalPnL = allEntries.reduce((s, e) => s + e.realizedPnL, 0);
  const totalClosed = allEntries.length;
  const hasNegativeExpectancy = totalClosed > 0 && totalPnL / totalClosed < 0;

  const MIN_CLOSED_FOR_TUNING = 20;
  const MIN_CLOSED_FOR_CAUSAL = 15;
  const dataSufficient = totalClosed >= MIN_CLOSED_FOR_TUNING;
  const dynamicSafestNextChange = dataSufficient
    ? "Add minRealizedEdgeThreshold: only open when capturableEdgeAtEntry exceeds a floor (e.g. 0.01) to filter marginal entries. Or tighten minNetCapturableEdgeToTrade slightly. Test on shadow profile first."
    : `Insufficient data for threshold tuning. Need ≥${MIN_CLOSED_FOR_TUNING} closed trades. Current: ${totalClosed}. Once sufficient, re-run audit.`;

  // 1. byCapturableEdgeDecile
  const byCapturableEdgeDecile: Record<string, ByCapturableEdgeDecileEntry> = {};
  const withCapturable = allEntries.filter((e) => e.capturableEdgeAtEntry != null);
  if (withCapturable.length >= 2) {
    const sorted = [...withCapturable].sort((a, b) => a.capturableEdgeAtEntry - b.capturableEdgeAtEntry);
    for (let d = 1; d <= 10; d++) {
      const lo = Math.floor(((d - 1) / 10) * sorted.length);
      const hi = Math.floor((d / 10) * sorted.length);
      const slice = sorted.slice(lo, hi);
      if (slice.length === 0) continue;
      const pnls = slice.map((e) => e.realizedPnL);
      const fillRatios = slice.map((e) => e.fillRatio ?? 0).filter((r) => r > 0);
      byCapturableEdgeDecile[decileKey(d)] = {
        tradeCount: slice.length,
        avgRealizedPnL: pnls.reduce((s, x) => s + x, 0) / pnls.length,
        medianRealizedPnL: median(pnls),
        winRate: slice.filter((e) => e.realizedPnL > 0).length / slice.length,
        avgFilledCapital: slice.reduce((s, e) => s + e.filledCapital, 0) / slice.length,
        avgFillRatio: fillRatios.length ? fillRatios.reduce((a, b) => a + b, 0) / fillRatios.length : 0,
      };
    }
  }

  // 2. byFillRatioBucket
  const byFillRatioBucket: Record<string, ByFillRatioBucketEntry> = {};
  for (const b of FILL_RATIO_BUCKETS) {
    byFillRatioBucket[b] = {
      tradeCount: 0,
      avgRealizedPnL: 0,
      medianRealizedPnL: 0,
      winRate: 0,
      avgCapturableEdgeAtEntry: 0,
    };
  }
  for (const e of allEntries) {
    const b = bucketByFillRatio(e.fillRatio ?? null);
    if (!b || !(b in byFillRatioBucket)) continue;
    const bucket = byFillRatioBucket[b];
    bucket.tradeCount++;
    bucket.avgRealizedPnL += e.realizedPnL;
    bucket.avgCapturableEdgeAtEntry += e.capturableEdgeAtEntry;
  }
  for (const b of FILL_RATIO_BUCKETS) {
    const bucket = byFillRatioBucket[b];
    if (bucket.tradeCount === 0) continue;
    bucket.avgRealizedPnL /= bucket.tradeCount;
    bucket.avgCapturableEdgeAtEntry /= bucket.tradeCount;
    const slice = allEntries.filter((e) => bucketByFillRatio(e.fillRatio ?? null) === b);
    bucket.medianRealizedPnL = median(slice.map((x) => x.realizedPnL));
    bucket.winRate = slice.filter((x) => x.realizedPnL > 0).length / slice.length;
  }

  // 3. worstPairs
  const byPair = new Map<string, typeof allEntries>();
  for (const e of allEntries) {
    const pk = e.pairKey ?? "";
    if (!pk) continue;
    const arr = byPair.get(pk) ?? [];
    arr.push(e);
    byPair.set(pk, arr);
  }
  const worstPairs: WorstPairEntry[] = Array.from(byPair.entries())
    .map(([pairKey, arr]) => {
      const pnls = arr.map((x) => x.realizedPnL);
      const fillRatios = arr.map((x) => x.fillRatio ?? 0).filter((r) => r > 0);
      const exitCounts: Record<string, number> = {};
      for (const x of arr) {
        const r = x.exitReason ?? "unknown";
        exitCounts[r] = (exitCounts[r] ?? 0) + 1;
      }
      let dominantExitReason: string | undefined;
      const maxCount = Math.max(...Object.values(exitCounts), 0);
      for (const [r, c] of Object.entries(exitCounts)) {
        if (c === maxCount) {
          dominantExitReason = r;
          break;
        }
      }
      return {
        pairKey,
        tradeCount: arr.length,
        avgRealizedPnL: pnls.reduce((a, b) => a + b, 0) / pnls.length,
        medianRealizedPnL: median(pnls),
        winRate: arr.filter((x) => x.realizedPnL > 0).length / arr.length,
        avgCapturableEdgeAtEntry: arr.reduce((s, x) => s + x.capturableEdgeAtEntry, 0) / arr.length,
        avgFillRatio: fillRatios.length ? fillRatios.reduce((a, b) => a + b, 0) / fillRatios.length : 0,
        dominantExitReason,
      };
    })
    .sort((a, b) => a.avgRealizedPnL - b.avgRealizedPnL)
    .slice(0, 20);

  // 4. exitReasonDiagnostics
  const exitReasonDiagnostics: Record<string, ExitReasonDiagnosticEntry> = {};
  const byExitForDiag = new Map<string, typeof allEntries>();
  for (const e of allEntries) {
    const r = e.exitReason ?? "unknown";
    const arr = byExitForDiag.get(r) ?? [];
    arr.push(e);
    byExitForDiag.set(r, arr);
  }
  for (const [reason, arr] of Array.from(byExitForDiag)) {
    const pnls = arr.map((x) => x.realizedPnL);
    const fillRatios = arr.map((x) => x.fillRatio ?? 0).filter((r) => r > 0);
    exitReasonDiagnostics[reason] = {
      tradeCount: arr.length,
      avgRealizedPnL: pnls.reduce((a, b) => a + b, 0) / pnls.length,
      medianRealizedPnL: median(pnls),
      winRate: arr.filter((x) => x.realizedPnL > 0).length / arr.length,
      avgHoldingTimeMs: arr.reduce((s, x) => s + x.holdingTimeMs, 0) / arr.length,
      avgFillRatio: fillRatios.length ? fillRatios.reduce((a, b) => a + b, 0) / fillRatios.length : 0,
    };
  }

  // 5. profileComparison
  const profileComparison: ProfileComparisonEntry[] = profiles.map((p) => {
    const closed = p.closedTrades.filter((t) => t.status === "closed" && t.closedAt);
    const entries = closed.map((t) => toAuditEntry(t, p.profileId));
    const baseCfg = getProfileById(p.profileId);
    const cfg = baseCfg ?? configResolver?.(p.profileId);
    const pnls = entries.map((e) => e.realizedPnL);
    const fillRatios = entries.map((e) => e.fillRatio ?? 0).filter((r) => r > 0);
    return {
      profileId: p.profileId,
      maxHoldingTimeMs: cfg?.maxHoldingTimeMs ?? 0,
      totalClosed: closed.length,
      avgRealizedPnL: closed.length ? entries.reduce((s, e) => s + e.realizedPnL, 0) / closed.length : 0,
      medianRealizedPnL: median(pnls),
      winRate: closed.length ? entries.filter((e) => e.realizedPnL > 0).length / closed.length : 0,
      avgHoldingTimeMs: closed.length ? entries.reduce((s, e) => s + e.holdingTimeMs, 0) / closed.length : 0,
      avgFilledCapital: closed.length ? entries.reduce((s, e) => s + e.filledCapital, 0) / closed.length : 0,
      avgFillRatio: fillRatios.length ? fillRatios.reduce((a, b) => a + b, 0) / fillRatios.length : 0,
    };
  });

  // 6. causalReadout
  const enoughDataForCausalReadout = totalClosed >= MIN_CLOSED_FOR_CAUSAL;
  let edgeMonotonicityHealthy = false;
  let fillQualityLikelyPrimaryDriver = false;
  let exitsLikelyPrimaryDriver = false;
  let pairSelectionLikelyPrimaryDriver = false;
  let leadingHypothesis = "Insufficient data for causal inference.";

  if (enoughDataForCausalReadout) {
    const decileKeys = Object.keys(byCapturableEdgeDecile).sort();
    if (decileKeys.length >= 3) {
      const avgPnls = decileKeys.map((k) => byCapturableEdgeDecile[k].avgRealizedPnL);
      let monotonic = true;
      for (let i = 1; i < avgPnls.length; i++) {
        if (avgPnls[i] < avgPnls[i - 1] - 0.001) {
          monotonic = false;
          break;
        }
      }
      edgeMonotonicityHealthy = monotonic;
    }

    const fillBucketsWithData = FILL_RATIO_BUCKETS.filter((b) => byFillRatioBucket[b].tradeCount >= 2);
    if (fillBucketsWithData.length >= 2) {
      const lowBucket = fillBucketsWithData[0];
      const highBucket = fillBucketsWithData[fillBucketsWithData.length - 1];
      const lowAvg = byFillRatioBucket[lowBucket].avgRealizedPnL;
      const highAvg = byFillRatioBucket[highBucket].avgRealizedPnL;
      fillQualityLikelyPrimaryDriver = highAvg - lowAvg > 0.005;
    }

    const exitReasons = Object.entries(exitReasonDiagnostics);
    const totalLoss = allEntries.filter((e) => e.realizedPnL < 0).reduce((s, e) => s + e.realizedPnL, 0);
    if (totalLoss < 0 && exitReasons.length >= 1) {
      const worstExit = exitReasons
        .filter(([, v]) => v.avgRealizedPnL < 0)
        .sort((a, b) => a[1].avgRealizedPnL - b[1].avgRealizedPnL)[0];
      if (worstExit && worstExit[1].tradeCount >= Math.ceil(totalClosed * 0.2)) {
        const lossShare = worstExit[1].tradeCount * Math.abs(worstExit[1].avgRealizedPnL);
        exitsLikelyPrimaryDriver = lossShare > Math.abs(totalLoss) * 0.4;
      }
    }

    if (worstPairs.length >= 1 && totalLoss < 0) {
      const topLossPairs = worstPairs.slice(0, 5);
      const pairLossShare = topLossPairs.reduce(
        (s, p) => s + p.tradeCount * Math.max(0, -p.avgRealizedPnL),
        0
      );
      pairSelectionLikelyPrimaryDriver = pairLossShare > Math.abs(totalLoss) * 0.5;
    }

    const drivers: string[] = [];
    if (!edgeMonotonicityHealthy) drivers.push("edge miscalibration at entry");
    if (fillQualityLikelyPrimaryDriver) drivers.push("poor fill quality / adverse selection");
    if (exitsLikelyPrimaryDriver) drivers.push("exit logic");
    if (pairSelectionLikelyPrimaryDriver) drivers.push("structurally bad cross-market pairs");
    leadingHypothesis =
      drivers.length > 0
        ? `Primary loss driver hypothesis: ${drivers.join("; ")}`
        : "No single driver dominates; losses may be distributed across causes.";
  }

  const causalReadout: CausalReadout = {
    edgeMonotonicityHealthy,
    fillQualityLikelyPrimaryDriver,
    exitsLikelyPrimaryDriver,
    pairSelectionLikelyPrimaryDriver,
    enoughDataForCausalReadout,
    leadingHypothesis,
  };

  return {
    timestamp: new Date().toISOString(),
    realizedPnLFormula:
      "realizedPnL = filledCapital * pnlPct where pnlPct = (effectiveExitPrice - effectiveEntryPrice) / max(0.001, effectiveEntryPrice); effectiveExitPrice = exitPrice - exitImpactResult.expectedPriceWorseningExit",
    codePath:
      "shadowSimulationService.runCycle -> simulateRealisticExit (realisticExecutionEngine.ts L326-371) -> closeShadowTrade (shadowSimulationStore.ts L155-202)",
    profileSummaries,
    byProfile,
    aggregationConsistencyDiagnostics,
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
    safestNextChange: dynamicSafestNextChange,
    dataSufficiency: {
      totalClosed,
      minRequiredForTuning: MIN_CLOSED_FOR_TUNING,
      sufficientForThresholdTuning: dataSufficient,
    },
    byCapturableEdgeDecile,
    byFillRatioBucket,
    worstPairs,
    exitReasonDiagnostics,
    profileComparison,
    causalReadout,
  };
}

/**
 * Segmentação para análise de ilhas locais de viabilidade.
 * Usar apenas com trades novos (não reidratados) para inferência econômica válida.
 * Não inventa positividade; retorna vazio se amostra insuficiente.
 */
export interface LocalViabilitySegment {
  tradeCount: number;
  avgRealizedPnL: number;
  winRate: number;
}

export interface LocalViabilityPrep {
  /** Só fluxo novo desta execução; nunca mistura com histórico */
  newTradesCount: number;
  sufficientForLocalAnalysis: boolean;
  minRequiredForLocalAnalysis: number;
  byPairKey: Record<string, LocalViabilitySegment>;
  byCapturableEdgeBucket: Record<string, LocalViabilitySegment>;
  byFillRatioBucket: Record<string, LocalViabilitySegment>;
  byPairKeyAndFillRatio: Record<string, LocalViabilitySegment>;
  byPairKeyAndEdge: Record<string, LocalViabilitySegment>;
  byExitReason: Record<string, LocalViabilitySegment>;
  byHoldingRegime: Record<string, LocalViabilitySegment>;
}

const MIN_FOR_LOCAL_ANALYSIS = 10;

export function computeLocalViabilitySegments(
  entries: ClosedTradeAuditEntry[],
  rehydratedIds: ReadonlySet<string>
): LocalViabilityPrep {
  const newEntries = entries.filter((e) => !rehydratedIds.has(e.tradeId));
  const n = newEntries.length;
  const sufficient = n >= MIN_FOR_LOCAL_ANALYSIS;

  type Acc = { tradeCount: number; totalPnL: number; wins: number };
  const byPairKey: Record<string, Acc> = {};
  const byFillRatioBucket: Record<string, Acc> = {};
  const byCapturableEdgeBucket: Record<string, Acc> = {};
  const byPairKeyAndFillRatio: Record<string, Acc> = {};
  const byPairKeyAndEdge: Record<string, Acc> = {};
  const byExitReason: Record<string, Acc> = {};
  const byHoldingRegime: Record<string, Acc> = {};

  function edgeBucket(edge: number): string {
    if (edge < 0.005) return "0-0.5%";
    if (edge < 0.01) return "0.5-1%";
    if (edge < 0.02) return "1-2%";
    if (edge < 0.05) return "2-5%";
    return ">5%";
  }

  function add(map: Record<string, Acc>, key: string, e: ClosedTradeAuditEntry): void {
    if (!map[key]) map[key] = { tradeCount: 0, totalPnL: 0, wins: 0 };
    map[key].tradeCount++;
    map[key].totalPnL += e.realizedPnL;
    map[key].wins += e.realizedPnL > 0 ? 1 : 0;
  }

  for (const e of newEntries) {
    const pk = e.pairKey ?? "_no_pair";
    const fr = bucketByFillRatio(e.fillRatio ?? null) ?? "_unknown";
    const eb = edgeBucket(e.capturableEdgeAtEntry ?? 0);
    add(byPairKey, pk, e);
    add(byFillRatioBucket, fr, e);
    add(byCapturableEdgeBucket, eb, e);
    add(byPairKeyAndFillRatio, `${pk}|${fr}`, e);
    add(byPairKeyAndEdge, `${pk}|${eb}`, e);
    add(byExitReason, e.exitReason ?? "unknown", e);
    add(byHoldingRegime, e.holdingTimeBucket ?? "unknown", e);
  }

  const toSeg = (m: Record<string, Acc>): Record<string, LocalViabilitySegment> =>
    Object.fromEntries(
      Object.entries(m).map(([k, v]) => [
        k,
        {
          tradeCount: v.tradeCount,
          avgRealizedPnL: v.tradeCount ? v.totalPnL / v.tradeCount : 0,
          winRate: v.tradeCount ? v.wins / v.tradeCount : 0,
        },
      ])
    );

  return {
    newTradesCount: n,
    sufficientForLocalAnalysis: sufficient,
    minRequiredForLocalAnalysis: MIN_FOR_LOCAL_ANALYSIS,
    byPairKey: toSeg(byPairKey),
    byCapturableEdgeBucket: toSeg(byCapturableEdgeBucket),
    byFillRatioBucket: toSeg(byFillRatioBucket),
    byPairKeyAndFillRatio: toSeg(byPairKeyAndFillRatio),
    byPairKeyAndEdge: toSeg(byPairKeyAndEdge),
    byExitReason: toSeg(byExitReason),
    byHoldingRegime: toSeg(byHoldingRegime),
  };
}
