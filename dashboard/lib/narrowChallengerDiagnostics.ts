/**
 * Narrow Challenger Diagnostics — observabilidade dedicada ao profile narrow.
 * In-memory only; no persistence.
 */

import type { ShadowProfileState } from "./shadowSimulationStore";
import type { ClosedTradeAuditEntry } from "./shadowClosedTradeAudit";
import { runStructuralPersistenceValidation } from "./structuralPersistenceValidation";
import { NARROW_CHALLENGER_TARGET } from "./narrowChallengerHelpers";

const NARROW_PROFILE_ID = "shadow_1000_narrow_562794_567561_edge2to5_fill25to50_v1";

interface ProfileCounters {
  opportunitiesSeen: number;
  rejectedByPairMismatch: number;
  rejectedByEdgeBucketMismatch: number;
  rejectedByFillBucketMismatch: number;
  passedAllNarrowFilters: number;
}

const counters = new Map<string, ProfileCounters>();

function ensureProfile(profileId: string): ProfileCounters {
  let c = counters.get(profileId);
  if (!c) {
    c = {
      opportunitiesSeen: 0,
      rejectedByPairMismatch: 0,
      rejectedByEdgeBucketMismatch: 0,
      rejectedByFillBucketMismatch: 0,
      passedAllNarrowFilters: 0,
    };
    counters.set(profileId, c);
  }
  return c;
}

export function recordOpportunitiesSeen(profileId: string): void {
  if (profileId !== NARROW_PROFILE_ID) return;
  const c = ensureProfile(profileId);
  c.opportunitiesSeen++;
}

export function recordRejectedByPairMismatch(profileId: string): void {
  if (profileId !== NARROW_PROFILE_ID) return;
  const c = ensureProfile(profileId);
  c.rejectedByPairMismatch++;
}

export function recordRejectedByEdgeBucketMismatch(profileId: string): void {
  if (profileId !== NARROW_PROFILE_ID) return;
  const c = ensureProfile(profileId);
  c.rejectedByEdgeBucketMismatch++;
}

export function recordRejectedByFillBucketMismatch(profileId: string): void {
  if (profileId !== NARROW_PROFILE_ID) return;
  const c = ensureProfile(profileId);
  c.rejectedByFillBucketMismatch++;
}

export function recordPassedAllNarrowFilters(profileId: string): void {
  if (profileId !== NARROW_PROFILE_ID) return;
  const c = ensureProfile(profileId);
  c.passedAllNarrowFilters++;
}

export const NARROW_CHALLENGER_PROFILE_ID = NARROW_PROFILE_ID;
export const BASELINE_PROFILE_ID = "shadow_1000";

export interface NarrowChallengerDiagnosticsResult {
  profileId: string;
  targetPairKey: string;
  targetCapturableEdgeBucket: string;
  targetFillRatioBucket: string;
  opportunitiesSeen: number;
  rejectedByPairMismatch: number;
  rejectedByEdgeBucketMismatch: number;
  rejectedByFillBucketMismatch: number;
  passedAllNarrowFilters: number;
  openedTradeCount: number;
  closedTradeCount: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  totalRealizedPnL: number;
  avgCapturableEdgeOpened: number;
  avgFillRatioOpened: number;
  discoveryCount: number;
  validationCount: number;
  discoveryAvgPnL: number;
  validationAvgPnL: number;
  persistenceMaintained?: boolean;
  eligibleForNarrowChallenger?: boolean;
  currentSubsetMatchesTargetDefinition?: boolean;
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const m = Math.floor(arr.length / 2);
  return arr.length % 2 ? sorted[m]! : (sorted[m - 1]! + sorted[m]!) / 2;
}

export function getNarrowChallengerDiagnostics(
  profiles: ShadowProfileState[],
  allAuditEntries: ClosedTradeAuditEntry[],
  rehydratedIds: ReadonlySet<string>
): NarrowChallengerDiagnosticsResult | null {
  const profile = profiles.find((p) => p.profileId === NARROW_PROFILE_ID);
  const c = ensureProfile(NARROW_PROFILE_ID);

  const closed = (profile?.closedTrades ?? []).filter((t) => t.status === "closed" && t.closedAt);
  /** Performance oficial do narrow: apenas trades com narrowFilterMatchAtOpen === true */
  const closedOfficial = closed.filter((t) => t.narrowFilterMatchAtOpen === true);
  const openedCount = closed.length + (profile?.activeTrades?.length ?? 0);

  const pnls = closedOfficial.map((t) => t.realizedPnL);
  const avgPnL = pnls.length ? pnls.reduce((a, b) => a + b, 0) / pnls.length : 0;
  const totalPnL = pnls.reduce((a, b) => a + b, 0);
  const avgCapturable = closedOfficial.length
    ? closedOfficial.reduce((s, t) => s + (t.capturableEdgeAtEntry ?? 0), 0) / closedOfficial.length
    : 0;
  const fillRatios = closedOfficial.map((t) =>
    t.fillRatio != null && typeof t.fillRatio === "number"
      ? t.fillRatio
      : (t.requestedCapital != null && t.requestedCapital > 0 ? t.filledCapital / t.requestedCapital : 1)
  );
  const avgFill = fillRatios.length ? fillRatios.reduce((a, b) => a + b, 0) / fillRatios.length : 0;

  const narrowEntries = allAuditEntries.filter(
    (e) => e.profileId === NARROW_PROFILE_ID && e.narrowFilterMatchAtOpen === true
  );
  const newEntries = narrowEntries.filter((e) => !rehydratedIds.has(e.tradeId));
  const sorted = [...newEntries].sort(
    (a, b) => new Date(a.closedAt).getTime() - new Date(b.closedAt).getTime()
  );
  const mid = Math.floor(sorted.length / 2);
  const discoveryEntries = sorted.slice(0, mid);
  const validationEntries = sorted.slice(mid);
  const discoveryCount = discoveryEntries.length;
  const validationCount = validationEntries.length;
  const discAvg =
    discoveryEntries.length > 0
      ? discoveryEntries.reduce((s, e) => s + e.realizedPnL, 0) / discoveryEntries.length
      : 0;
  const valAvg =
    validationEntries.length > 0
      ? validationEntries.reduce((s, e) => s + e.realizedPnL, 0) / validationEntries.length
      : 0;

  const persistence = runStructuralPersistenceValidation(allAuditEntries, rehydratedIds);
  const fineBlock = persistence.finePersistence;
  const targetSubsetKey = `${NARROW_CHALLENGER_TARGET.pairKey}|${NARROW_CHALLENGER_TARGET.capturableEdgeBucket}|${NARROW_CHALLENGER_TARGET.fillRatioBucket}`;
  const fineSubset = fineBlock.eligibleSubsets.find((s) => s.subsetKey === targetSubsetKey);
  const persistenceMaintained = fineSubset?.persistentLeastBad ?? undefined;
  const eligibleForNarrowChallenger = fineSubset?.eligibleForNarrowChallenger ?? undefined;
  const currentSubsetMatchesTargetDefinition =
    narrowEntries.length >= 5
      ? narrowEntries.every(
          (e) =>
            e.pairKey === NARROW_CHALLENGER_TARGET.pairKey &&
            (e.capturableEdgeAtEntry ?? 0) >= 0.02 &&
            (e.capturableEdgeAtEntry ?? 0) < 0.05 &&
            (e.fillRatio ?? 0) >= 0.25 &&
            (e.fillRatio ?? 0) < 0.5
        )
      : undefined;

  return {
    profileId: NARROW_PROFILE_ID,
    targetPairKey: NARROW_CHALLENGER_TARGET.pairKey,
    targetCapturableEdgeBucket: NARROW_CHALLENGER_TARGET.capturableEdgeBucket,
    targetFillRatioBucket: NARROW_CHALLENGER_TARGET.fillRatioBucket,
    opportunitiesSeen: c.opportunitiesSeen,
    rejectedByPairMismatch: c.rejectedByPairMismatch,
    rejectedByEdgeBucketMismatch: c.rejectedByEdgeBucketMismatch,
    rejectedByFillBucketMismatch: c.rejectedByFillBucketMismatch,
    passedAllNarrowFilters: c.passedAllNarrowFilters,
    openedTradeCount: openedCount,
    closedTradeCount: closedOfficial.length,
    avgRealizedPnL: avgPnL,
    medianRealizedPnL: median(pnls),
    totalRealizedPnL: totalPnL,
    avgCapturableEdgeOpened: avgCapturable,
    avgFillRatioOpened: avgFill,
    discoveryCount,
    validationCount,
    discoveryAvgPnL: discAvg,
    validationAvgPnL: valAvg,
    persistenceMaintained: narrowEntries.length >= 10 ? persistenceMaintained : undefined,
    eligibleForNarrowChallenger: narrowEntries.length >= 10 ? eligibleForNarrowChallenger : undefined,
    currentSubsetMatchesTargetDefinition,
  };
}

export interface NarrowChallengerComparisonResult {
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

function medianFromTrades(trades: Array<{ realizedPnL: number; fillRatio?: number | null; capturableEdgeAtEntry?: number }>): number {
  const pnls = trades.map((t) => t.realizedPnL);
  if (pnls.length === 0) return 0;
  const sorted = [...pnls].sort((a, b) => a - b);
  const m = Math.floor(pnls.length / 2);
  return pnls.length % 2 ? sorted[m]! : (sorted[m - 1]! + sorted[m]!) / 2;
}

export function getNarrowChallengerComparison(
  profiles: ShadowProfileState[],
  narrowDiagnostics: NarrowChallengerDiagnosticsResult | null
): NarrowChallengerComparisonResult | null {
  const baseline = profiles.find((p) => p.profileId === BASELINE_PROFILE_ID);
  const challenger = profiles.find((p) => p.profileId === NARROW_PROFILE_ID);
  if (!narrowDiagnostics) return null;

  const baselineClosed = baseline?.closedTrades ?? [];
  const challengerClosed =
    narrowDiagnostics != null
      ? (challenger?.closedTrades ?? []).filter((t) => t.narrowFilterMatchAtOpen === true)
      : (challenger?.closedTrades ?? []);

  const baselineAvgPnL =
    baselineClosed.length > 0
      ? baselineClosed.reduce((s, t) => s + t.realizedPnL, 0) / baselineClosed.length
      : 0;
  const baselineTotal = baselineClosed.reduce((s, t) => s + t.realizedPnL, 0);
  const baselineAvgFill =
    baselineClosed.length > 0
      ? baselineClosed.reduce((s, t) => s + (t.fillRatio ?? 0), 0) / baselineClosed.length
      : 0;
  const baselineAvgEdge =
    baselineClosed.length > 0
      ? baselineClosed.reduce((s, t) => s + (t.capturableEdgeAtEntry ?? 0), 0) / baselineClosed.length
      : 0;

  return {
    baselineProfileId: BASELINE_PROFILE_ID,
    challengerProfileId: NARROW_PROFILE_ID,
    baselineClosed: baselineClosed.length,
    challengerClosed: challengerClosed.length,
    baselineAvgRealizedPnL: baselineAvgPnL,
    challengerAvgRealizedPnL: narrowDiagnostics.avgRealizedPnL,
    baselineMedianRealizedPnL: medianFromTrades(baselineClosed),
    challengerMedianRealizedPnL: narrowDiagnostics.medianRealizedPnL,
    baselineTotalRealizedPnL: baselineTotal,
    challengerTotalRealizedPnL: narrowDiagnostics.totalRealizedPnL,
    baselineAvgFillRatio: baselineAvgFill,
    challengerAvgFillRatio: narrowDiagnostics.avgFillRatioOpened,
    baselineAvgCapturableEdgeAtEntry: baselineAvgEdge,
    challengerAvgCapturableEdgeAtEntry: narrowDiagnostics.avgCapturableEdgeOpened,
    sameUniverseNote:
      challengerClosed.length < 5
        ? "Amostra do challenger insuficiente para comparação objetiva (mín. 5 trades fechados)."
        : "Challenger opera apenas no subset 562794+567561 | edge 2-5% | fill 0.25-0.5; baseline opera no universo total.",
  };
}
