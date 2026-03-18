/**
 * Structural Challenger Diagnostics — observabilidade para challengers estruturais.
 * In-memory only; no persistence.
 */

import type { ShadowProfileState } from "./shadowSimulationStore";
import type { ClosedTradeAuditEntry } from "./shadowClosedTradeAudit";

const STRUCTURAL_PROFILE_IDS = [
  "shadow_1000_structural_narrow_gt5_fill10to25_v1",
  "shadow_1000_structural_narrow_fill10to25_v1",
] as const;

interface ProfileCounters {
  opportunitiesSeen: number;
  rejectedByPairMismatch: number;
  rejectedByFillBucketMismatch: number;
  rejectedByEdgeBucketMismatch: number;
  passedAllStructuralFilters: number;
}

const counters = new Map<string, ProfileCounters>();

function ensureProfile(profileId: string): ProfileCounters {
  let c = counters.get(profileId);
  if (!c) {
    c = {
      opportunitiesSeen: 0,
      rejectedByPairMismatch: 0,
      rejectedByFillBucketMismatch: 0,
      rejectedByEdgeBucketMismatch: 0,
      passedAllStructuralFilters: 0,
    };
    counters.set(profileId, c);
  }
  return c;
}

export function recordStructuralOpportunitiesSeen(profileId: string): void {
  if (!STRUCTURAL_PROFILE_IDS.includes(profileId as (typeof STRUCTURAL_PROFILE_IDS)[number])) return;
  const c = ensureProfile(profileId);
  c.opportunitiesSeen++;
}

export function recordStructuralRejectedByPairMismatch(profileId: string): void {
  if (!STRUCTURAL_PROFILE_IDS.includes(profileId as (typeof STRUCTURAL_PROFILE_IDS)[number])) return;
  const c = ensureProfile(profileId);
  c.rejectedByPairMismatch++;
}

export function recordStructuralRejectedByFillBucketMismatch(profileId: string): void {
  if (!STRUCTURAL_PROFILE_IDS.includes(profileId as (typeof STRUCTURAL_PROFILE_IDS)[number])) return;
  const c = ensureProfile(profileId);
  c.rejectedByFillBucketMismatch++;
}

export function recordStructuralRejectedByEdgeBucketMismatch(profileId: string): void {
  if (!STRUCTURAL_PROFILE_IDS.includes(profileId as (typeof STRUCTURAL_PROFILE_IDS)[number])) return;
  const c = ensureProfile(profileId);
  c.rejectedByEdgeBucketMismatch++;
}

export function recordStructuralPassedAllFilters(profileId: string): void {
  if (!STRUCTURAL_PROFILE_IDS.includes(profileId as (typeof STRUCTURAL_PROFILE_IDS)[number])) return;
  const c = ensureProfile(profileId);
  c.passedAllStructuralFilters++;
}

export const BASELINE_PROFILE_ID = "shadow_1000";

export interface StructuralChallengerDiagnosticsBlock {
  profileId: string;
  targetPairKeys: string[];
  targetFillRatioBucket: string;
  targetCapturableEdgeBucket: string | null;
  opportunitiesSeen: number;
  rejectedByPairMismatch: number;
  rejectedByFillBucketMismatch: number;
  rejectedByEdgeBucketMismatch: number;
  passedAllStructuralFilters: number;
  openedTradeCount: number;
  closedTradeCount: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  totalRealizedPnL: number;
  avgCapturableEdgeOpened: number;
  avgFillRatioOpened: number;
  avgObservedEdgeOpened: number;
  discoveryCount: number;
  validationCount: number;
  discoveryAvgPnL: number;
  validationAvgPnL: number;
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const m = Math.floor(arr.length / 2);
  return arr.length % 2 ? sorted[m]! : (sorted[m - 1]! + sorted[m]!) / 2;
}

export function getStructuralChallengerDiagnostics(
  profiles: ShadowProfileState[],
  allAuditEntries: ClosedTradeAuditEntry[],
  profileConfig: { profileId: string; targetPairKeys: string[]; targetFillRatioBucket: string; targetCapturableEdgeBucket: string | null }
): StructuralChallengerDiagnosticsBlock {
  const { profileId, targetPairKeys, targetFillRatioBucket, targetCapturableEdgeBucket } = profileConfig;
  const profile = profiles.find((p) => p.profileId === profileId);
  const c = ensureProfile(profileId);

  const closed = (profile?.closedTrades ?? []).filter(
    (t) => t.status === "closed" && t.closedAt && t.structuralFilterMatchAtOpen !== false
  );
  const openedCount = closed.length + (profile?.activeTrades?.length ?? 0);

  const pnls = closed.map((t) => t.realizedPnL);
  const avgPnL = pnls.length ? pnls.reduce((a, b) => a + b, 0) / pnls.length : 0;
  const totalPnL = pnls.reduce((a, b) => a + b, 0);
  const avgCapturable = closed.length
    ? closed.reduce((s, t) => s + (t.capturableEdgeAtEntry ?? 0), 0) / closed.length
    : 0;
  const fillRatios = closed.map((t) =>
    t.fillRatio != null && typeof t.fillRatio === "number"
      ? t.fillRatio
      : t.requestedCapital != null && t.requestedCapital > 0
        ? (t.filledCapital ?? 0) / t.requestedCapital
        : 0
  );
  const avgFill = fillRatios.length ? fillRatios.reduce((a, b) => a + b, 0) / fillRatios.length : 0;
  const avgObserved = closed.length
    ? closed.reduce((s, t) => s + (t.observedEdgeAtEntry ?? 0), 0) / closed.length
    : 0;

  const structuralEntries = allAuditEntries.filter(
    (e) => e.profileId === profileId && e.structuralFilterMatchAtOpen !== false
  );
  const newEntries = structuralEntries;
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

  return {
    profileId,
    targetPairKeys: [...targetPairKeys],
    targetFillRatioBucket,
    targetCapturableEdgeBucket,
    opportunitiesSeen: c.opportunitiesSeen,
    rejectedByPairMismatch: c.rejectedByPairMismatch,
    rejectedByFillBucketMismatch: c.rejectedByFillBucketMismatch,
    rejectedByEdgeBucketMismatch: c.rejectedByEdgeBucketMismatch,
    passedAllStructuralFilters: c.passedAllStructuralFilters,
    openedTradeCount: openedCount,
    closedTradeCount: closed.length,
    avgRealizedPnL: avgPnL,
    medianRealizedPnL: median(pnls),
    totalRealizedPnL: totalPnL,
    avgCapturableEdgeOpened: avgCapturable,
    avgFillRatioOpened: avgFill,
    avgObservedEdgeOpened: avgObserved,
    discoveryCount,
    validationCount,
    discoveryAvgPnL: discAvg,
    validationAvgPnL: valAvg,
  };
}

export interface StructuralChallengerComparisonBlock {
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

export function getStructuralChallengerComparison(
  profiles: ShadowProfileState[],
  structuralDiagnostics: StructuralChallengerDiagnosticsBlock
): StructuralChallengerComparisonBlock {
  const baseline = profiles.find((p) => p.profileId === BASELINE_PROFILE_ID);
  const challenger = profiles.find((p) => p.profileId === structuralDiagnostics.profileId);

  const baselineClosed = baseline?.closedTrades ?? [];
  const challengerClosed =
    challenger?.closedTrades?.filter((t) => t.structuralFilterMatchAtOpen === true) ?? [];

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
    challengerProfileId: structuralDiagnostics.profileId,
    baselineClosed: baselineClosed.length,
    challengerClosed: challengerClosed.length,
    baselineAvgRealizedPnL: baselineAvgPnL,
    challengerAvgRealizedPnL: structuralDiagnostics.avgRealizedPnL,
    baselineMedianRealizedPnL: medianFromTrades(baselineClosed),
    challengerMedianRealizedPnL: structuralDiagnostics.medianRealizedPnL,
    baselineTotalRealizedPnL: baselineTotal,
    challengerTotalRealizedPnL: structuralDiagnostics.totalRealizedPnL,
    baselineAvgFillRatio: baselineAvgFill,
    challengerAvgFillRatio: structuralDiagnostics.avgFillRatioOpened,
    baselineAvgCapturableEdgeAtEntry: baselineAvgEdge,
    challengerAvgCapturableEdgeAtEntry: structuralDiagnostics.avgCapturableEdgeOpened,
    sameUniverseNote:
      challengerClosed.length < 5
        ? `Amostra do challenger insuficiente (mín. 5 trades fechados).`
        : `Challenger opera apenas em pair×fill0.1-0.25${structuralDiagnostics.targetCapturableEdgeBucket ? "×edge>5%" : ""}; baseline opera no universo total.`,
  };
}
