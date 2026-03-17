/**
 * Fill Guard Diagnostics — observability for fillguard_v1 entry decisions.
 * Diagnoses why fillguard has totalClosed=0 / openedTradeCount=0.
 * In-memory only; no persistence. No business logic changes.
 */

const FILLGUARD_V1 = "shadow_1000_adapt_captrade_exitrefine_fillguard_v1";
const FILLGUARD_CAL_V1 = "shadow_1000_adapt_captrade_exitrefine_fillguard_cal_v1";
const TRACKED_PROFILES = new Set([FILLGUARD_V1, FILLGUARD_CAL_V1]);
const MAX_REJECTED_FILL_RATIOS = 2000;

interface ProfileCounters {
  evaluatedOpportunityCount: number;
  openedTradeCount: number;
  rejectedByEntryThresholdCount: number;
  rejectedByFillGuardCount: number;
  rejectedByOtherReasonCount: number;
  reachedFillGuardDecisionCount: number;
  failedAtFillGuardCount: number;
  rejectedFillRatios: number[];
}

const counters = new Map<string, ProfileCounters>();

function ensureProfile(profileId: string): ProfileCounters {
  let c = counters.get(profileId);
  if (!c) {
    c = {
      evaluatedOpportunityCount: 0,
      openedTradeCount: 0,
      rejectedByEntryThresholdCount: 0,
      rejectedByFillGuardCount: 0,
      rejectedByOtherReasonCount: 0,
      reachedFillGuardDecisionCount: 0,
      failedAtFillGuardCount: 0,
      rejectedFillRatios: [],
    };
    counters.set(profileId, c);
  }
  return c;
}

export function recordEvaluated(profileId: string): void {
  if (!TRACKED_PROFILES.has(profileId)) return;
  const c = ensureProfile(profileId);
  c.evaluatedOpportunityCount++;
}

export function recordRejectedByEntryThreshold(profileId: string): void {
  if (!TRACKED_PROFILES.has(profileId)) return;
  const c = ensureProfile(profileId);
  c.rejectedByEntryThresholdCount++;
}

export function recordRejectedByFillGuard(profileId: string, fillRatio: number): void {
  if (!TRACKED_PROFILES.has(profileId)) return;
  const c = ensureProfile(profileId);
  c.rejectedByFillGuardCount++;
  c.failedAtFillGuardCount++;
  c.rejectedFillRatios.push(fillRatio);
  if (c.rejectedFillRatios.length > MAX_REJECTED_FILL_RATIOS) {
    c.rejectedFillRatios.shift();
  }
}

export function recordRejectedByOther(profileId: string): void {
  if (!TRACKED_PROFILES.has(profileId)) return;
  const c = ensureProfile(profileId);
  c.rejectedByOtherReasonCount++;
}

export function recordReachedFillGuardDecision(profileId: string): void {
  if (!TRACKED_PROFILES.has(profileId)) return;
  const c = ensureProfile(profileId);
  c.reachedFillGuardDecisionCount++;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const i = Math.max(0, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[i] ?? 0;
}

export interface FillGuardProfileDiagnostics {
  evaluatedOpportunityCount: number;
  openedTradeCount: number;
  rejectedByEntryThresholdCount: number;
  rejectedByFillGuardCount: number;
  rejectedByOtherReasonCount: number;
  reachedFillGuardDecisionCount: number;
  failedAtFillGuardCount: number;
  avgRejectedFillRatio: number;
  minRejectedFillRatio: number;
  maxRejectedFillRatio: number;
  p10RejectedFillRatio: number;
  p50RejectedFillRatio: number;
  p90RejectedFillRatio: number;
}

export interface FillGuardDiagnosticsResult {
  byProfile: Record<string, FillGuardProfileDiagnostics>;
  generatedAt: string;
}

export function getFillGuardDiagnostics(
  profiles: Array<{ profileId: string; closedTrades: unknown[]; activeTrades: unknown[] }>
): FillGuardDiagnosticsResult {
  const byProfile: Record<string, FillGuardProfileDiagnostics> = {};

  for (const profileId of Array.from(TRACKED_PROFILES)) {
    const c = ensureProfile(profileId);
    const p = profiles.find((x) => x.profileId === profileId);
    const opened = (p?.closedTrades?.length ?? 0) + (p?.activeTrades?.length ?? 0);

    const sorted = [...c.rejectedFillRatios].sort((a, b) => a - b);

    byProfile[profileId] = {
      evaluatedOpportunityCount: c.evaluatedOpportunityCount,
      openedTradeCount: opened,
      rejectedByEntryThresholdCount: c.rejectedByEntryThresholdCount,
      rejectedByFillGuardCount: c.rejectedByFillGuardCount,
      rejectedByOtherReasonCount: c.rejectedByOtherReasonCount,
      reachedFillGuardDecisionCount: c.reachedFillGuardDecisionCount,
      failedAtFillGuardCount: c.failedAtFillGuardCount,
      avgRejectedFillRatio:
        c.rejectedFillRatios.length
          ? c.rejectedFillRatios.reduce((a, b) => a + b, 0) / c.rejectedFillRatios.length
          : 0,
      minRejectedFillRatio: c.rejectedFillRatios.length ? Math.min(...c.rejectedFillRatios) : 0,
      maxRejectedFillRatio: c.rejectedFillRatios.length ? Math.max(...c.rejectedFillRatios) : 0,
      p10RejectedFillRatio: percentile(sorted, 10),
      p50RejectedFillRatio: percentile(sorted, 50),
      p90RejectedFillRatio: percentile(sorted, 90),
    };
  }

  return {
    byProfile,
    generatedAt: new Date().toISOString(),
  };
}
