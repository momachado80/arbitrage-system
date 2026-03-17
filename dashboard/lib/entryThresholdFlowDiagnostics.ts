/**
 * Entry Threshold Flow Diagnostics — where does the flow stop before threshold check?
 * Diagnoses why entryThresholdCausalDiagnostics shows zeros (flow never reaches threshold).
 * In-memory only. No business logic changes.
 */

export type SkippedReason =
  | "no_opportunities_in_cycle"
  | "capacity_zero"
  | "confidence_below_threshold"
  | "trade_already_active"
  | "pair_excluded"
  | "no_enabled_profiles"
  | "evaluate_never_called";

interface ProfileFlowCounters {
  cycleReachedProfileExecution: number;
  opportunityReachedProfile: number;
  reachedSimulateRealisticEntry: number;
  reachedThresholdCheck: number;
  skippedBeforeThresholdReason: Record<string, number>;
}

const profileCounters = new Map<string, ProfileFlowCounters>();

// Cycle-level (runCycle)
let cycleCompletedCount = 0;
let lastCycleOpportunitiesMerged = 0;
let lastCycleProfilesCount = 0;

// evaluateOpportunity path
let evaluateOpportunityCallCount = 0;

function ensureProfile(profileId: string): ProfileFlowCounters {
  let c = profileCounters.get(profileId);
  if (!c) {
    c = {
      cycleReachedProfileExecution: 0,
      opportunityReachedProfile: 0,
      reachedSimulateRealisticEntry: 0,
      reachedThresholdCheck: 0,
      skippedBeforeThresholdReason: {},
    };
    profileCounters.set(profileId, c);
  }
  return c;
}

function incrementSkipped(profileId: string, reason: SkippedReason | string): void {
  const c = ensureProfile(profileId);
  const key = reason;
  c.skippedBeforeThresholdReason[key] = (c.skippedBeforeThresholdReason[key] ?? 0) + 1;
}

export function recordCycleCompleted(opportunitiesMerged: number, profilesCount: number): void {
  cycleCompletedCount++;
  lastCycleOpportunitiesMerged = opportunitiesMerged;
  lastCycleProfilesCount = profilesCount;
}

export function recordCycleOpportunityIteration(profileId: string): void {
  const c = ensureProfile(profileId);
  c.cycleReachedProfileExecution++;
}

export function recordOpportunityReachedProfile(profileId: string): void {
  const c = ensureProfile(profileId);
  c.opportunityReachedProfile++;
}

export function recordReachedSimulateRealisticEntry(profileId: string): void {
  const c = ensureProfile(profileId);
  c.reachedSimulateRealisticEntry++;
}

export function recordReachedThresholdCheck(profileId: string): void {
  const c = ensureProfile(profileId);
  c.reachedThresholdCheck++;
}

export function recordSkippedBeforeThreshold(profileId: string, reason: SkippedReason | string): void {
  incrementSkipped(profileId, reason);
}

export function recordNoOpportunitiesInCycle(): void {
  // No profile-specific skip; cycle had no opportunities
}

export function recordEvaluateOpportunityCalled(): void {
  evaluateOpportunityCallCount++;
}

export function recordEvaluateNoEnabledProfiles(): void {
  // Global: no profiles to iterate
}

export interface EntryThresholdFlowDiagnostics {
  cycleLevel: {
    cycleCompletedCount: number;
    lastCycleOpportunitiesMerged: number;
    lastCycleProfilesCount: number;
    evaluateOpportunityCallCount: number;
  };
  byProfile: Record<
    string,
    {
      cycleReachedProfileExecution: number;
      opportunityReachedProfile: number;
      reachedSimulateRealisticEntry: number;
      reachedThresholdCheck: number;
      skippedBeforeThresholdReason: Record<string, number>;
    }
  >;
  generatedAt: string;
}

export function getEntryThresholdFlowDiagnostics(
  profileIds: string[]
): EntryThresholdFlowDiagnostics {
  const byProfile: EntryThresholdFlowDiagnostics["byProfile"] = {};
  const allIds = new Set(profileIds);
  for (const [pid] of Array.from(profileCounters.entries())) {
    allIds.add(pid);
  }
  for (const pid of Array.from(allIds)) {
    const c = ensureProfile(pid);
    byProfile[pid] = {
      cycleReachedProfileExecution: c.cycleReachedProfileExecution,
      opportunityReachedProfile: c.opportunityReachedProfile,
      reachedSimulateRealisticEntry: c.reachedSimulateRealisticEntry,
      reachedThresholdCheck: c.reachedThresholdCheck,
      skippedBeforeThresholdReason: { ...c.skippedBeforeThresholdReason },
    };
  }
  return {
    cycleLevel: {
      cycleCompletedCount,
      lastCycleOpportunitiesMerged,
      lastCycleProfilesCount,
      evaluateOpportunityCallCount,
    },
    byProfile,
    generatedAt: new Date().toISOString(),
  };
}
