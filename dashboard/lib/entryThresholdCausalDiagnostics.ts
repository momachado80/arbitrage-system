/**
 * Entry Threshold Causal Diagnostics — causal instrumentation for entrycal_bind vs exitrefine.
 * Answers: is minNetCapturableEdgeToTrade applied? What metric is compared? Why no divergence?
 * In-memory only. No business logic changes.
 */

const EXITREFINE = "shadow_1000_adapt_captrade_exitrefine_v1";
const ENTRYCAL_BIND = "shadow_1000_adapt_captrade_exitrefine_entrycal_bind_v1";
const TRACKED_PROFILES = new Set([EXITREFINE, ENTRYCAL_BIND]);
const MAX_DIVERGENT_SAMPLES = 50;
const MAX_SAME_CYCLE_RECORDS = 5000;

interface ThresholdDecision {
  profileId: string;
  opportunityId: string;
  pairKey: string;
  metricCompared: number; // netEdgeAfterImpact (value actually compared to threshold)
  threshold: number;
  decision: "accepted" | "rejected";
  rejectionReason?: string;
  cycleBucket: number;
}

/** Per (opportunityId, cycleBucket): decisions from both profiles */
const decisionsByOppCycle = new Map<string, ThresholdDecision[]>();

/** Per profile: threshold evaluation counters */
const thresholdCheckCounters = new Map<
  string,
  { evaluated: number; passed: number; failed: number }
>();

function ensureCounters(profileId: string) {
  let c = thresholdCheckCounters.get(profileId);
  if (!c) {
    c = { evaluated: 0, passed: 0, failed: 0 };
    thresholdCheckCounters.set(profileId, c);
  }
  return c;
}

function pruneOldRecords() {
  if (decisionsByOppCycle.size <= MAX_SAME_CYCLE_RECORDS) return;
  const entries = Array.from(decisionsByOppCycle.entries());
  entries.sort((a, b) => {
    const bucketA = parseInt(a[0].split("|")[1] ?? "0", 10);
    const bucketB = parseInt(b[0].split("|")[1] ?? "0", 10);
    return bucketA - bucketB;
  });
  const toRemove = entries.length - MAX_SAME_CYCLE_RECORDS;
  for (let i = 0; i < toRemove && i < entries.length; i++) {
    decisionsByOppCycle.delete(entries[i][0]);
  }
}

/**
 * Record a threshold decision. Call from shadowSimulationService when evaluating entry.
 * metricCompared = netEdgeAfterImpact (the value actually compared to threshold in realisticExecutionEngine).
 */
export function recordThresholdDecision(
  profileId: string,
  opportunityId: string,
  pairKey: string,
  metricCompared: number,
  threshold: number,
  decision: "accepted" | "rejected",
  rejectionReason: string | undefined,
  cycleBucket: number
): void {
  if (!TRACKED_PROFILES.has(profileId)) return;

  const c = ensureCounters(profileId);
  c.evaluated++;
  if (decision === "accepted") c.passed++;
  else c.failed++;

  const key = `${opportunityId}|${cycleBucket}`;
  let recs = decisionsByOppCycle.get(key);
  if (!recs) {
    recs = [];
    decisionsByOppCycle.set(key, recs);
  }
  // Avoid duplicate for same profile (e.g. runCycle + evaluateOpportunity both firing)
  const existing = recs.find((r) => r.profileId === profileId);
  if (existing) return;
  recs.push({
    profileId,
    opportunityId,
    pairKey,
    metricCompared,
    threshold,
    decision,
    rejectionReason,
    cycleBucket,
  });
  pruneOldRecords();
}

export interface EntryThresholdCausalDiagnostics {
  effectiveEntryThresholdByProfile: Record<string, number>;
  opportunitiesEvaluatedByBoth: number;
  comparisonCounts: {
    passedBoth: number;
    passedOnlyExitrefine: number;
    passedOnlyEntrycalBind: number;
    rejectedOnlyByEntrycalBindThreshold: number;
  };
  perProfileCounters: Record<
    string,
    { thresholdCheckEvaluatedCount: number; thresholdCheckPassedCount: number; thresholdCheckFailedCount: number }
  >;
  divergentSamples: Array<{
    opportunityId: string;
    pairKey: string;
    metricCompared: number;
    thresholdExitrefine: number;
    thresholdEntrycalBind: number;
    decisionExitrefine: "accepted" | "rejected";
    decisionEntrycalBind: "accepted" | "rejected";
    rejectionReasonExitrefine?: string;
    rejectionReasonEntrycalBind?: string;
  }>;
  metricUsed: string;
  generatedAt: string;
}

export function getEntryThresholdCausalDiagnostics(
  getProfileConfig: (profileId: string) => { minNetCapturableEdgeToTrade?: number; minCapturableEdgeToTrade?: number } | undefined
): EntryThresholdCausalDiagnostics {
  const effectiveEntryThresholdByProfile: Record<string, number> = {};
  for (const pid of Array.from(TRACKED_PROFILES)) {
    const cfg = getProfileConfig(pid);
    const v = cfg?.minCapturableEdgeToTrade ?? cfg?.minNetCapturableEdgeToTrade;
    effectiveEntryThresholdByProfile[pid] = typeof v === "number" ? v : 0;
  }

  let opportunitiesEvaluatedByBoth = 0;
  let passedBoth = 0;
  let passedOnlyExitrefine = 0;
  let passedOnlyEntrycalBind = 0;
  let rejectedOnlyByEntrycalBindThreshold = 0;
  const divergentSamples: EntryThresholdCausalDiagnostics["divergentSamples"] = [];

  for (const [, recs] of Array.from(decisionsByOppCycle.entries())) {
    const exitRec = recs.find((r) => r.profileId === EXITREFINE);
    const bindRec = recs.find((r) => r.profileId === ENTRYCAL_BIND);
    if (!exitRec || !bindRec) continue;

    opportunitiesEvaluatedByBoth++;

    const exitAccepted = exitRec.decision === "accepted";
    const bindAccepted = bindRec.decision === "accepted";
    const bindRejectedByThreshold =
      !bindAccepted && bindRec.rejectionReason === "net_edge_below_threshold";

    if (exitAccepted && bindAccepted) passedBoth++;
    else if (exitAccepted && !bindAccepted) {
      passedOnlyExitrefine++;
      if (bindRejectedByThreshold) rejectedOnlyByEntrycalBindThreshold++;
      if (divergentSamples.length < MAX_DIVERGENT_SAMPLES) {
        divergentSamples.push({
          opportunityId: exitRec.opportunityId,
          pairKey: exitRec.pairKey,
          metricCompared: bindRec.metricCompared,
          thresholdExitrefine: exitRec.threshold,
          thresholdEntrycalBind: bindRec.threshold,
          decisionExitrefine: "accepted",
          decisionEntrycalBind: "rejected",
          rejectionReasonEntrycalBind: bindRec.rejectionReason,
        });
      }
    } else if (!exitAccepted && bindAccepted) {
      passedOnlyEntrycalBind++;
      if (divergentSamples.length < MAX_DIVERGENT_SAMPLES) {
        divergentSamples.push({
          opportunityId: exitRec.opportunityId,
          pairKey: exitRec.pairKey,
          metricCompared: exitRec.metricCompared,
          thresholdExitrefine: exitRec.threshold,
          thresholdEntrycalBind: bindRec.threshold,
          decisionExitrefine: "rejected",
          decisionEntrycalBind: "accepted",
          rejectionReasonExitrefine: exitRec.rejectionReason,
        });
      }
    }
  }

  const perProfileCounters: EntryThresholdCausalDiagnostics["perProfileCounters"] = {};
  for (const pid of Array.from(TRACKED_PROFILES)) {
    const c = ensureCounters(pid);
    perProfileCounters[pid] = {
      thresholdCheckEvaluatedCount: c.evaluated,
      thresholdCheckPassedCount: c.passed,
      thresholdCheckFailedCount: c.failed,
    };
  }

  return {
    effectiveEntryThresholdByProfile,
    opportunitiesEvaluatedByBoth,
    comparisonCounts: {
      passedBoth,
      passedOnlyExitrefine,
      passedOnlyEntrycalBind,
      rejectedOnlyByEntrycalBindThreshold,
    },
    perProfileCounters,
    divergentSamples,
    metricUsed: "netEdgeAfterImpact (value compared to threshold in realisticExecutionEngine)",
    generatedAt: new Date().toISOString(),
  };
}
