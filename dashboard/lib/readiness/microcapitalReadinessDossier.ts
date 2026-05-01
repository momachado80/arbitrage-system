/**
 * Microcapital Readiness Gate — dossier types.
 *
 * Pure data shapes consumed by the analyzer, the API endpoint, the CLI script
 * and the executive summary renderer. NO side effects, NO real execution.
 *
 * Policy preserved: shadow_only_no_api_submit_v1.
 *
 * "microcapital" = technical/paper readiness only. NOT a green light for real
 * trading, capital activation, signing, or any external submission.
 */

export type ReadinessVerdict =
  | "APPROVED_FOR_SHADOW_READINESS"
  | "APPROVED_WITH_REMARKS"
  | "BLOCKED_TECHNICAL_RISK"
  | "BLOCKED_ECONOMIC_RISK"
  | "BLOCKED_COMPLIANCE_RISK"
  | "BLOCKED_SAFETY_RISK";

export type ConfidenceLevel = "LOW" | "MEDIUM" | "HIGH";

export interface SystemIdentity {
  projectName: string;
  generatedAt: string;
  gitCommit: string | null;
  paperStateDir: string;
  executionMode: string;
  effectiveMode: string;
  workerName: string;
}

export interface SafetyStatus {
  shadowOnlyPolicyPresent: boolean;
  apiSubmitAllowedNeverTrue: boolean;
  prohibitedTermsScanPassed: boolean;
  noRealExecutionDetected: boolean;
  dispatcherAuditOnly: boolean;
  payloadSafetyPresent: boolean;
  notes: string[];
}

export interface PaperAssessmentSummaryView {
  paperExecutionAssessmentCount: number;
  paperCyclePositiveCount: number;
  paperBlockedByGateCount: number;
  paperNotViableUnderStressCount: number;
  paperBlockedByExecutionCount: number;
  clobUnavailableCount: number;
  estimatedNetAfterPaperExecutionTotal: number;
  estimatedNetAfterPaperExecutionPositiveTotal: number;
  estimatedNetAfterPaperExecutionMedian: number;
  estimatedNetAfterPaperExecutionBest: number;
  estimatedNetAfterPaperExecutionWorst: number;
  positiveAssessmentRate: number;
  blockedByGateRate: number;
  firstAssessmentAt: string | null;
  lastAssessmentAt: string | null;
}

export type PaperAssessmentEvidenceLabel =
  | "no_assessments_present"
  | "assessments_present_no_positive"
  | "positive_assessments_present_majority_blocked"
  | "positive_assessments_present";

export interface PaperAssessmentEvidence {
  label: PaperAssessmentEvidenceLabel;
  hasAnyAssessment: boolean;
  hasAnyPositiveAssessment: boolean;
  majorityBlockedByGate: boolean;
}

export interface EconomicReadiness {
  totalCyclesObserved: number;
  /** Number of fully closed paper trades observed (NOT assessments). */
  closedPaperCycles: number;
  positiveCycles: number;
  negativeCycles: number;
  positiveCycleRate: number;
  grossEdgeTotal: number;
  netEdgeTotal: number;
  estimatedFeesTotal: number;
  estimatedSlippageTotal: number;
  netAfterFeesAndSlippage: number;
  medianNetEdge: number;
  worstCycleNetEdge: number;
  bestCycleNetEdge: number;
  maxDrawdownPaper: number;
  profitFactorPaper: number;
  falsePositiveEstimate: number | null;
  confidenceLevel: ConfidenceLevel;
  /** Aggregate of paper-execution assessments (worker-emitted JSONL). NEVER counted as realized PnL. */
  paperExecutionAssessments: PaperAssessmentSummaryView;
  paperAssessmentEvidence: PaperAssessmentEvidence;
  /** Paper cycle lifecycle counters derived from assessments + markout followups. */
  openedPaperCycles: number;
  /** Paper cycles that have been formally CLOSED via markout evidence. NOT realized PnL. */
  closedPaperCyclesViaMarkout: number;
  insufficientEvidencePaperCycles: number;
  /** Sum of simulated PnL across markout-closed cycles (paper-only, never capital). */
  simulatedClosedCycleNetTotal: number;
  simulatedClosedCyclePositiveCount: number;
  simulatedClosedCycleNegativeCount: number;
  markoutBackedCycleCount: number;
  economicVerdict: ReadinessVerdict;
  notes: string[];
}

export interface ExecutionRealism {
  averageSignalToEnqueueLatencyMs: number | null;
  p95SignalToEnqueueLatencyMs: number | null;
  averageDispatchLatencyMs: number | null;
  p95DispatchLatencyMs: number | null;
  staleSignalCount: number;
  priceMovedAgainstSignalCount: number;
  priceMovedInFavorCount: number;
  simulatedFillQuality: "insufficient_data" | "poor" | "fair" | "good";
  simulatedSlippageBps: number;
  markout5s: number | null;
  markout30s: number | null;
  markout60s: number | null;
  realismVerdict: ReadinessVerdict;
  notes: string[];
}

export type ProbeStatus = "pass" | "fail" | "insufficient_evidence";

export interface ReliabilityReadiness {
  restartRecoveryPassed: boolean;
  idempotencyPassed: boolean;
  duplicatePreventionPassed: boolean;
  leaseRecoveryPassed: boolean;
  retryFinitePassed: boolean;
  permanentErrorsBecomeDeadPassed: boolean;
  circuitBreakerPassed: boolean;
  workerSurvivesDispatcherFailurePassed: boolean;
  /** Tri-state per-probe status — pass / fail / insufficient_evidence. */
  probes: {
    restartRecovery: ProbeStatus;
    idempotency: ProbeStatus;
    duplicatePrevention: ProbeStatus;
    leaseRecovery: ProbeStatus;
    retryFinite: ProbeStatus;
    permanentErrorsBecomeDead: ProbeStatus;
    circuitBreaker: ProbeStatus;
    workerSurvivesDispatcherFailure: ProbeStatus;
  };
  reliabilityVerdict: ReadinessVerdict;
  notes: string[];
}

export interface ObservabilityReadiness {
  hasJsonLogs: boolean;
  hasShadowAudit: boolean;
  hasPaperTrades: boolean;
  hasAnalyticsEndpoint: boolean;
  hasSystemEndpoint: boolean;
  hasTradesEndpoint: boolean;
  canReconcileCycleToPacket: boolean;
  canReconcilePacketToDispatch: boolean;
  canReconcileSignalToPaperTrade: boolean;
  observabilityVerdict: ReadinessVerdict;
  notes: string[];
}

export interface RiskLimitsSimulated {
  maxPaperLossPerCycle: number;
  maxPaperDailyLoss: number;
  maxConsecutiveNegativeCycles: number;
  killSwitchSimulated: boolean;
  cooldownAfterLossSimulated: boolean;
  /** Whether the simulated cooldown is currently engaged (paper-only state). */
  cooldownActiveNow: boolean;
  /** Epoch ms at which the simulated cooldown expires; null if not active. */
  cooldownActiveUntilMs: number | null;
  /** Tri-state evidence for the cooldown mechanism. */
  cooldownEvidence: ProbeStatus;
  exposureLimitSimulated: boolean;
  riskVerdict: ReadinessVerdict;
  notes: string[];
}

export interface MicrocapitalReadinessDossier {
  /** Always "shadow_only_no_api_submit_v1". */
  policyTag: string;
  systemIdentity: SystemIdentity;
  safetyStatus: SafetyStatus;
  economicReadiness: EconomicReadiness;
  executionRealism: ExecutionRealism;
  reliabilityReadiness: ReliabilityReadiness;
  observabilityReadiness: ObservabilityReadiness;
  riskLimitsSimulated: RiskLimitsSimulated;
  finalVerdict: ReadinessVerdict;
  blockers: string[];
  warnings: string[];
  nextActions: string[];
}

export const READINESS_FINAL_PRECEDENCE: ReadinessVerdict[] = [
  "BLOCKED_SAFETY_RISK",
  "BLOCKED_COMPLIANCE_RISK",
  "BLOCKED_ECONOMIC_RISK",
  "BLOCKED_TECHNICAL_RISK",
  "APPROVED_WITH_REMARKS",
  "APPROVED_FOR_SHADOW_READINESS",
];

export function pickWorstVerdict(verdicts: ReadinessVerdict[]): ReadinessVerdict {
  if (verdicts.length === 0) return "APPROVED_FOR_SHADOW_READINESS";
  for (const v of READINESS_FINAL_PRECEDENCE) {
    if (verdicts.includes(v)) return v;
  }
  return "APPROVED_FOR_SHADOW_READINESS";
}

export function isBlockedVerdict(v: ReadinessVerdict): boolean {
  return v.startsWith("BLOCKED_");
}
