/**
 * Microcapital Readiness Gate — analyzer.
 *
 * Pure function: takes already-collected paper/shadow/dispatcher snapshots and
 * returns a {@link MicrocapitalReadinessDossier}.
 *
 * Strict policy: shadow_only_no_api_submit_v1.
 * NEVER triggers execution. NEVER submits orders. NEVER touches wallet/signer/
 * private key. Reads only.
 */

import type { PaperAnalyticsResult } from "../paperAnalytics";
import type { PaperTrade } from "../paperTypes";

import {
  aggregateExecutionRealism,
  type ExecutionRealismAggregate,
  type ExecutionRealismSample,
} from "./executionRealismHarness";
import {
  isBlockedVerdict,
  pickWorstVerdict,
  type ConfidenceLevel,
  type EconomicReadiness,
  type ExecutionRealism,
  type MicrocapitalReadinessDossier,
  type ObservabilityReadiness,
  type ReadinessVerdict,
  type ProbeStatus,
  type ReliabilityReadiness,
  type RiskLimitsSimulated,
  type SafetyStatus,
  type SystemIdentity,
} from "./microcapitalReadinessDossier";
import {
  MICROCAPITAL_READINESS_THRESHOLDS,
  SHADOW_ONLY_POLICY_TAG,
} from "./microcapitalReadinessThresholds";
import type { ProhibitedTermsScanResult } from "./prohibitedTermsScanner";

export interface ReliabilityProbeInput {
  restartRecoveryPassed: boolean;
  idempotencyPassed: boolean;
  duplicatePreventionPassed: boolean;
  leaseRecoveryPassed: boolean;
  retryFinitePassed: boolean;
  permanentErrorsBecomeDeadPassed: boolean;
  circuitBreakerPassed: boolean;
  workerSurvivesDispatcherFailurePassed: boolean;
  /**
   * Optional per-probe tri-state status. When provided, the analyzer uses these
   * to distinguish "fail" from "insufficient_evidence" (mechanism not yet wired)
   * — preventing honest gaps from being misclassified as hard reliability failures.
   * If a probe is missing here, the boolean *Passed flag governs (legacy path).
   */
  probeStatus?: Partial<{
    restartRecovery: ProbeStatus;
    idempotency: ProbeStatus;
    duplicatePrevention: ProbeStatus;
    leaseRecovery: ProbeStatus;
    retryFinite: ProbeStatus;
    permanentErrorsBecomeDead: ProbeStatus;
    circuitBreaker: ProbeStatus;
    workerSurvivesDispatcherFailure: ProbeStatus;
  }>;
}

export interface ObservabilityProbeInput {
  hasJsonLogs: boolean;
  hasShadowAudit: boolean;
  hasPaperTrades: boolean;
  hasAnalyticsEndpoint: boolean;
  hasSystemEndpoint: boolean;
  hasTradesEndpoint: boolean;
  canReconcileCycleToPacket: boolean;
  canReconcilePacketToDispatch: boolean;
  canReconcileSignalToPaperTrade: boolean;
}

export interface RiskLimitsProbeInput {
  killSwitchSimulated: boolean;
  cooldownAfterLossSimulated: boolean;
  exposureLimitSimulated: boolean;
  maxConsecutiveNegativeCycles: number;
  maxPaperLossPerCycle: number;
  maxPaperDailyLoss: number;
}

export interface AnalyzerInput {
  systemIdentity: SystemIdentity;
  /** From `getPaperAnalyticsData()`. */
  paperAnalytics: PaperAnalyticsResult | null;
  /** Closed paper trades (oldest → newest). */
  closedTrades: PaperTrade[];
  /** Active paper trades (snapshot only — no submit). */
  activeTrades: PaperTrade[];
  /** Realism samples produced by {@link buildExecutionRealismSample}. */
  realismSamples: ExecutionRealismSample[];
  /** Result of the prohibited-terms scanner. */
  prohibitedTermsScan: ProhibitedTermsScanResult;
  /** Boolean: any source had `apiSubmitAllowed: true` literal. */
  apiSubmitAllowedTrueAnywhere: boolean;
  /** Boolean: dispatcher path is audit_only / no real submission. */
  dispatcherAuditOnly: boolean;
  /** Boolean: defensive payload guard `assertShadowOnlyNoRealSubmission` is wired in. */
  payloadSafetyPresent: boolean;
  reliability: ReliabilityProbeInput;
  observability: ObservabilityProbeInput;
  riskLimits: RiskLimitsProbeInput;
}

function median(nums: number[]): number {
  if (nums.length === 0) return 0;
  const s = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
}

function classifyConfidence(cycleCount: number): ConfidenceLevel {
  const min = MICROCAPITAL_READINESS_THRESHOLDS.MIN_OBSERVED_CYCLES;
  if (cycleCount >= min * 3) return "HIGH";
  if (cycleCount >= min) return "MEDIUM";
  return "LOW";
}

function buildSafetyStatus(input: AnalyzerInput): SafetyStatus {
  const notes: string[] = [];
  const shadowOnlyPolicyPresent = true; // tag is constant in this layer
  const apiSubmitAllowedNeverTrue = !input.apiSubmitAllowedTrueAnywhere;
  const prohibitedTermsScanPassed = input.prohibitedTermsScan.passed;
  const dispatcherAuditOnly = input.dispatcherAuditOnly;
  const payloadSafetyPresent = input.payloadSafetyPresent;

  if (!apiSubmitAllowedNeverTrue) {
    notes.push("apiSubmitAllowed:true detected somewhere in the codebase — SAFETY BLOCK");
  }
  if (!prohibitedTermsScanPassed) {
    const findings = input.prohibitedTermsScan.findings.slice(0, 3);
    notes.push(
      `Prohibited execution terms detected (${input.prohibitedTermsScan.findings.length} findings). ` +
        `First: ${findings.map(f => `${f.matchedTerm}@${f.file}:${f.line}`).join(", ")}`,
    );
  }
  if (!dispatcherAuditOnly) {
    notes.push("Dispatcher is not declared audit_only — SAFETY BLOCK");
  }
  if (!payloadSafetyPresent) {
    notes.push("Defensive payload guard (assertShadowOnlyNoRealSubmission) not wired");
  }

  return {
    shadowOnlyPolicyPresent,
    apiSubmitAllowedNeverTrue,
    prohibitedTermsScanPassed,
    noRealExecutionDetected:
      apiSubmitAllowedNeverTrue && prohibitedTermsScanPassed && dispatcherAuditOnly,
    dispatcherAuditOnly,
    payloadSafetyPresent,
    notes,
  };
}

function safetyVerdict(safety: SafetyStatus): ReadinessVerdict {
  if (
    !safety.shadowOnlyPolicyPresent ||
    !safety.apiSubmitAllowedNeverTrue ||
    !safety.prohibitedTermsScanPassed ||
    !safety.dispatcherAuditOnly
  ) {
    return "BLOCKED_SAFETY_RISK";
  }
  if (!safety.payloadSafetyPresent) return "APPROVED_WITH_REMARKS";
  return "APPROVED_FOR_SHADOW_READINESS";
}

interface CyclePnL {
  pnl: number;
  netEdge: number;
  fees: number;
  slippage: number;
}

function tradesToCycleSeries(closedTrades: PaperTrade[]): CyclePnL[] {
  return closedTrades
    .filter(t => t.status === "closed")
    .map(t => {
      const filled = Number.isFinite(t.filledCapital) ? t.filledCapital : 0;
      const fees = filled * MICROCAPITAL_READINESS_THRESHOLDS.DEFAULT_PAPER_FEE_RATE;
      const slippage =
        filled *
        (MICROCAPITAL_READINESS_THRESHOLDS.DEFAULT_PAPER_SLIPPAGE_BPS / 10_000);
      return {
        pnl: Number.isFinite(t.realizedPnL) ? t.realizedPnL : 0,
        netEdge: Number.isFinite(t.netEdgeAtEntry) ? t.netEdgeAtEntry * filled : 0,
        fees,
        slippage,
      };
    });
}

function buildEconomicReadiness(input: AnalyzerInput): EconomicReadiness {
  const notes: string[] = [];
  const series = tradesToCycleSeries(input.closedTrades);
  const total = series.length;
  const positives = series.filter(c => c.pnl > 0).length;
  const negatives = series.filter(c => c.pnl < 0).length;
  const positiveRate = total > 0 ? positives / total : 0;

  const grossEdgeTotal = series.reduce((s, c) => s + c.netEdge, 0); // best-effort
  const feesTotal = series.reduce((s, c) => s + c.fees, 0);
  const slipTotal = series.reduce((s, c) => s + c.slippage, 0);
  const pnlTotal = series.reduce((s, c) => s + c.pnl, 0);
  const netAfterFeesAndSlippage = pnlTotal - feesTotal - slipTotal;

  const pnls = series.map(c => c.pnl);
  const med = median(pnls);
  const worst = pnls.length > 0 ? Math.min(...pnls) : 0;
  const best = pnls.length > 0 ? Math.max(...pnls) : 0;

  const maxDrawdownPaper = input.paperAnalytics?.maxDrawdown ?? 0;
  const profitFactorPaper = input.paperAnalytics?.profitFactor ?? 0;

  const confidenceLevel = classifyConfidence(total);

  const T = MICROCAPITAL_READINESS_THRESHOLDS;
  let verdict: ReadinessVerdict = "APPROVED_FOR_SHADOW_READINESS";

  if (total < T.MIN_OBSERVED_CYCLES) {
    notes.push(
      `Only ${total} closed cycles observed (< ${T.MIN_OBSERVED_CYCLES}). Economic confidence: ${confidenceLevel}`,
    );
    verdict = "APPROVED_WITH_REMARKS";
  }
  if (total >= T.MIN_OBSERVED_CYCLES && positiveRate < T.MIN_POSITIVE_CYCLE_RATE) {
    notes.push(
      `Positive cycle rate ${(positiveRate * 100).toFixed(1)}% below threshold ${(T.MIN_POSITIVE_CYCLE_RATE * 100).toFixed(0)}%`,
    );
    verdict = "BLOCKED_ECONOMIC_RISK";
  }
  if (
    total >= T.MIN_OBSERVED_CYCLES &&
    netAfterFeesAndSlippage < T.MIN_NET_AFTER_FEES_AND_SLIPPAGE
  ) {
    notes.push(
      `Net edge after fees + slippage = ${netAfterFeesAndSlippage.toFixed(2)} below threshold ${T.MIN_NET_AFTER_FEES_AND_SLIPPAGE}`,
    );
    verdict = "BLOCKED_ECONOMIC_RISK";
  }
  if (maxDrawdownPaper > T.MAX_DRAWDOWN_PAPER_ALLOWED_ABS) {
    notes.push(
      `Paper drawdown ${maxDrawdownPaper.toFixed(2)} exceeds allowed ${T.MAX_DRAWDOWN_PAPER_ALLOWED_ABS}`,
    );
    verdict = pickWorstVerdict([verdict, "BLOCKED_ECONOMIC_RISK"]);
  }

  if (total > 0) {
    const positiveSorted = [...series.filter(c => c.pnl > 0)].sort((a, b) => b.pnl - a.pnl);
    const totalPositiveEdge = positiveSorted.reduce((s, c) => s + c.pnl, 0);
    if (totalPositiveEdge > 0) {
      const top1Share = positiveSorted[0]?.pnl / totalPositiveEdge || 0;
      const top3Share =
        positiveSorted.slice(0, 3).reduce((s, c) => s + c.pnl, 0) / totalPositiveEdge || 0;
      if (top1Share > T.MAX_TOP_CYCLE_EDGE_SHARE) {
        notes.push(
          `Edge concentration: top-1 cycle holds ${(top1Share * 100).toFixed(1)}% of positive PnL`,
        );
        verdict = pickWorstVerdict([verdict, "APPROVED_WITH_REMARKS"]);
      }
      if (top3Share > T.MAX_TOP_3_CYCLES_EDGE_SHARE) {
        notes.push(
          `Edge concentration: top-3 cycles hold ${(top3Share * 100).toFixed(1)}% of positive PnL`,
        );
        verdict = pickWorstVerdict([verdict, "APPROVED_WITH_REMARKS"]);
      }
    }
  }

  return {
    totalCyclesObserved: total,
    positiveCycles: positives,
    negativeCycles: negatives,
    positiveCycleRate: positiveRate,
    grossEdgeTotal,
    netEdgeTotal: pnlTotal,
    estimatedFeesTotal: feesTotal,
    estimatedSlippageTotal: slipTotal,
    netAfterFeesAndSlippage,
    medianNetEdge: med,
    worstCycleNetEdge: worst,
    bestCycleNetEdge: best,
    maxDrawdownPaper,
    profitFactorPaper,
    falsePositiveEstimate: null,
    confidenceLevel,
    economicVerdict: verdict,
    notes,
  };
}

function buildExecutionRealism(
  input: AnalyzerInput,
  agg: ExecutionRealismAggregate,
): ExecutionRealism {
  const notes: string[] = [];
  const T = MICROCAPITAL_READINESS_THRESHOLDS;
  let verdict: ReadinessVerdict = "APPROVED_FOR_SHADOW_READINESS";

  if (agg.sampleCount === 0) {
    notes.push("No execution realism samples recorded yet");
    verdict = "APPROVED_WITH_REMARKS";
  }

  const staleRate = agg.sampleCount > 0 ? agg.staleSignalCount / agg.sampleCount : 0;
  if (staleRate > T.MAX_STALE_SIGNAL_RATE) {
    notes.push(
      `Stale signal rate ${(staleRate * 100).toFixed(1)}% exceeds ${(T.MAX_STALE_SIGNAL_RATE * 100).toFixed(0)}%`,
    );
    verdict = pickWorstVerdict([verdict, "BLOCKED_TECHNICAL_RISK"]);
  }

  if (
    agg.averageSignalToEnqueueLatencyMs !== null &&
    agg.averageSignalToEnqueueLatencyMs > T.MAX_AVG_SIGNAL_TO_ENQUEUE_MS
  ) {
    notes.push(
      `Avg signal→enqueue latency ${agg.averageSignalToEnqueueLatencyMs.toFixed(0)}ms > ${T.MAX_AVG_SIGNAL_TO_ENQUEUE_MS}ms`,
    );
    verdict = pickWorstVerdict([verdict, "APPROVED_WITH_REMARKS"]);
  }
  if (
    agg.p95SignalToEnqueueLatencyMs !== null &&
    agg.p95SignalToEnqueueLatencyMs > T.MAX_P95_SIGNAL_TO_ENQUEUE_MS
  ) {
    notes.push(
      `p95 signal→enqueue latency ${agg.p95SignalToEnqueueLatencyMs.toFixed(0)}ms > ${T.MAX_P95_SIGNAL_TO_ENQUEUE_MS}ms`,
    );
    verdict = pickWorstVerdict([verdict, "APPROVED_WITH_REMARKS"]);
  }

  let fillQuality: ExecutionRealism["simulatedFillQuality"] = "insufficient_data";
  if (agg.sampleCount > 0) {
    if (staleRate < 0.05 && agg.priceMovedAgainstSignalCount === 0) fillQuality = "good";
    else if (staleRate < 0.1) fillQuality = "fair";
    else fillQuality = "poor";
  }

  return {
    averageSignalToEnqueueLatencyMs: agg.averageSignalToEnqueueLatencyMs,
    p95SignalToEnqueueLatencyMs: agg.p95SignalToEnqueueLatencyMs,
    averageDispatchLatencyMs: agg.averageDispatchLatencyMs,
    p95DispatchLatencyMs: agg.p95DispatchLatencyMs,
    staleSignalCount: agg.staleSignalCount,
    priceMovedAgainstSignalCount: agg.priceMovedAgainstSignalCount,
    priceMovedInFavorCount: agg.priceMovedInFavorCount,
    simulatedFillQuality: fillQuality,
    simulatedSlippageBps: agg.averageSlippageBps,
    markout5s: agg.markout5s,
    markout30s: agg.markout30s,
    markout60s: agg.markout60s,
    realismVerdict: verdict,
    notes,
  };
}

function buildReliability(input: AnalyzerInput): ReliabilityReadiness {
  const r = input.reliability;
  const notes: string[] = [];

  function status(name: keyof NonNullable<typeof r.probeStatus>, passed: boolean): ProbeStatus {
    const explicit = r.probeStatus?.[name];
    if (explicit) return explicit;
    return passed ? "pass" : "fail";
  }

  const probes = {
    restartRecovery: status("restartRecovery", r.restartRecoveryPassed),
    idempotency: status("idempotency", r.idempotencyPassed),
    duplicatePrevention: status("duplicatePrevention", r.duplicatePreventionPassed),
    leaseRecovery: status("leaseRecovery", r.leaseRecoveryPassed),
    retryFinite: status("retryFinite", r.retryFinitePassed),
    permanentErrorsBecomeDead: status(
      "permanentErrorsBecomeDead",
      r.permanentErrorsBecomeDeadPassed,
    ),
    circuitBreaker: status("circuitBreaker", r.circuitBreakerPassed),
    workerSurvivesDispatcherFailure: status(
      "workerSurvivesDispatcherFailure",
      r.workerSurvivesDispatcherFailurePassed,
    ),
  };

  const failed = Object.entries(probes)
    .filter(([, s]) => s === "fail")
    .map(([k]) => k);
  const insufficient = Object.entries(probes)
    .filter(([, s]) => s === "insufficient_evidence")
    .map(([k]) => k);

  let verdict: ReadinessVerdict = "APPROVED_FOR_SHADOW_READINESS";
  if (failed.length > 0) {
    notes.push(`Failed reliability probes: ${failed.join(", ")}`);
    const blockingSubset = ["restartRecovery", "idempotency", "duplicatePrevention"];
    if (failed.some(f => blockingSubset.includes(f))) verdict = "BLOCKED_TECHNICAL_RISK";
    else verdict = "APPROVED_WITH_REMARKS";
  }
  if (insufficient.length > 0) {
    notes.push(`Insufficient evidence (mechanism not wired): ${insufficient.join(", ")}`);
    if (verdict === "APPROVED_FOR_SHADOW_READINESS") verdict = "APPROVED_WITH_REMARKS";
  }

  return {
    restartRecoveryPassed: r.restartRecoveryPassed,
    idempotencyPassed: r.idempotencyPassed,
    duplicatePreventionPassed: r.duplicatePreventionPassed,
    leaseRecoveryPassed: r.leaseRecoveryPassed,
    retryFinitePassed: r.retryFinitePassed,
    permanentErrorsBecomeDeadPassed: r.permanentErrorsBecomeDeadPassed,
    circuitBreakerPassed: r.circuitBreakerPassed,
    workerSurvivesDispatcherFailurePassed: r.workerSurvivesDispatcherFailurePassed,
    probes,
    reliabilityVerdict: verdict,
    notes,
  };
}

function buildObservability(input: AnalyzerInput): ObservabilityReadiness {
  const o = input.observability;
  const notes: string[] = [];
  const missing: string[] = [];
  if (!o.hasJsonLogs) missing.push("hasJsonLogs");
  if (!o.hasShadowAudit) missing.push("hasShadowAudit");
  if (!o.hasPaperTrades) missing.push("hasPaperTrades");
  if (!o.hasAnalyticsEndpoint) missing.push("hasAnalyticsEndpoint");
  if (!o.hasSystemEndpoint) missing.push("hasSystemEndpoint");
  if (!o.hasTradesEndpoint) missing.push("hasTradesEndpoint");
  if (!o.canReconcileCycleToPacket) missing.push("canReconcileCycleToPacket");
  if (!o.canReconcilePacketToDispatch) missing.push("canReconcilePacketToDispatch");
  if (!o.canReconcileSignalToPaperTrade) missing.push("canReconcileSignalToPaperTrade");

  let verdict: ReadinessVerdict = "APPROVED_FOR_SHADOW_READINESS";
  if (missing.length > 0) {
    notes.push(`Missing observability: ${missing.join(", ")}`);
    if (!o.hasShadowAudit || !o.hasPaperTrades) verdict = "BLOCKED_TECHNICAL_RISK";
    else verdict = "APPROVED_WITH_REMARKS";
  }
  return {
    ...o,
    observabilityVerdict: verdict,
    notes,
  };
}

function buildRiskLimits(input: AnalyzerInput): RiskLimitsSimulated {
  const r = input.riskLimits;
  const notes: string[] = [];
  let verdict: ReadinessVerdict = "APPROVED_FOR_SHADOW_READINESS";

  if (!r.killSwitchSimulated) {
    notes.push("Simulated kill-switch not present");
    verdict = "APPROVED_WITH_REMARKS";
  }
  if (!r.cooldownAfterLossSimulated) {
    notes.push("Simulated cooldown-after-loss not present");
    verdict = "APPROVED_WITH_REMARKS";
  }
  if (!r.exposureLimitSimulated) {
    notes.push("Simulated exposure-limit not present");
    verdict = "APPROVED_WITH_REMARKS";
  }

  return {
    maxPaperLossPerCycle: r.maxPaperLossPerCycle,
    maxPaperDailyLoss: r.maxPaperDailyLoss,
    maxConsecutiveNegativeCycles: r.maxConsecutiveNegativeCycles,
    killSwitchSimulated: r.killSwitchSimulated,
    cooldownAfterLossSimulated: r.cooldownAfterLossSimulated,
    exposureLimitSimulated: r.exposureLimitSimulated,
    riskVerdict: verdict,
    notes,
  };
}

export function analyzeMicrocapitalReadiness(
  input: AnalyzerInput,
): MicrocapitalReadinessDossier {
  const safetyStatus = buildSafetyStatus(input);
  const economicReadiness = buildEconomicReadiness(input);
  const realismAggregate = aggregateExecutionRealism(input.realismSamples);
  const executionRealism = buildExecutionRealism(input, realismAggregate);
  const reliabilityReadiness = buildReliability(input);
  const observabilityReadiness = buildObservability(input);
  const riskLimitsSimulated = buildRiskLimits(input);

  const safetyVerd = safetyVerdict(safetyStatus);

  const finalVerdict = pickWorstVerdict([
    safetyVerd,
    economicReadiness.economicVerdict,
    executionRealism.realismVerdict,
    reliabilityReadiness.reliabilityVerdict,
    observabilityReadiness.observabilityVerdict,
    riskLimitsSimulated.riskVerdict,
  ]);

  const blockers: string[] = [];
  const warnings: string[] = [];
  const nextActions: string[] = [];

  function classify(prefix: string, v: ReadinessVerdict, ns: string[]): void {
    if (isBlockedVerdict(v)) blockers.push(...ns.map(n => `${prefix}: ${n}`));
    else warnings.push(...ns.map(n => `${prefix}: ${n}`));
  }

  classify("safety", safetyVerd, safetyStatus.notes);
  classify("economic", economicReadiness.economicVerdict, economicReadiness.notes);
  classify("realism", executionRealism.realismVerdict, executionRealism.notes);
  classify("reliability", reliabilityReadiness.reliabilityVerdict, reliabilityReadiness.notes);
  classify(
    "observability",
    observabilityReadiness.observabilityVerdict,
    observabilityReadiness.notes,
  );
  classify("risk", riskLimitsSimulated.riskVerdict, riskLimitsSimulated.notes);

  if (economicReadiness.totalCyclesObserved < MICROCAPITAL_READINESS_THRESHOLDS.MIN_OBSERVED_CYCLES) {
    nextActions.push(
      `Accumulate at least ${MICROCAPITAL_READINESS_THRESHOLDS.MIN_OBSERVED_CYCLES} closed paper cycles before re-running gate`,
    );
  }
  if (executionRealism.markout5s === null) {
    nextActions.push("Wire 5s/30s/60s markout sampling on paper-only path");
  }
  if (!reliabilityReadiness.idempotencyPassed) {
    nextActions.push("Add idempotency probe (paper) covering duplicate packet rejection");
  }
  if (!reliabilityReadiness.restartRecoveryPassed) {
    nextActions.push("Add restart-recovery probe (paper) covering queued/in-flight packets");
  }
  if (isBlockedVerdict(finalVerdict)) {
    nextActions.push("Resolve all blockers before any further microcapital readiness review");
  }

  return {
    policyTag: SHADOW_ONLY_POLICY_TAG,
    systemIdentity: input.systemIdentity,
    safetyStatus,
    economicReadiness,
    executionRealism,
    reliabilityReadiness,
    observabilityReadiness,
    riskLimitsSimulated,
    finalVerdict,
    blockers,
    warnings,
    nextActions,
  };
}
