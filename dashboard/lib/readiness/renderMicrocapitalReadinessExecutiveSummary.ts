/**
 * Microcapital Readiness Gate — executive summary renderer.
 *
 * Produces a human-readable, plain-text summary of the dossier.
 *
 * Strict policy: shadow_only_no_api_submit_v1. The summary makes NO claim about
 * financial returns, NEVER recommends activating capital, and NEVER recommends
 * real execution.
 */

import type {
  MicrocapitalReadinessDossier,
  ReadinessVerdict,
} from "./microcapitalReadinessDossier";

function fmtPct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}
function fmtNum(n: number, dp = 2): string {
  if (!Number.isFinite(n)) return "n/a";
  return n.toFixed(dp);
}
function fmtMs(n: number | null): string {
  if (n === null || !Number.isFinite(n)) return "n/a";
  return `${Math.round(n)}ms`;
}

const VERDICT_BADGE: Record<ReadinessVerdict, string> = {
  APPROVED_FOR_SHADOW_READINESS: "[OK] APPROVED_FOR_SHADOW_READINESS",
  APPROVED_WITH_REMARKS: "[!]  APPROVED_WITH_REMARKS",
  BLOCKED_TECHNICAL_RISK: "[X]  BLOCKED_TECHNICAL_RISK",
  BLOCKED_ECONOMIC_RISK: "[X]  BLOCKED_ECONOMIC_RISK",
  BLOCKED_COMPLIANCE_RISK: "[X]  BLOCKED_COMPLIANCE_RISK",
  BLOCKED_SAFETY_RISK: "[X]  BLOCKED_SAFETY_RISK",
};

export function renderMicrocapitalReadinessExecutiveSummary(
  d: MicrocapitalReadinessDossier,
): string {
  const lines: string[] = [];

  lines.push("=".repeat(70));
  lines.push("MICROCAPITAL READINESS GATE — EXECUTIVE SUMMARY");
  lines.push("(paper / shadow only — no real execution, no capital activation)");
  lines.push("=".repeat(70));
  lines.push(`Policy: ${d.policyTag}`);
  lines.push(`Project: ${d.systemIdentity.projectName}`);
  lines.push(`Generated: ${d.systemIdentity.generatedAt}`);
  lines.push(`Git: ${d.systemIdentity.gitCommit ?? "n/a"}`);
  lines.push(`Mode: ${d.systemIdentity.executionMode} (effective ${d.systemIdentity.effectiveMode})`);
  lines.push(`Worker: ${d.systemIdentity.workerName}`);
  lines.push(`Paper state dir: ${d.systemIdentity.paperStateDir}`);
  lines.push("");
  lines.push(`FINAL VERDICT: ${VERDICT_BADGE[d.finalVerdict]}`);
  lines.push("");

  lines.push("--- BLOCKERS ---");
  if (d.blockers.length === 0) lines.push("  (none)");
  else d.blockers.forEach(b => lines.push(`  - ${b}`));
  lines.push("");

  lines.push("--- WARNINGS ---");
  if (d.warnings.length === 0) lines.push("  (none)");
  else d.warnings.forEach(w => lines.push(`  - ${w}`));
  lines.push("");

  lines.push("--- KEY EVIDENCE ---");
  lines.push(`  shadowOnlyPolicyPresent: ${d.safetyStatus.shadowOnlyPolicyPresent}`);
  lines.push(`  apiSubmitAllowedNeverTrue: ${d.safetyStatus.apiSubmitAllowedNeverTrue}`);
  lines.push(`  prohibitedTermsScanPassed: ${d.safetyStatus.prohibitedTermsScanPassed}`);
  lines.push(`  dispatcherAuditOnly: ${d.safetyStatus.dispatcherAuditOnly}`);
  lines.push(`  payloadSafetyPresent: ${d.safetyStatus.payloadSafetyPresent}`);
  lines.push("");

  lines.push("--- ECONOMIC METRICS (paper) ---");
  const e = d.economicReadiness;
  lines.push(`  closed paper cycles: ${e.closedPaperCycles}`);
  const a = e.paperExecutionAssessments;
  lines.push(`  paper execution assessments: ${a.paperExecutionAssessmentCount}`);
  lines.push(`  paper positive assessments: ${a.paperCyclePositiveCount}`);
  lines.push(`  blocked by gate: ${a.paperBlockedByGateCount}`);
  lines.push(`  blocked by execution: ${a.paperBlockedByExecutionCount}`);
  lines.push(`  not viable under stress: ${a.paperNotViableUnderStressCount}`);
  lines.push(`  clob unavailable: ${a.clobUnavailableCount}`);
  lines.push(
    `  estimated net after paper execution total: ${fmtNum(a.estimatedNetAfterPaperExecutionTotal, 6)}`,
  );
  lines.push(
    `  positive-only paper-net total: ${fmtNum(a.estimatedNetAfterPaperExecutionPositiveTotal, 6)}`,
  );
  lines.push(
    `  paper-net median / best / worst: ${fmtNum(a.estimatedNetAfterPaperExecutionMedian, 6)} / ${fmtNum(a.estimatedNetAfterPaperExecutionBest, 6)} / ${fmtNum(a.estimatedNetAfterPaperExecutionWorst, 6)}`,
  );
  lines.push(
    `  positiveAssessmentRate / blockedByGateRate: ${fmtPct(a.positiveAssessmentRate)} / ${fmtPct(a.blockedByGateRate)}`,
  );
  lines.push(`  assessment evidence: ${e.paperAssessmentEvidence.label}`);
  lines.push("");
  lines.push(`  opened paper cycles: ${e.openedPaperCycles}`);
  lines.push(`  closed paper cycles (markout-backed): ${e.closedPaperCyclesViaMarkout}`);
  lines.push(`  insufficient evidence cycles: ${e.insufficientEvidencePaperCycles}`);
  lines.push(`  markout-backed cycles: ${e.markoutBackedCycleCount}`);
  lines.push(
    `  simulated closed-cycle net total: ${fmtNum(e.simulatedClosedCycleNetTotal, 6)} (positives ${e.simulatedClosedCyclePositiveCount}, negatives ${e.simulatedClosedCycleNegativeCount})`,
  );
  lines.push("");
  lines.push(`  closed-trade positives / negatives: ${e.positiveCycles} / ${e.negativeCycles}`);
  lines.push(`  positive cycle rate (closed): ${fmtPct(e.positiveCycleRate)}`);
  lines.push(`  net edge total (closed): ${fmtNum(e.netEdgeTotal)}`);
  lines.push(`  fees / slippage (sim, closed): ${fmtNum(e.estimatedFeesTotal)} / ${fmtNum(e.estimatedSlippageTotal)}`);
  lines.push(`  net after fees+slippage (closed): ${fmtNum(e.netAfterFeesAndSlippage)}`);
  lines.push(`  median / worst / best cycle PnL (closed): ${fmtNum(e.medianNetEdge)} / ${fmtNum(e.worstCycleNetEdge)} / ${fmtNum(e.bestCycleNetEdge)}`);
  lines.push(`  max drawdown (paper): ${fmtNum(e.maxDrawdownPaper)}`);
  lines.push(`  profit factor (paper): ${fmtNum(e.profitFactorPaper)}`);
  if (e.closedPaperCycles === 0 && a.paperExecutionAssessmentCount > 0) {
    lines.push(`  confidence: ${e.confidenceLevel} because no closed paper cycles yet`);
  } else {
    lines.push(`  confidence: ${e.confidenceLevel}`);
  }
  lines.push("");

  lines.push("--- EXECUTION REALISM (simulated) ---");
  const x = d.executionRealism;
  lines.push(`  avg / p95 signal→enqueue: ${fmtMs(x.averageSignalToEnqueueLatencyMs)} / ${fmtMs(x.p95SignalToEnqueueLatencyMs)}`);
  lines.push(`  avg / p95 dispatch: ${fmtMs(x.averageDispatchLatencyMs)} / ${fmtMs(x.p95DispatchLatencyMs)}`);
  lines.push(`  stale / against / favor: ${x.staleSignalCount} / ${x.priceMovedAgainstSignalCount} / ${x.priceMovedInFavorCount}`);
  lines.push(`  fill quality (sim): ${x.simulatedFillQuality}`);
  lines.push(`  slippage bps (sim): ${fmtNum(x.simulatedSlippageBps, 1)}`);
  lines.push(`  markout 5s / 30s / 60s: ${x.markout5s ?? "n/a"} / ${x.markout30s ?? "n/a"} / ${x.markout60s ?? "n/a"}`);
  if (x.markoutFollowupStats) {
    const s = x.markoutFollowupStats;
    lines.push(
      `  followups: ok=${s.okFollowups}, price_unavailable=${s.priceUnavailableFollowups}, sampler_error=${s.samplerErrorFollowups}, unavailableRate=${fmtPct(s.priceUnavailableRate)}`,
    );
    const srcs = Object.entries(s.priceSourcesUsed);
    if (srcs.length > 0) {
      lines.push(
        `  priceSourcesUsed: ${srcs.map(([k, v]) => `${k}=${v}`).join(", ")}`,
      );
    }
  }
  lines.push("");

  lines.push("--- FAILURE-MODE PROBES ---");
  const r = d.reliabilityReadiness;
  const p = r.probes;
  lines.push(`  restartRecovery: ${p?.restartRecovery ?? r.restartRecoveryPassed}`);
  lines.push(`  idempotency: ${p?.idempotency ?? r.idempotencyPassed}`);
  lines.push(`  duplicatePrevention: ${p?.duplicatePrevention ?? r.duplicatePreventionPassed}`);
  lines.push(`  leaseRecovery: ${p?.leaseRecovery ?? r.leaseRecoveryPassed}`);
  lines.push(`  retryFinite: ${p?.retryFinite ?? r.retryFinitePassed}`);
  lines.push(`  permanentErrorsBecomeDead: ${p?.permanentErrorsBecomeDead ?? r.permanentErrorsBecomeDeadPassed}`);
  lines.push(`  circuitBreaker: ${p?.circuitBreaker ?? r.circuitBreakerPassed}`);
  lines.push(`  workerSurvivesDispatcherFailure: ${p?.workerSurvivesDispatcherFailure ?? r.workerSurvivesDispatcherFailurePassed}`);
  lines.push("");

  lines.push("--- NEXT TECHNICAL STEPS ---");
  if (d.nextActions.length === 0) lines.push("  (none)");
  else d.nextActions.forEach(a => lines.push(`  - ${a}`));
  lines.push("");

  lines.push("Reminder: this gate is paper/shadow only. It does not authorize real");
  lines.push("execution, capital activation, signing, wallet usage, or any external submit.");
  lines.push("=".repeat(70));

  return lines.join("\n");
}
