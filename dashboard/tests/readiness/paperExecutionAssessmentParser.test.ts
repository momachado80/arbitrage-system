/**
 * Tests — paper execution assessment parser.
 *
 * Paper/shadow only. NEVER converts an assessment into a closed trade or into
 * realized PnL. NEVER touches submit/wallet/signer/private key.
 * Policy: shadow_only_no_api_submit_v1.
 */

import fs from "fs";
import os from "os";
import path from "path";

import {
  parsePaperExecutionAssessmentLine,
  readPaperExecutionAssessmentsFromJsonl,
  summarizePaperExecutionAssessments,
} from "../../lib/readiness/paperExecutionAssessmentParser";

import {
  analyzeMicrocapitalReadiness,
  type AnalyzerInput,
} from "../../lib/readiness/analyzeMicrocapitalReadiness";

import { assertEqual, assertTrue, describe, test } from "./_assert";

const POSITIVE_LINE = JSON.stringify({
  isoTimestamp: "2026-04-29T12:00:00.000Z",
  marketId: "1823789",
  marketTitle: "Cross-venue anchor 1823789",
  paperExecutionVerdict: "paper_cycle_positive",
  executionGateVerdict: "passed",
  narrowValidationVerdict: "passed",
  estimatedNetBeforeExecution: 0.02,
  estimatedNetAfterPaperExecution: 0.015164,
  estimatedEntrySlippage: 0.0015,
  estimatedExitSlippage: 0.001,
  estimatedHedgeCost: 0.0005,
  estimatedEntryCost: 0.001,
  estimatedExitCost: 0.001,
  estimatedLegRiskCost: 0.0005,
  microstructureObservationState: "clob_book_observed",
  microstructureDetail: "depth_within_window",
  clobBookStructure: "two_sided",
  clobBestBid: 0.49,
  clobBestAsk: 0.51,
  clobMid: 0.5,
  clobSpreadUsed: 0.02,
  clobBidLevels: 5,
  clobAskLevels: 5,
  entryDepthAvailable: true,
  exitDepthAvailable: true,
  supportingNote: "all gates green",
});

const BLOCKED_BY_GATE_LINE = JSON.stringify({
  isoTimestamp: "2026-04-29T12:05:00.000Z",
  marketId: "1823789",
  paperExecutionVerdict: "blocked_by_gate",
  executionGateVerdict: "blocked_by_execution",
  microstructureObservationState: "clob_book_unavailable",
  estimatedNetAfterPaperExecution: -0.001,
});

const NOT_VIABLE_LINE = JSON.stringify({
  isoTimestamp: "2026-04-29T12:10:00.000Z",
  marketId: "1823789",
  paperExecutionVerdict: "not_viable_under_stress",
  estimatedNetAfterPaperExecution: -0.005,
});

describe("tests/readiness/paperExecutionAssessmentParser.test.ts", () => {
  test("parses a paper_cycle_positive line with estimatedNetAfterPaperExecution", () => {
    const r = parsePaperExecutionAssessmentLine(POSITIVE_LINE);
    assertTrue(r !== null, "row parsed");
    assertEqual(r!.paperExecutionVerdict, "paper_cycle_positive", "verdict");
    assertEqual(r!.estimatedNetAfterPaperExecution, 0.015164, "net captured");
    assertEqual(r!.marketId, "1823789", "marketId");
    assertEqual(r!.entryDepthAvailable, true, "entryDepthAvailable=true");
  });

  test("parses a blocked_by_gate line including microstructure state", () => {
    const r = parsePaperExecutionAssessmentLine(BLOCKED_BY_GATE_LINE);
    assertTrue(r !== null, "parsed");
    assertEqual(r!.paperExecutionVerdict, "blocked_by_gate", "verdict");
    assertEqual(r!.microstructureObservationState, "clob_book_unavailable", "micro state");
  });

  test("counts blocked_by_execution via executionGateVerdict", () => {
    const summary = summarizePaperExecutionAssessments([
      parsePaperExecutionAssessmentLine(BLOCKED_BY_GATE_LINE)!,
    ]);
    assertEqual(summary.paperBlockedByExecutionCount, 1, "blocked_by_execution counted");
    assertEqual(summary.paperBlockedByGateCount, 1, "blocked_by_gate counted");
  });

  test("counts clob_book_unavailable via microstructureObservationState", () => {
    const summary = summarizePaperExecutionAssessments([
      parsePaperExecutionAssessmentLine(BLOCKED_BY_GATE_LINE)!,
    ]);
    assertEqual(summary.clobUnavailableCount, 1, "clob unavailable counted");
  });

  test("computes positiveAssessmentRate and blockedByGateRate", () => {
    const lines = [
      POSITIVE_LINE,
      BLOCKED_BY_GATE_LINE,
      BLOCKED_BY_GATE_LINE,
      BLOCKED_BY_GATE_LINE,
    ];
    const rows = lines
      .map(parsePaperExecutionAssessmentLine)
      .filter((r): r is NonNullable<typeof r> => r !== null);
    const summary = summarizePaperExecutionAssessments(rows);
    assertEqual(summary.paperExecutionAssessmentCount, 4, "4 rows");
    assertEqual(summary.paperCyclePositiveCount, 1, "1 positive");
    assertEqual(summary.paperBlockedByGateCount, 3, "3 blocked");
    assertTrue(
      Math.abs(summary.positiveAssessmentRate - 0.25) < 1e-9,
      "positive rate = 25%",
    );
    assertTrue(
      Math.abs(summary.blockedByGateRate - 0.75) < 1e-9,
      "blocked rate = 75%",
    );
    assertTrue(
      Math.abs(summary.estimatedNetAfterPaperExecutionPositiveTotal - 0.015164) < 1e-9,
      "positive-only total isolates winners",
    );
  });

  test("does NOT convert an assessment into a closed trade", () => {
    const baseInput: AnalyzerInput = {
      systemIdentity: {
        projectName: "test",
        generatedAt: "2026-04-30T00:00:00.000Z",
        gitCommit: null,
        paperStateDir: "/tmp/.paper",
        executionMode: "controlled_paper",
        effectiveMode: "shadow_only",
        workerName: "t",
      },
      paperAnalytics: null,
      closedTrades: [],
      activeTrades: [],
      realismSamples: [],
      prohibitedTermsScan: { scannedFileCount: 1, findings: [], passed: true },
      apiSubmitAllowedTrueAnywhere: false,
      dispatcherAuditOnly: true,
      payloadSafetyPresent: true,
      reliability: {
        restartRecoveryPassed: true,
        idempotencyPassed: true,
        duplicatePreventionPassed: true,
        leaseRecoveryPassed: true,
        retryFinitePassed: true,
        permanentErrorsBecomeDeadPassed: true,
        circuitBreakerPassed: true,
        workerSurvivesDispatcherFailurePassed: true,
      },
      observability: {
        hasJsonLogs: true,
        hasShadowAudit: true,
        hasPaperTrades: true,
        hasAnalyticsEndpoint: true,
        hasSystemEndpoint: true,
        hasTradesEndpoint: true,
        canReconcileCycleToPacket: true,
        canReconcilePacketToDispatch: true,
        canReconcileSignalToPaperTrade: true,
      },
      riskLimits: {
        killSwitchSimulated: true,
        cooldownAfterLossSimulated: true,
        exposureLimitSimulated: true,
        maxConsecutiveNegativeCycles: 0,
        maxPaperLossPerCycle: 0,
        maxPaperDailyLoss: 0,
      },
      paperExecutionAssessments: summarizePaperExecutionAssessments([
        parsePaperExecutionAssessmentLine(POSITIVE_LINE)!,
      ]),
    };
    const d = analyzeMicrocapitalReadiness(baseInput);
    assertEqual(d.economicReadiness.closedPaperCycles, 0, "closedPaperCycles stays 0");
    assertEqual(d.economicReadiness.totalCyclesObserved, 0, "totalCyclesObserved stays 0");
    assertEqual(
      d.economicReadiness.paperExecutionAssessments.paperExecutionAssessmentCount,
      1,
      "but assessment is reported",
    );
    assertEqual(
      d.economicReadiness.paperExecutionAssessments.paperCyclePositiveCount,
      1,
      "1 positive assessment reported",
    );
    assertEqual(
      d.economicReadiness.netEdgeTotal,
      0,
      "no realized PnL inferred from assessment",
    );
  });

  test("keeps confidence LOW when closedPaperCycles=0 even with positive assessments", () => {
    const baseInput: AnalyzerInput = {
      systemIdentity: {
        projectName: "test",
        generatedAt: "2026-04-30T00:00:00.000Z",
        gitCommit: null,
        paperStateDir: "/tmp/.paper",
        executionMode: "controlled_paper",
        effectiveMode: "shadow_only",
        workerName: "t",
      },
      paperAnalytics: null,
      closedTrades: [],
      activeTrades: [],
      realismSamples: [],
      prohibitedTermsScan: { scannedFileCount: 1, findings: [], passed: true },
      apiSubmitAllowedTrueAnywhere: false,
      dispatcherAuditOnly: true,
      payloadSafetyPresent: true,
      reliability: {
        restartRecoveryPassed: true,
        idempotencyPassed: true,
        duplicatePreventionPassed: true,
        leaseRecoveryPassed: true,
        retryFinitePassed: true,
        permanentErrorsBecomeDeadPassed: true,
        circuitBreakerPassed: true,
        workerSurvivesDispatcherFailurePassed: true,
      },
      observability: {
        hasJsonLogs: true,
        hasShadowAudit: true,
        hasPaperTrades: true,
        hasAnalyticsEndpoint: true,
        hasSystemEndpoint: true,
        hasTradesEndpoint: true,
        canReconcileCycleToPacket: true,
        canReconcilePacketToDispatch: true,
        canReconcileSignalToPaperTrade: true,
      },
      riskLimits: {
        killSwitchSimulated: true,
        cooldownAfterLossSimulated: true,
        exposureLimitSimulated: true,
        maxConsecutiveNegativeCycles: 0,
        maxPaperLossPerCycle: 0,
        maxPaperDailyLoss: 0,
      },
      paperExecutionAssessments: summarizePaperExecutionAssessments(
        [POSITIVE_LINE, POSITIVE_LINE, POSITIVE_LINE]
          .map(parsePaperExecutionAssessmentLine)
          .filter((r): r is NonNullable<typeof r> => r !== null),
      ),
    };
    const d = analyzeMicrocapitalReadiness(baseInput);
    assertEqual(d.economicReadiness.confidenceLevel, "LOW", "confidence stays LOW");
  });

  test("majority-blocked assessment evidence label is set when blockedByGateRate > 0.5", () => {
    const summary = summarizePaperExecutionAssessments(
      [
        POSITIVE_LINE,
        BLOCKED_BY_GATE_LINE,
        BLOCKED_BY_GATE_LINE,
        BLOCKED_BY_GATE_LINE,
      ]
        .map(parsePaperExecutionAssessmentLine)
        .filter((r): r is NonNullable<typeof r> => r !== null),
    );
    assertTrue(summary.blockedByGateRate > 0.5, "blocked majority confirmed");
    // Build a minimal AnalyzerInput to verify the evidence label.
    const d = analyzeMicrocapitalReadiness({
      systemIdentity: {
        projectName: "t",
        generatedAt: "2026-04-30T00:00:00.000Z",
        gitCommit: null,
        paperStateDir: "/tmp/.paper",
        executionMode: "controlled_paper",
        effectiveMode: "shadow_only",
        workerName: "t",
      },
      paperAnalytics: null,
      closedTrades: [],
      activeTrades: [],
      realismSamples: [],
      prohibitedTermsScan: { scannedFileCount: 1, findings: [], passed: true },
      apiSubmitAllowedTrueAnywhere: false,
      dispatcherAuditOnly: true,
      payloadSafetyPresent: true,
      reliability: {
        restartRecoveryPassed: true,
        idempotencyPassed: true,
        duplicatePreventionPassed: true,
        leaseRecoveryPassed: true,
        retryFinitePassed: true,
        permanentErrorsBecomeDeadPassed: true,
        circuitBreakerPassed: true,
        workerSurvivesDispatcherFailurePassed: true,
      },
      observability: {
        hasJsonLogs: true,
        hasShadowAudit: true,
        hasPaperTrades: true,
        hasAnalyticsEndpoint: true,
        hasSystemEndpoint: true,
        hasTradesEndpoint: true,
        canReconcileCycleToPacket: true,
        canReconcilePacketToDispatch: true,
        canReconcileSignalToPaperTrade: true,
      },
      riskLimits: {
        killSwitchSimulated: true,
        cooldownAfterLossSimulated: true,
        exposureLimitSimulated: true,
        maxConsecutiveNegativeCycles: 0,
        maxPaperLossPerCycle: 0,
        maxPaperDailyLoss: 0,
      },
      paperExecutionAssessments: summary,
    });
    assertEqual(
      d.economicReadiness.paperAssessmentEvidence.label,
      "positive_assessments_present_majority_blocked",
      "label classified correctly",
    );
  });

  test("readPaperExecutionAssessmentsFromJsonl reads + skips blank/invalid lines", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "paper-assessment-"));
    const file = path.join(dir, "history.jsonl");
    const content = [
      POSITIVE_LINE,
      "",
      "not json",
      BLOCKED_BY_GATE_LINE,
      NOT_VIABLE_LINE,
      "",
    ].join("\n");
    fs.writeFileSync(file, content, "utf8");
    const rows = readPaperExecutionAssessmentsFromJsonl(file);
    assertEqual(rows.length, 3, "3 valid rows");
    const summary = summarizePaperExecutionAssessments(rows);
    assertEqual(summary.paperCyclePositiveCount, 1, "1 positive");
    assertEqual(summary.paperBlockedByGateCount, 1, "1 blocked");
    assertEqual(summary.paperNotViableUnderStressCount, 1, "1 not_viable");
  });

  test("returns empty list when file is missing", () => {
    const rows = readPaperExecutionAssessmentsFromJsonl(
      "/nonexistent/path/should-not-exist.jsonl",
    );
    assertEqual(rows.length, 0, "missing file → empty list");
  });

  test("parser surface contains no submit/wallet/signer/privateKey method", () => {
    const surface = [
      "parsePaperExecutionAssessmentLine",
      "summarizePaperExecutionAssessments",
      "readPaperExecutionAssessmentsFromJsonl",
      "EMPTY_PAPER_EXECUTION_ASSESSMENT_SUMMARY",
    ];
    for (const sym of surface) {
      assertTrue(
        !sym.toLowerCase().includes("submit"),
        `${sym}: no submit verb`,
      );
      assertTrue(!sym.toLowerCase().includes("wallet"), `${sym}: no wallet`);
      assertTrue(!sym.toLowerCase().includes("signer"), `${sym}: no signer`);
      assertTrue(
        !sym.toLowerCase().includes("privatekey"),
        `${sym}: no privateKey`,
      );
    }
  });
});
