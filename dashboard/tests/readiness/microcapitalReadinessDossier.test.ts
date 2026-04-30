/**
 * Tests — microcapital readiness dossier shape + analyzer verdicts.
 *
 * Paper/shadow only. No execution, no submit.
 */

import {
  analyzeMicrocapitalReadiness,
  type AnalyzerInput,
} from "../../lib/readiness/analyzeMicrocapitalReadiness";
import {
  isBlockedVerdict,
  pickWorstVerdict,
} from "../../lib/readiness/microcapitalReadinessDossier";
import { SHADOW_ONLY_POLICY_TAG } from "../../lib/readiness/microcapitalReadinessThresholds";
import type { PaperTrade } from "../../lib/paperTypes";

import { assertEqual, assertTrue, describe, test } from "./_assert";

function baseInput(overrides: Partial<AnalyzerInput> = {}): AnalyzerInput {
  return {
    systemIdentity: {
      projectName: "test",
      generatedAt: "2026-04-30T00:00:00.000Z",
      gitCommit: null,
      paperStateDir: "/tmp/.paper",
      executionMode: "controlled_paper",
      effectiveMode: "shadow_only",
      workerName: "test-worker",
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
    ...overrides,
  };
}

function fakeTrade(pnl: number, filled = 100): PaperTrade {
  return {
    tradeId: `pt-${Math.random()}`,
    opportunityId: "op",
    sourceType: "standard",
    opportunityType: "overround",
    marketsInvolved: [],
    openedAt: new Date(0).toISOString(),
    closedAt: new Date(60_000).toISOString(),
    status: "closed",
    grossEdgeAtEntry: 0.01,
    netEdgeAtEntry: 0.008,
    recommendedCapital: filled,
    requestedCapital: filled,
    filledCapital: filled,
    entryPriceEstimate: 0.5,
    entryConfidence: 0.5,
    realizedPnL: pnl,
    realizedReturn: pnl / filled,
    holdingTimeMs: 60_000,
    maxAdverseExcursion: 0,
    maxFavorableExcursion: 0,
  };
}

describe("tests/readiness/microcapitalReadinessDossier.test.ts", () => {
  test("approves when everything is clean and there are no cycles", () => {
    const d = analyzeMicrocapitalReadiness(baseInput());
    assertEqual(d.policyTag, SHADOW_ONLY_POLICY_TAG, "policyTag is the shadow tag");
    assertTrue(
      d.finalVerdict === "APPROVED_FOR_SHADOW_READINESS" ||
        d.finalVerdict === "APPROVED_WITH_REMARKS",
      "no-cycles state should not be a hard BLOCK",
    );
    assertEqual(d.economicReadiness.totalCyclesObserved, 0, "no cycles");
    assertEqual(d.economicReadiness.confidenceLevel, "LOW", "confidence LOW with 0 cycles");
  });

  test("BLOCKS on safety: prohibited term findings", () => {
    const d = analyzeMicrocapitalReadiness(
      baseInput({
        prohibitedTermsScan: {
          scannedFileCount: 5,
          findings: [
            { file: "/proj/lib/x.ts", line: 1, text: "const wallet = 1;", matchedTerm: "wallet" },
          ],
          passed: false,
        },
      }),
    );
    assertEqual(d.finalVerdict, "BLOCKED_SAFETY_RISK", "prohibited term blocks");
    assertTrue(d.blockers.length > 0, "blockers populated");
  });

  test("BLOCKS on safety: apiSubmitAllowed:true detected", () => {
    const d = analyzeMicrocapitalReadiness(
      baseInput({
        apiSubmitAllowedTrueAnywhere: true,
        prohibitedTermsScan: {
          scannedFileCount: 5,
          findings: [
            {
              file: "/proj/lib/x.ts",
              line: 1,
              text: "apiSubmitAllowed: true",
              matchedTerm: "apiSubmitAllowed: true",
            },
          ],
          passed: false,
        },
      }),
    );
    assertEqual(d.finalVerdict, "BLOCKED_SAFETY_RISK", "apiSubmitAllowed:true blocks");
  });

  test("BLOCKS economic when ≥100 cycles and net negative after fees+slippage", () => {
    const trades: PaperTrade[] = Array.from({ length: 100 }, () => fakeTrade(-1));
    const d = analyzeMicrocapitalReadiness(
      baseInput({
        closedTrades: trades,
        paperAnalytics: {
          totalTrades: 100,
          closedTrades: 100,
          winRate: 0,
          avgReturn: -0.01,
          avgPnL: -1,
          totalPnL: -100,
          profitFactor: 0,
          maxDrawdown: 100,
          avgHoldingTimeMs: 1,
          avgNetEdgeAtEntry: 0,
          avgFilledCapital: 100,
          utilizationRate: 0,
          pnlByOpportunityType: {},
          pnlBySourceType: {},
          averageCapacityConfidence: 0,
        },
      }),
    );
    assertTrue(isBlockedVerdict(d.finalVerdict), "should block on economic risk");
  });

  test("APPROVED_WITH_REMARKS for low cycle count", () => {
    const trades: PaperTrade[] = Array.from({ length: 5 }, () => fakeTrade(1));
    const d = analyzeMicrocapitalReadiness(baseInput({ closedTrades: trades }));
    assertEqual(d.finalVerdict, "APPROVED_WITH_REMARKS", "few cycles → remarks");
  });

  test("pickWorstVerdict prefers BLOCKED over APPROVED", () => {
    assertEqual(
      pickWorstVerdict(["APPROVED_FOR_SHADOW_READINESS", "BLOCKED_TECHNICAL_RISK"]),
      "BLOCKED_TECHNICAL_RISK",
      "blocked wins",
    );
    assertEqual(
      pickWorstVerdict(["APPROVED_FOR_SHADOW_READINESS", "APPROVED_WITH_REMARKS"]),
      "APPROVED_WITH_REMARKS",
      "remarks > approved",
    );
    assertEqual(
      pickWorstVerdict(["BLOCKED_TECHNICAL_RISK", "BLOCKED_SAFETY_RISK"]),
      "BLOCKED_SAFETY_RISK",
      "safety > technical",
    );
  });

  test("dossier final verdict surfaces blockers/warnings/nextActions", () => {
    const d = analyzeMicrocapitalReadiness(
      baseInput({
        observability: {
          hasJsonLogs: true,
          hasShadowAudit: false,
          hasPaperTrades: false,
          hasAnalyticsEndpoint: true,
          hasSystemEndpoint: true,
          hasTradesEndpoint: true,
          canReconcileCycleToPacket: false,
          canReconcilePacketToDispatch: false,
          canReconcileSignalToPaperTrade: false,
        },
      }),
    );
    assertTrue(
      isBlockedVerdict(d.finalVerdict),
      "missing shadow audit + paper trades is a technical block",
    );
    assertTrue(d.nextActions.length > 0, "nextActions populated");
  });
});
