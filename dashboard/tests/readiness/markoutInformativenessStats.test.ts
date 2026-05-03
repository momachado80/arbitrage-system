/**
 * Tests — métricas de informatividade de markout (paper-only).
 */

import * as fs from "fs";
import * as os from "os";
import * as path from "path";

import { analyzeMicrocapitalReadiness } from "../../lib/readiness/analyzeMicrocapitalReadiness";
import { computeMarkoutInformativenessStats } from "../../lib/readiness/markoutInformativenessStats";
import type { PaperCycleLifecycleSummary } from "../../lib/readiness/paperCycleLifecycle";
import { renderMicrocapitalReadinessExecutiveSummary } from "../../lib/readiness/renderMicrocapitalReadinessExecutiveSummary";
import type { MarkoutHorizonMs } from "../../lib/readiness/markoutFollowupProducer";

import { assertIncludes, assertEqual, describe, test } from "./_assert";

function lifecycleClosed(...cycleIds: string[]): PaperCycleLifecycleSummary {
  const events = cycleIds.flatMap(cid => [
    {
      type: "paper_cycle_opened" as const,
      cycleId: cid,
      marketId: "1823789",
      openedAt: "2026-05-01T00:00:00.000Z",
      estimatedEntryNet: 0.01,
    },
    {
      type: "paper_cycle_closed" as const,
      cycleId: cid,
      marketId: "1823789",
      openedAt: "2026-05-01T00:00:00.000Z",
      closedAt: "2026-05-01T01:00:00.000Z",
      closeStatus: "closed" as const,
      closeReason: "markout_window_complete" as const,
      evidenceSource: "markout_followups" as const,
      estimatedEntryNet: 0.01,
      estimatedExitNet: 0,
      simulatedNetPnl: 0.01,
      averageMarkout: 0,
      markoutHorizonsObserved: [5_000, 30_000, 60_000] as MarkoutHorizonMs[],
    },
  ]);
  const closedCount = cycleIds.length;
  return {
    openedPaperCycles: cycleIds.length,
    closedPaperCycles: closedCount,
    insufficientEvidencePaperCycles: 0,
    simulatedClosedCycleNetTotal: 0,
    simulatedClosedCyclePositiveCount: closedCount,
    simulatedClosedCycleNegativeCount: 0,
    markoutBackedCycleCount: closedCount,
    events,
  };
}

function writeTempFollowups(rows: Record<string, unknown>[]): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "minf-tests-"));
  const p = path.join(dir, "followups.jsonl");
  fs.writeFileSync(p, `${rows.map(r => JSON.stringify(r)).join("\n")}\n`, "utf8");
  return p;
}

describe("tests/readiness/markoutInformativenessStats.test.ts", () => {
  const cidFlat = "pc::1823789::2026-05-01T02:22:47.082Z";

  test("flat: 3 horizons ok markout zero, best_ask_only, pinned low", () => {
    const rows = [5_000, 30_000, 60_000].map(h => ({
      type: "markout_followup",
      cycleId: cidFlat,
      marketId: "1823789",
      horizonMs: h,
      samplerStatus: "ok",
      priceSource: "best_ask_only",
      rawPriceFieldsSeen: ["bestAsk"],
      basePrice: 0.001,
      followupPrice: 0.001,
      markout: 0,
    }));
    const p = writeTempFollowups(rows);
    const lifecycle = lifecycleClosed(cidFlat);
    const s = computeMarkoutInformativenessStats(lifecycle, p)!;
    assertEqual(s.flatMarkoutCycleCount, 1, "flat");
    assertEqual(s.informativeMarkoutCycleCount, 0, "informative");
    assertEqual(s.allHorizonsFlatCycleCount, 1, "all horizons flat");
    assertEqual(s.lowPricePinnedCycleCount, 1, "pinned");
    assertEqual(s.bestAskOnlyCycleCount, 1, "best ask only cycles");
    assertEqual(s.nonZeroMarkoutFollowupCount, 0, "nonzero fu");
    assertEqual(s.zeroMarkoutFollowupCount, 3, "zero fu");
    assertEqual(s.markoutInformativenessVerdict, "WEAK_FLAT_MARKOUTS", "verdict");
  });

  test("informative cycle: non-zero markout", () => {
    const cid = "pc::1823789::2026-06-01T00:00:00.000Z";
    const rows = [
      {
        type: "markout_followup",
        cycleId: cid,
        marketId: "1823789",
        horizonMs: 5_000,
        samplerStatus: "ok",
        priceSource: "best_ask_only",
        basePrice: 0.4,
        followupPrice: 0.41,
        markout: 0.03,
      },
      {
        type: "markout_followup",
        cycleId: cid,
        marketId: "1823789",
        horizonMs: 30_000,
        samplerStatus: "ok",
        priceSource: "best_ask_only",
        basePrice: 0.4,
        followupPrice: 0.41,
        markout: 0.03,
      },
      {
        type: "markout_followup",
        cycleId: cid,
        marketId: "1823789",
        horizonMs: 60_000,
        samplerStatus: "ok",
        priceSource: "best_ask_only",
        basePrice: 0.4,
        followupPrice: 0.41,
        markout: 0.03,
      },
    ];
    const p = writeTempFollowups(rows);
    const s = computeMarkoutInformativenessStats(lifecycleClosed(cid), p)!;
    assertEqual(s.informativeMarkoutCycleCount, 1, "informative cycles");
    assertEqual(s.flatMarkoutCycleCount, 0, "flat");
    assertEqual(s.markoutInformativenessVerdict, "INFORMATIVE", "verdict");
    assertEqual(s.bestAskOnlyCycleCount, 1, "best ask counted");
    assertEqual(s.lowPricePinnedCycleCount, 0, "no pin");
  });

  test("informative cycle: flat markouts mas spread followupPrice across horizons", () => {
    const cid = "pc::1823789::2026-06-02T00:00:00.000Z";
    const rows = [
      {
        type: "markout_followup",
        cycleId: cid,
        marketId: "1823789",
        horizonMs: 5_000,
        samplerStatus: "ok",
        priceSource: "best_ask_only",
        basePrice: 0.5,
        followupPrice: 0.5,
        markout: 0,
      },
      {
        type: "markout_followup",
        cycleId: cid,
        marketId: "1823789",
        horizonMs: 30_000,
        samplerStatus: "ok",
        priceSource: "best_ask_only",
        basePrice: 0.5,
        followupPrice: 0.52,
        markout: 0,
      },
      {
        type: "markout_followup",
        cycleId: cid,
        marketId: "1823789",
        horizonMs: 60_000,
        samplerStatus: "ok",
        priceSource: "best_ask_only",
        basePrice: 0.5,
        followupPrice: 0.51,
        markout: 0,
      },
    ];
    const p = writeTempFollowups(rows);
    const s = computeMarkoutInformativenessStats(lifecycleClosed(cid), p)!;
    assertEqual(s.informativeMarkoutCycleCount, 1, "price spread qualifies");
    assertEqual(s.flatMarkoutCycleCount, 0, "flat");
  });

  test("legacy rows sem samplerStatus não quebram (tratadas como ok com números)", () => {
    const cid = "pc::1823789::2026-07-01T00:00:00.000Z";
    const rows = [5_000, 30_000, 60_000].map(h => ({
      type: "markout_followup",
      cycleId: cid,
      marketId: "1823789",
      horizonMs: h,
      basePrice: 0.001,
      followupPrice: 0.001,
      markout: 0,
      source: "paper_shadow_readonly_live_sampler",
    }));
    const p = writeTempFollowups(rows);
    const s = computeMarkoutInformativenessStats(lifecycleClosed(cid), p)!;
    assertEqual(s.flatMarkoutCycleCount, 1, "classified flat");
    assertEqual(s.markoutInformativenessVerdict, "WEAK_FLAT_MARKOUTS", "verdict");
  });

  test("executive summary menciona markout informativeness", () => {
    const d = analyzeMicrocapitalReadiness({
      systemIdentity: {
        projectName: "test",
        generatedAt: "2026-05-03T00:00:00.000Z",
        gitCommit: null,
        paperStateDir: "/tmp",
        executionMode: "controlled_paper",
        effectiveMode: "shadow_only",
        workerName: "w",
      },
      paperAnalytics: null,
      closedTrades: [],
      activeTrades: [],
      realismSamples: [],
      prohibitedTermsScan: { scannedFileCount: 0, findings: [], passed: true },
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
      markoutFollowupStats: {
        okFollowups: 3,
        priceUnavailableFollowups: 0,
        samplerErrorFollowups: 0,
        priceUnavailableRate: 0,
        priceSourcesUsed: { best_ask_only: 3 },
      },
      markoutInformativenessStats: {
        markoutBackedCycleCount: 8,
        informativeMarkoutCycleCount: 0,
        flatMarkoutCycleCount: 8,
        nonZeroMarkoutFollowupCount: 0,
        zeroMarkoutFollowupCount: 24,
        allHorizonsFlatCycleCount: 8,
        bestAskOnlyCycleCount: 8,
        lowPricePinnedCycleCount: 8,
        markoutInformativenessVerdict: "WEAK_FLAT_MARKOUTS",
      },
    });
    const text = renderMicrocapitalReadinessExecutiveSummary(d);
    assertIncludes(text, "markout informativeness:", "heading");
    assertIncludes(text, "verdict=WEAK_FLAT_MARKOUTS", "verdict text");
    assertIncludes(text, "informative=0", "inform count");
    assertIncludes(text, "flat=8", "flat count");
  });
});
