/**
 * Tests — execution realism harness (paper-only).
 *
 * Verifies that {@link buildExecutionRealismSample} performs SIMULATED arithmetic
 * only and never reaches any external service.
 */

import {
  aggregateExecutionRealism,
  buildExecutionRealismSample,
} from "../../lib/readiness/executionRealismHarness";

import { assertEqual, assertTrue, describe, test } from "./_assert";

describe("tests/readiness/executionRealismHarness.test.ts", () => {
  test("simulated sample reports markoutStatus=insufficient_data when no future prices given", () => {
    const s = buildExecutionRealismSample({
      cycleId: "c1",
      signalTimestamp: 1_000,
      enqueueTimestamp: 1_050,
      observedPriceOrProbabilitySnapshot: 0.5,
      notionalPaperUsd: 100,
    });
    assertEqual(s.markoutStatus, "insufficient_data", "no future data → insufficient");
    assertEqual(s.markout5s, null, "5s null");
    assertEqual(s.markout30s, null, "30s null");
  });

  test("simulated sample computes net = gross - fee - slippage (paper-only)", () => {
    const s = buildExecutionRealismSample({
      cycleId: "c2",
      signalTimestamp: 0,
      enqueueTimestamp: 0,
      observedPriceOrProbabilitySnapshot: 0.5,
      simulatedExecutablePrice: 0.51,
      notionalPaperUsd: 1000,
      simulatedFeeOverride: 2,
      simulatedSlippageBpsOverride: 25,
    });
    // gross = (0.51-0.50)*1000 = 10
    // fee   = 2
    // slip  = 1000 * 25/10000 = 2.5
    // net   = 10 - 2 - 2.5 = 5.5
    assertEqual(Math.round(s.simulatedNet * 100) / 100, 5.5, "net arithmetic correct");
  });

  test("staleSignal flagged when signal→enqueue >= 2000ms", () => {
    const s = buildExecutionRealismSample({
      cycleId: "c3",
      signalTimestamp: 0,
      enqueueTimestamp: 3_000,
      observedPriceOrProbabilitySnapshot: 0.5,
      notionalPaperUsd: 100,
    });
    assertEqual(s.staleSignal, true, "stale signal flagged");
  });

  test("aggregate reports zero stale rate when none stale", () => {
    const samples = [
      buildExecutionRealismSample({
        cycleId: "c1",
        signalTimestamp: 0,
        enqueueTimestamp: 50,
        observedPriceOrProbabilitySnapshot: 0.5,
        notionalPaperUsd: 100,
      }),
      buildExecutionRealismSample({
        cycleId: "c2",
        signalTimestamp: 0,
        enqueueTimestamp: 100,
        observedPriceOrProbabilitySnapshot: 0.5,
        notionalPaperUsd: 100,
      }),
    ];
    const agg = aggregateExecutionRealism(samples);
    assertEqual(agg.sampleCount, 2, "two samples");
    assertEqual(agg.staleSignalCount, 0, "no stales");
    assertTrue(
      (agg.averageSignalToEnqueueLatencyMs ?? 999) < 200,
      "avg latency low",
    );
  });

  test("aggregate marks markoutCoverage as insufficient_data when no markouts", () => {
    const samples = [
      buildExecutionRealismSample({
        cycleId: "c1",
        signalTimestamp: 0,
        enqueueTimestamp: 50,
        observedPriceOrProbabilitySnapshot: 0.5,
        notionalPaperUsd: 100,
      }),
    ];
    const agg = aggregateExecutionRealism(samples);
    assertEqual(agg.markoutCoverage, "insufficient_data", "no markouts");
  });

  test("aggregate full coverage with all markouts present", () => {
    const samples = [
      buildExecutionRealismSample({
        cycleId: "c1",
        signalTimestamp: 0,
        enqueueTimestamp: 10,
        observedPriceOrProbabilitySnapshot: 0.5,
        notionalPaperUsd: 100,
        priceAfter5s: 0.501,
        priceAfter30s: 0.502,
        priceAfter60s: 0.499,
      }),
    ];
    const agg = aggregateExecutionRealism(samples);
    assertEqual(agg.markoutCoverage, "full", "full coverage");
  });
});
