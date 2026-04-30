/**
 * Tests — markout follow-up producer (paper-only).
 *
 * Paper/shadow only. No execution, no submit. No wallet. No signer. No private
 * key. Policy: shadow_only_no_api_submit_v1.
 */

import {
  buildMarkoutFollowupRecord,
  createMarkoutFollowupProducer,
  MARKOUT_HORIZONS_MS,
} from "../../lib/readiness/markoutFollowupProducer";
import {
  applyMarkoutPriceSet,
  buildExecutionRealismSample,
} from "../../lib/readiness/executionRealismHarness";

import { assertEqual, assertThrows, assertTrue, describe, test } from "./_assert";

const T = 1_700_000_000_000;

const BASE = {
  cycleId: "c1",
  marketId: "m1",
  signalTimestamp: T,
  basePrice: 0.5,
};

describe("tests/readiness/markoutFollowupProducer.test.ts", () => {
  test("1. builds 5s follow-up record correctly", () => {
    const r = buildMarkoutFollowupRecord(BASE, {
      cycleId: "c1",
      horizonMs: 5_000,
      followupTimestamp: T + 5_000,
      followupPrice: 0.51,
    });
    assertEqual(r.type, "markout_followup", "type");
    assertEqual(r.horizonMs, 5_000, "horizon=5s");
    assertTrue(Math.abs(r.markout - 0.01) < 1e-9, "markout = followup - base");
    assertEqual(r.source, "paper_shadow_readonly", "source tag");
  });

  test("2. builds 30s follow-up record correctly", () => {
    const r = buildMarkoutFollowupRecord(BASE, {
      cycleId: "c1",
      horizonMs: 30_000,
      followupTimestamp: T + 30_000,
      followupPrice: 0.495,
    });
    assertEqual(r.horizonMs, 30_000, "horizon=30s");
    assertTrue(Math.abs(r.markout - -0.005) < 1e-9, "markout = -0.005");
  });

  test("3. builds 60s follow-up record correctly", () => {
    const r = buildMarkoutFollowupRecord(BASE, {
      cycleId: "c1",
      horizonMs: 60_000,
      followupTimestamp: T + 60_000,
      followupPrice: 0.502,
    });
    assertEqual(r.horizonMs, 60_000, "horizon=60s");
    assertTrue(Math.abs(r.markout - 0.002) < 1e-9, "markout = 0.002");
  });

  test("4. computes markout correctly when integrated into realism harness", () => {
    const producer = createMarkoutFollowupProducer();
    producer.registerBaseSample(BASE);
    producer.recordFollowup({
      cycleId: "c1",
      horizonMs: 5_000,
      followupTimestamp: T + 5_000,
      followupPrice: 0.51,
    });
    producer.recordFollowup({
      cycleId: "c1",
      horizonMs: 30_000,
      followupTimestamp: T + 30_000,
      followupPrice: 0.495,
    });
    producer.recordFollowup({
      cycleId: "c1",
      horizonMs: 60_000,
      followupTimestamp: T + 60_000,
      followupPrice: 0.502,
    });

    const baseInput = {
      cycleId: "c1",
      signalTimestamp: T,
      enqueueTimestamp: T + 10,
      observedPriceOrProbabilitySnapshot: 0.5,
      notionalPaperUsd: 100,
    };
    const merged = applyMarkoutPriceSet(baseInput, producer.toMarkoutPriceSet("c1"));
    const sample = buildExecutionRealismSample(merged);
    assertEqual(sample.markoutStatus, "ok", "all three present → ok");
    assertEqual(Math.round((sample.markout5s ?? 0) * 1000) / 1000, 0.01, "5s");
    assertEqual(Math.round((sample.markout30s ?? 0) * 1000) / 1000, -0.005, "30s");
    assertEqual(Math.round((sample.markout60s ?? 0) * 1000) / 1000, 0.002, "60s");
  });

  test("5. keeps insufficient_data when no followups present", () => {
    const producer = createMarkoutFollowupProducer();
    producer.registerBaseSample(BASE);
    const baseInput = {
      cycleId: "c1",
      signalTimestamp: T,
      enqueueTimestamp: T + 10,
      observedPriceOrProbabilitySnapshot: 0.5,
      notionalPaperUsd: 100,
    };
    const merged = applyMarkoutPriceSet(baseInput, producer.toMarkoutPriceSet("c1"));
    const sample = buildExecutionRealismSample(merged);
    assertEqual(
      sample.markoutStatus,
      "insufficient_data",
      "no followups → insufficient_data",
    );
  });

  test("6. rejects payload without cycleId", () => {
    assertThrows(
      () =>
        buildMarkoutFollowupRecord(BASE, {
          cycleId: "",
          horizonMs: 5_000,
          followupTimestamp: T + 5_000,
          followupPrice: 0.51,
        }),
      "empty cycleId rejected",
    );
    assertThrows(
      () =>
        buildMarkoutFollowupRecord(BASE, {
          horizonMs: 5_000,
          followupTimestamp: T + 5_000,
          followupPrice: 0.51,
        } as unknown as Parameters<typeof buildMarkoutFollowupRecord>[1]),
      "missing cycleId rejected",
    );
    assertThrows(
      () =>
        buildMarkoutFollowupRecord(BASE, {
          cycleId: "DIFFERENT",
          horizonMs: 5_000,
          followupTimestamp: T + 5_000,
          followupPrice: 0.51,
        }),
      "cycleId mismatch rejected",
    );
    const producer = createMarkoutFollowupProducer();
    assertThrows(
      () =>
        producer.recordFollowup({
          cycleId: "unknown",
          horizonMs: 5_000,
          followupTimestamp: T + 5_000,
          followupPrice: 0.51,
        }),
      "unknown cycle id rejected",
    );
  });

  test("7. surface is paper-only (no submit/wallet/signer/privateKey method)", () => {
    const producer = createMarkoutFollowupProducer();
    const keys = Object.keys(producer).sort();
    assertEqual(
      keys,
      [
        "forget",
        "getFollowupsForCycle",
        "recordFollowup",
        "registerBaseSample",
        "toMarkoutPriceSet",
      ],
      "surface is exactly the paper-only API",
    );
    // Sanity: source tag is hard-coded to paper_shadow_readonly.
    const r = buildMarkoutFollowupRecord(BASE, {
      cycleId: "c1",
      horizonMs: 5_000,
      followupTimestamp: T + 5_000,
      followupPrice: 0.5,
    });
    assertEqual(r.source, "paper_shadow_readonly", "source pinned to paper");
  });

  test("MARKOUT_HORIZONS_MS = [5_000, 30_000, 60_000]", () => {
    assertEqual([...MARKOUT_HORIZONS_MS], [5_000, 30_000, 60_000], "horizons");
  });
});
