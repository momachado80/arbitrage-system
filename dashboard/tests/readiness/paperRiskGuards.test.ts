/**
 * Tests — paper cooldown-after-loss guard.
 *
 * Paper-only. No execution, no submit, no wallet/signer/private key.
 * Policy: shadow_only_no_api_submit_v1.
 */

import {
  computePaperCooldownState,
  createPaperCooldownGuard,
  type PaperCycleOutcome,
} from "../../lib/readiness/paperRiskGuards";

import { assertEqual, assertThrows, assertTrue, describe, test } from "./_assert";

const T = 1_700_000_000_000;

function loss(id: string, pnl: number, t: number = T): PaperCycleOutcome {
  return { cycleId: id, pnl, timestamp: t };
}

describe("tests/readiness/paperRiskGuards.test.ts", () => {
  test("1. three consecutive losses activate cooldown", () => {
    const guard = createPaperCooldownGuard({
      maxConsecutiveLosses: 3,
      severeLossThresholdUsd: 10_000, // disable severe trigger
      cooldownDurationMs: 60_000,
    });
    guard.recordCycle(loss("a", -1, T + 0));
    guard.recordCycle(loss("b", -1, T + 1));
    guard.recordCycle(loss("c", -1, T + 2));
    const state = guard.getState(T + 5);
    assertEqual(state.cooldownActive, true, "cooldown active after 3rd loss");
    assertEqual(state.consecutiveLosses, 3, "3 consecutive losses");
    assertEqual(state.lastLossReason, "consecutive_losses", "reason set");
    assertEqual(state.shouldSkipPaperDispatch, true, "should skip dispatch");
  });

  test("2. two losses do NOT activate cooldown", () => {
    const guard = createPaperCooldownGuard({
      maxConsecutiveLosses: 3,
      severeLossThresholdUsd: 10_000,
      cooldownDurationMs: 60_000,
    });
    guard.recordCycle(loss("a", -1, T + 0));
    guard.recordCycle(loss("b", -1, T + 1));
    const state = guard.getState(T + 5);
    assertEqual(state.cooldownActive, false, "still cool after 2 losses");
    assertEqual(state.consecutiveLosses, 2, "2 trailing losses");
    assertEqual(state.shouldSkipPaperDispatch, false, "no skip");
  });

  test("3. one severe loss activates cooldown", () => {
    const guard = createPaperCooldownGuard({
      maxConsecutiveLosses: 100,
      severeLossThresholdUsd: 200,
      cooldownDurationMs: 60_000,
    });
    guard.recordCycle(loss("a", -250, T + 0));
    const state = guard.getState(T + 5);
    assertEqual(state.cooldownActive, true, "severe loss activates cooldown");
    assertEqual(state.lastLossReason, "severe_single_loss", "severe reason");
  });

  test("4. profit after losses resets consecutiveLosses", () => {
    const guard = createPaperCooldownGuard({
      maxConsecutiveLosses: 3,
      severeLossThresholdUsd: 10_000,
      cooldownDurationMs: 60_000,
    });
    guard.recordCycle(loss("a", -1, T + 0));
    guard.recordCycle(loss("b", -1, T + 1));
    guard.recordCycle(loss("c", +5, T + 2)); // win resets
    const state = guard.getState(T + 5);
    assertEqual(state.consecutiveLosses, 0, "win resets streak");
    assertEqual(state.cooldownActive, false, "no cooldown");
  });

  test("5. cooldown expires after configured duration", () => {
    const guard = createPaperCooldownGuard({
      maxConsecutiveLosses: 1,
      severeLossThresholdUsd: 10_000,
      cooldownDurationMs: 60_000,
    });
    guard.recordCycle(loss("a", -1, T + 0));
    assertEqual(guard.getState(T + 1_000).cooldownActive, true, "active in window");
    assertEqual(
      guard.getState(T + 60_001).cooldownActive,
      false,
      "expired after 60s",
    );
  });

  test("6. shouldSkipPaperDispatch is true during cooldown, false otherwise", () => {
    const guard = createPaperCooldownGuard({
      maxConsecutiveLosses: 1,
      severeLossThresholdUsd: 10_000,
      cooldownDurationMs: 60_000,
    });
    guard.recordCycle(loss("a", -1, T + 0));
    assertEqual(
      guard.getState(T + 5).shouldSkipPaperDispatch,
      true,
      "skip during cooldown",
    );
    assertEqual(
      guard.getState(T + 60_001).shouldSkipPaperDispatch,
      false,
      "no skip after expiry",
    );
  });

  test("7. no real path is invoked — only pure local memory", () => {
    // Inspect surface: the guard exposes only pure functions.
    const guard = createPaperCooldownGuard();
    const keys = Object.keys(guard).sort();
    assertEqual(keys, ["getState", "recordCycle", "snapshot"], "surface is pure");
  });

  test("rejects invalid cycle inputs", () => {
    const guard = createPaperCooldownGuard();
    assertThrows(
      () => guard.recordCycle({ cycleId: "", pnl: -1, timestamp: T } as PaperCycleOutcome),
      "empty cycleId rejected",
    );
    assertThrows(
      () =>
        guard.recordCycle({
          cycleId: "x",
          pnl: Number.NaN,
          timestamp: T,
        } as PaperCycleOutcome),
      "NaN pnl rejected",
    );
  });

  test("computePaperCooldownState convenience helper composes correctly", () => {
    const cycles: PaperCycleOutcome[] = [
      loss("a", -1, T + 0),
      loss("b", -1, T + 1),
      loss("c", -1, T + 2),
    ];
    const state = computePaperCooldownState({
      closedCycles: cycles,
      nowMs: T + 5,
      options: {
        maxConsecutiveLosses: 3,
        severeLossThresholdUsd: 10_000,
        cooldownDurationMs: 60_000,
      },
    });
    assertEqual(state.cooldownActive, true, "active");
    assertEqual(state.cooldownAfterLossSimulated, true, "mechanism present");
    assertTrue(
      typeof state.cooldownActiveUntilMs === "number",
      "cooldownActiveUntilMs is a number when active",
    );
  });
});
