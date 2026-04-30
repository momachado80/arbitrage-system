/**
 * Microcapital Readiness Gate — paper cooldown-after-loss guard.
 *
 * Stateful, paper/shadow-only loss tracker. Records simulated cycle outcomes
 * (paper PnL only) into a small ring buffer and computes whether a simulated
 * cooldown is active.
 *
 * GUARDRAILS — this module:
 *   - does NOT submit any order
 *   - does NOT call any financial endpoint
 *   - does NOT sign any transaction
 *   - does NOT touch a wallet, signer, or private key
 *   - does NOT authorize real execution
 *   - is paper/shadow-only and read-only against external systems
 *
 * Policy: shadow_only_no_api_submit_v1.
 */

import { MICROCAPITAL_READINESS_THRESHOLDS } from "./microcapitalReadinessThresholds";

export interface PaperCycleOutcome {
  cycleId: string;
  /** Paper PnL in paper-USD. Negative = paper loss. */
  pnl: number;
  /** Epoch ms at which the cycle closed. */
  timestamp: number;
}

export type CooldownReason =
  | "consecutive_losses"
  | "severe_single_loss"
  | "none";

export interface PaperCooldownState {
  cooldownAfterLossSimulated: true;
  cooldownActive: boolean;
  cooldownActiveUntilMs: number | null;
  consecutiveLosses: number;
  lastLossReason: CooldownReason;
  shouldSkipPaperDispatch: boolean;
}

export interface PaperCooldownGuardOptions {
  maxConsecutiveLosses?: number;
  severeLossThresholdUsd?: number;
  cooldownDurationMs?: number;
  ringSize?: number;
}

export interface PaperCooldownGuard {
  /** Append a paper cycle outcome to the ring buffer. */
  recordCycle(outcome: PaperCycleOutcome): void;
  /** Compute the current state at `nowMs`. */
  getState(nowMs: number): PaperCooldownState;
  /** Inspect the ring buffer (newest last). */
  snapshot(): ReadonlyArray<PaperCycleOutcome>;
}

/**
 * Build a stateful paper cooldown guard. Pure local memory; no I/O. Optional
 * overrides come from caller-provided test inputs only — production callers
 * should rely on {@link MICROCAPITAL_READINESS_THRESHOLDS} defaults.
 */
export function createPaperCooldownGuard(
  opts: PaperCooldownGuardOptions = {},
): PaperCooldownGuard {
  const T = MICROCAPITAL_READINESS_THRESHOLDS;
  const maxConsecutive = Math.max(1, opts.maxConsecutiveLosses ?? T.MAX_CONSECUTIVE_LOSSES_PAPER);
  const severeLoss = Math.max(0, opts.severeLossThresholdUsd ?? T.SEVERE_PAPER_LOSS_USD);
  const cooldownMs = Math.max(0, opts.cooldownDurationMs ?? T.PAPER_COOLDOWN_DURATION_MS);
  const ringSize = Math.max(1, opts.ringSize ?? T.PAPER_COOLDOWN_RING_SIZE);

  const ring: PaperCycleOutcome[] = [];
  let cooldownActiveUntilMs: number | null = null;
  let lastLossReason: CooldownReason = "none";

  function pushRing(outcome: PaperCycleOutcome): void {
    ring.push(outcome);
    if (ring.length > ringSize) ring.shift();
  }

  function consecutiveTrailingLosses(): number {
    let count = 0;
    for (let i = ring.length - 1; i >= 0; i--) {
      if (ring[i].pnl < 0) count++;
      else break;
    }
    return count;
  }

  function recordCycle(outcome: PaperCycleOutcome): void {
    if (!outcome || typeof outcome.cycleId !== "string" || outcome.cycleId.length === 0) {
      throw new Error("paperCooldownGuard.recordCycle_invalid_cycleId");
    }
    if (typeof outcome.pnl !== "number" || !Number.isFinite(outcome.pnl)) {
      throw new Error("paperCooldownGuard.recordCycle_invalid_pnl");
    }
    if (typeof outcome.timestamp !== "number" || !Number.isFinite(outcome.timestamp)) {
      throw new Error("paperCooldownGuard.recordCycle_invalid_timestamp");
    }
    pushRing(outcome);

    // Trigger evaluation (only on a loss; profits do not extend cooldown).
    if (outcome.pnl < 0) {
      const consec = consecutiveTrailingLosses();
      if (-outcome.pnl >= severeLoss && severeLoss > 0) {
        cooldownActiveUntilMs = outcome.timestamp + cooldownMs;
        lastLossReason = "severe_single_loss";
        return;
      }
      if (consec >= maxConsecutive) {
        cooldownActiveUntilMs = outcome.timestamp + cooldownMs;
        lastLossReason = "consecutive_losses";
        return;
      }
    }
  }

  function getState(nowMs: number): PaperCooldownState {
    const active =
      cooldownActiveUntilMs !== null && nowMs < cooldownActiveUntilMs;
    if (!active && cooldownActiveUntilMs !== null && nowMs >= cooldownActiveUntilMs) {
      // Expired — clear the deadline (lastLossReason kept for inspection).
      cooldownActiveUntilMs = null;
    }
    return {
      cooldownAfterLossSimulated: true,
      cooldownActive: active,
      cooldownActiveUntilMs: active ? cooldownActiveUntilMs : null,
      consecutiveLosses: consecutiveTrailingLosses(),
      lastLossReason,
      shouldSkipPaperDispatch: active,
    };
  }

  function snapshot(): ReadonlyArray<PaperCycleOutcome> {
    return [...ring];
  }

  return { recordCycle, getState, snapshot };
}

export interface BuildCooldownStateInput {
  closedCycles: ReadonlyArray<PaperCycleOutcome>;
  nowMs: number;
  options?: PaperCooldownGuardOptions;
}

/**
 * Convenience helper: build a guard, replay cycles in order, return state.
 *
 * Pure with respect to caller — does not retain a guard reference.
 */
export function computePaperCooldownState(
  input: BuildCooldownStateInput,
): PaperCooldownState {
  const guard = createPaperCooldownGuard(input.options);
  for (const c of input.closedCycles) guard.recordCycle(c);
  return guard.getState(input.nowMs);
}
