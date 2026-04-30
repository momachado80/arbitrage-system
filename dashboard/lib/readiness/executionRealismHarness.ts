/**
 * Microcapital Readiness Gate — execution realism harness (paper / shadow only).
 *
 * Records, for each positive paper cycle, simulated execution metadata (latencies,
 * markouts, simulated fill quality) so the readiness analyzer can assess realism.
 *
 * GUARDRAILS — this module:
 *   - does NOT call any order endpoint
 *   - does NOT sign any transaction
 *   - does NOT touch a wallet, signer, or private key
 *   - does NOT submit anything real
 *   - is paper/shadow only
 *
 * Policy: shadow_only_no_api_submit_v1.
 */

import { MICROCAPITAL_READINESS_THRESHOLDS } from "./microcapitalReadinessThresholds";

export interface ExecutionRealismSampleInput {
  cycleId: string;
  signalTimestamp: number;
  enqueueTimestamp: number;
  /** Optional — present if dispatcher path is observed. */
  dispatchTimestamp?: number;
  /** Reference price/probability snapshot at the moment the signal fired. */
  observedPriceOrProbabilitySnapshot: number;
  /** Optional simulated executable price (paper-only). */
  simulatedExecutablePrice?: number;
  /** Reference notional (paper). */
  notionalPaperUsd: number;
  /** Optional fee override (paper-only). */
  simulatedFeeOverride?: number;
  /** Optional slippage bps override (paper-only). */
  simulatedSlippageBpsOverride?: number;
  /** Optional later-observed prices for markout windows (paper-only). */
  priceAfter5s?: number;
  priceAfter30s?: number;
  priceAfter60s?: number;
}

export type MarkoutStatus = "ok" | "insufficient_data";

export interface ExecutionRealismSample {
  cycleId: string;
  signalTimestamp: number;
  enqueueTimestamp: number;
  dispatchTimestamp: number | null;
  signalToEnqueueLatencyMs: number;
  enqueueToDispatchLatencyMs: number | null;
  observedPriceOrProbabilitySnapshot: number;
  simulatedExecutablePrice: number;
  simulatedFee: number;
  simulatedSlippage: number;
  simulatedSlippageBps: number;
  simulatedNet: number;
  markout5s: number | null;
  markout30s: number | null;
  markout60s: number | null;
  markoutStatus: MarkoutStatus;
  staleSignal: boolean;
  wouldHaveBeenGoodAfterDelay: boolean;
  wouldHaveBeenBadAfterDelay: boolean;
}

const STALE_SIGNAL_LATENCY_MS = 2_000;

function bpsOf(value: number, bps: number): number {
  return value * (bps / 10_000);
}

function sanitizeNumber(n: number | undefined, fallback: number): number {
  if (typeof n !== "number" || !Number.isFinite(n)) return fallback;
  return n;
}

/**
 * Build an execution realism sample. Pure function — does NOT submit, sign, or
 * call any external service. All "execution" is simulated arithmetic.
 */
export function buildExecutionRealismSample(
  input: ExecutionRealismSampleInput,
): ExecutionRealismSample {
  const signalToEnqueueLatencyMs = Math.max(
    0,
    input.enqueueTimestamp - input.signalTimestamp,
  );
  const enqueueToDispatchLatencyMs =
    typeof input.dispatchTimestamp === "number"
      ? Math.max(0, input.dispatchTimestamp - input.enqueueTimestamp)
      : null;

  const observed = sanitizeNumber(input.observedPriceOrProbabilitySnapshot, 0);
  const executable = sanitizeNumber(input.simulatedExecutablePrice, observed);

  const slippageBps = sanitizeNumber(
    input.simulatedSlippageBpsOverride,
    MICROCAPITAL_READINESS_THRESHOLDS.DEFAULT_PAPER_SLIPPAGE_BPS,
  );

  const notional = Math.max(0, sanitizeNumber(input.notionalPaperUsd, 0));
  const simulatedSlippage = bpsOf(notional, slippageBps);
  const simulatedFee = sanitizeNumber(
    input.simulatedFeeOverride,
    notional * MICROCAPITAL_READINESS_THRESHOLDS.DEFAULT_PAPER_FEE_RATE,
  );

  const grossEdgePerUnit = executable - observed;
  const grossEdgeNotional = grossEdgePerUnit * notional;
  const simulatedNet = grossEdgeNotional - simulatedFee - simulatedSlippage;

  const haveAny =
    typeof input.priceAfter5s === "number" ||
    typeof input.priceAfter30s === "number" ||
    typeof input.priceAfter60s === "number";

  const markout5s =
    typeof input.priceAfter5s === "number" ? input.priceAfter5s - observed : null;
  const markout30s =
    typeof input.priceAfter30s === "number" ? input.priceAfter30s - observed : null;
  const markout60s =
    typeof input.priceAfter60s === "number" ? input.priceAfter60s - observed : null;

  const markoutStatus: MarkoutStatus = haveAny ? "ok" : "insufficient_data";

  const staleSignal = signalToEnqueueLatencyMs >= STALE_SIGNAL_LATENCY_MS;

  const lateRef = markout30s ?? markout5s ?? markout60s ?? 0;
  const wouldHaveBeenGoodAfterDelay = lateRef > 0 && simulatedNet < 0;
  const wouldHaveBeenBadAfterDelay = lateRef < 0 && simulatedNet > 0;

  return {
    cycleId: input.cycleId,
    signalTimestamp: input.signalTimestamp,
    enqueueTimestamp: input.enqueueTimestamp,
    dispatchTimestamp: input.dispatchTimestamp ?? null,
    signalToEnqueueLatencyMs,
    enqueueToDispatchLatencyMs,
    observedPriceOrProbabilitySnapshot: observed,
    simulatedExecutablePrice: executable,
    simulatedFee,
    simulatedSlippage,
    simulatedSlippageBps: slippageBps,
    simulatedNet,
    markout5s,
    markout30s,
    markout60s,
    markoutStatus,
    staleSignal,
    wouldHaveBeenGoodAfterDelay,
    wouldHaveBeenBadAfterDelay,
  };
}

export interface ExecutionRealismAggregate {
  sampleCount: number;
  averageSignalToEnqueueLatencyMs: number | null;
  p95SignalToEnqueueLatencyMs: number | null;
  averageDispatchLatencyMs: number | null;
  p95DispatchLatencyMs: number | null;
  staleSignalCount: number;
  priceMovedAgainstSignalCount: number;
  priceMovedInFavorCount: number;
  averageSlippageBps: number;
  markout5s: number | null;
  markout30s: number | null;
  markout60s: number | null;
  markoutCoverage: "insufficient_data" | "partial" | "full";
}

function average(nums: number[]): number | null {
  if (nums.length === 0) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function percentile(nums: number[], p: number): number | null {
  if (nums.length === 0) return null;
  const sorted = [...nums].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * sorted.length)));
  return sorted[idx];
}

export function aggregateExecutionRealism(
  samples: ExecutionRealismSample[],
): ExecutionRealismAggregate {
  const enqueueLats = samples.map(s => s.signalToEnqueueLatencyMs);
  const dispatchLats = samples
    .map(s => s.enqueueToDispatchLatencyMs)
    .filter((n): n is number => typeof n === "number");

  const stale = samples.filter(s => s.staleSignal).length;
  const movedFavor = samples.filter(s => s.wouldHaveBeenGoodAfterDelay).length;
  const movedAgainst = samples.filter(s => s.wouldHaveBeenBadAfterDelay).length;

  const slippageBpsArr = samples.map(s => s.simulatedSlippageBps);
  const avgSlippage = slippageBpsArr.length > 0 ? average(slippageBpsArr) ?? 0 : 0;

  const m5 = samples.map(s => s.markout5s).filter((n): n is number => typeof n === "number");
  const m30 = samples.map(s => s.markout30s).filter((n): n is number => typeof n === "number");
  const m60 = samples.map(s => s.markout60s).filter((n): n is number => typeof n === "number");

  const totalMarkoutSlots = samples.length * 3;
  const filledMarkoutSlots = m5.length + m30.length + m60.length;
  let markoutCoverage: "insufficient_data" | "partial" | "full" = "insufficient_data";
  if (samples.length > 0) {
    if (filledMarkoutSlots === totalMarkoutSlots) markoutCoverage = "full";
    else if (filledMarkoutSlots > 0) markoutCoverage = "partial";
  }

  return {
    sampleCount: samples.length,
    averageSignalToEnqueueLatencyMs: average(enqueueLats),
    p95SignalToEnqueueLatencyMs: percentile(enqueueLats, 95),
    averageDispatchLatencyMs: dispatchLats.length > 0 ? average(dispatchLats) : null,
    p95DispatchLatencyMs: dispatchLats.length > 0 ? percentile(dispatchLats, 95) : null,
    staleSignalCount: stale,
    priceMovedAgainstSignalCount: movedAgainst,
    priceMovedInFavorCount: movedFavor,
    averageSlippageBps: avgSlippage,
    markout5s: average(m5),
    markout30s: average(m30),
    markout60s: average(m60),
    markoutCoverage,
  };
}
