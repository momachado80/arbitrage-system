/**
 * Microcapital Readiness Gate — markout follow-up producer (paper-only).
 *
 * Pure builder + tiny in-memory registry that turns base price snapshots into
 * markout follow-up records at +5s / +30s / +60s. The integration with any
 * operational worker is intentionally minimal: we expose pure functions and
 * a registry. We do NOT modify the running worker in this commit; consumers
 * may opt-in by calling `recordFollowup()` from a paper-only sidecar.
 *
 * GUARDRAILS — this module:
 *   - does NOT submit any order
 *   - does NOT call any financial endpoint
 *   - does NOT sign any transaction
 *   - does NOT touch a wallet, signer, or private key
 *   - does NOT authorize real execution
 *   - reads only data already produced by paper/shadow paths
 *
 * Policy: shadow_only_no_api_submit_v1.
 */

import { MICROCAPITAL_READINESS_THRESHOLDS } from "./microcapitalReadinessThresholds";

export type MarkoutHorizonMs = 5_000 | 30_000 | 60_000;

export const MARKOUT_HORIZONS_MS: ReadonlyArray<MarkoutHorizonMs> =
  MICROCAPITAL_READINESS_THRESHOLDS.MARKOUT_HORIZONS_MS as ReadonlyArray<MarkoutHorizonMs>;

export interface MarkoutBaseSample {
  cycleId: string;
  marketId: string;
  /** Epoch ms at which the signal/observation was recorded. */
  signalTimestamp: number;
  /** Reference price/probability snapshot at signalTimestamp (paper-only). */
  basePrice: number;
}

export interface MarkoutFollowupInput {
  cycleId: string;
  /** Which horizon this follow-up resolves. */
  horizonMs: MarkoutHorizonMs;
  /** Epoch ms at which the follow-up read was taken. */
  followupTimestamp: number;
  /** Observed paper-only price/probability at followupTimestamp. */
  followupPrice: number;
}

export interface MarkoutFollowupRecord {
  type: "markout_followup";
  cycleId: string;
  marketId: string;
  baseObservedAt: number;
  followupObservedAt: number;
  horizonMs: MarkoutHorizonMs;
  basePrice: number;
  followupPrice: number;
  markout: number;
  source: "paper_shadow_readonly";
}

export interface MarkoutPriceSet {
  priceAfter5s?: number;
  priceAfter30s?: number;
  priceAfter60s?: number;
}

function validateBaseSample(sample: MarkoutBaseSample): void {
  if (typeof sample !== "object" || sample === null) {
    throw new Error("markoutFollowupProducer.invalid_base_sample");
  }
  if (typeof sample.cycleId !== "string" || sample.cycleId.length === 0) {
    throw new Error("markoutFollowupProducer.missing_cycleId");
  }
  if (typeof sample.marketId !== "string" || sample.marketId.length === 0) {
    throw new Error("markoutFollowupProducer.missing_marketId");
  }
  if (typeof sample.signalTimestamp !== "number" || !Number.isFinite(sample.signalTimestamp)) {
    throw new Error("markoutFollowupProducer.invalid_signalTimestamp");
  }
  if (typeof sample.basePrice !== "number" || !Number.isFinite(sample.basePrice)) {
    throw new Error("markoutFollowupProducer.invalid_basePrice");
  }
}

function validateFollowup(input: MarkoutFollowupInput): void {
  if (typeof input !== "object" || input === null) {
    throw new Error("markoutFollowupProducer.invalid_followup");
  }
  if (typeof input.cycleId !== "string" || input.cycleId.length === 0) {
    throw new Error("markoutFollowupProducer.missing_cycleId");
  }
  if (typeof input.followupTimestamp !== "number" || !Number.isFinite(input.followupTimestamp)) {
    throw new Error("markoutFollowupProducer.invalid_followupTimestamp");
  }
  if (typeof input.followupPrice !== "number" || !Number.isFinite(input.followupPrice)) {
    throw new Error("markoutFollowupProducer.invalid_followupPrice");
  }
  if (!MARKOUT_HORIZONS_MS.includes(input.horizonMs)) {
    throw new Error(`markoutFollowupProducer.invalid_horizon:${input.horizonMs}`);
  }
}

/**
 * Pure builder for a follow-up record. No I/O.
 */
export function buildMarkoutFollowupRecord(
  base: MarkoutBaseSample,
  followup: MarkoutFollowupInput,
): MarkoutFollowupRecord {
  validateBaseSample(base);
  validateFollowup(followup);
  if (followup.cycleId !== base.cycleId) {
    throw new Error("markoutFollowupProducer.cycleId_mismatch");
  }
  return {
    type: "markout_followup",
    cycleId: base.cycleId,
    marketId: base.marketId,
    baseObservedAt: base.signalTimestamp,
    followupObservedAt: followup.followupTimestamp,
    horizonMs: followup.horizonMs,
    basePrice: base.basePrice,
    followupPrice: followup.followupPrice,
    markout: followup.followupPrice - base.basePrice,
    source: "paper_shadow_readonly",
  };
}

export interface MarkoutFollowupProducer {
  /** Register a base sample. Idempotent on duplicate cycleId+marketId. */
  registerBaseSample(sample: MarkoutBaseSample): void;
  /**
   * Resolve a single follow-up. Returns the record for caller to append to
   * the JSONL audit stream (paper/shadow). Throws if no base sample exists.
   */
  recordFollowup(followup: MarkoutFollowupInput): MarkoutFollowupRecord;
  /** Snapshot of all follow-ups seen so far for a given cycle. */
  getFollowupsForCycle(cycleId: string): ReadonlyArray<MarkoutFollowupRecord>;
  /**
   * Aggregate all resolved follow-ups for a cycle into the
   * priceAfter5s/30s/60s shape consumed by ExecutionRealismHarness.
   */
  toMarkoutPriceSet(cycleId: string): MarkoutPriceSet;
  /** Drop a cycle (after sample emission, GC). */
  forget(cycleId: string): void;
}

/**
 * In-memory registry. Pure — does NOT touch disk. Consumers (e.g. a paper-only
 * sidecar) are responsible for persisting returned records to the existing
 * shadow-audit JSONL. We expose `buildMarkoutFollowupRecord` for callers that
 * want the record without the registry.
 */
export function createMarkoutFollowupProducer(): MarkoutFollowupProducer {
  const bases = new Map<string, MarkoutBaseSample>();
  const followups = new Map<string, MarkoutFollowupRecord[]>();

  function registerBaseSample(sample: MarkoutBaseSample): void {
    validateBaseSample(sample);
    bases.set(sample.cycleId, sample);
    if (!followups.has(sample.cycleId)) followups.set(sample.cycleId, []);
  }

  function recordFollowup(input: MarkoutFollowupInput): MarkoutFollowupRecord {
    validateFollowup(input);
    const base = bases.get(input.cycleId);
    if (!base) throw new Error("markoutFollowupProducer.unknown_cycleId");
    const record = buildMarkoutFollowupRecord(base, input);
    const list = followups.get(input.cycleId) ?? [];
    list.push(record);
    followups.set(input.cycleId, list);
    return record;
  }

  function getFollowupsForCycle(cycleId: string): ReadonlyArray<MarkoutFollowupRecord> {
    return followups.get(cycleId) ?? [];
  }

  function toMarkoutPriceSet(cycleId: string): MarkoutPriceSet {
    const list = followups.get(cycleId) ?? [];
    const set: MarkoutPriceSet = {};
    for (const r of list) {
      if (r.horizonMs === 5_000) set.priceAfter5s = r.followupPrice;
      else if (r.horizonMs === 30_000) set.priceAfter30s = r.followupPrice;
      else if (r.horizonMs === 60_000) set.priceAfter60s = r.followupPrice;
    }
    return set;
  }

  function forget(cycleId: string): void {
    bases.delete(cycleId);
    followups.delete(cycleId);
  }

  return {
    registerBaseSample,
    recordFollowup,
    getFollowupsForCycle,
    toMarkoutPriceSet,
    forget,
  };
}
