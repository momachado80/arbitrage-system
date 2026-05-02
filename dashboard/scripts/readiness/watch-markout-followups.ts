/**
 * Microcapital Readiness Gate — live markout follow-up watcher (paper-only).
 *
 * Watches PAPER_STATE_DIR for new `paper_cycle_positive` assessments emitted
 * by the cross-venue-anchor 1823789 paper-execution worker, and produces
 * markout follow-ups at +5s / +30s / +60s by sampling a public, read-only
 * price source.
 *
 * GUARDRAILS — this script:
 *   - is paper / shadow / read-only against external systems
 *   - reads the existing assessment JSONL (never writes to it)
 *   - appends to `cross-venue-anchor-1823789-markout-followups.jsonl`
 *   - persists progress to `cross-venue-anchor-1823789-markout-watch-state.json`
 *   - DOES NOT submit any order
 *   - DOES NOT call any financial endpoint
 *   - DOES NOT sign any transaction
 *   - DOES NOT touch a wallet, signer, or private key
 *
 * Policy: shadow_only_no_api_submit_v1.
 *
 * Usage:
 *   ts-node -P tsconfig.worker.json scripts/readiness/watch-markout-followups.ts --once
 *   ts-node -P tsconfig.worker.json scripts/readiness/watch-markout-followups.ts --watch
 */

import fs from "fs";
import path from "path";

import {
  resolvePaperStateDir,
  PAPER_TRAIL_FILENAMES,
} from "../../lib/paperStateDir";
import {
  readPaperExecutionAssessmentsFromJsonl,
  type PaperExecutionAssessment,
} from "../../lib/readiness/paperExecutionAssessmentParser";
import { MICROCAPITAL_READINESS_THRESHOLDS } from "../../lib/readiness/microcapitalReadinessThresholds";
import { buildPaperCycleId } from "../../lib/readiness/paperCycleLifecycle";
import { fetchGammaMarketRawJson } from "../../lib/clobMicrostructure";

export const WATCH_STATE_FILENAME =
  "cross-venue-anchor-1823789-markout-watch-state.json" as const;
export const WATCH_SCHEMA_VERSION = "v1" as const;

export type SamplerStatus = "ok" | "price_unavailable" | "sampler_error";

export type PriceSource =
  | "bid_ask_mid"
  | "best_ask_only"
  | "best_bid_only"
  | "last_trade_price"
  | "last_price"
  | "price"
  | "outcome_prices"
  | "tokens_outcomes_price";

export interface SamplerResult {
  status: SamplerStatus;
  price: number | null;
  errorMessage?: string;
  /** Which public field the price came from. Present when status === "ok". */
  priceSource?: PriceSource;
  /**
   * Short list of price-related field names actually observed in the public
   * payload. NEVER includes the raw payload itself; only field-name strings.
   */
  rawPriceFieldsSeen?: string[];
}

export interface ReadonlyPriceSampler {
  /**
   * Sample current mid price for a marketId. NEVER mutates remote state.
   * Implementations MUST be read-only — no submit, no signing, no wallet.
   */
  samplePrice(marketId: string): Promise<SamplerResult>;
}

/* ------------------------- public-payload extraction ------------------------ */

function asNumber(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function isProbabilityish(n: number): boolean {
  return Number.isFinite(n) && n > 0 && n < 1;
}

function parseStringArrayField(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(v => String(v)).filter(Boolean);
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value) as unknown;
      return Array.isArray(parsed) ? parsed.map(v => String(v)).filter(Boolean) : [];
    } catch {
      return [];
    }
  }
  return [];
}

interface PublicPriceExtraction {
  price: number;
  priceSource: PriceSource;
  rawPriceFieldsSeen: string[];
}

/**
 * Read-only fallback chain over public Gamma payload fields.
 *
 * Priority:
 *   1. bid + ask both valid     → (bid+ask)/2 with priceSource="bid_ask_mid"
 *   2. only valid bestAsk       → bestAsk          ("best_ask_only")
 *   3. only valid bestBid       → bestBid          ("best_bid_only")
 *   4. lastTradePrice valid     → lastTradePrice   ("last_trade_price")
 *   5. lastPrice valid          → lastPrice        ("last_price")
 *   6. top-level `price` valid  → price            ("price")
 *   7. outcomePrices array      → first valid      ("outcome_prices")
 *   8. tokens[] / outcomes[].price valid → ("tokens_outcomes_price")
 *
 * Returns null when no usable public price exists. NEVER touches private data.
 */
export function extractGammaPublicPriceWithFallback(
  raw: Record<string, unknown>,
): PublicPriceExtraction | null {
  const seen: string[] = [];

  const bid = asNumber(raw.bestBid);
  const ask = asNumber(raw.bestAsk);
  const bidValid = bid !== null && isProbabilityish(bid);
  const askValid = ask !== null && isProbabilityish(ask);
  if (bid !== null) seen.push("bestBid");
  if (ask !== null) seen.push("bestAsk");

  if (bidValid && askValid && (ask as number) > (bid as number)) {
    return {
      price: ((bid as number) + (ask as number)) / 2,
      priceSource: "bid_ask_mid",
      rawPriceFieldsSeen: seen,
    };
  }
  if (askValid && !bidValid) {
    return {
      price: ask as number,
      priceSource: "best_ask_only",
      rawPriceFieldsSeen: seen,
    };
  }
  if (bidValid && !askValid) {
    return {
      price: bid as number,
      priceSource: "best_bid_only",
      rawPriceFieldsSeen: seen,
    };
  }

  const lastTrade = asNumber(raw.lastTradePrice);
  if (lastTrade !== null) seen.push("lastTradePrice");
  if (lastTrade !== null && isProbabilityish(lastTrade)) {
    return {
      price: lastTrade,
      priceSource: "last_trade_price",
      rawPriceFieldsSeen: seen,
    };
  }

  const lastPrice = asNumber(raw.lastPrice);
  if (lastPrice !== null) seen.push("lastPrice");
  if (lastPrice !== null && isProbabilityish(lastPrice)) {
    return {
      price: lastPrice,
      priceSource: "last_price",
      rawPriceFieldsSeen: seen,
    };
  }

  const topLevelPrice = asNumber(raw.price);
  if (topLevelPrice !== null) seen.push("price");
  if (topLevelPrice !== null && isProbabilityish(topLevelPrice)) {
    return {
      price: topLevelPrice,
      priceSource: "price",
      rawPriceFieldsSeen: seen,
    };
  }

  const outcomePrices = parseStringArrayField(raw.outcomePrices);
  if (outcomePrices.length > 0) seen.push("outcomePrices");
  if (outcomePrices.length > 0) {
    // Pair with outcomes; prefer the YES slot if identifiable.
    const outcomes = parseStringArrayField(raw.outcomes);
    let chosenIdx = 0;
    if (outcomes.length === outcomePrices.length) {
      const yesIdx = outcomes.findIndex(o => /^yes$/i.test(o.trim()));
      if (yesIdx >= 0) chosenIdx = yesIdx;
    }
    const candidate = asNumber(outcomePrices[chosenIdx]);
    if (candidate !== null && isProbabilityish(candidate)) {
      return {
        price: candidate,
        priceSource: "outcome_prices",
        rawPriceFieldsSeen: seen,
      };
    }
  }

  // tokens / outcomes objects with embedded price fields.
  const tokens = Array.isArray(raw.tokens) ? (raw.tokens as unknown[]) : [];
  if (tokens.length > 0) seen.push("tokens");
  for (const t of tokens) {
    if (typeof t !== "object" || t === null) continue;
    const tObj = t as Record<string, unknown>;
    const candidate = asNumber(tObj.price);
    if (candidate !== null && isProbabilityish(candidate)) {
      return {
        price: candidate,
        priceSource: "tokens_outcomes_price",
        rawPriceFieldsSeen: seen,
      };
    }
  }

  return null;
}

/**
 * Public read-only sampler using the existing Gamma helpers from
 * `lib/clobMicrostructure.ts`. Fetches market raw JSON and returns
 * (bestBid + bestAsk) / 2 as mid. Errors are converted to safe statuses.
 */
export function createGammaPriceSampler(): ReadonlyPriceSampler {
  return {
    async samplePrice(marketId: string): Promise<SamplerResult> {
      try {
        const raw = await fetchGammaMarketRawJson(marketId);
        if (!raw) {
          return {
            status: "price_unavailable",
            price: null,
            errorMessage: "gamma_returned_null",
            rawPriceFieldsSeen: [],
          };
        }
        const extraction = extractGammaPublicPriceWithFallback(raw);
        if (!extraction) {
          return {
            status: "price_unavailable",
            price: null,
            errorMessage: "no_usable_public_price",
            rawPriceFieldsSeen: [],
          };
        }
        return {
          status: "ok",
          price: extraction.price,
          priceSource: extraction.priceSource,
          rawPriceFieldsSeen: extraction.rawPriceFieldsSeen,
        };
      } catch (err) {
        return {
          status: "sampler_error",
          price: null,
          errorMessage: err instanceof Error ? err.message : String(err),
        };
      }
    },
  };
}

/**
 * Stub sampler for tests. `prices` maps marketId → price. A `null` value
 * yields `price_unavailable`; missing keys yield `price_unavailable`.
 *
 * Optionally `pricesAdvanced` provides a per-market `priceSource` /
 * `rawPriceFieldsSeen` so tests can exercise the fallback chain shape
 * end-to-end without going through the network.
 *
 * Optionally `gammaPayloads` provides a per-market raw payload that the
 * stub feeds through the real {@link extractGammaPublicPriceWithFallback}
 * helper — exercising the production extraction in tests.
 */
export function createStubPriceSampler(
  prices: ReadonlyMap<string, number | null>,
  opts: {
    errorOnMarketIds?: ReadonlySet<string>;
    pricesAdvanced?: ReadonlyMap<
      string,
      { price: number; priceSource: PriceSource; rawPriceFieldsSeen: string[] }
    >;
    gammaPayloads?: ReadonlyMap<string, Record<string, unknown>>;
  } = {},
): ReadonlyPriceSampler {
  return {
    async samplePrice(marketId: string): Promise<SamplerResult> {
      if (opts.errorOnMarketIds?.has(marketId)) {
        return {
          status: "sampler_error",
          price: null,
          errorMessage: "stub_forced_error",
        };
      }
      if (opts.gammaPayloads?.has(marketId)) {
        const payload = opts.gammaPayloads.get(marketId)!;
        const extraction = extractGammaPublicPriceWithFallback(payload);
        if (!extraction) {
          return {
            status: "price_unavailable",
            price: null,
            errorMessage: "no_usable_public_price",
            rawPriceFieldsSeen: [],
          };
        }
        return {
          status: "ok",
          price: extraction.price,
          priceSource: extraction.priceSource,
          rawPriceFieldsSeen: extraction.rawPriceFieldsSeen,
        };
      }
      if (opts.pricesAdvanced?.has(marketId)) {
        const adv = opts.pricesAdvanced.get(marketId)!;
        return {
          status: "ok",
          price: adv.price,
          priceSource: adv.priceSource,
          rawPriceFieldsSeen: adv.rawPriceFieldsSeen,
        };
      }
      if (!prices.has(marketId)) {
        return {
          status: "price_unavailable",
          price: null,
          errorMessage: "stub_no_entry",
        };
      }
      const p = prices.get(marketId);
      if (p === null || p === undefined) {
        return { status: "price_unavailable", price: null };
      }
      return {
        status: "ok",
        price: p,
        priceSource: "bid_ask_mid",
        rawPriceFieldsSeen: ["bestBid", "bestAsk"],
      };
    },
  };
}

interface ScheduledFollowup {
  cycleId: string;
  marketId: string;
  basePrice: number;
  baseObservedAt: number;
  horizonMs: 5_000 | 30_000 | 60_000;
  dueAtMs: number;
}

interface WatchState {
  schemaVersion: typeof WATCH_SCHEMA_VERSION;
  seenAssessmentKeys: string[];
  scheduledFollowups: ScheduledFollowup[];
  completedFollowups: string[];
  failedFollowups: string[];
  updatedAt: string;
}

function defaultState(): WatchState {
  return {
    schemaVersion: WATCH_SCHEMA_VERSION,
    seenAssessmentKeys: [],
    scheduledFollowups: [],
    completedFollowups: [],
    failedFollowups: [],
    updatedAt: new Date(0).toISOString(),
  };
}

function readWatchState(filePath: string): WatchState {
  let raw: string;
  try {
    raw = fs.readFileSync(filePath, "utf8");
  } catch {
    return defaultState();
  }
  let parsed: Partial<WatchState>;
  try {
    parsed = JSON.parse(raw) as Partial<WatchState>;
  } catch {
    return defaultState();
  }
  return {
    schemaVersion: WATCH_SCHEMA_VERSION,
    seenAssessmentKeys: Array.isArray(parsed.seenAssessmentKeys)
      ? parsed.seenAssessmentKeys.filter((s): s is string => typeof s === "string")
      : [],
    scheduledFollowups: Array.isArray(parsed.scheduledFollowups)
      ? parsed.scheduledFollowups.filter(isScheduledFollowup)
      : [],
    completedFollowups: Array.isArray(parsed.completedFollowups)
      ? parsed.completedFollowups.filter((s): s is string => typeof s === "string")
      : [],
    failedFollowups: Array.isArray(parsed.failedFollowups)
      ? parsed.failedFollowups.filter((s): s is string => typeof s === "string")
      : [],
    updatedAt: typeof parsed.updatedAt === "string" ? parsed.updatedAt : new Date(0).toISOString(),
  };
}

function isScheduledFollowup(v: unknown): v is ScheduledFollowup {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return (
    typeof o.cycleId === "string" &&
    typeof o.marketId === "string" &&
    typeof o.basePrice === "number" &&
    typeof o.baseObservedAt === "number" &&
    (o.horizonMs === 5_000 || o.horizonMs === 30_000 || o.horizonMs === 60_000) &&
    typeof o.dueAtMs === "number"
  );
}

function writeWatchState(filePath: string, state: WatchState): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(
    filePath,
    JSON.stringify({ ...state, updatedAt: new Date().toISOString() }, null, 2),
    "utf8",
  );
}

function basePriceOf(a: PaperExecutionAssessment): number | null {
  if (a.clobMid !== null) return a.clobMid;
  if (a.clobBestBid !== null && a.clobBestAsk !== null) {
    return (a.clobBestBid + a.clobBestAsk) / 2;
  }
  return null;
}

function isPositive(a: PaperExecutionAssessment): boolean {
  return (a.paperExecutionVerdict ?? "").toLowerCase() === "paper_cycle_positive";
}

function assessmentKey(a: PaperExecutionAssessment): string {
  return `${a.isoTimestamp ?? "unknown"}::${a.marketId ?? "unknown"}::${a.estimatedNetAfterPaperExecution ?? "n/a"}`;
}

interface MarkoutFollowupRowV1 {
  type: "markout_followup";
  schemaVersion: typeof WATCH_SCHEMA_VERSION;
  source: "paper_shadow_readonly_live_sampler";
  cycleId: string;
  marketId: string;
  baseObservedAt: number;
  followupObservedAt: number;
  horizonMs: 5_000 | 30_000 | 60_000;
  basePrice: number;
  followupPrice: number | null;
  markout: number | null;
  samplerStatus: SamplerStatus;
  errorMessage?: string;
  /** Which public field the price came from. Present when samplerStatus="ok". */
  priceSource?: PriceSource;
  /** Short list of price-related field names actually observed (no payload). */
  rawPriceFieldsSeen?: string[];
}

function appendJsonl(filePath: string, rows: ReadonlyArray<object>): void {
  if (rows.length === 0) return;
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const lines = rows.map(r => JSON.stringify(r)).join("\n");
  fs.appendFileSync(filePath, lines + "\n", "utf8");
}

function readExistingFollowupKeys(filePath: string): Set<string> {
  const out = new Set<string>();
  let raw: string;
  try {
    raw = fs.readFileSync(filePath, "utf8");
  } catch {
    return out;
  }
  for (const line of raw.split(/\r?\n/)) {
    const t = line.trim();
    if (!t) continue;
    try {
      const obj = JSON.parse(t) as Record<string, unknown>;
      const cid = typeof obj.cycleId === "string" ? obj.cycleId : null;
      const h = typeof obj.horizonMs === "number" ? obj.horizonMs : null;
      if (cid && h !== null) out.add(`${cid}::${h}`);
    } catch {
      // skip
    }
  }
  return out;
}

export interface RunOnceResult {
  newPositivesSeen: number;
  followupsScheduledTotal: number;
  followupsCompleted: number;
  followupsSkippedDuplicate: number;
  followupsFailed: number;
  followupsPending: number;
}

export interface RunWatchOptions {
  paperStateDir?: string;
  sampler?: ReadonlyPriceSampler;
  /** Override Date.now() for tests / determinism. */
  nowMs?: number;
}

export async function runOnce(opts: RunWatchOptions = {}): Promise<RunOnceResult> {
  const stateDir = opts.paperStateDir ?? resolvePaperStateDir();
  const sampler = opts.sampler ?? createGammaPriceSampler();
  const nowMs = typeof opts.nowMs === "number" ? opts.nowMs : Date.now();

  const T = MICROCAPITAL_READINESS_THRESHOLDS;
  const horizons = T.MARKOUT_HORIZONS_MS;

  const historyPath = path.join(
    stateDir,
    PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789PaperExecutionHistory,
  );
  const followupsPath = path.join(stateDir, T.MARKOUT_FOLLOWUPS_FILENAME);
  const statePath = path.join(stateDir, WATCH_STATE_FILENAME);

  const state = readWatchState(statePath);
  const seenKeys = new Set(state.seenAssessmentKeys);
  const completed = new Set(state.completedFollowups);
  for (const k of Array.from(readExistingFollowupKeys(followupsPath))) {
    completed.add(k);
  }

  const assessments = readPaperExecutionAssessmentsFromJsonl(historyPath);
  const positives = assessments.filter(isPositive);

  let newPositivesSeen = 0;
  for (const a of positives) {
    const key = assessmentKey(a);
    if (seenKeys.has(key)) continue;
    if (!a.isoTimestamp) {
      seenKeys.add(key);
      continue;
    }
    const baseObservedAt = Date.parse(a.isoTimestamp);
    const basePrice = basePriceOf(a);
    if (!Number.isFinite(baseObservedAt) || basePrice === null) {
      seenKeys.add(key);
      continue;
    }
    const cycleId = buildPaperCycleId(a);
    for (const h of horizons) {
      const fkey = `${cycleId}::${h}`;
      if (completed.has(fkey)) continue;
      if (
        state.scheduledFollowups.find(
          s => s.cycleId === cycleId && s.horizonMs === h,
        )
      ) {
        continue;
      }
      state.scheduledFollowups.push({
        cycleId,
        marketId: a.marketId ?? "unknown",
        basePrice,
        baseObservedAt,
        horizonMs: h as 5_000 | 30_000 | 60_000,
        dueAtMs: baseObservedAt + h,
      });
    }
    seenKeys.add(key);
    newPositivesSeen++;
  }

  const followupsScheduledTotal = state.scheduledFollowups.length;
  let followupsCompleted = 0;
  let followupsFailed = 0;
  let followupsSkippedDuplicate = 0;
  const remaining: ScheduledFollowup[] = [];

  for (const sched of state.scheduledFollowups) {
    const fkey = `${sched.cycleId}::${sched.horizonMs}`;
    if (completed.has(fkey)) {
      followupsSkippedDuplicate++;
      continue;
    }
    if (sched.dueAtMs > nowMs) {
      remaining.push(sched);
      continue;
    }

    let sample: SamplerResult;
    try {
      sample = await sampler.samplePrice(sched.marketId);
    } catch (err) {
      sample = {
        status: "sampler_error",
        price: null,
        errorMessage: err instanceof Error ? err.message : String(err),
      };
    }

    const followupObservedAt = nowMs;
    const followupPrice = sample.status === "ok" ? sample.price : null;
    const markout =
      sample.status === "ok" && followupPrice !== null
        ? followupPrice - sched.basePrice
        : null;

    const row: MarkoutFollowupRowV1 = {
      type: "markout_followup",
      schemaVersion: WATCH_SCHEMA_VERSION,
      source: "paper_shadow_readonly_live_sampler",
      cycleId: sched.cycleId,
      marketId: sched.marketId,
      baseObservedAt: sched.baseObservedAt,
      followupObservedAt,
      horizonMs: sched.horizonMs,
      basePrice: sched.basePrice,
      followupPrice,
      markout,
      samplerStatus: sample.status,
      ...(sample.errorMessage ? { errorMessage: sample.errorMessage } : {}),
      ...(sample.priceSource ? { priceSource: sample.priceSource } : {}),
      ...(sample.rawPriceFieldsSeen
        ? { rawPriceFieldsSeen: sample.rawPriceFieldsSeen.slice(0, 12) }
        : {}),
    };

    appendJsonl(followupsPath, [row]);
    completed.add(fkey);
    if (sample.status === "ok") {
      followupsCompleted++;
    } else {
      followupsFailed++;
      state.failedFollowups.push(`${fkey}::${sample.status}`);
    }
  }

  state.scheduledFollowups = remaining;
  state.seenAssessmentKeys = Array.from(seenKeys);
  state.completedFollowups = Array.from(completed);
  writeWatchState(statePath, state);

  return {
    newPositivesSeen,
    followupsScheduledTotal,
    followupsCompleted,
    followupsSkippedDuplicate,
    followupsFailed,
    followupsPending: remaining.length,
  };
}

export interface RunWatchLoopOptions extends RunWatchOptions {
  pollIntervalMs?: number;
  maxDurationMs?: number;
  /** Test hook: stop after N iterations regardless of duration. */
  maxIterations?: number;
}

export async function runWatchLoop(
  opts: RunWatchLoopOptions = {},
): Promise<{ iterations: number; lastResult: RunOnceResult | null }> {
  const interval = opts.pollIntervalMs ?? 5_000;
  const maxDuration = opts.maxDurationMs ?? 30 * 60 * 1000;
  const maxIterations = opts.maxIterations ?? Number.POSITIVE_INFINITY;
  const start = Date.now();
  let iterations = 0;
  let lastResult: RunOnceResult | null = null;
  while (Date.now() - start < maxDuration && iterations < maxIterations) {
    try {
      lastResult = await runOnce(opts);
      console.log(`[watch-markouts] ${JSON.stringify(lastResult)}`);
    } catch (err) {
      console.error(
        `[watch-markouts] error: ${err instanceof Error ? err.message : err}`,
      );
    }
    iterations++;
    if (iterations >= maxIterations) break;
    await new Promise(res => setTimeout(res, interval));
  }
  return { iterations, lastResult };
}

function main(): void {
  const args = process.argv.slice(2);
  const watch = args.includes("--watch");
  const once = args.includes("--once") || !watch;

  if (once && !watch) {
    runOnce()
      .then(r => {
        process.stdout.write(JSON.stringify(r, null, 2) + "\n");
        process.exit(0);
      })
      .catch(err => {
        console.error("[watch-markouts] fatal:", err);
        process.exit(1);
      });
    return;
  }
  runWatchLoop().catch(err => {
    console.error("[watch-markouts] fatal:", err);
    process.exit(1);
  });
}

if (require.main === module) {
  main();
}
