/**
 * Tests — live markout follow-up watcher (paper-only, read-only).
 *
 * Paper/shadow only. Uses a stub sampler so no network calls. NEVER touches
 * submit/wallet/signer/private key surfaces.
 *
 * Policy: shadow_only_no_api_submit_v1.
 */

import fs from "fs";
import os from "os";
import path from "path";

import {
  createStubPriceSampler,
  extractGammaPublicPriceWithFallback,
  runOnce,
  runWatchLoop,
  WATCH_STATE_FILENAME,
} from "../../scripts/readiness/watch-markout-followups";

import { assertEqual, assertTrue, describe, test } from "./_assert";

const HISTORY_FILENAME = "cross-venue-anchor-1823789-paper-execution-history.jsonl";
const FOLLOWUPS_FILENAME = "cross-venue-anchor-1823789-markout-followups.jsonl";

const T0 = Date.parse("2026-04-29T12:00:00.000Z");

function isoAt(ms: number): string {
  return new Date(ms).toISOString();
}

function makeStateDir(jsonl: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "watch-markouts-"));
  fs.writeFileSync(path.join(dir, HISTORY_FILENAME), jsonl, "utf8");
  return dir;
}

function readJsonl(filePath: string): Record<string, unknown>[] {
  if (!fs.existsSync(filePath)) return [];
  return fs
    .readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .map(s => JSON.parse(s) as Record<string, unknown>);
}

function readState(stateDir: string): Record<string, unknown> {
  const p = path.join(stateDir, WATCH_STATE_FILENAME);
  if (!fs.existsSync(p)) return {};
  return JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, unknown>;
}

const POSITIVE_LINE_AT_T0 = JSON.stringify({
  isoTimestamp: isoAt(T0),
  marketId: "1823789",
  paperExecutionVerdict: "paper_cycle_positive",
  estimatedNetAfterPaperExecution: 0.015164,
  clobMid: 0.5,
  clobBestBid: 0.49,
  clobBestAsk: 0.51,
});

const BLOCKED_LINE_AT_T0 = JSON.stringify({
  isoTimestamp: isoAt(T0 + 1_000),
  marketId: "1823789",
  paperExecutionVerdict: "blocked_by_gate",
});

describe("tests/readiness/markoutWatcher.test.ts", () => {
  test("detects new positive assessment and schedules 5s/30s/60s", async () => {
    const dir = makeStateDir([POSITIVE_LINE_AT_T0, BLOCKED_LINE_AT_T0].join("\n"));
    // nowMs is BEFORE +5s — so all three followups are scheduled but pending.
    const r = await runOnce({
      paperStateDir: dir,
      sampler: createStubPriceSampler(new Map()),
      nowMs: T0 + 1_000,
    });
    assertEqual(r.newPositivesSeen, 1, "1 positive seen");
    assertEqual(r.followupsScheduledTotal, 3, "3 horizons scheduled");
    assertEqual(r.followupsCompleted, 0, "none due yet");
    assertEqual(r.followupsPending, 3, "all 3 pending");
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("ignores blocked_by_gate", async () => {
    const dir = makeStateDir(BLOCKED_LINE_AT_T0);
    const r = await runOnce({
      paperStateDir: dir,
      sampler: createStubPriceSampler(new Map()),
      nowMs: T0 + 1_000,
    });
    assertEqual(r.newPositivesSeen, 0, "no positives");
    assertEqual(r.followupsScheduledTotal, 0, "no schedule");
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("sampler ok writes markout rows when followups are due", async () => {
    const dir = makeStateDir(POSITIVE_LINE_AT_T0);
    const sampler = createStubPriceSampler(
      new Map<string, number | null>([["1823789", 0.51]]),
    );
    // First pass: schedule.
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 1_000 });
    // Second pass at T0+90s: all three horizons due.
    const r = await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 90_000 });
    assertEqual(r.followupsCompleted, 3, "3 completed");
    assertEqual(r.followupsFailed, 0, "no failures");
    const rows = readJsonl(path.join(dir, FOLLOWUPS_FILENAME));
    assertEqual(rows.length, 3, "3 rows on disk");
    for (const row of rows) {
      assertEqual(row.samplerStatus, "ok", "status ok");
      assertEqual(
        row.source,
        "paper_shadow_readonly_live_sampler",
        "live sampler source tag",
      );
      assertEqual(row.basePrice, 0.5, "basePrice from assessment mid");
      assertEqual(row.followupPrice, 0.51, "followupPrice from sampler");
      assertTrue(
        Math.abs(Number(row.markout) - 0.01) < 1e-9,
        `markout ≈ 0.01 (got ${row.markout})`,
      );
    }
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("sampler_error does NOT crash; records failure in state and JSONL", async () => {
    const dir = makeStateDir(POSITIVE_LINE_AT_T0);
    const sampler = createStubPriceSampler(new Map(), {
      errorOnMarketIds: new Set(["1823789"]),
    });
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 1_000 });
    const r = await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 90_000 });
    assertEqual(r.followupsFailed, 3, "3 failed");
    assertEqual(r.followupsCompleted, 0, "none completed");
    const rows = readJsonl(path.join(dir, FOLLOWUPS_FILENAME));
    assertEqual(rows.length, 3, "3 rows on disk anyway");
    for (const row of rows) {
      assertEqual(row.samplerStatus, "sampler_error", "samplerStatus=sampler_error");
      assertEqual(row.followupPrice, null, "no price");
      assertEqual(row.markout, null, "no markout");
      assertTrue(typeof row.errorMessage === "string", "errorMessage present");
    }
    const st = readState(dir);
    const failed = (st.failedFollowups ?? []) as string[];
    assertEqual(failed.length, 3, "state failed list has 3");
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("does NOT duplicate followup existing on disk", async () => {
    const dir = makeStateDir(POSITIVE_LINE_AT_T0);
    const sampler = createStubPriceSampler(
      new Map<string, number | null>([["1823789", 0.51]]),
    );
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 1_000 });
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 90_000 });
    const after1 = readJsonl(path.join(dir, FOLLOWUPS_FILENAME));
    const r3 = await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 200_000 });
    const after2 = readJsonl(path.join(dir, FOLLOWUPS_FILENAME));
    assertEqual(after1.length, 3, "first pass wrote 3");
    assertEqual(after2.length, 3, "second pass added none (idempotent)");
    assertEqual(r3.followupsCompleted, 0, "second-pass completed = 0");
    assertEqual(r3.followupsScheduledTotal, 0, "no new schedules");
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("writes state file with seen + completed + scheduled lists", async () => {
    const dir = makeStateDir(POSITIVE_LINE_AT_T0);
    const sampler = createStubPriceSampler(
      new Map<string, number | null>([["1823789", 0.51]]),
    );
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 1_000 });
    let st = readState(dir);
    assertEqual(st.schemaVersion, "v1", "schemaVersion v1");
    assertEqual((st.seenAssessmentKeys as string[]).length, 1, "1 seen key");
    assertEqual(
      (st.scheduledFollowups as unknown[]).length,
      3,
      "3 scheduled before due",
    );
    assertEqual(
      (st.completedFollowups as string[]).length,
      0,
      "none completed yet",
    );
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 90_000 });
    st = readState(dir);
    assertEqual(
      (st.scheduledFollowups as unknown[]).length,
      0,
      "all due → drained",
    );
    assertEqual(
      (st.completedFollowups as string[]).length,
      3,
      "3 completed in state",
    );
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("watch loop with maxIterations stops cleanly", async () => {
    const dir = makeStateDir(POSITIVE_LINE_AT_T0);
    const sampler = createStubPriceSampler(
      new Map<string, number | null>([["1823789", 0.51]]),
    );
    const r = await runWatchLoop({
      paperStateDir: dir,
      sampler,
      nowMs: T0 + 1_000,
      pollIntervalMs: 1,
      maxIterations: 3,
    });
    assertEqual(r.iterations, 3, "exactly 3 iterations");
    assertTrue(r.lastResult !== null, "lastResult populated");
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("price_unavailable status writes row but does NOT count as completed", async () => {
    const dir = makeStateDir(POSITIVE_LINE_AT_T0);
    const sampler = createStubPriceSampler(
      new Map<string, number | null>([["1823789", null]]),
    );
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 1_000 });
    const r = await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 90_000 });
    assertEqual(r.followupsCompleted, 0, "0 completed (sampler returned null)");
    assertEqual(r.followupsFailed, 3, "3 failed (price_unavailable)");
    const rows = readJsonl(path.join(dir, FOLLOWUPS_FILENAME));
    for (const row of rows) {
      assertEqual(
        row.samplerStatus,
        "price_unavailable",
        "samplerStatus=price_unavailable",
      );
      assertEqual(row.followupPrice, null, "no price");
      assertEqual(row.markout, null, "no markout");
    }
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("watcher source code does NOT contain submit/wallet/signer/privateKey verbs in executable positions", () => {
    const src = fs.readFileSync(
      path.resolve(
        __dirname,
        "..",
        "..",
        "scripts",
        "readiness",
        "watch-markout-followups.ts",
      ),
      "utf8",
    );
    // Strip JSDoc and line comments before scanning for executable references.
    const stripped = src
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/\/\/.*$/gm, "");
    assertTrue(
      !/\bsubmitOrder\b|\bplaceOrder\b|\bexecuteTrade\b|\brealSubmit\b|microLiveSubmit/.test(
        stripped,
      ),
      "no submit verbs",
    );
    assertTrue(
      !/sendTransaction|signTypedData|\bbroadcast\b/.test(stripped),
      "no transaction verbs",
    );
    assertTrue(!/\bprivateKey\b|\bMNEMONIC\b/.test(stripped), "no private key");
    assertTrue(
      !/\bwalletAddress\b|\bsignerAddress\b/.test(stripped),
      "no wallet/signer address",
    );
  });

  test("schemaVersion v1 is stamped on every emitted followup row", async () => {
    const dir = makeStateDir(POSITIVE_LINE_AT_T0);
    const sampler = createStubPriceSampler(
      new Map<string, number | null>([["1823789", 0.5]]),
    );
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 1_000 });
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 90_000 });
    const rows = readJsonl(path.join(dir, FOLLOWUPS_FILENAME));
    for (const row of rows) {
      assertEqual(row.schemaVersion, "v1", "schemaVersion v1");
      assertEqual(row.type, "markout_followup", "type=markout_followup");
    }
    fs.rmSync(dir, { recursive: true, force: true });
  });

  // -------------------------- Fallback chain tests --------------------------

  test("fallback: bid + ask present → bid_ask_mid", () => {
    const e = extractGammaPublicPriceWithFallback({ bestBid: 0.49, bestAsk: 0.51 });
    assertTrue(e !== null, "extracted");
    assertEqual(e!.priceSource, "bid_ask_mid", "source = bid_ask_mid");
    assertTrue(Math.abs(e!.price - 0.5) < 1e-9, "price = (0.49+0.51)/2");
    assertEqual(e!.rawPriceFieldsSeen.includes("bestBid"), true, "saw bestBid");
    assertEqual(e!.rawPriceFieldsSeen.includes("bestAsk"), true, "saw bestAsk");
  });

  test("fallback: only bestAsk valid → best_ask_only", () => {
    const e = extractGammaPublicPriceWithFallback({ bestAsk: 0.6 });
    assertTrue(e !== null, "extracted");
    assertEqual(e!.priceSource, "best_ask_only", "source = best_ask_only");
    assertEqual(e!.price, 0.6, "price = bestAsk");
  });

  test("fallback: only bestBid valid → best_bid_only", () => {
    const e = extractGammaPublicPriceWithFallback({ bestBid: 0.4 });
    assertTrue(e !== null, "extracted");
    assertEqual(e!.priceSource, "best_bid_only", "source = best_bid_only");
    assertEqual(e!.price, 0.4, "price = bestBid");
  });

  test("fallback: lastTradePrice falls through bid/ask", () => {
    const e = extractGammaPublicPriceWithFallback({ lastTradePrice: 0.55 });
    assertTrue(e !== null, "extracted");
    assertEqual(e!.priceSource, "last_trade_price", "source = last_trade_price");
    assertEqual(e!.price, 0.55, "price = lastTradePrice");
  });

  test("fallback: lastPrice when bid/ask/lastTradePrice absent", () => {
    const e = extractGammaPublicPriceWithFallback({ lastPrice: 0.62 });
    assertTrue(e !== null, "extracted");
    assertEqual(e!.priceSource, "last_price", "source = last_price");
    assertEqual(e!.price, 0.62, "price = lastPrice");
  });

  test("fallback: top-level price field", () => {
    const e = extractGammaPublicPriceWithFallback({ price: 0.42 });
    assertTrue(e !== null, "extracted");
    assertEqual(e!.priceSource, "price", "source = price");
    assertEqual(e!.price, 0.42, "price = price field");
  });

  test("fallback: outcomePrices array (with YES outcome)", () => {
    const e = extractGammaPublicPriceWithFallback({
      outcomes: ["No", "Yes"],
      outcomePrices: ["0.3", "0.7"],
    });
    assertTrue(e !== null, "extracted");
    assertEqual(e!.priceSource, "outcome_prices", "source = outcome_prices");
    assertEqual(e!.price, 0.7, "yes outcome picked");
  });

  test("fallback: outcomePrices as JSON-stringified array", () => {
    const e = extractGammaPublicPriceWithFallback({
      outcomes: '["Yes","No"]',
      outcomePrices: '["0.65","0.35"]',
    });
    assertTrue(e !== null, "extracted");
    assertEqual(e!.priceSource, "outcome_prices", "source");
    assertEqual(e!.price, 0.65, "first outcome (Yes) selected");
  });

  test("fallback: tokens with embedded price", () => {
    const e = extractGammaPublicPriceWithFallback({
      tokens: [
        { outcome: "Yes", price: 0.58 },
        { outcome: "No", price: 0.42 },
      ],
    });
    assertTrue(e !== null, "extracted");
    assertEqual(e!.priceSource, "tokens_outcomes_price", "source = tokens_outcomes_price");
    assertEqual(e!.price, 0.58, "first usable token price");
  });

  test("fallback: nothing usable → null", () => {
    const e = extractGammaPublicPriceWithFallback({});
    assertEqual(e, null, "no extraction");
    const e2 = extractGammaPublicPriceWithFallback({ bestBid: 0, bestAsk: 0 });
    assertEqual(e2, null, "zero bid/ask rejected");
    const e3 = extractGammaPublicPriceWithFallback({ price: 1.5 });
    assertEqual(e3, null, "out-of-range price rejected");
  });

  test("watcher emits priceSource + rawPriceFieldsSeen on JSONL rows", async () => {
    const dir = makeStateDir(POSITIVE_LINE_AT_T0);
    const sampler = createStubPriceSampler(new Map(), {
      gammaPayloads: new Map<string, Record<string, unknown>>([
        [
          "1823789",
          { bestBid: 0.49, bestAsk: 0.51, lastTradePrice: 0.5 },
        ],
      ]),
    });
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 1_000 });
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 90_000 });
    const rows = readJsonl(path.join(dir, FOLLOWUPS_FILENAME));
    assertEqual(rows.length, 3, "3 rows");
    for (const row of rows) {
      assertEqual(row.priceSource, "bid_ask_mid", "priceSource on row");
      assertTrue(
        Array.isArray(row.rawPriceFieldsSeen),
        "rawPriceFieldsSeen is array",
      );
      const seen = row.rawPriceFieldsSeen as string[];
      assertEqual(seen.includes("bestBid"), true, "saw bestBid");
      assertEqual(seen.includes("bestAsk"), true, "saw bestAsk");
    }
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("watcher records price_unavailable when payload has no usable price", async () => {
    const dir = makeStateDir(POSITIVE_LINE_AT_T0);
    const sampler = createStubPriceSampler(new Map(), {
      gammaPayloads: new Map<string, Record<string, unknown>>([
        ["1823789", { description: "no price fields here" }],
      ]),
    });
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 1_000 });
    await runOnce({ paperStateDir: dir, sampler, nowMs: T0 + 90_000 });
    const rows = readJsonl(path.join(dir, FOLLOWUPS_FILENAME));
    for (const row of rows) {
      assertEqual(
        row.samplerStatus,
        "price_unavailable",
        "no usable price → unavailable",
      );
      assertEqual(
        row.errorMessage,
        "no_usable_public_price",
        "errorMessage set",
      );
    }
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("safety: source code does NOT contain submit/order/wallet/signer/privateKey/mnemonic/seed phrase", () => {
    const src = fs.readFileSync(
      path.resolve(
        __dirname,
        "..",
        "..",
        "scripts",
        "readiness",
        "watch-markout-followups.ts",
      ),
      "utf8",
    );
    const stripped = src
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/\/\/.*$/gm, "");
    assertTrue(
      !/\bsubmitOrder\b|\bplaceOrder\b|\bexecuteTrade\b|\brealSubmit\b|microLiveSubmit/.test(
        stripped,
      ),
      "no submit verbs",
    );
    assertTrue(
      !/sendTransaction|signTypedData|\bbroadcast\b/.test(stripped),
      "no transaction verbs",
    );
    assertTrue(!/\bprivateKey\b|\bMNEMONIC\b/i.test(stripped), "no private key / mnemonic");
    assertTrue(!/seed\s*phrase|seedPhrase/i.test(stripped), "no seed phrase");
    assertTrue(
      !/\bwalletAddress\b|\bsignerAddress\b/.test(stripped),
      "no wallet/signer address",
    );
  });
});
