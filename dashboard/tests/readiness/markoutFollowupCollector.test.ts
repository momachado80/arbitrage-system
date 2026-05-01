/**
 * Tests — markout follow-up collector script (paper-only, read-only).
 *
 * Paper/shadow only. NO market client touched.
 * Policy: shadow_only_no_api_submit_v1.
 */

import fs from "fs";
import os from "os";
import path from "path";

import { collectMarkoutFollowups } from "../../scripts/readiness/collect-markout-followups";

import { assertEqual, assertTrue, describe, test } from "./_assert";

const HISTORY_FILENAME = "cross-venue-anchor-1823789-paper-execution-history.jsonl";
const FOLLOWUPS_FILENAME = "cross-venue-anchor-1823789-markout-followups.jsonl";

function makeStateDir(jsonl: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "markout-collector-"));
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

const T0 = Date.parse("2026-04-29T12:00:00.000Z");

function asIso(ms: number): string {
  return new Date(ms).toISOString();
}

describe("tests/readiness/markoutFollowupCollector.test.ts", () => {
  test("with later same-market snapshots near +5s/+30s/+60s, emits markout_followup rows", () => {
    const lines = [
      JSON.stringify({
        isoTimestamp: asIso(T0),
        marketId: "1823789",
        paperExecutionVerdict: "paper_cycle_positive",
        clobMid: 0.5,
        clobBestBid: 0.49,
        clobBestAsk: 0.51,
      }),
      JSON.stringify({
        isoTimestamp: asIso(T0 + 5_000),
        marketId: "1823789",
        paperExecutionVerdict: "blocked_by_gate",
        clobMid: 0.51,
        clobBestBid: 0.5,
        clobBestAsk: 0.52,
      }),
      JSON.stringify({
        isoTimestamp: asIso(T0 + 30_000),
        marketId: "1823789",
        paperExecutionVerdict: "blocked_by_gate",
        clobMid: 0.52,
      }),
      JSON.stringify({
        isoTimestamp: asIso(T0 + 60_000),
        marketId: "1823789",
        paperExecutionVerdict: "blocked_by_gate",
        clobMid: 0.515,
      }),
    ].join("\n");

    const stateDir = makeStateDir(lines);
    const result = collectMarkoutFollowups({
      paperStateDir: stateDir,
      nowMs: T0 + 200_000,
    });
    assertEqual(result.positiveAssessmentCount, 1, "1 positive");
    assertEqual(result.followupRowsWritten, 3, "3 followups written");

    const rows = readJsonl(path.join(stateDir, FOLLOWUPS_FILENAME));
    assertEqual(rows.length, 3, "3 rows on disk");
    const horizons = rows.map(r => r.horizonMs).sort((a, b) => Number(a) - Number(b));
    assertEqual(horizons, [5_000, 30_000, 60_000], "all 3 horizons present");
    fs.rmSync(stateDir, { recursive: true, force: true });
  });

  test("if no later snapshot is close enough AND assessment is older than threshold → historical_followup_unavailable", () => {
    const lines = [
      JSON.stringify({
        isoTimestamp: asIso(T0),
        marketId: "1823789",
        paperExecutionVerdict: "paper_cycle_positive",
        clobMid: 0.5,
      }),
    ].join("\n");
    const stateDir = makeStateDir(lines);
    const result = collectMarkoutFollowups({
      paperStateDir: stateDir,
      nowMs: T0 + 1_000_000, // far in the future
    });
    assertEqual(result.followupRowsWritten, 0, "no followup rows");
    assertEqual(
      result.historicalUnavailableRowsWritten,
      3,
      "3 historical_followup_unavailable rows (5s+30s+60s)",
    );
    const rows = readJsonl(path.join(stateDir, FOLLOWUPS_FILENAME));
    assertTrue(
      rows.every(r => r.markoutStatus === "historical_followup_unavailable"),
      "all rows marked historical",
    );
    fs.rmSync(stateDir, { recursive: true, force: true });
  });

  test("idempotent: re-running does not duplicate followup keys", () => {
    const lines = [
      JSON.stringify({
        isoTimestamp: asIso(T0),
        marketId: "1823789",
        paperExecutionVerdict: "paper_cycle_positive",
        clobMid: 0.5,
      }),
      JSON.stringify({
        isoTimestamp: asIso(T0 + 5_000),
        marketId: "1823789",
        paperExecutionVerdict: "blocked_by_gate",
        clobMid: 0.51,
      }),
    ].join("\n");
    const stateDir = makeStateDir(lines);
    const r1 = collectMarkoutFollowups({ paperStateDir: stateDir, nowMs: T0 + 1_000_000 });
    const r2 = collectMarkoutFollowups({ paperStateDir: stateDir, nowMs: T0 + 1_000_000 });
    const rowsAfterTwo = readJsonl(path.join(stateDir, FOLLOWUPS_FILENAME));
    assertEqual(r2.alreadyPresentRowsSkipped, 3, "2nd run skips all 3 keys");
    assertEqual(rowsAfterTwo.length, 3, "no duplication");
    fs.rmSync(stateDir, { recursive: true, force: true });
  });

  test("does NOT touch market client (script reads only existing JSONL)", () => {
    // Smoke test: parse the script source and verify there is no `fetch(`,
    // `clob`, `gamma`, `submit`, `wallet`, `signer`, `privateKey` invocation.
    const src = fs.readFileSync(
      path.resolve(__dirname, "..", "..", "scripts", "readiness", "collect-markout-followups.ts"),
      "utf8",
    );
    // Comments are allowed to mention these terms; check that there is no
    // fetch(...) call in executable code by stripping JSDoc and line comments.
    const stripped = src
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/\/\/.*$/gm, "");
    assertTrue(
      !/\bfetch\s*\(/.test(stripped),
      "no fetch() call in collector",
    );
    assertTrue(
      !/sendTransaction|signTypedData|broadcast|placeOrder|submitOrder/.test(stripped),
      "no submit-like verbs in collector",
    );
  });
});
