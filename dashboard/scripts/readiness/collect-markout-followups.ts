/**
 * Microcapital Readiness Gate — paper-only markout follow-up collector.
 *
 * Reads PAPER_STATE_DIR/cross-venue-anchor-1823789-paper-execution-history.jsonl
 * and joins each `paper_cycle_positive` assessment with later same-market
 * assessments to derive paper-only markout follow-ups at +5s / +30s / +60s.
 *
 * Writes:
 *   PAPER_STATE_DIR/cross-venue-anchor-1823789-markout-followups.jsonl
 *
 * Each row is either a {@link MarkoutFollowupRecord} (when a later snapshot
 * is found within the configured tolerance) or a "historical_followup_unavailable"
 * placeholder when no later snapshot is close enough AND the assessment is
 * older than `MARKOUT_HISTORICAL_AGE_MS` (i.e. there is no chance of a future
 * snapshot ever falling inside the window).
 *
 * GUARDRAILS — this script:
 *   - reads ONLY the existing audit JSONL (already paper/shadow only)
 *   - never connects to a market endpoint
 *   - never submits, signs, or touches a wallet/signer/private key
 *   - is idempotent: existing followup keys (cycleId+horizonMs) are not duplicated
 *
 * Policy: shadow_only_no_api_submit_v1.
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
import {
  buildMarkoutFollowupRecord,
  type MarkoutFollowupRecord,
  type MarkoutHorizonMs,
} from "../../lib/readiness/markoutFollowupProducer";
import { MICROCAPITAL_READINESS_THRESHOLDS } from "../../lib/readiness/microcapitalReadinessThresholds";
import { buildPaperCycleId } from "../../lib/readiness/paperCycleLifecycle";

interface HistoricalUnavailableRow {
  type: "markout_followup";
  cycleId: string;
  marketId: string | null;
  baseObservedAt: number;
  horizonMs: MarkoutHorizonMs;
  source: "paper_shadow_readonly";
  markoutStatus: "historical_followup_unavailable";
}

function epoch(iso: string | null): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : null;
}

function isPositive(a: PaperExecutionAssessment): boolean {
  return (a.paperExecutionVerdict ?? "").toLowerCase() === "paper_cycle_positive";
}

function basePriceOf(a: PaperExecutionAssessment): number | null {
  // Prefer mid; fall back to the average of bid/ask; null if neither is present.
  if (a.clobMid !== null) return a.clobMid;
  if (a.clobBestBid !== null && a.clobBestAsk !== null) {
    return (a.clobBestBid + a.clobBestAsk) / 2;
  }
  return null;
}

function followupPriceOf(a: PaperExecutionAssessment): number | null {
  return basePriceOf(a);
}

interface ExistingKey {
  cycleId: string;
  horizonMs: MarkoutHorizonMs;
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
    if (t.length === 0) continue;
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

function appendJsonl(filePath: string, rows: ReadonlyArray<object>): void {
  if (rows.length === 0) return;
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const lines = rows.map(r => JSON.stringify(r)).join("\n");
  fs.appendFileSync(filePath, lines + "\n", "utf8");
}

interface CollectorResult {
  positiveAssessmentCount: number;
  followupRowsWritten: number;
  historicalUnavailableRowsWritten: number;
  alreadyPresentRowsSkipped: number;
  pendingNewerWindowSkipped: number;
}

export function collectMarkoutFollowups(opts: {
  paperStateDir?: string;
  nowMs?: number;
} = {}): CollectorResult {
  const stateDir = opts.paperStateDir ?? resolvePaperStateDir();
  const nowMs = typeof opts.nowMs === "number" ? opts.nowMs : Date.now();

  const historyPath = path.join(
    stateDir,
    PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789PaperExecutionHistory,
  );
  const followupPath = path.join(
    stateDir,
    MICROCAPITAL_READINESS_THRESHOLDS.MARKOUT_FOLLOWUPS_FILENAME,
  );

  const assessments = readPaperExecutionAssessmentsFromJsonl(historyPath);
  const existing = readExistingFollowupKeys(followupPath);

  const positives = assessments.filter(isPositive);
  const T = MICROCAPITAL_READINESS_THRESHOLDS;
  const horizons = T.MARKOUT_HORIZONS_MS;

  const newRows: object[] = [];
  let historicalUnavailable = 0;
  let alreadyPresent = 0;
  let pendingNewer = 0;

  for (const base of positives) {
    const baseTs = epoch(base.isoTimestamp);
    const basePrice = basePriceOf(base);
    if (baseTs === null || basePrice === null) continue;
    const cycleId = buildPaperCycleId(base);

    const sameMarketLater = assessments.filter(a => {
      if (a.marketId !== base.marketId) return false;
      const ts = epoch(a.isoTimestamp);
      return ts !== null && ts > baseTs;
    });

    for (const horizon of horizons) {
      const key = `${cycleId}::${horizon}`;
      if (existing.has(key)) {
        alreadyPresent++;
        continue;
      }
      const targetTs = baseTs + horizon;
      const tolerance =
        T.MARKOUT_JOIN_TOLERANCE_MS[horizon as keyof typeof T.MARKOUT_JOIN_TOLERANCE_MS];

      // Find the closest later assessment within tolerance of the target.
      let closest: PaperExecutionAssessment | null = null;
      let closestDelta = Number.POSITIVE_INFINITY;
      for (const candidate of sameMarketLater) {
        const ts = epoch(candidate.isoTimestamp);
        if (ts === null) continue;
        const delta = Math.abs(ts - targetTs);
        if (delta <= tolerance && delta < closestDelta) {
          closest = candidate;
          closestDelta = delta;
        }
      }

      if (closest !== null) {
        const followupTs = epoch(closest.isoTimestamp);
        const followupPrice = followupPriceOf(closest);
        if (followupTs !== null && followupPrice !== null) {
          const record: MarkoutFollowupRecord = buildMarkoutFollowupRecord(
            {
              cycleId,
              marketId: base.marketId ?? "unknown",
              signalTimestamp: baseTs,
              basePrice,
            },
            {
              cycleId,
              horizonMs: horizon as MarkoutHorizonMs,
              followupTimestamp: followupTs,
              followupPrice,
            },
          );
          newRows.push(record);
          existing.add(key);
          continue;
        }
      }

      // No close-enough later assessment. Decide between
      // historical_followup_unavailable (no chance of future hit) and skip-now.
      const ageOfWindow = nowMs - (baseTs + horizon);
      if (ageOfWindow > T.MARKOUT_HISTORICAL_AGE_MS) {
        const placeholder: HistoricalUnavailableRow = {
          type: "markout_followup",
          cycleId,
          marketId: base.marketId ?? null,
          baseObservedAt: baseTs,
          horizonMs: horizon as MarkoutHorizonMs,
          source: "paper_shadow_readonly",
          markoutStatus: "historical_followup_unavailable",
        };
        newRows.push(placeholder);
        existing.add(key);
        historicalUnavailable++;
      } else {
        pendingNewer++;
      }
    }
  }

  // Tally followup vs historical-unavailable rows.
  const followupRowsWritten = newRows.filter(
    r => !("markoutStatus" in (r as Record<string, unknown>)),
  ).length;

  appendJsonl(followupPath, newRows);

  return {
    positiveAssessmentCount: positives.length,
    followupRowsWritten,
    historicalUnavailableRowsWritten: historicalUnavailable,
    alreadyPresentRowsSkipped: alreadyPresent,
    pendingNewerWindowSkipped: pendingNewer,
  };
}

function main(): void {
  const result = collectMarkoutFollowups();
  process.stdout.write(JSON.stringify(result, null, 2));
  process.stdout.write("\n");
}

if (require.main === module) {
  main();
}
