/**
 * Agrega o histórico JSONL da validação 31552 (stressAdjustedNetConversionEdge ao longo do tempo).
 *
 * Uso: npm run final-neg-risk-31552:summarize
 *       npm run final-neg-risk-31552:summarize -- --history=/data/final-negative-risk-31552-history.jsonl
 */

import fs from "fs";
import path from "path";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES } from "../lib/paperStateDir";

export type Final3DayVerdict =
  | "consistently_negative_close_path"
  | "episodic_but_not_promotable"
  | "borderline_candidate_keep_watchlist"
  | "repeatably_positive_candidate_survives";

interface Row {
  isoTimestamp: string;
  eventId?: string;
  stressAdjustedNetConversionEdge: number;
}

function parseArgs(argv: string[]): { historyPath: string } {
  let historyPath = defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.finalNegativeRisk31552History);
  for (const a of argv) {
    if (a.startsWith("--history=")) {
      historyPath = path.resolve(a.slice("--history=".length).trim());
    }
  }
  return { historyPath };
}

function medianSorted(sorted: number[]): number {
  if (sorted.length === 0) return NaN;
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function mean(nums: number[]): number {
  if (nums.length === 0) return NaN;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function dayKey(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "invalid_date";
  return d.toISOString().slice(0, 10);
}

function classifyFinal3Day(
  n: number,
  meanNet: number,
  medianNet: number,
  minNet: number,
  maxNet: number,
  pctPositive: number,
  pctAboveMinus0_005: number,
  weakestMean: number | null,
): Final3DayVerdict {
  if (n === 0) {
    return "borderline_candidate_keep_watchlist";
  }
  if (n < 12) {
    return "borderline_candidate_keep_watchlist";
  }

  const w = weakestMean != null && Number.isFinite(weakestMean) ? weakestMean : minNet;

  if (
    meanNet <= -0.002 &&
    medianNet <= -0.001 &&
    pctPositive <= 0.22 &&
    pctAboveMinus0_005 <= 0.45
  ) {
    return "consistently_negative_close_path";
  }

  if (
    meanNet >= 0.0008 &&
    medianNet >= 0.0005 &&
    pctPositive >= 0.42 &&
    w >= -0.004
  ) {
    return "repeatably_positive_candidate_survives";
  }

  if (Math.abs(meanNet) <= 0.0014 && pctAboveMinus0_005 >= 0.58) {
    return "borderline_candidate_keep_watchlist";
  }

  if (pctPositive >= 0.38 && meanNet < 0.0004) {
    return "episodic_but_not_promotable";
  }

  if (meanNet > -0.002 && meanNet < 0.0008 && pctPositive >= 0.25 && pctPositive <= 0.55) {
    return "borderline_candidate_keep_watchlist";
  }

  return "episodic_but_not_promotable";
}

function windowLabel(day: string, m: number): string {
  return `${day} UTC (mean=${m.toFixed(6)})`;
}

function main(): void {
  const { historyPath: fp } = parseArgs(process.argv.slice(2));
  if (!fs.existsSync(fp)) {
    console.error(`[summarize-31552] missing history file: ${fp}`);
    process.exit(1);
    return;
  }

  const text = fs.readFileSync(fp, "utf8");
  const lines = text
    .split("\n")
    .map(l => l.trim())
    .filter(Boolean);
  const rows: Row[] = [];
  for (const line of lines) {
    try {
      const j = JSON.parse(line) as Row;
      if (typeof j.stressAdjustedNetConversionEdge !== "number") continue;
      rows.push(j);
    } catch {
      continue;
    }
  }

  const nets = rows.map(r => r.stressAdjustedNetConversionEdge);
  const n = nets.length;
  const sorted = [...nets].sort((a, b) => a - b);
  const meanNet = mean(nets);
  const medianNet = medianSorted(sorted);
  const minNet = n ? sorted[0] : NaN;
  const maxNet = n ? sorted[sorted.length - 1] : NaN;
  const pctPositive = n ? nets.filter(x => x > 0).length / n : 0;
  const pctAboveMinus0_005 = n ? nets.filter(x => x > -0.005).length / n : 0;

  const byDay = new Map<string, number[]>();
  for (const r of rows) {
    const k = dayKey(r.isoTimestamp);
    if (k === "invalid_date") continue;
    const arr = byDay.get(k) ?? [];
    arr.push(r.stressAdjustedNetConversionEdge);
    byDay.set(k, arr);
  }

  let strongestWindow = "";
  let weakestWindow = "";
  let weakestMean: number | null = null;
  if (byDay.size > 0) {
    let bestDay = "";
    let bestM = -Infinity;
    let worstDay = "";
    let worstM = Infinity;
    for (const [day, vals] of Array.from(byDay.entries())) {
      const m = mean(vals);
      if (m > bestM || (m === bestM && day > bestDay)) {
        bestM = m;
        bestDay = day;
      }
      if (m < worstM || (m === worstM && day < worstDay)) {
        worstM = m;
        worstDay = day;
      }
    }
    strongestWindow = windowLabel(bestDay, bestM);
    weakestWindow = windowLabel(worstDay, worstM);
    weakestMean = worstM;
  }

  const final3DayVerdict = classifyFinal3Day(
    n,
    meanNet,
    medianNet,
    minNet,
    maxNet,
    pctPositive,
    pctAboveMinus0_005,
    weakestMean,
  );

  const final3DaySummaryLine = [
    `final3Day_31552 n=${n}`,
    `verdict=${final3DayVerdict}`,
    `mean=${meanNet.toFixed(6)}`,
    `median=${medianNet.toFixed(6)}`,
    `min=${minNet.toFixed(6)}`,
    `max=${maxNet.toFixed(6)}`,
    `pctPos=${(pctPositive * 100).toFixed(1)}%`,
    `pctAbove-0.005=${(pctAboveMinus0_005 * 100).toFixed(1)}%`,
    `strongest=${strongestWindow || "n/a"}`,
    `weakest=${weakestWindow || "n/a"}`,
  ].join(" | ");

  const out = {
    historyFile: fp,
    totalObservations: n,
    meanStressAdjustedNet: meanNet,
    medianStressAdjustedNet: medianNet,
    minStressAdjustedNet: minNet,
    maxStressAdjustedNet: maxNet,
    pctPositive,
    pctAboveMinus0_005,
    strongestWindow: strongestWindow || "n/a",
    weakestWindow: weakestWindow || "n/a",
    final3DayVerdict,
    final3DaySummaryLine,
  };

  console.log(JSON.stringify(out, null, 2));
}

main();
