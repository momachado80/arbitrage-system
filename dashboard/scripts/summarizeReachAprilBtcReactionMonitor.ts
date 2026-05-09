/**
 * Resume histórico JSONL do monitor BTC narrow (uma linha por mercado por ciclo).
 *
 * Uso:
 *   npx ts-node -P tsconfig.worker.json scripts/summarizeReachAprilBtcReactionMonitor.ts
 * Opcional: REACH_APRIL_BTC_MONITOR_HISTORY_PATH=<ficheiro.jsonl>
 */

import fs from "fs";
import path from "path";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES } from "../lib/paperStateDir";

export type FinalBtcReactionVerdict =
  | "consistently_negative_close_reaction_path"
  | "episodic_but_not_promotable"
  | "borderline_candidate_keep_watchlist"
  | "repeatably_near_zero_or_positive_candidate_survives";

interface Row {
  isoTimestamp?: string;
  marketId?: string;
  marketTitle?: string;
  triggerCount?: number;
  avgLagObserved?: number;
  medianNetPerReactionCycle?: number;
  worstNetPerReactionCycle?: number;
  calibratedReactionVerdictPerMarket?: string;
  supportingNote?: string;
}

interface MarketAgg {
  marketId: string;
  marketTitle: string;
  totalObservations: number;
  totalTriggers: number;
  meanMedianNet: number;
  medianMedianNet: number;
  minMedianNet: number;
  maxMedianNet: number;
  pctAboveZero: number;
  pctAboveMinus0_003: number;
  strongestWindow: string | null;
  weakestWindow: string | null;
  finalBtcReactionVerdict: FinalBtcReactionVerdict;
  finalBtcReactionSummaryLine: string;
}

function medianSorted(xs: number[]): number {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function mean(xs: number[]): number {
  if (xs.length === 0) return NaN;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function verdictForMarket(
  totalObservations: number,
  medianNets: number[],
  triggersPerObs: number[],
): FinalBtcReactionVerdict {
  const meanMed = mean(medianNets);
  const pctAboveZero = medianNets.filter(x => x > 0).length / Math.max(1, medianNets.length);
  const pctAboveM003 = medianNets.filter(x => x > -0.003).length / Math.max(1, medianNets.length);
  const totalTrig = triggersPerObs.reduce((a, b) => a + b, 0);

  if (totalObservations < 3) {
    return "episodic_but_not_promotable";
  }

  if (pctAboveZero >= 0.15 || (meanMed > -0.002 && pctAboveM003 >= 0.55)) {
    return "repeatably_near_zero_or_positive_candidate_survives";
  }

  if (meanMed < -0.008 && pctAboveM003 < 0.15 && totalTrig > 0) {
    return "consistently_negative_close_reaction_path";
  }

  if (meanMed >= -0.006 && meanMed <= -0.002 && pctAboveM003 >= 0.22) {
    return "borderline_candidate_keep_watchlist";
  }

  return "episodic_but_not_promotable";
}

function summarizeHistoryFile(filePath: string): Record<string, MarketAgg> {
  const raw = fs.readFileSync(filePath, "utf8");
  const lines = raw.split("\n").filter(Boolean);

  const byMarket = new Map<
    string,
    {
      title: string;
      rows: { iso: string; median: number; triggers: number }[];
    }
  >();

  for (const line of lines) {
    let j: Row;
    try {
      j = JSON.parse(line) as Row;
    } catch {
      continue;
    }
    const mid = String(j.marketId ?? "");
    if (!mid) continue;
    const med = typeof j.medianNetPerReactionCycle === "number" ? j.medianNetPerReactionCycle : 0;
    const tc = typeof j.triggerCount === "number" ? j.triggerCount : 0;
    const iso = String(j.isoTimestamp ?? "");
    const title = String(j.marketTitle ?? "");

    const cur = byMarket.get(mid) ?? { title, rows: [] };
    if (title && !cur.title) cur.title = title;
    cur.rows.push({ iso, median: med, triggers: tc });
    byMarket.set(mid, cur);
  }

  const out: Record<string, MarketAgg> = {};

  Array.from(byMarket.entries()).forEach(([marketId, pack]) => {
    const medianNets = pack.rows.map((r: { iso: string; median: number; triggers: number }) => r.median);
    const triggersObs = pack.rows.map((r: { iso: string; median: number; triggers: number }) => r.triggers);

    let maxM = -Infinity;
    let minM = Infinity;
    let strongestIso: string | null = null;
    let weakestIso: string | null = null;
    for (const r of pack.rows) {
      if (r.median > maxM) {
        maxM = r.median;
        strongestIso = r.iso || null;
      }
      if (r.median < minM) {
        minM = r.median;
        weakestIso = r.iso || null;
      }
    }
    if (!Number.isFinite(maxM)) maxM = 0;
    if (!Number.isFinite(minM)) minM = 0;

    const pctAboveZero = medianNets.filter((x: number) => x > 0).length / Math.max(1, medianNets.length);
    const pctAboveM003 = medianNets.filter((x: number) => x > -0.003).length / Math.max(1, medianNets.length);

    const vv = verdictForMarket(pack.rows.length, medianNets, triggersObs);

    const agg: MarketAgg = {
      marketId,
      marketTitle: pack.title || marketId,
      totalObservations: pack.rows.length,
      totalTriggers: triggersObs.reduce((a, b) => a + b, 0),
      meanMedianNet: mean(medianNets),
      medianMedianNet: medianSorted(medianNets),
      minMedianNet: medianNets.length ? minM : 0,
      maxMedianNet: medianNets.length ? maxM : 0,
      pctAboveZero: Math.round(pctAboveZero * 1000) / 1000,
      pctAboveMinus0_003: Math.round(pctAboveM003 * 1000) / 1000,
      strongestWindow: strongestIso,
      weakestWindow: weakestIso,
      finalBtcReactionVerdict: vv,
      finalBtcReactionSummaryLine: `btc_narrow_watch market=${marketId} n=${pack.rows.length} verdict=${vv} mean_med_net=${mean(medianNets).toFixed(6)} pct_gt_m003=${(pctAboveM003 * 100).toFixed(1)}%`,
    };

    out[marketId] = agg;
  });

  return out;
}

function resolveHistoryPath(): string {
  const raw = process.env.REACH_APRIL_BTC_MONITOR_HISTORY_PATH?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.reachAprilBtcReactionMonitorHistory);
}

function main(): void {
  const fp = resolveHistoryPath();
  if (!fs.existsSync(fp)) {
    console.error(JSON.stringify({ error: "history_file_missing", path: fp }));
    process.exit(2);
  }
  const summary = summarizeHistoryFile(fp);
  console.log(JSON.stringify({ historyPath: fp, probe: "reach-april-btc-narrow-monitor-v1", perMarket: summary }, null, 2));
}

main();
