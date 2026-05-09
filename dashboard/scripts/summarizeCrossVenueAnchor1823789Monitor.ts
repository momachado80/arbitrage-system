/**
 * Resumo JSONL do monitor cross-venue anchor refinado — mercado 1823789 apenas.
 *
 * Uso (após 2–3 dias de histórico no volume Railway ou cópia local):
 *   npx ts-node -P tsconfig.worker.json scripts/summarizeCrossVenueAnchor1823789Monitor.ts
 * Opcional: `CROSS_VENUE_ANCHOR_1823789_MONITOR_HISTORY_PATH=<ficheiro.jsonl>`
 */

import fs from "fs";
import path from "path";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES } from "../lib/paperStateDir";

export type FinalAnchor1823789Verdict =
  | "consistently_negative_close_anchor_path"
  | "episodic_but_not_promotable"
  | "borderline_candidate_keep_watchlist"
  | "repeatably_positive_candidate_survives";

interface Row {
  isoTimestamp?: string;
  marketId?: string;
  estimatedNetAnchorCycle?: number;
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

/**
 * Distribuição temporal do net refinado: promoção vs ruído episódico.
 * Thresholds alinhados à escala típica do piloto refinado (frações 0–1).
 */
function verdictForNetSeries(nets: number[]): FinalAnchor1823789Verdict {
  const n = nets.length;
  if (n < 3) return "episodic_but_not_promotable";

  const m = mean(nets);
  const pctPositive = nets.filter(x => x > 0).length / n;
  const pct005 = nets.filter(x => x > 0.005).length / n;

  if (m < -0.008 && pctPositive < 0.15) {
    return "consistently_negative_close_anchor_path";
  }

  if ((m >= 0.004 && pctPositive >= 0.5 && pct005 >= 0.33) || (m >= 0.0025 && pctPositive >= 0.62)) {
    return "repeatably_positive_candidate_survives";
  }

  if (m >= -0.006 && m <= 0.012 && pctPositive >= 0.22 && pctPositive < 0.52 && pct005 < 0.35) {
    return "borderline_candidate_keep_watchlist";
  }

  return "episodic_but_not_promotable";
}

function summarizeHistoryFile(filePath: string): {
  totalObservations: number;
  meanEstimatedNetAnchorCycle: number;
  medianEstimatedNetAnchorCycle: number;
  minEstimatedNetAnchorCycle: number;
  maxEstimatedNetAnchorCycle: number;
  pctPositive: number;
  pctAbove0_005: number;
  strongestWindow: string | null;
  weakestWindow: string | null;
  finalAnchor1823789Verdict: FinalAnchor1823789Verdict;
  finalAnchor1823789SummaryLine: string;
} {
  const raw = fs.readFileSync(filePath, "utf8");
  const lines = raw.split("\n").filter(Boolean);

  const nets: number[] = [];
  const rows: { iso: string; net: number }[] = [];

  for (const line of lines) {
    let j: Row;
    try {
      j = JSON.parse(line) as Row;
    } catch {
      continue;
    }
    if (String(j.marketId ?? "") !== "1823789") continue;
    const net = typeof j.estimatedNetAnchorCycle === "number" ? j.estimatedNetAnchorCycle : NaN;
    if (!Number.isFinite(net)) continue;
    const iso = String(j.isoTimestamp ?? "");
    nets.push(net);
    rows.push({ iso, net });
  }

  if (nets.length === 0) {
    return {
      totalObservations: 0,
      meanEstimatedNetAnchorCycle: NaN,
      medianEstimatedNetAnchorCycle: NaN,
      minEstimatedNetAnchorCycle: NaN,
      maxEstimatedNetAnchorCycle: NaN,
      pctPositive: 0,
      pctAbove0_005: 0,
      strongestWindow: null,
      weakestWindow: null,
      finalAnchor1823789Verdict: "episodic_but_not_promotable",
      finalAnchor1823789SummaryLine:
        "anchor_1823789_watch n=0 verdict=episodic_but_not_promotable (no valid rows)",
    };
  }

  let maxN = -Infinity;
  let minN = Infinity;
  let strongestIso: string | null = null;
  let weakestIso: string | null = null;
  for (const r of rows) {
    if (r.net > maxN) {
      maxN = r.net;
      strongestIso = r.iso || null;
    }
    if (r.net < minN) {
      minN = r.net;
      weakestIso = r.iso || null;
    }
  }

  const pctPositive = nets.filter(x => x > 0).length / nets.length;
  const pctAbove0_005 = nets.filter(x => x > 0.005).length / nets.length;
  const vv = verdictForNetSeries(nets);
  const meanNet = mean(nets);
  const medNet = medianSorted(nets);

  const finalAnchor1823789SummaryLine = `anchor_1823789_watch n=${nets.length} verdict=${vv} mean_net=${meanNet.toFixed(6)} median_net=${medNet.toFixed(6)} pct_pos=${(pctPositive * 100).toFixed(1)}% pct_gt_0.005=${(pctAbove0_005 * 100).toFixed(1)}%`;

  return {
    totalObservations: nets.length,
    meanEstimatedNetAnchorCycle: Math.round(meanNet * 1_000_000) / 1_000_000,
    medianEstimatedNetAnchorCycle: Math.round(medNet * 1_000_000) / 1_000_000,
    minEstimatedNetAnchorCycle: Math.round(minN * 1_000_000) / 1_000_000,
    maxEstimatedNetAnchorCycle: Math.round(maxN * 1_000_000) / 1_000_000,
    pctPositive: Math.round(pctPositive * 1000) / 1000,
    pctAbove0_005: Math.round(pctAbove0_005 * 1000) / 1000,
    strongestWindow: strongestIso,
    weakestWindow: weakestIso,
    finalAnchor1823789Verdict: vv,
    finalAnchor1823789SummaryLine,
  };
}

function resolveHistoryPath(): string {
  const raw = process.env.CROSS_VENUE_ANCHOR_1823789_MONITOR_HISTORY_PATH?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789MonitorHistory);
}

function main(): void {
  const fp = resolveHistoryPath();
  if (!fs.existsSync(fp)) {
    console.error(JSON.stringify({ error: "history_file_missing", path: fp }));
    process.exit(2);
  }
  const summary = summarizeHistoryFile(fp);
  console.log(
    JSON.stringify(
      {
        historyPath: fp,
        probe: "cross-venue-anchor-1823789-monitor-v1",
        marketId: "1823789",
        ...summary,
      },
      null,
      2,
    ),
  );
}

main();
