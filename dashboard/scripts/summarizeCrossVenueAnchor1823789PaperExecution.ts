/**
 * Resumo JSONL — paper execution narrow 1823789.
 *
 * Uso após 24h / 72h no volume Railway (ou cópia local):
 *   npx ts-node -P tsconfig.worker.json scripts/summarizeCrossVenueAnchor1823789PaperExecution.ts
 * Opcional: `CROSS_VENUE_ANCHOR_1823789_PAPER_EXECUTION_HISTORY_PATH=<ficheiro.jsonl>`
 */

import fs from "fs";
import path from "path";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES } from "../lib/paperStateDir";
import {
  ACTIVE_SINGLE_TRACK_EXPERIMENT_ID,
  getStrategyTrackClassification,
} from "../lib/strategyTrackPolicy";

export type FinalPaperExecutionVerdict =
  | "blocked_by_gate"
  | "paper_execution_negative"
  | "borderline_paper_candidate"
  | "positive_paper_candidate";

/** Disponibilidade de ciclos “executáveis” em papel (gates abertos), independentemente do net. */
export type PaperExecutionAvailabilityVerdict =
  | "low_availability"
  | "moderate_availability"
  | "high_availability";

export type ContinuationDecision =
  | "continue_controlled_paper"
  | "pause_and_review"
  | "stop_track";

interface Row {
  probeVersion?: string;
  isoTimestamp?: string;
  marketId?: string;
  narrowValidationVerdict?: string;
  executionGateVerdict?: string;
  paperExecutionVerdict?: string;
  estimatedNetAfterPaperExecution?: number | null;
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

function finalVerdict(params: {
  totalPaperExecutedCycles: number;
  totalBlockedCycles: number;
  nets: number[];
}): FinalPaperExecutionVerdict {
  const { totalPaperExecutedCycles, totalBlockedCycles, nets } = params;
  if (totalPaperExecutedCycles === 0) {
    if (totalBlockedCycles > 0) return "blocked_by_gate";
    return "blocked_by_gate";
  }
  if (nets.length < 3) return "borderline_paper_candidate";

  const m = mean(nets);
  const pctPositive = nets.filter(x => x > 0).length / nets.length;

  if (m < -0.003 || pctPositive < 0.28) {
    return "paper_execution_negative";
  }
  if (m >= 0.004 && pctPositive >= 0.55) {
    return "positive_paper_candidate";
  }
  return "borderline_paper_candidate";
}

function availabilityVerdictFromExecutionRate(
  executionRate: number,
  totalObservations: number,
): PaperExecutionAvailabilityVerdict {
  if (totalObservations === 0 || !Number.isFinite(executionRate)) return "low_availability";
  if (executionRate >= 0.85) return "high_availability";
  if (executionRate >= 0.55) return "moderate_availability";
  return "low_availability";
}

function continuationDecisionFromDailyGates(params: {
  totalObservations: number;
  executionRate: number;
  blockedRate: number;
  meanNetPerObservation: number;
  latestNarrowValidationVerdict: string | null;
  latestExecutionGateVerdict: string | null;
  latestPaperExecutionVerdict: string | null;
}): ContinuationDecision {
  const {
    totalObservations,
    executionRate,
    blockedRate,
    meanNetPerObservation,
    latestNarrowValidationVerdict,
    latestExecutionGateVerdict,
    latestPaperExecutionVerdict,
  } = params;

  if (totalObservations === 0) return "pause_and_review";
  if (executionRate < 0.2 || blockedRate > 0.8 || meanNetPerObservation < -0.002) {
    return "stop_track";
  }
  if (
    latestNarrowValidationVerdict !== "survives_stress_suite" ||
    (latestExecutionGateVerdict !== "survives_execution_gate" &&
      latestExecutionGateVerdict !== "survives_and_scalable") ||
    latestPaperExecutionVerdict === "blocked_by_gate" ||
    executionRate < 0.55 ||
    blockedRate > 0.45 ||
    !Number.isFinite(meanNetPerObservation) ||
    meanNetPerObservation <= 0
  ) {
    return "pause_and_review";
  }
  return "continue_controlled_paper";
}

function summarizeHistoryFile(filePath: string): {
  totalObservations: number;
  totalBlockedCycles: number;
  totalPaperExecutedCycles: number;
  currentClassification: string;
  narrowValidationVerdict: string | null;
  executionGateVerdict: string | null;
  paperExecutionVerdict: string | null;
  executionRate: number;
  blockedRate: number;
  meanNetPerObservation: number;
  meanNetConditionalOnExecution: number;
  medianNetConditionalOnExecution: number;
  continuationDecision: ContinuationDecision;
  availabilityVerdict: PaperExecutionAvailabilityVerdict;
  meanEstimatedNetAfterPaperExecution: number;
  medianEstimatedNetAfterPaperExecution: number;
  minEstimatedNetAfterPaperExecution: number;
  maxEstimatedNetAfterPaperExecution: number;
  pctPositivePaperCycles: number;
  strongestWindow: string | null;
  weakestWindow: string | null;
  finalPaperExecutionVerdict: FinalPaperExecutionVerdict;
  finalPaperExecutionSummaryLine: string;
} {
  const raw = fs.readFileSync(filePath, "utf8");
  const lines = raw.split("\n").filter(Boolean);

  let totalObservations = 0;
  let totalBlockedCycles = 0;
  let sumNetPerObservation = 0;
  let latestIso = "";
  let latestNarrowValidationVerdict: string | null = null;
  let latestExecutionGateVerdict: string | null = null;
  let latestPaperExecutionVerdict: string | null = null;
  const executedRows: { iso: string; net: number }[] = [];

  for (const line of lines) {
    let j: Row;
    try {
      j = JSON.parse(line) as Row;
    } catch {
      continue;
    }
    if (String(j.marketId ?? "") !== "1823789") continue;
    if (j.probeVersion !== "cross-venue-anchor-1823789-paper-execution-v1") continue;
    totalObservations += 1;
    const iso = String(j.isoTimestamp ?? "");
    if (iso >= latestIso) {
      latestIso = iso;
      latestNarrowValidationVerdict =
        typeof j.narrowValidationVerdict === "string" ? j.narrowValidationVerdict : null;
      latestExecutionGateVerdict =
        typeof j.executionGateVerdict === "string" ? j.executionGateVerdict : null;
      latestPaperExecutionVerdict =
        typeof j.paperExecutionVerdict === "string" ? j.paperExecutionVerdict : null;
    }

    const v = j.paperExecutionVerdict;
    if (v === "blocked_by_gate") {
      totalBlockedCycles += 1;
      continue;
    }
    const net = j.estimatedNetAfterPaperExecution;
    if (typeof net !== "number" || !Number.isFinite(net)) {
      continue;
    }
    sumNetPerObservation += net;
    executedRows.push({ iso: String(j.isoTimestamp ?? ""), net });
  }

  const totalPaperExecutedCycles = executedRows.length;
  const nets = executedRows.map(r => r.net);

  let maxN = -Infinity;
  let minN = Infinity;
  let strongestIso: string | null = null;
  let weakestIso: string | null = null;
  for (const r of executedRows) {
    if (r.net > maxN) {
      maxN = r.net;
      strongestIso = r.iso || null;
    }
    if (r.net < minN) {
      minN = r.net;
      weakestIso = r.iso || null;
    }
  }

  const pctPositivePaperCycles = nets.length === 0 ? 0 : nets.filter(x => x > 0).length / nets.length;
  const meanNet = mean(nets);
  const medNet = medianSorted(nets);
  const fv = finalVerdict({ totalPaperExecutedCycles, totalBlockedCycles, nets });

  const executionRate =
    totalObservations > 0 ? Math.round((totalPaperExecutedCycles / totalObservations) * 1_000_000) / 1_000_000 : 0;
  const blockedRate =
    totalObservations > 0 ? Math.round((totalBlockedCycles / totalObservations) * 1_000_000) / 1_000_000 : 0;
  const meanNetPerObservation =
    totalObservations > 0
      ? Math.round((sumNetPerObservation / totalObservations) * 1_000_000) / 1_000_000
      : NaN;
  const meanNetConditionalOnExecution = meanNet;
  const medianNetConditionalOnExecution = medNet;
  const availabilityVerdict = availabilityVerdictFromExecutionRate(executionRate, totalObservations);
  const continuationDecision = continuationDecisionFromDailyGates({
    totalObservations,
    executionRate,
    blockedRate,
    meanNetPerObservation,
    latestNarrowValidationVerdict,
    latestExecutionGateVerdict,
    latestPaperExecutionVerdict,
  });

  const finalPaperExecutionSummaryLine = [
    `paper_exec_1823789 n=${totalObservations}`,
    `blocked=${totalBlockedCycles}`,
    `executed=${totalPaperExecutedCycles}`,
    `exec_rate=${(executionRate * 100).toFixed(1)}%`,
    `blocked_rate=${(blockedRate * 100).toFixed(1)}%`,
    `mean_net_per_obs=${Number.isFinite(meanNetPerObservation) ? meanNetPerObservation.toFixed(6) : "NaN"}`,
    `continuation=${continuationDecision}`,
    `avail=${availabilityVerdict}`,
    `verdict=${fv}`,
    `mean_net_after=${Number.isFinite(meanNet) ? meanNet.toFixed(6) : "NaN"}`,
    `median_net_after=${Number.isFinite(medNet) ? medNet.toFixed(6) : "NaN"}`,
    `pct_pos=${(pctPositivePaperCycles * 100).toFixed(1)}%`,
  ].join(" ");

  return {
    totalObservations,
    totalBlockedCycles,
    totalPaperExecutedCycles,
    currentClassification: getStrategyTrackClassification(ACTIVE_SINGLE_TRACK_EXPERIMENT_ID),
    narrowValidationVerdict: latestNarrowValidationVerdict,
    executionGateVerdict: latestExecutionGateVerdict,
    paperExecutionVerdict: latestPaperExecutionVerdict,
    executionRate,
    blockedRate,
    meanNetPerObservation,
    meanNetConditionalOnExecution: Number.isFinite(meanNetConditionalOnExecution)
      ? Math.round(meanNetConditionalOnExecution * 1_000_000) / 1_000_000
      : NaN,
    medianNetConditionalOnExecution: Number.isFinite(medianNetConditionalOnExecution)
      ? Math.round(medianNetConditionalOnExecution * 1_000_000) / 1_000_000
      : NaN,
    continuationDecision,
    availabilityVerdict,
    meanEstimatedNetAfterPaperExecution: Number.isFinite(meanNet) ? Math.round(meanNet * 1_000_000) / 1_000_000 : NaN,
    medianEstimatedNetAfterPaperExecution: Number.isFinite(medNet) ? Math.round(medNet * 1_000_000) / 1_000_000 : NaN,
    minEstimatedNetAfterPaperExecution:
      nets.length === 0 ? NaN : Math.round(minN * 1_000_000) / 1_000_000,
    maxEstimatedNetAfterPaperExecution:
      nets.length === 0 ? NaN : Math.round(maxN * 1_000_000) / 1_000_000,
    pctPositivePaperCycles: Math.round(pctPositivePaperCycles * 1000) / 1000,
    strongestWindow: strongestIso,
    weakestWindow: weakestIso,
    finalPaperExecutionVerdict: fv,
    finalPaperExecutionSummaryLine,
  };
}

function resolveHistoryPath(): string {
  const raw = process.env.CROSS_VENUE_ANCHOR_1823789_PAPER_EXECUTION_HISTORY_PATH?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789PaperExecutionHistory);
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
        probe: "cross-venue-anchor-1823789-paper-execution-v1",
        marketId: "1823789",
        ...summary,
      },
      null,
      2,
    ),
  );
}

main();
