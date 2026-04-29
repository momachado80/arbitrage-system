/**
 * Estatísticas partilhadas entre workers (summaries) e scoreboard — sem alterar lógica de estratégia.
 */

import fs from "fs";

export type LiveExperimentId =
  | "final_neg_risk_31552"
  | "reach_april_btc_reaction_monitor"
  | "cross_venue_anchor_1823789";

export type PromotabilityVerdict =
  | "insufficient_history"
  | "consistently_negative"
  | "episodic_but_not_promotable"
  | "borderline_keep_running"
  | "promotable_candidate";

export interface ExperimentObservationFrame {
  isoTimestamp: string;
  netPrimary: number;
}

/** Mínimo de linhas para ranking global do scoreboard. */
export const ENOUGH_HISTORY_FOR_RANKING = 10;

export const NEAR_ZERO_ABS_BAND = 0.005;

function mean(xs: number[]): number {
  if (xs.length === 0) return NaN;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function medianSorted(xs: number[]): number {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export function readJsonlLines(fp: string): string[] {
  if (!fs.existsSync(fp)) return [];
  const text = fs.readFileSync(fp, { encoding: "utf8" }) as string;
  return text
    .split("\n")
    .map(l => l.trim())
    .filter(Boolean);
}

/** Evento 31552: uma linha por ciclo; net = stress ajustado conversão. */
export function parseFinalNegRisk31552(lines: string[]): {
  frames: ExperimentObservationFrame[];
  lastVerdict: string | null;
} {
  const frames: ExperimentObservationFrame[] = [];
  let lastIso = "";
  let lastVerdict: string | null = null;
  for (const line of lines) {
    try {
      const j = JSON.parse(line) as {
        isoTimestamp?: string;
        stressAdjustedNetConversionEdge?: number;
        finalNegativeRiskValidationVerdict?: string;
      };
      const ts = typeof j.isoTimestamp === "string" ? j.isoTimestamp : "";
      const net = typeof j.stressAdjustedNetConversionEdge === "number" ? j.stressAdjustedNetConversionEdge : NaN;
      if (!Number.isFinite(net)) continue;
      frames.push({ isoTimestamp: ts || new Date(0).toISOString(), netPrimary: net });
      if (ts >= lastIso) {
        lastIso = ts;
        lastVerdict =
          typeof j.finalNegativeRiskValidationVerdict === "string"
            ? j.finalNegativeRiskValidationVerdict
            : lastVerdict;
      }
    } catch {
      continue;
    }
  }
  return { frames, lastVerdict };
}

/** BTC narrow: uma linha por mercado por ciclo. */
export function parseReachAprilBtc(lines: string[]): {
  frames: ExperimentObservationFrame[];
  lastVerdict: string | null;
} {
  const frames: ExperimentObservationFrame[] = [];
  let lastIso = "";
  let lastVerdict: string | null = null;
  for (const line of lines) {
    try {
      const j = JSON.parse(line) as {
        isoTimestamp?: string;
        medianNetPerReactionCycle?: number;
        calibratedReactionVerdictPerMarket?: string;
      };
      const ts = typeof j.isoTimestamp === "string" ? j.isoTimestamp : "";
      const net =
        typeof j.medianNetPerReactionCycle === "number" ? j.medianNetPerReactionCycle : NaN;
      if (!Number.isFinite(net)) continue;
      frames.push({ isoTimestamp: ts || new Date(0).toISOString(), netPrimary: net });
      if (ts >= lastIso) {
        lastIso = ts;
        lastVerdict =
          typeof j.calibratedReactionVerdictPerMarket === "string"
            ? j.calibratedReactionVerdictPerMarket
            : lastVerdict;
      }
    } catch {
      continue;
    }
  }
  return { frames, lastVerdict };
}

/** Anchor 1823789: uma linha por ciclo. */
export function parseCrossVenueAnchor1823789(lines: string[]): {
  frames: ExperimentObservationFrame[];
  lastVerdict: string | null;
} {
  const frames: ExperimentObservationFrame[] = [];
  let lastIso = "";
  let lastVerdict: string | null = null;
  for (const line of lines) {
    try {
      const j = JSON.parse(line) as {
        isoTimestamp?: string;
        estimatedNetAnchorCycle?: number;
        refinedPilotVerdictPerMarket?: string;
      };
      const ts = typeof j.isoTimestamp === "string" ? j.isoTimestamp : "";
      const net =
        typeof j.estimatedNetAnchorCycle === "number" ? j.estimatedNetAnchorCycle : NaN;
      if (!Number.isFinite(net)) continue;
      frames.push({ isoTimestamp: ts || new Date(0).toISOString(), netPrimary: net });
      if (ts >= lastIso) {
        lastIso = ts;
        lastVerdict =
          typeof j.refinedPilotVerdictPerMarket === "string"
            ? j.refinedPilotVerdictPerMarket
            : lastVerdict;
      }
    } catch {
      continue;
    }
  }
  return { frames, lastVerdict };
}

export function summarizeObservationFrames(
  frames: ExperimentObservationFrame[],
  nearZeroBand: number,
): {
  meanNet: number;
  medianNet: number;
  minNet: number;
  maxNet: number;
  pctPositive: number;
  pctNearZero: number;
  strongestWindow: string | null;
  weakestWindow: string | null;
} {
  const nets = frames.map(f => f.netPrimary);
  if (nets.length === 0) {
    return {
      meanNet: NaN,
      medianNet: NaN,
      minNet: NaN,
      maxNet: NaN,
      pctPositive: 0,
      pctNearZero: 0,
      strongestWindow: null,
      weakestWindow: null,
    };
  }
  let maxN = -Infinity;
  let minN = Infinity;
  let strongIso: string | null = null;
  let weakIso: string | null = null;
  for (const f of frames) {
    if (f.netPrimary > maxN) {
      maxN = f.netPrimary;
      strongIso = f.isoTimestamp || null;
    }
    if (f.netPrimary < minN) {
      minN = f.netPrimary;
      weakIso = f.isoTimestamp || null;
    }
  }
  const pctPositive = nets.filter(x => x > 0).length / nets.length;
  const pctNearZero = nets.filter(x => Math.abs(x) <= nearZeroBand).length / nets.length;
  return {
    meanNet: Math.round(mean(nets) * 1_000_000) / 1_000_000,
    medianNet: Math.round(medianSorted(nets) * 1_000_000) / 1_000_000,
    minNet: Math.round(minN * 1_000_000) / 1_000_000,
    maxNet: Math.round(maxN * 1_000_000) / 1_000_000,
    pctPositive: Math.round(pctPositive * 1000) / 1000,
    pctNearZero: Math.round(pctNearZero * 1000) / 1000,
    strongestWindow: strongIso,
    weakestWindow: weakIso,
  };
}

export function derivePromotabilityVerdict(
  n: number,
  meanNet: number,
  medianNet: number,
  pctPositive: number,
): PromotabilityVerdict {
  if (n < ENOUGH_HISTORY_FOR_RANKING) {
    return "insufficient_history";
  }
  if (medianNet <= 0 && meanNet <= 0 && pctPositive < 0.12) {
    return "consistently_negative";
  }
  if (medianNet > 0 && meanNet > 0 && pctPositive >= 0.42) {
    return "promotable_candidate";
  }
  if (
    medianNet > -0.003 &&
    medianNet < 0.012 &&
    pctPositive >= 0.22 &&
    pctPositive <= 0.52
  ) {
    return "borderline_keep_running";
  }
  return "episodic_but_not_promotable";
}
