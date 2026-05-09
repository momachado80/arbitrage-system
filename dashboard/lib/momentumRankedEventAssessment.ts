/**
 * Ranked Event Assessment — camada analítica pura sobre MomentumEvent[].
 * Ordena eventos por qualidade (captureProxy + magnitude − penalidade),
 * compara fatias (all, topQuartile, topDecile, top1) e avalia se o topo
 * salva a tese de microestrutura. Não altera eventos nem probe state.
 */

import type { MomentumEvent, MomentumEventType } from "./momentumSnipingProbe";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}
function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

const MIN_EVENTS_FOR_RANKING = () =>
  Math.max(4, Math.floor(envNum("MOMENTUM_RANKING_MIN_EVENTS", 8)));
const CONCENTRATION_RISK_THRESHOLD = () =>
  envNum("MOMENTUM_RANKING_CONCENTRATION_RISK", 0.5);
const IMPROVEMENT_THRESHOLD = () =>
  envNum("MOMENTUM_RANKING_IMPROVEMENT_THRESHOLD", 0.003);

export type RankedSignalVerdict =
  | "insufficient_sample"
  | "no_top_slice_improvement"
  | "weak_top_slice_signal"
  | "promising_top_slice_signal"
  | "concentrated_but_interesting"
  | "unstable_or_negative";

export interface RankedEventAssessment {
  totalEventsConsidered: number;
  eventsEligibleForRanking: number;
  rankingMethodDescription: string;
  topDecileCount: number;
  topQuartileCount: number;
  top1Count: number;
  averageCapturableProxyAll: number | null;
  averageCapturableProxyTopQuartile: number | null;
  averageCapturableProxyTopDecile: number | null;
  averageCapturableProxyTop1: number | null;
  capturableRateAll: number;
  capturableRateTopQuartile: number;
  capturableRateTopDecile: number;
  capturableRateTop1: number;
  topEventTypesBreakdown: Record<string, number>;
  topMarketsBreakdown: Record<string, number>;
  concentrationRiskInTopSlice: number;
  topSliceShowsImprovement: boolean;
  topSliceImprovementMagnitude: number | null;
  rankedSignalVerdict: RankedSignalVerdict;
  supportingReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: Record<string, number>;
}

function scoreEvent(e: MomentumEvent): number {
  const proxy = e.conservativeCaptureProxy;
  const mag = e.magnitude;
  const liqBonus = e.liquidityAtDetection >= 10_000 ? 0.001 : 0;
  return proxy + mag * 0.15 + liqBonus;
}

function avg(nums: number[]): number | null {
  if (nums.length === 0) return null;
  return r4(nums.reduce((a, b) => a + b, 0) / nums.length);
}

function capturableRate(events: MomentumEvent[]): number {
  if (events.length === 0) return 0;
  return r4(events.filter(e => e.capturable).length / events.length);
}

function typeBreakdown(events: MomentumEvent[]): Record<string, number> {
  const m: Record<string, number> = {};
  for (const e of events) {
    m[e.eventType] = (m[e.eventType] ?? 0) + 1;
  }
  return m;
}

function marketBreakdown(events: MomentumEvent[]): Record<string, number> {
  const m: Record<string, number> = {};
  for (const e of events) {
    const label = e.marketQuestion.slice(0, 80);
    m[label] = (m[label] ?? 0) + 1;
  }
  return m;
}

function concentrationRisk(events: MomentumEvent[]): number {
  if (events.length === 0) return 0;
  const counts: Record<string, number> = {};
  for (const e of events) {
    counts[e.marketId] = (counts[e.marketId] ?? 0) + 1;
  }
  const maxCount = Math.max(...Object.values(counts));
  return r4(maxCount / events.length);
}

export function buildRankedEventAssessment(
  allEvents: readonly MomentumEvent[],
): RankedEventAssessment {
  const minEv = MIN_EVENTS_FOR_RANKING();
  const concRiskT = CONCENTRATION_RISK_THRESHOLD();
  const improvT = IMPROVEMENT_THRESHOLD();

  const total = allEvents.length;
  const ranked = [...allEvents].sort((a, b) => scoreEvent(b) - scoreEvent(a));

  const topQuartileN = Math.max(1, Math.floor(total * 0.25));
  const topDecileN = Math.max(1, Math.floor(total * 0.10));
  const top1N = Math.max(1, Math.min(3, Math.ceil(total * 0.01)));

  const topQuartile = ranked.slice(0, topQuartileN);
  const topDecile = ranked.slice(0, topDecileN);
  const top1 = ranked.slice(0, top1N);

  const proxiesAll = allEvents.map(e => e.conservativeCaptureProxy);
  const proxiesQ = topQuartile.map(e => e.conservativeCaptureProxy);
  const proxiesD = topDecile.map(e => e.conservativeCaptureProxy);
  const proxies1 = top1.map(e => e.conservativeCaptureProxy);

  const avgAll = avg(proxiesAll);
  const avgQ = avg(proxiesQ);
  const avgD = avg(proxiesD);
  const avg1 = avg(proxies1);

  const concRisk = concentrationRisk(topDecile);

  const bestSliceAvg = avgD ?? avgQ ?? avg1 ?? null;
  const improvement =
    bestSliceAvg !== null && avgAll !== null
      ? r4(bestSliceAvg - avgAll)
      : null;
  const topSliceShowsImprovement =
    improvement !== null && improvement > improvT;

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];

  let verdict: RankedSignalVerdict;

  if (total < minEv) {
    verdict = "insufficient_sample";
    blockingReasons.push(
      `Total de eventos ${total} < mínimo para ranking ${minEv}.`,
    );
  } else if (
    avgAll !== null &&
    avgAll < 0 &&
    (bestSliceAvg === null || bestSliceAvg < 0)
  ) {
    verdict = "unstable_or_negative";
    blockingReasons.push(
      `Média global (${avgAll}) e melhor fatia (${bestSliceAvg}) ambas negativas.`,
    );
  } else if (!topSliceShowsImprovement) {
    verdict = "no_top_slice_improvement";
    blockingReasons.push(
      `Topo não melhora: avgAll=${avgAll}, topDecile=${avgD}, melhoria=${improvement} vs threshold=${improvT}.`,
    );
  } else if (concRisk > concRiskT) {
    const capRateD = capturableRate(topDecile);
    if (capRateD >= 0.5 && improvement !== null && improvement > improvT * 2) {
      verdict = "concentrated_but_interesting";
      supportingReasons.push(
        `Top decile melhora ${improvement} mas concentrado (${r4(concRisk * 100)}% num mercado). capturableRate topDecile=${capRateD}.`,
      );
    } else {
      verdict = "weak_top_slice_signal";
      blockingReasons.push(
        `Concentração de ${r4(concRisk * 100)}% num único mercado no top decile; risco de artefato. melhoria=${improvement}.`,
      );
    }
  } else {
    const capRateD = capturableRate(topDecile);
    if (
      capRateD >= 0.6 &&
      improvement !== null &&
      improvement > improvT * 3
    ) {
      verdict = "promising_top_slice_signal";
      supportingReasons.push(
        `Top decile: avgProxy=${avgD}, capturableRate=${capRateD}, melhoria=${improvement} vs média global=${avgAll}. Diversificado (concentração=${r4(concRisk * 100)}%).`,
      );
    } else if (capRateD >= 0.3 && topSliceShowsImprovement) {
      verdict = "weak_top_slice_signal";
      supportingReasons.push(
        `Melhoria detectada no topo (${improvement}) mas capturableRate apenas ${capRateD}. avgTopDecile=${avgD} vs avgAll=${avgAll}.`,
      );
    } else {
      verdict = "no_top_slice_improvement";
      blockingReasons.push(
        `Melhoria marginal: topDecile capturableRate=${capRateD}, avgD=${avgD}, avgAll=${avgAll}.`,
      );
    }
  }

  return {
    totalEventsConsidered: total,
    eventsEligibleForRanking: total,
    rankingMethodDescription:
      "score = captureProxy + magnitude×0.15 + liquidityBonus(0.001 se liq≥10k). Ranking descendente por score; fatias: top25%, top10%, top1% (min 1, max 3).",
    topDecileCount: total >= minEv ? topDecileN : 0,
    topQuartileCount: total >= minEv ? topQuartileN : 0,
    top1Count: total >= minEv ? top1N : 0,
    averageCapturableProxyAll: avgAll,
    averageCapturableProxyTopQuartile: avgQ,
    averageCapturableProxyTopDecile: avgD,
    averageCapturableProxyTop1: avg1,
    capturableRateAll: capturableRate(allEvents as MomentumEvent[]),
    capturableRateTopQuartile: total >= minEv ? capturableRate(topQuartile) : 0,
    capturableRateTopDecile: total >= minEv ? capturableRate(topDecile) : 0,
    capturableRateTop1: total >= minEv ? capturableRate(top1) : 0,
    topEventTypesBreakdown: total >= minEv ? typeBreakdown(topDecile) : {},
    topMarketsBreakdown: total >= minEv ? marketBreakdown(topDecile) : {},
    concentrationRiskInTopSlice: concRisk,
    topSliceShowsImprovement,
    topSliceImprovementMagnitude: improvement,
    rankedSignalVerdict: verdict,
    supportingReasons,
    blockingReasons,
    thresholdsUsed: {
      MOMENTUM_RANKING_MIN_EVENTS: minEv,
      MOMENTUM_RANKING_CONCENTRATION_RISK: concRiskT,
      MOMENTUM_RANKING_IMPROVEMENT_THRESHOLD: improvT,
    },
  };
}

export function buildRankedSignalSummaryLine(a: RankedEventAssessment): string {
  if (a.totalEventsConsidered < MIN_EVENTS_FOR_RANKING()) {
    return `ranked: insufficient_sample (${a.totalEventsConsidered} eventos)`;
  }
  const imp = a.topSliceImprovementMagnitude !== null
    ? (a.topSliceImprovementMagnitude > 0 ? "+" : "") + String(a.topSliceImprovementMagnitude)
    : "n/a";
  return `ranked: ${a.rankedSignalVerdict} | topDecile avgProxy=${a.averageCapturableProxyTopDecile} capRate=${a.capturableRateTopDecile} imp=${imp} conc=${r4(a.concentrationRiskInTopSlice * 100)}%`;
}
