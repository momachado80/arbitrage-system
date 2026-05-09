/**
 * Top-Slice Robustness Assessment — camada temporal sobre RankedEventAssessment.
 * Mantém histórico leve in-memory de snapshots do ranking e avalia se o sinal
 * do top slice é persistente, diversificado e robusto, ou apenas um artefato
 * concentrado e efémero. Não altera eventos, ranking ou probe state.
 */

import type { RankedEventAssessment } from "./momentumRankedEventAssessment";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}
function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

const MAX_WINDOWS = () =>
  Math.max(5, Math.floor(envNum("MOMENTUM_ROBUSTNESS_MAX_WINDOWS", 20)));
const MIN_WINDOWS_FOR_READ = () =>
  Math.max(3, Math.floor(envNum("MOMENTUM_ROBUSTNESS_MIN_WINDOWS", 5)));
const DIVERSITY_GOOD_THRESHOLD = () =>
  envNum("MOMENTUM_ROBUSTNESS_DIVERSITY_GOOD", 3);
const PERSISTENCE_STRONG_RATE = () =>
  envNum("MOMENTUM_ROBUSTNESS_PERSISTENCE_STRONG", 0.7);
const CONCENTRATION_DECLINING_THRESHOLD = () =>
  envNum("MOMENTUM_ROBUSTNESS_CONC_DECLINING", -0.05);

const GLOBAL_KEY = "__momentumTopSliceRobustness_v1";

export type TopSliceRobustnessVerdict =
  | "insufficient_history"
  | "concentrated_artifact_risk"
  | "weak_but_persistent"
  | "improving_and_diversifying"
  | "robust_top_slice_signal"
  | "unstable";

interface WindowSnapshot {
  ts: number;
  totalEvents: number;
  topSliceShowsImprovement: boolean;
  improvementMagnitude: number | null;
  concentrationRisk: number;
  avgProxyTopDecile: number | null;
  avgProxyAll: number | null;
  capturableRateTopDecile: number;
  distinctMarketsInTopDecile: number;
}

interface RobustnessState {
  windows: WindowSnapshot[];
}

function getState(): RobustnessState {
  const g = globalThis as unknown as Record<string, RobustnessState | undefined>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = { windows: [] };
  }
  return g[GLOBAL_KEY]!;
}

function recordWindow(r: RankedEventAssessment): void {
  const st = getState();
  const distinctMarkets = Object.keys(r.topMarketsBreakdown).length;
  st.windows.push({
    ts: Date.now(),
    totalEvents: r.totalEventsConsidered,
    topSliceShowsImprovement: r.topSliceShowsImprovement,
    improvementMagnitude: r.topSliceImprovementMagnitude,
    concentrationRisk: r.concentrationRiskInTopSlice,
    avgProxyTopDecile: r.averageCapturableProxyTopDecile,
    avgProxyAll: r.averageCapturableProxyAll,
    capturableRateTopDecile: r.capturableRateTopDecile,
    distinctMarketsInTopDecile: distinctMarkets,
  });
  const max = MAX_WINDOWS();
  if (st.windows.length > max) {
    st.windows = st.windows.slice(-max);
  }
}

function linearTrend(values: number[]): number {
  const n = values.length;
  if (n < 2) return 0;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += values[i]!;
    sumXY += i * values[i]!;
    sumX2 += i * i;
  }
  const denom = n * sumX2 - sumX * sumX;
  if (denom === 0) return 0;
  return r4((n * sumXY - sumX * sumY) / denom);
}

export interface TopSliceRobustnessAssessment {
  totalEvaluationWindows: number;
  windowsWithTopSliceImprovement: number;
  windowsWithPositiveTopDecileProxy: number;
  windowsWithPositiveTopQuartileProxy: number;
  repeatedTopSliceImprovement: boolean;
  repeatedTopSliceConcentration: boolean;
  topSliceMarketDiversityCount: number;
  topSliceMarketDiversityScore: number;
  topSliceTemporalPersistenceScore: number;
  topSliceConcentrationTrend: number;
  topSliceImprovementTrend: number;
  topSliceRobustnessVerdict: TopSliceRobustnessVerdict;
  supportingReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: Record<string, number>;
}

export function buildTopSliceRobustnessAssessment(
  currentRanking: RankedEventAssessment,
): TopSliceRobustnessAssessment {
  recordWindow(currentRanking);

  const st = getState();
  const w = st.windows;
  const n = w.length;
  const minW = MIN_WINDOWS_FOR_READ();
  const divGood = DIVERSITY_GOOD_THRESHOLD();
  const persStrong = PERSISTENCE_STRONG_RATE();
  const concDeclT = CONCENTRATION_DECLINING_THRESHOLD();

  const improvementWindows = w.filter(s => s.topSliceShowsImprovement).length;
  const posDecileWindows = w.filter(
    s => s.avgProxyTopDecile !== null && s.avgProxyTopDecile > 0,
  ).length;
  const posQuartileWindows = w.filter(
    s => s.avgProxyAll !== null && s.capturableRateTopDecile > 0,
  ).length;

  const improvRate = n > 0 ? r4(improvementWindows / n) : 0;
  const repeatedImprovement = improvementWindows >= 3 && improvRate >= 0.5;

  const concValues = w.map(s => s.concentrationRisk);
  const highConcWindows = concValues.filter(c => c > 0.5).length;
  const highConcRate = n > 0 ? highConcWindows / n : 0;
  const repeatedConcentration = highConcWindows >= 3 && highConcRate >= 0.5;

  const allDistinctMarkets = new Set<number>();
  for (const s of w) {
    allDistinctMarkets.add(s.distinctMarketsInTopDecile);
  }
  const latestDiversity = w.length > 0 ? w[w.length - 1]!.distinctMarketsInTopDecile : 0;
  const maxDiversity = w.reduce((mx, s) => Math.max(mx, s.distinctMarketsInTopDecile), 0);
  const avgDiversity =
    n > 0
      ? r4(w.reduce((s, x) => s + x.distinctMarketsInTopDecile, 0) / n)
      : 0;
  const diversityScore = r4(Math.min(1, avgDiversity / Math.max(1, divGood)));

  const persistenceScore = improvRate;

  const concTrend = linearTrend(concValues);
  const improvMagnitudes = w.map(s => s.improvementMagnitude ?? 0);
  const improvTrend = linearTrend(improvMagnitudes);

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];

  let verdict: TopSliceRobustnessVerdict;

  if (n < minW) {
    verdict = "insufficient_history";
    blockingReasons.push(
      `Janelas avaliadas ${n} < mínimo ${minW}. Aguardar mais ciclos de scan.`,
    );
  } else if (!repeatedImprovement && improvementWindows <= 1) {
    verdict = "unstable";
    blockingReasons.push(
      `Melhoria do topo em apenas ${improvementWindows}/${n} janelas (rate=${improvRate}). Sinal instável.`,
    );
  } else if (repeatedConcentration && avgDiversity < 2) {
    if (repeatedImprovement && improvRate >= persStrong) {
      verdict = "concentrated_artifact_risk";
      supportingReasons.push(
        `Topo melhora em ${improvementWindows}/${n} janelas (persistente), mas concentrado (>50% num mercado) em ${highConcWindows}/${n} janelas. avgDiversity=${avgDiversity}. Risco elevado de artefato.`,
      );
    } else {
      verdict = "concentrated_artifact_risk";
      blockingReasons.push(
        `Concentração alta em ${highConcWindows}/${n} janelas com diversidade média=${avgDiversity}. Melhoria fraca (${improvementWindows}/${n}). Provável artefato.`,
      );
    }
  } else {
    const concDeclining = concTrend < concDeclT;
    const improvIncreasing = improvTrend > 0;
    const diversityGood = avgDiversity >= divGood;
    const persistenceGood = persistenceScore >= persStrong;

    if (persistenceGood && diversityGood && !repeatedConcentration) {
      verdict = "robust_top_slice_signal";
      supportingReasons.push(
        `Topo melhora em ${improvementWindows}/${n} janelas (${improvRate}), diversidade boa (avg=${avgDiversity}, score=${diversityScore}), concentração controlada (trend=${concTrend}). Sinal robusto.`,
      );
    } else if (
      (concDeclining || improvIncreasing) &&
      improvRate >= 0.4 &&
      avgDiversity >= 1.5
    ) {
      verdict = "improving_and_diversifying";
      supportingReasons.push(
        `Tendência positiva: concTrend=${concTrend}, improvTrend=${improvTrend}. Melhoria em ${improvementWindows}/${n} janelas. avgDiversity=${avgDiversity}. Diversificação em curso.`,
      );
    } else if (repeatedImprovement) {
      verdict = "weak_but_persistent";
      supportingReasons.push(
        `Melhoria persistente (${improvementWindows}/${n}) mas diversidade limitada (avg=${avgDiversity}) ou concentração ainda elevada (rate=${highConcRate}).`,
      );
    } else {
      verdict = "unstable";
      blockingReasons.push(
        `Melhoria intermitente (${improvementWindows}/${n}). Diversidade=${avgDiversity}, persistência=${persistenceScore}. Sinal não se sustenta.`,
      );
    }
  }

  return {
    totalEvaluationWindows: n,
    windowsWithTopSliceImprovement: improvementWindows,
    windowsWithPositiveTopDecileProxy: posDecileWindows,
    windowsWithPositiveTopQuartileProxy: posQuartileWindows,
    repeatedTopSliceImprovement: repeatedImprovement,
    repeatedTopSliceConcentration: repeatedConcentration,
    topSliceMarketDiversityCount: latestDiversity,
    topSliceMarketDiversityScore: diversityScore,
    topSliceTemporalPersistenceScore: persistenceScore,
    topSliceConcentrationTrend: concTrend,
    topSliceImprovementTrend: improvTrend,
    topSliceRobustnessVerdict: verdict,
    supportingReasons,
    blockingReasons,
    thresholdsUsed: {
      MOMENTUM_ROBUSTNESS_MAX_WINDOWS: MAX_WINDOWS(),
      MOMENTUM_ROBUSTNESS_MIN_WINDOWS: minW,
      MOMENTUM_ROBUSTNESS_DIVERSITY_GOOD: divGood,
      MOMENTUM_ROBUSTNESS_PERSISTENCE_STRONG: persStrong,
      MOMENTUM_ROBUSTNESS_CONC_DECLINING: concDeclT,
    },
  };
}

export function buildRobustnessSummaryLine(a: TopSliceRobustnessAssessment): string {
  if (a.totalEvaluationWindows < MIN_WINDOWS_FOR_READ()) {
    return `robustness: insufficient_history (${a.totalEvaluationWindows} windows)`;
  }
  return `robustness: ${a.topSliceRobustnessVerdict} | persist=${a.topSliceTemporalPersistenceScore} div=${a.topSliceMarketDiversityScore} concTrend=${a.topSliceConcentrationTrend} improvTrend=${a.topSliceImprovementTrend}`;
}
