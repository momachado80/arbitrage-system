/**
 * Recalibração só leitura de custo de execução: compara o proxy estático (executionTruthEngine)
 * com um estado derivado das séries persistidas market-data-truth (spread/depth/mid no tempo).
 * Não altera veredictos de viabilidade nem thresholds de estratégia.
 */

import { loadMarketDataTruthStore, type MarketTruthSeriesDisk } from "./marketDataTruthCapture";
import {
  adverseSelectionAtSpread,
  estimatedNetFromState,
  feeProxy,
  fillPlausibilityAtSpread,
  inventoryRiskAtSpread,
  observedDepth,
  observedSpread,
  unwindCostAtSpread,
} from "./executionTruthEngine";
import { fetchNormalizedMarketById } from "./polymarketClient";
import { getAllMarkets } from "./marketDataService";
import type { NormalizedMarket } from "./polymarketClient";

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function mean(nums: number[]): number {
  if (nums.length === 0) return 0;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function stdev(nums: number[]): number {
  if (nums.length < 2) return 0;
  const mu = mean(nums);
  const v = nums.reduce((s, x) => s + (x - mu) ** 2, 0) / (nums.length - 1);
  return Math.sqrt(v);
}

function median(nums: number[]): number {
  if (nums.length === 0) return 0;
  const s = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

const MIN_SNAPSHOTS = 12;
const NEAR_ZERO_ABS = 0.0006;
/** Redução mínima de |net| quando o proxy era défice — “mais perto de zero”. */
const MEANINGFUL_CLOSER_TO_ZERO_ABS = 0.0035;
const PAPER_FILL_CANDIDATE_MIN_DELTA = 0.006;
const PAPER_FILL_CANDIDATE_MAX_ABS_REC = 0.008;
const PAPER_FILL_CANDIDATE_TOP_REC_RANK = 3;

export type ExecutionCostRecalibrationVerdict =
  | "insufficient_truth_series_for_recalibration"
  | "recalibration_neutral_vs_proxy"
  | "recalibration_materially_more_positive_than_proxy"
  | "recalibration_materially_more_negative_than_proxy";

export type ProxyModelHarshnessVsObserved = "materially_too_harsh" | "broadly_correct" | "materially_too_lenient";

export interface StrongestMarketAfterRecalibration {
  marketId: string;
  marketTitle: string;
  snapshotsUsed: number;
  estimatedNetPerCycleProxy: number;
  recalibratedNetPerCycle: number;
  deltaNetFromRecalibration: number;
}

/** Vista compacta só leitura: top = conjunto analisado, ordenado por rank recalibrado. */
export interface MarketRecalibrationComparisonCompact {
  marketId: string;
  marketTitle: string;
  estimatedNetPerCycle: number;
  recalibratedNetPerCycle: number;
  deltaNetFromRecalibration: number;
  rankByRecalibratedNet: number;
  rankByDeltaNet: number;
}

export type AutonomousPolymarketPathClosureAssessment =
  | "economically_unsupported"
  | "one_market_warrants_final_validation";

export interface ExecutionCostRecalibrationDigest {
  probeVersion: "execution-cost-recalibration-v1";
  readDisclaimer: string;
  marketsAnalyzed: number;
  marketsSkippedShortSeries: number;
  proxyExecutionCost: number;
  recalibratedExecutionCost: number;
  spreadCostProxyVsObserved: number;
  depthCostProxyVsObserved: number;
  driftPenaltyProxyVsObserved: number;
  recalibratedNetPerCycle: number;
  deltaNetFromRecalibration: number;
  strongestMarketsAfterRecalibration: StrongestMarketAfterRecalibration[];
  executionCostRecalibrationVerdict: ExecutionCostRecalibrationVerdict;
  executionCostRecalibrationSummaryLine: string;
  proxyModelHarshnessVsObserved: ProxyModelHarshnessVsObserved;
  marketsNearZeroAfterRecalibration: number;
  marketsPositiveAfterRecalibration: number;
  marketRecalibrationComparisonCompact: MarketRecalibrationComparisonCompact[];
  anyMarketMeaningfullyCloserToZeroAfterRecalibration: boolean;
  anyMarketDeservesFinalPaperFillValidation: boolean;
  autonomousPolymarketPathClosureAssessment: AutonomousPolymarketPathClosureAssessment;
  finalRecommendationLine: string;
  computedAt: string;
}

function seriesObservedSpreadDepth(row: MarketTruthSeriesDisk): {
  sMean: number;
  dMean: number;
  midDriftAddon: number;
} {
  const spreads = row.snapshots.map(s => s.spread).filter(x => Number.isFinite(x));
  const depths = row.snapshots.map(s => s.depth).filter(x => Number.isFinite(x) && x >= 0);
  const mids = row.snapshots.map(s => s.mid).filter(x => Number.isFinite(x));
  const steps = row.snapshots.map(s => s.midStepDelta).filter((x): x is number => x != null && Number.isFinite(x));
  const sMean = r6(Math.min(0.48, Math.max(0.0001, mean(spreads))));
  const dMean = r6(Math.min(24_000, Math.max(0, mean(depths.length ? depths : [0]))));
  const stepPart = steps.length ? mean(steps) : 0;
  const volPart = mids.length >= 2 ? stdev(mids) : 0;
  const midDriftAddon = r6(Math.min(0.045, 1.85 * stepPart + 0.55 * volPart));
  return { sMean, dMean: dMean > 0 ? dMean : 0, midDriftAddon };
}

function depthLiquidityScale(dProxy: number, dObs: number): number {
  if (!(dObs > 1e-6) || !(dProxy > 1e-6)) return 1;
  const ratio = dProxy / dObs;
  return r6(Math.min(1.35, Math.max(0.62, ratio)));
}

export async function buildExecutionCostRecalibrationDigest(): Promise<ExecutionCostRecalibrationDigest> {
  const store = await loadMarketDataTruthStore();
  const all = getAllMarkets();
  const byId = new Map(all.map(m => [String(m.id), m]));

  const perMarket: {
    m: NormalizedMarket;
    row: MarketTruthSeriesDisk;
    proxyDrag: number;
    recDrag: number;
    grossP: number;
    grossO: number;
    depthRatio: number;
    driftRatio: number;
    netP: number;
    netR: number;
  }[] = [];

  let skipped = 0;

  for (const id of Object.keys(store.markets)) {
    const row = store.markets[id];
    if (row.snapshots.length < MIN_SNAPSHOTS) {
      skipped++;
      continue;
    }
    let m: NormalizedMarket | undefined = byId.get(String(id));
    if (!m) {
      m = (await fetchNormalizedMarketById(String(id))) ?? undefined;
    }
    if (!m) {
      skipped++;
      continue;
    }

    const sP = observedSpread(m);
    const dP = observedDepth(m);
    const fillP = fillPlausibilityAtSpread(m, sP);
    const advP = adverseSelectionAtSpread(m, sP);
    const invP = inventoryRiskAtSpread(m, sP);
    const unwP = unwindCostAtSpread(m, sP);
    const fee = feeProxy();
    const proxyDrag = r6(advP + invP + unwP + fee);
    const grossP = r6(sP * 0.13 * fillP);

    const { sMean, dMean, midDriftAddon } = seriesObservedSpreadDepth(row);
    const sO = sMean;
    const dObs = dMean > 0 ? dMean : dP;
    const fillO = fillPlausibilityAtSpread(m, sO);
    const advBase = adverseSelectionAtSpread(m, sO);
    const advR = r6(advBase + midDriftAddon);
    const liqScale = depthLiquidityScale(dP, dObs);
    const invR = r6(inventoryRiskAtSpread(m, sO) * liqScale);
    const unwR = r6(unwindCostAtSpread(m, sO) * liqScale);
    const recDrag = r6(advR + invR + unwR + fee);
    const grossO = r6(sO * 0.13 * fillO);

    const netP = estimatedNetFromState(sP, fillP, advP, invP, unwP, fee);
    const netR = estimatedNetFromState(sO, fillO, advR, invR, unwR, fee);

    const depthRatio = dP > 1e-6 ? r6(dObs / dP) : 1;
    const driftRatio = advP > 1e-8 ? r6(advR / advP) : 1;

    perMarket.push({
      m,
      row,
      proxyDrag,
      recDrag,
      grossP,
      grossO,
      depthRatio,
      driftRatio,
      netP: netP,
      netR: netR,
    });
  }

  const n = perMarket.length;

  if (n < 2) {
    const line = `execution_cost_recalibration: verdict=insufficient_truth_series_for_recalibration | analyzed=${n} skipped_short=${skipped}`;
    return {
      probeVersion: "execution-cost-recalibration-v1",
      readDisclaimer:
        "Só leitura: séries market-data-truth vs proxies executionTruthEngine. Não altera veredictos nem estratégia. Custos observados são agregados temporais (spread/depth CLOB+Gamma); drift aditivo a partir de mid/step.",
      marketsAnalyzed: n,
      marketsSkippedShortSeries: skipped,
      proxyExecutionCost: 0,
      recalibratedExecutionCost: 0,
      spreadCostProxyVsObserved: 1,
      depthCostProxyVsObserved: 1,
      driftPenaltyProxyVsObserved: 1,
      recalibratedNetPerCycle: 0,
      deltaNetFromRecalibration: 0,
      strongestMarketsAfterRecalibration: [],
      executionCostRecalibrationVerdict: "insufficient_truth_series_for_recalibration",
      executionCostRecalibrationSummaryLine: line,
      proxyModelHarshnessVsObserved: "broadly_correct",
      marketsNearZeroAfterRecalibration: 0,
      marketsPositiveAfterRecalibration: 0,
      marketRecalibrationComparisonCompact: [],
      anyMarketMeaningfullyCloserToZeroAfterRecalibration: false,
      anyMarketDeservesFinalPaperFillValidation: false,
      autonomousPolymarketPathClosureAssessment: "economically_unsupported",
      finalRecommendationLine:
        "Encerrar caminho autónomo até haver store temporal suficiente; sem mercado para validação paper-fill final.",
      computedAt: new Date().toISOString(),
    };
  }

  const proxyExecutionCost = r6(mean(perMarket.map(x => x.proxyDrag)));
  const recalibratedExecutionCost = r6(mean(perMarket.map(x => x.recDrag)));
  const spreadCostProxyVsObserved = r6(
    mean(perMarket.map(x => (x.grossP > 1e-12 ? x.grossO / x.grossP : 1))),
  );
  const depthCostProxyVsObserved = r6(mean(perMarket.map(x => x.depthRatio)));
  const driftPenaltyProxyVsObserved = r6(mean(perMarket.map(x => x.driftRatio)));

  const deltas = perMarket.map(x => r6(x.netR - x.netP));
  const deltaNetFromRecalibration = r6(mean(deltas));
  const recalibratedNetPerCycle = r6(mean(perMarket.map(x => x.netR)));

  const nearZero = perMarket.filter(x => Math.abs(x.netR) < NEAR_ZERO_ABS).length;
  const positive = perMarket.filter(x => x.netR > 0).length;

  const meanDelta = deltaNetFromRecalibration;
  const medDelta = median(deltas);
  const posFrac = positive / n;

  let executionCostRecalibrationVerdict: ExecutionCostRecalibrationVerdict;
  if (meanDelta > 0.0012 && (posFrac >= 0.35 || medDelta > 0.0008)) {
    executionCostRecalibrationVerdict = "recalibration_materially_more_positive_than_proxy";
  } else if (meanDelta < -0.0012 || medDelta < -0.001) {
    executionCostRecalibrationVerdict = "recalibration_materially_more_negative_than_proxy";
  } else {
    executionCostRecalibrationVerdict = "recalibration_neutral_vs_proxy";
  }

  let proxyModelHarshnessVsObserved: ProxyModelHarshnessVsObserved;
  if (medDelta > 0.002) proxyModelHarshnessVsObserved = "materially_too_harsh";
  else if (medDelta < -0.002) proxyModelHarshnessVsObserved = "materially_too_lenient";
  else proxyModelHarshnessVsObserved = "broadly_correct";

  const strongestMarketsAfterRecalibration: StrongestMarketAfterRecalibration[] = [...perMarket]
    .sort((a, b) => b.netR - a.netR)
    .slice(0, 8)
    .map(x => ({
      marketId: String(x.m.id),
      marketTitle: x.m.question.length > 120 ? `${x.m.question.slice(0, 117)}…` : x.m.question,
      snapshotsUsed: x.row.snapshots.length,
      estimatedNetPerCycleProxy: x.netP,
      recalibratedNetPerCycle: x.netR,
      deltaNetFromRecalibration: r6(x.netR - x.netP),
    }));

  const titleShort = (q: string) => (q.length > 120 ? `${q.slice(0, 117)}…` : q);
  const baseRows = perMarket.map(x => {
    const marketId = String(x.m.id);
    const delta = r6(x.netR - x.netP);
    return {
      marketId,
      marketTitle: titleShort(x.m.question),
      estimatedNetPerCycle: x.netP,
      recalibratedNetPerCycle: x.netR,
      deltaNetFromRecalibration: delta,
    };
  });

  const sortedByRec = [...baseRows].sort(
    (a, b) =>
      b.recalibratedNetPerCycle - a.recalibratedNetPerCycle || a.marketId.localeCompare(b.marketId),
  );
  const rankByRec = new Map<string, number>();
  sortedByRec.forEach((row, i) => rankByRec.set(row.marketId, i + 1));

  const sortedByDelta = [...baseRows].sort(
    (a, b) =>
      b.deltaNetFromRecalibration - a.deltaNetFromRecalibration || a.marketId.localeCompare(b.marketId),
  );
  const rankByDelta = new Map<string, number>();
  sortedByDelta.forEach((row, i) => rankByDelta.set(row.marketId, i + 1));

  const marketRecalibrationComparisonCompact: MarketRecalibrationComparisonCompact[] = sortedByRec.map(row => ({
    ...row,
    rankByRecalibratedNet: rankByRec.get(row.marketId)!,
    rankByDeltaNet: rankByDelta.get(row.marketId)!,
  }));

  const anyMarketMeaningfullyCloserToZeroAfterRecalibration = perMarket.some(x => {
    if (!(x.netP < 0)) return false;
    if (!(x.netR > x.netP)) return false;
    return Math.abs(x.netP) - Math.abs(x.netR) >= MEANINGFUL_CLOSER_TO_ZERO_ABS;
  });

  const anyMarketDeservesFinalPaperFillValidation = marketRecalibrationComparisonCompact.some(
    row =>
      row.rankByRecalibratedNet <= PAPER_FILL_CANDIDATE_TOP_REC_RANK &&
      row.recalibratedNetPerCycle >= -PAPER_FILL_CANDIDATE_MAX_ABS_REC &&
      row.deltaNetFromRecalibration >= PAPER_FILL_CANDIDATE_MIN_DELTA,
  );

  const autonomousPolymarketPathClosureAssessment: AutonomousPolymarketPathClosureAssessment =
    anyMarketDeservesFinalPaperFillValidation ? "one_market_warrants_final_validation" : "economically_unsupported";

  const finalRecommendationLine =
    autonomousPolymarketPathClosureAssessment === "one_market_warrants_final_validation"
      ? "Permitir apenas uma validação paper-fill observacional no melhor mercado por net recalibrado (top rank); caso contrário encerrar expansão autónoma."
      : "Encerrar o caminho Polymarket autónomo como economicamente não suportado pelos nets recalibrados; não abrir novos ramos até evidência de fill.";

  const executionCostRecalibrationSummaryLine = `execution_cost_recalibration: verdict=${executionCostRecalibrationVerdict} | analyzed=${n} skipped=${skipped} mean_delta_net=${deltaNetFromRecalibration} mean_rec_net=${recalibratedNetPerCycle} spread_gross_ratio=${spreadCostProxyVsObserved} depth_ratio=${depthCostProxyVsObserved} drift_ratio=${driftPenaltyProxyVsObserved} harshness=${proxyModelHarshnessVsObserved} near_zero=${nearZero} positive=${positive} closer_zero=${anyMarketMeaningfullyCloserToZeroAfterRecalibration} paper_fill=${anyMarketDeservesFinalPaperFillValidation} closure=${autonomousPolymarketPathClosureAssessment}`;

  return {
    probeVersion: "execution-cost-recalibration-v1",
    readDisclaimer:
      "Só leitura: comparação entre proxy instantâneo (spread/liquidez normalizados) e médias temporais do store market-data-truth + ajuste de inventário por depth observada e aditivo de drift a partir de mid. Não altera veredictos de viabilidade.",
    marketsAnalyzed: n,
    marketsSkippedShortSeries: skipped,
    proxyExecutionCost,
    recalibratedExecutionCost,
    spreadCostProxyVsObserved,
    depthCostProxyVsObserved,
    driftPenaltyProxyVsObserved,
    recalibratedNetPerCycle,
    deltaNetFromRecalibration,
    strongestMarketsAfterRecalibration,
    executionCostRecalibrationVerdict,
    executionCostRecalibrationSummaryLine,
    proxyModelHarshnessVsObserved,
    marketsNearZeroAfterRecalibration: nearZero,
    marketsPositiveAfterRecalibration: positive,
    marketRecalibrationComparisonCompact,
    anyMarketMeaningfullyCloserToZeroAfterRecalibration,
    anyMarketDeservesFinalPaperFillValidation,
    autonomousPolymarketPathClosureAssessment,
    finalRecommendationLine,
    computedAt: new Date().toISOString(),
  };
}
