/**
 * Validação observacional final (paper-fill) — um único mercado (629558).
 * Sem trading live, sem expansão, sem novas famílias. Só leitura + modelo numérico.
 */

import { loadMarketDataTruthStore, type MarketTruthSeriesDisk } from "./marketDataTruthCapture";
import {
  adverseSelectionAtSpread,
  estimatedNetFromState,
  estimatedNetPerCycle,
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

const TARGET_MARKET_ID = "629558";
const MIN_SNAPSHOTS = 12;
/** Alinhado a marginal_edge do engine; não altera gates globais. */
const SURVIVES_MIN_NET = 0.0008;
const FAILS_MAX_NET = -0.002;

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

function seriesStats(row: MarketTruthSeriesDisk): {
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

function recalibratedNet(m: NormalizedMarket, row: MarketTruthSeriesDisk): number {
  const sP = observedSpread(m);
  const dP = observedDepth(m);
  const fillP = fillPlausibilityAtSpread(m, sP);
  const advP = adverseSelectionAtSpread(m, sP);
  const invP = inventoryRiskAtSpread(m, sP);
  const unwP = unwindCostAtSpread(m, sP);
  const fee = feeProxy();
  const { sMean, dMean, midDriftAddon } = seriesStats(row);
  const sO = sMean;
  const dObs = dMean > 0 ? dMean : dP;
  const fillO = fillPlausibilityAtSpread(m, sO);
  const advR = r6(adverseSelectionAtSpread(m, sO) + midDriftAddon);
  const liq = depthLiquidityScale(dP, dObs);
  const invR = r6(inventoryRiskAtSpread(m, sO) * liq);
  const unwR = r6(unwindCostAtSpread(m, sO) * liq);
  return estimatedNetFromState(sO, fillO, advR, invR, unwR, fee);
}

export type ObservationalPaperFillVerdict =
  | "survives_final_validation"
  | "fails_final_validation"
  | "inconclusive_but_not_promotable";

export type FinalPaperFillClosureDecision = "continue_to_narrow_paper_testing" | "close_autonomous_polymarket_path";

export interface FinalPaperFillValidation629558Digest {
  probeVersion: "final-paper-fill-validation-629558-v1";
  readDisclaimer: string;
  marketId: string;
  marketTitle: string;
  proxyEstimatedNetPerCycle: number;
  recalibratedNetPerCycle: number;
  observationalPaperFillVerdict: ObservationalPaperFillVerdict;
  assumedEntrySpread: number;
  assumedExitSpread: number;
  fillPlausibilityUsed: number;
  slippageUsed: number;
  adverseMoveObservedOrEstimated: number;
  estimatedNetAfterObservationalFill: number;
  closureDecision: FinalPaperFillClosureDecision;
  finalValidationSummaryLine: string;
  snapshotsUsed: number;
  computedAt: string;
}

export async function buildFinalPaperFillValidation629558Digest(): Promise<FinalPaperFillValidation629558Digest> {
  const store = await loadMarketDataTruthStore();
  const row = store.markets[TARGET_MARKET_ID];
  const all = getAllMarkets();
  let m: NormalizedMarket | undefined = all.find(x => String(x.id) === TARGET_MARKET_ID);
  if (!m) {
    m = (await fetchNormalizedMarketById(TARGET_MARKET_ID)) ?? undefined;
  }

  if (!m || !row || row.snapshots.length < MIN_SNAPSHOTS) {
    const verdict: ObservationalPaperFillVerdict = "inconclusive_but_not_promotable";
    const closureDecision: FinalPaperFillClosureDecision = "close_autonomous_polymarket_path";
    const line = `final_paper_fill_629558: verdict=${verdict} net_after=n/a closure=${closureDecision} reason=missing_market_or_short_series`;
    return {
      probeVersion: "final-paper-fill-validation-629558-v1",
      readDisclaimer:
        "Observacional apenas: mercado 629558, séries market-data-truth em disco, sem ordens reais. Inconclusivo => não promovível e encerramento conservador do caminho autónomo.",
      marketId: TARGET_MARKET_ID,
      marketTitle: m?.question ?? "(unresolved)",
      proxyEstimatedNetPerCycle: m ? estimatedNetPerCycle(m) : 0,
      recalibratedNetPerCycle: m && row && row.snapshots.length >= MIN_SNAPSHOTS ? recalibratedNet(m, row) : 0,
      observationalPaperFillVerdict: verdict,
      assumedEntrySpread: 0,
      assumedExitSpread: 0,
      fillPlausibilityUsed: 0,
      slippageUsed: 0,
      adverseMoveObservedOrEstimated: 0,
      estimatedNetAfterObservationalFill: 0,
      closureDecision,
      finalValidationSummaryLine: line,
      snapshotsUsed: row?.snapshots.length ?? 0,
      computedAt: new Date().toISOString(),
    };
  }

  const spreads = row.snapshots.map(s => s.spread).filter(x => Number.isFinite(x));
  const mids = row.snapshots.map(s => s.mid).filter(x => Number.isFinite(x));
  const { sMean, dMean, midDriftAddon } = seriesStats(row);

  const assumedEntrySpread = r6(Math.min(0.48, Math.max(0.0001, mean(spreads))));
  const spreadVol = spreads.length >= 2 ? stdev(spreads) : 0;
  const assumedExitSpread = r6(Math.min(0.48, assumedEntrySpread + spreadVol));

  const sRound = r6((assumedEntrySpread + assumedExitSpread) / 2);
  const fillPlausibilityUsed = fillPlausibilityAtSpread(m, sRound);
  const dP = observedDepth(m);
  const dObs = dMean > 0 ? dMean : dP;
  const liq = depthLiquidityScale(dP, dObs);

  const spreadSlip = r6(0.13 * fillPlausibilityUsed * Math.max(0, assumedExitSpread - assumedEntrySpread));
  const midSlip = r6((mids.length >= 2 ? stdev(mids) : 0) * 0.35);
  const slippageUsed = r6(spreadSlip + midSlip);

  const adverseMoveObservedOrEstimated = midDriftAddon;

  const gross = r6(sRound * 0.13 * fillPlausibilityUsed);
  const adv = r6(adverseSelectionAtSpread(m, sRound) + adverseMoveObservedOrEstimated);
  const inv = r6(inventoryRiskAtSpread(m, sRound) * liq);
  const unw = r6(unwindCostAtSpread(m, sRound) * liq);
  const fee = feeProxy();
  const estimatedNetAfterObservationalFill = r6(gross - adv - inv - unw - fee - slippageUsed);

  let observationalPaperFillVerdict: ObservationalPaperFillVerdict;
  if (estimatedNetAfterObservationalFill >= SURVIVES_MIN_NET) {
    observationalPaperFillVerdict = "survives_final_validation";
  } else if (estimatedNetAfterObservationalFill <= FAILS_MAX_NET) {
    observationalPaperFillVerdict = "fails_final_validation";
  } else {
    observationalPaperFillVerdict = "inconclusive_but_not_promotable";
  }

  const closureDecision: FinalPaperFillClosureDecision =
    observationalPaperFillVerdict === "survives_final_validation"
      ? "continue_to_narrow_paper_testing"
      : "close_autonomous_polymarket_path";

  const proxyEstimatedNetPerCycle = estimatedNetPerCycle(m);
  const recalibratedNetPerCycle = recalibratedNet(m, row);

  const finalValidationSummaryLine = `final_paper_fill_629558: verdict=${observationalPaperFillVerdict} net_after=${estimatedNetAfterObservationalFill} proxy_net=${proxyEstimatedNetPerCycle} rec_net=${recalibratedNetPerCycle} entry_spread=${assumedEntrySpread} exit_spread=${assumedExitSpread} fill=${fillPlausibilityUsed} slip=${slippageUsed} adverse_obs=${adverseMoveObservedOrEstimated} closure=${closureDecision} snaps=${row.snapshots.length}`;

  return {
    probeVersion: "final-paper-fill-validation-629558-v1",
    readDisclaimer:
      "Último teste discriminatório observacional para 629558 apenas: entradas/saídas derivadas da série temporal em disco; slippage composto por alargamento de spread + volatilidade de mid. Sem live trading. Inconclusivo => não promovível.",
    marketId: TARGET_MARKET_ID,
    marketTitle: m.question.length > 160 ? `${m.question.slice(0, 157)}…` : m.question,
    proxyEstimatedNetPerCycle,
    recalibratedNetPerCycle,
    observationalPaperFillVerdict,
    assumedEntrySpread,
    assumedExitSpread,
    fillPlausibilityUsed,
    slippageUsed,
    adverseMoveObservedOrEstimated,
    estimatedNetAfterObservationalFill,
    closureDecision,
    finalValidationSummaryLine,
    snapshotsUsed: row.snapshots.length,
    computedAt: new Date().toISOString(),
  };
}
