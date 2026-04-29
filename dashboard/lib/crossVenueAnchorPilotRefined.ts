/**
 * Cross-Venue Anchor Pilot — refinamento narrow (2 mercados).
 * Mesma classe estratégica; fair alinhado ao tipo de resolução Binance “1m HIGH ≥ strike no mês”;
 * custos menos proxy (fees taker crypto Polymarket + hedge venue + spread).
 */

import {
  fetchGammaMarketRawJson,
  fetchParsedClobBook,
  resolveYesClobTokenId,
} from "./clobMicrostructure";

export type ExternalAnchorTypeKey =
  | "coinbase_executable_spot_btc_usd"
  | "coinbase_executable_spot_eth_usd";

export type RefinedPilotVerdictPerMarket =
  | "refined_positive"
  | "refined_weak"
  | "refined_not_viable";

export type RefinedCrossVenueAnchorVerdict =
  | "no_viable_refined_anchor_market"
  | "weak_refined_anchor_candidate"
  | "one_viable_refined_anchor_candidate"
  | "multiple_viable_refined_anchor_candidates";

export interface RefinedCrossVenueAnchorMarketRow {
  marketId: string;
  marketTitle: string;
  refinedFairValue: number;
  anchorPriceObserved: number;
  polymarketPriceObserved: number;
  rawAnchorGap: number;
  estimatedEntryCost: number;
  estimatedExitCost: number;
  estimatedHedgeCost: number;
  estimatedLegRiskCost: number;
  estimatedNetAnchorCycle: number;
  refinedPilotVerdictPerMarket: RefinedPilotVerdictPerMarket;
  supportingNote: string;
}

export interface StrongestRefinedAnchorMarket {
  marketId: string;
  marketTitle: string;
  estimatedNetAnchorCycle: number;
}

export interface RefinedCrossVenueAnchorPilotDigest {
  probeVersion: "cross-venue-anchor-pilot-refined-v1";
  readDisclaimer: string;
  refinedCrossVenueAnchorVerdict: RefinedCrossVenueAnchorVerdict;
  marketsEvaluated: number;
  marketsWithPositiveNetAnchorCycle: number;
  strongestAnchorMarkets: StrongestRefinedAnchorMarket[];
  refinedCrossVenueAnchorSummaryLine: string;
  markets: RefinedCrossVenueAnchorMarketRow[];
  computedAt: string;
}

/** Universo mínimo: apenas os dois mercados mais fortes no piloto v1. */
const REFINED_UNIVERSE: Array<{
  marketId: string;
  marketTitle: string;
  strikeUsd: number;
  asset: "btc" | "eth";
  anchor: ExternalAnchorTypeKey;
}> = [
  {
    marketId: "1823775",
    marketTitle: "Will Bitcoin reach $85,000 in April?",
    strikeUsd: 85_000,
    asset: "btc",
    anchor: "coinbase_executable_spot_btc_usd",
  },
  {
    marketId: "1823789",
    marketTitle: "Will Ethereum reach $4,000 in April?",
    strikeUsd: 4000,
    asset: "eth",
    anchor: "coinbase_executable_spot_eth_usd",
  },
];

/** Taker fee fraction aplicada ao preço YES (crypto schedule proxy). */
const PM_TAKER_FEE_FRAC = 0.0072;

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function num(x: unknown): number {
  const n = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(n) ? n : NaN;
}

/**
 * Fair YES ~ alinhamento “touch high” Binance-style (proxy com spot executável Coinbase).
 * - Se spot ≥ strike: evento HIGH≥K pode já ter ocorrido no mês → YES muito provável.
 * - Se spot < strike: decay com shortfall relativo (mais conservador no rabo que logistic v1).
 */
function refinedReachFairBinanceStyle(
  spotUsd: number,
  strikeUsd: number,
  asset: "btc" | "eth",
): number {
  if (!(spotUsd > 0 && strikeUsd > 0)) return 0.5;
  const ratio = spotUsd / strikeUsd;

  if (ratio >= 1) {
    const slack = ratio - 1;
    const lambda = asset === "btc" ? 32 : 28;
    return r6(Math.min(0.96, Math.max(0.85, 0.85 + slack * lambda)));
  }

  const shortfall = (strikeUsd - spotUsd) / strikeUsd;
  const k = asset === "btc" ? 38 : 42;
  const p = 0.03 + 0.85 * Math.exp(-shortfall * k);
  return r6(Math.min(0.91, Math.max(0.03, p)));
}

async function fetchCoinbaseSpot(pair: "BTC-USD" | "ETH-USD"): Promise<number | null> {
  try {
    const res = await fetch(`https://api.coinbase.com/v2/prices/${pair}/spot`, {
      signal: AbortSignal.timeout(8000),
      headers: { Accept: "application/json" },
    });
    const j = (await res.json()) as { data?: { amount?: string } };
    const v = num(j.data?.amount);
    return Number.isFinite(v) ? v : null;
  } catch {
    return null;
  }
}

function hedgeCostFor(asset: "btc" | "eth"): number {
  return asset === "btc" ? 0.0043 : 0.0055;
}

type RefinedUniverseMeta = (typeof REFINED_UNIVERSE)[number];

async function computeRefinedMarketRow(
  meta: RefinedUniverseMeta,
  anchorPx: number | null,
): Promise<RefinedCrossVenueAnchorMarketRow> {
  const raw = await fetchGammaMarketRawJson(meta.marketId);
  const yesToken = raw ? resolveYesClobTokenId(raw) : "";
  const book = yesToken ? await fetchParsedClobBook(yesToken) : null;

  if (!Number.isFinite(anchorPx ?? NaN) || !book) {
    return {
      marketId: meta.marketId,
      marketTitle: meta.marketTitle,
      refinedFairValue: 0,
      anchorPriceObserved: Number.isFinite(anchorPx ?? NaN) ? r6(anchorPx!) : 0,
      polymarketPriceObserved: 0,
      rawAnchorGap: 0,
      estimatedEntryCost: 0,
      estimatedExitCost: 0,
      estimatedHedgeCost: 0,
      estimatedLegRiskCost: 0,
      estimatedNetAnchorCycle: 0,
      refinedPilotVerdictPerMarket: "refined_not_viable",
      supportingNote: !Number.isFinite(anchorPx ?? NaN)
        ? "anchor_spot_unavailable"
        : "polymarket_clob_book_unavailable",
    };
  }

  const pmMid = book.mid;
  const refinedFairValue = refinedReachFairBinanceStyle(anchorPx!, meta.strikeUsd, meta.asset);
  const edgeMag = r6(refinedFairValue - pmMid);
  const rawAnchorGap = r6(pmMid - refinedFairValue);

  const half = r6(book.spread / 2);
  const takerHit = r6(PM_TAKER_FEE_FRAC * pmMid);
  const latencySkew = 0.002;

  const estimatedEntryCost = r6(half + takerHit + latencySkew);
  const estimatedExitCost = r6(half + takerHit + latencySkew);
  const estimatedHedgeCost = hedgeCostFor(meta.asset);
  const estimatedLegRiskCost = r6(
    Math.min(0.018, book.spread * 1.35 + 0.045 * Math.abs(pmMid - refinedFairValue)),
  );

  const estimatedNetAnchorCycle = r6(
    edgeMag -
      estimatedEntryCost -
      estimatedExitCost -
      estimatedHedgeCost -
      estimatedLegRiskCost,
  );

  let refinedPilotVerdictPerMarket: RefinedPilotVerdictPerMarket = "refined_not_viable";
  if (estimatedNetAnchorCycle > 0.006) refinedPilotVerdictPerMarket = "refined_positive";
  else if (estimatedNetAnchorCycle > 0) refinedPilotVerdictPerMarket = "refined_weak";

  const supportingNote = [
    `fair_model=binance_touch_high_proxy_month`,
    `strike_usd=${meta.strikeUsd}`,
    `spot_usd=${r6(anchorPx!)}`,
    `refined_fair=${refinedFairValue}`,
    `pm_mid=${r6(pmMid)}`,
    `edge_yes_vs_fair=${edgeMag}`,
    `spread=${r6(book.spread)}`,
    `pm_taker_frac=${PM_TAKER_FEE_FRAC}`,
    `hedge_asset=${meta.asset}`,
  ].join("|");

  return {
    marketId: meta.marketId,
    marketTitle: meta.marketTitle,
    refinedFairValue,
    anchorPriceObserved: r6(anchorPx!),
    polymarketPriceObserved: r6(pmMid),
    rawAnchorGap,
    estimatedEntryCost,
    estimatedExitCost,
    estimatedHedgeCost,
    estimatedLegRiskCost,
    estimatedNetAnchorCycle,
    refinedPilotVerdictPerMarket,
    supportingNote,
  };
}

/** Observação refinada só para ETH reach $4k April — monitor temporal Railway (1823789). */
export type CrossVenueAnchor1823789MonitorObservation = {
  isoTimestamp: string;
  probeVersion: "cross-venue-anchor-1823789-monitor-v1";
  row: RefinedCrossVenueAnchorMarketRow;
};

export async function buildCrossVenueAnchor1823789RefinedObservation(): Promise<CrossVenueAnchor1823789MonitorObservation> {
  const meta = REFINED_UNIVERSE.find(m => m.marketId === "1823789");
  if (!meta) throw new Error("internal: 1823789 missing from refined universe");
  const anchorPx = await fetchCoinbaseSpot("ETH-USD");
  const row = await computeRefinedMarketRow(meta, anchorPx);
  return {
    isoTimestamp: new Date().toISOString(),
    probeVersion: "cross-venue-anchor-1823789-monitor-v1",
    row,
  };
}

export async function buildCrossVenueAnchorPilotRefinedDigest(): Promise<RefinedCrossVenueAnchorPilotDigest> {
  const [btcSpot, ethSpot] = await Promise.all([
    fetchCoinbaseSpot("BTC-USD"),
    fetchCoinbaseSpot("ETH-USD"),
  ]);

  const markets: RefinedCrossVenueAnchorMarketRow[] = [];

  for (const meta of REFINED_UNIVERSE) {
    const anchorPx = meta.asset === "btc" ? btcSpot : ethSpot;
    markets.push(await computeRefinedMarketRow(meta, anchorPx));
  }

  const marketsEvaluated = markets.length;
  const marketsWithPositiveNetAnchorCycle = markets.filter(m => m.estimatedNetAnchorCycle > 0).length;
  const strong = markets.filter(m => m.refinedPilotVerdictPerMarket === "refined_positive").length;
  const weakOnly =
    markets.filter(m => m.refinedPilotVerdictPerMarket === "refined_weak").length > 0 &&
    strong === 0 &&
    marketsWithPositiveNetAnchorCycle > 0;

  let refinedCrossVenueAnchorVerdict: RefinedCrossVenueAnchorVerdict =
    "no_viable_refined_anchor_market";
  if (strong >= 2) refinedCrossVenueAnchorVerdict = "multiple_viable_refined_anchor_candidates";
  else if (strong === 1) refinedCrossVenueAnchorVerdict = "one_viable_refined_anchor_candidate";
  else if (weakOnly || (strong === 0 && marketsWithPositiveNetAnchorCycle >= 1)) {
    refinedCrossVenueAnchorVerdict = "weak_refined_anchor_candidate";
  }

  const strongestAnchorMarkets: StrongestRefinedAnchorMarket[] = [...markets]
    .sort((a, b) => b.estimatedNetAnchorCycle - a.estimatedNetAnchorCycle)
    .map(m => ({
      marketId: m.marketId,
      marketTitle: m.marketTitle,
      estimatedNetAnchorCycle: m.estimatedNetAnchorCycle,
    }));

  const refinedCrossVenueAnchorSummaryLine = `cross_venue_anchor_refined_v1: verdict=${refinedCrossVenueAnchorVerdict} | mkts=${marketsEvaluated} | pos_net=${marketsWithPositiveNetAnchorCycle}`;

  return {
    probeVersion: "cross-venue-anchor-pilot-refined-v1",
    readDisclaimer:
      "Refinamento robot-only (v1): só mercados 1823775 e 1823789. Fair proxy Binance-style touch HIGH≥strike no mês; âncora Coinbase spot para hedge executável; custos com fee taker PM estimada + hedge BTC/ETH + spread. Sem alargamento de universo.",
    refinedCrossVenueAnchorVerdict,
    marketsEvaluated,
    marketsWithPositiveNetAnchorCycle,
    strongestAnchorMarkets,
    refinedCrossVenueAnchorSummaryLine,
    markets,
    computedAt: new Date().toISOString(),
  };
}
