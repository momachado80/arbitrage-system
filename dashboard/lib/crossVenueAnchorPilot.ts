/**
 * Cross-Venue Anchor / Hedge Pilot — robot-only, universo mínimo fixo.
 * Compara mid YES Polymarket (CLOB) vs fair elementar derivado de spot executável Coinbase
 * e strike explícito por mercado (âncora externa limpa). Sem scan alargado.
 */

import {
  fetchGammaMarketRawJson,
  fetchParsedClobBook,
  parseClobTokenIds,
} from "./clobMicrostructure";

export type ExternalAnchorTypeKey =
  | "coinbase_executable_spot_btc_usd"
  | "coinbase_executable_spot_eth_usd";

export type PilotVerdictPerMarket = "anchor_edge_positive" | "anchor_edge_weak" | "anchor_not_viable";

export type CrossVenueAnchorVerdict =
  | "no_viable_anchor_market"
  | "weak_anchor_candidate_only"
  | "one_viable_anchor_candidate"
  | "multiple_viable_anchor_candidates";

export interface CrossVenueAnchorMarketRow {
  marketId: string;
  marketTitle: string;
  externalAnchorType: ExternalAnchorTypeKey | string;
  anchorPriceObserved: number;
  polymarketPriceObserved: number;
  rawAnchorGap: number;
  estimatedEntryCost: number;
  estimatedExitCost: number;
  estimatedHedgeCost: number;
  estimatedLegRiskCost: number;
  estimatedNetAnchorCycle: number;
  pilotVerdictPerMarket: PilotVerdictPerMarket;
  supportingNote: string;
}

export interface StrongestAnchorMarket {
  marketId: string;
  marketTitle: string;
  estimatedNetAnchorCycle: number;
}

export interface CrossVenueAnchorPilotDigest {
  probeVersion: "cross-venue-anchor-pilot-v1";
  readDisclaimer: string;
  crossVenueAnchorVerdict: CrossVenueAnchorVerdict;
  marketsEvaluated: number;
  marketsWithValidAnchor: number;
  marketsWithPositiveNetAnchorCycle: number;
  strongestAnchorMarkets: StrongestAnchorMarket[];
  crossVenueAnchorSummaryLine: string;
  markets: CrossVenueAnchorMarketRow[];
  computedAt: string;
}

/** Universo fixo (4 mercados): BTC reach April ×3 + ETH reach April ×1 — mapeamento strike limpo. */
const PILOT_MARKETS: Array<{
  marketId: string;
  marketTitle: string;
  strikeUsd: number;
  asset: "btc" | "eth";
  anchor: ExternalAnchorTypeKey;
}> = [
  {
    marketId: "1823776",
    marketTitle: "Will Bitcoin reach $80,000 in April?",
    strikeUsd: 80_000,
    asset: "btc",
    anchor: "coinbase_executable_spot_btc_usd",
  },
  {
    marketId: "1823775",
    marketTitle: "Will Bitcoin reach $85,000 in April?",
    strikeUsd: 85_000,
    asset: "btc",
    anchor: "coinbase_executable_spot_btc_usd",
  },
  {
    marketId: "1823774",
    marketTitle: "Will Bitcoin reach $90,000 in April?",
    strikeUsd: 90_000,
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

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function num(x: unknown): number {
  const n = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(n) ? n : NaN;
}

/** Fair YES (0–1) elementar para “touch / reach strike” antes do fim do período — logística sobre spot/strike. */
function crudeReachFair01(spotUsd: number, strikeUsd: number): number {
  if (!(spotUsd > 0 && strikeUsd > 0)) return 0.5;
  const r = spotUsd / strikeUsd;
  const x = (r - 1) * 6;
  const p = 1 / (1 + Math.exp(-x));
  return Math.min(0.97, Math.max(0.03, p));
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

export async function buildCrossVenueAnchorPilotDigest(): Promise<CrossVenueAnchorPilotDigest> {
  const [btcSpot, ethSpot] = await Promise.all([
    fetchCoinbaseSpot("BTC-USD"),
    fetchCoinbaseSpot("ETH-USD"),
  ]);

  const markets: CrossVenueAnchorMarketRow[] = [];

  for (const meta of PILOT_MARKETS) {
    const anchorPx = meta.asset === "btc" ? btcSpot : ethSpot;
    const raw = await fetchGammaMarketRawJson(meta.marketId);
    const tokenIds = raw ? parseClobTokenIds(raw) : [];
    const yesToken = tokenIds[0] ?? "";
    const book = yesToken ? await fetchParsedClobBook(yesToken) : null;

    if (!Number.isFinite(anchorPx ?? NaN) || !book) {
      markets.push({
        marketId: meta.marketId,
        marketTitle: meta.marketTitle,
        externalAnchorType: meta.anchor,
        anchorPriceObserved: Number.isFinite(anchorPx ?? NaN) ? r6(anchorPx!) : 0,
        polymarketPriceObserved: 0,
        rawAnchorGap: 0,
        estimatedEntryCost: 0,
        estimatedExitCost: 0,
        estimatedHedgeCost: 0,
        estimatedLegRiskCost: 0,
        estimatedNetAnchorCycle: 0,
        pilotVerdictPerMarket: "anchor_not_viable",
        supportingNote:
          !Number.isFinite(anchorPx ?? NaN)
            ? "anchor_spot_unavailable"
            : "polymarket_clob_book_unavailable",
      });
      continue;
    }

    const pmMid = book.mid;
    const fair01 = crudeReachFair01(anchorPx!, meta.strikeUsd);
    /** Positivo se YES na Polymarket está barato vs fair (comprar YES). */
    const edgeMag = r6(fair01 - pmMid);
    const rawAnchorGap = r6(pmMid - fair01);

    const entryHalf = r6(book.spread / 2);
    const exitHalf = entryHalf;
    const tickProxy = 0.005;
    const estimatedEntryCost = r6(entryHalf + tickProxy);
    const estimatedExitCost = r6(exitHalf + tickProxy);
    const estimatedHedgeCost = 0.004;
    const estimatedLegRiskCost = r6(Math.min(0.012, book.spread * 0.75));

    const estimatedNetAnchorCycle = r6(
      edgeMag - estimatedEntryCost - estimatedExitCost - estimatedHedgeCost - estimatedLegRiskCost,
    );

    let pilotVerdictPerMarket: PilotVerdictPerMarket = "anchor_not_viable";
    if (estimatedNetAnchorCycle > 0.004) pilotVerdictPerMarket = "anchor_edge_positive";
    else if (estimatedNetAnchorCycle > 0) pilotVerdictPerMarket = "anchor_edge_weak";

    const supportingNote = [
      `fair01_model=logistic_spot_strike`,
      `strike_usd=${meta.strikeUsd}`,
      `spot_usd=${r6(anchorPx!)}`,
      `pm_mid=${r6(pmMid)}`,
      `edge_yes_vs_fair=${edgeMag}`,
      `spread=${r6(book.spread)}`,
      `source=coinbase_spot_clob_book`,
    ].join("|");

    markets.push({
      marketId: meta.marketId,
      marketTitle: meta.marketTitle,
      externalAnchorType: meta.anchor,
      anchorPriceObserved: r6(anchorPx!),
      polymarketPriceObserved: r6(pmMid),
      rawAnchorGap,
      estimatedEntryCost,
      estimatedExitCost,
      estimatedHedgeCost,
      estimatedLegRiskCost,
      estimatedNetAnchorCycle,
      pilotVerdictPerMarket,
      supportingNote,
    });
  }

  const marketsEvaluated = markets.length;
  const marketsWithValidAnchor = markets.filter(
    m => m.supportingNote.indexOf("unavailable") === -1,
  ).length;
  const marketsWithPositiveNetAnchorCycle = markets.filter(
    m => m.estimatedNetAnchorCycle > 0,
  ).length;

  const posStrong = markets.filter(m => m.pilotVerdictPerMarket === "anchor_edge_positive").length;
  const posWeak = markets.filter(m => m.pilotVerdictPerMarket === "anchor_edge_weak").length;

  let crossVenueAnchorVerdict: CrossVenueAnchorVerdict = "no_viable_anchor_market";
  if (posStrong >= 2) {
    crossVenueAnchorVerdict = "multiple_viable_anchor_candidates";
  } else if (posStrong === 1) {
    crossVenueAnchorVerdict = "one_viable_anchor_candidate";
  } else if (posStrong === 0 && posWeak >= 1) {
    crossVenueAnchorVerdict = "weak_anchor_candidate_only";
  } else if (marketsWithPositiveNetAnchorCycle >= 1 && posStrong === 0) {
    crossVenueAnchorVerdict = "weak_anchor_candidate_only";
  }

  const strongestAnchorMarkets: StrongestAnchorMarket[] = [...markets]
    .sort((a, b) => b.estimatedNetAnchorCycle - a.estimatedNetAnchorCycle)
    .slice(0, 4)
    .map(m => ({
      marketId: m.marketId,
      marketTitle: m.marketTitle,
      estimatedNetAnchorCycle: m.estimatedNetAnchorCycle,
    }));

  const crossVenueAnchorSummaryLine = `cross_venue_anchor_v1: verdict=${crossVenueAnchorVerdict} | mkts=${marketsEvaluated} | valid_anchor=${marketsWithValidAnchor} | pos_net_cycle=${marketsWithPositiveNetAnchorCycle}`;

  return {
    probeVersion: "cross-venue-anchor-pilot-v1",
    readDisclaimer:
      "Pilot robot-only (v1): âncora = Coinbase spot executável (BTC-USD / ETH-USD); Polymarket = mid CLOB token YES. Fair elementar ~ logística(spot/strike) só para gap observacional; não é modelo de pricing de produção. Custos são proxies (spread+ticks+ hedge leg). Sem scan alargado.",
    crossVenueAnchorVerdict,
    marketsEvaluated,
    marketsWithValidAnchor,
    marketsWithPositiveNetAnchorCycle,
    strongestAnchorMarkets,
    crossVenueAnchorSummaryLine,
    markets,
    computedAt: new Date().toISOString(),
  };
}
