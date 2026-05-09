/**
 * Reward-Aware Maker Pilot — âncora externa (Gamma bestBid/bestAsk), recompensa proxy,
 * inventário e hedge modelados. Universo mínimo; sem trading live.
 */

import { extractGammaBestBidAsk, fetchGammaMarketRawJson } from "./clobMicrostructure";
import {
  adverseSelectionAtSpread,
  feeProxy,
  fillPlausibilityAtSpread,
  inventoryRiskAtSpread,
  observedSpread,
  unwindCostAtSpread,
  estimatedNetPerCycle,
  isMachineObserved,
  isRobotQuoteableGate,
} from "./executionTruthEngine";
import { getAllMarkets } from "./marketDataService";
import { getMarketDataRuntime } from "./nodeProcessRuntimeState";
import type { NormalizedMarket } from "./polymarketClient";

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

const MAX_PILOT_MARKETS = 6;
const VIABLE_NET = 0.002;
const GAP_MS = 100;

export type PilotVerdictPerMarket = "positive_expected" | "marginal" | "not_viable";

export type RewardAwareMakerPilotVerdict =
  | "no_viable_reward_aware_maker_market"
  | "weak_reward_aware_maker_candidate"
  | "one_viable_reward_aware_maker_candidate"
  | "multiple_viable_reward_aware_maker_candidates";

export interface RewardAwareMakerPilotMarketRow {
  marketId: string;
  marketTitle: string;
  externalFairValueAvailable: boolean;
  observedSpread: number;
  quoteWidthCandidate: number;
  makerFillPlausibility: number;
  expectedSpreadCapture: number;
  expectedRewardContribution: number;
  expectedInventoryRiskCost: number;
  expectedHedgeCost: number;
  expectedAdverseSelectionCost: number;
  estimatedNetMakerCycle: number;
  pilotVerdictPerMarket: PilotVerdictPerMarket;
  supportingNote: string;
}

export interface StrongestPilotMarket {
  marketId: string;
  marketTitle: string;
  estimatedNetMakerCycle: number;
  externalFairValueAvailable: boolean;
  pilotVerdictPerMarket: PilotVerdictPerMarket;
}

export interface RewardAwareMakerPilotDigest {
  probeVersion: "reward-aware-maker-pilot-v1";
  readDisclaimer: string;
  rewardAwareMakerPilotVerdict: RewardAwareMakerPilotVerdict;
  marketsEvaluated: number;
  marketsWithExternalAnchor: number;
  marketsWithPositiveNetCycle: number;
  strongestPilotMarkets: StrongestPilotMarket[];
  rewardAwareMakerPilotSummaryLine: string;
  markets: RewardAwareMakerPilotMarketRow[];
  computedAt: string;
}

/** Proxy de recompensa de liquidez (sem API de rewards); teto conservador. */
export function rewardProxyPerCycle(m: NormalizedMarket): number {
  if (m.liquidity <= 0) return 0;
  return r6(Math.min(0.0018, m.volume / (m.liquidity * 2_400 + 9_000)));
}

function midFromPrices(m: NormalizedMarket): number {
  if (m.prices.length < 2) return r6(m.prices[0] ?? 0);
  return r6((m.prices[0] + m.prices[1]) / 2);
}

export function pickPilotUniverse(): NormalizedMarket[] {
  const all = getAllMarkets();
  const quoteable = [...all]
    .filter(isRobotQuoteableGate)
    .sort((a, b) => estimatedNetPerCycle(b) - estimatedNetPerCycle(a))
    .slice(0, MAX_PILOT_MARKETS);
  if (quoteable.length > 0) return quoteable;
  return [...all]
    .filter(isMachineObserved)
    .sort((a, b) => estimatedNetPerCycle(b) - estimatedNetPerCycle(a))
    .slice(0, MAX_PILOT_MARKETS);
}

export async function evaluatePilotMarket(m: NormalizedMarket): Promise<RewardAwareMakerPilotMarketRow> {
  const raw = await fetchGammaMarketRawJson(m.id);
  const gbb = raw ? extractGammaBestBidAsk(raw) : null;
  const externalFairValueAvailable = !!gbb;
  const extSpread = gbb ? r6(gbb.ask - gbb.bid) : observedSpread(m);
  const obs = observedSpread(m);
  const quoteWidthCandidate = r6(
    Math.max(0.0024, Math.min(extSpread * 0.4, obs * 0.48, 0.06)),
  );
  const effectiveSpreadForCosts = r6(Math.min(0.48, quoteWidthCandidate * 2.1));
  const makerFillPlausibility = fillPlausibilityAtSpread(m, effectiveSpreadForCosts);
  const expectedSpreadCapture = r6(quoteWidthCandidate * makerFillPlausibility * 0.58);
  const expectedRewardContribution = rewardProxyPerCycle(m);
  const expectedInventoryRiskCost = r6(inventoryRiskAtSpread(m, effectiveSpreadForCosts) * 0.82);
  const expectedHedgeCost = r6(unwindCostAtSpread(m, effectiveSpreadForCosts) * 0.28 + feeProxy() * 0.35);
  const expectedAdverseSelectionCost = adverseSelectionAtSpread(m, effectiveSpreadForCosts);
  const fee = feeProxy();
  const estimatedNetMakerCycle = r6(
    expectedSpreadCapture +
      expectedRewardContribution -
      expectedInventoryRiskCost -
      expectedHedgeCost -
      expectedAdverseSelectionCost -
      fee,
  );

  let pilotVerdictPerMarket: PilotVerdictPerMarket;
  if (estimatedNetMakerCycle >= VIABLE_NET) pilotVerdictPerMarket = "positive_expected";
  else if (estimatedNetMakerCycle > 0) pilotVerdictPerMarket = "marginal";
  else pilotVerdictPerMarket = "not_viable";

  const fairHint = gbb
    ? `gamma_bbo_anchor|fair_mid=${r6((gbb.bid + gbb.ask) / 2)}|ext_spread=${extSpread}|outcome_mid=${midFromPrices(m)}`
    : `no_gamma_bbo|outcome_mid=${midFromPrices(m)}|spread_proxy=${obs}`;

  const supportingNote = `${fairHint}|quote_w=${quoteWidthCandidate}|reward_proxy`;

  const marketTitle = m.question.length > 140 ? `${m.question.slice(0, 137)}…` : m.question;

  return {
    marketId: String(m.id),
    marketTitle,
    externalFairValueAvailable,
    observedSpread: obs,
    quoteWidthCandidate,
    makerFillPlausibility,
    expectedSpreadCapture,
    expectedRewardContribution,
    expectedInventoryRiskCost,
    expectedHedgeCost,
    expectedAdverseSelectionCost,
    estimatedNetMakerCycle,
    pilotVerdictPerMarket,
    supportingNote,
  };
}

export async function waitForCatalogSnapshot(maxMs = 35_000): Promise<void> {
  getAllMarkets();
  const t0 = Date.now();
  while (Date.now() - t0 < maxMs) {
    const st = getMarketDataRuntime();
    if (st.markets.length >= 200) return;
    if (st.bootstrapCompleted && st.markets.length > 0) return;
    if (st.bootstrapFailed) return;
    await new Promise(r => setTimeout(r, 350));
  }
}

export async function buildRewardAwareMakerPilotDigest(): Promise<RewardAwareMakerPilotDigest> {
  await waitForCatalogSnapshot();
  const candidates = pickPilotUniverse();
  const markets: RewardAwareMakerPilotMarketRow[] = [];

  for (let i = 0; i < candidates.length; i++) {
    markets.push(await evaluatePilotMarket(candidates[i]));
    if (i < candidates.length - 1) {
      await new Promise(r => setTimeout(r, GAP_MS));
    }
  }

  const marketsEvaluated = markets.length;
  const marketsWithExternalAnchor = markets.filter(m => m.externalFairValueAvailable).length;
  const marketsWithPositiveNetCycle = markets.filter(m => m.estimatedNetMakerCycle > 0).length;
  const viableCount = markets.filter(m => m.estimatedNetMakerCycle >= VIABLE_NET).length;
  const marginalCount = markets.filter(m => m.pilotVerdictPerMarket === "marginal").length;

  let rewardAwareMakerPilotVerdict: RewardAwareMakerPilotVerdict;
  if (marketsEvaluated === 0) {
    rewardAwareMakerPilotVerdict = "no_viable_reward_aware_maker_market";
  } else if (viableCount >= 2) {
    rewardAwareMakerPilotVerdict = "multiple_viable_reward_aware_maker_candidates";
  } else if (viableCount === 1) {
    rewardAwareMakerPilotVerdict = "one_viable_reward_aware_maker_candidate";
  } else if (marginalCount >= 1 || marketsWithPositiveNetCycle >= 1) {
    rewardAwareMakerPilotVerdict = "weak_reward_aware_maker_candidate";
  } else {
    rewardAwareMakerPilotVerdict = "no_viable_reward_aware_maker_market";
  }

  const strongestPilotMarkets: StrongestPilotMarket[] = [...markets]
    .sort((a, b) => b.estimatedNetMakerCycle - a.estimatedNetMakerCycle)
    .slice(0, 4)
    .map(m => ({
      marketId: m.marketId,
      marketTitle: m.marketTitle,
      estimatedNetMakerCycle: m.estimatedNetMakerCycle,
      externalFairValueAvailable: m.externalFairValueAvailable,
      pilotVerdictPerMarket: m.pilotVerdictPerMarket,
    }));

  const rewardAwareMakerPilotSummaryLine = `reward_aware_maker_pilot: verdict=${rewardAwareMakerPilotVerdict} | evaluated=${marketsEvaluated} ext_anchor=${marketsWithExternalAnchor} pos_net=${marketsWithPositiveNetCycle} viable_ge_${VIABLE_NET}=${viableCount} | top_net=${strongestPilotMarkets[0]?.estimatedNetMakerCycle ?? "n/a"}`;

  return {
    probeVersion: "reward-aware-maker-pilot-v1",
    readDisclaimer:
      "Pilot maker com âncora externa Gamma (bestBid/bestAsk quando existente), custos de inventário/adversário/hedge derivados do executionTruthEngine com spread efetivo apertado, recompensa apenas como proxy volume/liquidez (sem API de incentives). Universo = até 6 mercados robot-quoteable; se vazio, fallback máquina-observado (ainda ≤6). Sem ordens live.",
    rewardAwareMakerPilotVerdict,
    marketsEvaluated,
    marketsWithExternalAnchor,
    marketsWithPositiveNetCycle,
    strongestPilotMarkets,
    rewardAwareMakerPilotSummaryLine,
    markets,
    computedAt: new Date().toISOString(),
  };
}
