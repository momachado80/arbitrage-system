/**
 * Maker / Inventory Edge Probe — auditoria estrutural económica (sem live quoting nem ordens).
 * Proxies observáveis: spread, liquidez, volume. Optimizado para falsificação rápida.
 */

import type { NormalizedMarket } from "./polymarketClient";
import { getAllMarkets } from "./marketDataService";

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

export type MakerInventoryVerdict =
  | "no_quoteable_markets"
  | "maker_edge_unproven"
  | "weak_maker_candidate_present"
  | "viable_maker_candidate_present";

export interface MakerKillCriteriaStatus {
  marketsLoadedGtZero: boolean;
  evaluatedGtZero: boolean;
  quoteableGtZero: boolean;
  anyEstimatedMakerNetPositive: boolean;
  anyEstimatedMakerNetAboveWeakThreshold: boolean;
  avgSpreadAboveNoiseFloor: boolean;
}

export interface MakerCandidateMarket {
  marketId: string;
  marketTitle: string;
  observedSpread: number;
  topOfBookDepth: number;
  fillPlausibilityMaker: number;
  quoteWidthCandidate: number;
  inventoryRiskProxy: number;
  adverseSelectionRiskProxy: number;
  estimatedMakerGross: number;
  estimatedHedgeCost: number;
  estimatedMakerNet: number;
  supportingNote: string;
}

export interface MakerInventoryProbeDigest {
  probeVersion: "maker-inventory-v1";
  readDisclaimer: string;
  makerInventoryVerdict: MakerInventoryVerdict;
  marketsEvaluated: number;
  marketsQuoteable: number;
  avgObservedSpread: number;
  avgTopOfBookDepth: number;
  avgQueueCompetitionProxy: number;
  estimatedMakerCapturePerTurn: number;
  estimatedAdverseSelectionCost: number;
  estimatedInventoryRiskCost: number;
  estimatedHedgeCost: number;
  estimatedNetMakerEdge: number;
  candidateMarkets: MakerCandidateMarket[];
  killCriteriaStatus: MakerKillCriteriaStatus;
  makerInventorySummaryLine: string;
  computedAt: string;
}

const WEAK_NET_THRESHOLD = 0.003;
const VIABLE_NET_THRESHOLD = 0.006;

function eligibleForEvaluation(m: NormalizedMarket): boolean {
  return (
    m.active &&
    !m.closed &&
    m.liquidity >= 400 &&
    m.outcomes.length === 2 &&
    m.prices.length === m.outcomes.length &&
    m.spread > 0.001 &&
    m.spread <= 0.55
  );
}

function isQuoteable(m: NormalizedMarket): boolean {
  return m.spread >= 0.012 && m.spread <= 0.48 && m.liquidity >= 2_000 && m.volume >= 800;
}

function topOfBookDepthProxy(m: NormalizedMarket): number {
  return r6(Math.min(22_000, m.liquidity * 0.0045));
}

function queueCompetitionProxy(m: NormalizedMarket): number {
  if (m.liquidity <= 0) return 1;
  return r6(clamp01(m.volume / (m.liquidity * 6 + 500)));
}

function evaluateMarket(m: NormalizedMarket): MakerCandidateMarket {
  const spreadEff = r6(Math.min(0.48, m.spread));
  const qc = queueCompetitionProxy(m);
  const depth = topOfBookDepthProxy(m);
  const observedSpread = spreadEff;
  const fillPlausibilityMaker = r6(
    clamp01(0.18 + 0.55 * clamp01(m.liquidity / 28_000) + 0.27 * (1 - clamp01(spreadEff / 0.5))),
  );
  const quoteWidthCandidate = r6(spreadEff * 0.4);
  const inventoryRiskProxy = r6(
    (1 - clamp01(m.liquidity / 45_000)) * 0.007 + spreadEff * 0.045,
  );
  const adverseSelectionRiskProxy = r6(qc * 0.014 + spreadEff * 0.12);
  const hedgeApplicable = spreadEff > 0.32;
  const estimatedHedgeCost = hedgeApplicable ? 0.0045 : 0.0012;
  const makerFeeProxy = 0.003;
  const estimatedMakerGross = r6(spreadEff * 0.21);
  const estimatedMakerNet = r6(
    estimatedMakerGross -
      adverseSelectionRiskProxy -
      inventoryRiskProxy -
      estimatedHedgeCost -
      makerFeeProxy,
  );
  const supportingNote = `proxy_audit: depth~liquidity*0.0045_cap | adverse~queueComp*0.014+spread*0.12 | inv~(1-liq/45k)*0.007+spread*0.045 | gross~spread*0.21_half_spread_capture_proxy | no_live_orders`;
  return {
    marketId: m.id,
    marketTitle: m.question.length > 140 ? `${m.question.slice(0, 137)}…` : m.question,
    observedSpread,
    topOfBookDepth: depth,
    fillPlausibilityMaker,
    quoteWidthCandidate,
    inventoryRiskProxy,
    adverseSelectionRiskProxy,
    estimatedMakerGross,
    estimatedHedgeCost,
    estimatedMakerNet,
    supportingNote,
  };
}

export function buildMakerInventoryProbeDigest(): MakerInventoryProbeDigest {
  const all = getAllMarkets();
  const evaluated = all.filter(eligibleForEvaluation);
  const quoteable = evaluated.filter(isQuoteable);

  const marketsEvaluated = evaluated.length;
  const marketsQuoteable = quoteable.length;

  const avgObservedSpread =
    evaluated.length > 0
      ? r6(evaluated.reduce((s, m) => s + Math.min(0.48, m.spread), 0) / evaluated.length)
      : 0;
  const avgTopOfBookDepth =
    evaluated.length > 0
      ? r6(evaluated.reduce((s, m) => s + topOfBookDepthProxy(m), 0) / evaluated.length)
      : 0;
  const avgQueueCompetitionProxy =
    evaluated.length > 0
      ? r6(evaluated.reduce((s, m) => s + queueCompetitionProxy(m), 0) / evaluated.length)
      : 0;

  const quoteableRows = quoteable.map(evaluateMarket);
  const estimatedMakerCapturePerTurn =
    quoteableRows.length > 0
      ? r6(quoteableRows.reduce((s, r) => s + r.estimatedMakerGross, 0) / quoteableRows.length)
      : 0;
  const estimatedAdverseSelectionCost =
    quoteableRows.length > 0
      ? r6(quoteableRows.reduce((s, r) => s + r.adverseSelectionRiskProxy, 0) / quoteableRows.length)
      : 0;
  const estimatedInventoryRiskCost =
    quoteableRows.length > 0
      ? r6(quoteableRows.reduce((s, r) => s + r.inventoryRiskProxy, 0) / quoteableRows.length)
      : 0;
  const estimatedHedgeCost =
    quoteableRows.length > 0
      ? r6(quoteableRows.reduce((s, r) => s + r.estimatedHedgeCost, 0) / quoteableRows.length)
      : 0;
  const estimatedNetMakerEdge =
    quoteableRows.length > 0
      ? r6(quoteableRows.reduce((s, r) => s + r.estimatedMakerNet, 0) / quoteableRows.length)
      : 0;

  const candidateMarkets = [...quoteableRows].sort((a, b) => b.estimatedMakerNet - a.estimatedMakerNet).slice(0, 24);

  const maxNet = candidateMarkets.length > 0 ? Math.max(...candidateMarkets.map(c => c.estimatedMakerNet)) : -999;

  const killCriteriaStatus: MakerKillCriteriaStatus = {
    marketsLoadedGtZero: all.length > 0,
    evaluatedGtZero: marketsEvaluated > 0,
    quoteableGtZero: marketsQuoteable > 0,
    anyEstimatedMakerNetPositive: candidateMarkets.some(c => c.estimatedMakerNet > 0),
    anyEstimatedMakerNetAboveWeakThreshold: candidateMarkets.some(c => c.estimatedMakerNet >= WEAK_NET_THRESHOLD),
    avgSpreadAboveNoiseFloor: avgObservedSpread >= 0.012,
  };

  let makerInventoryVerdict: MakerInventoryVerdict;
  if (!killCriteriaStatus.marketsLoadedGtZero || !killCriteriaStatus.evaluatedGtZero) {
    makerInventoryVerdict = "no_quoteable_markets";
  } else if (!killCriteriaStatus.quoteableGtZero) {
    makerInventoryVerdict = "no_quoteable_markets";
  } else if (maxNet >= VIABLE_NET_THRESHOLD) {
    makerInventoryVerdict = "viable_maker_candidate_present";
  } else if (maxNet >= WEAK_NET_THRESHOLD) {
    makerInventoryVerdict = "weak_maker_candidate_present";
  } else if (killCriteriaStatus.anyEstimatedMakerNetPositive) {
    makerInventoryVerdict = "weak_maker_candidate_present";
  } else {
    makerInventoryVerdict = "maker_edge_unproven";
  }

  const makerInventorySummaryLine = `maker_inventory: verdict=${makerInventoryVerdict} | evaluated=${marketsEvaluated} quoteable=${marketsQuoteable} avgSpread=${avgObservedSpread} avgNetProxy=${estimatedNetMakerEdge} maxNetProxy=${r6(maxNet)} | candidates_returned=${candidateMarkets.length}`;

  return {
    probeVersion: "maker-inventory-v1",
    readDisclaimer:
      "Maker/inventory: proxies estruturais a partir de spread/liquidez/volume. Não executa ordens nem simula fila real. Net maker é ilustrativo; falsificação requer dados de quoting e inventário.",
    makerInventoryVerdict,
    marketsEvaluated,
    marketsQuoteable,
    avgObservedSpread,
    avgTopOfBookDepth,
    avgQueueCompetitionProxy,
    estimatedMakerCapturePerTurn,
    estimatedAdverseSelectionCost,
    estimatedInventoryRiskCost,
    estimatedHedgeCost,
    estimatedNetMakerEdge,
    candidateMarkets,
    killCriteriaStatus,
    makerInventorySummaryLine,
    computedAt: new Date().toISOString(),
  };
}
