/**
 * Realistic Execution Engine — combines latency, decay, impact, and fill realism.
 * Produces capturable edge, realistic fills, and execution diagnostics.
 */

import type { NormalizedPaperOpportunity } from "./paperTypes";
import type { CapacityResult } from "./capitalCapacityEngine";
import {
  sampleEntryLatency,
  sampleExitLatency,
  estimateLatencyPenalty,
  type LatencyProfile,
  type LatencySample,
} from "./latencyModel";
import {
  estimateEntryImpact,
  estimateExitImpact,
  estimatePostImpactNetEdge,
  type MarketImpactConfig,
} from "./marketImpactModel";
import {
  estimateCapturableEdge,
  estimateDecayRate,
  type EdgeDecayResult,
  type PersistenceData,
} from "./edgeDecayModel";

export interface RealisticEntryConfig {
  latencyProfile: LatencyProfile;
  impactConfig: Partial<MarketImpactConfig>;
  minCapturableEdgeToTrade: number;
  feeBuffer: number;
  liquidityHaircut: number;
}

export interface RealisticEntryResult {
  observedEdge: number;
  capturableEdgeBeforeImpact: number;
  netEdgeAfterImpact: number;
  requestedCapital: number;
  recommendedCapital: number;
  filledCapital: number;
  fillProbability: number;
  entryLatencyMs: number;
  effectiveEntryPrice: number;
  entrySlippage: number;
  entryImpact: number;
  rejectionReason?: string;
  diagnostics?: Record<string, unknown>;
}

export interface RealisticExitConfig {
  latencyProfile: LatencyProfile;
  impactConfig: Partial<MarketImpactConfig>;
}

export interface RealisticExitResult {
  exitLatencyMs: number;
  effectiveExitPrice: number;
  exitSlippage: number;
  exitImpact: number;
  realizedPnL: number;
  realizedReturn: number;
  exitReason: string;
}

export interface ActiveShadowTradeState {
  tradeId: string;
  opportunityId: string;
  openedAt: string;
  entryEdge: number;
  capturableEdgeAtEntry: number;
  effectiveEntryPrice: number;
  filledCapital: number;
  maxAdverseExcursion: number;
  maxFavorableExcursion: number;
}

const DEFAULT_ENTRY_CONFIG: RealisticEntryConfig = {
  latencyProfile: "normal",
  impactConfig: {},
  minCapturableEdgeToTrade: 0.005,
  feeBuffer: 0.002,
  liquidityHaircut: 0.7,
};

function fillProbability(
  liquidity: number,
  spread: number,
  confidence: number,
  requestedCapital: number,
  liquidityHaircut: number
): number {
  const liq = Math.max(liquidity * liquidityHaircut, 1);
  const sizeRatio = requestedCapital / liq;
  const liqScore = Math.min(1, Math.log10(liq) / 5);
  const spreadPenalty = Math.max(0.3, 1 - spread * 3);
  const sizePenalty = Math.max(0.2, 1 - sizeRatio * 2);
  return Math.min(0.9, 0.65 * liqScore * spreadPenalty * sizePenalty * confidence);
}

function deterministicFill(prob: number, requested: number): number {
  if (prob < 0.15) return 0;
  return requested * Math.min(1, prob);
}

export function simulateRealisticEntry(
  opportunity: NormalizedPaperOpportunity,
  capacity: CapacityResult,
  profileState: {
    availableCapital: number;
    maxCapitalPerTrade: number;
    maxCapitalPerCluster: number;
    maxCapitalPerMarket: number;
    exposureByCluster: Record<string, number>;
    exposureByMarket: Record<string, number>;
  },
  persistenceData: PersistenceData | null,
  config?: Partial<RealisticEntryConfig>
): RealisticEntryResult {
  const cfg = { ...DEFAULT_ENTRY_CONFIG, ...config };
  const observedEdge = Math.max(0, opportunity.edge);
  const spread = Math.max(0.01, opportunity.spread);
  const liquidity = Math.max(1, opportunity.liquidity) * cfg.liquidityHaircut;

  const latencySample = sampleEntryLatency(cfg.latencyProfile);
  const latencyPenalty = estimateLatencyPenalty(
    latencySample.totalEntryLatencyMs,
    persistenceData
  );
  const decayResult = estimateCapturableEdge(opportunity, latencySample.totalEntryLatencyMs, persistenceData);
  const capturableBeforeImpact = Math.max(0, decayResult.capturableEdge - latencyPenalty - cfg.feeBuffer);

  const recommendedRaw = Math.min(
    capacity.recommendedCapital,
    profileState.availableCapital,
    profileState.maxCapitalPerTrade,
    liquidity * 0.1,
    opportunity.liquidity * 0.08
  );

  const clusterExposure = opportunity.clusterId
    ? profileState.exposureByCluster[opportunity.clusterId] || 0
    : 0;
  const marketExposure = opportunity.marketsInvolved.reduce(
    (s, m) => s + (profileState.exposureByMarket[m.marketId] || 0),
    0
  );
  const remainingCluster = Math.max(0, profileState.maxCapitalPerCluster - clusterExposure);
  const remainingMarket = Math.max(0, profileState.maxCapitalPerMarket - marketExposure);
  const requestedCapital = Math.min(recommendedRaw, remainingCluster, remainingMarket);

  if (requestedCapital <= 0) {
    const marketId = opportunity.marketsInvolved[0]?.marketId ?? opportunity.opportunityId;
    console.log("[DIAGNOSTICS] CAPITAL REJECTION", {
      marketId,
      recommendedCapital: recommendedRaw,
      requestedCapital: 0,
    });
    return {
      observedEdge,
      capturableEdgeBeforeImpact: capturableBeforeImpact,
      netEdgeAfterImpact: 0,
      requestedCapital: 0,
      recommendedCapital: recommendedRaw,
      filledCapital: 0,
      fillProbability: 0,
      entryLatencyMs: latencySample.totalEntryLatencyMs,
      effectiveEntryPrice: 1 - observedEdge,
      entrySlippage: 0,
      entryImpact: 0,
      rejectionReason: "insufficient_capital_or_exposure_limit",
    };
  }

  const impactResult = estimateEntryImpact(opportunity, requestedCapital, cfg.impactConfig);
  const netEdgeAfterImpact = estimatePostImpactNetEdge(
    opportunity,
    requestedCapital,
    cfg.impactConfig
  );

  if (netEdgeAfterImpact < cfg.minCapturableEdgeToTrade) {
    const marketId = opportunity.marketsInvolved[0]?.marketId ?? opportunity.opportunityId;
    console.log("[DIAGNOSTICS] FILL REJECTION", {
      marketId,
      filledCapital: 0,
      reason: "net_edge_below_threshold",
    });
    return {
      observedEdge,
      capturableEdgeBeforeImpact: capturableBeforeImpact,
      netEdgeAfterImpact,
      requestedCapital,
      recommendedCapital: recommendedRaw,
      filledCapital: 0,
      fillProbability: 0,
      entryLatencyMs: latencySample.totalEntryLatencyMs,
      effectiveEntryPrice: 1 - observedEdge,
      entrySlippage: impactResult.expectedPriceWorseningEntry * 0.5,
      entryImpact: impactResult.expectedPriceWorseningEntry * 0.5,
      rejectionReason: "net_edge_below_threshold",
      diagnostics: { netEdge: netEdgeAfterImpact, threshold: cfg.minCapturableEdgeToTrade },
    };
  }

  const prob = fillProbability(
    opportunity.liquidity,
    spread,
    opportunity.confidence,
    requestedCapital,
    cfg.liquidityHaircut
  );
  const filled = deterministicFill(prob, requestedCapital);

  if (filled <= 0) {
    const marketId = opportunity.marketsInvolved[0]?.marketId ?? opportunity.opportunityId;
    const scenarios = [
      { cutoff: 0.15, filled: prob < 0.15 ? 0 : requestedCapital * Math.min(1, prob) },
      { cutoff: 0.1, filled: prob < 0.1 ? 0 : requestedCapital * Math.min(1, prob) },
      { cutoff: 0.08, filled: prob < 0.08 ? 0 : requestedCapital * Math.min(1, prob) },
      { cutoff: 0.05, filled: prob < 0.05 ? 0 : requestedCapital * Math.min(1, prob) },
      { cutoff: "proportional", filled: requestedCapital * prob },
    ] as const;
    const MIN_ECONOMIC_USD = 1;
    const MIN_EXPECTED_PROFIT_USD = 0.01;
    const economicAnalysis = scenarios.map((s) => {
      const h = s.filled;
      const netEdgeCaptured = h * netEdgeAfterImpact;
      const fillFraction = requestedCapital > 0 ? h / requestedCapital : 0;
      const exceedsMinSize = h >= MIN_ECONOMIC_USD;
      const exceedsMinProfit = netEdgeCaptured >= MIN_EXPECTED_PROFIT_USD;
      const likelyNoise = h < MIN_ECONOMIC_USD || netEdgeCaptured < MIN_EXPECTED_PROFIT_USD;
      return {
        cutoff: s.cutoff,
        hypotheticalFilledCapital: h,
        hypotheticalNetEdgeCaptured: netEdgeCaptured,
        hypotheticalFillFraction: fillFraction,
        exceedsMinEconomicSize: exceedsMinSize,
        exceedsMinExpectedProfit: exceedsMinProfit,
        likelyNoiseSized: likelyNoise,
      };
    });
    console.log("[DIAGNOSTICS] FILL REJECTION ECONOMIC WHAT-IF", {
      marketId,
      currentFillProbability: prob,
      currentRequestedCapital: requestedCapital,
      netEdgeAfterImpact,
      economicAnalysis,
      minEconomicFloorUsd: MIN_ECONOMIC_USD,
      minExpectedProfitUsd: MIN_EXPECTED_PROFIT_USD,
    });
    return {
      observedEdge,
      capturableEdgeBeforeImpact: capturableBeforeImpact,
      netEdgeAfterImpact,
      requestedCapital,
      recommendedCapital: recommendedRaw,
      filledCapital: 0,
      fillProbability: prob,
      entryLatencyMs: latencySample.totalEntryLatencyMs,
      effectiveEntryPrice: 1 - observedEdge,
      entrySlippage: 0,
      entryImpact: 0,
      rejectionReason: "fill_rejected",
      diagnostics: { fillProbability: prob },
    };
  }

  const effectiveSlippage = impactResult.expectedPriceWorseningEntry * (filled / requestedCapital);
  const effectiveEntryPrice = 1 - observedEdge + effectiveSlippage;

  return {
    observedEdge,
    capturableEdgeBeforeImpact: capturableBeforeImpact,
    netEdgeAfterImpact,
    requestedCapital,
    recommendedCapital: recommendedRaw,
    filledCapital: filled,
    fillProbability: prob,
    entryLatencyMs: latencySample.totalEntryLatencyMs,
    effectiveEntryPrice,
    entrySlippage: effectiveSlippage * 0.5,
    entryImpact: effectiveSlippage * 0.5,
    diagnostics: {
      latencyMs: latencySample.totalEntryLatencyMs,
      decayRate: decayResult.decayRate,
      fragilityScore: decayResult.fragilityScore,
    },
  };
}

export function simulateRealisticExit(
  activeTrade: ActiveShadowTradeState,
  latestOpportunity: { edge: number; confidence: number } | null,
  config?: Partial<RealisticExitConfig>
): RealisticExitResult {
  const cfg = { ...DEFAULT_ENTRY_CONFIG, ...config } as RealisticExitConfig;
  const latencySample = sampleExitLatency(cfg.latencyProfile);

  const oppForImpact = {
    edge: latestOpportunity?.edge ?? activeTrade.entryEdge,
    liquidity: 1,
    spread: 0.02,
    confidence: latestOpportunity?.confidence ?? 0.5,
  };

  const exitImpactResult = estimateExitImpact(
    oppForImpact as NormalizedPaperOpportunity,
    activeTrade.filledCapital,
    cfg.impactConfig
  );

  const exitPrice = latestOpportunity
    ? 1 - latestOpportunity.edge
    : activeTrade.effectiveEntryPrice;
  const effectiveExitPrice = exitPrice - exitImpactResult.expectedPriceWorseningExit;
  const pnlPct = (effectiveExitPrice - activeTrade.effectiveEntryPrice) / Math.max(0.001, activeTrade.effectiveEntryPrice);
  const realizedPnL = activeTrade.filledCapital * pnlPct;
  const realizedReturn = pnlPct;

  let exitReason = "edge_normalization";
  const now = Date.now();
  const holdingMs = now - new Date(activeTrade.openedAt).getTime();
  if (holdingMs >= 300_000) exitReason = "max_holding_time";
  else if (pnlPct <= -0.03) exitReason = "stop_loss";
  else if (pnlPct >= 0.05) exitReason = "take_profit";
  else if (!latestOpportunity) exitReason = "opportunity_disappeared";

  return {
    exitLatencyMs: latencySample.totalExitLatencyMs,
    effectiveExitPrice,
    exitSlippage: exitImpactResult.expectedPriceWorseningExit * 0.5,
    exitImpact: exitImpactResult.expectedPriceWorseningExit * 0.5,
    realizedPnL,
    realizedReturn,
    exitReason,
  };
}

export function buildExecutionDiagnostics(
  entry: RealisticEntryResult,
  exit?: RealisticExitResult
): Record<string, unknown> {
  const d: Record<string, unknown> = {
    observedEdge: entry.observedEdge,
    capturableBeforeImpact: entry.capturableEdgeBeforeImpact,
    netAfterImpact: entry.netEdgeAfterImpact,
    entryLatencyMs: entry.entryLatencyMs,
    fillProbability: entry.fillProbability,
    filledCapital: entry.filledCapital,
    requestedCapital: entry.requestedCapital,
  };
  if (exit) {
    d.exitLatencyMs = exit.exitLatencyMs;
    d.realizedPnL = exit.realizedPnL;
    d.realizedReturn = exit.realizedReturn;
    d.exitReason = exit.exitReason;
  }
  return d;
}
