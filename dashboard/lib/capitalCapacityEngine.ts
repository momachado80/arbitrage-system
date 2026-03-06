/**
 * Capital Capacity Engine — estimates max deployable capital before opportunity becomes non-viable.
 * Conservative model using liquidity, spread, edge, confidence.
 */

import type { NormalizedPaperOpportunity } from "./paperTypes";

export interface CapacityConfig {
  minNetEdgeThreshold: number;
  feeBuffer: number;
  edgeDecayFactor: number;
  slippageMultiplier: number;
  impactMultiplier: number;
  recommendedCapitalFraction: number;
}

const DEFAULT_CONFIG: CapacityConfig = {
  minNetEdgeThreshold: 0.005,
  feeBuffer: 0.002,
  edgeDecayFactor: 1.5,
  slippageMultiplier: 1.2,
  impactMultiplier: 0.8,
  recommendedCapitalFraction: 0.5,
};

export interface CapacityResult {
  opportunityId: string;
  estimatedGrossEdge: number;
  estimatedSlippageRate: number;
  estimatedImpactRate: number;
  estimatedNetEdge: number;
  maxDeployableCapital: number;
  recommendedCapital: number;
  capacityConfidence: number;
  reasoning: string;
}

function solveMaxCapital(
  grossEdge: number,
  spread: number,
  liquidity: number,
  config: CapacityConfig
): { maxCapital: number; slippageRate: number; impactRate: number } {
  const liq = Math.max(liquidity, 1);
  const minNet = config.minNetEdgeThreshold;
  const fee = config.feeBuffer;
  const decay = config.edgeDecayFactor;
  const slipMult = config.slippageMultiplier;
  const impactMult = config.impactMultiplier;

  if (grossEdge <= fee + minNet) {
    return { maxCapital: 0, slippageRate: 0, impactRate: 0 };
  }

  const edgeAfterFee = grossEdge - fee;
  const maxNet = edgeAfterFee - minNet;
  if (maxNet <= 0) {
    return { maxCapital: 0, slippageRate: 0, impactRate: 0 };
  }

  const slipCoeff = (spread * slipMult) / liq;
  const impactCoeff = (decay * impactMult) / liq;
  const totalCoeff = slipCoeff + impactCoeff;

  if (totalCoeff <= 0) {
    return { maxCapital: Math.min(liq * 0.1, 10000), slippageRate: 0, impactRate: 0 };
  }

  const maxOrderSize = maxNet / totalCoeff;
  const capped = Math.min(maxOrderSize, liq * 0.15, 5000);

  const orderSize = Math.max(0, capped);
  const slippageRate = slipCoeff * orderSize;
  const impactRate = impactCoeff * orderSize;

  return {
    maxCapital: orderSize,
    slippageRate,
    impactRate,
  };
}

export function estimateOpportunityCapacity(
  opportunity: NormalizedPaperOpportunity,
  config?: Partial<CapacityConfig>
): CapacityResult {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const grossEdge = Math.max(0, opportunity.edge);
  const spread = Math.max(0.01, opportunity.spread);
  const liquidity = Math.max(1, opportunity.liquidity);
  const confidence = Math.max(0, Math.min(1, opportunity.confidence));

  const { maxCapital, slippageRate, impactRate } = solveMaxCapital(
    grossEdge,
    spread,
    liquidity,
    cfg
  );

  const netEdge = grossEdge - slippageRate - impactRate - cfg.feeBuffer;
  const capacityConfidence = Math.min(
    1,
    confidence * Math.min(1, liquidity / 2000) * (netEdge > cfg.minNetEdgeThreshold ? 1 : 0.5)
  );

  let recommendedCapital = 0;
  let reasoning = "Net edge below threshold; no capital recommended.";

  if (netEdge > cfg.minNetEdgeThreshold && maxCapital > 0) {
    recommendedCapital = Math.min(
      maxCapital * cfg.recommendedCapitalFraction,
      maxCapital
    );
    reasoning = `Net edge ${(netEdge * 100).toFixed(2)}% above threshold; max ${maxCapital.toFixed(0)} USD, recommended ${recommendedCapital.toFixed(0)} USD.`;
  } else if (netEdge <= 0) {
    reasoning = "Slippage + impact + fees exceed gross edge.";
  }

  return {
    opportunityId: opportunity.opportunityId,
    estimatedGrossEdge: grossEdge,
    estimatedSlippageRate: slippageRate,
    estimatedImpactRate: impactRate,
    estimatedNetEdge: netEdge,
    maxDeployableCapital: maxCapital,
    recommendedCapital,
    capacityConfidence,
    reasoning,
  };
}

export function estimateBatchCapacity(
  opportunities: NormalizedPaperOpportunity[],
  config?: Partial<CapacityConfig>
): CapacityResult[] {
  return opportunities.map((opp) => estimateOpportunityCapacity(opp, config));
}
