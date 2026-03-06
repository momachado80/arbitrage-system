/**
 * Shared types for the Paper Trading + Capital Capacity Simulation layer.
 * Normalized opportunity shape used by capacity engine, execution simulator, and paper trade engine.
 */

export type PaperSourceType = "standard" | "graph";

export type PaperOpportunityType =
  | "overround"
  | "underround"
  | "cross_market"
  | "graph_subset"
  | "graph_complement"
  | "graph_exclusive"
  | "graph_equivalence"
  | "graph_cycle";

export interface NormalizedPaperOpportunity {
  opportunityId: string;
  sourceType: PaperSourceType;
  opportunityType: PaperOpportunityType;
  clusterId?: string;
  marketsInvolved: Array<{ marketId: string; question: string }>;
  edge: number;
  confidence: number;
  liquidity: number;
  spread: number;
  compositeScore?: number;
  rank?: number;
}

export interface CapacityEstimate {
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

export type PaperTradeStatus = "active" | "closed" | "rejected";

export type ExitCondition =
  | "edge_normalization"
  | "max_holding_time"
  | "stop_loss"
  | "take_profit"
  | "manual";

export interface PaperTrade {
  tradeId: string;
  opportunityId: string;
  sourceType: PaperSourceType;
  opportunityType: PaperOpportunityType;
  clusterId?: string;
  marketsInvolved: Array<{ marketId: string; question: string }>;
  openedAt: string;
  closedAt: string | null;
  status: PaperTradeStatus;
  rejectionReason?: string;
  grossEdgeAtEntry: number;
  netEdgeAtEntry: number;
  recommendedCapital: number;
  requestedCapital: number;
  filledCapital: number;
  entryPriceEstimate: number;
  exitPriceEstimate?: number;
  entryConfidence: number;
  exitConfidence?: number;
  realizedPnL: number;
  realizedReturn: number;
  holdingTimeMs: number;
  maxAdverseExcursion: number;
  maxFavorableExcursion: number;
  exitCondition?: ExitCondition;
  notes?: string;
}
