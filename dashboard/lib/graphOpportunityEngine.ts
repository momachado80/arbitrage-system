import type { GraphOpportunity } from "./graphArbitrageEngine";

export interface RankedGraphOpportunity extends GraphOpportunity {
  rank: number;
  compositeScore: number;
}

export interface GraphSummary {
  graphOpportunitiesDetected: number;
  averageConfidence: number;
  averageEdge: number;
  numberOfClustersScanned: number;
  byType: Record<string, number>;
}

const MIN_CONFIDENCE = 0.03;
const MIN_LIQUIDITY = 100;
const MAX_RESULTS = 50;

export function rankGraphOpportunities(
  opportunities: GraphOpportunity[],
  options?: { minConfidence?: number; minLiquidity?: number; maxResults?: number }
): RankedGraphOpportunity[] {
  const minConf = options?.minConfidence ?? MIN_CONFIDENCE;
  const minLiq = options?.minLiquidity ?? MIN_LIQUIDITY;
  const max = options?.maxResults ?? MAX_RESULTS;

  const filtered = opportunities.filter(
    (o) => o.confidence >= minConf && o.liquidity >= minLiq
  );

  const scored: RankedGraphOpportunity[] = filtered.map((o) => ({
    ...o,
    rank: 0,
    compositeScore: o.edge * Math.log10(Math.max(1, 1 + o.liquidity)) * o.confidence,
  }));

  scored.sort((a, b) => b.compositeScore - a.compositeScore);

  return scored.slice(0, max).map((s, i) => ({ ...s, rank: i + 1 }));
}

export function computeGraphSummary(
  ranked: RankedGraphOpportunity[],
  clustersScanned: number
): GraphSummary {
  const count = ranked.length;
  const byType: Record<string, number> = {};

  let totalConf = 0;
  let totalEdge = 0;

  for (const r of ranked) {
    totalConf += r.confidence;
    totalEdge += r.edge;
    byType[r.type] = (byType[r.type] || 0) + 1;
  }

  return {
    graphOpportunitiesDetected: count,
    averageConfidence: count > 0 ? totalConf / count : 0,
    averageEdge: count > 0 ? totalEdge / count : 0,
    numberOfClustersScanned: clustersScanned,
    byType,
  };
}
