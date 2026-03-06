import type { DetectedEdge, EdgeType } from "./probabilityScanner";

export interface RankedOpportunity extends DetectedEdge {
  rank: number;
  compositeScore: number;
}

const MIN_CONFIDENCE = 0.05;
const MIN_LIQUIDITY = 500;
const MAX_RESULTS = 50;

function liquidityScore(liquidity: number): number {
  if (liquidity <= 0) return 0;
  return Math.min(1, Math.log10(liquidity) / 6);
}

export function rankOpportunities(
  edges: DetectedEdge[],
  options?: { minConfidence?: number; minLiquidity?: number; maxResults?: number }
): RankedOpportunity[] {
  const minConf = options?.minConfidence ?? MIN_CONFIDENCE;
  const minLiq = options?.minLiquidity ?? MIN_LIQUIDITY;
  const max = options?.maxResults ?? MAX_RESULTS;

  const filtered = edges.filter(
    (e) => e.confidence >= minConf && e.liquidity >= minLiq
  );

  const scored: RankedOpportunity[] = filtered.map((e) => ({
    ...e,
    rank: 0,
    compositeScore: e.edge * liquidityScore(e.liquidity) * e.confidence,
  }));

  scored.sort((a, b) => b.compositeScore - a.compositeScore);

  return scored.slice(0, max).map((s, i) => ({ ...s, rank: i + 1 }));
}

export function filterByType(
  opportunities: RankedOpportunity[],
  type: EdgeType
): RankedOpportunity[] {
  return opportunities.filter((o) => o.type === type);
}

export function computeSystemMetrics(
  ranked: RankedOpportunity[],
  totalMarkets: number
) {
  const avgConfidence =
    ranked.length > 0
      ? ranked.reduce((s, r) => s + r.confidence, 0) / ranked.length
      : 0;

  const avgEdge =
    ranked.length > 0
      ? ranked.reduce((s, r) => s + r.edge, 0) / ranked.length
      : 0;

  let qualityScore: number;
  if (ranked.length === 0) {
    qualityScore = 0;
  } else {
    const densityFactor = Math.min(1, ranked.length / 20);
    qualityScore = Math.min(1, avgConfidence * 0.4 + densityFactor * 0.3 + Math.min(1, avgEdge * 10) * 0.3);
  }

  return {
    marketQuality: qualityScore,
    marketsTracked: totalMarkets,
    opportunitiesDetected: ranked.length,
    avgConfidence,
    avgEdge,
    overroundCount: ranked.filter((r) => r.type === "overround").length,
    underroundCount: ranked.filter((r) => r.type === "underround").length,
    crossMarketCount: ranked.filter((r) => r.type === "cross_market").length,
  };
}
