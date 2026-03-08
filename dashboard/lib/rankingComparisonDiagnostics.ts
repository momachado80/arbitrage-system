/**
 * Ranking Comparison Diagnostics — diagnostic-only comparison of current vs hybrid ranking.
 * Does NOT change production ranking behavior.
 */

import type { NormalizedPaperOpportunity } from "./paperTypes";
import { estimateOpportunityCapacity } from "./capitalCapacityEngine";
import { getEnabledProfiles } from "./shadowSimulationProfiles";

function estimateSpread(opp: { prices?: number[]; confidence?: number }): number {
  if (opp.prices && Array.isArray(opp.prices) && opp.prices.length >= 2) {
    const sorted = [...opp.prices].sort((a, b) => b - a);
    return Math.max(0.01, sorted[0] - sorted[sorted.length - 1]);
  }
  return Math.max(0.01, 0.02 * (1 - (opp.confidence ?? 0.5)));
}

function fillProbabilityEstimate(
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

function normalizeForDiagnostic(opp: Record<string, unknown>, source: "standard" | "graph"): NormalizedPaperOpportunity {
  const isGraph = source === "graph";
  const spread = estimateSpread(opp as { prices?: number[]; confidence?: number });
  if (isGraph) {
    const o = opp as { id: string; type: string; edge: number; confidence: number; liquidity: number; marketsInvolved?: unknown[] };
    return {
      opportunityId: o.id,
      sourceType: "graph",
      opportunityType: (o.type ?? "graph_subset") as NormalizedPaperOpportunity["opportunityType"],
      clusterId: (opp as { clusterId?: string }).clusterId,
      marketsInvolved: (o.marketsInvolved ?? []) as Array<{ marketId: string; question: string }>,
      edge: o.edge,
      confidence: o.confidence,
      liquidity: o.liquidity,
      spread,
    };
  }
  return {
    opportunityId: String(opp.marketId ?? opp.id ?? ""),
    sourceType: "standard",
    opportunityType: String(opp.type ?? "overround") as NormalizedPaperOpportunity["opportunityType"],
    marketsInvolved: [{ marketId: String(opp.marketId ?? ""), question: String(opp.question ?? "") }],
    edge: Number(opp.edge ?? 0),
    confidence: Number(opp.confidence ?? 0),
    liquidity: Number(opp.liquidity ?? 0),
    spread,
  };
}

export interface ExecutableMetrics {
  requestedCapital: number;
  recommendedCapital: number;
  fillProbability: number;
  netEdgeEstimate: number;
  executableExpectedValue: number;
  likelyNoiseSized: boolean;
}

function inferSource(opp: Record<string, unknown>): "standard" | "graph" {
  const hasGraphShape =
    typeof opp.id === "string" &&
    Array.isArray(opp.marketsInvolved) &&
    (opp.marketsInvolved as unknown[]).length > 0;
  return hasGraphShape ? "graph" : "standard";
}

export function estimateExecutableValueForDispatch(opp: Record<string, unknown>): ExecutableMetrics | null {
  const profiles = getEnabledProfiles();
  if (profiles.length === 0) return null;
  const source = inferSource(opp);
  const normalized = normalizeForDiagnostic(opp, source);
  return estimateExecutableMetrics(normalized, profiles[0]);
}

function estimateExecutableMetrics(
  opp: NormalizedPaperOpportunity,
  profile: { maxCapitalPerTrade: number; maxCapitalPerCluster: number; maxCapitalPerMarket: number; liquidityHaircut: number; startingCapital: number }
): ExecutableMetrics {
  const capacity = estimateOpportunityCapacity(opp);
  const liquidity = Math.max(1, opp.liquidity) * profile.liquidityHaircut;
  const recommendedRaw = Math.min(
    capacity.recommendedCapital,
    profile.startingCapital,
    profile.maxCapitalPerTrade,
    liquidity * 0.1,
    opp.liquidity * 0.08
  );
  const requestedCapital = Math.min(recommendedRaw, profile.maxCapitalPerCluster, profile.maxCapitalPerMarket);
  const netEdgeEstimate = Math.max(0, capacity.estimatedNetEdge ?? opp.edge);
  const fillProbability = requestedCapital > 0
    ? fillProbabilityEstimate(opp.liquidity, opp.spread, opp.confidence, requestedCapital, profile.liquidityHaircut)
    : 0;
  const executableExpectedValue = requestedCapital * fillProbability * netEdgeEstimate;
  const likelyNoiseSized = executableExpectedValue < 0.01 || requestedCapital < 1;

  return {
    requestedCapital,
    recommendedCapital: capacity.recommendedCapital,
    fillProbability,
    netEdgeEstimate,
    executableExpectedValue,
    likelyNoiseSized,
  };
}

export type CompositeScoreFn = (opp: Record<string, unknown>) => number;

export function liquidityScoreStandard(liq: number): number {
  if (liq <= 0) return 0;
  return Math.min(1, Math.log10(liq) / 6);
}

export function getCompositeScoreStandard(opp: Record<string, unknown>): number {
  const edge = Number(opp.edge ?? 0);
  const liq = Number(opp.liquidity ?? 0);
  const conf = Number(opp.confidence ?? 0);
  return edge * liquidityScoreStandard(liq) * conf;
}

export function getCompositeScoreGraph(opp: Record<string, unknown>): number {
  const edge = Number(opp.edge ?? 0);
  const liq = Number(opp.liquidity ?? 0);
  const conf = Number(opp.confidence ?? 0);
  return edge * Math.log10(Math.max(1, 1 + liq)) * conf;
}

const EEV_NORMALIZE_FACTOR = 100;

function normalizeEEVForScore(eev: number): number {
  return Math.min(1, Math.max(0, eev * EEV_NORMALIZE_FACTOR));
}

export function runRankingComparisonDiagnostics(
  opportunities: Array<Record<string, unknown>>,
  source: "standard" | "graph",
  sampleSize = 20
): void {
  const profiles = getEnabledProfiles();
  if (profiles.length === 0) return;

  const profile = profiles[0];
  const getCompositeScore = source === "standard" ? getCompositeScoreStandard : getCompositeScoreGraph;

  const sample = opportunities.slice(0, sampleSize);
  const enriched: Array<{
    opp: Record<string, unknown>;
    currentRank: number;
    compositeScore: number;
    metrics: ExecutableMetrics;
    hybridScoreA: number;
    hybridScoreB: number;
    hybridScoreC: number;
  }> = [];

  for (let i = 0; i < sample.length; i++) {
    const opp = sample[i];
    const normalized = normalizeForDiagnostic(opp, source);
    const metrics = estimateExecutableMetrics(normalized, profile);
    const compositeScore = getCompositeScore(opp);
    const eevNorm = normalizeEEVForScore(metrics.executableExpectedValue);

    const hybridScoreA = compositeScore * eevNorm;
    const hybridScoreB =
      0.4 * Number(opp.edge ?? 0) +
      0.2 * Number(opp.confidence ?? 0) +
      0.2 * (source === "standard" ? liquidityScoreStandard(Number(opp.liquidity ?? 0)) : Math.min(1, Math.log10(Math.max(1, 1 + Number(opp.liquidity ?? 0))) / 6)) +
      0.2 * eevNorm;
    const hybridScoreC = metrics.likelyNoiseSized ? compositeScore * 0.1 : compositeScore;

    enriched.push({
      opp,
      currentRank: i + 1,
      compositeScore,
      metrics,
      hybridScoreA,
      hybridScoreB,
      hybridScoreC,
    });
  }

  const byHybridA = [...enriched].sort((a, b) => b.hybridScoreA - a.hybridScoreA);
  const byHybridB = [...enriched].sort((a, b) => b.hybridScoreB - a.hybridScoreB);
  const byHybridC = [...enriched].sort((a, b) => b.hybridScoreC - a.hybridScoreC);

  const comparison = enriched.map((e) => {
    const hybridARank = byHybridA.findIndex((x) => x.opp === e.opp) + 1;
    const hybridBRank = byHybridB.findIndex((x) => x.opp === e.opp) + 1;
    const hybridCRank = byHybridC.findIndex((x) => x.opp === e.opp) + 1;
    return {
      marketId: e.opp.marketId ?? e.opp.id ?? "?",
      currentRank: e.currentRank,
      compositeScore: e.compositeScore,
      executableExpectedValue: e.metrics.executableExpectedValue,
      likelyNoiseSized: e.metrics.likelyNoiseSized,
      hybridScoreA: e.hybridScoreA,
      hybridScoreB: e.hybridScoreB,
      hybridScoreC: e.hybridScoreC,
      rankUnderHybridA: hybridARank,
      rankUnderHybridB: hybridBRank,
      rankUnderHybridC: hybridCRank,
    };
  });

  console.log("[DIAGNOSTICS] RANKING COMPARISON", {
    source,
    sampleSize: comparison.length,
    comparison,
    description: {
      hybridScoreA: "compositeScore × normalizedEEV",
      hybridScoreB: "0.4×edge + 0.2×confidence + 0.2×liquidityScore + 0.2×EEV",
      hybridScoreC: "compositeScore × 0.1 if likelyNoiseSized else compositeScore",
    },
  });
}
