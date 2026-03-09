/**
 * EEV Filter Quality Tracker — diagnostic aggregation for passed vs filtered opportunities.
 * Does NOT change production behavior. Only accumulates stats for analysis.
 */

import type { ExecutableMetrics } from "./rankingComparisonDiagnostics";

interface AggregateStats {
  count: number;
  sumEdge: number;
  sumConfidence: number;
  sumLiquidity: number;
  sumRequestedCapital: number;
  sumRecommendedCapital: number;
  sumFillProbability: number;
  sumNetEdge: number;
  sumEEV: number;
  countLikelyNoise: number;
  minEEV: number;
  maxEEV: number;
  lastExample: Record<string, unknown> | null;
}

const emptyStats = (): AggregateStats => ({
  count: 0,
  sumEdge: 0,
  sumConfidence: 0,
  sumLiquidity: 0,
  sumRequestedCapital: 0,
  sumRecommendedCapital: 0,
  sumFillProbability: 0,
  sumNetEdge: 0,
  sumEEV: 0,
  countLikelyNoise: 0,
  minEEV: Infinity,
  maxEEV: -Infinity,
  lastExample: null,
});

const filtered: AggregateStats = emptyStats();
const passed: AggregateStats = emptyStats();

function addToStats(
  stats: AggregateStats,
  opp: { edge?: unknown; confidence?: unknown; liquidity?: unknown },
  metrics: ExecutableMetrics,
  example: Record<string, unknown>
): void {
  stats.count++;
  stats.sumEdge += Number(opp.edge ?? 0);
  stats.sumConfidence += Number(opp.confidence ?? 0);
  stats.sumLiquidity += Number(opp.liquidity ?? 0);
  stats.sumRequestedCapital += metrics.requestedCapital;
  stats.sumRecommendedCapital += metrics.recommendedCapital;
  stats.sumFillProbability += metrics.fillProbability;
  stats.sumNetEdge += metrics.netEdgeEstimate;
  stats.sumEEV += metrics.executableExpectedValue;
  if (metrics.likelyNoiseSized) stats.countLikelyNoise++;
  stats.minEEV = Math.min(stats.minEEV, metrics.executableExpectedValue);
  stats.maxEEV = Math.max(stats.maxEEV, metrics.executableExpectedValue);
  stats.lastExample = example;
}

function toSummary(stats: AggregateStats, label: string): Record<string, unknown> {
  if (stats.count === 0) {
    return { label, count: 0 };
  }
  return {
    label,
    count: stats.count,
    avgEdge: stats.sumEdge / stats.count,
    avgConfidence: stats.sumConfidence / stats.count,
    avgLiquidity: stats.sumLiquidity / stats.count,
    avgRequestedCapital: stats.sumRequestedCapital / stats.count,
    avgRecommendedCapital: stats.sumRecommendedCapital / stats.count,
    avgFillProbability: stats.sumFillProbability / stats.count,
    avgNetEdge: stats.sumNetEdge / stats.count,
    avgEEV: stats.sumEEV / stats.count,
    likelyNoisePct: (stats.countLikelyNoise / stats.count) * 100,
    minEEV: stats.minEEV === Infinity ? null : stats.minEEV,
    maxEEV: stats.maxEEV === -Infinity ? null : stats.maxEEV,
  };
}

export function recordFiltered(
  opp: Record<string, unknown>,
  metrics: ExecutableMetrics
): void {
  addToStats(
    filtered,
    { edge: opp.edge, confidence: opp.confidence, liquidity: opp.liquidity },
    metrics,
    {
      marketId: opp.marketId ?? opp.id,
      edge: opp.edge,
      confidence: opp.confidence,
      liquidity: opp.liquidity,
      requestedCapital: metrics.requestedCapital,
      fillProbability: metrics.fillProbability,
      executableExpectedValue: metrics.executableExpectedValue,
    }
  );
}

export function recordPassed(
  opp: Record<string, unknown>,
  metrics: ExecutableMetrics
): void {
  addToStats(
    passed,
    { edge: opp.edge, confidence: opp.confidence, liquidity: opp.liquidity },
    metrics,
    {
      marketId: opp.marketId ?? opp.id,
      edge: opp.edge,
      confidence: opp.confidence,
      liquidity: opp.liquidity,
      requestedCapital: metrics.requestedCapital,
      fillProbability: metrics.fillProbability,
      executableExpectedValue: metrics.executableExpectedValue,
    }
  );
}

export function getEEVFilterQualitySummary(): {
  filtered: Record<string, unknown>;
  passed: Record<string, unknown>;
  comparison: {
    passedHasHigherAvgEEV: boolean;
    passedHasHigherAvgRequestedCapital: boolean;
    passedHasLowerLikelyNoisePct: boolean;
  };
  timestamp: string;
} {
  const filteredSummary = toSummary(filtered, "filtered");
  const passedSummary = toSummary(passed, "passed");

  const passedAvgEEV = passed.count > 0 ? passed.sumEEV / passed.count : 0;
  const filteredAvgEEV = filtered.count > 0 ? filtered.sumEEV / filtered.count : 0;
  const passedAvgReq = passed.count > 0 ? passed.sumRequestedCapital / passed.count : 0;
  const filteredAvgReq = filtered.count > 0 ? filtered.sumRequestedCapital / filtered.count : 0;
  const passedNoisePct = passed.count > 0 ? (passed.countLikelyNoise / passed.count) * 100 : 0;
  const filteredNoisePct = filtered.count > 0 ? (filtered.countLikelyNoise / filtered.count) * 100 : 100;

  return {
    filtered: filteredSummary,
    passed: passedSummary,
    comparison: {
      passedHasHigherAvgEEV: passedAvgEEV > filteredAvgEEV,
      passedHasHigherAvgRequestedCapital: passedAvgReq > filteredAvgReq,
      passedHasLowerLikelyNoisePct: passedNoisePct < filteredNoisePct,
    },
    timestamp: new Date().toISOString(),
  };
}
