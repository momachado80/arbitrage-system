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

const weakSubgroupCounts = {
  passedButTinyCapital: 0,
  passedButLowFillProbability: 0,
};

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
  if (metrics.requestedCapital < 0.5) weakSubgroupCounts.passedButTinyCapital++;
  if (metrics.fillProbability < 0.15) weakSubgroupCounts.passedButLowFillProbability++;
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

export interface PipelineDiagnosticsInput {
  totalPassedEEVFilter: number;
  totalEvaluateCalls: number;
  totalExecutionCalls: number;
  totalShadowTradesOpened: number;
  earlyExitCounts: Record<string, number>;
}

export function getPassedEEVDownstreamSummary(pipeline: PipelineDiagnosticsInput): {
  passedCohort: {
    count: number;
    reachedEvaluate: number;
    reachedExecution: number;
    shadowTradesOpened: number;
    conversionRate: number;
  };
  passedAverages: {
    avgRequestedCapital: number;
    avgFillProbability: number;
    avgNetEdge: number;
    avgExecutableExpectedValue: number;
  };
  rejectionBreakdown: Record<string, number>;
  fillRejectionDominance: boolean;
  fillRejectionPct: number;
  weakSubgroups: {
    passedButTinyCapital: number;
    passedButLowFillProbability: number;
    tinyCapitalPct: number;
    lowFillPct: number;
  };
  timestamp: string;
} {
  const passedCount = pipeline.totalPassedEEVFilter;
  const reachedEvaluate = pipeline.totalEvaluateCalls;
  const reachedExecution = pipeline.totalExecutionCalls;
  const shadowTradesOpened = pipeline.totalShadowTradesOpened;
  const rejectionBreakdown = pipeline.earlyExitCounts;

  const totalEarlyExits = Object.values(rejectionBreakdown).reduce((s, v) => s + v, 0);
  const fillRejected = rejectionBreakdown.FILL_REJECTED ?? 0;
  const fillRejectionPct = reachedExecution > 0 ? (fillRejected / reachedExecution) * 100 : 0;
  const fillRejectionDominance = totalEarlyExits > 0 && fillRejected >= totalEarlyExits * 0.4;

  const weakSubgroups = {
    passedButTinyCapital: weakSubgroupCounts.passedButTinyCapital,
    passedButLowFillProbability: weakSubgroupCounts.passedButLowFillProbability,
    tinyCapitalPct: passedCount > 0 ? (weakSubgroupCounts.passedButTinyCapital / passedCount) * 100 : 0,
    lowFillPct: passedCount > 0 ? (weakSubgroupCounts.passedButLowFillProbability / passedCount) * 100 : 0,
  };

  return {
    passedCohort: {
      count: passedCount,
      reachedEvaluate,
      reachedExecution,
      shadowTradesOpened,
      conversionRate: passedCount > 0 ? shadowTradesOpened / passedCount : 0,
    },
    passedAverages: passed.count > 0
      ? {
          avgRequestedCapital: passed.sumRequestedCapital / passed.count,
          avgFillProbability: passed.sumFillProbability / passed.count,
          avgNetEdge: passed.sumNetEdge / passed.count,
          avgExecutableExpectedValue: passed.sumEEV / passed.count,
        }
      : {
          avgRequestedCapital: 0,
          avgFillProbability: 0,
          avgNetEdge: 0,
          avgExecutableExpectedValue: 0,
        },
    rejectionBreakdown,
    fillRejectionDominance,
    fillRejectionPct,
    weakSubgroups,
    timestamp: new Date().toISOString(),
  };
}
