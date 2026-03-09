/**
 * Execution Dispatcher — connects scanner output to shadow execution pipeline.
 * Dispatches opportunities to the shadow simulation for evaluation.
 */

import { evaluateOpportunity } from "./shadowSimulationService";
import { incrementDispatch, incrementEarlyExit, incrementFilteredByEEV, incrementPassedEEVFilter } from "./shadowPipelineDiagnostics";
import { estimateExecutableValueForDispatch } from "./rankingComparisonDiagnostics";
import { MIN_EXECUTABLE_EXPECTED_VALUE_USD } from "./executionFilterConfig";
import { recordFiltered, recordPassed } from "./eevFilterQualityTracker";

export type DispatchedOpportunity = Record<string, unknown>;

export function dispatchOpportunity(opportunity: DispatchedOpportunity): void {
  const marketId = String(opportunity.marketId ?? opportunity.id ?? "?");
  const type = String(opportunity.type ?? "?");
  const edge = Number(opportunity.edge ?? 0);
  const rank = opportunity.rank ?? "?";
  console.log("DISPATCH START", { marketId, type, edge, rank });

  incrementDispatch();

  const metrics = estimateExecutableValueForDispatch(opportunity);
  if (metrics && metrics.executableExpectedValue < MIN_EXECUTABLE_EXPECTED_VALUE_USD) {
    incrementFilteredByEEV();
    recordFiltered(opportunity, metrics);
    console.log("[DIAGNOSTICS] EEV FILTERED OUT", {
      marketId,
      edge,
      confidence: opportunity.confidence,
      liquidity: opportunity.liquidity,
      requestedCapital: metrics.requestedCapital,
      fillProbability: metrics.fillProbability,
      netEdgeAfterImpact: metrics.netEdgeEstimate,
      executableExpectedValue: metrics.executableExpectedValue,
      reason: "BELOW_MIN_EXECUTABLE_VALUE",
    });
    return;
  }

  incrementPassedEEVFilter();
  if (metrics) recordPassed(opportunity, metrics);

  try {
    console.log("CALLING EXECUTION ENGINE", marketId);
    evaluateOpportunity(opportunity);
  } catch (err) {
    incrementEarlyExit("DISPATCH_CATCH_ERROR");
    console.log("[DIAGNOSTICS] EARLY EXIT", { reason: "DISPATCH_CATCH_ERROR", marketId });
    console.error("[ExecutionDispatcher] Dispatch error:", err);
  }
}
