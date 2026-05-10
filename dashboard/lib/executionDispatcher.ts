/**
 * Execution Dispatcher — connects scanner output to shadow execution pipeline.
 * Dispatches opportunities to the shadow simulation for evaluation.
 */

import { evaluateOpportunity } from "./shadowSimulationService";
import { incrementDispatch, incrementEarlyExit, incrementFilteredByEEV, incrementPassedEEVFilter } from "./shadowPipelineDiagnostics";
import { estimateExecutableValueForDispatch } from "./rankingComparisonDiagnostics";
import { MIN_EXECUTABLE_EXPECTED_VALUE_USD } from "./executionFilterConfig";
import { recordFiltered, recordPassed } from "./eevFilterQualityTracker";
import { getAllMarkets } from "./marketDataService";
import { evaluateOpportunityUniverseQuality } from "./universeQualityGate";

export type DispatchedOpportunity = Record<string, unknown>;

const VERBOSE = process.env.WORKER_VERBOSE_LOGS === "1";

export function dispatchOpportunity(opportunity: DispatchedOpportunity): void {
  const marketId = String(opportunity.marketId ?? opportunity.id ?? "?");
  const type = String(opportunity.type ?? "?");
  const edge = Number(opportunity.edge ?? 0);
  const rank = opportunity.rank ?? "?";
  if (VERBOSE) console.log("DISPATCH START", { marketId, type, edge, rank });

  /**
   * UNIVERSE QUALITY GATE — antes de incrementDispatch / EEV / evaluateOpportunity.
   * Cobre standard E graph (ambos chegam aqui). Fail-closed quando perna falta no cache.
   * Sem efeito colateral em .paper, sem tocar shadow store.
   */
  const cachedMarkets = getAllMarkets();
  const lookup = (legId: string) =>
    cachedMarkets.find(m => m.id === legId) ?? null;
  const uqGate = evaluateOpportunityUniverseQuality(
    opportunity,
    lookup,
    new Date().toISOString(),
  );
  if (uqGate.rejected) {
    incrementEarlyExit(`BLOCKED_BY_UNIVERSE_QUALITY:${uqGate.verdict}`);
    if (VERBOSE) {
      console.log("[DIAGNOSTICS] UQ BLOCKED", {
        opportunityId: opportunity.opportunityId ?? opportunity.id ?? null,
        opportunityType: opportunity.opportunityType ?? null,
        sourceType: opportunity.sourceType ?? null,
        verdict: uqGate.verdict,
        legMarketId: uqGate.legMarketId,
        question: uqGate.question,
        mid: uqGate.mid,
        liquidity: uqGate.liquidity,
        suitabilityVerdict: uqGate.suitabilityVerdict,
        reasons: uqGate.reasons,
        disqualifiers: uqGate.disqualifiers,
      });
    }
    return;
  }

  incrementDispatch();

  const metrics = estimateExecutableValueForDispatch(opportunity);
  if (metrics && metrics.executableExpectedValue < MIN_EXECUTABLE_EXPECTED_VALUE_USD) {
    incrementFilteredByEEV();
    recordFiltered(opportunity, metrics);
    if (VERBOSE) console.log("[DIAGNOSTICS] EEV FILTERED OUT", {
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
    if (VERBOSE) console.log("CALLING EXECUTION ENGINE", marketId);
    evaluateOpportunity(opportunity);
  } catch (err) {
    incrementEarlyExit("DISPATCH_CATCH_ERROR");
    if (VERBOSE) console.log("[DIAGNOSTICS] EARLY EXIT", { reason: "DISPATCH_CATCH_ERROR", marketId });
    console.error("[ExecutionDispatcher] Dispatch error:", err);
  }
}
