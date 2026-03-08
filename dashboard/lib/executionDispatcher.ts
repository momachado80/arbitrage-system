/**
 * Execution Dispatcher — connects scanner output to shadow execution pipeline.
 * Dispatches opportunities to the shadow simulation for evaluation.
 */

import { evaluateOpportunity } from "./shadowSimulationService";
import { incrementDispatch, incrementEarlyExit } from "./shadowPipelineDiagnostics";

export type DispatchedOpportunity = Record<string, unknown>;

export function dispatchOpportunity(opportunity: DispatchedOpportunity): void {
  const marketId = String(opportunity.marketId ?? opportunity.id ?? "?");
  const type = String(opportunity.type ?? "?");
  const edge = Number(opportunity.edge ?? 0);
  const rank = opportunity.rank ?? "?";
  console.log("DISPATCH START", { marketId, type, edge, rank });

  incrementDispatch();

  try {
    console.log("CALLING EXECUTION ENGINE", marketId);
    evaluateOpportunity(opportunity);
  } catch (err) {
    incrementEarlyExit("DISPATCH_CATCH_ERROR");
    console.log("[DIAGNOSTICS] EARLY EXIT", { reason: "DISPATCH_CATCH_ERROR", marketId });
    console.error("[ExecutionDispatcher] Dispatch error:", err);
  }
}
