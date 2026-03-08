/**
 * Execution Dispatcher — connects scanner output to shadow execution pipeline.
 * Dispatches opportunities to the shadow simulation for evaluation.
 */

import { evaluateOpportunity } from "./shadowSimulationService";

export type DispatchedOpportunity = Record<string, unknown>;

export function dispatchOpportunity(opportunity: DispatchedOpportunity): void {
  const marketId = String(opportunity.marketId ?? opportunity.id ?? "?");
  const type = String(opportunity.type ?? "?");
  const edge = Number(opportunity.edge ?? 0);
  const rank = opportunity.rank ?? "?";
  console.log("DISPATCH START", { marketId, type, edge, rank });

  try {
    console.log("CALLING EXECUTION ENGINE", marketId);
    evaluateOpportunity(opportunity);
  } catch (err) {
    console.log("DISPATCH EARLY EXIT", "dispatch threw error");
    console.error("[ExecutionDispatcher] Dispatch error:", err);
  }
}
