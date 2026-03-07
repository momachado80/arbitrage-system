/**
 * Execution Dispatcher — connects scanner output to shadow execution pipeline.
 * Dispatches opportunities to the shadow simulation for evaluation.
 */

import { evaluateOpportunity } from "./shadowSimulationService";

export type DispatchedOpportunity = Record<string, unknown>;

export function dispatchOpportunity(opportunity: DispatchedOpportunity): void {
  try {
    evaluateOpportunity(opportunity);
  } catch (err) {
    console.error("[ExecutionDispatcher] Dispatch error:", err);
  }
}
