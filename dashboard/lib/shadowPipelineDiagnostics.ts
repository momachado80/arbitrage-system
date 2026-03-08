/**
 * Shadow Pipeline Diagnostics — in-memory counters for observability.
 * Tracks where opportunities die in the post-dispatch execution pipeline.
 * No persistence, no business logic changes.
 */

export type EarlyExitReason =
  | "NO_ENABLED_PROFILES"
  | "TRADE_ALREADY_ACTIVE"
  | "RECOMMENDED_CAPITAL_ZERO_OR_NEGATIVE"
  | "CONFIDENCE_BELOW_THRESHOLD"
  | "INSUFFICIENT_CAPITAL_OR_EXPOSURE_LIMIT"
  | "NET_EDGE_BELOW_THRESHOLD"
  | "FILL_REJECTED"
  | "FILLED_CAPITAL_ZERO"
  | "DISPATCH_CATCH_ERROR"
  | "EVALUATION_CATCH_ERROR";

const counters = {
  totalDispatches: 0,
  totalEvaluateCalls: 0,
  totalExecutionCalls: 0,
  totalShadowTradesOpened: 0,
  totalFilteredByEEV: 0,
  totalPassedEEVFilter: 0,
  earlyExitCounts: {} as Record<string, number>,
};

function ensureKey(reason: string): void {
  if (!(reason in counters.earlyExitCounts)) {
    counters.earlyExitCounts[reason] = 0;
  }
}

export function incrementDispatch(): void {
  counters.totalDispatches++;
}

export function incrementEvaluateCall(): void {
  counters.totalEvaluateCalls++;
}

export function incrementExecutionCall(): void {
  counters.totalExecutionCalls++;
}

export function incrementShadowTradeOpened(): void {
  counters.totalShadowTradesOpened++;
}

export function incrementEarlyExit(reason: EarlyExitReason | string): void {
  ensureKey(reason);
  counters.earlyExitCounts[reason]++;
}

export function incrementFilteredByEEV(): void {
  counters.totalFilteredByEEV++;
}

export function incrementPassedEEVFilter(): void {
  counters.totalPassedEEVFilter++;
}

export function getPipelineDiagnostics(): {
  totalDispatches: number;
  totalEvaluateCalls: number;
  totalExecutionCalls: number;
  totalShadowTradesOpened: number;
  totalFilteredByEEV: number;
  totalPassedEEVFilter: number;
  earlyExitCounts: Record<string, number>;
  timestamp: string;
} {
  return {
    ...counters,
    earlyExitCounts: { ...counters.earlyExitCounts },
    timestamp: new Date().toISOString(),
  };
}
