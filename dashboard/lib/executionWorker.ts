/**
 * Execution Worker — runs scanner and dispatcher continuously in background.
 * Independent of dashboard requests; ensures opportunities are always processed.
 */

import { getAllMarkets } from "./marketDataService";
import { scanMarkets } from "./probabilityScanner";
import { rankOpportunities } from "./opportunityEngine";
import { getGraphOpportunities } from "./graphScanService";
import { dispatchOpportunity } from "./executionDispatcher";
import { getExecutionWorkerLock } from "./nodeProcessRuntimeState";

const CYCLE_INTERVAL_MS = 5_000;

async function executionCycle(): Promise<void> {
  try {
    const markets = getAllMarkets();
    if (markets.length === 0) return;

    const edges = scanMarkets(markets);
    const ranked = rankOpportunities(edges);
    const graphOpps = getGraphOpportunities();

    for (const opp of ranked) {
      dispatchOpportunity(opp as unknown as Record<string, unknown>);
    }
    for (const opp of graphOpps) {
      dispatchOpportunity(opp as unknown as Record<string, unknown>);
    }
  } catch (err) {
    console.warn("[ExecutionWorker] Cycle error:", err instanceof Error ? err.message : err);
  }
}

export function startExecutionWorker(): void {
  const lock = getExecutionWorkerLock();
  if (lock.started) {
    console.log("[ExecutionWorker] loop_skip_already_active_globalThis");
    return;
  }
  lock.started = true;
  console.log("[ExecutionWorker] Background execution worker started (effective; interval: 5s)");
  void executionCycle();
  setInterval(() => {
    void executionCycle();
  }, CYCLE_INTERVAL_MS);
}
