/**
 * Execution Worker — runs scanner and dispatcher continuously in background.
 * Independent of dashboard requests; ensures opportunities are always processed.
 */

import { getAllMarkets } from "./marketDataService";
import { scanMarkets } from "./probabilityScanner";
import { rankOpportunities } from "./opportunityEngine";
import { getGraphOpportunities } from "./graphScanService";
import { dispatchOpportunity } from "./executionDispatcher";

const CYCLE_INTERVAL_MS = 5_000;
let workerStarted = false;

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
  if (workerStarted) return;
  workerStarted = true;
  console.log("[ExecutionWorker] Background execution worker started (interval: 5s)");
  executionCycle();
  setInterval(executionCycle, CYCLE_INTERVAL_MS);
}
