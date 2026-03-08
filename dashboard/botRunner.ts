/**
 * Bot Runner — persistent arbitrage worker process.
 * Runs scanner and execution pipeline continuously, independent of Next.js.
 * Use: npm run bot (or as Railway worker service)
 */

import { getAllMarkets } from "./lib/marketDataService";
import { scanMarkets } from "./lib/probabilityScanner";
import { rankOpportunities } from "./lib/opportunityEngine";
import { getGraphOpportunities } from "./lib/graphScanService";
import { dispatchOpportunity } from "./lib/executionDispatcher";
import { getPipelineDiagnostics } from "./lib/shadowPipelineDiagnostics";
import { runRankingComparisonDiagnostics } from "./lib/rankingComparisonDiagnostics";

console.log("BOT RUNNER FILE LOADED");

const CYCLE_INTERVAL_MS = 5_000;
const SNAPSHOT_INTERVAL_MS = 60_000;

function startDiagnosticsSnapshot(): void {
  setInterval(() => {
    const d = getPipelineDiagnostics();
    console.log("[DIAGNOSTICS] WORKER SNAPSHOT", {
      totalDispatches: d.totalDispatches,
      totalEvaluateCalls: d.totalEvaluateCalls,
      totalExecutionCalls: d.totalExecutionCalls,
      totalShadowTradesOpened: d.totalShadowTradesOpened,
      earlyExitCounts: d.earlyExitCounts,
      timestamp: d.timestamp,
    });
  }, SNAPSHOT_INTERVAL_MS);
}

async function runBot(): Promise<void> {
  console.log("ARBITRAGE WORKER ONLINE");
  startDiagnosticsSnapshot();

  while (true) {
    try {
      const markets = getAllMarkets();
      console.log("MARKETS FETCHED:", markets.length);

      if (markets.length > 0) {
        const edges = scanMarkets(markets);
        console.log("OPPORTUNITIES DETECTED:", edges.length);

        const ranked = rankOpportunities(edges);
        console.log("OPPORTUNITIES RANKED:", ranked.length);

        if (ranked.length > 0) {
          runRankingComparisonDiagnostics(ranked as unknown as Record<string, unknown>[], "standard", 15);
        }

        for (const opp of ranked) {
          console.log("DISPATCHING OPPORTUNITY", opp);
          dispatchOpportunity(opp as unknown as Record<string, unknown>);
        }
      }

      const graphOpps = getGraphOpportunities();
      console.log("GRAPH OPPORTUNITIES:", graphOpps.length);

      if (graphOpps.length > 0) {
        runRankingComparisonDiagnostics(graphOpps as unknown as Record<string, unknown>[], "graph", 15);
      }

      for (const opp of graphOpps) {
        console.log("DISPATCHING OPPORTUNITY", opp);
        dispatchOpportunity(opp as unknown as Record<string, unknown>);
      }
    } catch (err) {
      console.error("WORKER LOOP ERROR:", err);
    }

    await new Promise((r) => setTimeout(r, CYCLE_INTERVAL_MS));
  }
}

runBot();
