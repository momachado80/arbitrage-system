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

const CYCLE_INTERVAL_MS = 5_000;

async function runBot(): Promise<void> {
  console.log("[BotRunner] Arbitrage worker started (interval: 5s)");

  while (true) {
    try {
      const markets = getAllMarkets();
      if (markets.length > 0) {
        const edges = scanMarkets(markets);
        const ranked = rankOpportunities(edges);

        for (const opp of ranked) {
          dispatchOpportunity(opp as unknown as Record<string, unknown>);
        }
      }

      const graphOpps = getGraphOpportunities();
      for (const opp of graphOpps) {
        dispatchOpportunity(opp as unknown as Record<string, unknown>);
      }
    } catch (err) {
      console.warn("[BotRunner] Cycle error:", err instanceof Error ? err.message : err);
    }

    await new Promise((r) => setTimeout(r, CYCLE_INTERVAL_MS));
  }
}

runBot();
