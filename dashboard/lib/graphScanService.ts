import { getAllMarkets } from "./marketDataService";
import { buildClusters } from "./marketRelationBuilder";
import { scanClusters } from "./graphArbitrageEngine";
import {
  rankGraphOpportunities,
  computeGraphSummary,
  type RankedGraphOpportunity,
  type GraphSummary,
} from "./graphOpportunityEngine";

const SCAN_INTERVAL_MS = 6_000;

let cachedRanked: RankedGraphOpportunity[] = [];
let cachedSummary: GraphSummary = {
  graphOpportunitiesDetected: 0,
  averageConfidence: 0,
  averageEdge: 0,
  numberOfClustersScanned: 0,
  byType: {},
};
let lastScanMs = 0;
let scanning = false;
let loopStarted = false;

async function runScan(): Promise<void> {
  if (scanning) return;
  scanning = true;
  try {
    const markets = getAllMarkets();
    if (markets.length === 0) return;

    const clusters = buildClusters(markets);
    const opportunities = scanClusters(clusters);
    const ranked = rankGraphOpportunities(opportunities);
    const summary = computeGraphSummary(ranked, clusters.length);

    cachedRanked = ranked;
    cachedSummary = summary;
    lastScanMs = Date.now();
  } catch (err) {
    console.error("[GraphScanService] Scan failed:", err);
  } finally {
    scanning = false;
  }
}

function startLoop(): void {
  if (loopStarted) return;
  loopStarted = true;
  console.log("[GraphScanService] Background graph scan loop started");
  setTimeout(runScan, 2000);
  setInterval(runScan, SCAN_INTERVAL_MS);
}

export function ensureGraphScanning(): void {
  startLoop();
}

export function getGraphOpportunities(): RankedGraphOpportunity[] {
  ensureGraphScanning();
  return cachedRanked;
}

export function getGraphSummary(): GraphSummary {
  ensureGraphScanning();
  return cachedSummary;
}

export function getGraphScanStats() {
  return {
    lastScanMs,
    isScanning: scanning,
    opportunitiesCount: cachedRanked.length,
  };
}
