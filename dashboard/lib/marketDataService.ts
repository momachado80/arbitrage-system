import {
  fetchAllMarkets,
  getCachedMarkets,
  type NormalizedMarket,
} from "./polymarketClient";

const REFRESH_INTERVAL_MS = 5_000;

let markets: NormalizedMarket[] = [];
let lastRefresh = 0;
let refreshing = false;
let loopStarted = false;
let fetchCount = 0;
let lastError: string | null = null;

async function refresh(): Promise<void> {
  if (refreshing) return;
  refreshing = true;
  try {
    markets = await fetchAllMarkets();
    lastRefresh = Date.now();
    fetchCount++;
    lastError = null;
  } catch (err) {
    lastError = err instanceof Error ? err.message : "unknown error";
    console.error(`[MarketDataService] Refresh failed: ${lastError}`);
    const cached = getCachedMarkets();
    if (cached.length > 0) markets = cached;
  } finally {
    refreshing = false;
  }
}

function startLoop(): void {
  if (loopStarted) return;
  loopStarted = true;
  console.log("[MarketDataService] Background refresh loop started");
  refresh();
  setInterval(refresh, REFRESH_INTERVAL_MS);
}

export function ensureRunning(): void {
  startLoop();
}

export function getAllMarkets(): NormalizedMarket[] {
  ensureRunning();
  return markets;
}

export function getMarketById(id: string): NormalizedMarket | undefined {
  ensureRunning();
  return markets.find((m) => m.id === id);
}

export function getServiceStats() {
  return {
    marketsTracked: markets.length,
    lastRefreshMs: lastRefresh,
    fetchCount,
    lastError,
    isRefreshing: refreshing,
  };
}
