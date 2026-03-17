import {
  fetchAllMarkets,
  getCachedMarkets,
  type NormalizedMarket,
} from "./polymarketClient";

const REFRESH_INTERVAL_MS = 5_000;
const REFRESH_TIMEOUT_MS = 12_000;

let markets: NormalizedMarket[] = [];
let lastRefresh = 0;
let refreshing = false;
let loopStarted = false;
let fetchCount = 0;
let lastError: string | null = null;

// Bootstrap and refresh diagnostics (for shadowRuntimeDiagnostics / marketSourceDiagnostics)
let bootstrapAttempted = false;
let bootstrapCompleted = false;
let bootstrapFailed = false;
let bootstrapErrorMessage: string | null = null;
let lastBootstrapAt: string | null = null;
let refreshAttemptedCount = 0;
let refreshSuccessCount = 0;
let refreshFailureCount = 0;
let lastRefreshError: string | null = null;

function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    p,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timeout after ${ms}ms`)), ms)
    ),
  ]);
}

async function refresh(): Promise<void> {
  if (refreshing) return;
  refreshing = true;
  refreshAttemptedCount++;
  try {
    markets = await withTimeout(fetchAllMarkets(), REFRESH_TIMEOUT_MS, "market refresh");
    lastRefresh = Date.now();
    fetchCount++;
    lastError = null;
    lastRefreshError = null;
    refreshSuccessCount++;
    if (!bootstrapCompleted && markets.length > 0) {
      bootstrapCompleted = true;
      bootstrapFailed = false;
      bootstrapErrorMessage = null;
      lastBootstrapAt = new Date().toISOString();
    } else if (!bootstrapCompleted && bootstrapAttempted && markets.length === 0) {
      bootstrapFailed = true;
      bootstrapErrorMessage = "no markets returned";
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : "unknown error";
    lastError = msg;
    lastRefreshError = msg;
    refreshFailureCount++;
    if (!bootstrapCompleted && bootstrapAttempted) {
      bootstrapFailed = true;
      bootstrapErrorMessage = msg;
    }
    console.error(`[MarketDataService] Refresh failed: ${msg}`);
    const cached = getCachedMarkets();
    if (cached.length > 0) markets = cached;
  } finally {
    refreshing = false;
  }
}

function startLoop(): void {
  if (loopStarted) return;
  loopStarted = true;
  bootstrapAttempted = true;
  lastBootstrapAt = new Date().toISOString();
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
    marketBootstrapAttempted: bootstrapAttempted,
    marketBootstrapCompleted: bootstrapCompleted,
    marketBootstrapFailed: bootstrapFailed,
    marketBootstrapErrorMessage: bootstrapErrorMessage,
    lastMarketBootstrapAt: lastBootstrapAt,
    refreshAttemptedCount,
    refreshSuccessCount,
    refreshFailureCount,
    lastRefreshError,
  };
}
