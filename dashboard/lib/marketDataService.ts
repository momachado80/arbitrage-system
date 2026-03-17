/**
 * Market Data Service — bootstrap e refresh de mercados.
 * Instrumentação explícita; cleanup obrigatório de locks; timeouts defensivos.
 */

import {
  fetchAllMarkets,
  getCachedMarkets,
  type NormalizedMarket,
} from "./polymarketClient";

const REFRESH_INTERVAL_MS = 5_000;
const REFRESH_TIMEOUT_MS = 12_000;
const STALE_REFRESH_MS = REFRESH_TIMEOUT_MS * 2;

let markets: NormalizedMarket[] = [];
let lastRefresh = 0;
let refreshing = false;
let loopStarted = false;
let fetchCount = 0;
let lastError: string | null = null;

// Instrumentação explícita
let bootstrapAttempted = false;
let bootstrapCompleted = false;
let bootstrapFailed = false;
let bootstrapErrorMessage: string | null = null;
let lastBootstrapAt: string | null = null;
let bootstrapPhase: "idle" | "refresh_started" | "fetch_completed" | "fetch_failed" = "idle";
let refreshAttemptedCount = 0;
let refreshSuccessCount = 0;
let refreshFailureCount = 0;
let lastRefreshError: string | null = null;
let refreshStartedAt: number | null = null;
let lastRefreshCompletedAt: string | null = null;
let lastRefreshFailedAt: string | null = null;

function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    p,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timeout after ${ms}ms`)), ms)
    ),
  ]);
}

async function refresh(): Promise<void> {
  if (refreshing) {
    const stuckMs = refreshStartedAt != null ? Date.now() - refreshStartedAt : 0;
    if (stuckMs > STALE_REFRESH_MS) {
      console.error(`[MarketDataService] STALE REFRESH (${stuckMs}ms) — force cleanup`);
      refreshing = false;
      refreshFailureCount++;
      if (!bootstrapCompleted && bootstrapAttempted) {
        bootstrapFailed = true;
        bootstrapErrorMessage = `refresh hung ${stuckMs}ms`;
      }
      lastRefreshError = `stale_refresh_${stuckMs}ms`;
      lastRefreshFailedAt = new Date().toISOString();
    }
    return;
  }
  refreshing = true;
  refreshStartedAt = Date.now();
  refreshAttemptedCount++;
  bootstrapPhase = "refresh_started";

  try {
    markets = await withTimeout(fetchAllMarkets(), REFRESH_TIMEOUT_MS, "market refresh");
    bootstrapPhase = "fetch_completed";
    lastRefresh = Date.now();
    fetchCount++;
    lastError = null;
    lastRefreshError = null;
    refreshSuccessCount++;
    lastRefreshCompletedAt = new Date().toISOString();
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
    bootstrapPhase = "fetch_failed";
    lastError = msg;
    lastRefreshError = msg;
    refreshFailureCount++;
    lastRefreshFailedAt = new Date().toISOString();
    if (!bootstrapCompleted && bootstrapAttempted) {
      bootstrapFailed = true;
      bootstrapErrorMessage = msg;
    }
    console.error(`[MarketDataService] Refresh failed: ${msg}`);
    const cached = getCachedMarkets();
    if (cached.length > 0) markets = cached;
  } finally {
    refreshing = false;
    refreshStartedAt = null;
  }
}

function startLoop(): void {
  if (loopStarted) return;
  loopStarted = true;
  bootstrapAttempted = true;
  lastBootstrapAt = new Date().toISOString();
  console.log("[MarketDataService] Bootstrap started");
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
  const stuckMs = refreshStartedAt != null ? Date.now() - refreshStartedAt : 0;
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
    bootstrapPhase,
    refreshStartedAt,
    refreshStuckMs: stuckMs,
    lastRefreshCompletedAt,
    lastRefreshFailedAt,
  };
}
