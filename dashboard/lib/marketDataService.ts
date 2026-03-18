/**
 * Market Data Service — bootstrap e refresh de mercados.
 * Etapas explícitas, rastreamento persistido, timeouts por etapa, guard contra stuck.
 */

import {
  fetchAllMarkets,
  getCachedMarkets,
  type RefreshStepReporter,
} from "./polymarketClient";

const REFRESH_INTERVAL_MS = 5_000;
const STUCK_REFRESH_GUARD_MS = 120_000;

const FETCH_STANDARD_TIMEOUT_MS = 60_000;
const PARSE_TIMEOUT_MS = 15_000;
const MERGE_TIMEOUT_MS = 5_000;

let markets: import("./polymarketClient").NormalizedMarket[] = [];
let lastRefresh = 0;
let refreshing = false;
let loopStarted = false;
let fetchCount = 0;
let lastError: string | null = null;

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

let lastCompletedStep: string | null = null;
let lastFailedStep: string | null = null;
let lastStepStartedAt: string | null = null;
let lastStepCompletedAt: string | null = null;
let lastRefreshErrorMessage: string | null = null;
let refreshAttemptDurationMs: number | null = null;
let bootstrapAttemptDurationMs: number | null = null;
let timeoutOperationalOccurred = false;

function forceCleanupStuck(reason: string): void {
  console.error(`[MarketDataService] STUCK REFRESH GUARD (${reason}) — force cleanup`);
  refreshing = false;
  refreshStartedAt = null;
  if (!bootstrapCompleted && bootstrapAttempted) {
    bootstrapFailed = true;
    bootstrapErrorMessage = reason;
  }
  lastRefreshError = reason;
  lastRefreshFailedAt = new Date().toISOString();
  refreshFailureCount++;
  timeoutOperationalOccurred = true;
  lastFailedStep = lastCompletedStep ?? "unknown_stuck_step";
}

function setStep(step: string): void {
  lastCompletedStep = step;
  lastStepStartedAt = new Date().toISOString();
}

function completeStep(step: string): void {
  lastCompletedStep = step;
  lastStepCompletedAt = new Date().toISOString();
}

async function refresh(): Promise<void> {
  if (refreshing) {
    const stuckMs = refreshStartedAt != null ? Date.now() - refreshStartedAt : 0;
    if (stuckMs > STUCK_REFRESH_GUARD_MS) {
      forceCleanupStuck(`refresh_hung_${stuckMs}ms`);
    }
    return;
  }

  refreshing = true;
  refreshStartedAt = Date.now();
  refreshAttemptedCount++;
  bootstrapPhase = "refresh_started";
  lastFailedStep = null;
  lastRefreshErrorMessage = null;
  timeoutOperationalOccurred = false;

  const reportStep: RefreshStepReporter = (step: string, detail?: { count?: number }) => {
    completeStep(step);
    if (detail?.count != null) {
      console.log(`[MarketDataService] ${step} count=${detail.count}`);
    } else {
      console.log(`[MarketDataService] ${step}`);
    }
  };

  console.log("[MarketDataService] refresh_begin");

  try {
    setStep("refresh_begin");
    completeStep("refresh_begin");

    const rawMarkets = await fetchAllMarkets({
      reportStep,
      fetchStandardTimeoutMs: FETCH_STANDARD_TIMEOUT_MS,
      parseTimeoutMs: PARSE_TIMEOUT_MS,
      mergeTimeoutMs: MERGE_TIMEOUT_MS,
    });

    setStep("publish_done");
    markets = rawMarkets;
    completeStep("publish_done");
    console.log(`[MarketDataService] publish_done count=${markets.length}`);

    lastRefresh = Date.now();
    fetchCount++;
    lastError = null;
    lastRefreshError = null;
    refreshSuccessCount++;
    lastRefreshCompletedAt = new Date().toISOString();
    refreshAttemptDurationMs = Date.now() - (refreshStartedAt ?? 0);

    if (!bootstrapCompleted && markets.length > 0) {
      bootstrapCompleted = true;
      bootstrapFailed = false;
      bootstrapErrorMessage = null;
      lastBootstrapAt = new Date().toISOString();
      bootstrapAttemptDurationMs = refreshAttemptDurationMs;
    } else if (!bootstrapCompleted && bootstrapAttempted && markets.length === 0) {
      bootstrapFailed = true;
      bootstrapErrorMessage = "no markets returned";
    }

    setStep("refresh_finalize_done");
    completeStep("refresh_finalize_done");
    console.log("[MarketDataService] refresh_finalize_done");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    const isTimeout = msg.includes("timeout");
    bootstrapPhase = "fetch_failed";
    lastError = msg;
    lastRefreshError = msg;
    lastRefreshErrorMessage = msg;
    refreshFailureCount++;
    lastRefreshFailedAt = new Date().toISOString();
    refreshAttemptDurationMs = refreshStartedAt != null ? Date.now() - refreshStartedAt : null;
    lastFailedStep = lastCompletedStep ?? "unknown";
    if (isTimeout) timeoutOperationalOccurred = true;

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

export function getAllMarkets(): import("./polymarketClient").NormalizedMarket[] {
  ensureRunning();
  return markets;
}

export function getMarketById(id: string): import("./polymarketClient").NormalizedMarket | undefined {
  ensureRunning();
  return markets.find((m) => m.id === id);
}

export function getServiceStats() {
  const stuckMs = refreshStartedAt != null ? Date.now() - refreshStartedAt : 0;
  if (stuckMs > STUCK_REFRESH_GUARD_MS) {
    forceCleanupStuck(`guard_triggered_${stuckMs}ms`);
  }
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
    lastCompletedStep,
    lastFailedStep,
    lastStepStartedAt,
    lastStepCompletedAt,
    lastRefreshErrorMessage,
    refreshAttemptDurationMs,
    bootstrapAttemptDurationMs,
    timeoutOperationalOccurred,
  };
}
