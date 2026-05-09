/**
 * Market Data Service — bootstrap e refresh de mercados.
 * Etapas explícitas, rastreamento persistido, timeouts por etapa, guard contra stuck.
 * Estado em globalThis: um refresh em voo por processo (reutilização de promise + dedupe).
 */

import {
  fetchAllMarkets,
  getCachedMarkets,
  type RefreshStepReporter,
} from "./polymarketClient";
import { getMarketDataRuntime } from "./nodeProcessRuntimeState";

const REFRESH_INTERVAL_MS = 5_000;
const STUCK_REFRESH_GUARD_MS = 120_000;

const FETCH_STANDARD_TIMEOUT_MS = 60_000;
const PARSE_TIMEOUT_MS = 15_000;
const MERGE_TIMEOUT_MS = 5_000;

function forceCleanupStuck(reason: string): void {
  const st = getMarketDataRuntime();
  console.error(`[MarketDataService] STUCK REFRESH GUARD (${reason}) — force cleanup`);
  st.refreshing = false;
  st.refreshStartedAt = null;
  st.refreshInFlight = null;
  if (!st.bootstrapCompleted && st.bootstrapAttempted) {
    st.bootstrapFailed = true;
    st.bootstrapErrorMessage = reason;
  }
  st.lastRefreshError = reason;
  st.lastRefreshFailedAt = new Date().toISOString();
  st.refreshFailureCount++;
  st.timeoutOperationalOccurred = true;
  st.lastFailedStep = st.lastCompletedStep ?? "unknown_stuck_step";
}

function setStep(step: string): void {
  const st = getMarketDataRuntime();
  st.lastCompletedStep = step;
  st.lastStepStartedAt = new Date().toISOString();
}

function completeStep(step: string): void {
  const st = getMarketDataRuntime();
  st.lastCompletedStep = step;
  st.lastStepCompletedAt = new Date().toISOString();
}

function refresh(): Promise<void> {
  const st = getMarketDataRuntime();
  const m = st.refreshMetrics;
  m.refreshRequested++;
  const inFlightCount = st.refreshInFlight != null ? 1 : 0;
  console.log(
    `[MarketDataService] refresh_requested total_req=${m.refreshRequested} in_flight_refresh_count=${inFlightCount}`
  );

  if (st.refreshInFlight) {
    const stuckMs = st.refreshStartedAt != null ? Date.now() - st.refreshStartedAt : 0;
    if (stuckMs > STUCK_REFRESH_GUARD_MS) {
      forceCleanupStuck(`refresh_hung_${stuckMs}ms`);
    } else {
      m.refreshDeduped++;
      console.log(
        `[MarketDataService] refresh_deduped_or_reused deduped_total=${m.refreshDeduped} in_flight_refresh_count=1`
      );
      return st.refreshInFlight;
    }
  }

  m.refreshStartedEffective++;
  console.log(
    `[MarketDataService] refresh_started effective=${m.refreshStartedEffective} | refresh_begin`
  );

  const workStartedAt = Date.now();
  st.refreshInFlight = (async () => {
    st.refreshing = true;
    st.refreshStartedAt = Date.now();
    st.refreshAttemptedCount++;
    st.bootstrapPhase = "refresh_started";
    st.lastFailedStep = null;
    st.lastRefreshErrorMessage = null;
    st.timeoutOperationalOccurred = false;

    const reportStep: RefreshStepReporter = (step: string, detail?: { count?: number }) => {
      completeStep(step);
      if (detail?.count != null) {
        console.log(`[MarketDataService] ${step} count=${detail.count}`);
      } else {
        console.log(`[MarketDataService] ${step}`);
      }
    };

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
      st.markets = rawMarkets;
      completeStep("publish_done");
      console.log(`[MarketDataService] publish_done count=${st.markets.length}`);

      st.lastRefresh = Date.now();
      st.fetchCount++;
      st.lastError = null;
      st.lastRefreshError = null;
      st.refreshSuccessCount++;
      st.lastRefreshCompletedAt = new Date().toISOString();
      st.refreshAttemptDurationMs = Date.now() - (st.refreshStartedAt ?? 0);
      m.refreshCompleted++;
      m.lastDurationMs = Date.now() - workStartedAt;
      console.log(`[MarketDataService] refresh_completed duration_ms=${m.lastDurationMs}`);

      if (!st.bootstrapCompleted && st.markets.length > 0) {
        st.bootstrapCompleted = true;
        st.bootstrapFailed = false;
        st.bootstrapErrorMessage = null;
        st.lastBootstrapAt = new Date().toISOString();
        st.bootstrapAttemptDurationMs = st.refreshAttemptDurationMs;
      } else if (!st.bootstrapCompleted && st.bootstrapAttempted && st.markets.length === 0) {
        st.bootstrapFailed = true;
        st.bootstrapErrorMessage = "no markets returned";
      }

      setStep("refresh_finalize_done");
      completeStep("refresh_finalize_done");
      console.log("[MarketDataService] refresh_finalize_done");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const isTimeout = msg.includes("timeout");
      st.bootstrapPhase = "fetch_failed";
      st.lastError = msg;
      st.lastRefreshError = msg;
      st.lastRefreshErrorMessage = msg;
      st.refreshFailureCount++;
      st.lastRefreshFailedAt = new Date().toISOString();
      st.refreshAttemptDurationMs = st.refreshStartedAt != null ? Date.now() - st.refreshStartedAt : null;
      m.refreshFailed++;
      m.lastDurationMs = Date.now() - workStartedAt;
      st.lastFailedStep = st.lastCompletedStep ?? "unknown";
      if (isTimeout) st.timeoutOperationalOccurred = true;

      if (!st.bootstrapCompleted && st.bootstrapAttempted) {
        st.bootstrapFailed = true;
        st.bootstrapErrorMessage = msg;
      }
      console.error(`[MarketDataService] refresh_failed: ${msg} duration_ms=${m.lastDurationMs}`);
      const cached = getCachedMarkets();
      if (cached.length > 0) st.markets = cached;
    } finally {
      st.refreshing = false;
      st.refreshStartedAt = null;
      st.refreshInFlight = null;
      console.log("[MarketDataService] refresh_finalize in_flight_refresh_count=0");
    }
  })();

  return st.refreshInFlight;
}

function startLoop(): void {
  const st = getMarketDataRuntime();
  if (st.loopStarted) {
    console.log("[MarketDataService] bootstrap_skip loop_already_active_globalThis");
    return;
  }
  st.loopStarted = true;
  st.bootstrapAttempted = true;
  st.lastBootstrapAt = new Date().toISOString();
  console.log("[MarketDataService] Bootstrap started (effective; single process loop)");
  void refresh();
  setInterval(() => {
    void refresh();
  }, REFRESH_INTERVAL_MS);
}

export function ensureRunning(): void {
  startLoop();
}

export function getAllMarkets(): import("./polymarketClient").NormalizedMarket[] {
  ensureRunning();
  return getMarketDataRuntime().markets;
}

export function getMarketById(id: string): import("./polymarketClient").NormalizedMarket | undefined {
  ensureRunning();
  return getMarketDataRuntime().markets.find((m) => m.id === id);
}

export function getServiceStats() {
  const st = getMarketDataRuntime();
  const stuckMs = st.refreshStartedAt != null ? Date.now() - st.refreshStartedAt : 0;
  if (stuckMs > STUCK_REFRESH_GUARD_MS) {
    forceCleanupStuck(`guard_triggered_${stuckMs}ms`);
  }
  const rm = st.refreshMetrics;
  return {
    marketsTracked: st.markets.length,
    lastRefreshMs: st.lastRefresh,
    fetchCount: st.fetchCount,
    lastError: st.lastError,
    isRefreshing: st.refreshing,
    marketBootstrapAttempted: st.bootstrapAttempted,
    marketBootstrapCompleted: st.bootstrapCompleted,
    marketBootstrapFailed: st.bootstrapFailed,
    marketBootstrapErrorMessage: st.bootstrapErrorMessage,
    lastMarketBootstrapAt: st.lastBootstrapAt,
    refreshAttemptedCount: st.refreshAttemptedCount,
    refreshSuccessCount: st.refreshSuccessCount,
    refreshFailureCount: st.refreshFailureCount,
    lastRefreshError: st.lastRefreshError,
    bootstrapPhase: st.bootstrapPhase,
    refreshStartedAt: st.refreshStartedAt,
    refreshStuckMs: stuckMs,
    lastRefreshCompletedAt: st.lastRefreshCompletedAt,
    lastRefreshFailedAt: st.lastRefreshFailedAt,
    lastCompletedStep: st.lastCompletedStep,
    lastFailedStep: st.lastFailedStep,
    lastStepStartedAt: st.lastStepStartedAt,
    lastStepCompletedAt: st.lastStepCompletedAt,
    lastRefreshErrorMessage: st.lastRefreshErrorMessage,
    refreshAttemptDurationMs: st.refreshAttemptDurationMs,
    bootstrapAttemptDurationMs: st.bootstrapAttemptDurationMs,
    timeoutOperationalOccurred: st.timeoutOperationalOccurred,
    refreshMetrics: { ...rm },
    inFlightRefreshCount: st.refreshInFlight != null ? 1 : 0,
  };
}
