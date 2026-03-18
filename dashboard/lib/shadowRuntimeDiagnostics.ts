/**
 * Shadow Runtime Diagnostics — bootstrap and lifecycle observability.
 * Diagnoses why upstream/markets/cycles are dead. In-memory only.
 *
 * CAUSA RAIZ (quando bootstrap/loop mortos):
 * 1. refresh() pendente: fetchAllMarkets() sem timeout → promise nunca resolve → attempted=1, success=0, failure=0.
 * 2. runCycle nunca chamado: recordShadowLoopStarted só disparava no .then; fetch para localhost pendente → loopStarted=false.
 * 3. instrumentation bloqueava HTTP server 15s → fetch /api/opportunities falhava (server não ready).
 * 4. INITIAL_DELAY_MS=6s → primeiro audit antes de 6s mostra loopStarted=false.
 */

let serviceBootAttempted = false;
let serviceBootCompleted = false;
let serviceBootFailed = false;
let serviceBootErrorMessage: string | null = null;
let shadowLoopStarted = false;
let shadowLoopHeartbeatCount = 0;
let lastShadowLoopStartedAt: string | null = null;
let lastShadowLoopCompletedAt: string | null = null;
let instrumentationRanAt: string | null = null;
let schedulerScheduledAt: string | null = null;

export function recordShadowBootAttempted(): void {
  serviceBootAttempted = true;
}

export function recordShadowBootCompleted(): void {
  serviceBootCompleted = true;
  serviceBootFailed = false;
}

export function recordShadowBootFailed(msg: string): void {
  serviceBootFailed = true;
  serviceBootErrorMessage = msg;
}

export function recordShadowLoopStarted(): void {
  shadowLoopStarted = true;
  shadowLoopHeartbeatCount++;
  lastShadowLoopStartedAt = new Date().toISOString();
}

export function recordShadowLoopCompleted(): void {
  lastShadowLoopCompletedAt = new Date().toISOString();
}

export function recordInstrumentationRan(): void {
  if (!instrumentationRanAt) instrumentationRanAt = new Date().toISOString();
}

export function recordSchedulerScheduled(): void {
  if (!schedulerScheduledAt) schedulerScheduledAt = new Date().toISOString();
}

export function getInstrumentationRanAt(): string | null {
  return instrumentationRanAt;
}

export function getSchedulerScheduledAt(): string | null {
  return schedulerScheduledAt;
}

export interface ShadowRuntimeDiagnostics {
  serviceBootAttempted: boolean;
  serviceBootCompleted: boolean;
  serviceBootFailed: boolean;
  serviceBootErrorMessage: string | null;
  shadowLoopStarted: boolean;
  shadowLoopHeartbeatCount: number;
  lastShadowLoopStartedAt: string | null;
  lastShadowLoopCompletedAt: string | null;
  instrumentationRanAt: string | null;
  schedulerScheduledAt: string | null;
  /** Gate explícito quando loop não iniciou */
  loopBlockReason: string | null;
  marketBootstrapAttempted: boolean;
  marketBootstrapCompleted: boolean;
  marketBootstrapFailed: boolean;
  marketBootstrapErrorMessage: string | null;
  lastMarketBootstrapAt: string | null;
  /** pending | completed | failed */
  marketBootstrapStatus: "pending" | "completed" | "failed";
  marketRefreshAttemptedCount: number;
  marketRefreshSuccessCount: number;
  marketRefreshFailureCount: number;
  lastMarketRefreshError: string | null;
  /** true quando attempted > 0 e success+failure < attempted */
  marketRefreshPending: boolean;
  schedulerRegistered: boolean;
  intervalMs: number;
  runtimeEnvironmentSummary: Record<string, string | number | boolean>;
  generatedAt: string;
  bootstrapOperationalTroubleshooting?: {
    bootstrapPhase: string;
    refreshStuckMs: number;
    lockStuck: boolean;
    schedulerRegisteredButBlocked: boolean;
    timeoutOperationalOccurred: boolean;
    lastCompletedStep: string | null;
    lastFailedStep: string | null;
    lastStepStartedAt: string | null;
    lastStepCompletedAt: string | null;
    lastRefreshErrorMessage: string | null;
    refreshAttemptDurationMs: number | null;
    bootstrapAttemptDurationMs: number | null;
  };
}

function deriveLoopBlockReason(
  marketStats: { isRefreshing?: boolean; refreshAttemptedCount?: number; refreshSuccessCount?: number; refreshFailureCount?: number; refreshStuckMs?: number }
): string | null {
  if (shadowLoopStarted) return null;
  if (!serviceBootCompleted) return "shadow_boot_not_completed";
  if (!schedulerScheduledAt) return "scheduler_not_scheduled";
  const attempted = marketStats.refreshAttemptedCount ?? 0;
  const success = marketStats.refreshSuccessCount ?? 0;
  const failure = marketStats.refreshFailureCount ?? 0;
  if (attempted === 0) return "market_refresh_never_called";
  if (success + failure < attempted && marketStats.isRefreshing) return "market_refresh_pending";
  if (success + failure < attempted) {
    const stuck = marketStats.refreshStuckMs ?? 0;
    return stuck > 15000 ? `market_refresh_hung_${Math.round(stuck / 1000)}s` : "market_refresh_hung";
  }
  return "awaiting_first_cycle_or_cycle_not_yet_invoked";
}

export function getShadowRuntimeDiagnostics(
  marketStats: {
    marketBootstrapAttempted?: boolean;
    marketBootstrapCompleted?: boolean;
    marketBootstrapFailed?: boolean;
    marketBootstrapErrorMessage?: string | null;
    lastMarketBootstrapAt?: string | null;
    refreshAttemptedCount?: number;
    refreshSuccessCount?: number;
    refreshFailureCount?: number;
    lastRefreshError?: string | null;
    isRefreshing?: boolean;
    bootstrapPhase?: string;
    refreshStuckMs?: number;
    lastCompletedStep?: string | null;
    lastFailedStep?: string | null;
    lastStepStartedAt?: string | null;
    lastStepCompletedAt?: string | null;
    lastRefreshErrorMessage?: string | null;
    refreshAttemptDurationMs?: number | null;
    bootstrapAttemptDurationMs?: number | null;
    timeoutOperationalOccurred?: boolean;
  },
  scheduler: { registered: boolean; intervalMs: number }
): ShadowRuntimeDiagnostics {
  const attempted = marketStats.refreshAttemptedCount ?? 0;
  const success = marketStats.refreshSuccessCount ?? 0;
  const failure = marketStats.refreshFailureCount ?? 0;
  const refreshPending = attempted > 0 && success + failure < attempted;
  const bootstrapCompleted = marketStats.marketBootstrapCompleted ?? false;
  const bootstrapFailed = marketStats.marketBootstrapFailed ?? false;
  const bootstrapStatus: "pending" | "completed" | "failed" =
    bootstrapCompleted ? "completed" : bootstrapFailed ? "failed" : "pending";
  const stuckMs = marketStats.refreshStuckMs ?? 0;
  const lockStuck = refreshPending && stuckMs > 20000;

  return {
    serviceBootAttempted,
    serviceBootCompleted,
    serviceBootFailed,
    serviceBootErrorMessage,
    shadowLoopStarted,
    shadowLoopHeartbeatCount,
    lastShadowLoopStartedAt,
    lastShadowLoopCompletedAt,
    instrumentationRanAt,
    schedulerScheduledAt,
    loopBlockReason: deriveLoopBlockReason(marketStats),
    marketBootstrapAttempted: marketStats.marketBootstrapAttempted ?? false,
    marketBootstrapCompleted: bootstrapCompleted,
    marketBootstrapFailed: bootstrapFailed,
    marketBootstrapErrorMessage: marketStats.marketBootstrapErrorMessage ?? null,
    lastMarketBootstrapAt: marketStats.lastMarketBootstrapAt ?? null,
    marketBootstrapStatus: bootstrapStatus,
    marketRefreshAttemptedCount: attempted,
    marketRefreshSuccessCount: success,
    marketRefreshFailureCount: failure,
    lastMarketRefreshError: marketStats.lastRefreshError ?? null,
    marketRefreshPending: refreshPending,
    schedulerRegistered: scheduler.registered,
    intervalMs: scheduler.intervalMs,
    runtimeEnvironmentSummary: {
      nodeVersion: process.version,
      env: typeof process.env.NODE_ENV === "string" ? process.env.NODE_ENV : "unknown",
      hasPort: typeof process.env.PORT === "string",
      hasRailwayDomain: typeof process.env.RAILWAY_PUBLIC_DOMAIN === "string",
    },
    generatedAt: new Date().toISOString(),
    bootstrapOperationalTroubleshooting: {
      bootstrapPhase: marketStats.bootstrapPhase ?? "idle",
      refreshStuckMs: stuckMs,
      lockStuck,
      schedulerRegisteredButBlocked: !!scheduler.registered && !shadowLoopStarted && refreshPending,
      timeoutOperationalOccurred:
        (marketStats.lastRefreshError ?? "").includes("timeout") ||
        (marketStats.timeoutOperationalOccurred ?? false),
      lastCompletedStep: marketStats.lastCompletedStep ?? null,
      lastFailedStep: marketStats.lastFailedStep ?? null,
      lastStepStartedAt: marketStats.lastStepStartedAt ?? null,
      lastStepCompletedAt: marketStats.lastStepCompletedAt ?? null,
      lastRefreshErrorMessage: marketStats.lastRefreshErrorMessage ?? null,
      refreshAttemptDurationMs: marketStats.refreshAttemptDurationMs ?? null,
      bootstrapAttemptDurationMs: marketStats.bootstrapAttemptDurationMs ?? null,
    },
  };
}
