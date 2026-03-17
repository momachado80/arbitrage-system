/**
 * Shadow Runtime Diagnostics — bootstrap and lifecycle observability.
 * Diagnoses why upstream/markets/cycles are dead. In-memory only.
 */

let serviceBootAttempted = false;
let serviceBootCompleted = false;
let serviceBootFailed = false;
let serviceBootErrorMessage: string | null = null;
let shadowLoopStarted = false;
let shadowLoopHeartbeatCount = 0;
let lastShadowLoopStartedAt: string | null = null;
let lastShadowLoopCompletedAt: string | null = null;

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

export interface ShadowRuntimeDiagnostics {
  serviceBootAttempted: boolean;
  serviceBootCompleted: boolean;
  serviceBootFailed: boolean;
  serviceBootErrorMessage: string | null;
  shadowLoopStarted: boolean;
  shadowLoopHeartbeatCount: number;
  lastShadowLoopStartedAt: string | null;
  lastShadowLoopCompletedAt: string | null;
  marketBootstrapAttempted: boolean;
  marketBootstrapCompleted: boolean;
  marketBootstrapFailed: boolean;
  marketBootstrapErrorMessage: string | null;
  lastMarketBootstrapAt: string | null;
  marketRefreshAttemptedCount: number;
  marketRefreshSuccessCount: number;
  marketRefreshFailureCount: number;
  lastMarketRefreshError: string | null;
  schedulerRegistered: boolean;
  intervalMs: number;
  runtimeEnvironmentSummary: Record<string, string | number | boolean>;
  generatedAt: string;
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
  },
  scheduler: { registered: boolean; intervalMs: number }
): ShadowRuntimeDiagnostics {
  return {
    serviceBootAttempted,
    serviceBootCompleted,
    serviceBootFailed,
    serviceBootErrorMessage,
    shadowLoopStarted,
    shadowLoopHeartbeatCount,
    lastShadowLoopStartedAt,
    lastShadowLoopCompletedAt,
    marketBootstrapAttempted: marketStats.marketBootstrapAttempted ?? false,
    marketBootstrapCompleted: marketStats.marketBootstrapCompleted ?? false,
    marketBootstrapFailed: marketStats.marketBootstrapFailed ?? false,
    marketBootstrapErrorMessage: marketStats.marketBootstrapErrorMessage ?? null,
    lastMarketBootstrapAt: marketStats.lastMarketBootstrapAt ?? null,
    marketRefreshAttemptedCount: marketStats.refreshAttemptedCount ?? 0,
    marketRefreshSuccessCount: marketStats.refreshSuccessCount ?? 0,
    marketRefreshFailureCount: marketStats.refreshFailureCount ?? 0,
    lastMarketRefreshError: marketStats.lastRefreshError ?? null,
    schedulerRegistered: scheduler.registered,
    intervalMs: scheduler.intervalMs,
    runtimeEnvironmentSummary: {
      nodeVersion: process.version,
      env: typeof process.env.NODE_ENV === "string" ? process.env.NODE_ENV : "unknown",
      hasPort: typeof process.env.PORT === "string",
      hasRailwayDomain: typeof process.env.RAILWAY_PUBLIC_DOMAIN === "string",
    },
    generatedAt: new Date().toISOString(),
  };
}
