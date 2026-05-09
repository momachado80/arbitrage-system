/**
 * Estado de runtime partilhado no processo Node (globalThis).
 * Em `next dev`, HMR pode recarregar módulos; variáveis de módulo repostas.
 * Um único store por processo evita múltiplos loops e refreshes paralelos.
 */

import type { NormalizedMarket } from "./polymarketClient";
import type { RankedGraphOpportunity, GraphSummary } from "./graphOpportunityEngine";
import type { GraphOpportunity } from "./graphArbitrageEngine";
import type { PaperTrade } from "./paperTypes";
import type { GraphPipelineDiagnosticsSnapshot } from "./graphPipelineDiagnostics";
import type { StructuralMicroLaneScanSnapshot } from "./graphStructuralMicroLane";

export const ARBITRAGE_DASHBOARD_RUNTIME_ROOT_KEY = "__arbitrageDashboard_processRuntime_v1";

const ROOT_KEY = ARBITRAGE_DASHBOARD_RUNTIME_ROOT_KEY;

type Root = {
  marketData: MarketDataRuntimeState;
  graphScan: GraphScanRuntimeState;
  paperSim: PaperSimRuntimeState;
  shadowLoop: { started: boolean };
  executionWorker: { started: boolean };
};

export type MarketDataRefreshMetrics = {
  refreshRequested: number;
  refreshStartedEffective: number;
  refreshDeduped: number;
  refreshCompleted: number;
  refreshFailed: number;
  lastDurationMs: number | null;
};

export type MarketDataRuntimeState = {
  markets: NormalizedMarket[];
  lastRefresh: number;
  /** Uma única operação de refresh em voo; chamadas concurrentes reutilizam esta promise. */
  refreshInFlight: Promise<void> | null;
  refreshing: boolean;
  loopStarted: boolean;
  fetchCount: number;
  lastError: string | null;
  bootstrapAttempted: boolean;
  bootstrapCompleted: boolean;
  bootstrapFailed: boolean;
  bootstrapErrorMessage: string | null;
  lastBootstrapAt: string | null;
  bootstrapPhase: "idle" | "refresh_started" | "fetch_completed" | "fetch_failed";
  refreshAttemptedCount: number;
  refreshSuccessCount: number;
  refreshFailureCount: number;
  lastRefreshError: string | null;
  refreshStartedAt: number | null;
  lastRefreshCompletedAt: string | null;
  lastRefreshFailedAt: string | null;
  lastCompletedStep: string | null;
  lastFailedStep: string | null;
  lastStepStartedAt: string | null;
  lastStepCompletedAt: string | null;
  lastRefreshErrorMessage: string | null;
  refreshAttemptDurationMs: number | null;
  bootstrapAttemptDurationMs: number | null;
  timeoutOperationalOccurred: boolean;
  refreshMetrics: MarketDataRefreshMetrics;
};

const defaultGraphSummary = (): GraphSummary => ({
  graphOpportunitiesDetected: 0,
  averageConfidence: 0,
  averageEdge: 0,
  numberOfClustersScanned: 0,
  byType: {},
});

export type GraphScanLastCapture = {
  marketCount: number;
  clusterCount: number;
  rawOpportunityCount: number;
};

export type GraphScanRuntimeState = {
  loopStarted: boolean;
  cachedRanked: RankedGraphOpportunity[];
  /** Pool completo pré-rank (paper expansion usa orçamento sem alterar rank global da UI). */
  cachedGraphRaw: GraphOpportunity[];
  cachedSummary: GraphSummary;
  lastScanMs: number;
  scanning: boolean;
  /** Último scan completo (mesmo com pool vazio); explica `cachedGraphRaw` vazio sem adivinhar. */
  lastGraphScanCapture: GraphScanLastCapture | null;
  /** Funil relation/cluster/source (último ciclo graph scan; não recalculado em GET). */
  lastGraphPipelineDiagnostics: GraphPipelineDiagnosticsSnapshot | null;
  /** Contagens/amostras por micro-lane estrutural (último raw pool). */
  lastStructuralMicroLaneScan: StructuralMicroLaneScanSnapshot | null;
};

export type PaperSimRuntimeState = {
  loopStarted: boolean;
  lastUpdateMs: number;
  lastCycleOk: boolean;
  /** Última falha no `.then` assíncrono de `runCycle` (ingestão); `null` se o último ciclo completou ou ainda não houve tentativa. */
  lastPaperCycleAsyncError: string | null;
  lastPaperCycleAsyncErrorAt: string | null;
  tradesSnapshot: { active: PaperTrade[]; recentClosed: PaperTrade[] };
  /** ISO do último refreshPaperTradesApiSnapshot (cópia para API /paper/trades). */
  lastTradesSnapshotRefreshAt: string | null;
};

function initialMarketData(): MarketDataRuntimeState {
  return {
    markets: [],
    lastRefresh: 0,
    refreshInFlight: null,
    refreshing: false,
    loopStarted: false,
    fetchCount: 0,
    lastError: null,
    bootstrapAttempted: false,
    bootstrapCompleted: false,
    bootstrapFailed: false,
    bootstrapErrorMessage: null,
    lastBootstrapAt: null,
    bootstrapPhase: "idle",
    refreshAttemptedCount: 0,
    refreshSuccessCount: 0,
    refreshFailureCount: 0,
    lastRefreshError: null,
    refreshStartedAt: null,
    lastRefreshCompletedAt: null,
    lastRefreshFailedAt: null,
    lastCompletedStep: null,
    lastFailedStep: null,
    lastStepStartedAt: null,
    lastStepCompletedAt: null,
    lastRefreshErrorMessage: null,
    refreshAttemptDurationMs: null,
    bootstrapAttemptDurationMs: null,
    timeoutOperationalOccurred: false,
    refreshMetrics: {
      refreshRequested: 0,
      refreshStartedEffective: 0,
      refreshDeduped: 0,
      refreshCompleted: 0,
      refreshFailed: 0,
      lastDurationMs: null,
    },
  };
}

function initialGraphScan(): GraphScanRuntimeState {
  return {
    loopStarted: false,
    cachedRanked: [],
    cachedGraphRaw: [],
    cachedSummary: defaultGraphSummary(),
    lastScanMs: 0,
    scanning: false,
    lastGraphScanCapture: null,
    lastGraphPipelineDiagnostics: null,
    lastStructuralMicroLaneScan: null,
  };
}

function initialPaperSim(): PaperSimRuntimeState {
  return {
    loopStarted: false,
    lastUpdateMs: 0,
    lastCycleOk: true,
    lastPaperCycleAsyncError: null,
    lastPaperCycleAsyncErrorAt: null,
    tradesSnapshot: { active: [], recentClosed: [] },
    lastTradesSnapshotRefreshAt: null,
  };
}

function getRoot(): Root {
  const g = globalThis as unknown as Record<string, Root | undefined>;
  if (!g[ROOT_KEY]) {
    g[ROOT_KEY] = {
      marketData: initialMarketData(),
      graphScan: initialGraphScan(),
      paperSim: initialPaperSim(),
      shadowLoop: { started: false },
      executionWorker: { started: false },
    };
    if (process.env.NODE_ENV === "development") {
      console.log(
        "[Runtime] nodeProcessRuntimeState attached to globalThis (single-process guards; HMR-safe loops/refresh)"
      );
    }
  }
  return g[ROOT_KEY]!;
}

export function getMarketDataRuntime(): MarketDataRuntimeState {
  return getRoot().marketData;
}

export function getGraphScanRuntime(): GraphScanRuntimeState {
  const st = getRoot().graphScan;
  if (!st.cachedGraphRaw) st.cachedGraphRaw = [];
  if (st.lastGraphScanCapture === undefined) st.lastGraphScanCapture = null;
  if (st.lastGraphPipelineDiagnostics === undefined) st.lastGraphPipelineDiagnostics = null;
  if (st.lastStructuralMicroLaneScan === undefined) st.lastStructuralMicroLaneScan = null;
  return st;
}

export function getPaperSimRuntime(): PaperSimRuntimeState {
  const st = getRoot().paperSim;
  if (st.lastPaperCycleAsyncError === undefined) st.lastPaperCycleAsyncError = null;
  if (st.lastPaperCycleAsyncErrorAt === undefined) st.lastPaperCycleAsyncErrorAt = null;
  return st;
}

export function getShadowLoopLock(): { started: boolean } {
  return getRoot().shadowLoop;
}

export function getExecutionWorkerLock(): { started: boolean } {
  return getRoot().executionWorker;
}

/** Para inspecção em API ou debug; não necessário para o fluxo normal. */
export function getProcessRuntimeSummary(): {
  marketLoopStarted: boolean;
  graphLoopStarted: boolean;
  paperLoopStarted: boolean;
  shadowLoopStarted: boolean;
  executionWorkerStarted: boolean;
  marketRefreshMetrics: MarketDataRefreshMetrics;
  inFlightRefreshCount: 0 | 1;
} {
  const r = getRoot();
  const md = r.marketData;
  return {
    marketLoopStarted: md.loopStarted,
    graphLoopStarted: r.graphScan.loopStarted,
    paperLoopStarted: r.paperSim.loopStarted,
    shadowLoopStarted: r.shadowLoop.started,
    executionWorkerStarted: r.executionWorker.started,
    marketRefreshMetrics: { ...md.refreshMetrics },
    inFlightRefreshCount: md.refreshInFlight != null ? 1 : 0,
  };
}
