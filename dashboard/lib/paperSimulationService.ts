/**
 * Paper Simulation Service — background loop for paper trading simulation.
 * Integrates with existing market/opportunity refresh, non-blocking.
 */

import type { GraphOpportunity } from "./graphArbitrageEngine";
import type { RankedGraphOpportunity } from "./graphOpportunityEngine";
import { getGraphOpportunities } from "./graphScanService";
import { getGraphScanRuntime } from "./nodeProcessRuntimeState";
import { applyUpstreamScannerExpansion } from "./paperUpstreamScannerExpansion";
import { getAllMarkets } from "./marketDataService";
import { fetchNormalizedMarketById, type NormalizedMarket } from "./polymarketClient";
import { scanMarkets } from "./probabilityScanner";
import { estimateBatchCapacity } from "./capitalCapacityEngine";
import { processOpportunities, resolvePaperPolicyFromEnv, getPaperEntryPolicySnapshot } from "./paperTradeEngine";
import {
  getPaperPortfolio,
  getPaperPortfolioSummary,
  initPaperPortfolio,
  getActivePaperTrades,
  getClosedPaperTrades,
  getPaperPortfolioStateIntegrity,
  type PaperPortfolioStateIntegrity,
} from "./paperPortfolioStore";
import { computePaperAnalytics, computeEquityCurve } from "./paperAnalytics";
import type { NormalizedPaperOpportunity, PaperTrade } from "./paperTypes";
import {
  getPaperMarketWhitelist,
  filterPaperOpportunitiesByWhitelist,
  gammaIdsFromMarketIdField,
} from "./paperMarketWhitelist";
import {
  recordPaperCycleOpportunityMetrics,
  getPaperOpportunityMetricsToday,
  getPaperOpportunityMetricsByDay,
} from "./paperDailyMetricsStore";
import {
  recordPaperPreFilterBatch,
  getPaperOpenDiagnostics,
  getEconomicDedupeCountsFromRecentBuffer,
  recordPaperExplorationLastCycle,
} from "./paperOpenDiagnostics";
import {
  resetGraphOpportunityPaperImpactCycle,
  finalizeGraphOpportunityPaperImpactCycle,
  recordGraphMergedExpandedIfApplicable,
  recordGraphCapacityPositiveIfApplicable,
  resolveGraphDiagnosticProvenanceForRawGraphOpportunity,
} from "./graphOpportunityPaperImpact";
import { sortOpportunitiesForExploration } from "./paperExploration";
import {
  recordPaperUpstreamCycleComplete,
  recordPaperUpstreamRunCycleStart,
  getPaperUpstreamDiagnostics,
} from "./paperUpstreamDiagnostics";
import { getGammaFetchByIdDiagnostics } from "./gammaFetchByIdDiagnostics";
import {
  recordPaperWhitelistHealthAfterScan,
  getPaperWhitelistHealth,
  clearPaperWhitelistHealth,
} from "./paperWhitelistHealth";
import {
  buildPaperOperationalWhitelist,
  getPaperAdaptiveWhitelistDiagnostics,
  clearPaperAdaptiveWhitelistDiagnostics,
} from "./paperAdaptiveWhitelist";
import { getPaperTradeLifecycleDiagnostics } from "./paperTradeLifecycleDiagnostics";
import { getPaperSimulateEntryDiagnostics } from "./paperSimulateEntryDiagnostics";
import { getPaperSimRuntime, ARBITRAGE_DASHBOARD_RUNTIME_ROOT_KEY } from "./nodeProcessRuntimeState";
import {
  getClosedTradesWithFiniteRealizedPnl,
  getPaperApiRecentClosedLimit,
} from "./paperClosedTradesMetrics";
import { safeFeeBufferPerLeg } from "./paperRealizedPnlSemantics";

const CYCLE_INTERVAL_MS = 8_000;
const INITIAL_DELAY_MS = 5_000;

/** Cópia rasa para o snapshot da API (evita partilhar arrays `marketsInvolved` com o store). */
function shallowSnapshotTradeForApi(t: PaperTrade): PaperTrade {
  return {
    ...t,
    marketsInvolved: (t.marketsInvolved || []).map((m) => ({ ...m })),
  };
}

/**
 * Snapshot em `getPaperSimRuntime().tradesSnapshot` — actualizado no fim de cada ciclo paper.
 * GET /api/paper/trades lê sempre do store canónico (evita drift vs analytics).
 */
function refreshPaperTradesApiSnapshot(): void {
  const lim = getPaperApiRecentClosedLimit();
  const rt = getPaperSimRuntime();
  rt.tradesSnapshot = {
    active: getActivePaperTrades().map(shallowSnapshotTradeForApi),
    recentClosed: getClosedPaperTrades(lim).map(shallowSnapshotTradeForApi),
  };
  rt.lastTradesSnapshotRefreshAt = new Date().toISOString();
}

export type PaperStateIntegrity = PaperPortfolioStateIntegrity & {
  lastSnapshotRefreshAt: string | null;
  snapshotActiveCount: number;
  snapshotRecentClosedCount: number;
  /** true se cópia da API diverge do store canónico (sinal de módulos duplicados antes do fix). */
  snapshotVsStoreActiveMismatch: boolean;
  processRuntimeRootKey: string;
};

function buildPaperStateIntegrity(): PaperStateIntegrity {
  const base = getPaperPortfolioStateIntegrity();
  const rt = getPaperSimRuntime();
  const snapA = rt.tradesSnapshot.active.length;
  const storeA = base.activeTradesStoreCount;
  return {
    ...base,
    lastSnapshotRefreshAt: rt.lastTradesSnapshotRefreshAt,
    snapshotActiveCount: snapA,
    snapshotRecentClosedCount: rt.tradesSnapshot.recentClosed.length,
    snapshotVsStoreActiveMismatch: snapA !== storeA,
    processRuntimeRootKey: ARBITRAGE_DASHBOARD_RUNTIME_ROOT_KEY,
  };
}

function estimateSpread(opp: { prices?: number[]; liquidity?: number; confidence?: number }): number {
  if (opp.prices && Array.isArray(opp.prices) && opp.prices.length >= 2) {
    const sorted = [...opp.prices].sort((a, b) => b - a);
    return Math.max(0.01, sorted[0] - sorted[sorted.length - 1]);
  }
  return Math.max(0.01, 0.02 * (1 - (opp.confidence || 0.5)));
}

function normalizeStandard(opp: {
  marketId: string;
  question: string;
  edge: number;
  type: string;
  confidence: number;
  liquidity: number;
  outcomes?: string[];
  prices?: number[];
}): NormalizedPaperOpportunity {
  const spread = estimateSpread(opp);
  return {
    opportunityId: opp.marketId,
    sourceType: "standard",
    opportunityType: opp.type as NormalizedPaperOpportunity["opportunityType"],
    marketsInvolved: [{ marketId: opp.marketId, question: opp.question }],
    edge: opp.edge,
    confidence: opp.confidence,
    liquidity: opp.liquidity,
    spread,
    compositeScore: (opp as { compositeScore?: number }).compositeScore,
    rank: (opp as { rank?: number }).rank,
  };
}

function normalizeGraph(opp: GraphOpportunity | RankedGraphOpportunity): NormalizedPaperOpportunity {
  const spread = estimateSpread(opp);
  const ranked = opp as RankedGraphOpportunity;
  const out: NormalizedPaperOpportunity = {
    opportunityId: opp.id,
    sourceType: "graph",
    opportunityType: opp.type as NormalizedPaperOpportunity["opportunityType"],
    clusterId: opp.clusterId,
    marketsInvolved: opp.marketsInvolved || [],
    edge: opp.edge,
    confidence: opp.confidence,
    liquidity: opp.liquidity,
    spread,
    compositeScore: ranked.compositeScore,
    rank: ranked.rank,
    graphDiagnosticProvenance: resolveGraphDiagnosticProvenanceForRawGraphOpportunity(opp),
    structuralMicroLaneReason: opp.structuralMicroLaneReason,
  };
  return out;
}

async function fetchStandardOpportunities(): Promise<NormalizedPaperOpportunity[]> {
  try {
    const base = typeof window !== "undefined" ? "" : "http://localhost:3000";
    const res = await fetch(`${base}/api/opportunities`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return [];
    const data = await res.json();
    const opps = data.opportunities || [];
    return opps.map((o: Record<string, unknown>) =>
      normalizeStandard({
        marketId: String(o.marketId ?? o.id ?? ""),
        question: String(o.question ?? ""),
        edge: Number(o.edge ?? 0),
        type: String(o.type ?? "overround"),
        confidence: Number(o.confidence ?? 0),
        liquidity: Number(o.liquidity ?? 0),
        outcomes: (o.outcomes as string[]) || [],
        prices: (o.prices as number[]) || [],
      })
    );
  } catch {
    return [];
  }
}

/**
 * Scan in-process completo nos mercados da whitelist (sem top-50 do HTTP).
 * Garante ids em falta no cache paginado via GET /markets/{id}.
 * Mantém cross_market quando todos os gamma ids (incl. idA+idB) estão na whitelist.
 */
async function standardOpportunitiesFromWhitelistedScan(whitelist: Set<string>): Promise<{
  opps: NormalizedPaperOpportunity[];
  standard: import("./paperUpstreamDiagnostics").UpstreamStandardMetrics;
  operationalWhitelist: Set<string>;
  marketsForExpansion: NormalizedMarket[];
}> {
  const fail = (
    operational: Set<string>,
    partial: {
      allMarketsCacheCount?: number;
      whitelistHitsFromAllMarkets?: number;
      fetchNormalizedAttempts?: number;
      fetchNormalizedSucceeded?: number;
      subsetMarketsPassedToScan?: number;
      scanEdgesRaw?: number;
      scanEdgesAfterWhitelistFilter?: number;
    }
  ): {
    opps: NormalizedPaperOpportunity[];
    standard: import("./paperUpstreamDiagnostics").UpstreamStandardMetrics;
    operationalWhitelist: Set<string>;
    marketsForExpansion: NormalizedMarket[];
  } => ({
    opps: [],
    standard: {
      mode: "whitelist_scan",
      whitelistMarketIdsCount: whitelist.size,
      operationalWhitelistMarketIdsCount: operational.size,
      allMarketsCacheCount: partial.allMarketsCacheCount ?? 0,
      whitelistHitsFromAllMarkets: partial.whitelistHitsFromAllMarkets ?? 0,
      fetchNormalizedAttempts: partial.fetchNormalizedAttempts ?? 0,
      fetchNormalizedSucceeded: partial.fetchNormalizedSucceeded ?? 0,
      subsetMarketsPassedToScan: partial.subsetMarketsPassedToScan ?? 0,
      scanEdgesRaw: partial.scanEdgesRaw ?? 0,
      scanEdgesAfterWhitelistFilter: partial.scanEdgesAfterWhitelistFilter ?? 0,
    },
    operationalWhitelist: operational,
    marketsForExpansion: [],
  });
  try {
    const allMarkets = getAllMarkets();
    const allMarketsCacheCount = allMarkets.length;
    const byId = new Map(allMarkets.map((m) => [m.id, m]));
    let whitelistHitsFromAllMarkets = 0;
    for (const id of Array.from(whitelist)) {
      if (byId.has(id)) whitelistHitsFromAllMarkets += 1;
    }
    let fetchNormalizedAttempts = 0;
    let fetchNormalizedSucceeded = 0;
    for (const id of Array.from(whitelist)) {
      if (!byId.has(id)) {
        fetchNormalizedAttempts += 1;
        const fetched = await fetchNormalizedMarketById(id);
        if (fetched) {
          fetchNormalizedSucceeded += 1;
          byId.set(id, fetched);
        }
      }
    }
    recordPaperWhitelistHealthAfterScan(whitelist, byId);
    const { operational } = buildPaperOperationalWhitelist(whitelist, byId, allMarkets);
    const markets = Array.from(operational)
      .map((id) => byId.get(id))
      .filter((m): m is NormalizedMarket => m != null);
    const subsetMarketsPassedToScan = markets.length;
    if (markets.length === 0) {
      return fail(operational, {
        allMarketsCacheCount,
        whitelistHitsFromAllMarkets,
        fetchNormalizedAttempts,
        fetchNormalizedSucceeded,
        subsetMarketsPassedToScan: 0,
      });
    }
    const edges = scanMarkets(markets);
    const scanEdgesRaw = edges.length;
    const filtered = edges.filter((e) => {
      const parts = gammaIdsFromMarketIdField(e.marketId);
      return parts.length > 0 && parts.every((wid) => operational.has(wid));
    });
    const scanEdgesAfterWhitelistFilter = filtered.length;
    const opps = filtered.map((e) =>
      normalizeStandard({
        marketId: e.marketId,
        question: e.question,
        edge: e.edge,
        type: e.type,
        confidence: e.confidence,
        liquidity: e.liquidity,
        outcomes: e.outcomes,
        prices: e.prices,
      })
    );
    return {
      opps,
      standard: {
        mode: "whitelist_scan",
        whitelistMarketIdsCount: whitelist.size,
        operationalWhitelistMarketIdsCount: operational.size,
        allMarketsCacheCount,
        whitelistHitsFromAllMarkets,
        fetchNormalizedAttempts,
        fetchNormalizedSucceeded,
        subsetMarketsPassedToScan,
        scanEdgesRaw,
        scanEdgesAfterWhitelistFilter,
      },
      operationalWhitelist: operational,
      marketsForExpansion: markets,
    };
  } catch {
    return fail(new Set(), {});
  }
}

function fetchGraphOpportunities(): NormalizedPaperOpportunity[] {
  try {
    const ranked = getGraphOpportunities();
    return ranked.map((o) => normalizeGraph(o));
  } catch {
    return [];
  }
}

function runCycle(): void {
  const t0 = Date.now();
  try {
    const whitelist = getPaperMarketWhitelist();
    recordPaperUpstreamRunCycleStart(whitelist);
    if (!whitelist) {
      clearPaperWhitelistHealth();
      clearPaperAdaptiveWhitelistDiagnostics();
    }
    const standardPromise = whitelist
      ? standardOpportunitiesFromWhitelistedScan(whitelist)
      : fetchStandardOpportunities().then((opps) => ({
          opps,
          standard: { mode: "http_opportunities" as const, httpOpportunitiesReturned: opps.length },
          operationalWhitelist: undefined as Set<string> | undefined,
          marketsForExpansion: getAllMarkets(),
        }));

    Promise.all([standardPromise, Promise.resolve(fetchGraphOpportunities())]).then(([stdResult, graphOpps]) => {
      resetGraphOpportunityPaperImpactCycle();
      const stdOpps = stdResult.opps;
      const mergedRaw = [...graphOpps, ...stdOpps];
      const paperFilterWhitelist = stdResult.operationalWhitelist ?? whitelist;
      const merged = filterPaperOpportunitiesByWhitelist(mergedRaw, paperFilterWhitelist);
      const graphAfterWhitelist = filterPaperOpportunitiesByWhitelist(graphOpps, paperFilterWhitelist).length;

      const marketsForCrossExpansion =
        stdResult.marketsForExpansion.length > 0
          ? stdResult.marketsForExpansion
          : paperFilterWhitelist
            ? getAllMarkets().filter((m) => paperFilterWhitelist.has(m.id))
            : getAllMarkets();

      const graphRt = getGraphScanRuntime();
      const { merged: mergedExpanded } = applyUpstreamScannerExpansion({
        merged,
        graphRaw: graphRt.cachedGraphRaw,
        marketsForCrossExpansion,
        normalizeGraph: (o) => normalizeGraph(o),
        normalizeStandardCross: (e) => normalizeStandard(e),
        sourceContext: {
          scannerMode: whitelist ? "whitelist_scan" : "http_opportunities",
          graphScanLastScanMs: graphRt.lastScanMs > 0 ? graphRt.lastScanMs : null,
          graphRankedOpportunitiesCount: graphOpps.length,
          graphScanCapture: graphRt.lastGraphScanCapture,
        },
      });

      const capacityResults = estimateBatchCapacity(mergedExpanded);

      for (const o of mergedExpanded) {
        recordGraphMergedExpandedIfApplicable(o);
      }

      const withCapacity = mergedExpanded
        .map((opp, i) => ({ opp, capacity: capacityResults[i] }))
        .filter((x) => x.capacity.recommendedCapital > 0);

      for (const x of withCapacity) {
        recordGraphCapacityPositiveIfApplicable(x.opp);
      }

      recordPaperUpstreamCycleComplete({
        standard: stdResult.standard,
        standardOppsReturned: stdOpps.length,
        graphOpportunitiesRaw: graphOpps.length,
        graphAfterWhitelist,
        mergedRawCount: mergedRaw.length,
        mergedAfterWhitelistCount: mergedExpanded.length,
        enteringRecommendedCapitalPositiveCount: withCapacity.length,
      });

      recordPaperPreFilterBatch(mergedExpanded.length, withCapacity.length);
      recordPaperCycleOpportunityMetrics(mergedExpanded.length, withCapacity.length);

      const policy = resolvePaperPolicyFromEnv();

      const rt = getPaperSimRuntime();
      if (rt.lastUpdateMs === 0) {
        initPaperPortfolio(policy.initialCapital);
      }

      const dedupeCounts = getEconomicDedupeCountsFromRecentBuffer();
      const { sorted: sortedForExploration, lastCycle: explorationLastCycle } = sortOpportunitiesForExploration(
        withCapacity,
        dedupeCounts
      );
      recordPaperExplorationLastCycle(explorationLastCycle);

      const result = processOpportunities(sortedForExploration, policy);

      finalizeGraphOpportunityPaperImpactCycle();

      refreshPaperTradesApiSnapshot();

      rt.lastUpdateMs = Date.now();
      rt.lastCycleOk = true;
      rt.lastPaperCycleAsyncError = null;
      rt.lastPaperCycleAsyncErrorAt = null;

      const summary = getPaperPortfolioSummary();
      const elapsed = Date.now() - t0;

      if (result.opened > 0 || result.closed > 0 || mergedExpanded.length > 0) {
        console.log(
          `[PaperSim] cycle ok | opps=${mergedExpanded.length} opened=${result.opened} closed=${result.closed} equity=${summary.currentEquity.toFixed(0)} ms=${elapsed}`
        );
      }
    }).catch((err) => {
      const rtFail = getPaperSimRuntime();
      rtFail.lastCycleOk = false;
      const msg = err instanceof Error ? err.message : String(err);
      rtFail.lastPaperCycleAsyncError = msg.slice(0, 500);
      rtFail.lastPaperCycleAsyncErrorAt = new Date().toISOString();
      console.warn("[PaperSim] cycle failed:", err?.message || err);
    });
  } catch (err) {
    const rtFail = getPaperSimRuntime();
    rtFail.lastCycleOk = false;
    rtFail.lastPaperCycleAsyncError =
      err instanceof Error ? err.message.slice(0, 500) : String(err).slice(0, 500);
    rtFail.lastPaperCycleAsyncErrorAt = new Date().toISOString();
    console.warn("[PaperSim] cycle error:", err instanceof Error ? err.message : err);
  }
}

export function ensurePaperSimulation(): void {
  const rt = getPaperSimRuntime();
  if (rt.loopStarted) {
    console.log("[PaperSim] loop_skip_already_active_globalThis");
    return;
  }
  rt.loopStarted = true;
  console.log("[PaperSim] Background paper simulation loop started (effective; single process)");
  setTimeout(runCycle, INITIAL_DELAY_MS);
  setInterval(runCycle, CYCLE_INTERVAL_MS);

  try {
    const { ensureLowLiquidityProbe } = require("./lowLiquidityEdgeProbe") as typeof import("./lowLiquidityEdgeProbe");
    ensureLowLiquidityProbe();
  } catch (e) {
    console.warn("[PaperSim] lowLiquidityEdgeProbe start failed (non-fatal):", e instanceof Error ? e.message : e);
  }
}

export function getPaperSystemStatus(): {
  status: string;
  lastUpdate: string | null;
  lastPaperCycleAsyncError: string | null;
  lastPaperCycleAsyncErrorAt: string | null;
  startingCapital: number;
  currentEquity: number;
  availableCapital: number;
  reservedCapital: number;
  activeTrades: number;
  closedTrades: number;
  realizedPnL: number;
  unrealizedPnL: number;
  openEntryDiagnostics: ReturnType<typeof getPaperOpenDiagnostics>;
  openUpstreamDiagnostics: ReturnType<typeof getPaperUpstreamDiagnostics>;
  gammaFetchByIdDiagnostics: ReturnType<typeof getGammaFetchByIdDiagnostics>;
  paperWhitelistHealth: ReturnType<typeof getPaperWhitelistHealth>;
  paperAdaptiveWhitelist: ReturnType<typeof getPaperAdaptiveWhitelistDiagnostics>;
  paperTradeLifecycleDiagnostics: ReturnType<typeof getPaperTradeLifecycleDiagnostics>;
  simulateEntryDiagnostics: ReturnType<typeof getPaperSimulateEntryDiagnostics>;
  paperEntryPolicy: ReturnType<typeof getPaperEntryPolicySnapshot>;
  paperStateIntegrity: PaperStateIntegrity;
} {
  ensurePaperSimulation();
  const s = getPaperPortfolioSummary();
  const rt = getPaperSimRuntime();
  return {
    status: rt.lastCycleOk ? "ok" : "degraded",
    lastUpdate: rt.lastUpdateMs > 0 ? new Date(rt.lastUpdateMs).toISOString() : null,
    lastPaperCycleAsyncError: rt.lastPaperCycleAsyncError,
    lastPaperCycleAsyncErrorAt: rt.lastPaperCycleAsyncErrorAt,
    startingCapital: s.startingCapital,
    currentEquity: s.currentEquity,
    availableCapital: s.availableCapital,
    reservedCapital: s.reservedCapital,
    activeTrades: s.activeTrades,
    closedTrades: s.closedTrades,
    realizedPnL: s.realizedPnL,
    unrealizedPnL: s.unrealizedPnL,
    openEntryDiagnostics: getPaperOpenDiagnostics(),
    openUpstreamDiagnostics: getPaperUpstreamDiagnostics(),
    gammaFetchByIdDiagnostics: getGammaFetchByIdDiagnostics(),
    paperWhitelistHealth: getPaperWhitelistHealth(),
    paperAdaptiveWhitelist: getPaperAdaptiveWhitelistDiagnostics(),
    paperTradeLifecycleDiagnostics: getPaperTradeLifecycleDiagnostics(),
    simulateEntryDiagnostics: getPaperSimulateEntryDiagnostics(),
    paperEntryPolicy: getPaperEntryPolicySnapshot(),
    paperStateIntegrity: buildPaperStateIntegrity(),
  };
}

export function getPaperTradesData(): {
  active: import("./paperTypes").PaperTrade[];
  recentClosed: import("./paperTypes").PaperTrade[];
} {
  ensurePaperSimulation();
  const lim = getPaperApiRecentClosedLimit();
  return {
    active: getActivePaperTrades().map(shallowSnapshotTradeForApi),
    recentClosed: getClosedPaperTrades(lim).map(shallowSnapshotTradeForApi),
  };
}

export function getPaperPortfolioData(): {
  summary: import("./paperPortfolioStore").PaperPortfolioSummary;
  exposureByType: Record<string, number>;
  exposureByCluster: Record<string, number>;
  exposureByMarket: Record<string, number>;
} {
  ensurePaperSimulation();
  const p = getPaperPortfolio();
  return {
    summary: getPaperPortfolioSummary(),
    exposureByType: p.exposureByType,
    exposureByCluster: p.exposureByCluster,
    exposureByMarket: p.exposureByMarket,
  };
}

export function getPaperAnalyticsData(): {
  analytics: import("./paperAnalytics").PaperAnalyticsResult;
  equityCurve: import("./paperAnalytics").EquityPoint[];
  dailyOpportunityMetrics: {
    todayUtc: string;
    today: import("./paperDailyMetricsStore").PaperDayOpportunityMetrics | null;
    byDay: Record<string, import("./paperDailyMetricsStore").PaperDayOpportunityMetrics>;
    opportunitiesSeenToday: number;
    opportunitiesExecutableToday: number;
  };
} {
  ensurePaperSimulation();
  const p = getPaperPortfolio();
  const policy = resolvePaperPolicyFromEnv();
  const feeBuf = safeFeeBufferPerLeg(policy.feeBuffer);
  const todayMetrics = getPaperOpportunityMetricsToday();
  const opportunitiesSeenToday = todayMetrics?.opportunitiesSeen ?? 0;
  const opportunitiesExecutableToday = todayMetrics?.opportunitiesExecutable ?? 0;
  const closedForMetrics = getClosedTradesWithFiniteRealizedPnl();
  const analytics = computePaperAnalytics(
    closedForMetrics,
    p.activeTrades,
    p.startingCapital,
    p.currentEquity,
    p.maxDrawdown,
    feeBuf,
    { seen: opportunitiesSeenToday, executable: opportunitiesExecutableToday }
  );
  const equityCurve = computeEquityCurve(closedForMetrics, p.startingCapital, feeBuf);
  const todayUtc = new Date().toISOString().slice(0, 10);
  return {
    analytics,
    equityCurve,
    dailyOpportunityMetrics: {
      todayUtc,
      today: todayMetrics,
      byDay: getPaperOpportunityMetricsByDay(),
      opportunitiesSeenToday,
      opportunitiesExecutableToday,
    },
  };
}
