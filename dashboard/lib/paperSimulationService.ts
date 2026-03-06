/**
 * Paper Simulation Service — background loop for paper trading simulation.
 * Integrates with existing market/opportunity refresh, non-blocking.
 */

import { getGraphOpportunities } from "./graphScanService";
import { estimateBatchCapacity } from "./capitalCapacityEngine";
import { processOpportunities } from "./paperTradeEngine";
import {
  getPaperPortfolio,
  getPaperPortfolioSummary,
  initPaperPortfolio,
  getActivePaperTrades,
  getClosedPaperTrades,
} from "./paperPortfolioStore";
import { computePaperAnalytics, computeEquityCurve } from "./paperAnalytics";
import type { NormalizedPaperOpportunity } from "./paperTypes";
import { DEFAULT_PAPER_POLICY } from "./paperTradeEngine";

const CYCLE_INTERVAL_MS = 8_000;
const INITIAL_DELAY_MS = 5_000;

let loopStarted = false;
let lastUpdateMs = 0;
let lastCycleOk = true;

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

function normalizeGraph(opp: {
  id: string;
  type: string;
  edge: number;
  confidence: number;
  liquidity: number;
  clusterId?: string;
  marketsInvolved: Array<{ marketId: string; question: string }>;
}): NormalizedPaperOpportunity {
  const spread = estimateSpread(opp);
  return {
    opportunityId: opp.id,
    sourceType: "graph",
    opportunityType: opp.type as NormalizedPaperOpportunity["opportunityType"],
    clusterId: opp.clusterId,
    marketsInvolved: opp.marketsInvolved || [],
    edge: opp.edge,
    confidence: opp.confidence,
    liquidity: opp.liquidity,
    spread,
    compositeScore: (opp as { compositeScore?: number }).compositeScore,
    rank: (opp as { rank?: number }).rank,
  };
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
    const [standard, graph] = [
      fetchStandardOpportunities(),
      Promise.resolve(fetchGraphOpportunities()),
    ];

    Promise.all([standard, graph]).then(([stdOpps, graphOpps]) => {
      const merged = [...graphOpps, ...stdOpps];
      const capacityResults = estimateBatchCapacity(merged);

      const withCapacity = merged
        .map((opp, i) => ({ opp, capacity: capacityResults[i] }))
        .filter((x) => x.capacity.recommendedCapital > 0);

      if (lastUpdateMs === 0) {
        initPaperPortfolio(DEFAULT_PAPER_POLICY.initialCapital);
      }

      const result = processOpportunities(withCapacity, DEFAULT_PAPER_POLICY);

      lastUpdateMs = Date.now();
      lastCycleOk = true;

      const summary = getPaperPortfolioSummary();
      const elapsed = Date.now() - t0;

      if (result.opened > 0 || result.closed > 0 || merged.length > 0) {
        console.log(
          `[PaperSim] cycle ok | opps=${merged.length} opened=${result.opened} closed=${result.closed} equity=${summary.currentEquity.toFixed(0)} ms=${elapsed}`
        );
      }
    }).catch((err) => {
      lastCycleOk = false;
      console.warn("[PaperSim] cycle failed:", err?.message || err);
    });
  } catch (err) {
    lastCycleOk = false;
    console.warn("[PaperSim] cycle error:", err instanceof Error ? err.message : err);
  }
}

export function ensurePaperSimulation(): void {
  if (loopStarted) return;
  loopStarted = true;
  console.log("[PaperSim] Background paper simulation loop started");
  setTimeout(runCycle, INITIAL_DELAY_MS);
  setInterval(runCycle, CYCLE_INTERVAL_MS);
}

export function getPaperSystemStatus(): {
  status: string;
  lastUpdate: string | null;
  startingCapital: number;
  currentEquity: number;
  availableCapital: number;
  reservedCapital: number;
  activeTrades: number;
  closedTrades: number;
  realizedPnL: number;
  unrealizedPnL: number;
} {
  ensurePaperSimulation();
  const s = getPaperPortfolioSummary();
  return {
    status: lastCycleOk ? "ok" : "degraded",
    lastUpdate: lastUpdateMs > 0 ? new Date(lastUpdateMs).toISOString() : null,
    startingCapital: s.startingCapital,
    currentEquity: s.currentEquity,
    availableCapital: s.availableCapital,
    reservedCapital: s.reservedCapital,
    activeTrades: s.activeTrades,
    closedTrades: s.closedTrades,
    realizedPnL: s.realizedPnL,
    unrealizedPnL: s.unrealizedPnL,
  };
}

export function getPaperTradesData(): {
  active: import("./paperTypes").PaperTrade[];
  recentClosed: import("./paperTypes").PaperTrade[];
} {
  ensurePaperSimulation();
  return {
    active: getActivePaperTrades(),
    recentClosed: getClosedPaperTrades(50),
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
} {
  ensurePaperSimulation();
  const p = getPaperPortfolio();
  const analytics = computePaperAnalytics(
    p.closedTrades,
    p.activeTrades,
    p.startingCapital,
    p.currentEquity,
    p.maxDrawdown
  );
  const equityCurve = computeEquityCurve(p.closedTrades, p.startingCapital);
  return { analytics, equityCurve };
}
