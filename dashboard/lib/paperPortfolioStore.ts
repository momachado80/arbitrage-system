/**
 * Paper Portfolio Store — maintains current paper portfolio state.
 * Bounded in memory, safe against malformed trades.
 */

import type { PaperTrade } from "./paperTypes";

const DEFAULT_STARTING_CAPITAL = 10_000;
const MAX_ACTIVE_TRADES = 100;
const MAX_CLOSED_TRADES = 500;

let startingCapital = DEFAULT_STARTING_CAPITAL;
let activeTrades: PaperTrade[] = [];
let closedTrades: PaperTrade[] = [];
let peakEquity = startingCapital;

function safeNumber(v: unknown, def: number): number {
  if (typeof v === "number" && !Number.isNaN(v)) return v;
  return def;
}

export function initPaperPortfolio(initialCapital?: number): void {
  startingCapital = Math.max(100, safeNumber(initialCapital, DEFAULT_STARTING_CAPITAL));
  peakEquity = startingCapital;
}

export function getPaperPortfolio() {
  const reserved = activeTrades.reduce((s, t) => s + (t.filledCapital || 0), 0);
  const realized = closedTrades.reduce((s, t) => s + (t.realizedPnL || 0), 0);
  const unrealized = activeTrades.reduce((s, t) => {
    const entry = t.entryPriceEstimate || 0;
    const exitEst = 1 - (t.grossEdgeAtEntry || 0);
    const pnl = (t.filledCapital || 0) * ((exitEst - entry) / Math.max(0.001, entry));
    return s + pnl;
  }, 0);
  const currentEquity = startingCapital + realized + unrealized;
  const available = Math.max(0, currentEquity - reserved);
  const maxDrawdown = peakEquity > 0 ? (peakEquity - Math.min(peakEquity, currentEquity)) / peakEquity : 0;
  if (currentEquity > peakEquity) peakEquity = currentEquity;

  const exposureByType: Record<string, number> = {};
  const exposureByCluster: Record<string, number> = {};
  const exposureByMarket: Record<string, number> = {};

  for (const t of activeTrades) {
    const cap = t.filledCapital || 0;
    exposureByType[t.opportunityType] = (exposureByType[t.opportunityType] || 0) + cap;
    if (t.clusterId) exposureByCluster[t.clusterId] = (exposureByCluster[t.clusterId] || 0) + cap;
    for (const m of t.marketsInvolved || []) {
      exposureByMarket[m.marketId] = (exposureByMarket[m.marketId] || 0) + cap;
    }
  }

  return {
    startingCapital,
    currentEquity,
    availableCapital: available,
    reservedCapital: reserved,
    realizedPnL: realized,
    unrealizedPnL: unrealized,
    maxDrawdown,
    activeTrades,
    closedTrades,
    exposureByType,
    exposureByCluster,
    exposureByMarket,
  };
}

export function getActivePaperTrades(): PaperTrade[] {
  return [...activeTrades];
}

export function getClosedPaperTrades(limit = 50): PaperTrade[] {
  const start = Math.max(0, closedTrades.length - limit);
  return closedTrades.slice(start);
}

export interface PaperPortfolioSummary {
  startingCapital: number;
  currentEquity: number;
  availableCapital: number;
  reservedCapital: number;
  realizedPnL: number;
  unrealizedPnL: number;
  maxDrawdown: number;
  activeTrades: number;
  closedTrades: number;
  exposureByType: Record<string, number>;
  exposureByCluster: Record<string, number>;
  exposureByMarket: Record<string, number>;
}

export function getPaperPortfolioSummary(): PaperPortfolioSummary {
  const p = getPaperPortfolio();
  return {
    startingCapital: p.startingCapital,
    currentEquity: p.currentEquity,
    availableCapital: p.availableCapital,
    reservedCapital: p.reservedCapital,
    realizedPnL: p.realizedPnL,
    unrealizedPnL: p.unrealizedPnL,
    maxDrawdown: p.maxDrawdown,
    activeTrades: p.activeTrades.length,
    closedTrades: p.closedTrades.length,
    exposureByType: p.exposureByType,
    exposureByCluster: p.exposureByCluster,
    exposureByMarket: p.exposureByMarket,
  };
}

export function addActiveTrade(trade: PaperTrade): void {
  try {
    if (activeTrades.length >= MAX_ACTIVE_TRADES) {
      console.warn("[PaperPortfolio] Max active trades reached");
      return;
    }
    if (!trade?.tradeId || !trade?.opportunityId) return;
    activeTrades.push({
      ...trade,
      filledCapital: safeNumber(trade.filledCapital, 0),
      realizedPnL: 0,
      realizedReturn: 0,
      holdingTimeMs: 0,
      maxAdverseExcursion: safeNumber(trade.maxAdverseExcursion, 0),
      maxFavorableExcursion: safeNumber(trade.maxFavorableExcursion, 0),
    });
  } catch {
    // non-fatal
  }
}

export function closeTrade(tradeId: string, updates: Partial<PaperTrade>): void {
  const idx = activeTrades.findIndex((t) => t.tradeId === tradeId);
  if (idx < 0) return;
  const t = activeTrades[idx];
  activeTrades.splice(idx, 1);
  const closed: PaperTrade = {
    ...t,
    ...updates,
    status: "closed",
    closedAt: updates.closedAt || new Date().toISOString(),
  };
  closedTrades.push(closed);
  if (closedTrades.length > MAX_CLOSED_TRADES) {
    closedTrades = closedTrades.slice(-MAX_CLOSED_TRADES);
  }
}

export function getAvailableCapital(): number {
  const p = getPaperPortfolio();
  return p.availableCapital;
}
