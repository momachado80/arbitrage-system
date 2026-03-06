/**
 * Paper Analytics — computes portfolio and strategy metrics from paper trades.
 */

import type { PaperTrade } from "./paperTypes";

export interface PaperAnalyticsResult {
  totalTrades: number;
  closedTrades: number;
  winRate: number;
  avgReturn: number;
  avgPnL: number;
  totalPnL: number;
  profitFactor: number;
  maxDrawdown: number;
  avgHoldingTimeMs: number;
  avgNetEdgeAtEntry: number;
  avgFilledCapital: number;
  utilizationRate: number;
  pnlByOpportunityType: Record<string, number>;
  pnlBySourceType: Record<string, number>;
  averageCapacityConfidence: number;
  sharpeLikeMetric?: number;
}

export interface EquityPoint {
  ts: string;
  equity: number;
  cumulativePnL: number;
}

function safeNum(v: unknown): number {
  if (typeof v === "number" && !Number.isNaN(v)) return v;
  return 0;
}

export function computePaperAnalytics(
  closedTrades: PaperTrade[],
  activeTrades: PaperTrade[],
  startingCapital: number,
  currentEquity: number,
  maxDrawdown: number
): PaperAnalyticsResult {
  const closed = closedTrades.filter((t) => t.status === "closed" && t.realizedPnL !== undefined);
  const total = closed.length + activeTrades.length;

  let winRate = 0;
  let avgReturn = 0;
  let avgPnL = 0;
  let totalPnL = 0;
  let grossProfit = 0;
  let grossLoss = 0;
  let avgHolding = 0;
  let avgNetEdge = 0;
  let avgFilled = 0;
  const pnlByType: Record<string, number> = {};
  const pnlBySource: Record<string, number> = {};
  let capConfSum = 0;
  let capConfCount = 0;

  if (closed.length > 0) {
    const wins = closed.filter((t) => safeNum(t.realizedPnL) > 0).length;
    winRate = wins / closed.length;

    const returns = closed.map((t) => safeNum(t.realizedReturn));
    avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;

    const pnls = closed.map((t) => safeNum(t.realizedPnL));
    totalPnL = pnls.reduce((a, b) => a + b, 0);
    avgPnL = totalPnL / closed.length;

    for (const t of closed) {
      const p = safeNum(t.realizedPnL);
      if (p > 0) grossProfit += p;
      else grossLoss += Math.abs(p);
    }

    const holdings = closed.map((t) => safeNum(t.holdingTimeMs));
    avgHolding = holdings.reduce((a, b) => a + b, 0) / holdings.length;

    const edges = closed.map((t) => safeNum(t.netEdgeAtEntry));
    avgNetEdge = edges.reduce((a, b) => a + b, 0) / edges.length;

    const filled = closed.map((t) => safeNum(t.filledCapital));
    avgFilled = filled.reduce((a, b) => a + b, 0) / filled.length;

    for (const t of closed) {
      const type = t.opportunityType || "unknown";
      pnlByType[type] = (pnlByType[type] || 0) + safeNum(t.realizedPnL);
      const src = t.sourceType || "standard";
      pnlBySource[src] = (pnlBySource[src] || 0) + safeNum(t.realizedPnL);
    }
  }

  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 10 : 0;

  const reserved = activeTrades.reduce((s, t) => s + safeNum(t.filledCapital), 0);
  const utilizationRate = startingCapital > 0 ? reserved / startingCapital : 0;

  let sharpeLike: number | undefined;
  if (closed.length >= 5) {
    const returns = closed.map((t) => safeNum(t.realizedReturn));
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / returns.length;
    const std = Math.sqrt(variance) || 0.0001;
    sharpeLike = std > 0 ? (mean / std) * Math.sqrt(252) : 0;
  }

  return {
    totalTrades: total,
    closedTrades: closed.length,
    winRate,
    avgReturn,
    avgPnL,
    totalPnL,
    profitFactor,
    maxDrawdown,
    avgHoldingTimeMs: Math.round(avgHolding),
    avgNetEdgeAtEntry: avgNetEdge,
    avgFilledCapital: avgFilled,
    utilizationRate,
    pnlByOpportunityType: pnlByType,
    pnlBySourceType: pnlBySource,
    averageCapacityConfidence: capConfCount > 0 ? capConfSum / capConfCount : 0,
    sharpeLikeMetric: sharpeLike,
  };
}

export function computeEquityCurve(
  closedTrades: PaperTrade[],
  startingCapital: number
): EquityPoint[] {
  const sorted = [...closedTrades]
    .filter((t) => t.closedAt && t.realizedPnL !== undefined)
    .sort((a, b) => new Date(a.closedAt!).getTime() - new Date(b.closedAt!).getTime());

  const curve: EquityPoint[] = [{ ts: new Date(0).toISOString(), equity: startingCapital, cumulativePnL: 0 }];
  let cum = 0;

  for (const t of sorted) {
    cum += safeNum(t.realizedPnL);
    curve.push({
      ts: t.closedAt!,
      equity: startingCapital + cum,
      cumulativePnL: cum,
    });
  }

  return curve;
}
