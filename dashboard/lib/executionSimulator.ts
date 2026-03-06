/**
 * Execution Simulator — simulates trade entry and exit for paper trading.
 * No real orders. Fill probability depends on liquidity, spread, confidence, capital.
 */

import type { NormalizedPaperOpportunity } from "./paperTypes";
import type { CapacityResult } from "./capitalCapacityEngine";

export interface SimulatedEntry {
  entryTimestamp: string;
  entryEdge: number;
  entryPriceEstimate: number;
  entrySlippage: number;
  filledCapital: number;
  partialFillFlag: boolean;
  fillProbability: number;
}

export interface SimulatedExit {
  exitTimestamp: string;
  exitPriceEstimate: number;
  exitSlippage: number;
  realizedPnL: number;
  realizedReturn: number;
  maxAdverseExcursion: number;
  maxFavorableExcursion: number;
  exitCondition: "edge_normalization" | "max_holding_time" | "stop_loss" | "take_profit";
}

export interface ActivePaperTradeState {
  tradeId: string;
  opportunityId: string;
  openedAt: string;
  entryEdge: number;
  entryPriceEstimate: number;
  filledCapital: number;
  maxAdverseExcursion: number;
  maxFavorableExcursion: number;
}

const FILL_PROB_BASE = 0.7;
const PARTIAL_FILL_THRESHOLD = 0.5;

function fillProbability(
  liquidity: number,
  spread: number,
  confidence: number,
  requestedCapital: number
): number {
  const liq = Math.max(liquidity, 1);
  const sizeRatio = requestedCapital / liq;
  const liqScore = Math.min(1, Math.log10(liq) / 5);
  const spreadPenalty = Math.max(0.3, 1 - spread * 3);
  const sizePenalty = Math.max(0.2, 1 - sizeRatio * 2);
  return Math.min(0.95, FILL_PROB_BASE * liqScore * spreadPenalty * sizePenalty * confidence);
}

function deterministicFill(prob: number, requested: number): number {
  if (prob < 0.2) return 0;
  return requested * Math.min(1, prob);
}

export function simulateEntry(
  opportunity: NormalizedPaperOpportunity,
  capacity: CapacityResult,
  portfolioAvailableCapital: number
): SimulatedEntry {
  const now = new Date().toISOString();
  const requested = Math.min(
    capacity.recommendedCapital,
    portfolioAvailableCapital,
    opportunity.liquidity * 0.1
  );

  if (requested <= 0) {
    return {
      entryTimestamp: now,
      entryEdge: opportunity.edge,
      entryPriceEstimate: 1 - opportunity.edge,
      entrySlippage: 0,
      filledCapital: 0,
      partialFillFlag: false,
      fillProbability: 0,
    };
  }

  const prob = fillProbability(
    opportunity.liquidity,
    opportunity.spread,
    opportunity.confidence,
    requested
  );

  const filled = deterministicFill(prob, requested);
  const partialFill = filled > 0 && filled < requested * 0.9;

  const slippagePct = (opportunity.spread * (requested / Math.max(1, opportunity.liquidity))) / 2;
  const entryPrice = 1 - opportunity.edge;
  const effectiveSlippage = filled > 0 ? slippagePct * (filled / requested) : 0;

  return {
    entryTimestamp: now,
    entryEdge: opportunity.edge,
    entryPriceEstimate: entryPrice,
    entrySlippage: effectiveSlippage,
    filledCapital: filled,
    partialFillFlag: partialFill,
    fillProbability: prob,
  };
}

export function simulateExit(
  activeTrade: ActivePaperTradeState,
  latestOpportunity: { edge: number; confidence: number } | null,
  options: {
    maxHoldingTimeMs: number;
    stopLossPct: number;
    takeProfitPct: number;
  }
): SimulatedExit {
  const now = Date.now();
  const openedAt = new Date(activeTrade.openedAt).getTime();
  const holdingMs = now - openedAt;

  const entryPrice = activeTrade.entryPriceEstimate;
  const exitPrice = latestOpportunity
    ? 1 - latestOpportunity.edge
    : entryPrice;
  const pnlPct = (exitPrice - entryPrice) / entryPrice;
  const realizedPnL = activeTrade.filledCapital * pnlPct;
  const realizedReturn = pnlPct;

  let exitCondition: SimulatedExit["exitCondition"] = "edge_normalization";

  if (holdingMs >= options.maxHoldingTimeMs) {
    exitCondition = "max_holding_time";
  } else if (pnlPct <= -options.stopLossPct) {
    exitCondition = "stop_loss";
  } else if (pnlPct >= options.takeProfitPct) {
    exitCondition = "take_profit";
  }

  const exitSlippage = latestOpportunity ? 0.001 : 0.002;

  return {
    exitTimestamp: new Date(now).toISOString(),
    exitPriceEstimate: exitPrice,
    exitSlippage: exitSlippage,
    realizedPnL,
    realizedReturn,
    maxAdverseExcursion: activeTrade.maxAdverseExcursion,
    maxFavorableExcursion: activeTrade.maxFavorableExcursion,
    exitCondition,
  };
}

export function shouldClosePaperTrade(
  activeTrade: ActivePaperTradeState,
  latestOpportunity: { edge: number; confidence: number } | null,
  options: {
    maxHoldingTimeMs: number;
    stopLossPct: number;
    takeProfitPct: number;
    edgeNormalizationThreshold: number;
  }
): boolean {
  const now = Date.now();
  const openedAt = new Date(activeTrade.openedAt).getTime();
  const holdingMs = now - openedAt;

  if (holdingMs >= options.maxHoldingTimeMs) return true;

  const entryPrice = activeTrade.entryPriceEstimate;
  const exitPrice = latestOpportunity ? 1 - latestOpportunity.edge : entryPrice;
  const pnlPct = (exitPrice - entryPrice) / Math.max(0.001, entryPrice);

  if (pnlPct <= -options.stopLossPct) return true;
  if (pnlPct >= options.takeProfitPct) return true;

  if (latestOpportunity && Math.abs(latestOpportunity.edge) < options.edgeNormalizationThreshold) {
    return true;
  }

  return false;
}
