/**
 * Paper Trade Engine — manages lifecycle of simulated trades.
 * Opens, updates, closes paper trades based on opportunities and exit conditions.
 */

import type { NormalizedPaperOpportunity } from "./paperTypes";
import type { CapacityResult } from "./capitalCapacityEngine";
import {
  simulateEntry,
  simulateExit,
  shouldClosePaperTrade,
  type SimulatedEntry,
  type SimulatedExit,
  type ActivePaperTradeState,
} from "./executionSimulator";
import {
  getPaperPortfolio,
  addActiveTrade,
  closeTrade,
  getActivePaperTrades,
} from "./paperPortfolioStore";
import { estimateOpportunityCapacity } from "./capitalCapacityEngine";

export interface PaperTradePolicy {
  initialCapital: number;
  minConfidenceToTrade: number;
  minNetEdgeToTrade: number;
  maxCapitalPerTrade: number;
  maxCapitalPerCluster: number;
  maxCapitalPerMarket: number;
  maxHoldingTimeMs: number;
  stopLossPct: number;
  takeProfitPct: number;
  feeBuffer: number;
  edgeDecayFactor: number;
  edgeNormalizationThreshold: number;
  minEpisodeDurationMs?: number;
}

export const DEFAULT_PAPER_POLICY: PaperTradePolicy = {
  initialCapital: 10_000,
  minConfidenceToTrade: 0.15,
  minNetEdgeToTrade: 0.008,
  maxCapitalPerTrade: 500,
  maxCapitalPerCluster: 1000,
  maxCapitalPerMarket: 300,
  maxHoldingTimeMs: 300_000,
  stopLossPct: 0.03,
  takeProfitPct: 0.05,
  feeBuffer: 0.002,
  edgeDecayFactor: 1.5,
  edgeNormalizationThreshold: 0.005,
  minEpisodeDurationMs: 5000,
};

let tradeSeq = 0;

function makeTradeId(): string {
  return `pt-${++tradeSeq}-${Date.now()}`;
}

function hasActiveTradeForOpportunity(opportunityId: string): boolean {
  return getActivePaperTrades().some((t) => t.opportunityId === opportunityId);
}

function toActiveState(t: {
  tradeId: string;
  opportunityId: string;
  openedAt: string;
  entryPriceEstimate: number;
  filledCapital: number;
  maxAdverseExcursion?: number;
  maxFavorableExcursion?: number;
}): ActivePaperTradeState {
  return {
    tradeId: t.tradeId,
    opportunityId: t.opportunityId,
    openedAt: t.openedAt,
    entryEdge: 0,
    entryPriceEstimate: t.entryPriceEstimate,
    filledCapital: t.filledCapital,
    maxAdverseExcursion: t.maxAdverseExcursion ?? 0,
    maxFavorableExcursion: t.maxFavorableExcursion ?? 0,
  };
}

export function processOpportunities(
  opportunities: Array<{ opp: NormalizedPaperOpportunity; capacity: CapacityResult }>,
  policy: Partial<PaperTradePolicy> = {}
): { opened: number; closed: number; rejected: number } {
  const pol = { ...DEFAULT_PAPER_POLICY, ...policy };
  const portfolio = getPaperPortfolio();
  const oppMap = new Map(opportunities.map((o) => [o.opp.opportunityId, o]));

  let opened = 0;
  let closed = 0;
  let rejected = 0;

  for (const { opp, capacity } of opportunities) {
    if (capacity.recommendedCapital <= 0 || capacity.estimatedNetEdge < pol.minNetEdgeToTrade) {
      rejected++;
      continue;
    }
    if (opp.confidence < pol.minConfidenceToTrade) {
      rejected++;
      continue;
    }
    if (hasActiveTradeForOpportunity(opp.opportunityId)) continue;

    const clusterExposure = (opp.clusterId && portfolio.exposureByCluster[opp.clusterId]) || 0;
    const marketExposure = opp.marketsInvolved.reduce(
      (s, m) => s + (portfolio.exposureByMarket[m.marketId] || 0),
      0
    );
    if (clusterExposure >= pol.maxCapitalPerCluster || marketExposure >= pol.maxCapitalPerMarket) {
      rejected++;
      continue;
    }

    const requested = Math.min(
      capacity.recommendedCapital,
      pol.maxCapitalPerTrade,
      portfolio.availableCapital
    );
    if (requested <= 0) {
      rejected++;
      continue;
    }

    const entry: SimulatedEntry = simulateEntry(opp, capacity, portfolio.availableCapital);
    if (entry.filledCapital <= 0) {
      rejected++;
      continue;
    }

    const tradeId = makeTradeId();
    const netEdge = capacity.estimatedNetEdge;
    addActiveTrade({
      tradeId,
      opportunityId: opp.opportunityId,
      sourceType: opp.sourceType,
      opportunityType: opp.opportunityType,
      clusterId: opp.clusterId,
      marketsInvolved: opp.marketsInvolved,
      openedAt: entry.entryTimestamp,
      closedAt: null,
      status: "active",
      grossEdgeAtEntry: opp.edge,
      netEdgeAtEntry: netEdge,
      recommendedCapital: capacity.recommendedCapital,
      requestedCapital: requested,
      filledCapital: entry.filledCapital,
      entryPriceEstimate: entry.entryPriceEstimate,
      entryConfidence: opp.confidence,
      realizedPnL: 0,
      realizedReturn: 0,
      holdingTimeMs: 0,
      maxAdverseExcursion: 0,
      maxFavorableExcursion: 0,
      notes: entry.partialFillFlag ? "partial_fill" : undefined,
    });
    opened++;
  }

  const active = getActivePaperTrades();
  for (const t of active) {
    const latest = oppMap.get(t.opportunityId);
    const latestState = latest
      ? { edge: latest.opp.edge, confidence: latest.opp.confidence }
      : null;

    const shouldClose = shouldClosePaperTrade(toActiveState(t), latestState, {
      maxHoldingTimeMs: pol.maxHoldingTimeMs,
      stopLossPct: pol.stopLossPct,
      takeProfitPct: pol.takeProfitPct,
      edgeNormalizationThreshold: pol.edgeNormalizationThreshold,
    });

    if (shouldClose) {
      const exit: SimulatedExit = simulateExit(toActiveState(t), latestState, {
        maxHoldingTimeMs: pol.maxHoldingTimeMs,
        stopLossPct: pol.stopLossPct,
        takeProfitPct: pol.takeProfitPct,
      });
      closeTrade(t.tradeId, {
        closedAt: exit.exitTimestamp,
        exitPriceEstimate: exit.exitPriceEstimate,
        exitConfidence: latest?.opp.confidence,
        realizedPnL: exit.realizedPnL,
        realizedReturn: exit.realizedReturn,
        holdingTimeMs: Date.now() - new Date(t.openedAt).getTime(),
        maxAdverseExcursion: exit.maxAdverseExcursion,
        maxFavorableExcursion: exit.maxFavorableExcursion,
        exitCondition: exit.exitCondition,
      });
      closed++;
    }
  }

  return { opened, closed, rejected };
}
