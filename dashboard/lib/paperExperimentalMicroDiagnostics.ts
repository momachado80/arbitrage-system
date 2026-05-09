/**
 * Diagnósticos das micro-lanes experimentais (split estrutural pós-violação equivalence).
 * Leitura pura sobre portfolio + último snapshot do graph scan.
 */

import { getPaperPortfolio } from "./paperPortfolioStore";
import { isClosedTradeWithFiniteRealizedPnl } from "./paperClosedTradesMetrics";
import {
  getClosedTradeGrossRealizedPnL,
  getClosedTradeNetRealizedPnL,
  safeFeeBufferPerLeg,
} from "./paperRealizedPnlSemantics";
import { resolvePaperPolicyFromEnv } from "./paperTradeEngine";
import type { PaperOpportunityType, PaperTrade } from "./paperTypes";
import { getGraphScanRuntime } from "./nodeProcessRuntimeState";
import type { StructuralMicroLaneScanSnapshot } from "./graphStructuralMicroLane";

const GROSS_ZERO_EPS = 0.01;
const MIN_FOR_VERDICT = 5;

const STRUCTURAL_MICRO_TYPES = [
  "graph_equivalence_micro",
  "graph_subset_micro",
  "graph_exclusive_micro",
] as const satisfies readonly PaperOpportunityType[];

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function avg(arr: number[]): number | null {
  if (arr.length === 0) return null;
  return round4(arr.reduce((a, b) => a + b, 0) / arr.length);
}

export type ExperimentalStructuralMicroPortfolioLane = {
  opportunityType: PaperOpportunityType;
  activeTradesCount: number;
  closedTradesCount: number;
  avgGrossRealizedPnl: number | null;
  avgNetRealizedPnl: number | null;
  avgEdgeAtEntry: number | null;
  avgExitPriceEstimate: number | null;
  fallbackNoLatestCloseCount: number;
  grossZeroNetNegCount: number;
  verdict: "insufficient_data" | "promising" | "destructive";
};

function buildPortfolioLaneStats(oppType: PaperOpportunityType): ExperimentalStructuralMicroPortfolioLane {
  const portfolio = getPaperPortfolio();
  const feeBuf = safeFeeBufferPerLeg(resolvePaperPolicyFromEnv().feeBuffer);

  const active = portfolio.activeTrades.filter((t) => t.opportunityType === oppType);
  const closed = portfolio.closedTrades.filter(
    (t) => t.opportunityType === oppType && isClosedTradeWithFiniteRealizedPnl(t)
  );

  const grossPnls: number[] = [];
  const netPnls: number[] = [];
  const edges: number[] = [];
  const exitPrices: number[] = [];
  let fallbackNoLatest = 0;
  let grossZeroNetNeg = 0;

  for (const t of closed) {
    const gross = getClosedTradeGrossRealizedPnL(t);
    const net = getClosedTradeNetRealizedPnL(t, feeBuf);
    grossPnls.push(gross);
    netPnls.push(net);

    if (typeof t.grossEdgeAtEntry === "number" && Number.isFinite(t.grossEdgeAtEntry)) {
      edges.push(t.grossEdgeAtEntry);
    }
    if (typeof t.exitPriceEstimate === "number" && Number.isFinite(t.exitPriceEstimate)) {
      exitPrices.push(t.exitPriceEstimate);
    }
    if (t.exitPriceMarkSourceAtClose === "fallback_no_latest") fallbackNoLatest += 1;
    if (Math.abs(gross) < GROSS_ZERO_EPS && net < -GROSS_ZERO_EPS) grossZeroNetNeg += 1;
  }

  const n = closed.length;
  let verdict: "insufficient_data" | "promising" | "destructive" = "insufficient_data";
  if (n >= MIN_FOR_VERDICT) {
    const avgNet = avg(netPnls);
    const fallbackShare = fallbackNoLatest / n;
    const grossZeroShare = grossZeroNetNeg / n;
    if (fallbackShare >= 0.5 || grossZeroShare >= 0.5 || (avgNet != null && avgNet < 0)) {
      verdict = "destructive";
    } else {
      verdict = "promising";
    }
  }

  return {
    opportunityType: oppType,
    activeTradesCount: active.length,
    closedTradesCount: n,
    avgGrossRealizedPnl: avg(grossPnls),
    avgNetRealizedPnl: avg(netPnls),
    avgEdgeAtEntry: avg(edges),
    avgExitPriceEstimate: avg(exitPrices),
    fallbackNoLatestCloseCount: fallbackNoLatest,
    grossZeroNetNegCount: grossZeroNetNeg,
    verdict,
  };
}

export type ExperimentalMicroLaneDiagnostics = {
  structuralMicroHypothesisNote: string;
  portfolioByStructuralLane: {
    graphEquivalenceMicro: ExperimentalStructuralMicroPortfolioLane;
    graphSubsetMicro: ExperimentalStructuralMicroPortfolioLane;
    graphExclusiveMicro: ExperimentalStructuralMicroPortfolioLane;
  };
  /** Último graph scan: contagens + amostras por tipo (cachedGraphRaw). */
  lastStructuralMicroLaneScan: StructuralMicroLaneScanSnapshot | null;
  /** Post-mortem da família graph após falha económica em todas as lanes. */
  graphFamilyPostMortem: GraphFamilyPostMortem;
};

export function buildExperimentalMicroLaneDiagnostics(): ExperimentalMicroLaneDiagnostics {
  const scan = getGraphScanRuntime().lastStructuralMicroLaneScan;

  return {
    structuralMicroHypothesisNote:
      "FROZEN. Todas as micro-lanes graph falharam economicamente. Seed blocklist expandido para bloquear graph_equivalence_micro, graph_subset_micro, graph_exclusive_micro. Ver graphFamilyPostMortem para análise detalhada.",
    portfolioByStructuralLane: {
      graphEquivalenceMicro: buildPortfolioLaneStats("graph_equivalence_micro"),
      graphSubsetMicro: buildPortfolioLaneStats("graph_subset_micro"),
      graphExclusiveMicro: buildPortfolioLaneStats("graph_exclusive_micro"),
    },
    lastStructuralMicroLaneScan: scan,
    graphFamilyPostMortem: buildGraphFamilyPostMortem(),
  };
}

/** Para métricas que somam todas as micro-lanes estruturais. */
export function isStructuralGraphMicroType(ot: PaperTrade["opportunityType"]): boolean {
  return (STRUCTURAL_MICRO_TYPES as readonly string[]).includes(ot);
}

// ---------------------------------------------------------------------------
// Post-mortem: graph experimental family failure analysis
// ---------------------------------------------------------------------------

export type GraphFamilyPostMortemTradeSample = {
  tradeId: string;
  opportunityType: PaperOpportunityType;
  grossEdgeAtEntry: number;
  entryPriceEstimate: number;
  exitPriceEstimate: number | null;
  exitPriceMarkSource: string | null;
  edgeAtExit: number | null;
  exitCondition: string | null;
  grossRealizedPnL: number;
  netRealizedPnL: number;
  holdingTimeMs: number;
};

export type GraphFamilyPostMortem = {
  computedAt: string;
  status: "frozen_after_failed_economic_validation";
  totalClosedGraphMicroTrades: number;
  byLane: Record<
    string,
    {
      closedCount: number;
      avgGross: number | null;
      avgNet: number | null;
      grossZeroNetNegCount: number;
      dominantExitCondition: string | null;
      dominantExitConditionShare: number | null;
      avgEntryPrice: number | null;
      avgExitPrice: number | null;
      entryExitPriceDeltaZeroCount: number;
    }
  >;
  findings: {
    whyGrossRemainedZero: string;
    whyNetStayedNegative: string;
    validMarkButNoConvergence: string;
    structuralConclusion: string;
  };
  tradeSamples: GraphFamilyPostMortemTradeSample[];
};

export function buildGraphFamilyPostMortem(): GraphFamilyPostMortem {
  const portfolio = getPaperPortfolio();
  const feeBuf = safeFeeBufferPerLeg(resolvePaperPolicyFromEnv().feeBuffer);

  const allMicroClosed = portfolio.closedTrades.filter(
    (t) => isStructuralGraphMicroType(t.opportunityType) && isClosedTradeWithFiniteRealizedPnl(t)
  );

  const byLane: GraphFamilyPostMortem["byLane"] = {};

  for (const laneType of STRUCTURAL_MICRO_TYPES) {
    const trades = allMicroClosed.filter((t) => t.opportunityType === laneType);
    const n = trades.length;
    const grossPnls: number[] = [];
    const netPnls: number[] = [];
    const entryPrices: number[] = [];
    const exitPrices: number[] = [];
    let grossZeroNetNeg = 0;
    let entryExitDeltaZero = 0;
    const exitConditionCounts = new Map<string, number>();

    for (const t of trades) {
      const gross = getClosedTradeGrossRealizedPnL(t);
      const net = getClosedTradeNetRealizedPnL(t, feeBuf);
      grossPnls.push(gross);
      netPnls.push(net);
      if (Math.abs(gross) < GROSS_ZERO_EPS && net < -GROSS_ZERO_EPS) grossZeroNetNeg += 1;
      const ep = t.entryPriceEstimate;
      const xp = typeof t.exitPriceEstimate === "number" ? t.exitPriceEstimate : null;
      if (typeof ep === "number") entryPrices.push(ep);
      if (xp != null) {
        exitPrices.push(xp);
        if (Math.abs(xp - ep) < 1e-6) entryExitDeltaZero += 1;
      }
      const ec = t.exitCondition ?? "unknown";
      exitConditionCounts.set(ec, (exitConditionCounts.get(ec) ?? 0) + 1);
    }

    let dominantExitCondition: string | null = null;
    let dominantExitConditionCount = 0;
    for (const [ec, cnt] of Array.from(exitConditionCounts.entries())) {
      if (cnt > dominantExitConditionCount) {
        dominantExitConditionCount = cnt;
        dominantExitCondition = ec;
      }
    }

    byLane[laneType] = {
      closedCount: n,
      avgGross: avg(grossPnls),
      avgNet: avg(netPnls),
      grossZeroNetNegCount: grossZeroNetNeg,
      dominantExitCondition,
      dominantExitConditionShare: n > 0 ? round4(dominantExitConditionCount / n) : null,
      avgEntryPrice: avg(entryPrices),
      avgExitPrice: avg(exitPrices),
      entryExitPriceDeltaZeroCount: entryExitDeltaZero,
    };
  }

  const totalClosed = allMicroClosed.length;
  const totalGrossZeroNetNeg = allMicroClosed.filter((t) => {
    const gross = getClosedTradeGrossRealizedPnL(t);
    const net = getClosedTradeNetRealizedPnL(t, feeBuf);
    return Math.abs(gross) < GROSS_ZERO_EPS && net < -GROSS_ZERO_EPS;
  }).length;
  const totalEntryExitDeltaZero = allMicroClosed.filter((t) => {
    const xp = typeof t.exitPriceEstimate === "number" ? t.exitPriceEstimate : null;
    return xp != null && Math.abs(xp - t.entryPriceEstimate) < 1e-6;
  }).length;
  const maxHoldingTimeExits = allMicroClosed.filter((t) => t.exitCondition === "max_holding_time").length;
  const validMarkNotFallback = allMicroClosed.filter(
    (t) => t.exitPriceMarkSourceAtClose != null && t.exitPriceMarkSourceAtClose !== "fallback_no_latest"
  ).length;

  const samples: GraphFamilyPostMortemTradeSample[] = allMicroClosed.slice(-10).map((t) => ({
    tradeId: t.tradeId,
    opportunityType: t.opportunityType,
    grossEdgeAtEntry: t.grossEdgeAtEntry,
    entryPriceEstimate: t.entryPriceEstimate,
    exitPriceEstimate: typeof t.exitPriceEstimate === "number" ? t.exitPriceEstimate : null,
    exitPriceMarkSource: t.exitPriceMarkSourceAtClose ?? null,
    edgeAtExit: typeof t.edgeAtExit === "number" ? t.edgeAtExit : null,
    exitCondition: t.exitCondition ?? null,
    grossRealizedPnL: getClosedTradeGrossRealizedPnL(t),
    netRealizedPnL: getClosedTradeNetRealizedPnL(t, feeBuf),
    holdingTimeMs: t.holdingTimeMs,
  }));

  return {
    computedAt: new Date().toISOString(),
    status: "frozen_after_failed_economic_validation",
    totalClosedGraphMicroTrades: totalClosed,
    byLane,
    findings: {
      whyGrossRemainedZero:
        `${totalGrossZeroNetNeg}/${totalClosed} trades closed with |gross| < ${GROSS_ZERO_EPS} and net < 0. ` +
        `${totalEntryExitDeltaZero}/${totalClosed} had exitPrice ≈ entryPrice (delta < 1e-6). ` +
        "The graph-detected edge never materialized as a price movement captured by the paper engine. " +
        "Edge existed at entry but the mark price at exit converged back to entry, yielding zero gross return.",
      whyNetStayedNegative:
        "With gross ≈ 0, the estimated round-trip fees (entry + exit legs) produce a strictly negative net PnL on every trade. " +
        "The fee structure is not the cause — the cause is zero gross capture.",
      validMarkButNoConvergence:
        `${validMarkNotFallback}/${totalClosed} exits used valid mark (mtm or opp_map), not fallback_no_latest. ` +
        `${maxHoldingTimeExits}/${totalClosed} exited via max_holding_time. ` +
        "The mark infrastructure is working correctly. The problem is that the underlying arbitrage signal " +
        "(graph constraint violation on equivalent/subset/exclusive markets) does not translate into a " +
        "realizable price convergence within the holding window.",
      structuralConclusion:
        "The graph family in its current economic formulation is not monetizing. " +
        "The detected structural violations are real in probability space but do not " +
        "produce directional price movements capturable by the paper engine's entry/exit model. " +
        "All three micro-lanes (equivalence, subset, exclusive) exhibit the same pattern. " +
        "Family is frozen pending a fundamentally different monetization approach or evidence of edge capture.",
    },
    tradeSamples: samples,
  };
}
