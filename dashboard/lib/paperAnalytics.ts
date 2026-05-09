/**
 * Paper Analytics — computes portfolio and strategy metrics from paper trades.
 */

import type { PaperTrade } from "./paperTypes";
import { aggregateClosedTradesByGraphProvenance } from "./graphOpportunityPaperImpact";
import {
  isClosedTradeWithFiniteRealizedPnl,
  isGraphTradeForProvenanceMetrics,
} from "./paperClosedTradesMetrics";
import {
  buildFeeImpactAudit,
  type FeeImpactAudit,
} from "./graphProvenanceQualityAudit";
import {
  DEFAULT_PAPER_FEE_BUFFER_PER_LEG,
  getClosedTradeEstimatedTotalFees,
  getClosedTradeGrossRealizedPnL,
  getClosedTradeNetRealizedPnL,
  safeFeeBufferPerLeg,
} from "./paperRealizedPnlSemantics";

export interface PaperCausePnlRow {
  cause: string;
  pnl: number;
}

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
  /** Soma PnL bruto (simulateExit / grossRealizedPnL ou legado). */
  pnlGrossRealized: number;
  /** Soma taxas persistidas ou filledCapital × feeBuffer × 2. */
  estimatedRoundTripFees: number;
  /** Alias semântico de `estimatedRoundTripFees` (APIs / jq). */
  totalEstimatedFees: number;
  /** Igual a `totalPnL` (soma líquida); também gross − fees quando consistente. */
  estimatedNetPnl: number;
  /** Impacto de taxas por proveniência (fechados graph com PnL finito). */
  feeImpactAudit: FeeImpactAudit;
  /** Top chaves exitCondition|opportunityType por PnL positivo. */
  topGainCauses: PaperCausePnlRow[];
  /** Top chaves por PnL negativo (mais negativo primeiro). */
  topLossCauses: PaperCausePnlRow[];
  /** Trades paper (ativos + fechados), igual a totalTrades. */
  paperTradesCount: number;
  /** Agregado hoje (UTC) no ciclo paper; 0 se ainda não houve ciclos. */
  opportunitiesSeenToday: number;
  /** Com recommendedCapital > 0 após capacity, agregado hoje (UTC). */
  opportunitiesExecutableToday: number;
  /** Agregados por `exitCondition` (inclui causas dinâmicas quando presentes). */
  countByExitReason: Record<string, number>;
  avgPnLByExitReason: Record<string, number>;
  avgHoldingTimeByExitReason: Record<string, number>;
  avgCapturedEdgeRatioByExitReason: Record<string, number>;
  avgExpectedRemainingEdgeValueAtExit: number;
  avgDrawdownFromPeakAtExit: number;
  /** Fechados `graph` por proveniência à abertura (O(n) sobre trades). */
  graphProvenanceClosedTrades: ReturnType<typeof aggregateClosedTradesByGraphProvenance>;
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

function aggregateExitReasonMetrics(closed: PaperTrade[], feeBufferPerLeg: number): {
  countByExitReason: Record<string, number>;
  avgPnLByExitReason: Record<string, number>;
  avgHoldingTimeByExitReason: Record<string, number>;
  avgCapturedEdgeRatioByExitReason: Record<string, number>;
  avgExpectedRemainingEdgeValueAtExit: number;
  avgDrawdownFromPeakAtExit: number;
} {
  type Acc = {
    n: number;
    pnl: number;
    hold: number;
    capSum: number;
    capN: number;
    remSum: number;
    remN: number;
    ddSum: number;
    ddN: number;
  };
  const m = new Map<string, Acc>();
  let remAll = 0;
  let remAllN = 0;
  let ddAll = 0;
  let ddAllN = 0;

  for (const t of closed) {
    const reason = t.exitCondition ?? "unknown";
    const pnl = getClosedTradeNetRealizedPnL(t, feeBufferPerLeg);
    const hold = safeNum(t.holdingTimeMs);
    const snap = t.exitDecisionSnapshot;
    let acc = m.get(reason);
    if (!acc) {
      acc = { n: 0, pnl: 0, hold: 0, capSum: 0, capN: 0, remSum: 0, remN: 0, ddSum: 0, ddN: 0 };
      m.set(reason, acc);
    }
    acc.n += 1;
    acc.pnl += pnl;
    acc.hold += hold;
    if (snap && typeof snap.capturedEdgeRatio === "number" && Number.isFinite(snap.capturedEdgeRatio)) {
      acc.capSum += snap.capturedEdgeRatio;
      acc.capN += 1;
    }
    if (
      snap &&
      typeof snap.expectedRemainingEdgeValue === "number" &&
      Number.isFinite(snap.expectedRemainingEdgeValue)
    ) {
      acc.remSum += snap.expectedRemainingEdgeValue;
      acc.remN += 1;
      remAll += snap.expectedRemainingEdgeValue;
      remAllN += 1;
    }
    if (snap && typeof snap.drawdownFromPeakPnL === "number" && Number.isFinite(snap.drawdownFromPeakPnL)) {
      acc.ddSum += snap.drawdownFromPeakPnL;
      acc.ddN += 1;
      ddAll += snap.drawdownFromPeakPnL;
      ddAllN += 1;
    }
  }

  const countByExitReason: Record<string, number> = {};
  const avgPnLByExitReason: Record<string, number> = {};
  const avgHoldingTimeByExitReason: Record<string, number> = {};
  const avgCapturedEdgeRatioByExitReason: Record<string, number> = {};

  for (const [reason, acc] of Array.from(m.entries())) {
    countByExitReason[reason] = acc.n;
    avgPnLByExitReason[reason] = acc.n > 0 ? acc.pnl / acc.n : 0;
    avgHoldingTimeByExitReason[reason] = acc.n > 0 ? acc.hold / acc.n : 0;
    avgCapturedEdgeRatioByExitReason[reason] = acc.capN > 0 ? acc.capSum / acc.capN : 0;
  }

  return {
    countByExitReason,
    avgPnLByExitReason,
    avgHoldingTimeByExitReason,
    avgCapturedEdgeRatioByExitReason,
    avgExpectedRemainingEdgeValueAtExit: remAllN > 0 ? remAll / remAllN : 0,
    avgDrawdownFromPeakAtExit: ddAllN > 0 ? ddAll / ddAllN : 0,
  };
}

function topCausesByPnl(
  closed: PaperTrade[],
  winners: boolean,
  limit: number,
  feeBufferPerLeg: number
): PaperCausePnlRow[] {
  const m = new Map<string, number>();
  for (const t of closed) {
    const p = getClosedTradeNetRealizedPnL(t, feeBufferPerLeg);
    if (winners && p <= 0) continue;
    if (!winners && p >= 0) continue;
    const key = `${t.exitCondition ?? "unknown"}|${t.opportunityType ?? "unknown"}`;
    m.set(key, (m.get(key) ?? 0) + p);
  }
  const rows = Array.from(m.entries()).map(([cause, pnl]) => ({ cause, pnl }));
  if (winners) {
    rows.sort((a, b) => b.pnl - a.pnl);
  } else {
    rows.sort((a, b) => a.pnl - b.pnl);
  }
  return rows.slice(0, limit);
}

export function computePaperAnalytics(
  closedTrades: PaperTrade[],
  activeTrades: PaperTrade[],
  startingCapital: number,
  currentEquity: number,
  maxDrawdown: number,
  feeBufferPerLeg: number = 0.002,
  opportunityDayCounts?: { seen: number; executable: number }
): PaperAnalyticsResult {
  const feeBuf = safeFeeBufferPerLeg(feeBufferPerLeg);
  const actives = Array.isArray(activeTrades) ? activeTrades : [];
  const closed = closedTrades.filter(isClosedTradeWithFiniteRealizedPnl);
  const total = closed.length + actives.length;

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
  let estimatedRoundTripFees = 0;
  let grossRealizedSum = 0;

  if (closed.length > 0) {
    const wins = closed.filter((t) => getClosedTradeNetRealizedPnL(t, feeBuf) > 0).length;
    winRate = wins / closed.length;

    const returns = closed.map((t) => {
      const fc = safeNum(t.filledCapital);
      const net = getClosedTradeNetRealizedPnL(t, feeBuf);
      return fc > 0 ? net / fc : safeNum(t.realizedReturn);
    });
    avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;

    const pnls = closed.map((t) => getClosedTradeNetRealizedPnL(t, feeBuf));
    totalPnL = pnls.reduce((a, b) => a + b, 0);
    avgPnL = totalPnL / closed.length;

    for (const t of closed) {
      const p = getClosedTradeNetRealizedPnL(t, feeBuf);
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
      const net = getClosedTradeNetRealizedPnL(t, feeBuf);
      pnlByType[type] = (pnlByType[type] || 0) + net;
      const src = t.sourceType || "standard";
      pnlBySource[src] = (pnlBySource[src] || 0) + net;
      estimatedRoundTripFees += getClosedTradeEstimatedTotalFees(t, feeBuf);
      grossRealizedSum += getClosedTradeGrossRealizedPnL(t);
    }
  }

  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 10 : 0;
  const pnlGrossRealized = grossRealizedSum;
  /** Soma líquida por trade; deve alinhar com `pnlGrossRealized - estimatedRoundTripFees` salvo arredondamentos. */
  const estimatedNetPnl = totalPnL;
  const topGainCauses = topCausesByPnl(closed, true, 5, feeBuf);
  const topLossCauses = topCausesByPnl(closed, false, 5, feeBuf);

  const reserved = actives.reduce((s, t) => s + safeNum(t.filledCapital), 0);
  const utilizationRate = startingCapital > 0 ? reserved / startingCapital : 0;

  let sharpeLike: number | undefined;
  if (closed.length >= 5) {
    const returns = closed.map((t) => {
      const fc = safeNum(t.filledCapital);
      const net = getClosedTradeNetRealizedPnL(t, feeBuf);
      return fc > 0 ? net / fc : safeNum(t.realizedReturn);
    });
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / returns.length;
    const std = Math.sqrt(variance) || 0.0001;
    sharpeLike = std > 0 ? (mean / std) * Math.sqrt(252) : 0;
  }

  const exitAgg = aggregateExitReasonMetrics(closed, feeBuf);
  const graphProvClosed = aggregateClosedTradesByGraphProvenance(closed, feeBuf);
  const graphClosedForFeeAudit = closed.filter(isGraphTradeForProvenanceMetrics);
  const feeImpactAudit = buildFeeImpactAudit(graphClosedForFeeAudit, feeBuf);

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
    pnlGrossRealized,
    estimatedRoundTripFees,
    totalEstimatedFees: estimatedRoundTripFees,
    estimatedNetPnl,
    feeImpactAudit,
    topGainCauses,
    topLossCauses,
    paperTradesCount: total,
    opportunitiesSeenToday: opportunityDayCounts?.seen ?? 0,
    opportunitiesExecutableToday: opportunityDayCounts?.executable ?? 0,
    countByExitReason: exitAgg.countByExitReason,
    avgPnLByExitReason: exitAgg.avgPnLByExitReason,
    avgHoldingTimeByExitReason: exitAgg.avgHoldingTimeByExitReason,
    avgCapturedEdgeRatioByExitReason: exitAgg.avgCapturedEdgeRatioByExitReason,
    avgExpectedRemainingEdgeValueAtExit: exitAgg.avgExpectedRemainingEdgeValueAtExit,
    avgDrawdownFromPeakAtExit: exitAgg.avgDrawdownFromPeakAtExit,
    graphProvenanceClosedTrades: graphProvClosed,
  };
}

export function computeEquityCurve(
  closedTrades: PaperTrade[],
  startingCapital: number,
  feeBufferPerLeg: number = DEFAULT_PAPER_FEE_BUFFER_PER_LEG
): EquityPoint[] {
  const sorted = [...closedTrades]
    .filter((t) => isClosedTradeWithFiniteRealizedPnl(t) && t.closedAt)
    .sort((a, b) => new Date(a.closedAt!).getTime() - new Date(b.closedAt!).getTime());

  const curve: EquityPoint[] = [{ ts: new Date(0).toISOString(), equity: startingCapital, cumulativePnL: 0 }];
  let cum = 0;

  const buf = safeFeeBufferPerLeg(feeBufferPerLeg);
  for (const t of sorted) {
    cum += getClosedTradeNetRealizedPnL(t, buf);
    curve.push({
      ts: t.closedAt!,
      equity: startingCapital + cum,
      cumulativePnL: cum,
    });
  }

  return curve;
}
