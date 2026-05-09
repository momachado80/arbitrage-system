/**
 * Motor marginal de saída dinâmica (paper): métricas normalizadas e prioridade fixa entre causas.
 */

import type {
  ExitCondition,
  PaperExitDecisionSnapshot,
  PaperExitDiagnosticInputs,
  PaperMtmEdgeDivergenceExitSnapshot,
  PaperNoProgressExitSnapshot,
  PaperTrade,
} from "./paperTypes";
import type { PaperDynamicExitRuntimeConfig } from "./paperDynamicExitConfig";

export interface DynamicExitComputedMetrics {
  timeInTradeMs: number;
  entryNetEdge: number;
  currentNetEdge: number;
  currentGrossEdge: number;
  capturedEdgeRatio: number;
  edgeVelocity: number;
  expectedRemainingEdgeValue: number;
  bestMarkedPnL: number;
  worstMarkedPnL: number;
  currentMarkedPnL: number;
  drawdownFromPeakPnL: number;
  capitalEfficiencyScore: number;
  diagnosticInputs: PaperExitDiagnosticInputs;
}

export interface DynamicExitResult {
  shouldExit: boolean;
  cause: ExitCondition | null;
  decisionSnapshot: PaperExitDecisionSnapshot | null;
  computedMetrics: DynamicExitComputedMetrics;
}

function safeNum(v: unknown, d = 0): number {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  return d;
}

/** Gross edge corrente: live opp > MTM sintético > entrada. */
export function resolveCurrentGrossEdge(
  trade: PaperTrade,
  latestOpp: { edge: number } | null,
  syntheticGrossEdge: number | null
): number {
  if (latestOpp != null && typeof latestOpp.edge === "number" && Number.isFinite(latestOpp.edge)) {
    return latestOpp.edge;
  }
  if (syntheticGrossEdge != null && Number.isFinite(syntheticGrossEdge)) {
    return syntheticGrossEdge;
  }
  return safeNum(trade.grossEdgeAtEntry);
}

/**
 * Actualiza métricas dinâmicas no objecto do trade (mutação no store).
 * Sem live opp usa fallback sintético coerente com o engine.
 */
export function applyDynamicMetricsToTrade(
  trade: PaperTrade,
  latestOpp: { edge: number } | null,
  syntheticGrossEdge: number | null,
  feeBuffer: number,
  nowMs: number
): DynamicExitComputedMetrics {
  const openedAt = new Date(trade.openedAt).getTime();
  const timeInTradeMs = Math.max(0, nowMs - openedAt);
  const entryNetEdge = safeNum(trade.netEdgeAtEntry);
  const g0 = safeNum(trade.grossEdgeAtEntry);
  const filled = Math.max(0, safeNum(trade.filledCapital));
  const entry = Math.max(1e-9, safeNum(trade.entryPriceEstimate));

  /** Preço de marcação — mesma base que o MTM; antes da edge vinha só de latestOpp e podia divergir de lastMarkPx. */
  let markPxSource: PaperExitDiagnosticInputs["markPxSource"];
  let markPx: number;
  if (trade.lastMarkPx != null && Number.isFinite(trade.lastMarkPx)) {
    markPxSource = "lastMarkPx";
    markPx = trade.lastMarkPx;
  } else if (latestOpp != null) {
    markPxSource = "latestOpp";
    markPx = 1 - latestOpp.edge;
  } else {
    markPxSource = "entry_fallback";
    markPx = entry;
  }

  /**
   * Edge corrente implícita no mark (convenção do simulador: markPx = 1 - edge_mtm).
   * Isto alinha currentNetEdge / capturedEdge / valor restante com currentMarkedPnL.
   * Não usar resolveCurrentGrossEdge(latestOpp) quando lastMarkPx existe — era a fonte do desvio.
   */
  const currentGrossEdge = 1 - markPx;
  const currentNetEdge = currentGrossEdge - feeBuffer;

  const prevNetForVelocity = trade.lastNetEdge ?? entryNetEdge;
  const prevTick = trade.lastDynamicTickAt ?? openedAt;
  const deltaT = Math.max(1000, nowMs - prevTick);
  const edgeVelocity = (currentNetEdge - prevNetForVelocity) / deltaT;

  const capDen = g0;
  const capNum = g0 > 1e-9 ? g0 - currentGrossEdge : 0;
  const ratio = capDen > 1e-9 ? Math.max(0, Math.min(1, capNum / capDen)) : 0;

  // Proxy marginal (netEdge × capital), não um modelo económico maduro — recalibrar thresholds com analytics.
  const expectedRemainingEdgeValue = currentNetEdge * filled;
  const capitalEfficiencyScore =
    filled > 0 ? expectedRemainingEdgeValue / filled : currentNetEdge;

  const currentMarkedPnL = filled * ((markPx - entry) / entry);

  void syntheticGrossEdge; /* resolvido por markPx; mantido na assinatura para o engine */

  const prevBest = trade.bestMarkedPnL;
  const prevWorst = trade.worstMarkedPnL;
  const bestMarkedPnL =
    prevBest === undefined ? currentMarkedPnL : Math.max(prevBest, currentMarkedPnL);
  const worstMarkedPnL =
    prevWorst === undefined ? currentMarkedPnL : Math.min(prevWorst, currentMarkedPnL);

  const peakForDrawdown = prevBest === undefined ? bestMarkedPnL : Math.max(prevBest, currentMarkedPnL);
  const drawdownFromPeakPnL = Math.max(0, peakForDrawdown - currentMarkedPnL);

  const diagnosticInputs: PaperExitDiagnosticInputs = {
    markPx,
    markPxSource,
    grossEdgeFromMark: currentGrossEdge,
    grossEdgeAtEntry: g0,
    feeBuffer,
    capturedEdgeNumerator: capNum,
    capturedEdgeDenominator: capDen,
    remainingValueInputs: { currentNetEdge, filledCapital: filled },
    mtmPnlInputs: { markPx, entryPriceEstimate: entry, filledCapital: filled },
    exitSignalConsistency: {
      currentNetEdgePositiveButPnlNegative: currentNetEdge > 1e-9 && currentMarkedPnL < -1e-9,
      capturedEdgeZeroButPnlNonZero: ratio < 1e-9 && Math.abs(currentMarkedPnL) > 1e-6,
      remainingValueHighButDrawdownHigh:
        drawdownFromPeakPnL > 1e-6 &&
        expectedRemainingEdgeValue > drawdownFromPeakPnL &&
        currentMarkedPnL < -1e-6,
    },
  };

  trade.currentNetEdge = currentNetEdge;
  trade.capturedEdgeRatio = ratio;
  trade.edgeVelocity = edgeVelocity;
  trade.expectedRemainingEdgeValue = expectedRemainingEdgeValue;
  trade.capitalEfficiencyScore = capitalEfficiencyScore;
  trade.drawdownFromPeakPnL = drawdownFromPeakPnL;
  trade.bestMarkedPnL = bestMarkedPnL;
  trade.worstMarkedPnL = worstMarkedPnL;
  if (prevBest === undefined || currentMarkedPnL >= (prevBest ?? -Infinity)) {
    trade.bestMarkedAt = new Date(nowMs).toISOString();
  }
  trade.lastDynamicTickAt = nowMs;
  trade.lastNetEdge = currentNetEdge;

  return {
    timeInTradeMs,
    entryNetEdge,
    currentNetEdge,
    currentGrossEdge,
    capturedEdgeRatio: ratio,
    edgeVelocity,
    expectedRemainingEdgeValue,
    bestMarkedPnL,
    worstMarkedPnL,
    currentMarkedPnL,
    drawdownFromPeakPnL,
    capitalEfficiencyScore,
    diagnosticInputs,
  };
}

function thresholdsRecord(cfg: PaperDynamicExitRuntimeConfig): Record<string, number | boolean | string> {
  const th = cfg.thresholds;
  return {
    ENABLE_DYNAMIC_EXIT_ENGINE: cfg.engine,
    ENABLE_EXIT_PROFIT_GIVEBACK: cfg.profitGiveback,
    ENABLE_EXIT_INCREMENTAL_VALUE: cfg.incrementalValue,
    ENABLE_EXIT_CAPTURED_EDGE: cfg.capturedEdge,
    ENABLE_EXIT_EDGE_DETERIORATION: cfg.edgeDeteriorationFast,
    ENABLE_EXIT_CAPITAL_EFFICIENCY: cfg.capitalEfficiency,
    ENABLE_EXIT_MTM_EDGE_DIVERGENCE: cfg.mtmEdgeDivergence,
    REQUIRE_REMAINING_HIGH_FOR_MTM_DIVERGENCE: cfg.requireRemainingHighForMtmDivergence,
    minPnLToActivateTrailing: th.minPnLToActivateTrailing,
    allowedGivebackUsd: th.allowedGivebackUsd,
    minIncrementalEdgeUsd: th.minIncrementalEdgeUsd,
    capturedEdgeRatioExit: th.capturedEdgeRatioExit,
    minRemainingValueThresholdUsd: th.minRemainingValueThresholdUsd,
    negativeVelocityThreshold: th.negativeVelocityThreshold,
    deteriorationFloorRatio: th.deteriorationFloorRatio,
    minCapitalEfficiencyThreshold: th.minCapitalEfficiencyThreshold,
    minHoldBeforeEfficiencyExitMs: th.minHoldBeforeEfficiencyExitMs,
    minHoldBeforeIncrementalExitMs: th.minHoldBeforeIncrementalExitMs,
    emergencyExitTimeMs: th.emergencyExitTimeMs,
    minHoldBeforeMtmDivergenceExitMs: th.minHoldBeforeMtmDivergenceExitMs,
    minNegativePnlForMtmDivergenceExitUsd: th.minNegativePnlForMtmDivergenceExitUsd,
    minDrawdownForMtmDivergenceExitUsd: th.minDrawdownForMtmDivergenceExitUsd,
    ENABLE_EXIT_NO_PROGRESS: cfg.noProgress,
    minHoldBeforeNoProgressExitMs: th.minHoldBeforeNoProgressExitMs,
    maxFlatPnlAbsUsd: th.maxFlatPnlAbsUsd,
    maxCapturedEdgeRatioForNoProgressExit: th.maxCapturedEdgeRatioForNoProgressExit,
    maxFlatEdgeVelocityAbs: th.maxFlatEdgeVelocityAbs,
    noProgressMaxDrawdownFromPeakUsd: th.noProgressMaxDrawdownFromPeakUsd,
    skipNoProgressIfRemainingAboveUsd: th.skipNoProgressIfRemainingAboveUsd,
  };
}

function buildMtmEdgeDivergenceExitSnapshot(
  m: DynamicExitComputedMetrics,
  cfg: PaperDynamicExitRuntimeConfig
): PaperMtmEdgeDivergenceExitSnapshot {
  const th = cfg.thresholds;
  const div = m.diagnosticInputs.exitSignalConsistency;
  return {
    mtmEdgeDivergenceTriggered: true,
    thresholdsApplied: {
      minHoldBeforeMtmDivergenceExitMs: th.minHoldBeforeMtmDivergenceExitMs,
      minNegativePnlForMtmDivergenceExitUsd: th.minNegativePnlForMtmDivergenceExitUsd,
      minDrawdownForMtmDivergenceExitUsd: th.minDrawdownForMtmDivergenceExitUsd,
      requireRemainingHighForMtmDivergence: cfg.requireRemainingHighForMtmDivergence,
    },
    valuesAtDecision: {
      timeInTradeMs: m.timeInTradeMs,
      currentNetEdge: m.currentNetEdge,
      currentMarkedPnL: m.currentMarkedPnL,
      drawdownFromPeakPnL: m.drawdownFromPeakPnL,
      expectedRemainingEdgeValue: m.expectedRemainingEdgeValue,
      exitSignalConsistency: {
        currentNetEdgePositiveButPnlNegative: div.currentNetEdgePositiveButPnlNegative,
        capturedEdgeZeroButPnlNonZero: div.capturedEdgeZeroButPnlNonZero,
        remainingValueHighButDrawdownHigh: div.remainingValueHighButDrawdownHigh,
      },
    },
  };
}

function shouldExitMtmEdgeDivergence(
  m: DynamicExitComputedMetrics,
  cfg: PaperDynamicExitRuntimeConfig
): boolean {
  if (!cfg.mtmEdgeDivergence) return false;
  const th = cfg.thresholds;
  const div = m.diagnosticInputs.exitSignalConsistency;
  if (!div.currentNetEdgePositiveButPnlNegative) return false;
  if (m.currentNetEdge <= 1e-9) return false;
  if (m.currentMarkedPnL >= -th.minNegativePnlForMtmDivergenceExitUsd) return false;
  if (m.timeInTradeMs < th.minHoldBeforeMtmDivergenceExitMs) return false;
  if (m.drawdownFromPeakPnL < th.minDrawdownForMtmDivergenceExitUsd) return false;
  if (cfg.requireRemainingHighForMtmDivergence && !div.remainingValueHighButDrawdownHigh) return false;
  return true;
}

function buildNoProgressExitSnapshot(
  m: DynamicExitComputedMetrics,
  cfg: PaperDynamicExitRuntimeConfig
): PaperNoProgressExitSnapshot {
  const th = cfg.thresholds;
  return {
    noProgressTriggered: true,
    thresholdsApplied: {
      minHoldBeforeNoProgressExitMs: th.minHoldBeforeNoProgressExitMs,
      maxFlatPnlAbsUsd: th.maxFlatPnlAbsUsd,
      maxCapturedEdgeRatioForNoProgressExit: th.maxCapturedEdgeRatioForNoProgressExit,
      maxFlatEdgeVelocityAbs: th.maxFlatEdgeVelocityAbs,
      noProgressMaxDrawdownFromPeakUsd: th.noProgressMaxDrawdownFromPeakUsd,
      skipNoProgressIfRemainingAboveUsd: th.skipNoProgressIfRemainingAboveUsd,
    },
    valuesAtDecision: {
      timeInTradeMs: m.timeInTradeMs,
      currentMarkedPnL: m.currentMarkedPnL,
      capturedEdgeRatio: m.capturedEdgeRatio,
      edgeVelocity: m.edgeVelocity,
      drawdownFromPeakPnL: m.drawdownFromPeakPnL,
      expectedRemainingEdgeValue: m.expectedRemainingEdgeValue,
    },
  };
}

function shouldExitNoProgress(
  m: DynamicExitComputedMetrics,
  cfg: PaperDynamicExitRuntimeConfig
): boolean {
  if (!cfg.noProgress) return false;
  const th = cfg.thresholds;
  if (m.expectedRemainingEdgeValue > th.skipNoProgressIfRemainingAboveUsd) return false;
  if (m.timeInTradeMs < th.minHoldBeforeNoProgressExitMs) return false;
  if (Math.abs(m.currentMarkedPnL) > th.maxFlatPnlAbsUsd) return false;
  if (m.capturedEdgeRatio > th.maxCapturedEdgeRatioForNoProgressExit) return false;
  if (Math.abs(m.edgeVelocity) > th.maxFlatEdgeVelocityAbs) return false;
  if (th.noProgressMaxDrawdownFromPeakUsd > 0 && m.drawdownFromPeakPnL > th.noProgressMaxDrawdownFromPeakUsd) {
    return false;
  }
  return true;
}

export function buildExitDecisionSnapshot(
  cause: ExitCondition,
  m: DynamicExitComputedMetrics,
  cfg: PaperDynamicExitRuntimeConfig
): PaperExitDecisionSnapshot {
  const base: PaperExitDecisionSnapshot = {
    chosenExitCause: cause,
    timeInTradeMs: m.timeInTradeMs,
    entryNetEdge: m.entryNetEdge,
    currentNetEdge: m.currentNetEdge,
    capturedEdgeRatio: m.capturedEdgeRatio,
    edgeVelocity: m.edgeVelocity,
    expectedRemainingEdgeValue: m.expectedRemainingEdgeValue,
    bestMarkedPnL: m.bestMarkedPnL,
    currentMarkedPnL: m.currentMarkedPnL,
    drawdownFromPeakPnL: m.drawdownFromPeakPnL,
    capitalEfficiencyScore: m.capitalEfficiencyScore,
    thresholdsAtDecision: thresholdsRecord(cfg),
    diagnosticInputs: m.diagnosticInputs,
  };
  if (cause === "mtm_edge_divergence_exit") {
    base.mtmEdgeDivergenceExit = buildMtmEdgeDivergenceExitSnapshot(m, cfg);
  }
  if (cause === "no_progress_exit") {
    base.noProgressExit = buildNoProgressExitSnapshot(m, cfg);
  }
  return base;
}

/**
 * Avalia causas dinâmicas por ordem de prioridade; só causas com flag activa.
 */
export function evaluateDynamicExit(
  trade: PaperTrade,
  m: DynamicExitComputedMetrics,
  cfg: PaperDynamicExitRuntimeConfig
): DynamicExitResult {
  const th = cfg.thresholds;
  const causes: Array<{ on: boolean; cause: ExitCondition; fire: boolean }> = [
    {
      on: cfg.profitGiveback,
      cause: "profit_giveback",
      fire:
        m.bestMarkedPnL >= th.minPnLToActivateTrailing &&
        m.drawdownFromPeakPnL >= th.allowedGivebackUsd,
    },
    {
      on: cfg.capturedEdge,
      cause: "edge_fully_captured",
      fire:
        m.capturedEdgeRatio >= th.capturedEdgeRatioExit &&
        m.expectedRemainingEdgeValue < th.minRemainingValueThresholdUsd,
    },
    {
      on: cfg.incrementalValue,
      cause: "incremental_value_too_low",
      fire:
        m.timeInTradeMs >= th.minHoldBeforeIncrementalExitMs &&
        m.expectedRemainingEdgeValue < th.minIncrementalEdgeUsd,
    },
    {
      on: cfg.edgeDeteriorationFast,
      cause: "edge_deteriorating_fast",
      fire:
        m.edgeVelocity < th.negativeVelocityThreshold &&
        m.entryNetEdge > 1e-9 &&
        m.currentNetEdge / m.entryNetEdge < th.deteriorationFloorRatio,
    },
    {
      on: cfg.capitalEfficiency,
      cause: "capital_efficiency_exit",
      fire:
        m.timeInTradeMs >= th.minHoldBeforeEfficiencyExitMs &&
        m.capitalEfficiencyScore < th.minCapitalEfficiencyThreshold,
    },
    {
      on: cfg.mtmEdgeDivergence,
      cause: "mtm_edge_divergence_exit",
      fire: shouldExitMtmEdgeDivergence(m, cfg),
    },
    {
      on: cfg.noProgress,
      cause: "no_progress_exit",
      fire: shouldExitNoProgress(m, cfg),
    },
    {
      on: true,
      cause: "emergency_time_stop",
      fire: m.timeInTradeMs >= th.emergencyExitTimeMs,
    },
  ];

  for (const row of causes) {
    if (!row.on) continue;
    if (row.fire) {
      return {
        shouldExit: true,
        cause: row.cause,
        decisionSnapshot: buildExitDecisionSnapshot(row.cause, m, cfg),
        computedMetrics: m,
      };
    }
  }

  return {
    shouldExit: false,
    cause: null,
    decisionSnapshot: null,
    computedMetrics: m,
  };
}
