/**
 * Feature flags e limiares do motor de saída dinâmica (paper).
 * Desligado por defeito; sub-flags só aplicam com o master ligado.
 *
 * Defaults numéricos: conservadores para a escala típica de PnL/capital preenchido do paper
 * (ordens de frações de dólar a poucos USD). Recalibração futura deve partir de analytics
 * observados (ex. distribuição de realizedPnL, holding, exitDecisionSnapshot), não de suposições.
 */

function envBool(name: string, defaultValue: boolean): boolean {
  const raw = process.env[name]?.trim().toLowerCase();
  if (!raw) return defaultValue;
  return raw === "1" || raw === "true" || raw === "yes";
}

function envNum(name: string, defaultValue: number): number {
  const raw = process.env[name]?.trim();
  if (!raw) return defaultValue;
  const n = Number(raw);
  return Number.isFinite(n) ? n : defaultValue;
}

export interface PaperDynamicExitThresholds {
  minPnLToActivateTrailing: number;
  allowedGivebackUsd: number;
  minIncrementalEdgeUsd: number;
  capturedEdgeRatioExit: number;
  minRemainingValueThresholdUsd: number;
  negativeVelocityThreshold: number;
  deteriorationFloorRatio: number;
  minCapitalEfficiencyThreshold: number;
  minHoldBeforeEfficiencyExitMs: number;
  /** Tempo mínimo em trade antes de permitir `incremental_value_too_low` apenas. */
  minHoldBeforeIncrementalExitMs: number;
  emergencyExitTimeMs: number;
  minHoldBeforeMtmDivergenceExitMs: number;
  /** Perda MTM mínima (valor absoluto USD) para permitir saída por divergência. */
  minNegativePnlForMtmDivergenceExitUsd: number;
  minDrawdownForMtmDivergenceExitUsd: number;
  minHoldBeforeNoProgressExitMs: number;
  maxFlatPnlAbsUsd: number;
  maxCapturedEdgeRatioForNoProgressExit: number;
  maxFlatEdgeVelocityAbs: number;
  /** 0 = ignorar; senão exige drawdownFromPeakPnL <= este valor (trade “plano” vs pico). */
  noProgressMaxDrawdownFromPeakUsd: number;
  /** Se expectedRemainingEdgeValue > este valor, não dispara no-progress (hipótese ainda com “carne”). */
  skipNoProgressIfRemainingAboveUsd: number;
}

export interface PaperDynamicExitFlags {
  engine: boolean;
  profitGiveback: boolean;
  incrementalValue: boolean;
  capturedEdge: boolean;
  edgeDeteriorationFast: boolean;
  capitalEfficiency: boolean;
  /** Saída quando edge net > 0 mas PnL MTM negativo persistente (guard rails). */
  mtmEdgeDivergence: boolean;
  /** Se true, exige também `remainingValueHighButDrawdownHigh` para disparar divergência MTM. */
  requireRemainingHighForMtmDivergence: boolean;
  /** Encerra posições flat / sem captura nem velocidade após tempo mínimo. */
  noProgress: boolean;
}

export interface PaperDynamicExitRuntimeConfig extends PaperDynamicExitFlags {
  thresholds: PaperDynamicExitThresholds;
}

export function getPaperDynamicExitConfig(_maxHoldingTimeMsFallback: number): PaperDynamicExitRuntimeConfig {
  const engine = envBool("ENABLE_DYNAMIC_EXIT_ENGINE", false);
  // Limiares por defeito alinhados à escala económica actual do paper; sobrescrever via env.
  const thresholds: PaperDynamicExitThresholds = {
    minPnLToActivateTrailing: envNum("MIN_PNL_TO_ACTIVATE_TRAILING", 0.1),
    allowedGivebackUsd: envNum("ALLOWED_GIVEBACK_USD", 0.04),
    minIncrementalEdgeUsd: envNum("MIN_INCREMENTAL_EDGE_USD", 0.02),
    capturedEdgeRatioExit: envNum("CAPTURED_EDGE_RATIO_EXIT", 0.7),
    minRemainingValueThresholdUsd: envNum("MIN_REMAINING_VALUE_THRESHOLD_USD", 0.03),
    negativeVelocityThreshold: envNum("NEGATIVE_VELOCITY_THRESHOLD", -0.00003),
    deteriorationFloorRatio: envNum("DETERIORATION_FLOOR_RATIO", 0.45),
    minCapitalEfficiencyThreshold: envNum("MIN_CAPITAL_EFFICIENCY_THRESHOLD", 0.0008),
    minHoldBeforeEfficiencyExitMs: envNum("MIN_HOLD_BEFORE_EFFICIENCY_EXIT_MS", 30_000),
    minHoldBeforeIncrementalExitMs: envNum("MIN_HOLD_BEFORE_INCREMENTAL_EXIT_MS", 30_000),
    emergencyExitTimeMs: envNum("EMERGENCY_EXIT_TIME_MS", 240_000),
    minHoldBeforeMtmDivergenceExitMs: envNum("MIN_HOLD_BEFORE_MTM_DIVERGENCE_EXIT_MS", 45_000),
    minNegativePnlForMtmDivergenceExitUsd: envNum("MIN_NEGATIVE_PNL_FOR_MTM_DIVERGENCE_EXIT_USD", 0.02),
    minDrawdownForMtmDivergenceExitUsd: envNum("MIN_DRAWDOWN_FOR_MTM_DIVERGENCE_EXIT_USD", 0.02),
    minHoldBeforeNoProgressExitMs: envNum("MIN_HOLD_BEFORE_NO_PROGRESS_EXIT_MS", 60_000),
    maxFlatPnlAbsUsd: envNum("MAX_FLAT_PNL_ABS_USD", 0.03),
    maxCapturedEdgeRatioForNoProgressExit: envNum("MAX_CAPTURED_EDGE_RATIO_FOR_NO_PROGRESS_EXIT", 0.05),
    maxFlatEdgeVelocityAbs: envNum("MAX_FLAT_EDGE_VELOCITY_ABS", 0.000005),
    noProgressMaxDrawdownFromPeakUsd: envNum("NO_PROGRESS_MAX_DRAWDOWN_FROM_PEAK_USD", 0),
    /**
     * Se `expectedRemainingEdgeValue` (USD) continuar acima disto, não dispara `no_progress_exit`
     * (hipótese ainda com valor remanescente material). Default baixo vs 1e12: evita fechar como “flat”
     * quando o motor ainda reporta vários USD de edge remanescente.
     */
    skipNoProgressIfRemainingAboveUsd: envNum("SKIP_NO_PROGRESS_IF_REMAINING_ABOVE_USD", 0.25),
  };

  if (!engine) {
    return {
      engine: false,
      profitGiveback: false,
      incrementalValue: false,
      capturedEdge: false,
      edgeDeteriorationFast: false,
      capitalEfficiency: false,
      mtmEdgeDivergence: false,
      requireRemainingHighForMtmDivergence: false,
      noProgress: false,
      thresholds,
    };
  }

  return {
    engine: true,
    profitGiveback: envBool("ENABLE_EXIT_PROFIT_GIVEBACK", true),
    incrementalValue: envBool("ENABLE_EXIT_INCREMENTAL_VALUE", true),
    capturedEdge: envBool("ENABLE_EXIT_CAPTURED_EDGE", false),
    edgeDeteriorationFast: envBool("ENABLE_EXIT_EDGE_DETERIORATION", false),
    capitalEfficiency: envBool("ENABLE_EXIT_CAPITAL_EFFICIENCY", false),
    mtmEdgeDivergence: envBool("ENABLE_EXIT_MTM_EDGE_DIVERGENCE", false),
    requireRemainingHighForMtmDivergence: envBool("REQUIRE_REMAINING_HIGH_FOR_MTM_DIVERGENCE", false),
    noProgress: envBool("ENABLE_EXIT_NO_PROGRESS", false),
    thresholds,
  };
}
