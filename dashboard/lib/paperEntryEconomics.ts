/**
 * Entrada paper: lucro líquido realizável esperado × factor de progresso / monetização.
 * O factor penaliza fills fracos, pouca confiança, pouco headroom de edge, gross/fees fraco
 * e (opcionalmente) histórico de `no_progress_exit` por perfil — não só PnL esperado baixo.
 */

import type { CapacityResult } from "./capitalCapacityEngine";
import type { SimulatedEntry } from "./executionSimulator";
import {
  getEffectiveMinProgressProbabilityFactor,
  getHistoricalNoProgressRate,
} from "./paperEntryProfileMemory";

/** Perfil observado em runtime com `no_progress_exit` apesar de score/progress altos — falta de “headroom” líquido vs gross. */
export const STANDARD_CROSS_MARKET_PROFILE_KEY = "standard|cross_market" as const;

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

export type PaperEntryEconomicRejectionReason =
  | "FILLED_CAPITAL_BELOW_MIN"
  | "PROGRESS_PROBABILITY_FACTOR_BELOW_MIN"
  | "ENTRY_ECONOMIC_SCORE_BELOW_MIN"
  | "EXPECTED_REALIZABLE_NET_PNL_BELOW_MIN"
  | "EXPECTED_NET_PNL_BELOW_MIN"
  | "EXPECTED_NET_MARGIN_BELOW_MIN"
  | "GROSS_TO_FEES_RATIO_BELOW_MIN"
  /** Só `standard|cross_market`: net/gross demasiado baixo (edge líquida fina vs gross). */
  | "CROSS_MARKET_NET_TO_GROSS_EDGE_BELOW_MIN";

export interface PaperEntryEconomicFilterConfig {
  enabled: boolean;
  /**
   * Expoente ≥1 em `progressProbabilityFactor`: >1 penaliza mais trades com factor baixo (ex.: ~0,11)
   * sem baixar o PnL esperado na mesma proporção — recalibra a escala do score vs runtime.
   */
  entryScoreProgressExponent: number;
  /** Guard rail: rejeita se o factor de progresso puro for demasiado baixo (evidência: ~0,11 → no_progress). */
  enableMinProgressFactorGuard: boolean;
  minProgressProbabilityFactorToOpen: number;
  /** Acima do global quando histórico do perfil mostra muitos no_progress vs take_profit. */
  enableAdaptiveProgressGuard: boolean;
  minSamplesForAdaptiveProgressGuard: number;
  adaptiveProgressGuardExtraMax: number;
  /** Limiar no score = realizableUsd × progress^exponent (escala depende do expoente). */
  minEntryEconomicScore: number;
  /** Piso duro em USD no lucro líquido esperado (ex.: 0 evita trades estruturalmente negativos). */
  minRealizableNetPnlFloorUsd: number;
  /** Se true, mantém também o corte legacy em `minExpectedNetPnlToOpenUsd`. */
  requireMinRawExpectedNetPnl: boolean;
  minExpectedNetPnlToOpenUsd: number;
  minExpectedNetProfitMargin: number;
  minFilledCapitalToOpenUsd: number;
  minGrossToFeesRatioToOpen: number;
  /** Referência para normalizar gross/fees dentro do factor de progresso. */
  grossToFeesRefForProgress: number;
  /** Peso [0,1] da taxa histórica de no_progress no perfil. */
  historicalNoProgressWeight: number;
  /** Mínimo de fechos registados por perfil antes de usar histórico. */
  minSamplesForHistoricalPrior: number;
  /** Segundo sinal para cross_market: `estimatedNetEdge / estimatedGrossEdge` (headroom líquido). */
  enableCrossMarketNetGrossEntryGuard: boolean;
  minNetToGrossEdgeRatioCrossMarket: number;
}

export function getPaperEntryEconomicFilterConfig(): PaperEntryEconomicFilterConfig {
  return {
    enabled: envBool("ENABLE_ENTRY_EXPECTED_NET_FILTER", true),
    entryScoreProgressExponent: envNum("ENTRY_SCORE_PROGRESS_EXPONENT", 1.3),
    enableMinProgressFactorGuard: envBool("ENTRY_ENABLE_MIN_PROGRESS_GUARD", true),
    minProgressProbabilityFactorToOpen: envNum("MIN_PROGRESS_PROBABILITY_FACTOR_TO_OPEN", 0.12),
    enableAdaptiveProgressGuard: envBool("ENTRY_ENABLE_ADAPTIVE_PROGRESS_GUARD", true),
    minSamplesForAdaptiveProgressGuard: Math.max(
      1,
      Math.floor(envNum("ENTRY_MIN_SAMPLES_FOR_ADAPTIVE_PROGRESS_GUARD", 8))
    ),
    adaptiveProgressGuardExtraMax: envNum("ADAPTIVE_PROGRESS_GUARD_EXTRA_MAX", 0.08),
    /** Default > faixa observada de trades estéreis (~0,0044 USD com progress~0,11 na fórmula antiga linear). */
    minEntryEconomicScore: envNum("MIN_ENTRY_ECONOMIC_SCORE", 0.006),
    minRealizableNetPnlFloorUsd: envNum("MIN_REALIZABLE_NET_PNL_FLOOR_USD", 0),
    requireMinRawExpectedNetPnl: envBool("ENTRY_REQUIRE_MIN_RAW_EXPECTED_NET_PNL", false),
    minExpectedNetPnlToOpenUsd: envNum("MIN_EXPECTED_NET_PNL_TO_OPEN_USD", 0.05),
    minExpectedNetProfitMargin: envNum("MIN_EXPECTED_NET_PROFIT_MARGIN", 0.0005),
    minFilledCapitalToOpenUsd: envNum("MIN_FILLED_CAPITAL_TO_OPEN_USD", 20),
    minGrossToFeesRatioToOpen: envNum("MIN_GROSS_TO_FEES_RATIO_TO_OPEN", 1.2),
    grossToFeesRefForProgress: envNum("ENTRY_GROSS_TO_FEES_REF_FOR_PROGRESS", 1.2),
    historicalNoProgressWeight: envNum("ENTRY_HISTORICAL_NO_PROGRESS_WEIGHT", 0.35),
    minSamplesForHistoricalPrior: Math.max(1, Math.floor(envNum("ENTRY_MIN_SAMPLES_FOR_HISTORICAL_PRIOR", 5))),
    enableCrossMarketNetGrossEntryGuard: envBool("ENTRY_CROSS_MARKET_NET_GROSS_GUARD", true),
    minNetToGrossEdgeRatioCrossMarket: envNum("MIN_NET_TO_GROSS_EDGE_RATIO_CROSS_MARKET", 0.12),
  };
}

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

/**
 * Factor explícito [0,1]: “probabilidade operacional” de progresso / monetização.
 * Combina fill×confiança, headroom de edge vs minNetEdge, qualidade gross/fees e histórico leve.
 */
/** Componentes internos do factor de progresso (só diagnóstico; não altera política). */
export function computeProgressProbabilityInternals(
  capacity: CapacityResult,
  entry: SimulatedEntry,
  grossToFeesRatio: number,
  minNetEdgeToTrade: number,
  historicalNoProgressRate: number,
  cfg: PaperEntryEconomicFilterConfig
): {
  liquiditySignal: number;
  headroomFactor: number;
  monetizationFactor: number;
  historicalNoProgressFactor: number;
} {
  const conf = clamp01(capacity.capacityConfidence);
  const fill = clamp01(entry.fillProbability);
  const liquiditySignal = Math.sqrt(Math.max(1e-12, conf * fill));

  const edge = capacity.estimatedNetEdge;
  const minE = Math.max(1e-9, minNetEdgeToTrade);
  const headroom = edge > minE ? (edge - minE) / minE : 0;
  const headroomFactor = 0.2 + 0.8 * clamp01(headroom / 2);

  const monetizationFactor = clamp01(grossToFeesRatio / Math.max(1e-9, cfg.grossToFeesRefForProgress));

  const w = clamp01(cfg.historicalNoProgressWeight);
  const hist = clamp01(historicalNoProgressRate);
  const historicalNoProgressFactor = 1 - w * hist;

  return { liquiditySignal, headroomFactor, monetizationFactor, historicalNoProgressFactor };
}

export function computeProgressProbabilityFactor(
  capacity: CapacityResult,
  entry: SimulatedEntry,
  grossToFeesRatio: number,
  minNetEdgeToTrade: number,
  historicalNoProgressRate: number,
  cfg: PaperEntryEconomicFilterConfig
): number {
  const { liquiditySignal, headroomFactor, monetizationFactor, historicalNoProgressFactor } =
    computeProgressProbabilityInternals(
      capacity,
      entry,
      grossToFeesRatio,
      minNetEdgeToTrade,
      historicalNoProgressRate,
      cfg
    );
  return liquiditySignal * headroomFactor * monetizationFactor * historicalNoProgressFactor;
}

export interface PaperEntryEconomicsMetrics {
  expectedNetPnlToOpenUsd: number;
  /** Igual a expectedNetPnlToOpenUsd (lucro líquido USD esperado à abertura, fees ida+volta explícitas). */
  expectedRealizableNetPnlUsd: number;
  expectedGrossPnlToOpenUsd: number;
  expectedFeesUsd: number;
  expectedNetProfitMargin: number;
  grossToFeesRatio: number;
  filledCapital: number;
  progressProbabilityFactor: number;
  /** Expoente aplicado em `progressProbabilityFactor` no score (diagnóstico). */
  entryScoreProgressExponent: number;
  /**
   * realizableUsd × progress^exponent — com exponent>1, combinações com progress muito baixo descem mais na escala
   * do que no produto linear (mesmo realizableUsd moderado).
   */
  entryEconomicScore: number;
  historicalNoProgressPrior: number;
  entryProfileKey: string;
  /** Baseline global; o efectivo vem do adaptativo por perfil quando activo. */
  globalMinProgressProbabilityFactorToOpen?: number;
  effectiveMinProgressProbabilityFactorToOpen?: number;
  adaptiveProgressGuardApplied?: boolean;
  /** estimatedNetEdge / estimatedGrossEdge no capacity (0–1). */
  netToGrossEdgeRatioAtEntry: number;
}

/**
 * `estimatedNetEdge` no capacity inclui modelo de fees; comparamos também custo ida+volta explícito (2×feeBuffer),
 * alinhado ao analytics.
 */
export function computePaperEntryEconomicsMetrics(
  capacity: CapacityResult,
  entry: SimulatedEntry,
  feeBuffer: number,
  minNetEdgeToTrade: number,
  profileKey: string
): PaperEntryEconomicsMetrics {
  const cfg = getPaperEntryEconomicFilterConfig();
  const filled = Math.max(0, entry.filledCapital);
  const expectedFeesUsd = filled * feeBuffer * 2;
  const edgeValueUsd = filled * capacity.estimatedNetEdge;
  const expectedNetPnlToOpenUsd = edgeValueUsd - expectedFeesUsd;
  const expectedRealizableNetPnlUsd = expectedNetPnlToOpenUsd;
  const expectedGrossPnlToOpenUsd = filled * capacity.estimatedGrossEdge;
  const expectedNetProfitMargin = filled > 1e-9 ? expectedRealizableNetPnlUsd / filled : 0;
  const grossToFeesRatio =
    expectedFeesUsd > 1e-12 ? expectedGrossPnlToOpenUsd / expectedFeesUsd : 0;

  const historicalNoProgressPrior = getHistoricalNoProgressRate(profileKey, cfg.minSamplesForHistoricalPrior);

  const progressProbabilityFactor = computeProgressProbabilityFactor(
    capacity,
    entry,
    grossToFeesRatio,
    minNetEdgeToTrade,
    historicalNoProgressPrior,
    cfg
  );

  const exp = Math.max(1, cfg.entryScoreProgressExponent);
  const entryEconomicScore =
    expectedRealizableNetPnlUsd * Math.pow(Math.max(1e-12, progressProbabilityFactor), exp);

  const gEdge = Math.max(1e-12, capacity.estimatedGrossEdge);
  const netToGrossEdgeRatioAtEntry = capacity.estimatedNetEdge / gEdge;

  return {
    expectedNetPnlToOpenUsd,
    expectedRealizableNetPnlUsd,
    expectedGrossPnlToOpenUsd,
    expectedFeesUsd,
    expectedNetProfitMargin,
    grossToFeesRatio,
    filledCapital: filled,
    progressProbabilityFactor,
    entryScoreProgressExponent: exp,
    entryEconomicScore,
    historicalNoProgressPrior,
    entryProfileKey: profileKey,
    netToGrossEdgeRatioAtEntry,
  };
}

export function evaluatePaperEntryEconomics(
  capacity: CapacityResult,
  entry: SimulatedEntry,
  feeBuffer: number,
  minNetEdgeToTrade: number,
  profileKey: string,
  cfg: PaperEntryEconomicFilterConfig
):
  | { ok: true; metrics: PaperEntryEconomicsMetrics }
  | { ok: false; reason: PaperEntryEconomicRejectionReason; metrics: PaperEntryEconomicsMetrics } {
  const base = computePaperEntryEconomicsMetrics(capacity, entry, feeBuffer, minNetEdgeToTrade, profileKey);

  const guardRow = getEffectiveMinProgressProbabilityFactor(profileKey, cfg.minProgressProbabilityFactorToOpen, {
    enableAdaptive: cfg.enableAdaptiveProgressGuard,
    minSamples: cfg.minSamplesForAdaptiveProgressGuard,
    extraMax: cfg.adaptiveProgressGuardExtraMax,
  });

  const metrics: PaperEntryEconomicsMetrics = {
    ...base,
    globalMinProgressProbabilityFactorToOpen: cfg.minProgressProbabilityFactorToOpen,
    effectiveMinProgressProbabilityFactorToOpen: guardRow.effectiveMin,
    adaptiveProgressGuardApplied: guardRow.adaptiveApplied,
  };

  const filled = metrics.filledCapital;

  if (filled < cfg.minFilledCapitalToOpenUsd) {
    return { ok: false, reason: "FILLED_CAPITAL_BELOW_MIN", metrics };
  }
  if (metrics.expectedRealizableNetPnlUsd < cfg.minRealizableNetPnlFloorUsd) {
    return { ok: false, reason: "EXPECTED_REALIZABLE_NET_PNL_BELOW_MIN", metrics };
  }
  const minProgressCut = metrics.effectiveMinProgressProbabilityFactorToOpen ?? cfg.minProgressProbabilityFactorToOpen;
  if (cfg.enableMinProgressFactorGuard && metrics.progressProbabilityFactor < minProgressCut) {
    return { ok: false, reason: "PROGRESS_PROBABILITY_FACTOR_BELOW_MIN", metrics };
  }
  if (metrics.entryEconomicScore < cfg.minEntryEconomicScore) {
    return { ok: false, reason: "ENTRY_ECONOMIC_SCORE_BELOW_MIN", metrics };
  }
  if (
    profileKey === STANDARD_CROSS_MARKET_PROFILE_KEY &&
    cfg.enableCrossMarketNetGrossEntryGuard &&
    metrics.netToGrossEdgeRatioAtEntry < cfg.minNetToGrossEdgeRatioCrossMarket
  ) {
    return { ok: false, reason: "CROSS_MARKET_NET_TO_GROSS_EDGE_BELOW_MIN", metrics };
  }
  if (cfg.requireMinRawExpectedNetPnl && metrics.expectedNetPnlToOpenUsd < cfg.minExpectedNetPnlToOpenUsd) {
    return { ok: false, reason: "EXPECTED_NET_PNL_BELOW_MIN", metrics };
  }
  if (metrics.expectedNetProfitMargin < cfg.minExpectedNetProfitMargin) {
    return { ok: false, reason: "EXPECTED_NET_MARGIN_BELOW_MIN", metrics };
  }
  if (metrics.grossToFeesRatio < cfg.minGrossToFeesRatioToOpen) {
    return { ok: false, reason: "GROSS_TO_FEES_RATIO_BELOW_MIN", metrics };
  }
  return { ok: true, metrics };
}

/**
 * Lista todas as regras económicas que falhariam com as mesmas métricas (ordem igual a `evaluatePaperEntryEconomics`).
 * Não altera decisão do motor — só observabilidade.
 */
export function collectAllPaperEntryEconomicsFailures(
  metrics: PaperEntryEconomicsMetrics,
  profileKey: string,
  cfg: PaperEntryEconomicFilterConfig
): PaperEntryEconomicRejectionReason[] {
  const reasons: PaperEntryEconomicRejectionReason[] = [];
  const filled = metrics.filledCapital;
  if (filled < cfg.minFilledCapitalToOpenUsd) reasons.push("FILLED_CAPITAL_BELOW_MIN");
  if (metrics.expectedRealizableNetPnlUsd < cfg.minRealizableNetPnlFloorUsd) {
    reasons.push("EXPECTED_REALIZABLE_NET_PNL_BELOW_MIN");
  }
  const minProgressCut = metrics.effectiveMinProgressProbabilityFactorToOpen ?? cfg.minProgressProbabilityFactorToOpen;
  if (cfg.enableMinProgressFactorGuard && metrics.progressProbabilityFactor < minProgressCut) {
    reasons.push("PROGRESS_PROBABILITY_FACTOR_BELOW_MIN");
  }
  if (metrics.entryEconomicScore < cfg.minEntryEconomicScore) reasons.push("ENTRY_ECONOMIC_SCORE_BELOW_MIN");
  if (
    profileKey === STANDARD_CROSS_MARKET_PROFILE_KEY &&
    cfg.enableCrossMarketNetGrossEntryGuard &&
    metrics.netToGrossEdgeRatioAtEntry < cfg.minNetToGrossEdgeRatioCrossMarket
  ) {
    reasons.push("CROSS_MARKET_NET_TO_GROSS_EDGE_BELOW_MIN");
  }
  if (cfg.requireMinRawExpectedNetPnl && metrics.expectedNetPnlToOpenUsd < cfg.minExpectedNetPnlToOpenUsd) {
    reasons.push("EXPECTED_NET_PNL_BELOW_MIN");
  }
  if (metrics.expectedNetProfitMargin < cfg.minExpectedNetProfitMargin) reasons.push("EXPECTED_NET_MARGIN_BELOW_MIN");
  if (metrics.grossToFeesRatio < cfg.minGrossToFeesRatioToOpen) reasons.push("GROSS_TO_FEES_RATIO_BELOW_MIN");
  return reasons;
}
