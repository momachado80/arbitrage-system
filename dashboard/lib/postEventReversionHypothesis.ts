/**
 * Post-Event Reversion Hypothesis — pure logic.
 *
 * Hipótese narrow #1 (aprovada): em mercados Polymarket "Will [team] win the [SPORT]
 * championship?" durante NBA Finals + NHL Stanley Cup Playoffs, o mid-price
 * overshoota entre PRE_EVENT_15M e POST_EVENT_15M (≈T-15 → ≈T+120) e reverte
 * parcialmente até POST_EVENT_60M (≈T+240). Quando o move ≥ 3% absoluto, fading
 * o move e segurando até POST_EVENT_60M produz reversão média esperada ≥ 0.8%.
 *
 * Este arquivo contém SOMENTE funções puras testáveis sem rede:
 *  - isHypothesisEligibleMarket: filtra alvo
 *  - computeReversionMetric: dado um EventSnapshot fechado, calcula realized reversion
 *  - judgeHypothesis: agrega N métricas e retorna verdict (alive/refinement/dead)
 *
 * Não importa de: shadowSimulationService, shadowSimulationStore, paperPortfolioStore,
 * executionDispatcher, probabilityScanner, opportunityEngine, graphScanService,
 * graphArbitrageEngine. Sem trade. Sem .paper. Sem worker.
 */

import type { NormalizedMarket } from "./polymarketClient";
import {
  evaluateMarketSuitability,
  type MarketSuitabilityVerdict,
} from "./marketSuitabilityGate";
import {
  evaluateMarketUniverseQuality,
  type UniverseQualityVerdict,
} from "./marketUniverseQuality";

export const HYPOTHESIS_VERSION = "post_event_reversion_v1";

/** Threshold absoluto do move PRE_EVENT_15M → POST_EVENT_15M para disparar o sinal. */
export const SIGNAL_THRESHOLD = 0.03;
/** Banda saudável de mid (espelha UQ): fora dela invalida amostra ou bloqueia. */
export const TAIL_LOWER = 0.06;
export const TAIL_UPPER = 0.94;
/** Spread máximo aceitável no POST_EVENT_15M para considerar o evento mensurável. */
export const MAX_POST_IMMEDIATE_SPREAD = 0.05;
/** Sample size mínimo qualificado antes de emitir veredito. */
export const MIN_QUALIFIED_N = 50;

/** Critérios de vida/morte/refino. */
export const DEATH_MEAN = 0.005;
export const TARGET_MEAN = 0.008;
export const DEATH_HIT_RATE = 0.48;
export const TARGET_HIT_RATE = 0.55;
export const DEATH_SHARPE = 0.3;
export const TARGET_SHARPE = 0.6;
export const MAX_DRAWDOWN_RATIO = 5;

const NBA_FINALS_PATTERN = /\bwin\s+the\s+(\d{4}\s+)?nba\s+finals?\b/i;
const NHL_STANLEY_CUP_PATTERN = /\bwin\s+the\s+(\d{4}\s+)?(nhl\s+)?stanley\s+cup\b/i;

/** Verdicts hard de UQ que bloqueiam elegibilidade (espelha HARD_REJECT_VERDICTS
 *  do universeQualityGate). REJECT_AMBIGUOUS é permitido (data-limited). */
const HARD_UQ_REJECTS: ReadonlySet<UniverseQualityVerdict> = new Set<UniverseQualityVerdict>([
  "REJECT_POLITICAL_LEGAL",
  "REJECT_MEME_OR_ABSURD",
  "REJECT_TAIL_OR_TICK_FENCE",
  "REJECT_LONG_HORIZON_NO_CATALYST",
  "REJECT_NOT_SUITABLE",
]);

const SUITABILITY_DATA_LIMITED: ReadonlySet<MarketSuitabilityVerdict> = new Set<MarketSuitabilityVerdict>([
  "UNSUITABLE_NO_BOOK",
  "UNSUITABLE_MISSING_DATA",
]);

export type HypothesisSport = "NBA" | "NHL";

export interface SnapshotData {
  capturedAtUtc: string;
  mid: number;
  bid: number | null;
  ask: number | null;
  spread: number;
  liquidity: number;
  volume: number;
}

export interface EventSnapshotsByWindow {
  /** Catalyst window PRE_EVENT_15M ≈ T-15 (15 min antes do início do jogo). */
  preEvent15m?: SnapshotData;
  /** Catalyst window POST_EVENT_15M ≈ T+120 (≈ fim do jogo, captura overshoot). */
  postEvent15m?: SnapshotData;
  /** Catalyst window POST_EVENT_60M ≈ T+240 (1h+ pós-fim, captura reversão). */
  postEvent60m?: SnapshotData;
}

export interface EventSnapshot {
  marketId: string;
  question: string;
  sport: HypothesisSport;
  /** ISO UTC do início do jogo (= EVENT_START do catalyst). */
  catalystEventStartUtc: string;
  snapshots: EventSnapshotsByWindow;
}

export type EligibilityResult =
  | { eligible: true; sport: HypothesisSport }
  | { eligible: false; reason: string };

export interface ReversionMetric {
  marketId: string;
  question: string;
  sport: HypothesisSport;
  signalFired: boolean;
  invalidationReason: string | null;
  midPre: number | null;
  midPostImmediate: number | null;
  midPostLate: number | null;
  move: number | null;
  signalDir: "long" | "short" | null;
  realizedReversion: number | null;
}

export type VerdictStatus =
  | "alive_collecting"
  | "alive_surviving"
  | "needs_refinement"
  | "dead";

export interface HypothesisVerdict {
  status: VerdictStatus;
  n: number;
  meanRealizedReversion: number | null;
  hitRate: number | null;
  sharpeProxy: number | null;
  maxDrawdownAbs: number | null;
  drawdownToMeanRatio: number | null;
  deathReason: string | null;
  refinementReason: string | null;
  details: {
    samplesTotal: number;
    samplesQualified: number;
    samplesInvalidated: number;
    samplesSignalBelowThreshold: number;
    samplesIncomplete: number;
  };
}

/**
 * Verifica se um mercado se encaixa na hipótese: binário, NBA Finals ou NHL
 * Stanley Cup championship, mid em safe band, suitability aceitável e UQ não
 * rejeita por motivos fortes. Data-limited suitability é tolerada (espelha
 * universeQualityGate).
 */
export function isHypothesisEligibleMarket(
  market: NormalizedMarket,
  nowIso: string,
): EligibilityResult {
  if (market.closed) return { eligible: false, reason: "market_closed" };
  if (!market.active) return { eligible: false, reason: "market_inactive" };
  if (market.outcomes.length !== 2) return { eligible: false, reason: "not_binary" };

  const isNBA = NBA_FINALS_PATTERN.test(market.question);
  const isNHL = NHL_STANLEY_CUP_PATTERN.test(market.question);
  if (!isNBA && !isNHL) return { eligible: false, reason: "not_target_sport" };

  const mid = market.prices.length > 0 ? market.prices[0]! : null;
  if (mid === null || !Number.isFinite(mid)) {
    return { eligible: false, reason: "missing_mid" };
  }
  if (mid < TAIL_LOWER || mid > TAIL_UPPER) {
    return { eligible: false, reason: "mid_in_tail" };
  }

  const suit = evaluateMarketSuitability({
    marketId: market.id,
    question: market.question,
    closed: market.closed,
    liquidity: market.liquidity,
    volume: market.volume,
    lastPrice: mid,
    nowIso,
  });
  const dataLimited = SUITABILITY_DATA_LIMITED.has(suit.suitabilityVerdict);
  if (suit.suitabilityVerdict !== "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION" && !dataLimited) {
    return { eligible: false, reason: `suitability_${suit.suitabilityVerdict}` };
  }

  const uq = evaluateMarketUniverseQuality({
    marketId: market.id,
    question: market.question,
    slug: market.slug,
    category: market.category,
    nowIso,
    closed: market.closed,
    liquidity: market.liquidity,
    volume: market.volume,
    mid,
    suitabilityVerdict: dataLimited
      ? "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION"
      : suit.suitabilityVerdict,
  });
  if (HARD_UQ_REJECTS.has(uq.universeQualityVerdict)) {
    return { eligible: false, reason: `uq_${uq.universeQualityVerdict}` };
  }

  return { eligible: true, sport: isNBA ? "NBA" : "NHL" };
}

/**
 * Dado um EventSnapshot, computa a métrica de reversão para julgamento.
 * Retorna invalidationReason quando o evento não qualifica como sample.
 *
 * Definições:
 *  - move = midPostImmediate - midPre
 *  - signal fires se |move| ≥ SIGNAL_THRESHOLD
 *  - signalDir = "short" quando move > 0 (fade up); "long" quando move < 0 (fade down)
 *  - realizedReversion = -sign(move) * (midPostLate - midPostImmediate)
 *      → positivo quando o preço reverte favorável à direção do sinal
 */
export function computeReversionMetric(event: EventSnapshot): ReversionMetric {
  const s = event.snapshots;
  const midPre = s.preEvent15m?.mid ?? null;
  const midPostImmediate = s.postEvent15m?.mid ?? null;
  const midPostLate = s.postEvent60m?.mid ?? null;

  const base: ReversionMetric = {
    marketId: event.marketId,
    question: event.question,
    sport: event.sport,
    signalFired: false,
    invalidationReason: null,
    midPre,
    midPostImmediate,
    midPostLate,
    move: null,
    signalDir: null,
    realizedReversion: null,
  };

  if (midPre === null || !Number.isFinite(midPre)) {
    return { ...base, invalidationReason: "missing_pre_event_15m" };
  }
  if (midPostImmediate === null || !Number.isFinite(midPostImmediate)) {
    return { ...base, invalidationReason: "missing_post_event_15m" };
  }
  if (midPostLate === null || !Number.isFinite(midPostLate)) {
    return { ...base, invalidationReason: "missing_post_event_60m" };
  }
  if (midPostImmediate < TAIL_LOWER || midPostImmediate > TAIL_UPPER) {
    return { ...base, invalidationReason: "post_event_15m_mid_in_tail" };
  }
  const spreadPostImmediate = s.postEvent15m?.spread ?? 0;
  if (spreadPostImmediate > MAX_POST_IMMEDIATE_SPREAD) {
    return { ...base, invalidationReason: "post_event_15m_spread_too_wide" };
  }

  const move = midPostImmediate - midPre;
  if (Math.abs(move) < SIGNAL_THRESHOLD) {
    return { ...base, move, invalidationReason: "signal_below_threshold" };
  }

  const signalDir: "long" | "short" = move > 0 ? "short" : "long";
  const realizedReversion = -Math.sign(move) * (midPostLate - midPostImmediate);
  return { ...base, signalFired: true, move, signalDir, realizedReversion };
}

/**
 * Julga a hipótese contra os critérios de vida/morte/refino.
 *  - n < MIN_QUALIFIED_N → alive_collecting (continua amostrando)
 *  - mean abaixo do DEATH, hit_rate abaixo do random, sharpe abaixo do mínimo,
 *    drawdown dominante → dead (com deathReason)
 *  - mean/hit_rate/sharpe entre DEATH e TARGET → needs_refinement
 *  - todos acima do TARGET → alive_surviving
 */
export function judgeHypothesis(metrics: ReversionMetric[]): HypothesisVerdict {
  const total = metrics.length;
  const qualified = metrics.filter(m => m.signalFired && m.realizedReversion !== null);
  const signalBelowThreshold = metrics.filter(
    m => m.invalidationReason === "signal_below_threshold",
  ).length;
  const invalidated = metrics.filter(
    m =>
      m.invalidationReason !== null && m.invalidationReason !== "signal_below_threshold",
  ).length;
  const incomplete = total - qualified.length - signalBelowThreshold - invalidated;

  const details = {
    samplesTotal: total,
    samplesQualified: qualified.length,
    samplesInvalidated: invalidated,
    samplesSignalBelowThreshold: signalBelowThreshold,
    samplesIncomplete: incomplete,
  };

  const n = qualified.length;

  if (n < MIN_QUALIFIED_N) {
    return {
      status: "alive_collecting",
      n,
      meanRealizedReversion: null,
      hitRate: null,
      sharpeProxy: null,
      maxDrawdownAbs: null,
      drawdownToMeanRatio: null,
      deathReason: null,
      refinementReason: null,
      details,
    };
  }

  const reversions = qualified.map(m => m.realizedReversion!);
  const mean = reversions.reduce((a, b) => a + b, 0) / n;
  const hits = reversions.filter(r => r > 0).length;
  const hitRate = hits / n;
  const variance = reversions.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const sharpe = std > 0 ? mean / std : 0;
  const worst = Math.min(...reversions);
  const maxDrawdownAbs = Math.max(0, -worst);
  const drawdownToMeanRatio = mean > 0 ? maxDrawdownAbs / mean : Infinity;

  if (mean < DEATH_MEAN) {
    return buildVerdict(
      "dead",
      n, mean, hitRate, sharpe, maxDrawdownAbs, drawdownToMeanRatio,
      "mean_below_death_threshold", null, details,
    );
  }
  if (hitRate < DEATH_HIT_RATE) {
    return buildVerdict(
      "dead",
      n, mean, hitRate, sharpe, maxDrawdownAbs, drawdownToMeanRatio,
      "hit_rate_below_random", null, details,
    );
  }
  if (sharpe < DEATH_SHARPE) {
    return buildVerdict(
      "dead",
      n, mean, hitRate, sharpe, maxDrawdownAbs, drawdownToMeanRatio,
      "sharpe_below_minimum", null, details,
    );
  }
  if (drawdownToMeanRatio > MAX_DRAWDOWN_RATIO) {
    return buildVerdict(
      "dead",
      n, mean, hitRate, sharpe, maxDrawdownAbs, drawdownToMeanRatio,
      "drawdown_dominates_mean", null, details,
    );
  }

  const reasons: string[] = [];
  if (mean < TARGET_MEAN) reasons.push("mean_below_target");
  if (hitRate < TARGET_HIT_RATE) reasons.push("hit_rate_below_target");
  if (sharpe < TARGET_SHARPE) reasons.push("sharpe_below_target");
  if (reasons.length > 0) {
    return buildVerdict(
      "needs_refinement",
      n, mean, hitRate, sharpe, maxDrawdownAbs, drawdownToMeanRatio,
      null, reasons.join(","), details,
    );
  }

  return buildVerdict(
    "alive_surviving",
    n, mean, hitRate, sharpe, maxDrawdownAbs, drawdownToMeanRatio,
    null, null, details,
  );
}

function buildVerdict(
  status: VerdictStatus,
  n: number,
  mean: number,
  hitRate: number,
  sharpe: number,
  maxDrawdownAbs: number,
  drawdownToMeanRatio: number,
  deathReason: string | null,
  refinementReason: string | null,
  details: HypothesisVerdict["details"],
): HypothesisVerdict {
  return {
    status,
    n,
    meanRealizedReversion: mean,
    hitRate,
    sharpeProxy: sharpe,
    maxDrawdownAbs,
    drawdownToMeanRatio,
    deathReason,
    refinementReason,
    details,
  };
}
