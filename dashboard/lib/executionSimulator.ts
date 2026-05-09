/**
 * Execution Simulator — simulates trade entry and exit for paper trading.
 * No real orders. Fill probability depends on liquidity, spread, confidence, capital.
 */

import type { NormalizedPaperOpportunity, ExitCondition } from "./paperTypes";
import type { CapacityResult } from "./capitalCapacityEngine";
import { recordSimulateEntryDiagnostic } from "./paperSimulateEntryDiagnostics";

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
  /** PnL **bruto** só por movimento de preço × filledCapital; taxas não são deduzidas aqui. */
  realizedPnL: number;
  realizedReturn: number;
  maxAdverseExcursion: number;
  maxFavorableExcursion: number;
  exitCondition: ExitCondition;
}

export interface ActivePaperTradeState {
  tradeId: string;
  opportunityId: string;
  openedAt: string;
  entryEdge: number;
  /** Gross edge à entrada (mesma convenção que oportunidade / MTM). */
  entryGrossEdge: number;
  entryPriceEstimate: number;
  filledCapital: number;
  maxAdverseExcursion: number;
  maxFavorableExcursion: number;
}

export interface PaperExitPolicyOptions {
  maxHoldingTimeMs: number;
  stopLossPct: number;
  takeProfitPct: number;
  edgeNormalizationThreshold: number;
  edgeCaptureDelta: number;
  edgeDeteriorationDelta: number;
}

/** Opções para encadear saídas dinâmicas sem duplicar regras legadas. */
export interface PaperExitChainOptions {
  /** Quando true, `max_holding_time` não dispara (substituído por `emergency_time_stop` no motor dinâmico). */
  skipLegacyMaxHold?: boolean;
  /** Quando true, não aplica o ramo legado `edge_capture` (dinâmico `edge_fully_captured` assume). */
  skipLegacyEdgeCapture?: boolean;
  /** Quando true, não aplica o ramo legado `edge_deterioration` (dinâmico `edge_deteriorating_fast` assume). */
  skipLegacyEdgeDeterioration?: boolean;
}

/** Só stop-loss / take-profit (sempre antes do motor dinâmico e do resto legado). */
export function resolvePaperExitSafety(
  activeTrade: ActivePaperTradeState,
  latestOpportunity: { edge: number; confidence: number } | null,
  options: Pick<PaperExitPolicyOptions, "stopLossPct" | "takeProfitPct">
): ExitCondition | null {
  const entryPrice = activeTrade.entryPriceEstimate;
  const exitPrice = latestOpportunity ? 1 - latestOpportunity.edge : entryPrice;
  const pnlPct = (exitPrice - entryPrice) / Math.max(0.001, entryPrice);
  if (pnlPct <= -options.stopLossPct) return "stop_loss";
  if (pnlPct >= options.takeProfitPct) return "take_profit";
  return null;
}

/**
 * Prioridade: stop → take profit → deterioração/captura de edge (com latestState) → normalização → tempo máximo.
 */
export function resolvePaperExitReason(
  activeTrade: ActivePaperTradeState,
  latestOpportunity: { edge: number; confidence: number } | null,
  options: PaperExitPolicyOptions,
  chain?: PaperExitChainOptions
): ExitCondition | null {
  const now = Date.now();
  const openedAt = new Date(activeTrade.openedAt).getTime();
  const holdingMs = now - openedAt;

  const safety = resolvePaperExitSafety(activeTrade, latestOpportunity, options);
  if (safety) return safety;

  const g0 = activeTrade.entryGrossEdge;
  const cur = latestOpportunity?.edge;
  if (
    !chain?.skipLegacyEdgeDeterioration &&
    latestOpportunity != null &&
    typeof cur === "number" &&
    Number.isFinite(cur) &&
    Number.isFinite(g0)
  ) {
    if (cur < 0 || cur <= g0 - options.edgeDeteriorationDelta) return "edge_deterioration";
  }
  if (
    !chain?.skipLegacyEdgeCapture &&
    latestOpportunity != null &&
    typeof cur === "number" &&
    Number.isFinite(cur) &&
    Number.isFinite(g0)
  ) {
    if (g0 - cur >= options.edgeCaptureDelta) return "edge_capture";
  }

  if (latestOpportunity && Math.abs(latestOpportunity.edge) < options.edgeNormalizationThreshold) {
    return "edge_normalization";
  }

  if (!chain?.skipLegacyMaxHold && holdingMs >= options.maxHoldingTimeMs) return "max_holding_time";

  return null;
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

/** Fill proporcional contínuo (sem corte binário em prob). */
function deterministicFill(prob: number, requested: number): number {
  return requested * prob;
}

export function simulateEntry(
  opportunity: NormalizedPaperOpportunity,
  capacity: CapacityResult,
  portfolioAvailableCapital: number,
  options?: { requestedCapital?: number }
): SimulatedEntry {
  const now = new Date().toISOString();
  const capByLiquidity = opportunity.liquidity * 0.1;
  const requested =
    options?.requestedCapital != null && options.requestedCapital > 0
      ? Math.min(options.requestedCapital, portfolioAvailableCapital, capByLiquidity)
      : Math.min(capacity.recommendedCapital, portfolioAvailableCapital, capByLiquidity);

  const optionReq =
    options?.requestedCapital != null && options.requestedCapital > 0
      ? options.requestedCapital
      : null;

  if (requested <= 0) {
    recordSimulateEntryDiagnostic({
      opportunity,
      capacity,
      portfolioAvailableCapital,
      optionRequestedCapital: optionReq,
      finalRequestedCapital: requested,
      fillProbability: 0,
      filledCapital: 0,
    });
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

  recordSimulateEntryDiagnostic({
    opportunity,
    capacity,
    portfolioAvailableCapital,
    optionRequestedCapital: optionReq,
    finalRequestedCapital: requested,
    fillProbability: prob,
    filledCapital: filled,
  });

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

/**
 * Preço de saída (convénio: probabilidade implícita do outcome, 0–1):
 * - Com `latestOpportunity` (em runtime = `latestState` vindo do oppMap ou MTM no `paperTradeEngine`):
 *   `exitPriceEstimate = 1 - latestOpportunity.edge`. Em particular, **`edge === 0` ⇒ exit = 1**
 *   (modelo a tratar como certeza / resolução a favor do lado marcado, não “slippage” explícito).
 * - Sem `latestOpportunity` (`latestState === null`): **fallback** `exitPriceEstimate = entryPriceEstimate`
 *   (PnL de saída nulo em termos de mark; ver `recordPaperTradeLifecycleClose.exitEqualsEntryBecauseNoLatest`).
 */
export function simulateExit(
  activeTrade: ActivePaperTradeState,
  latestOpportunity: { edge: number; confidence: number } | null,
  options: PaperExitPolicyOptions,
  chain?: PaperExitChainOptions,
  forcedExitCondition?: ExitCondition | null
): SimulatedExit {
  const now = Date.now();

  const entryPrice = activeTrade.entryPriceEstimate;
  const exitPrice = latestOpportunity
    ? 1 - latestOpportunity.edge
    : entryPrice;
  const pnlPct = (exitPrice - entryPrice) / entryPrice;
  const realizedPnL = activeTrade.filledCapital * pnlPct;
  const realizedReturn = pnlPct;

  const exitCondition =
    forcedExitCondition ??
    resolvePaperExitReason(activeTrade, latestOpportunity, options, chain) ??
    "max_holding_time";

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
  options: PaperExitPolicyOptions,
  chain?: PaperExitChainOptions
): boolean {
  return resolvePaperExitReason(activeTrade, latestOpportunity, options, chain) != null;
}
