/**
 * Semântica gross vs net do paper PnL.
 *
 * Trades fechadas **após** a introdução dos campos explícitos:
 * - `grossRealizedPnL` = saída do `simulateExit` (variação de preço × filledCapital).
 * - `estimatedEntryFees` / `estimatedExitFees` = filledCapital × feeBuffer por perna (alinhado a paperEntryEconomics).
 * - `netRealizedPnL` = gross − estimatedTotalFees.
 * - `realizedPnL` espelha `netRealizedPnL` (KPI principal); `realizedReturn` = net / filledCapital.
 *
 * **Legado** (sem `grossRealizedPnL`): `realizedPnL` era só bruto; o líquido infere-se com `feeBufferPerLeg × 2 × filledCapital`.
 */

import type { PaperTrade } from "./paperTypes";

/** Igual ao default de `DEFAULT_PAPER_POLICY.feeBuffer` — só para inferência em trades legados. */
export const DEFAULT_PAPER_FEE_BUFFER_PER_LEG = 0.002;

/** Evita NaN/negativos propagarem PnL/taxas em analytics e APIs. */
export function safeFeeBufferPerLeg(
  v: number | undefined | null,
  fallback: number = DEFAULT_PAPER_FEE_BUFFER_PER_LEG
): number {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : fallback;
}

/** Há sinal PnL persistido finito (bruto, líquido explícito, ou legado `realizedPnL`). */
export function hasClosedPaperTradeFinitePnlSignal(t: PaperTrade): boolean {
  if (t.status !== "closed") return false;
  return (
    (typeof t.netRealizedPnL === "number" && Number.isFinite(t.netRealizedPnL)) ||
    (typeof t.grossRealizedPnL === "number" && Number.isFinite(t.grossRealizedPnL)) ||
    (typeof t.realizedPnL === "number" && Number.isFinite(t.realizedPnL))
  );
}

function safeFilled(t: PaperTrade): number {
  const f = t.filledCapital;
  return typeof f === "number" && Number.isFinite(f) && f > 0 ? f : 0;
}

/** Bruto: campo explícito ou, em legado, `realizedPnL` assumido bruto. */
export function getClosedTradeGrossRealizedPnL(t: PaperTrade): number {
  if (typeof t.grossRealizedPnL === "number" && Number.isFinite(t.grossRealizedPnL)) {
    return t.grossRealizedPnL;
  }
  if (t.status === "closed" && typeof t.realizedPnL === "number" && Number.isFinite(t.realizedPnL)) {
    return t.realizedPnL;
  }
  return 0;
}

/** Taxas persistidas ou modelo ida+volta alinhado à entrada económica. */
export function getClosedTradeEstimatedTotalFees(
  t: PaperTrade,
  feeBufferPerLeg: number = DEFAULT_PAPER_FEE_BUFFER_PER_LEG
): number {
  const buf = safeFeeBufferPerLeg(feeBufferPerLeg);
  if (typeof t.estimatedTotalFees === "number" && Number.isFinite(t.estimatedTotalFees)) {
    return Math.max(0, t.estimatedTotalFees);
  }
  const fc = safeFilled(t);
  if (fc <= 0) return 0;
  return fc * buf * 2;
}

/** Líquido: campo explícito ou gross inferido − taxas inferidas. */
export function getClosedTradeNetRealizedPnL(
  t: PaperTrade,
  feeBufferPerLeg: number = DEFAULT_PAPER_FEE_BUFFER_PER_LEG
): number {
  const buf = safeFeeBufferPerLeg(feeBufferPerLeg);
  if (typeof t.netRealizedPnL === "number" && Number.isFinite(t.netRealizedPnL)) {
    return t.netRealizedPnL;
  }
  const gross = getClosedTradeGrossRealizedPnL(t);
  const fees = getClosedTradeEstimatedTotalFees(t, buf);
  const n = gross - fees;
  return Number.isFinite(n) ? n : 0;
}

/**
 * Líquido para agregados onde trade sem sinal PnL finito conta 0 (ex.: auditoria quality sobre graph+fechado).
 */
export function getClosedTradeNetRealizedPnLOrZero(
  t: PaperTrade,
  feeBufferPerLeg: number = DEFAULT_PAPER_FEE_BUFFER_PER_LEG
): number {
  if (!hasClosedPaperTradeFinitePnlSignal(t)) return 0;
  return getClosedTradeNetRealizedPnL(t, feeBufferPerLeg);
}
