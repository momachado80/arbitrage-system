/**
 * Plan-time market quality gate para o pipeline de Post-Event Reversion.
 *
 * Razão de ser: a auditoria da Fase 1.B mostrou que a maioria das observações
 * coletadas era invalidada downstream por (a) spread pós-evento > 0.05 (média
 * 0.54, 5.8×–15× o threshold da hipótese) e (b) POST_EVENT_60M perdido por mid
 * drifta para fora da safe band [0.06, 0.94] ou por resolução do mercado. As
 * duas falhas têm causa comum: o universo selecionado em plan-time já contém
 * mercados de cauda com spread crônico e mid próximo de 0/1.
 *
 * Este gate adiciona uma filtragem ANTES da construção do plano, usando o
 * snapshot de book já buscado por probeBook em `buildCatalystObservationPlan`.
 * Zero custo adicional de rede.
 *
 * Não toca em SIGNAL_THRESHOLD, MAX_POST_IMMEDIATE_SPREAD, TAIL_LOWER nem
 * TAIL_UPPER (thresholds da hipótese, em lib/postEventReversionHypothesis.ts).
 * Não toca no scout, no schema do ledger, nem na lógica de julgamento.
 *
 * Funções puras, sem rede, sem I/O, sem .paper, sem execução, sem microcapital.
 */

/** Spread máximo aceitável em plan-time. Mais apertado que o threshold de
 *  julgamento da hipótese (MAX_POST_IMMEDIATE_SPREAD=0.05) para deixar headroom
 *  para alargamento natural pós-evento. */
export const PLAN_TIME_MAX_SPREAD = 0.04;

/** Piso de mid em plan-time. Mais conservador que TAIL_LOWER=0.06 da hipótese
 *  para excluir mercados de cauda propensos a drift→0 (time eliminado). */
export const PLAN_TIME_MID_FLOOR = 0.10;

/** Teto de mid em plan-time. Simétrico ao piso. Mais conservador que
 *  TAIL_UPPER=0.94 da hipótese para excluir favoritos virtualmente garantidos. */
export const PLAN_TIME_MID_CEILING = 0.90;

export interface PlanTimeMarketGateInput {
  bestBid: number | null;
  bestAsk: number | null;
}

export interface PlanTimeMarketGateConfig {
  maxSpread: number;
  midFloor: number;
  midCeiling: number;
}

export type PlanTimeMarketGateReason =
  | "passed"
  | "missing_book_prices"
  | "inverted_book"
  | "spread_above_plan_max"
  | "mid_below_plan_floor"
  | "mid_above_plan_ceiling";

export interface PlanTimeMarketGateResult {
  accepted: boolean;
  reason: PlanTimeMarketGateReason;
  probedSpread: number | null;
  probedMid: number | null;
}

const DEFAULT_CONFIG: PlanTimeMarketGateConfig = {
  maxSpread: PLAN_TIME_MAX_SPREAD,
  midFloor: PLAN_TIME_MID_FLOOR,
  midCeiling: PLAN_TIME_MID_CEILING,
};

/**
 * Avalia se um mercado passa pelo gate plan-time. Função pura.
 *
 * Boundary: aceita inclusivamente nos limites. `spread <= maxSpread` passa,
 * `spread > maxSpread` rejeita. `mid >= midFloor && mid <= midCeiling` passa.
 */
export function evaluatePlanTimeMarketGate(
  input: PlanTimeMarketGateInput,
  config: PlanTimeMarketGateConfig = DEFAULT_CONFIG,
): PlanTimeMarketGateResult {
  const { bestBid, bestAsk } = input;

  if (
    bestBid === null ||
    bestAsk === null ||
    !Number.isFinite(bestBid) ||
    !Number.isFinite(bestAsk)
  ) {
    return {
      accepted: false,
      reason: "missing_book_prices",
      probedSpread: null,
      probedMid: null,
    };
  }

  if (bestAsk < bestBid) {
    return {
      accepted: false,
      reason: "inverted_book",
      probedSpread: null,
      probedMid: null,
    };
  }

  const probedSpread = bestAsk - bestBid;
  const probedMid = (bestBid + bestAsk) / 2;

  if (probedSpread > config.maxSpread) {
    return {
      accepted: false,
      reason: "spread_above_plan_max",
      probedSpread,
      probedMid,
    };
  }

  if (probedMid < config.midFloor) {
    return {
      accepted: false,
      reason: "mid_below_plan_floor",
      probedSpread,
      probedMid,
    };
  }

  if (probedMid > config.midCeiling) {
    return {
      accepted: false,
      reason: "mid_above_plan_ceiling",
      probedSpread,
      probedMid,
    };
  }

  return {
    accepted: true,
    reason: "passed",
    probedSpread,
    probedMid,
  };
}
