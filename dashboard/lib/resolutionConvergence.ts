/**
 * Resolution Convergence — scorer puro (filão #4).
 *
 * AVISO DE HONESTIDADE: convergência NÃO é arbitragem mecânica. Um mercado perto
 * da resolução negociando longe de {0,1} pode refletir risco real residual (o lado
 * favorito ainda pode virar) ou iliquidez — não necessariamente mispricing. Saber
 * se o desconto excede a probabilidade VERDADEIRA de virada é um julgamento de
 * fair-value (informacional), que este censo NÃO faz. Aqui apenas:
 *   - surfamos candidatos: mid perto de extremo + perto da resolução
 *   - medimos o desconto e o yield de convergência líquido de custos MECÂNICOS
 *   - marcamos verdict "convergence_candidate_needs_fair_value" — nunca "viable"
 * O campo `anchorable` sinaliza o subconjunto (crypto_feed) onde um spot externo
 * PODERIA fundamentar o julgamento de fair-value depois.
 *
 * Função pura, sem rede, sem I/O, sem execução, sem .paper, sem microcapital.
 */

/** Mid do lado favorito ≥ este (ou ≤ 1 − este) para contar como "perto de extremo". */
export const CONV_NEAR_EXTREME_MID = 0.92;
/** Dias até a resolução ≤ este para contar como "perto da resolução". */
export const CONV_NEAR_RESOLUTION_DAYS = 7;

export type ConvergenceVerdict =
  | "invalid_input"
  | "not_near_extreme"
  | "not_near_resolution"
  | "costs_exceed_discount"
  | "convergence_candidate_needs_fair_value";

export interface ConvergenceCostModel {
  costOfCapitalAnnual: number;
  gasPerTxUsd: number;
  targetSizeUsd: number;
  umaHaircutByCategory: Record<string, number>;
}

export interface ConvergenceScoreInput {
  /** Mid do YES (usado para detectar lado favorito e proximidade de extremo). */
  yesMid: number;
  /** Melhor ask do YES e do NO (preço para comprar o lado favorito). */
  yesAsk: number;
  noAsk: number;
  daysToResolution: number;
  category: string;
}

export interface ConvergenceScore {
  favoredSide: "YES" | "NO" | null;
  favoredAsk: number | null;
  /** Retorno bruto se resolver no lado favorito: 1 − favoredAsk (NÃO risk-free). */
  discount: number;
  /** Yield anualizado do desconto (apenas descritivo; ignora risco de virada). */
  convergenceYieldAnnualized: number;
  /** Desconto menos custos MECÂNICOS (gas, lockup, UMA). NÃO ajustado a risco. */
  netDiscountAfterCosts: number;
  daysToResolution: number;
  /** crypto_feed → desconto poderia ser fundamentado por spot externo depois. */
  anchorable: boolean;
  verdict: ConvergenceVerdict;
  note: string;
}

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function invalid(note: string): ConvergenceScore {
  return {
    favoredSide: null,
    favoredAsk: null,
    discount: 0,
    convergenceYieldAnnualized: 0,
    netDiscountAfterCosts: 0,
    daysToResolution: 0,
    anchorable: false,
    verdict: "invalid_input",
    note,
  };
}

export function scoreConvergenceCandidate(
  input: ConvergenceScoreInput,
  costs: ConvergenceCostModel,
): ConvergenceScore {
  const { yesMid, yesAsk, noAsk } = input;
  if (![yesMid, yesAsk, noAsk].every(x => Number.isFinite(x) && x > 0 && x < 1)) {
    return invalid("invalid_price_inputs");
  }
  const days = Math.max(0, input.daysToResolution);
  const anchorable = input.category === "crypto_feed";

  /** Lado favorito: YES perto de 1, ou NO perto de 1 (YES perto de 0). */
  let favoredSide: "YES" | "NO" | null = null;
  let favoredAsk = 0;
  if (yesMid >= CONV_NEAR_EXTREME_MID) {
    favoredSide = "YES";
    favoredAsk = yesAsk;
  } else if (yesMid <= 1 - CONV_NEAR_EXTREME_MID) {
    favoredSide = "NO";
    favoredAsk = noAsk;
  }

  if (favoredSide === null) {
    return {
      favoredSide: null,
      favoredAsk: null,
      discount: 0,
      convergenceYieldAnnualized: 0,
      netDiscountAfterCosts: 0,
      daysToResolution: days,
      anchorable,
      verdict: "not_near_extreme",
      note: `yes_mid=${r6(yesMid)}_not_near_0_or_1`,
    };
  }

  const discount = r6(1 - favoredAsk);
  const convergenceYieldAnnualized =
    favoredAsk > 0 && days > 0
      ? r6((discount / favoredAsk) * (365 / days))
      : favoredAsk > 0
        ? r6(discount / favoredAsk)
        : 0;

  const cGas = r6(costs.gasPerTxUsd / costs.targetSizeUsd);
  const cLockup = r6(costs.costOfCapitalAnnual * (days / 365) * favoredAsk);
  const cUma = r6(
    costs.umaHaircutByCategory[input.category] ?? costs.umaHaircutByCategory.unknown ?? 0.01,
  );
  const netDiscountAfterCosts = r6(discount - cGas - cLockup - cUma);

  let verdict: ConvergenceVerdict;
  if (days > CONV_NEAR_RESOLUTION_DAYS) {
    verdict = "not_near_resolution";
  } else if (netDiscountAfterCosts <= 0) {
    verdict = "costs_exceed_discount";
  } else {
    verdict = "convergence_candidate_needs_fair_value";
  }

  const note = [
    `favored=${favoredSide}`,
    `favored_ask=${r6(favoredAsk)}`,
    `discount=${discount}`,
    `yield_ann=${convergenceYieldAnnualized}`,
    `days=${days}`,
    `net_after_mech_costs=${netDiscountAfterCosts}`,
    `anchorable=${anchorable}`,
  ].join("|");

  return {
    favoredSide,
    favoredAsk: r6(favoredAsk),
    discount,
    convergenceYieldAnnualized,
    netDiscountAfterCosts,
    daysToResolution: days,
    anchorable,
    verdict,
    note,
  };
}
