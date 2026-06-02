/**
 * Mechanical Edge Census (MEC) v1 — lógica pura de avaliação de ineficiência mecânica.
 *
 * Diferente da Hipótese #1 (alfa estatístico, arquivada por throughput), o MEC mede
 * incoerência de preço verificável numa única foto do livro: cestas cujo custo de
 * montagem é menor que o payout garantido. Cada avaliação é auto-falsificável — ou o
 * gap sobrevive aos 6 custos reais, ou não sobrevive.
 *
 * Este arquivo contém SOMENTE a função pura `evaluateMecBasket` (sem rede, sem I/O).
 * O fetch de livro (clobMicrostructure), o runner e o teste de persistência (Tier 3)
 * ficam fora — esta lib é o núcleo testável de custo, espelhando o padrão de
 * `finalNegativeRiskValidation31552.ts` mas generalizado e sem fudge factor.
 *
 * Não importa de: executionDispatcher, paperPortfolioStore, shadowSimulationStore,
 * opportunityEngine, probabilityScanner, graphArbitrageEngine. Sem trade. Sem .paper.
 * Sem microcapital. Sem execução real. Read-only census.
 */

export const MEC_VERSION = "mechanical_edge_census_v1";

/** Bandas de verdict de snapshot (item 4 da calibração: faixa 0.5%–1%). */
export const MEC_VIABLE_NET = 0.01; // ≥ 1.0% → viable_pending_persistence
export const MEC_MARGINAL_NET = 0.005; // 0.5%–1.0% → marginal_pending_persistence

export type MecEdgeType =
  | "BINARY_UNDERROUND"
  | "BINARY_OVERROUND"
  | "PARTITION_UNDERROUND"
  | "NEGRISK_CONVERSION";

/**
 * Verdict de UMA SNAPSHOT. NÃO afirma "viable_candidate" — isso exige o teste de
 * persistência (Tier 3, runtime, multi-snapshot). A função pura só promove até
 * "*_pending_persistence", que o runner eleva depois de confirmar persistência.
 */
export type MecSnapshotVerdict =
  | "negative_after_costs"
  | "not_viable"
  | "capacity_insufficient"
  | "marginal_pending_persistence"
  | "viable_pending_persistence"
  | "invalid_input";

export interface MecLeg {
  marketId: string;
  /** "buy" para underround/partition/negrisk; "sell" para overround. */
  side: "buy" | "sell";
  /** VWAP para preencher o tamanho-alvo (ask p/ buy, bid p/ sell). */
  vwapPrice: number;
  /** Melhor nível do lado relevante (p/ diagnóstico de slippage de profundidade). */
  bestPrice: number;
  /** Profundidade (shares) disponível no lado relevante (top 3 níveis). */
  depthTop3: number;
  /** Spread do livro (ask − bid). */
  spread: number;
}

export interface MecCostModel {
  /** Custo de oportunidade do capital, anual (ex. 0.10). */
  costOfCapitalAnnual: number;
  /** Gas por transação on-chain, em USD (Polygon ~baixo). */
  gasPerTxUsd: number;
  /** Tamanho-alvo do trade, em USD (define units e checagem de capacidade). */
  targetSizeUsd: number;
  /** Haircut UMA por categoria de resolução (risco de disputa/ambiguidade). */
  umaHaircutByCategory: Record<string, number>;
  /** Coeficiente do buffer de leg-risk (escala com k e spread). */
  legRiskCoeff: number;
}

export interface MecBasketInput {
  legs: MecLeg[];
  edgeType: MecEdgeType;
  /** Dias até a resolução (de endDate da Gamma). Negativo é tratado como 0. */
  daysToResolution: number;
  /** Categoria de resolução: crypto_feed | sports | macro_data | electoral | subjective | unknown. */
  category: string;
  /** Fração de fee de conversão (apenas NEGRISK_CONVERSION). Default 0. */
  conversionFeeFrac?: number;
}

export interface MecEvaluation {
  mecVersion: typeof MEC_VERSION;
  edgeType: MecEdgeType;
  k: number;
  grossEdge: number;
  capitalPerUnit: number;
  costs: {
    cSlippageDepth: number;
    cGas: number;
    cLockup: number;
    cUma: number;
    cConversion: number;
    cLegRisk: number;
  };
  netEdge: number;
  unitsTargeted: number;
  capacityShares: number;
  capacityOk: boolean;
  daysToResolution: number;
  verdict: MecSnapshotVerdict;
  /** Guard de governança: este censo nunca autoriza execução. */
  canUseForExecution: false;
  note: string;
}

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function validLeg(l: MecLeg): boolean {
  return (
    Number.isFinite(l.vwapPrice) &&
    l.vwapPrice > 0 &&
    l.vwapPrice < 1 &&
    Number.isFinite(l.bestPrice) &&
    Number.isFinite(l.depthTop3) &&
    l.depthTop3 >= 0 &&
    Number.isFinite(l.spread) &&
    l.spread >= 0
  );
}

function invalid(edgeType: MecEdgeType, k: number, note: string): MecEvaluation {
  return {
    mecVersion: MEC_VERSION,
    edgeType,
    k,
    grossEdge: 0,
    capitalPerUnit: 0,
    costs: { cSlippageDepth: 0, cGas: 0, cLockup: 0, cUma: 0, cConversion: 0, cLegRisk: 0 },
    netEdge: 0,
    unitsTargeted: 0,
    capacityShares: 0,
    capacityOk: false,
    daysToResolution: 0,
    verdict: "invalid_input",
    canUseForExecution: false,
    note,
  };
}

/**
 * Avalia uma cesta mecânica num único snapshot. Função pura.
 *
 * gross (buy: underround/partition/negrisk) = max(0, 1 − Σ vwapAsk_i)
 * gross (sell: overround)                   = max(0, Σ vwapBid_i − 1)
 *
 * net = gross − cGas − cLockup − cUma − cConversion − cLegRisk
 * (o slippage de profundidade já está embutido no gross via VWAP; cSlippageDepth é
 *  registrado à parte só para diagnóstico, comparando VWAP vs best level.)
 */
export function evaluateMecBasket(input: MecBasketInput, costs: MecCostModel): MecEvaluation {
  const { legs, edgeType } = input;
  const k = legs.length;

  if (k < 2) return invalid(edgeType, k, "insufficient_legs_need_ge_2");
  if (!legs.every(validLeg)) return invalid(edgeType, k, "invalid_leg_price_or_depth");
  if (!(costs.targetSizeUsd > 0)) return invalid(edgeType, k, "invalid_target_size");

  const isSell = edgeType === "BINARY_OVERROUND";
  const sumVwap = r6(legs.reduce((a, l) => a + l.vwapPrice, 0));
  const sumBest = r6(legs.reduce((a, l) => a + l.bestPrice, 0));

  const grossRaw = isSell ? sumVwap - 1 : 1 - sumVwap;
  const grossEdge = r6(Math.max(0, grossRaw));

  /** Slippage de profundidade: quanto o VWAP se afasta do best level (diagnóstico). */
  const cSlippageDepth = r6(Math.abs(sumVwap - sumBest));

  /** Capital efetivamente imobilizado por unidade de payout $1. */
  const capitalPerUnit = isSell ? 1 : sumVwap;

  /** Unidades que $targetSize compra (cada unidade = 1 share por perna). */
  const unitsTargeted = r6(costs.targetSizeUsd / Math.max(1e-9, capitalPerUnit));
  const capacityShares = r6(Math.min(...legs.map(l => l.depthTop3)));
  const capacityOk = capacityShares >= unitsTargeted;

  const days = Math.max(0, input.daysToResolution);

  const cGas = r6((k * costs.gasPerTxUsd) / costs.targetSizeUsd);
  const cLockup = r6(costs.costOfCapitalAnnual * (days / 365) * capitalPerUnit);
  const cUma = r6(
    costs.umaHaircutByCategory[input.category] ?? costs.umaHaircutByCategory.unknown ?? 0.01,
  );
  const cConversion = r6(
    edgeType === "NEGRISK_CONVERSION" ? Math.max(0, input.conversionFeeFrac ?? 0) : 0,
  );
  const meanSpread = r6(legs.reduce((a, l) => a + l.spread, 0) / k);
  const cLegRisk = r6(costs.legRiskCoeff * meanSpread * Math.sqrt(Math.max(0, k - 1)));

  const netEdge = r6(grossEdge - cGas - cLockup - cUma - cConversion - cLegRisk);

  let verdict: MecSnapshotVerdict;
  if (netEdge < 0) {
    verdict = "negative_after_costs";
  } else if (netEdge < MEC_MARGINAL_NET) {
    verdict = "not_viable";
  } else if (!capacityOk) {
    verdict = "capacity_insufficient";
  } else if (netEdge >= MEC_VIABLE_NET) {
    verdict = "viable_pending_persistence";
  } else {
    verdict = "marginal_pending_persistence";
  }

  const note = [
    `sum_vwap=${sumVwap}`,
    `sum_best=${sumBest}`,
    `gross=${grossEdge}`,
    `cap_per_unit=${capitalPerUnit}`,
    `units=${unitsTargeted}`,
    `cap_shares=${capacityShares}`,
    `days=${days}`,
    `mean_spread=${meanSpread}`,
  ].join("|");

  return {
    mecVersion: MEC_VERSION,
    edgeType,
    k,
    grossEdge,
    capitalPerUnit: r6(capitalPerUnit),
    costs: { cSlippageDepth, cGas, cLockup, cUma, cConversion, cLegRisk },
    netEdge,
    unitsTargeted,
    capacityShares,
    capacityOk,
    daysToResolution: days,
    verdict,
    canUseForExecution: false,
    note,
  };
}

/** Modelo de custo default calibrado (sessão 2026-06): capital 10%, $100, UMA por categoria. */
export const MEC_DEFAULT_COST_MODEL: MecCostModel = {
  costOfCapitalAnnual: 0.1,
  gasPerTxUsd: 0.03,
  targetSizeUsd: 100,
  umaHaircutByCategory: {
    crypto_feed: 0.001,
    sports: 0.003,
    macro_data: 0.004,
    electoral: 0.006,
    subjective: 0.02,
    unknown: 0.01,
  },
  legRiskCoeff: 0.5,
};
