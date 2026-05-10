/**
 * Universe Quality Gate — bloqueia oportunidades antes do dispatch quando o
 * marketUniverseQuality cascade rejeita qualquer perna do trade.
 *
 * Cobre standard E graph: ambos passam por executionDispatcher.dispatchOpportunity,
 * que invoca este gate uma única vez por opportunity.
 *
 * Pure read-only — sem .paper, sem worker de execução, sem chamadas econômicas.
 * Apenas leituras do cache NormalizedMarket + evaluateMarketSuitability +
 * evaluateMarketUniverseQuality (libs já existentes e já auditadas).
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

/** Verdicts que o gate pode retornar — estende `UniverseQualityVerdict` com a
 *  rejeição derivada de `opp.edge` (Gate A): captura o caso onde cache prices
 *  diferem do preço de execução real (`1 - opp.edge` em
 *  realisticExecutionEngine.ts:170/200/295). */
export type GateRejectionVerdict =
  | UniverseQualityVerdict
  | "REJECT_TAIL_VIA_OPP_EDGE";

export interface UniverseQualityGateRejection {
  rejected: true;
  verdict: GateRejectionVerdict;
  legMarketId: string;
  question: string;
  mid: number | null;
  liquidity: number;
  volume: number;
  closed: boolean;
  suitabilityVerdict: MarketSuitabilityVerdict;
  reasons: string[];
  disqualifiers: string[];
  legsChecked: number;
}

export interface UniverseQualityGateAllow {
  rejected: false;
  verdict: UniverseQualityVerdict;
  legsChecked: number;
}

export type UniverseQualityGateResult =
  | UniverseQualityGateRejection
  | UniverseQualityGateAllow;

/** Lookup função injetada — produção usa cache do marketDataService; testes injetam fixtures. */
export type LegMarketLookup = (
  legMarketId: string,
) => NormalizedMarket | null | undefined;

/** Estrutura mínima esperada de uma opportunity no momento do dispatch. */
type DispatchOpportunityShape = Record<string, unknown> & {
  marketsInvolved?: Array<{ marketId?: string; question?: string }>;
  marketId?: unknown;
  id?: unknown;
};

/**
 * Extrai pernas individuais respeitando IDs compostos como "A+B".
 * Prioriza opportunity.marketsInvolved[].marketId; fallback para top-level
 * marketId/id apenas quando o token resultante for puramente numérico
 * (evita capturar identificadores de oportunidade do tipo "graph-cluster-...").
 */
export function extractLegMarketIds(
  opportunity: DispatchOpportunityShape,
): string[] {
  const ids = new Set<string>();
  const involved = Array.isArray(opportunity.marketsInvolved)
    ? opportunity.marketsInvolved
    : [];
  for (const m of involved) {
    const raw = m && typeof m.marketId === "string" ? m.marketId : null;
    if (!raw) continue;
    for (const leg of raw.split("+").map(s => s.trim()).filter(Boolean)) {
      ids.add(leg);
    }
  }
  if (ids.size === 0) {
    const topRaw =
      (typeof opportunity.marketId === "string" ? opportunity.marketId : null) ??
      (typeof opportunity.id === "string" ? opportunity.id : null);
    if (topRaw) {
      for (const leg of topRaw.split("+").map(s => s.trim()).filter(Boolean)) {
        if (/^[0-9]+$/.test(leg)) ids.add(leg);
      }
    }
  }
  return Array.from(ids);
}

/**
 * Verdicts de suitability "data-limited" — quando a falta de book no
 * NormalizedMarket impede prova plena, sintetizamos SUITABLE no input do UQ
 * para que o pipeline avalie topic / tail-via-mid / horizon mesmo assim.
 * Verdicts "hard" (CLOSED, EXPIRED, PRICE_PINNED, LOW_LIQUIDITY,
 * WEAK_FLAT_MARKOUTS) são honrados — UQ propaga REJECT_NOT_SUITABLE.
 */
const SUITABILITY_DATA_LIMITED: ReadonlySet<MarketSuitabilityVerdict> = new Set<
  MarketSuitabilityVerdict
>(["UNSUITABLE_NO_BOOK", "UNSUITABLE_MISSING_DATA"]);

/**
 * UQ verdicts tratados como bloqueio forte pelo gate (cobrem 92% do PnL negativo
 * identificado na auditoria do bot). REJECT_AMBIGUOUS é DELIBERADAMENTE excluído:
 * ocorre quando endDate ausente, condição comum em produção (NormalizedMarket
 * não carrega endDate). Tratamos AMBIGUOUS como "data-limited → allow".
 */
const HARD_REJECT_VERDICTS: ReadonlySet<UniverseQualityVerdict> = new Set<
  UniverseQualityVerdict
>([
  "REJECT_POLITICAL_LEGAL",
  "REJECT_MEME_OR_ABSURD",
  "REJECT_TAIL_OR_TICK_FENCE",
  "REJECT_LONG_HORIZON_NO_CATALYST",
  "REJECT_NOT_SUITABLE",
]);

function evaluateLeg(
  market: NormalizedMarket & { endDate?: string | null },
  nowIso: string,
): {
  uqVerdict: UniverseQualityVerdict;
  suitVerdict: MarketSuitabilityVerdict;
  reasons: string[];
  disqualifiers: string[];
  mid: number | null;
} {
  const lastPrice = market.prices.length ? market.prices[0]! : null;
  const minPrice = market.prices.length ? Math.min(...market.prices) : null;
  /** Para mercados binários YES/NO, prices[0] = P(YES); usar como lastPrice
   *  permite o suitability gate detectar pinning. */
  const suitResult = evaluateMarketSuitability({
    marketId: market.id,
    question: market.question,
    closed: market.closed,
    liquidity: market.liquidity,
    volume: market.volume,
    lastPrice,
    endDate: market.endDate ?? null,
    nowIso,
  });

  const uqSuitInput = SUITABILITY_DATA_LIMITED.has(suitResult.suitabilityVerdict)
    ? "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION"
    : suitResult.suitabilityVerdict;

  const uqResult = evaluateMarketUniverseQuality({
    marketId: market.id,
    question: market.question,
    slug: market.slug,
    category: market.category,
    nowIso,
    closed: market.closed,
    liquidity: market.liquidity,
    volume: market.volume,
    /** mid = lowest outcome probability — para binário é o "long-shot side";
     *  para múltiplos outcomes, o mais cauda. Permite tail check de UQ. */
    mid: minPrice,
    suitabilityVerdict: uqSuitInput,
    endDate: market.endDate ?? null,
  });

  return {
    uqVerdict: uqResult.universeQualityVerdict,
    suitVerdict: suitResult.suitabilityVerdict,
    reasons: uqResult.reasons,
    disqualifiers: uqResult.disqualifiers,
    mid: minPrice,
  };
}

/**
 * Banda saudável de preço efetivo de entrada — espelha a rejeição tail do UQ
 * (`mid < 0.06 || mid > 0.94`). Estritamente fora dessa banda dispara rejeição.
 * Boundary inclusivo do safe band: `priceProxy === 0.06` e `priceProxy === 0.94`
 * são SAFE (não rejeita).
 */
const PRICE_SAFE_BAND_LOWER = 0.06;
const PRICE_SAFE_BAND_UPPER = 0.94;

/**
 * Gate A — bloqueia oportunidades cujo preço efetivo de entrada (proxy)
 * implica entrada em cauda, INDEPENDENTE do que o cache de NormalizedMarket
 * diga sobre `prices`. Necessário porque `realisticExecutionEngine.ts` calcula
 * `effectiveEntryPrice = 1 - Math.max(0, opp.edge)` (linhas 170, 200, 295) e
 * NÃO consulta `getAllMarkets()`. Cache pode reportar mid saudável (Gamma
 * default 0.5) enquanto execução abre em 0.0095.
 *
 * Símétrico:
 *  - `1 - clampedEdge < 0.06`  → tail inferior  (overround com edge alto)
 *  - `1 - clampedEdge > 0.94`  → tail superior  (underround/edge baixo ou negativo)
 *
 * Sem `opp.edge`: retorna não-rejeitado, deixa fluxo seguir para leg check.
 */
export function evaluateOpportunityEdgeTail(
  opportunity: DispatchOpportunityShape,
): {
  rejected: boolean;
  rationale?: string;
  effectiveEntryPriceProxy?: number;
  rawEdge?: number;
} {
  const edgeRaw = opportunity.edge as unknown;
  if (typeof edgeRaw !== "number" || !Number.isFinite(edgeRaw)) {
    return { rejected: false };
  }
  /** Espelha o clamp do executor: realisticExecutionEngine.ts:124. */
  const clampedEdge = Math.max(0, edgeRaw);
  const effectiveEntryPriceProxy = 1 - clampedEdge;
  if (effectiveEntryPriceProxy < PRICE_SAFE_BAND_LOWER) {
    return {
      rejected: true,
      rationale: "edge_implies_lower_tail_entry_below_0p06",
      effectiveEntryPriceProxy,
      rawEdge: edgeRaw,
    };
  }
  if (effectiveEntryPriceProxy > PRICE_SAFE_BAND_UPPER) {
    return {
      rejected: true,
      rationale: "edge_implies_upper_tail_entry_above_0p94",
      effectiveEntryPriceProxy,
      rawEdge: edgeRaw,
    };
  }
  return { rejected: false, effectiveEntryPriceProxy, rawEdge: edgeRaw };
}

/**
 * Avalia a oportunidade contra o universe quality cascade.
 * Bloqueia se qualquer perna retornar verdict REJECT_*.
 * Fail-closed quando perna não está no cache: sem dado, sem prova, sem dispatch.
 */
export function evaluateOpportunityUniverseQuality(
  opportunity: DispatchOpportunityShape,
  lookupMarket: LegMarketLookup,
  nowIso: string,
): UniverseQualityGateResult {
  /**
   * Gate A — bloqueio por opp.edge implicar entrada em cauda. Roda ANTES da
   * extração de legs e do lookup de cache: fail-fast sem custo de I/O quando
   * a oportunidade carrega edge extremo. Necessário porque o executor usa
   * `1 - opp.edge` como effectiveEntryPrice, divergindo do cache prices.
   */
  const edgeTail = evaluateOpportunityEdgeTail(opportunity);
  if (edgeTail.rejected) {
    const legIdsForReport = extractLegMarketIds(opportunity);
    return {
      rejected: true,
      verdict: "REJECT_TAIL_VIA_OPP_EDGE",
      legMarketId: legIdsForReport[0] ?? "(no_legs_extractable)",
      question: "(blocked via opp.edge — pre-leg-lookup; cache prices not consulted)",
      mid: edgeTail.effectiveEntryPriceProxy ?? null,
      liquidity: 0,
      volume: 0,
      closed: false,
      suitabilityVerdict: "UNSUITABLE_MISSING_DATA",
      reasons: [edgeTail.rationale ?? "edge_implies_tail_entry"],
      disqualifiers: [
        `opp_edge_value:${edgeTail.rawEdge ?? "?"}`,
        `effective_entry_price_proxy:${edgeTail.effectiveEntryPriceProxy ?? "?"}`,
      ],
      legsChecked: 0,
    };
  }

  const legIds = extractLegMarketIds(opportunity);
  if (legIds.length === 0) {
    return {
      rejected: true,
      verdict: "REJECT_NOT_SUITABLE",
      legMarketId: "(none)",
      question: "(no legs extracted)",
      mid: null,
      liquidity: 0,
      volume: 0,
      closed: false,
      suitabilityVerdict: "UNSUITABLE_MISSING_DATA",
      reasons: ["no_leg_markets_to_check"],
      disqualifiers: ["empty_marketsInvolved"],
      legsChecked: 0,
    };
  }
  let lastUqVerdict: UniverseQualityVerdict = "ACCEPTABLE_BUT_SECONDARY";
  let checked = 0;
  for (const legId of legIds) {
    checked++;
    const market = lookupMarket(legId);
    if (!market) {
      return {
        rejected: true,
        verdict: "REJECT_NOT_SUITABLE",
        legMarketId: legId,
        question: "(leg not in cache)",
        mid: null,
        liquidity: 0,
        volume: 0,
        closed: false,
        suitabilityVerdict: "UNSUITABLE_MISSING_DATA",
        reasons: ["leg_market_not_found_in_cache"],
        disqualifiers: ["unknown_leg_id"],
        legsChecked: checked,
      };
    }
    const ev = evaluateLeg(
      market as NormalizedMarket & { endDate?: string | null },
      nowIso,
    );
    if (HARD_REJECT_VERDICTS.has(ev.uqVerdict)) {
      return {
        rejected: true,
        verdict: ev.uqVerdict,
        legMarketId: market.id,
        question: market.question,
        mid: ev.mid,
        liquidity: market.liquidity,
        volume: market.volume,
        closed: market.closed,
        suitabilityVerdict: ev.suitVerdict,
        reasons: ev.reasons,
        disqualifiers: ev.disqualifiers,
        legsChecked: checked,
      };
    }
    lastUqVerdict = ev.uqVerdict;
  }
  return { rejected: false, verdict: lastUqVerdict, legsChecked: checked };
}
