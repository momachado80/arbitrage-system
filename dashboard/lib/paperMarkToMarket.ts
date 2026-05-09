/**
 * Mark-to-market mínimo para paper trades, alinhado à convenção do scanner/simulador:
 * exitPrice / markPx = 1 - edge (edge como em DetectedEdge).
 *
 * `graph_complement`: a oportunidade usa `edge = |pA+pB-1|` (violação complementary em
 * `probabilityGraph`) e `entryPriceEstimate = 1 - edge`. O MTM tem de usar as **duas** pernas
 * com os mesmos `prices[0]`; o proxy single-market `2 - (pY+pN)` dum único binário ≈ 1
 * sempre que pY+pN≈1, o que degenerava `exitPriceEstimate` para 1 e PnL irreal.
 */

import type { NormalizedMarket } from "./polymarketClient";
import type { PaperTrade } from "./paperTypes";

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return x;
  return Math.min(1, Math.max(0, x));
}

function marketPrices(m: NormalizedMarket): number[] | null {
  if (!m.prices || m.prices.length < 2) return null;
  return m.prices;
}

function resolveCrossLegIds(trade: PaperTrade): [string, string] | null {
  const inv = trade.marketsInvolved || [];
  if (inv.length >= 2 && inv[0].marketId && inv[1].marketId) {
    return [inv[0].marketId, inv[1].marketId];
  }
  const parts = (trade.opportunityId || "")
    .split("+")
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length >= 2) return [parts[0], parts[1]];
  return null;
}

/**
 * Retorna markPx coerente com simulateExit (markPx = 1 - edge_mtm).
 * null se faltar mercado ou preços.
 */
export function resolveMarkPxFromTrade(
  trade: PaperTrade,
  marketsById: Map<string, NormalizedMarket>
): number | null {
  if (trade.opportunityType === "cross_market") {
    const legs = resolveCrossLegIds(trade);
    if (!legs) return null;
    const [idA, idB] = legs;
    const ma = marketsById.get(idA);
    const mb = marketsById.get(idB);
    if (!ma || !mb) return null;
    const pa = marketPrices(ma);
    const pb = marketPrices(mb);
    if (!pa || !pb) return null;
    const edgeMtm = pa[0] + pb[0] - 1;
    return 1 - edgeMtm;
  }

  if (trade.opportunityType === "overround" || trade.opportunityType === "underround") {
    const id =
      trade.marketsInvolved?.[0]?.marketId ||
      (trade.opportunityId.includes("+") ? null : trade.opportunityId) ||
      null;
    if (!id) return null;
    const m = marketsById.get(id);
    if (!m) return null;
    const prices = marketPrices(m);
    if (!prices) return null;
    const sum = prices.reduce((s, p) => s + p, 0);
    if (trade.opportunityType === "overround") {
      const edgeMtm = sum - 1;
      return 1 - edgeMtm;
    }
    const edgeMtm = 1 - sum;
    return 1 - edgeMtm;
  }

  /**
   * Complemento no grafo: mesma convenção que `getConstraintViolations` (complementary) —
   * `severity = |pA+pB-1|` e no paper `entryPriceEstimate = 1 - opportunity.edge`.
   */
  if (trade.opportunityType === "graph_complement") {
    const legs = resolveCrossLegIds(trade);
    if (!legs) return null;
    const [idA, idB] = legs;
    const ma = marketsById.get(idA);
    const mb = marketsById.get(idB);
    if (!ma || !mb) return null;
    const pa = marketPrices(ma);
    const pb = marketPrices(mb);
    if (!pa || !pb) return null;
    const pA = pa[0];
    const pB = pb[0];
    if (!Number.isFinite(pA) || !Number.isFinite(pB)) return null;
    const deviation = Math.abs(pA + pB - 1);
    return clamp01(1 - deviation);
  }

  /**
   * Micro-lanes 2-pernas derivadas de violação equivalence (classificação estrutural):
   * edge observada = |pA - pB|; markPx = 1 - |pA_now - pB_now|.
   */
  if (
    trade.opportunityType === "graph_equivalence_micro" ||
    trade.opportunityType === "graph_subset_micro" ||
    trade.opportunityType === "graph_exclusive_micro"
  ) {
    const legs = resolveCrossLegIds(trade);
    if (!legs) return null;
    const [idA, idB] = legs;
    const ma = marketsById.get(idA);
    const mb = marketsById.get(idB);
    if (!ma || !mb) return null;
    const pa = marketPrices(ma);
    const pb = marketPrices(mb);
    if (!pa || !pb) return null;
    const pA = pa[0];
    const pB = pb[0];
    if (!Number.isFinite(pA) || !Number.isFinite(pB)) return null;
    const divergence = Math.abs(pA - pB);
    return clamp01(1 - divergence);
  }

  /**
   * Outros `graph_*` (subset, exclusive, equivalence, cycle): não há proxy single-market seguro
   * (o ramo binário `2-sum` dum mercado só devolve ≈1). Sem MTM → `latestState` só via oppMap
   * ou fallback no fecho.
   */
  if (trade.opportunityType.startsWith("graph_")) {
    return null;
  }

  /**
   * Tipos não-graph restantes: proxy single-market overround (binário / multi-outcome no mesmo mercado).
   */
  const id =
    trade.marketsInvolved?.[0]?.marketId ||
    (trade.opportunityId.includes("+") ? null : trade.opportunityId) ||
    null;
  if (!id) return null;
  const m = marketsById.get(id);
  if (!m) return null;
  const prices = marketPrices(m);
  if (!prices) return null;
  const sum = prices.reduce((s, p) => s + p, 0);
  const edgeMtm = sum - 1;
  return 1 - edgeMtm;
}
