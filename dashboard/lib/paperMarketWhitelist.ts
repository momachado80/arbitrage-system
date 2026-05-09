/**
 * Subset fixo de mercados para o ciclo paper (prova económica).
 * PAPER_MARKET_IDS: vírgula-separados (Gamma market id). Vazio = sem filtro.
 */

import type { NormalizedPaperOpportunity } from "./paperTypes";

/** Um mercado ou cross_market (idA+idB) como no scanner. */
export function gammaIdsFromMarketIdField(marketId: string): string[] {
  if (!marketId) return [];
  if (marketId.includes("+")) {
    return marketId.split("+").map((s) => s.trim()).filter(Boolean);
  }
  return [marketId];
}

export function getPaperMarketWhitelist(): Set<string> | null {
  const raw = process.env.PAPER_MARKET_IDS?.trim();
  if (!raw) return null;
  const ids = raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (ids.length === 0) return null;
  if (ids.length < 3 || ids.length > 5) {
    console.warn(
      `[PaperWhitelist] PAPER_MARKET_IDS tem ${ids.length} ids (recomendado 3–5 para sprint)`
    );
  }
  return new Set(ids);
}

export function paperOpportunityMatchesWhitelist(
  opp: NormalizedPaperOpportunity,
  whitelist: Set<string> | null
): boolean {
  if (!whitelist) return true;
  const involved = opp.marketsInvolved;
  if (!involved.length) return false;
  const ids: string[] = [];
  for (const m of involved) {
    ids.push(...gammaIdsFromMarketIdField(m.marketId));
  }
  if (ids.length === 0) return false;
  return ids.every((id) => whitelist.has(id));
}

export function filterPaperOpportunitiesByWhitelist<T extends NormalizedPaperOpportunity>(
  opps: T[],
  whitelist: Set<string> | null
): T[] {
  if (!whitelist) return opps;
  return opps.filter((o) => paperOpportunityMatchesWhitelist(o, whitelist));
}
