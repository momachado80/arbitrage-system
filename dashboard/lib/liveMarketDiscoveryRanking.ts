/**
 * Ranking económico read-only para candidatos do discovery Gamma/CLOB.
 * Sem execução, sem ordens, sem credenciais. `canUseForMicrocapitalCandidate` deve ser false até fase própria.
 */

import {
  evaluateMarketUniverseQuality,
  type MarketUniverseQualityInput,
  type MarketUniverseQualityResult,
} from "./marketUniverseQuality";

export interface LiveDiscoveryRankInput {
  bestBidUsed: unknown;
  bestAskUsed: unknown;
  liquidity?: unknown;
  volume?: unknown;
  clobBookStructure?: unknown;
}

export interface LiveDiscoveryEconomicRankAugment extends LiveDiscoveryRankInput {
  quotesNearTickFence: boolean;
  microProbabilityTail: boolean;
  wideSpread: boolean;
  healthyMidRange: boolean;
  strongTwoSidedBook: boolean;
  mid: number | null;
  spread: number | null;
  /** Maior é melhor (empacotado para ordenar candidatos antes de watcher/worker). */
  economicRankScore: number;
  canUseForMicrocapitalCandidate: false;
}

export const DISCOVERY_TOP_CANDIDATES_CAP = 20;

function fin(x: unknown): number | null {
  if (typeof x === "number" && Number.isFinite(x)) return x;
  if (typeof x === "string" && x.trim()) {
    const n = parseFloat(x);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

/**
 * Cotas coladas aos extremos probabilísticos (degraus ~0 ou ~100 % no YES típico).
 * Inclui pares tipo 0.002 / 0.003 vistas em outsiders FIFA liquídos mas economicamente quasi-nulos no YES.
 */
export function deriveQuotesNearTickFence(bid: number | null, ask: number | null, mid: number | null): boolean {
  if (bid === null || ask === null || mid === null) return true;
  if (!(bid > 0 && ask > 0 && ask >= bid)) return true;
  if (bid <= 0.012 || ask >= 0.988) return true;
  if (mid < 0.05 && ask <= 0.08) return true;
  if (mid > 0.95 && bid >= 0.92) return true;
  return false;
}

export function deriveMicroProbabilityTail(mid: number | null): boolean {
  if (mid === null || !Number.isFinite(mid)) return true;
  return mid < 0.03 || mid > 0.97;
}

export function deriveHealthyMidRange(mid: number | null): boolean {
  if (mid === null || !Number.isFinite(mid)) return false;
  return mid >= 0.05 && mid <= 0.7;
}

export function deriveWideSpread(mid: number | null, spread: number | null): boolean {
  if (mid === null || spread === null || !Number.isFinite(mid) || !Number.isFinite(spread)) return true;
  const denom = Math.max(Math.min(mid, 1 - mid), 0.02);
  return spread / denom > 0.28;
}

export function computeLiveDiscoveryEconomicRank(input: LiveDiscoveryRankInput): Omit<LiveDiscoveryEconomicRankAugment, keyof LiveDiscoveryRankInput> {
  const bid = fin(input.bestBidUsed);
  const ask = fin(input.bestAskUsed);
  let mid: number | null = null;
  let spread: number | null = null;

  if (bid !== null && ask !== null && bid > 0 && ask > 0 && ask >= bid) {
    mid = (bid + ask) / 2;
    spread = ask - bid;
  }

  const liquidity = fin(input.liquidity) ?? 0;
  const volume = fin(input.volume) ?? 0;
  const bs = typeof input.clobBookStructure === "string" ? input.clobBookStructure : "";

  const quotesNearTickFence = deriveQuotesNearTickFence(bid, ask, mid);
  const microProbabilityTail = deriveMicroProbabilityTail(mid);
  const wideSpread = deriveWideSpread(mid, spread);
  const healthyMidRange = deriveHealthyMidRange(mid);
  const twoSidedBook = bs === "two_sided";

  const strongTwoSidedBook =
    twoSidedBook &&
    !quotesNearTickFence &&
    !microProbabilityTail &&
    spread !== null &&
    spread <= 0.06 &&
    mid !== null &&
    mid >= 0.05 &&
    mid <= 0.85;

  let score = 0;
  score += Math.log1p(Math.max(liquidity, 0)) * 2.15;
  score += Math.log1p(Math.max(volume, 0)) * 0.42;
  if (healthyMidRange) score += 95;
  if (strongTwoSidedBook) score += 52;
  if (twoSidedBook) score += 12;
  else score -= 28;
  if (microProbabilityTail) score -= 135;
  if (quotesNearTickFence) score -= 118;
  if (wideSpread) score -= 55;
  score += Math.min(mid ?? 0, 1 - (mid ?? 0)) * 18;

  return {
    quotesNearTickFence,
    microProbabilityTail,
    wideSpread,
    healthyMidRange,
    strongTwoSidedBook,
    mid,
    spread,
    economicRankScore: score,
    canUseForMicrocapitalCandidate: false,
  };
}

export function enrichDiscoverySuitableRow(summary: Record<string, unknown>): Record<string, unknown> {
  const econ = computeLiveDiscoveryEconomicRank({
    bestBidUsed: summary.bestBidUsed,
    bestAskUsed: summary.bestAskUsed,
    liquidity: summary.liquidity,
    volume: summary.volume,
    clobBookStructure: summary.clobBookStructure,
  });
  return { ...summary, ...econ };
}

export function finalizeDiscoveryRankingSplit(
  enrichedSuitable: Array<Record<string, unknown>>,
): { candidatesSorted: Record<string, unknown>[]; topCandidates: Record<string, unknown>[] } {
  const sorted = [...enrichedSuitable].sort((a, b) => {
    const sa = typeof a.economicRankScore === "number" && Number.isFinite(a.economicRankScore) ? a.economicRankScore : -Infinity;
    const sb = typeof b.economicRankScore === "number" && Number.isFinite(b.economicRankScore) ? b.economicRankScore : -Infinity;
    return sb - sa;
  });
  return {
    candidatesSorted: sorted,
    topCandidates: sorted.slice(0, DISCOVERY_TOP_CANDIDATES_CAP),
  };
}

/**
 * Enriquecimento adicional com filtro de qualidade de universo (read-only).
 * Não substitui o ranking econômico — adiciona campos `universeQuality*` e `disqualifiers`.
 */
function numLike(x: unknown): number | null {
  if (typeof x === "number" && Number.isFinite(x)) return x;
  if (typeof x === "string" && x.trim()) {
    const n = parseFloat(x);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

export function enrichRowWithUniverseQuality(
  row: Record<string, unknown>,
  nowIso?: string,
): Record<string, unknown> {
  const econ = (row.economicRankScore as number | undefined) ?? null;
  const input: MarketUniverseQualityInput = {
    marketId: typeof row.id === "string" || typeof row.id === "number" ? String(row.id) : null,
    question: typeof row.question === "string" ? row.question : null,
    slug: typeof row.slug === "string" ? row.slug : null,
    category: typeof row.category === "string" ? row.category : null,
    tags: row.tags ?? null,
    endDate: (row.endDate as string | null | undefined) ?? null,
    nowIso: nowIso ?? null,
    active: row.active,
    closed: row.closed,
    resolved: row.resolved,
    liquidity: numLike(row.liquidity),
    volume: numLike(row.volume),
    bestBid: numLike(row.bestBidUsed ?? row.bestBid),
    bestAsk: numLike(row.bestAskUsed ?? row.bestAsk),
    mid: numLike(row.mid),
    spread: numLike(row.spread),
    suitabilityVerdict: typeof row.suitabilityVerdict === "string" ? row.suitabilityVerdict : null,
    economicRankScore: econ,
    clobBookStructure: typeof row.clobBookStructure === "string" ? row.clobBookStructure : null,
    strongTwoSidedBook: row.strongTwoSidedBook === true,
    healthyMidRange: row.healthyMidRange === true,
    microProbabilityTail: row.microProbabilityTail === true,
    quotesNearTickFence: row.quotesNearTickFence === true,
  };
  const q: MarketUniverseQualityResult = evaluateMarketUniverseQuality(input);
  return {
    ...row,
    universeQualityVerdict: q.universeQualityVerdict,
    universeQualityScore: q.universeQualityScore,
    universeQualityReasons: q.reasons,
    disqualifiers: q.disqualifiers,
    isCleanForObservationScout: q.isCleanForObservationScout,
    eventProximityClass: q.eventProximityClass,
    universeAmbiguityRisk: q.ambiguityRisk,
    universeMemeOrAbsurdityRisk: q.memeOrAbsurdityRisk,
    universePoliticalLegalRisk: q.politicalLegalRisk,
    universeLongHorizonNoCatalystRisk: q.longHorizonNoCatalystRisk,
    universeTailRisk: q.tailRisk,
  };
}

function rankCleanScore(row: Record<string, unknown>): number {
  const uq = typeof row.universeQualityScore === "number" && Number.isFinite(row.universeQualityScore)
    ? row.universeQualityScore
    : -Infinity;
  return uq;
}

function rankEconScore(row: Record<string, unknown>): number {
  return typeof row.economicRankScore === "number" && Number.isFinite(row.economicRankScore)
    ? row.economicRankScore
    : -Infinity;
}

/**
 * Resultado completo de ranking com universe quality.
 * - candidates: todos os suitable enriquecidos (preserva contrato anterior).
 * - topCandidates: top do ranking econômico (preserva contrato anterior).
 * - topCleanCandidates: subconjunto isCleanForObservationScout=true, ordenado por universeQualityScore depois economicRankScore.
 * - rejectedByUniverseQuality: candidatos suitable mas reprovados pelo filtro de qualidade de universo.
 * - universeQualityRejectionReasons: contagem por verdict de rejeição.
 */
export function finalizeDiscoveryRankingWithUniverseQuality(
  enrichedSuitable: Array<Record<string, unknown>>,
  nowIso?: string,
): {
  candidatesSorted: Record<string, unknown>[];
  topCandidates: Record<string, unknown>[];
  topCleanCandidates: Record<string, unknown>[];
  rejectedByUniverseQuality: Record<string, unknown>[];
  universeQualityRejectionReasons: Array<{ verdict: string; count: number }>;
} {
  const enrichedWithQuality = enrichedSuitable.map(r => enrichRowWithUniverseQuality(r, nowIso));
  const econSorted = [...enrichedWithQuality].sort((a, b) => rankEconScore(b) - rankEconScore(a));
  const topCandidates = econSorted.slice(0, DISCOVERY_TOP_CANDIDATES_CAP);

  const cleanRows = enrichedWithQuality.filter(r => r.universeQualityVerdict === "CLEAN_OBSERVATION_UNIVERSE");
  const topCleanCandidates = [...cleanRows].sort((a, b) => {
    const ucDiff = rankCleanScore(b) - rankCleanScore(a);
    if (ucDiff !== 0) return ucDiff;
    const econDiff = rankEconScore(b) - rankEconScore(a);
    if (econDiff !== 0) return econDiff;
    const sb = (b.strongTwoSidedBook === true ? 1 : 0) - (a.strongTwoSidedBook === true ? 1 : 0);
    if (sb !== 0) return sb;
    const hr = (b.healthyMidRange === true ? 1 : 0) - (a.healthyMidRange === true ? 1 : 0);
    if (hr !== 0) return hr;
    const sa = typeof a.spread === "number" && Number.isFinite(a.spread) ? a.spread : Infinity;
    const sbN = typeof b.spread === "number" && Number.isFinite(b.spread) ? b.spread : Infinity;
    return sa - sbN;
  });

  const rejectedByUniverseQuality = enrichedWithQuality
    .filter(r => typeof r.universeQualityVerdict === "string" && (r.universeQualityVerdict as string).startsWith("REJECT_"))
    .map(r => ({
      id: r.id ?? null,
      question: r.question ?? null,
      slug: r.slug ?? null,
      universeQualityVerdict: r.universeQualityVerdict ?? null,
      disqualifiers: r.disqualifiers ?? [],
      universeAmbiguityRisk: r.universeAmbiguityRisk ?? null,
      universeMemeOrAbsurdityRisk: r.universeMemeOrAbsurdityRisk ?? null,
      universePoliticalLegalRisk: r.universePoliticalLegalRisk ?? null,
      universeLongHorizonNoCatalystRisk: r.universeLongHorizonNoCatalystRisk ?? null,
      universeTailRisk: r.universeTailRisk ?? null,
      eventProximityClass: r.eventProximityClass ?? null,
    }));

  const counts: Record<string, number> = {};
  for (const r of rejectedByUniverseQuality) {
    const k = String(r.universeQualityVerdict ?? "UNKNOWN");
    counts[k] = (counts[k] ?? 0) + 1;
  }
  const universeQualityRejectionReasons = Object.entries(counts)
    .map(([verdict, count]) => ({ verdict, count }))
    .sort((a, b) => b.count - a.count);

  return {
    candidatesSorted: enrichedWithQuality,
    topCandidates,
    topCleanCandidates: topCleanCandidates.slice(0, DISCOVERY_TOP_CANDIDATES_CAP),
    rejectedByUniverseQuality,
    universeQualityRejectionReasons,
  };
}
