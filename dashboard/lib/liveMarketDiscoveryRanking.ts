/**
 * Ranking económico read-only para candidatos do discovery Gamma/CLOB.
 * Sem execução, sem ordens, sem credenciais. `canUseForMicrocapitalCandidate` deve ser false até fase própria.
 */

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
