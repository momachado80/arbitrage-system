/**
 * Mechanical Edge Census — camada de livro (pura).
 *
 * Computa VWAP por profundidade, melhor preço, profundidade agregada e os flags
 * de coerência do Tier 1 a partir de níveis brutos de order book. Sem rede, sem
 * I/O. O runner (scripts/runMechanicalEdgeCensus.ts) faz o fetch e passa os níveis
 * para cá; toda a aritmética testável vive neste arquivo.
 *
 * Sem trade, sem .paper, sem microcapital, sem execução. Read-only census.
 */

export interface BookLevel {
  price: number;
  size: number;
}

export interface VwapResult {
  /** Preço médio ponderado por volume para preencher o alvo (null se nada preenche). */
  vwap: number | null;
  /** Shares efetivamente preenchidas (≤ alvo se profundidade insuficiente). */
  filledShares: number;
  /** True se a profundidade cobre o tamanho-alvo inteiro. */
  fullyFilled: boolean;
  /** Custo total das shares preenchidas (Σ price·size). */
  cost: number;
}

function cleanLevels(levels: BookLevel[]): BookLevel[] {
  return levels.filter(
    l =>
      Number.isFinite(l.price) &&
      Number.isFinite(l.size) &&
      l.price > 0 &&
      l.price < 1 &&
      l.size > 0,
  );
}

/**
 * VWAP para preencher `targetShares` consumindo o livro do melhor preço para o pior.
 * side "buy" varre asks ascendente; "sell" varre bids descendente.
 */
export function computeVwap(
  levels: BookLevel[],
  targetShares: number,
  side: "buy" | "sell",
): VwapResult {
  if (!(targetShares > 0)) {
    return { vwap: null, filledShares: 0, fullyFilled: false, cost: 0 };
  }
  const clean = cleanLevels(levels);
  const sorted =
    side === "buy"
      ? clean.sort((a, b) => a.price - b.price)
      : clean.sort((a, b) => b.price - a.price);

  let remaining = targetShares;
  let cost = 0;
  let filled = 0;
  for (const lvl of sorted) {
    const take = Math.min(remaining, lvl.size);
    cost += take * lvl.price;
    filled += take;
    remaining -= take;
    if (remaining <= 1e-9) break;
  }
  if (filled <= 0) {
    return { vwap: null, filledShares: 0, fullyFilled: false, cost: 0 };
  }
  const fullyFilled = filled >= targetShares - 1e-9;
  return { vwap: cost / filled, filledShares: filled, fullyFilled, cost };
}

/** Melhor preço do lado relevante (menor ask / maior bid). null se vazio. */
export function bestPrice(levels: BookLevel[], side: "buy" | "sell"): number | null {
  const clean = cleanLevels(levels);
  if (clean.length === 0) return null;
  return side === "buy"
    ? Math.min(...clean.map(l => l.price))
    : Math.max(...clean.map(l => l.price));
}

/** Profundidade agregada (Σ size) dos N melhores níveis do lado relevante. */
export function depthTopN(levels: BookLevel[], n: number, side: "buy" | "sell"): number {
  const clean = cleanLevels(levels);
  const sorted =
    side === "buy"
      ? clean.sort((a, b) => a.price - b.price)
      : clean.sort((a, b) => b.price - a.price);
  return sorted.slice(0, Math.max(0, n)).reduce((s, l) => s + l.size, 0);
}

/**
 * Tier 1 — flag barato de underround binário usando só o melhor nível.
 * True quando ask_yes + ask_no < 1 − minGross (comprar ambos < payout 1).
 */
export function binaryUnderroundFlag(
  bestAskYes: number,
  bestAskNo: number,
  minGross: number,
): boolean {
  if (!Number.isFinite(bestAskYes) || !Number.isFinite(bestAskNo)) return false;
  return bestAskYes + bestAskNo < 1 - minGross;
}

/**
 * Tier 1 — flag barato de overround binário (vender ambos > payout 1).
 * True quando bid_yes + bid_no > 1 + minGross.
 */
export function binaryOverroundFlag(
  bestBidYes: number,
  bestBidNo: number,
  minGross: number,
): boolean {
  if (!Number.isFinite(bestBidYes) || !Number.isFinite(bestBidNo)) return false;
  return bestBidYes + bestBidNo > 1 + minGross;
}

/**
 * Tier 1 — flag de underround de partição (mutuamente exclusiva, exaustiva).
 * True quando Σ ask_yes_i < 1 − minGross. Requer ≥ 2 pernas com ask finito.
 */
export function partitionUnderroundFlag(bestAsks: number[], minGross: number): boolean {
  const valid = bestAsks.filter(a => Number.isFinite(a) && a > 0 && a < 1);
  if (valid.length < 2) return false;
  const sum = valid.reduce((a, b) => a + b, 0);
  return sum < 1 - minGross;
}

/**
 * Estima quantas shares por perna o tamanho-alvo em USD compra, dado o capital por
 * unidade (≈ Σ best ask). Cada unidade de cesta exige 1 share por perna.
 */
export function targetSharesPerLeg(targetSizeUsd: number, capitalPerUnit: number): number {
  if (!(targetSizeUsd > 0) || !(capitalPerUnit > 0)) return 0;
  return targetSizeUsd / capitalPerUnit;
}

/**
 * Classifica a categoria de resolução de um mercado a partir do texto (question/slug),
 * para selecionar o haircut UMA. Heurístico, conservador: o que não casar cai em
 * "unknown" (1%) ou "subjective" (2%) — nunca subestima risco de resolução.
 */
export function classifyResolutionCategory(text: string): string {
  const t = (text || "").toLowerCase();
  if (/\b(bitcoin|btc|ethereum|eth|solana|\bsol\b|crypto|dogecoin|\$\s?\d|reach\s+\$|price\s+of)\b/.test(t)) {
    return "crypto_feed";
  }
  if (
    /\b(nba|nfl|nhl|mlb|ncaa|premier\s+league|champions\s+league|world\s+cup|super\s+bowl|stanley\s+cup|finals?|playoffs?|win\s+the\s+(?:\d{4}\s+)?(?:nba|nhl|nfl|series|championship)|vs\.?\b|defeat|beat)\b/.test(
      t,
    )
  ) {
    return "sports";
  }
  if (/\b(election|president|presidential|senate|governor|primary|nominee|electoral|congress|parliament|prime\s+minister)\b/.test(t)) {
    return "electoral";
  }
  if (/\b(cpi|inflation|fed|federal\s+reserve|interest\s+rate|gdp|unemployment|fomc|rate\s+cut|rate\s+hike|jobs\s+report|recession)\b/.test(t)) {
    return "macro_data";
  }
  if (/\b(will\s+.*\b(say|tweet|post|announce|resign|fire|mention)|by\s+(?:the\s+)?end\s+of|before\s+\d{4})\b/.test(t)) {
    return "subjective";
  }
  return "unknown";
}
