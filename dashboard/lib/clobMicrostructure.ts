/**
 * Microestrutura pública CLOB + campos Gamma (bestBid/bestAsk) — sem auth.
 * Usado pela infra market-data-truth apenas.
 */

const GAMMA_MARKET_URL = "https://gamma-api.polymarket.com/markets";
const CLOB_BASE = (process.env.POLYMARKET_CLOB_HOST || "https://clob.polymarket.com").replace(/\/$/, "");
const FETCH_TIMEOUT_MS = 8_000;

export type MicroPriceSource = "clob_rest" | "gamma_best_hint" | "gamma_outcome_only";

export interface ParsedClobBook {
  bestBid: number;
  bestAsk: number;
  mid: number;
  spread: number;
  depthBidTop3: number;
  depthAskTop3: number;
  bidLevels: number;
  askLevels: number;
}

interface ClobLevel {
  price?: string;
  size?: string;
}

function num(x: unknown): number {
  const n = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(n) ? n : NaN;
}

export async function fetchGammaMarketRawJson(marketId: string): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`${GAMMA_MARKET_URL}/${encodeURIComponent(marketId)}`, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return null;
    const j = (await res.json()) as unknown;
    return j && typeof j === "object" && !Array.isArray(j) ? (j as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

/** Gamma devolve `clobTokenIds` / `outcomes` como string JSON ou array — alinhado ao probe CLOB 1823789. */
function parseGammaStringArrayField(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(v => String(v)).filter(Boolean);
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value) as unknown;
      return Array.isArray(parsed) ? parsed.map(v => String(v)).filter(Boolean) : [];
    } catch {
      return [];
    }
  }
  return [];
}

export function parseClobTokenIds(raw: Record<string, unknown>): string[] {
  return parseGammaStringArrayField(raw.clobTokenIds);
}

export function parseGammaOutcomes(raw: Record<string, unknown>): string[] {
  return parseGammaStringArrayField(raw.outcomes);
}

/** Token outcome YES (por ordem outcomes); fallback ao primeiro id — mesmo critério que `crossVenueAnchor1823789ClobConnectivityProbe`. */
export function resolveYesClobTokenId(raw: Record<string, unknown>): string {
  const outcomes = parseGammaOutcomes(raw);
  const tokenIds = parseClobTokenIds(raw);
  const yesIdx = outcomes.findIndex(o => /^yes$/i.test(o.trim()));
  if (yesIdx >= 0 && yesIdx < tokenIds.length) return tokenIds[yesIdx]!;
  return tokenIds[0] ?? "";
}

export function extractGammaBestBidAsk(raw: Record<string, unknown>): { bid: number; ask: number } | null {
  const bid = num(raw.bestBid);
  const ask = num(raw.bestAsk);
  if (!Number.isFinite(bid) || !Number.isFinite(ask) || ask <= bid) return null;
  return { bid, ask };
}

/**
 * Melhor bid/ask a partir do livro CLOB.
 * Livros one-sided (asks_only / bids_only) não são `null`: observabilidade alinha-se ao probe de conectividade;
 * spread pode ser 0 quando só há um lado (spread intrínseco desconhecido no REST).
 */
export async function fetchParsedClobBook(tokenId: string): Promise<ParsedClobBook | null> {
  if (!tokenId) return null;
  try {
    const url = `${CLOB_BASE}/book?token_id=${encodeURIComponent(tokenId)}`;
    const res = await fetch(url, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return null;
    const j = (await res.json()) as { bids?: ClobLevel[]; asks?: ClobLevel[] };
    const bids = Array.isArray(j.bids) ? j.bids : [];
    const asks = Array.isArray(j.asks) ? j.asks : [];
    const bidPrices = bids.map(b => num(b.price)).filter(x => Number.isFinite(x));
    const askPrices = asks.map(a => num(a.price)).filter(x => Number.isFinite(x));

    if (bidPrices.length === 0 && askPrices.length === 0) return null;

    const bidSorted = [...bids].sort((a, b) => num(b.price) - num(a.price));
    const askSorted = [...asks].sort((a, b) => num(a.price) - num(b.price));
    const sumTop = (arr: ClobLevel[], n: number) =>
      arr.slice(0, n).reduce((s, x) => s + (Number.isFinite(num(x.size)) ? Math.max(0, num(x.size)) : 0), 0);

    let bestBid: number;
    let bestAsk: number;
    let mid: number;
    let spread: number;

    if (bidPrices.length > 0 && askPrices.length > 0) {
      bestBid = Math.max(...bidPrices);
      bestAsk = Math.min(...askPrices);
      if (!(bestAsk > bestBid)) return null;
      mid = (bestBid + bestAsk) / 2;
      spread = bestAsk - bestBid;
    } else if (askPrices.length > 0) {
      bestBid = 0;
      bestAsk = Math.min(...askPrices);
      if (!(bestAsk > 0)) return null;
      mid = bestAsk;
      spread = 0;
    } else {
      bestBid = Math.max(...bidPrices);
      bestAsk = 0;
      if (!(bestBid > 0)) return null;
      mid = bestBid;
      spread = 0;
    }

    return {
      bestBid,
      bestAsk,
      mid,
      spread,
      depthBidTop3: sumTop(bidSorted, 3),
      depthAskTop3: sumTop(askSorted, 3),
      bidLevels: bids.length,
      askLevels: asks.length,
    };
  } catch {
    return null;
  }
}
