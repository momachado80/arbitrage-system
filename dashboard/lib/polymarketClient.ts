export interface PolymarketRawMarket {
  id: string;
  question: string;
  outcomes: string;
  outcomePrices: string;
  liquidity: string;
  volume: string;
  active: boolean;
  closed: boolean;
  market_slug: string;
  category?: string;
}

export interface NormalizedMarket {
  id: string;
  question: string;
  slug: string;
  category: string;
  outcomes: string[];
  prices: number[];
  liquidity: number;
  volume: number;
  active: boolean;
  closed: boolean;
  spread: number;
  probSum: number;
}

const GAMMA_URL = "https://gamma-api.polymarket.com/markets";
const FETCH_TIMEOUT_MS = 10_000;
const PAGE_LIMIT = 100;
const MAX_PAGES = 10;

let cache: NormalizedMarket[] = [];
let cacheTs = 0;
const CACHE_TTL_MS = 5_000;

function normalize(raw: PolymarketRawMarket): NormalizedMarket | null {
  try {
    const outcomes: string[] = JSON.parse(raw.outcomes || "[]");
    const prices: number[] = JSON.parse(raw.outcomePrices || "[]").map(Number);

    if (outcomes.length === 0 || prices.length === 0) return null;
    if (outcomes.length !== prices.length) return null;

    const liquidity = parseFloat(raw.liquidity) || 0;
    const volume = parseFloat(raw.volume) || 0;
    const probSum = prices.reduce((s, p) => s + p, 0);

    const sorted = [...prices].sort((a, b) => b - a);
    const spread = sorted.length >= 2 ? sorted[0] - sorted[sorted.length - 1] : 0;

    return {
      id: raw.id,
      question: raw.question,
      slug: raw.market_slug || raw.id,
      category: raw.category || "general",
      outcomes,
      prices,
      liquidity,
      volume,
      active: raw.active,
      closed: raw.closed,
      spread,
      probSum,
    };
  } catch {
    return null;
  }
}

async function fetchPage(offset: number): Promise<PolymarketRawMarket[]> {
  const url = `${GAMMA_URL}?limit=${PAGE_LIMIT}&offset=${offset}&active=true&closed=false`;
  const res = await fetch(url, {
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`Gamma API ${res.status}`);
  const body = await res.text();
  try {
    return JSON.parse(body) as PolymarketRawMarket[];
  } catch (e) {
    throw new Error(`Gamma API invalid JSON: ${e instanceof Error ? e.message : "parse error"}`);
  }
}

const TOTAL_FETCH_TIMEOUT_MS = 90_000;

export async function fetchAllMarkets(): Promise<NormalizedMarket[]> {
  const now = Date.now();
  if (cache.length > 0 && now - cacheTs < CACHE_TTL_MS) {
    return cache;
  }

  const t0 = Date.now();
  const all: NormalizedMarket[] = [];
  const timeoutPromise = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error(`fetchAllMarkets timeout ${TOTAL_FETCH_TIMEOUT_MS}ms`)), TOTAL_FETCH_TIMEOUT_MS)
  );

  const fetchPromise = (async () => {
    for (let page = 0; page < MAX_PAGES; page++) {
      const raw = await fetchPage(page * PAGE_LIMIT);
      for (const r of raw) {
        const m = normalize(r);
        if (m && !m.closed && m.active) all.push(m);
      }
      if (raw.length < PAGE_LIMIT) break;
    }
    return all;
  })();

  const result = await Promise.race([fetchPromise, timeoutPromise]);
  cache = result;
  cacheTs = Date.now();
  console.log(`[PolymarketClient] Fetched ${result.length} markets in ${Date.now() - t0}ms`);
  return result;
}

export function getCachedMarkets(): NormalizedMarket[] {
  return cache;
}
