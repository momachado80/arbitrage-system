import {
  recordGammaFetchByIdAttempt,
  type GammaByIdFailureCode,
} from "./gammaFetchByIdDiagnostics";

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

function gammaByIdNormalizeHint(raw: PolymarketRawMarket): string {
  try {
    const oc = raw.outcomes ?? "[]";
    const pc = raw.outcomePrices ?? "[]";
    const outcomes = JSON.parse(typeof oc === "string" ? oc : JSON.stringify(oc));
    const prices = JSON.parse(typeof pc === "string" ? pc : JSON.stringify(pc));
    if (!Array.isArray(outcomes) || outcomes.length === 0) return "outcomes_empty_or_missing";
    if (!Array.isArray(prices) || prices.length === 0) return "prices_empty_or_missing";
    if (outcomes.length !== prices.length) return `outcomes_prices_length_mismatch_${outcomes.length}_${prices.length}`;
    return "normalize_returned_null_other";
  } catch (e) {
    return `outcomes_prices_json_error:${e instanceof Error ? e.message : String(e)}`;
  }
}

function classifyGammaByIdFetchError(e: unknown): { code: GammaByIdFailureCode; detail: string } {
  const msg = e instanceof Error ? e.message : String(e);
  const name = e instanceof Error ? e.name : "";
  if (name === "AbortError" || /aborted|timeout/i.test(msg)) {
    return { code: "TIMEOUT_OR_ABORT", detail: `${name}: ${msg}` };
  }
  if (e instanceof TypeError) {
    return { code: "NETWORK_ERROR", detail: `${name}: ${msg}` };
  }
  return { code: "UNKNOWN_ERROR", detail: `${name}: ${msg}` };
}

/** Um mercado por id Gamma (paper whitelist quando o id não está no cache paginado). */
export async function fetchNormalizedMarketById(marketId: string): Promise<NormalizedMarket | null> {
  const baseDiag = {
    httpRequested: true,
    httpStatus: null as number | null,
    failureCode: null as GammaByIdFailureCode | null,
    detail: null as string | null,
    jsonReceived: false,
    jsonWasObject: false,
    normalizeReturnedNull: false,
    marketActive: null as boolean | null,
    marketClosed: null as boolean | null,
  };
  try {
    const url = `${GAMMA_URL}/${encodeURIComponent(marketId)}`;
    const res = await fetch(url, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { Accept: "application/json" },
    });
    baseDiag.httpStatus = res.status;
    if (res.status === 429) {
      recordGammaFetchByIdAttempt(marketId, {
        ...baseDiag,
        outcome: "failure",
        failureCode: "HTTP_429_RATE_LIMIT",
        detail: "rate_limited",
      });
      return null;
    }
    if (!res.ok) {
      recordGammaFetchByIdAttempt(marketId, {
        ...baseDiag,
        outcome: "failure",
        failureCode: "HTTP_NON_OK",
        detail: `status_${res.status}`,
      });
      return null;
    }
    let raw: unknown;
    try {
      raw = await res.json();
    } catch (e) {
      recordGammaFetchByIdAttempt(marketId, {
        ...baseDiag,
        outcome: "failure",
        failureCode: "JSON_PARSE_ERROR",
        detail: e instanceof Error ? e.message : String(e),
      });
      return null;
    }
    baseDiag.jsonReceived = true;
    if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
      recordGammaFetchByIdAttempt(marketId, {
        ...baseDiag,
        outcome: "failure",
        failureCode: "JSON_NOT_OBJECT",
        detail: raw === null ? "body_null" : Array.isArray(raw) ? "body_array" : typeof raw,
      });
      return null;
    }
    baseDiag.jsonWasObject = true;
    const rawMarket = raw as PolymarketRawMarket;
    const m = normalize(rawMarket);
    if (!m) {
      baseDiag.normalizeReturnedNull = true;
      recordGammaFetchByIdAttempt(marketId, {
        ...baseDiag,
        outcome: "failure",
        failureCode: "NORMALIZE_REJECTED",
        detail: gammaByIdNormalizeHint(rawMarket),
      });
      return null;
    }
    baseDiag.marketActive = m.active;
    baseDiag.marketClosed = m.closed;
    if (m.closed || !m.active) {
      recordGammaFetchByIdAttempt(marketId, {
        ...baseDiag,
        outcome: "failure",
        failureCode: "MARKET_INACTIVE_OR_CLOSED",
        detail: `active=${m.active} closed=${m.closed}`,
      });
      return null;
    }
    recordGammaFetchByIdAttempt(marketId, {
      ...baseDiag,
      outcome: "success",
      failureCode: null,
      detail: null,
      normalizeReturnedNull: false,
    });
    return m;
  } catch (e) {
    const { code, detail } = classifyGammaByIdFetchError(e);
    recordGammaFetchByIdAttempt(marketId, {
      ...baseDiag,
      outcome: "failure",
      failureCode: code,
      detail,
    });
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

export type RefreshStepReporter = (step: string, detail?: { count?: number }) => void;

function withStepTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    p,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timeout after ${ms}ms`)), ms)
    ),
  ]);
}

export async function fetchAllMarkets(options?: {
  reportStep?: RefreshStepReporter;
  fetchStandardTimeoutMs?: number;
  parseTimeoutMs?: number;
  mergeTimeoutMs?: number;
}): Promise<NormalizedMarket[]> {
  const now = Date.now();
  if (cache.length > 0 && now - cacheTs < CACHE_TTL_MS) {
    return cache;
  }

  const report = options?.reportStep;
  const fetchTimeout = options?.fetchStandardTimeoutMs ?? 60_000;
  const parseTimeout = options?.parseTimeoutMs ?? 15_000;
  const mergeTimeout = options?.mergeTimeoutMs ?? 5_000;

  const t0 = Date.now();
  const timeoutPromise = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error(`fetchAllMarkets total timeout ${TOTAL_FETCH_TIMEOUT_MS}ms`)), TOTAL_FETCH_TIMEOUT_MS)
  );

  const doFetch = async (): Promise<NormalizedMarket[]> => {
    report?.("fetch_standard_begin");
    const allRaw: PolymarketRawMarket[] = [];
    const fetchPhase = (async () => {
      for (let page = 0; page < MAX_PAGES; page++) {
        const raw = await fetchPage(page * PAGE_LIMIT);
        allRaw.push(...raw);
        if (raw.length < PAGE_LIMIT) break;
      }
    })();
    await withStepTimeout(fetchPhase, fetchTimeout, "fetch_standard");
    report?.("fetch_standard_done", { count: allRaw.length });

    report?.("parse_standard_begin");
    const parsePhase = (async () => {
      const all: NormalizedMarket[] = [];
      for (const r of allRaw) {
        const m = normalize(r);
        if (m && !m.closed && m.active) all.push(m);
      }
      return all;
    })();
    const parsed = await withStepTimeout(parsePhase, parseTimeout, "parse_standard");
    report?.("parse_standard_done", { count: parsed.length });

    report?.("fetch_graph_begin");
    report?.("fetch_graph_done");
    report?.("merge_begin");
    const mergePhase = Promise.resolve(parsed);
    const result = await withStepTimeout(mergePhase, mergeTimeout, "merge");
    report?.("merge_done", { count: result.length });

    return result;
  };

  const result = await Promise.race([doFetch(), timeoutPromise]);
  cache = result;
  cacheTs = Date.now();
  console.log(`[PolymarketClient] Fetched ${result.length} markets in ${Date.now() - t0}ms`);
  return result;
}

export function getCachedMarkets(): NormalizedMarket[] {
  return cache;
}
