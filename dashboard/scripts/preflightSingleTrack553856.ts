/**
 * Pré-flight read-only do mercado 553856 (Thunder NBA Finals).
 * Apenas GET públicos Gamma + CLOB. Sem worker, watcher, ordem nem chaves privadas.
 */

import type { ClobBookStructureHint } from "../lib/marketSuitabilityGate";
import { evaluateMarketSuitability } from "../lib/marketSuitabilityGate";

export const SINGLE_TRACK_553856_MARKET_ID = "553856";

const GAMMA_BASE = "https://gamma-api.polymarket.com";
const CLOB_BOOK = "https://clob.polymarket.com/book";
const HTTP_GAMMA_MS = 12_000;
const HTTP_BOOK_MS = 5_000;

const Q_TRUNC = 220;
const SLUG_TRUNC = 120;

export type PreflightVerdict553856 =
  | "READY_FOR_PAPER_SHADOW_OBSERVATION"
  | "NOT_READY_FOR_PAPER_SHADOW_OBSERVATION";

export interface GammaYesBookProbe {
  bookStructure: ClobBookStructureHint;
  bestBidBook: number | null;
  bestAskBook: number | null;
  yesTokenPrefix: string | null;
}

export interface Preflight553856Payload {
  scannedAtUtc: string;
  marketId: string;
  gammaRequestPath: string;
  preflightVerdict: PreflightVerdict553856;
  reasons: string[];
  canUseForMicrocapitalCandidate: false;
  suitability: ReturnType<typeof evaluateMarketSuitability>;
  bookSummary: {
    bookStructure: ClobBookStructureHint;
    bestBidBook: number | null;
    bestAskBook: number | null;
    yesTokenPrefix: string | null;
  };
  tokenSummary: {
    yesOutcomeTokenChosen: boolean;
    inspectedTokenPrefixes: string[];
  };
  gammaSummary: {
    id: string | null;
    question: string | null;
    slug: string | null;
    active: unknown;
    closed: unknown;
    resolved: unknown;
    endDate: unknown;
    volume: unknown;
    liquidity: unknown;
    conditionId: unknown;
    clobTokenIds: unknown;
    outcomePrices: unknown;
  };
}

function truncStr(s: string, n: number): string {
  if (s.length <= n) return s;
  return `${s.slice(0, n)}…`;
}

/** URL Gamma usando path REST — nunca `?id=` legacy. */
export function buildGamma553856MarketPreflightUrl(): string {
  return `${GAMMA_BASE}/markets/${encodeURIComponent(SINGLE_TRACK_553856_MARKET_ID)}`;
}

export function jsonStringArray(value: unknown): string[] | null {
  if (Array.isArray(value)) return value.map(v => String(v)).filter(Boolean);
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const p = JSON.parse(value) as unknown;
    return Array.isArray(p) ? p.map(v => String(v)).filter(Boolean) : null;
  } catch {
    return null;
  }
}

export function pickYesToken(row: Record<string, unknown>): string | null {
  const outs = jsonStringArray(row.outcomes);
  const toks = jsonStringArray(row.clobTokenIds);
  if (!toks?.length) return null;
  if (outs?.length) {
    const idx = outs.findIndex(o => /^yes$/i.test(String(o).trim()));
    if (idx >= 0 && idx < toks.length) return toks[idx]!;
  }
  return toks[0] ?? null;
}

function numLike(x: unknown): number | null {
  if (typeof x === "number" && Number.isFinite(x)) return x;
  if (typeof x === "string" && x.trim()) {
    const n = parseFloat(x);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function gammaBestHints(row: Record<string, unknown>): { bid: unknown; ask: unknown } {
  return {
    bid: row.bestBid ?? row.best_bid ?? row.bestBidClob ?? null,
    ask: row.bestAsk ?? row.best_ask ?? row.bestAskClob ?? null,
  };
}

export function bookStructure(raw: unknown): ClobBookStructureHint {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return "unusable";
  const j = raw as Record<string, unknown>;
  const bids = Array.isArray(j.bids) ? j.bids : [];
  const asks = Array.isArray(j.asks) ? j.asks : [];
  if (bids.length > 0 && asks.length > 0) return "two_sided";
  if (bids.length === 0 && asks.length > 0) return "asks_only";
  if (bids.length > 0 && asks.length === 0) return "bids_only";
  if (!bids.length && !asks.length) return "empty";
  return "unusable";
}

export function topPricesFromBook(raw: unknown): { bid: number | null; ask: number | null } {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return { bid: null, ask: null };
  const j = raw as Record<string, unknown>;
  let bestBid: number | null = null;
  let bestAsk: number | null = null;
  for (const lvl of Array.isArray(j.bids) ? j.bids : []) {
    const p = numLike((lvl as Record<string, unknown>).price);
    if (p !== null) bestBid = bestBid === null ? p : Math.max(bestBid, p);
  }
  for (const lvl of Array.isArray(j.asks) ? j.asks : []) {
    const p = numLike((lvl as Record<string, unknown>).price);
    if (p !== null) bestAsk = bestAsk === null ? p : Math.min(bestAsk, p);
  }
  return { bid: bestBid, ask: bestAsk };
}

export function unwrapGamma553856MarketRow(body: unknown): Record<string, unknown> | null {
  const want = SINGLE_TRACK_553856_MARKET_ID;
  const sameId = (o: Record<string, unknown>): boolean =>
    o.id !== undefined && o.id !== null && String(o.id) === want;

  const probe = (o: unknown): Record<string, unknown> | null => {
    if (!o || typeof o !== "object" || Array.isArray(o)) return null;
    const row = o as Record<string, unknown>;
    return sameId(row) ? row : null;
  };

  const direct = probe(body);
  if (direct) return direct;

  if (body && typeof body === "object" && !Array.isArray(body)) {
    const root = body as Record<string, unknown>;
    for (const k of ["market", "markets", "data", "results"] as const) {
      const v = root[k];
      if (Array.isArray(v) && v[0]) {
        const r = probe(v[0]);
        if (r) return r;
      } else if (v && typeof v === "object" && !Array.isArray(v)) {
        const outer = v as Record<string, unknown>;
        const inner = probe(outer);
        if (inner) return inner;
        if (Array.isArray(outer.data)) {
          for (const it of outer.data) {
            const r = probe(it);
            if (r) return r;
          }
        }
      }
    }
  }
  if (Array.isArray(body) && body[0]) return probe(body[0]);
  return null;
}

function summarizeGammaSafe(r: Record<string, unknown>): Preflight553856Payload["gammaSummary"] {
  const qRaw =
    typeof r.question === "string" ? r.question : typeof r.title === "string" ? r.title : null;
  const slugRaw =
    typeof r.slug === "string"
      ? r.slug
      : typeof r.market_slug === "string"
        ? r.market_slug
        : null;
  let outcomePrices = r.outcomePrices ?? null;
  if (typeof outcomePrices === "string") outcomePrices = truncStr(outcomePrices, 160);
  let clobTok = r.clobTokenIds ?? null;
  if (typeof clobTok === "string") clobTok = truncStr(clobTok, 220);
  return {
    id: r.id != null ? String(r.id) : null,
    question: qRaw ? truncStr(qRaw, Q_TRUNC) : null,
    slug: slugRaw ? truncStr(slugRaw, SLUG_TRUNC) : null,
    active: r.active ?? null,
    closed: r.closed ?? null,
    resolved: r.resolved ?? r.isResolved ?? r.marketResolved ?? null,
    endDate: r.endDate ?? r.end_date ?? null,
    volume: r.volume ?? null,
    liquidity: r.liquidity ?? null,
    conditionId: r.conditionId ?? r.condition_id ?? null,
    clobTokenIds: clobTok,
    outcomePrices,
  };
}

export function coerceBool(x: unknown): boolean | undefined {
  if (x === undefined) return undefined;
  return typeof x === "boolean" ? x : undefined;
}

/** Núcleo puro utilizado pelos testes e pelo runtime GET. */
export function compute553856PreflightPayload(args: {
  nowIso: string;
  gammaRow: Record<string, unknown> | null;
  yesBookProbe: GammaYesBookProbe;
}): Preflight553856Payload {
  const scannedAtUtc = new Date(args.nowIso).toISOString();
  const yesTok = args.gammaRow ? pickYesToken(args.gammaRow) : null;

  let row = args.gammaRow;
  if (row && String(row.id) !== SINGLE_TRACK_553856_MARKET_ID) {
    row = null;
  }

  const reasons: string[] = [];

  if (!row) {
    reasons.push("gamma_row_missing_or_id_mismatch");
  }

  const toksParsed = row ? jsonStringArray(row.clobTokenIds) : null;
  if (!row || !toksParsed || toksParsed.length === 0) {
    reasons.push("missing_gamma_clobTokenIds");
  }
  if (!yesTok && row && toksParsed && toksParsed.length > 0) {
    reasons.push("could_not_pick_yes_token");
  }

  const gh = row ? gammaBestHints(row) : { bid: null, ask: null };
  const bidComb = yesTok ? args.yesBookProbe.bestBidBook ?? numLike(gh.bid) : null;
  const askComb = yesTok ? args.yesBookProbe.bestAskBook ?? numLike(gh.ask) : null;

  const mid = row && row.id != null ? String(row.id) : SINGLE_TRACK_553856_MARKET_ID;

  let evalStruct: ClobBookStructureHint | undefined = args.yesBookProbe.bookStructure;
  if (!yesTok) evalStruct = "skipped";

  const suitability = evaluateMarketSuitability({
    marketId: mid,
    question: typeof row?.question === "string" ? row.question : undefined,
    title: typeof row?.title === "string" ? row.title : undefined,
    endDate: (row?.endDate ?? row?.end_date ?? null) as string | null,
    closed: coerceBool(row?.closed) ?? false,
    resolved:
      coerceBool(row?.resolved) ??
      coerceBool(row?.isResolved) ??
      coerceBool(row?.marketResolved) ??
      false,
    active: coerceBool(row?.active),
    bestBid: bidComb ?? undefined,
    bestAsk: askComb ?? undefined,
    volume: numLike(row?.volume ?? null) ?? undefined,
    liquidity: numLike(row?.liquidity ?? null) ?? undefined,
    conditionId: typeof row?.conditionId === "string" ? row.conditionId : null,
    updatedAt:
      typeof row?.updatedAt === "string"
        ? row.updatedAt
        : typeof row?.updated_at === "string"
          ? row.updated_at
          : undefined,
    category: typeof row?.category === "string" ? row.category : undefined,
    tags: row?.tags,
    clobBookStructure: evalStruct,
    nowIso: args.nowIso,
  });

  let bookStructuralOk =
    !!yesTok &&
    args.yesBookProbe.bookStructure === "two_sided" &&
    args.yesBookProbe.bestBidBook !== null &&
    args.yesBookProbe.bestAskBook !== null &&
    Number.isFinite(args.yesBookProbe.bestBidBook) &&
    Number.isFinite(args.yesBookProbe.bestAskBook);

  if (!yesTok || !toksParsed?.length) {
    reasons.push("book_probe_skipped_or_token_missing");
    bookStructuralOk = false;
  } else if (args.yesBookProbe.bookStructure !== "two_sided") {
    reasons.push(`clob_book_structure_not_two_sided:${args.yesBookProbe.bookStructure}`);
    bookStructuralOk = false;
  } else if (
    args.yesBookProbe.bestBidBook === null ||
    args.yesBookProbe.bestAskBook === null ||
    !Number.isFinite(args.yesBookProbe.bestBidBook) ||
    !Number.isFinite(args.yesBookProbe.bestAskBook)
  ) {
    reasons.push("clob_book_best_prices_missing");
    bookStructuralOk = false;
  }

  const preflightReady =
    Boolean(row) &&
    String(row?.id) === SINGLE_TRACK_553856_MARKET_ID &&
    bookStructuralOk &&
    suitability.canUseForPaperShadowCandidate === true &&
    suitability.suitabilityVerdict === "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION";

  const preflightVerdict: PreflightVerdict553856 = preflightReady
    ? "READY_FOR_PAPER_SHADOW_OBSERVATION"
    : "NOT_READY_FOR_PAPER_SHADOW_OBSERVATION";

  const reasonsOut: string[] = [];
  if (preflightReady) {
    reasonsOut.push("preflight_readonly_ready");
    for (const r of suitability.reasons) {
      if (!reasonsOut.includes(r)) reasonsOut.push(r);
    }
  } else {
    for (const r of reasons) {
      if (!reasonsOut.includes(r)) reasonsOut.push(r);
    }
    reasonsOut.push(`gate_verdict:${suitability.suitabilityVerdict}`);
    for (const r of suitability.reasons) {
      const tagged = `suitability_reason:${r}`;
      if (!reasonsOut.includes(tagged)) reasonsOut.push(tagged);
    }
    if (reasonsOut.length === 1 && reasonsOut[0]!.startsWith("gate_verdict:")) {
      reasonsOut.unshift("preflight_not_ready_no_additional_local_flags");
    }
  }

  const base = {
    scannedAtUtc,
    marketId: SINGLE_TRACK_553856_MARKET_ID,
    gammaRequestPath: `/markets/${SINGLE_TRACK_553856_MARKET_ID}`,
    preflightVerdict,
    reasons: reasonsOut,
    canUseForMicrocapitalCandidate: false as const,
    suitability,
    bookSummary: {
      bookStructure: args.yesBookProbe.bookStructure,
      bestBidBook: args.yesBookProbe.bestBidBook,
      bestAskBook: args.yesBookProbe.bestAskBook,
      yesTokenPrefix: args.yesBookProbe.yesTokenPrefix,
    },
    tokenSummary: {
      yesOutcomeTokenChosen: Boolean(yesTok),
      inspectedTokenPrefixes: yesTok ? [truncStr(yesTok, 18)] : [],
    },
    gammaSummary: row ? summarizeGammaSafe(row) : summarizeGammaSafe({ id: null }),
  };

  return base;
}

async function fetchJson(url: string, ms: number): Promise<unknown> {
  const res = await fetch(url, {
    signal: AbortSignal.timeout(ms),
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`http_${res.status}`);
  return (await res.json()) as unknown;
}

async function probeYesBook(tokenId: string): Promise<GammaYesBookProbe> {
  const url = `${CLOB_BOOK}?token_id=${encodeURIComponent(tokenId)}`;
  try {
    const raw = await fetchJson(url, HTTP_BOOK_MS);
    const { bid, ask } = topPricesFromBook(raw);
    return {
      bookStructure: bookStructure(raw),
      bestBidBook: bid,
      bestAskBook: ask,
      yesTokenPrefix: truncStr(tokenId, 18),
    };
  } catch {
    return { bookStructure: "unknown", bestBidBook: null, bestAskBook: null, yesTokenPrefix: truncStr(tokenId, 18) };
  }
}

async function main(): Promise<void> {
  const nowIso = new Date().toISOString();
  const gammaUrl = buildGamma553856MarketPreflightUrl();
  const gammaBody = await fetchJson(gammaUrl, HTTP_GAMMA_MS);
  const row = unwrapGamma553856MarketRow(gammaBody);
  let yesProbe: GammaYesBookProbe = {
    bookStructure: "skipped",
    bestBidBook: null,
    bestAskBook: null,
    yesTokenPrefix: null,
  };
  if (row && pickYesToken(row)) {
    const tok = pickYesToken(row)!;
    yesProbe = await probeYesBook(tok);
  }

  const payload = compute553856PreflightPayload({ nowIso, gammaRow: row, yesBookProbe: yesProbe });

  process.stdout.write(
    `${JSON.stringify(
      {
        ...payload,
        gammaUrlUsedPreview: truncStr(gammaUrl, 80),
      },
      null,
      2,
    )}\n`,
  );
}

if (require.main === module) {
  main().catch(err => {
    process.stderr.write(
      `[preflightSingleTrack553856] fatal: ${err instanceof Error ? err.message : String(err)}\n`,
    );
    process.exitCode = 1;
  });
}
