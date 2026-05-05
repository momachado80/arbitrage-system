/**
 * Discovery read-only sobre endpoints públicos Gamma + CLOB.
 * Sem ordem, carteira, credenciais de assinatura ou chave privada. Sem persistir payloads completos —
 * apenas resumo truncado como JSON único em stdout.
 */

import {
  enrichDiscoverySuitableRow,
  finalizeDiscoveryRankingWithUniverseQuality,
} from "../lib/liveMarketDiscoveryRanking";
import type { ClobBookStructureHint } from "../lib/marketSuitabilityGate";
import { evaluateMarketSuitability } from "../lib/marketSuitabilityGate";

const GAMMA_LIST = "https://gamma-api.polymarket.com/markets";
const CLOB_BOOK = "https://clob.polymarket.com/book";
const PAGE_LIMIT = 20;
const MAX_PAGES = 4;
const GAMMA_HTTP_MS = 12_000;
const BOOK_HTTP_MS = 5_000;
const Q_TRUNC = 200;

interface BookProbe {
  bookStructure?: ClobBookStructureHint;
  bestBidBook: number | null;
  bestAskBook: number | null;
  yesTokenPrefix: string | null;
}

function truncStr(s: string, n: number): string {
  if (s.length <= n) return s;
  return `${s.slice(0, n)}…`;
}

function jsonStringArray(value: unknown): string[] | null {
  if (Array.isArray(value)) return value.map(v => String(v)).filter(Boolean);
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const p = JSON.parse(value) as unknown;
    return Array.isArray(p) ? p.map(v => String(v)).filter(Boolean) : null;
  } catch {
    return null;
  }
}

function pickYesToken(row: Record<string, unknown>): string | null {
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

function bookStructure(raw: unknown): ClobBookStructureHint {
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

function topPricesFromBook(raw: unknown): { bid: number | null; ask: number | null } {
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

async function fetchJson(url: string, ms: number): Promise<unknown> {
  const res = await fetch(url, {
    signal: AbortSignal.timeout(ms),
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`http_${res.status}`);
  return (await res.json()) as unknown;
}

async function probeBook(tokenId: string): Promise<BookProbe> {
  const url = `${CLOB_BOOK}?token_id=${encodeURIComponent(tokenId)}`;
  try {
    const raw = await fetchJson(url, BOOK_HTTP_MS);
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

async function loadGammaRows(): Promise<Record<string, unknown>[]> {
  const rows: Record<string, unknown>[] = [];
  for (let p = 0; p < MAX_PAGES; p++) {
    const url = `${GAMMA_LIST}?active=true&closed=false&limit=${PAGE_LIMIT}&offset=${p * PAGE_LIMIT}`;
    const body = await fetchJson(url, GAMMA_HTTP_MS);
    if (!Array.isArray(body)) break;
    for (const item of body) {
      if (item && typeof item === "object" && !Array.isArray(item)) {
        rows.push(item as Record<string, unknown>);
      }
    }
    if (body.length < PAGE_LIMIT) break;
  }
  return rows;
}

async function main(): Promise<void> {
  const nowIso = new Date().toISOString();
  const rows = await loadGammaRows();
  let stdoutFieldsTruncated = false;

  const rowsOut: Array<{
    summary: Record<string, unknown>;
    evaluation: ReturnType<typeof evaluateMarketSuitability>;
  }> = [];

  const verdictCounts: Record<string, number> = {};

  const markTruncate = (full: string, max: number) => full.length > max;

  for (const r of rows) {
    const tok = pickYesToken(r);
    const book = tok ? await probeBook(tok) : { bookStructure: "skipped" as const, bestBidBook: null, bestAskBook: null, yesTokenPrefix: null };

    const gh = gammaBestHints(r);
    const bidComb = book.bestBidBook ?? numLike(gh.bid);
    const askComb = book.bestAskBook ?? numLike(gh.ask);

    const mid =
      typeof r.id === "string" || typeof r.id === "number" ? String(r.id) : null;
    const qRaw = typeof r.question === "string" ? r.question : typeof r.title === "string" ? r.title : null;

    let questionOut: string | null = null;
    if (qRaw) {
      if (markTruncate(qRaw, Q_TRUNC)) stdoutFieldsTruncated = true;
      questionOut = truncStr(qRaw, Q_TRUNC);
    }

    const slugSrc =
      typeof r.slug === "string" ? r.slug : typeof r.market_slug === "string" ? (r.market_slug as string) : null;
    let slugOut: string | null = null;
    if (slugSrc) {
      if (markTruncate(slugSrc, 120)) stdoutFieldsTruncated = true;
      slugOut = truncStr(slugSrc, 120);
    }

    let outcomePricesField = r.outcomePrices ?? null;
    if (typeof r.outcomePrices === "string" && markTruncate(r.outcomePrices, 160)) stdoutFieldsTruncated = true;
    if (typeof r.outcomePrices === "string") outcomePricesField = truncStr(r.outcomePrices, 160);

    let clobTokField = r.clobTokenIds ?? null;
    if (typeof r.clobTokenIds === "string" && markTruncate(r.clobTokenIds, 220)) stdoutFieldsTruncated = true;
    if (typeof r.clobTokenIds === "string") clobTokField = truncStr(r.clobTokenIds, 220);

    let tagsField = r.tags != null ? String(r.tags) : null;
    if (tagsField && markTruncate(tagsField, 120)) stdoutFieldsTruncated = true;
    if (tagsField) tagsField = truncStr(tagsField, 120);

    if (tok && tok.length > 18) stdoutFieldsTruncated = true;

    const summary = {
      id: mid,
      question: questionOut,
      slug: slugOut,
      active: r.active ?? null,
      closed: r.closed ?? null,
      resolved: r.resolved ?? r.isResolved ?? r.marketResolved ?? null,
      endDate: r.endDate ?? r.end_date ?? null,
      volume: r.volume ?? null,
      liquidity: r.liquidity ?? null,
      outcomePrices: outcomePricesField,
      clobTokenIds: clobTokField,
      conditionId: r.conditionId ?? r.condition_id ?? null,
      updatedAt: r.updatedAt ?? r.updated_at ?? null,
      category: r.category ?? null,
      tags: tagsField,
      yesTokenPrefix: book.yesTokenPrefix,
      clobBookStructure: book.bookStructure,
      bestBidUsed: bidComb,
      bestAskUsed: askComb,
    };

    const evalRes = evaluateMarketSuitability({
      marketId: mid ?? undefined,
      question: typeof r.question === "string" ? r.question : undefined,
      title: typeof r.title === "string" ? r.title : undefined,
      endDate: (r.endDate ?? r.end_date ?? null) as string | null,
      closed: Boolean(r.closed),
      resolved:
        typeof r.resolved === "boolean"
          ? r.resolved
          : typeof r.isResolved === "boolean"
            ? r.isResolved
            : typeof r.marketResolved === "boolean"
              ? r.marketResolved
              : undefined,
      active: typeof r.active === "boolean" ? r.active : undefined,
      bestBid: bidComb ?? undefined,
      bestAsk: askComb ?? undefined,
      volume: numLike(r.volume ?? null) ?? undefined,
      liquidity: numLike(r.liquidity ?? null) ?? undefined,
      conditionId: typeof r.conditionId === "string" ? r.conditionId : null,
      updatedAt: typeof r.updatedAt === "string" ? r.updatedAt : undefined,
      category: typeof r.category === "string" ? r.category : undefined,
      tags: r.tags,
      clobBookStructure: book.bookStructure === "skipped" ? undefined : book.bookStructure,
      nowIso,
    });

    verdictCounts[evalRes.suitabilityVerdict] = (verdictCounts[evalRes.suitabilityVerdict] ?? 0) + 1;
    rowsOut.push({ summary, evaluation: evalRes });
  }

  const suitable = rowsOut.filter(x => x.evaluation.canUseForPaperShadowCandidate);
  const enrichedSuitable = suitable.map(x =>
    enrichDiscoverySuitableRow({
      ...(x.summary as Record<string, unknown>),
      suitabilityVerdict: x.evaluation.suitabilityVerdict,
      reasons: x.evaluation.reasons,
    }),
  );
  const {
    candidatesSorted,
    topCandidates,
    topCleanCandidates,
    rejectedByUniverseQuality,
    universeQualityRejectionReasons,
  } = finalizeDiscoveryRankingWithUniverseQuality(enrichedSuitable, nowIso);

  const rejected = rowsOut
    .filter(x => !x.evaluation.canUseForPaperShadowCandidate)
    .map(x => ({
      id: x.summary.id,
      question: x.summary.question,
      slug: x.summary.slug,
      suitabilityVerdict: x.evaluation.suitabilityVerdict,
      reasons: x.evaluation.reasons,
    }));

  const topRejectionReasons = Object.entries(verdictCounts)
    .filter(([k]) => k !== "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION")
    .map(([verdict, count]) => ({ verdict, count }))
    .sort((a, b) => b.count - a.count);

  const baseReport = {
    scannedAtUtc: nowIso,
    totalMarketsScanned: rows.length,
    suitableForPaperShadowCount: suitable.length,
    rejectedCount: rows.length - suitable.length,
    topRejectionReasons,
    candidates: candidatesSorted,
    topCandidates,
    topCleanCandidates,
    rejectedByUniverseQuality,
    universeQualityRejectionReasons,
    rejected,
    topCandidatesCount: topCandidates.length,
    topCleanCandidatesCount: topCleanCandidates.length,
    note: "read_only_discovery_no_execution",
    ...(stdoutFieldsTruncated
      ? {
          truncationNotice:
            "stdout_fields_truncated: question_slug_outcomePrices_clobTokenIds_tags_yesTokenPrefix_capped_for_safe_json;",
        }
      : {}),
  };

  process.stdout.write(`${JSON.stringify(baseReport, null, 2)}\n`);
}

main().catch(err => {
  process.stderr.write(`[discoverLiveMarketSuitabilityCandidates] fatal: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exitCode = 1;
});
