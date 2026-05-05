/**
 * Scout read-only: descobre janelas observáveis em candidatos topo do discovery Gamma/CLOB.
 * Sem worker, watcher, execução real ou escrita `.paper`.
 */

import fs from "fs";
import path from "path";

import {
  enrichDiscoverySuitableRow,
  finalizeDiscoveryRankingWithUniverseQuality,
} from "../lib/liveMarketDiscoveryRanking";
import {
  evaluateObservationWindow,
  type ObservationWindowEvaluateResult,
  type ObservationWindowSnapshotInput,
  type ObservationWindowVerdict,
} from "../lib/observationWindowSuitability";
import type { ClobBookStructureHint } from "../lib/marketSuitabilityGate";
import { evaluateMarketSuitability, nearPinnedPrice, parsedEndUtc, parsedNowUtc } from "../lib/marketSuitabilityGate";
import { snapshotRowFromPrices } from "../lib/singleTrackMarkoutSampler";

const GAMMA_LIST = "https://gamma-api.polymarket.com/markets";
const CLOB_BOOK = "https://clob.polymarket.com/book";
const PAGE_LIMIT = 20;
const MAX_PAGES = 4;
const GAMMA_HTTP_MS = 12_000;
const BOOK_HTTP_MS = 5_000;
const REPO_ABS = path.resolve(__dirname, "..", "..");

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

function bookStructureHint(raw: unknown): ClobBookStructureHint {
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

async function probeBook(tokenId: string): Promise<{
  bookStructure: ClobBookStructureHint;
  bestBidBook: number | null;
  bestAskBook: number | null;
  yesTokenPrefix: string | null;
}> {
  const url = `${CLOB_BOOK}?token_id=${encodeURIComponent(tokenId)}`;
  try {
    const raw = await fetchJson(url, BOOK_HTTP_MS);
    const { bid, ask } = topPricesFromBook(raw);
    return {
      bookStructure: bookStructureHint(raw),
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

function delayMs(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

function assertOutPath(absPath: string): void {
  const norm = path.resolve(absPath);
  if (norm.includes(`${path.sep}.paper${path.sep}`) || norm.endsWith(`${path.sep}.paper`)) {
    throw new Error("output_path_blocked:.paper");
  }
  const inside =
    norm === REPO_ABS || norm.startsWith(REPO_ABS + path.sep);
  const tmpOk =
    norm === "/tmp" ||
    norm.startsWith("/tmp/") ||
    norm === "/private/tmp" ||
    norm.startsWith("/private/tmp/");
  if (!(tmpOk || !inside)) throw new Error("output_must_be_tmp_or_outside_repo");
}

type UniverseMode = "clean" | "all";

function parseCli(argv: string[]): {
  limit: number;
  snapshots: number;
  gapSeconds: number;
  outPath?: string;
  universeMode: UniverseMode;
} {
  let limit = 10;
  let snapshots = 4;
  let gapSeconds = 30;
  let outPath: string | undefined;
  let universeMode: UniverseMode = "clean";
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === "--limit" && argv[i + 1]) limit = Math.max(1, parseInt(argv[++i]!, 10) || 10);
    else if (a.startsWith("--limit=")) limit = Math.max(1, parseInt(a.slice("--limit=".length), 10) || 10);
    else if (a === "--snapshots" && argv[i + 1])
      snapshots = Math.max(1, parseInt(argv[++i]!, 10) || 4);
    else if (a.startsWith("--snapshots="))
      snapshots = Math.max(1, parseInt(a.slice("--snapshots=".length), 10) || 4);
    else if (a === "--gap-seconds" && argv[i + 1])
      gapSeconds = Math.max(0, parseInt(argv[++i]!, 10) || 0);
    else if (a.startsWith("--gap-seconds="))
      gapSeconds = Math.max(0, parseInt(a.slice("--gap-seconds=".length), 10) || 0);
    else if (a === "--out" && argv[i + 1]) outPath = argv[++i];
    else if (a.startsWith("--out=")) outPath = a.slice("--out=".length);
    else if (a === "--universe" && argv[i + 1]) {
      const v = String(argv[++i]).toLowerCase();
      universeMode = v === "all" ? "all" : "clean";
    } else if (a.startsWith("--universe=")) {
      const v = a.slice("--universe=".length).toLowerCase();
      universeMode = v === "all" ? "all" : "clean";
    }
  }
  return { limit, snapshots, gapSeconds, outPath, universeMode };
}

interface RankRow {
  marketId: string;
  label: string | null;
  endDate: string | null;
  observationWindowVerdict: ObservationWindowVerdict;
  observationWindowScore: number;
  midRange: number | null;
  avgSpread: number | null;
  avgMid: number | null;
  twoSidedSnapshotCount: number;
  snapshotsCollected: number;
  midCentrality: number;
  canUseForPaperShadowObservation: boolean;
}

function sortRanked(a: RankRow, b: RankRow): number {
  const tier = (v: ObservationWindowVerdict) =>
    v === "WINDOW_INFORMATIVE" ? 0 : v === "WINDOW_QUIET_BUT_VALID" ? 1 : 2;
  const ta = tier(a.observationWindowVerdict);
  const tb = tier(b.observationWindowVerdict);
  if (ta !== tb) return ta - tb;
  if (b.observationWindowScore !== a.observationWindowScore) return b.observationWindowScore - a.observationWindowScore;
  const mra = a.midRange ?? 0;
  const mrb = b.midRange ?? 0;
  if (mrb !== mra) return mrb - mra;
  const sa = a.avgSpread ?? 1;
  const sb = b.avgSpread ?? 1;
  if (sa !== sb) return sa - sb;
  if (b.twoSidedSnapshotCount !== a.twoSidedSnapshotCount) return b.twoSidedSnapshotCount - a.twoSidedSnapshotCount;
  if (b.midCentrality !== a.midCentrality) return b.midCentrality - a.midCentrality;
  const da = a.endDate ? parsedEndUtc(a.endDate)?.getTime() ?? 0 : 0;
  const db = b.endDate ? parsedEndUtc(b.endDate)?.getTime() ?? 0 : 0;
  return db - da;
}

async function collectSnapshotsForToken(tokenId: string, count: number, gapSec: number): Promise<ObservationWindowSnapshotInput[]> {
  const out: ObservationWindowSnapshotInput[] = [];
  for (let i = 0; i < count; i++) {
    if (i > 0 && gapSec > 0) await delayMs(gapSec * 1000);
    const probe = await probeBook(tokenId);
    const row = snapshotRowFromPrices({
      bestBid: probe.bestBidBook,
      bestAsk: probe.bestAskBook,
      collectedAtIso: new Date().toISOString(),
    });
    out.push({
      collectedAtIso: row.collectedAtIso,
      bestBid: row.bestBid,
      bestAsk: row.bestAsk,
      mid: row.mid,
      spread: row.spread,
      bookType: row.bookType,
      priceSource: row.priceSource,
    });
  }
  return out;
}

async function main(): Promise<void> {
  const cli = parseCli(process.argv);
  const nowIso = new Date().toISOString();
  const rows = await loadGammaRows();

  type RowEval = { summary: Record<string, unknown>; evaluation: ReturnType<typeof evaluateMarketSuitability>; row: Record<string, unknown> };
  const rowsOut: RowEval[] = [];

  for (const r of rows) {
    const tok = pickYesToken(r);
    const book = tok
      ? await probeBook(tok)
      : { bookStructure: "skipped" as const, bestBidBook: null, bestAskBook: null, yesTokenPrefix: null };

    const gh = gammaBestHints(r);
    const bidComb = book.bestBidBook ?? numLike(gh.bid);
    const askComb = book.bestAskBook ?? numLike(gh.ask);

    const mid = typeof r.id === "string" || typeof r.id === "number" ? String(r.id) : null;
    const qRaw = typeof r.question === "string" ? r.question : typeof r.title === "string" ? r.title : null;
    const questionOut = qRaw ? truncStr(qRaw, 200) : null;
    const slugSrc =
      typeof r.slug === "string" ? r.slug : typeof r.market_slug === "string" ? (r.market_slug as string) : null;
    const slugOut = slugSrc ? truncStr(slugSrc, 120) : null;

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
      conditionId: r.conditionId ?? r.condition_id ?? null,
      category: r.category ?? null,
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
      clobBookStructure: book.bookStructure === "skipped" ? undefined : book.bookStructure,
      nowIso,
    });

    rowsOut.push({ summary, evaluation: evalRes, row: r });
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
    topCandidates,
    topCleanCandidates,
    rejectedByUniverseQuality,
    universeQualityRejectionReasons,
  } = finalizeDiscoveryRankingWithUniverseQuality(enrichedSuitable, nowIso);

  const universeMode = cli.universeMode;
  const sourcePool = universeMode === "clean" ? topCleanCandidates : topCandidates;
  const universeStatus =
    universeMode === "clean" && topCleanCandidates.length === 0
      ? "NO_CLEAN_OBSERVATION_UNIVERSE"
      : "OK";

  if (universeStatus === "NO_CLEAN_OBSERVATION_UNIVERSE") {
    const payloadSkip = {
      generatedAtUtc: nowIso,
      note: "read_only_observation_window_scout_no_execution",
      universeMode,
      universeStatus,
      discoveryScanTotalMarkets: rows.length,
      suitableForPaperShadowCount: suitable.length,
      topCandidatesCount: topCandidates.length,
      topCleanCandidatesCount: topCleanCandidates.length,
      universeQualityRejectionReasons,
      rejectedByUniverseQualityCount: rejectedByUniverseQuality.length,
      marketsScouted: 0,
      informativeWindows: 0,
      quietButValidWindows: 0,
      notReadyWindows: 0,
      rankedWindows: [] as RankRow[],
      topWindows: [] as RankRow[],
      details: [] as Array<Record<string, unknown>>,
    };
    const text = `${JSON.stringify(payloadSkip, null, 2)}\n`;
    process.stdout.write(text);
    if (cli.outPath) {
      assertOutPath(cli.outPath);
      await fs.promises.mkdir(path.dirname(path.resolve(cli.outPath)), { recursive: true });
      await fs.promises.writeFile(path.resolve(cli.outPath), text, "utf8");
    }
    return;
  }

  const picked = sourcePool.slice(0, cli.limit);
  let marketsScouted = 0;
  let informativeWindows = 0;
  let quietButValidWindows = 0;
  let notReadyWindows = 0;

  const rankedWindows: RankRow[] = [];
  const detailBlocks: Array<{
    marketId: string;
    label: string | null;
    suitabilityVerdict: string;
    observation: ObservationWindowEvaluateResult;
    snapshotsCollected: number;
    skippedReason?: string;
  }> = [];

  for (const cand of picked) {
    const id = cand.id != null ? String(cand.id) : null;
    if (!id) continue;
    marketsScouted++;

    const raw = rows.find(y => String(y.id) === id) ?? null;
    const label =
      (typeof cand.question === "string" ? cand.question : null) ??
      (typeof cand.slug === "string" ? cand.slug : null);

    const token = raw ? pickYesToken(raw) : null;
    if (!token || !raw) {
      const obs = evaluateObservationWindow({
        marketId: id,
        label,
        nowIso,
        endDate: (raw?.endDate ?? raw?.end_date ?? null) as string | null,
        active: raw?.active,
        closed: raw?.closed,
        resolved: raw?.resolved ?? raw?.isResolved,
        snapshots: [],
        suitabilityVerdict: String(cand.suitabilityVerdict ?? "UNSUITABLE_MISSING_DATA"),
      });
      detailBlocks.push({ marketId: id, label, suitabilityVerdict: String(cand.suitabilityVerdict), observation: obs, snapshotsCollected: 0, skippedReason: "missing_row_or_token" });
      if (obs.observationWindowVerdict === "WINDOW_INFORMATIVE") informativeWindows++;
      else if (obs.observationWindowVerdict === "WINDOW_QUIET_BUT_VALID") quietButValidWindows++;
      else notReadyWindows++;
      rankedWindows.push(rankEntry(id, label, (raw?.endDate ?? raw?.end_date ?? null) as string | null, obs, 0));
      continue;
    }

    const bookFirst = await probeBook(token);
    const now = parsedNowUtc(nowIso);
    const endDt = parsedEndUtc((raw.endDate ?? raw.end_date ?? null) as string | null);
    const expired = !!(endDt && now.getTime() > endDt.getTime());
    const closedB = raw.closed === true;
    const resolvedB =
      raw.resolved === true || raw.isResolved === true || raw.marketResolved === true;
    const activeB = raw.active !== false;

    let snapshots: ObservationWindowSnapshotInput[] = [];
    let skipReason: string | undefined;

    if (closedB || resolvedB || expired || !activeB) {
      skipReason = "market_closed_resolved_expired_or_inactive_precheck";
    } else if (bookFirst.bookStructure !== "two_sided" || bookFirst.bestBidBook === null || bookFirst.bestAskBook === null) {
      skipReason = "first_snapshot_not_two_sided_precheck";
      const row0 = snapshotRowFromPrices({
        bestBid: bookFirst.bestBidBook,
        bestAsk: bookFirst.bestAskBook,
        collectedAtIso: new Date().toISOString(),
      });
      snapshots = [
        {
          collectedAtIso: row0.collectedAtIso,
          bestBid: row0.bestBid,
          bestAsk: row0.bestAsk,
          mid: row0.mid,
          spread: row0.spread,
          bookType: row0.bookType,
          priceSource: row0.priceSource,
        },
      ];
    } else if (
      nearPinnedPrice(bookFirst.bestBidBook) ||
      nearPinnedPrice(bookFirst.bestAskBook)
    ) {
      skipReason = "price_pinned_precheck_first_quote";
      const rowp = snapshotRowFromPrices({
        bestBid: bookFirst.bestBidBook,
        bestAsk: bookFirst.bestAskBook,
        collectedAtIso: new Date().toISOString(),
      });
      snapshots = [
        {
          collectedAtIso: rowp.collectedAtIso,
          bestBid: rowp.bestBid,
          bestAsk: rowp.bestAsk,
          mid: rowp.mid,
          spread: rowp.spread,
          bookType: rowp.bookType,
          priceSource: rowp.priceSource,
        },
      ];
    } else {
      snapshots = await collectSnapshotsForToken(token, cli.snapshots, cli.gapSeconds);
    }

    const suitVerdict = String(cand.suitabilityVerdict ?? "UNSUITABLE_MISSING_DATA");
    const obs = evaluateObservationWindow({
      marketId: id,
      label,
      nowIso,
      endDate: (raw.endDate ?? raw.end_date ?? null) as string | null,
      active: raw.active,
      closed: raw.closed,
      resolved: raw.resolved ?? raw.isResolved,
      snapshots,
      suitabilityVerdict: suitVerdict,
    });

    detailBlocks.push({
      marketId: id,
      label,
      suitabilityVerdict: suitVerdict,
      observation: obs,
      snapshotsCollected: snapshots.length,
      skippedReason: skipReason,
    });

    if (obs.observationWindowVerdict === "WINDOW_INFORMATIVE") informativeWindows++;
    else if (obs.observationWindowVerdict === "WINDOW_QUIET_BUT_VALID") quietButValidWindows++;
    else notReadyWindows++;
    rankedWindows.push(
      rankEntry(id, label, (raw.endDate ?? raw.end_date ?? null) as string | null, obs, snapshots.length),
    );
  }

  rankedWindows.sort(sortRanked);

  const payload = {
    generatedAtUtc: nowIso,
    note: "read_only_observation_window_scout_no_execution",
    universeMode,
    universeStatus,
    discoveryScanTotalMarkets: rows.length,
    suitableForPaperShadowCount: suitable.length,
    topCandidatesCount: topCandidates.length,
    topCleanCandidatesCount: topCleanCandidates.length,
    universeQualityRejectionReasons,
    rejectedByUniverseQualityCount: rejectedByUniverseQuality.length,
    marketsScouted,
    informativeWindows,
    quietButValidWindows,
    notReadyWindows,
    rankedWindows,
    topWindows: rankedWindows.slice(0, 5),
    details: detailBlocks.map(d => ({
      marketId: d.marketId,
      label: d.label ? truncStr(String(d.label), 180) : null,
      suitabilityVerdict: d.suitabilityVerdict,
      skippedReason: d.skippedReason ?? null,
      snapshotsCollected: d.snapshotsCollected,
      observationWindowVerdict: d.observation.observationWindowVerdict,
      observationWindowScore: d.observation.observationWindowScore,
      isWindowInformative: d.observation.isWindowInformative,
      reasons: d.observation.reasons,
      midRange: d.observation.midRange,
      spreadRange: d.observation.spreadRange,
      canUseForMicrocapitalCandidate: false as const,
    })),
  };

  const text = `${JSON.stringify(payload, null, 2)}\n`;
  process.stdout.write(text);
  if (cli.outPath) {
    assertOutPath(cli.outPath);
    await fs.promises.mkdir(path.dirname(path.resolve(cli.outPath)), { recursive: true });
    await fs.promises.writeFile(path.resolve(cli.outPath), text, "utf8");
  }
}

function rankEntry(
  marketId: string,
  label: string | null,
  endDate: string | null,
  obs: ObservationWindowEvaluateResult,
  snapshotsCollected: number,
): RankRow {
  const centrality =
    obs.avgMid !== null && Number.isFinite(obs.avgMid) ? Math.min(obs.avgMid, 1 - obs.avgMid) : 0;
  return {
    marketId,
    label: label ? truncStr(String(label), 160) : null,
    endDate,
    observationWindowVerdict: obs.observationWindowVerdict,
    observationWindowScore: obs.observationWindowScore,
    midRange: obs.midRange,
    avgSpread: obs.avgSpread,
    avgMid: obs.avgMid,
    twoSidedSnapshotCount: obs.twoSidedSnapshotCount,
    snapshotsCollected,
    canUseForPaperShadowObservation: obs.canUseForPaperShadowObservation,
    midCentrality: centrality,
  };
}

if (require.main === module) {
  main().catch(err => {
    process.stderr.write(
      `[scoutObservationWindows] fatal: ${err instanceof Error ? err.message : String(err)}\n`,
    );
    process.exitCode = 1;
  });
}
