/**
 * Resolve agenda read-only de catalisadores para mercados limpos (Gamma + ESPN público).
 * Sem ordens, capital ou `.paper`. Sem persistir payload bruto completo da ESPN —
 * apenas campos truncados no JSON do plano.
 */

import fs from "fs";
import path from "path";

import {
  buildObservationWindowsForCatalyst,
  evaluateCatalystObservationReadiness,
  inferCatalystProfileFromMarket,
  type CatalystProfile,
  type ObservationWindowSlot,
  type ScheduleFetchStatus,
} from "../lib/catalystObservationSchedule";
import {
  enrichDiscoverySuitableRow,
  finalizeDiscoveryRankingWithUniverseQuality,
} from "../lib/liveMarketDiscoveryRanking";
import type { ClobBookStructureHint } from "../lib/marketSuitabilityGate";
import { evaluateMarketSuitability } from "../lib/marketSuitabilityGate";

const GAMMA_LIST = "https://gamma-api.polymarket.com/markets";
const CLOB_BOOK = "https://clob.polymarket.com/book";
const ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard";
const ESPN_NHL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard";
const PAGE_LIMIT = 20;
const MAX_PAGES = 4;
const GAMMA_HTTP_MS = 12_000;
const BOOK_HTTP_MS = 5_000;
const ESPN_HTTP_MS = 10_000;
const Q_TRUNC = 200;

const REPO_ABS = path.resolve(__dirname, "..", "..");

interface BookProbe {
  bookStructure?: ClobBookStructureHint;
  bestBidBook: number | null;
  bestAskBook: number | null;
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
    return { bookStructure: bookStructure(raw), bestBidBook: bid, bestAskBook: ask };
  } catch {
    return { bookStructure: "unknown", bestBidBook: null, bestAskBook: null };
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

function assertOutPath(absPath: string): void {
  const norm = path.resolve(absPath);
  if (norm.includes(`${path.sep}.paper${path.sep}`) || norm.endsWith(`${path.sep}.paper`)) {
    throw new Error("output_path_blocked:.paper");
  }
  const inside = norm === REPO_ABS || norm.startsWith(REPO_ABS + path.sep);
  const tmpOk =
    norm === "/tmp" ||
    norm.startsWith("/tmp/") ||
    norm === "/private/tmp" ||
    norm.startsWith("/private/tmp/");
  if (!(tmpOk || !inside)) throw new Error("output_must_be_tmp_or_outside_repo");
}

function parseCli(argv: string[]): { limit: number; horizonHours: number; outPath: string } {
  let limit = 8;
  let horizonHours = 72;
  let outPath = "/tmp/catalyst-observation-plan.json";
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === "--limit" && argv[i + 1]) limit = Math.max(1, parseInt(argv[++i]!, 10) || 8);
    else if (a.startsWith("--limit=")) limit = Math.max(1, parseInt(a.slice("--limit=".length), 10) || 8);
    else if (a === "--horizon-hours" && argv[i + 1])
      horizonHours = Math.max(1, parseInt(argv[++i]!, 10) || 72);
    else if (a.startsWith("--horizon-hours="))
      horizonHours = Math.max(1, parseInt(a.slice("--horizon-hours=".length), 10) || 72);
    else if (a === "--out" && argv[i + 1]) outPath = argv[++i]!;
    else if (a.startsWith("--out=")) outPath = a.slice("--out=".length);
  }
  return { limit, horizonHours, outPath };
}

function formatEspnDate(d: Date): string {
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}${m}${day}`;
}

function normalizeTeamLabel(s: string): string {
  return s.toLowerCase().replace(/^the\s+/i, "").trim();
}

function tokenHits(subject: string, displayName: string): number {
  const dn = normalizeTeamLabel(displayName);
  const parts = normalizeTeamLabel(subject)
    .split(/\s+/)
    .filter(p => p.length >= 3);
  if (parts.length === 0) return dn.includes(normalizeTeamLabel(subject)) ? 1 : 0;
  let hits = 0;
  for (const p of parts) if (dn.includes(p)) hits++;
  return hits;
}

function teamRoughMatch(subject: string, displayName: string): boolean {
  const hits = tokenHits(subject, displayName);
  const parts = normalizeTeamLabel(subject)
    .split(/\s+/)
    .filter(p => p.length >= 3);
  const need = Math.min(2, Math.max(1, parts.length));
  return hits >= need;
}

interface NextGamePick {
  eventName: string;
  eventStartUtc: string;
  opponentShort: string | null;
}

function extractCompetitors(comp: Record<string, unknown>): Array<{ name: string }> {
  const out: Array<{ name: string }> = [];
  const comps = Array.isArray(comp.competitors) ? comp.competitors : [];
  for (const c of comps) {
    if (!c || typeof c !== "object") continue;
    const team = (c as Record<string, unknown>).team as Record<string, unknown> | undefined;
    const dn = team?.displayName ?? team?.shortDisplayName ?? team?.abbreviation;
    if (typeof dn === "string" && dn.trim()) out.push({ name: dn.trim() });
  }
  return out;
}

function parseEspnScoreboardForTeam(raw: unknown, subject: string): NextGamePick | null {
  if (!raw || typeof raw !== "object") return null;
  const events = Array.isArray((raw as Record<string, unknown>).events)
    ? ((raw as Record<string, unknown>).events as unknown[])
    : [];
  let best: NextGamePick | null = null;
  let bestT = Infinity;
  for (const ev of events) {
    if (!ev || typeof ev !== "object") continue;
    const e = ev as Record<string, unknown>;
    const comps = Array.isArray(e.competitions) ? e.competitions : [];
    const comp0 = comps[0];
    if (!comp0 || typeof comp0 !== "object") continue;
    const comp = comp0 as Record<string, unknown>;
    const teams = extractCompetitors(comp);
    const matched = teams.filter(t => teamRoughMatch(subject, t.name));
    if (matched.length === 0) continue;
    const dateStr =
      (typeof comp.date === "string" && comp.date) ||
      (typeof e.date === "string" && e.date) ||
      null;
    if (!dateStr) continue;
    const dt = new Date(dateStr);
    if (!Number.isFinite(dt.getTime())) continue;
    if (dt.getTime() < bestT) {
      bestT = dt.getTime();
      const opp = teams.find(t => !teamRoughMatch(subject, t.name));
      best = {
        eventName: truncStr(
          teams.map(t => t.name).join(" vs "),
          120,
        ),
        eventStartUtc: dt.toISOString(),
        opponentShort: opp ? truncStr(opp.name, 60) : null,
      };
    }
  }
  return best;
}

async function findNextEspnGame(
  league: "NBA" | "NHL",
  subject: string,
  now: Date,
  horizonEnd: Date,
): Promise<{ pick: NextGamePick | null; status: ScheduleFetchStatus }> {
  const base = league === "NBA" ? ESPN_NBA : ESPN_NHL;
  const cache = new Map<string, unknown>();
  let best: NextGamePick | null = null;
  let bestT = Infinity;
  try {
    const dayMs = 86_400_000;
    for (let t = now.getTime(); t <= horizonEnd.getTime() + dayMs; t += dayMs) {
      const d = new Date(t);
      const ds = formatEspnDate(d);
      if (!cache.has(ds)) {
        const url = `${base}?dates=${encodeURIComponent(ds)}`;
        cache.set(ds, await fetchJson(url, ESPN_HTTP_MS));
      }
      const raw = cache.get(ds);
      const pick = parseEspnScoreboardForTeam(raw, subject);
      if (pick) {
        const ts = new Date(pick.eventStartUtc).getTime();
        if (ts >= now.getTime() && ts <= horizonEnd.getTime() && ts < bestT) {
          bestT = ts;
          best = pick;
        }
      }
    }
    return { pick: best, status: "ok" };
  } catch {
    return { pick: null, status: "failed" };
  }
}

async function main(): Promise<void> {
  const cli = parseCli(process.argv);
  const nowIso = new Date().toISOString();
  const now = new Date(nowIso);
  const horizonEnd = new Date(now.getTime() + cli.horizonHours * 60 * 60 * 1000);

  const rows = await loadGammaRows();
  const rowsOut: Array<{
    summary: Record<string, unknown>;
    evaluation: ReturnType<typeof evaluateMarketSuitability>;
    raw: Record<string, unknown>;
  }> = [];

  for (const r of rows) {
    const tok = pickYesToken(r);
    const book = tok ? await probeBook(tok) : { bookStructure: "skipped" as const, bestBidBook: null, bestAskBook: null };
    const gh = gammaBestHints(r);
    const bidComb = book.bestBidBook ?? numLike(gh.bid);
    const askComb = book.bestAskBook ?? numLike(gh.ask);
    const mid = typeof r.id === "string" || typeof r.id === "number" ? String(r.id) : null;
    const qRaw = typeof r.question === "string" ? r.question : typeof r.title === "string" ? r.title : null;
    const questionOut = qRaw ? truncStr(qRaw, Q_TRUNC) : null;
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
      category: r.category ?? null,
      tags: r.tags != null ? truncStr(String(r.tags), 120) : null,
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

    rowsOut.push({ summary, evaluation: evalRes, raw: r });
  }

  const suitable = rowsOut.filter(x => x.evaluation.canUseForPaperShadowCandidate);
  const enrichedSuitable = suitable.map(x =>
    enrichDiscoverySuitableRow({
      ...(x.summary as Record<string, unknown>),
      suitabilityVerdict: x.evaluation.suitabilityVerdict,
      reasons: x.evaluation.reasons,
    }),
  );
  const { topCleanCandidates } = finalizeDiscoveryRankingWithUniverseQuality(enrichedSuitable, nowIso);
  const picked = topCleanCandidates.slice(0, cli.limit);

  let marketsWithNearCatalyst = 0;
  let marketsWithoutNearCatalyst = 0;
  let marketsNeedingScheduleSource = 0;

  const plan: Array<{
    marketId: string;
    label: string | null;
    catalystType: CatalystProfile["catalystType"];
    subject: string;
    league: CatalystProfile["league"];
    catalystPriority: CatalystProfile["catalystPriority"];
    catalystReadinessVerdict: string;
    nextEvent: {
      eventName: string | null;
      eventStartUtc: string | null;
      opponentShort: string | null;
      source: string;
      confidence: string;
    } | null;
    observationWindows: ObservationWindowSlot[];
    nextObservationWindowUtc: string | null;
    canUseForMicrocapitalCandidate: false;
    profileReasons: string[];
    readinessReasons: string[];
  }> = [];

  for (const row of picked) {
    const id = row.id != null ? String(row.id) : "";
    const rawQ =
      typeof row.question === "string"
        ? row.question
        : typeof row.slug === "string"
          ? row.slug
          : "";
    const profile = inferCatalystProfileFromMarket({
      marketId: id,
      question: rawQ,
      slug: typeof row.slug === "string" ? row.slug : null,
      category: typeof row.category === "string" ? row.category : null,
      tags: row.tags,
      endDate: typeof row.endDate === "string" ? row.endDate : null,
      nowIso,
    });

    let scheduleStatus: ScheduleFetchStatus = "not_applicable";
    let nextStart: string | null = null;
    let nextMeta: NextGamePick | null = null;

    if (profile.league === "NBA" || profile.league === "NHL") {
      const { pick, status } = await findNextEspnGame(profile.league, profile.subject, now, horizonEnd);
      scheduleStatus = status;
      nextMeta = pick;
      nextStart = pick?.eventStartUtc ?? null;
    }

    const readiness = evaluateCatalystObservationReadiness({
      nowIso,
      horizonHours: cli.horizonHours,
      profile,
      nextEventStartUtc: nextStart,
      scheduleFetchStatus: scheduleStatus,
    });

    let observationWindows: ObservationWindowSlot[] = [];
    if (nextStart && readiness.catalystReadinessVerdict === "HAS_NEAR_CATALYST") {
      observationWindows = buildObservationWindowsForCatalyst({
        marketId: id,
        label: rawQ ? truncStr(rawQ, 180) : null,
        catalystType: profile.catalystType,
        eventName: nextMeta?.eventName ?? "espn_scoreboard_match",
        eventStartUtc: nextStart,
        source: "espn_public_scoreboard",
        confidence: "medium",
      });
    }

    if (readiness.catalystReadinessVerdict === "HAS_NEAR_CATALYST") marketsWithNearCatalyst++;
    else if (readiness.catalystReadinessVerdict === "NEEDS_SCHEDULE_SOURCE") marketsNeedingScheduleSource++;
    else marketsWithoutNearCatalyst++;

    plan.push({
      marketId: id,
      label: rawQ ? truncStr(rawQ, 180) : null,
      catalystType: profile.catalystType,
      subject: truncStr(profile.subject, 120),
      league: profile.league,
      catalystPriority: profile.catalystPriority,
      catalystReadinessVerdict: readiness.catalystReadinessVerdict,
      nextEvent:
        nextMeta && nextStart
          ? {
              eventName: nextMeta.eventName,
              eventStartUtc: nextStart,
              opponentShort: nextMeta.opponentShort,
              source: "espn_public_scoreboard",
              confidence: "medium",
            }
          : profile.league === "FIFA"
            ? null
            : scheduleStatus === "failed"
              ? null
              : null,
      observationWindows,
      nextObservationWindowUtc: readiness.nextObservationWindowUtc,
      canUseForMicrocapitalCandidate: false,
      profileReasons: profile.reasons.map(r => truncStr(r, 160)),
      readinessReasons: readiness.reasons.map(r => truncStr(r, 160)),
    });
  }

  const payload = {
    generatedAtUtc: nowIso,
    horizonHours: cli.horizonHours,
    candidatesEvaluated: picked.length,
    marketsWithNearCatalyst,
    marketsWithoutNearCatalyst,
    marketsNeedingScheduleSource,
    note: "read_only_catalyst_plan_no_execution",
    plan,
  };

  const text = `${JSON.stringify(payload, null, 2)}\n`;
  process.stdout.write(text);
  assertOutPath(cli.outPath);
  await fs.promises.mkdir(path.dirname(path.resolve(cli.outPath)), { recursive: true });
  await fs.promises.writeFile(path.resolve(cli.outPath), text, "utf8");
}

main().catch(err => {
  process.stderr.write(
    `[buildCatalystObservationPlan] fatal: ${err instanceof Error ? err.message : String(err)}\n`,
  );
  process.exitCode = 1;
});
