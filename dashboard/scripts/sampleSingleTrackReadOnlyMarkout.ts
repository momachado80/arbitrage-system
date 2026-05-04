/**
 * Preflight + amostra read-only de markouts para qualquer `marketId` (Gamma `/markets/<id>`).
 * Apenas GET públicos Gamma + topo do livro CLOB; sem watcher, worker, capital nem escrita `.paper`.
 */

import fs from "fs";
import path from "path";

import type { ClobBookStructureHint } from "../lib/marketSuitabilityGate";
import { evaluateMarketSuitability } from "../lib/marketSuitabilityGate";
import {
  buildGammaMarketByIdUrl,
  computeHorizonMarkoutDigest,
  parseSamplingCli,
  PRICE_SOURCE_INSUFFICIENT_BID_ASK,
  snapshotRowFromPrices,
  type BookBidAskSnapshotPure,
  type HorizonMarkoutDigest,
  type SampleInformativenessVerdict553856,
} from "../lib/singleTrackMarkoutSampler";
import {
  type GammaYesBookProbe,
  type Preflight553856Payload,
  type PreflightVerdict553856,
  bookStructure,
  coerceBool,
  jsonStringArray,
  pickYesToken,
  topPricesFromBook,
} from "./preflightSingleTrack553856";

export type SampleOutcomeVerdict = "NOT_READY" | "COMPLETE_READ_ONLY_MARKOUT_SAMPLING";

const CLOB_BOOK = "https://clob.polymarket.com/book";
const HTTP_BOOK_MS = 5_000;
const HTTP_GAMMA_MS = 12_000;
const REPO_ABS = path.resolve(__dirname, "..", "..");

const Q_TRUNC = 220;
const SLUG_TRUNC = 120;

function truncStr(s: string, n: number): string {
  if (s.length <= n) return s;
  return `${s.slice(0, n)}…`;
}

function numLikeLocal(x: unknown): number | null {
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

function unwrapGammaMarketRowById(body: unknown, want: string): Record<string, unknown> | null {
  const wantTrim = want.trim();
  const sameId = (o: Record<string, unknown>): boolean =>
    o.id !== undefined && o.id !== null && String(o.id) === wantTrim;

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

function summarizeGammaSafeGeneric(r: Record<string, unknown>): Preflight553856Payload["gammaSummary"] {
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

/** Pre-flight read-only alinhado a `evaluateMarketSuitability`, sem payloads Gamma crus no output. */
function computeSingleTrackMarketPreflightPayload(args: {
  marketId: string;
  nowIso: string;
  gammaRow: Record<string, unknown> | null;
  yesBookProbe: GammaYesBookProbe;
  gammaRequestPath: string;
}): Preflight553856Payload {
  const mid = args.marketId.trim();
  let row = args.gammaRow;
  if (row && String(row.id) !== mid) {
    row = null;
  }

  const scannedAtUtc = new Date(args.nowIso).toISOString();
  const yesTok = row ? pickYesToken(row) : null;

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
  const bidComb = yesTok ? args.yesBookProbe.bestBidBook ?? numLikeLocal(gh.bid) : null;
  const askComb = yesTok ? args.yesBookProbe.bestAskBook ?? numLikeLocal(gh.ask) : null;

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
    volume: numLikeLocal(row?.volume ?? null) ?? undefined,
    liquidity: numLikeLocal(row?.liquidity ?? null) ?? undefined,
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
    String(row?.id) === mid &&
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

  return {
    scannedAtUtc,
    marketId: mid,
    gammaRequestPath: args.gammaRequestPath,
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
    gammaSummary: row ? summarizeGammaSafeGeneric(row) : summarizeGammaSafeGeneric({ id: null }),
  };
}

export function assertSamplerOutputPath(writePathAbs: string): void {
  if (writePathAbs.includes(`${path.sep}.paper${path.sep}`) || writePathAbs.endsWith(`${path.sep}.paper`)) {
    throw new Error("output_path_blocked:.paper_directory");
  }
  const normalized = path.resolve(writePathAbs);
  const insideRepo = normalized === REPO_ABS || normalized.startsWith(REPO_ABS + path.sep);
  const tmpOk =
    normalized === "/tmp" ||
    normalized.startsWith("/tmp/") ||
    normalized === "/private/tmp" ||
    normalized.startsWith("/private/tmp/");
  if (!(tmpOk || !insideRepo)) {
    throw new Error("output_path_must_be_under_/tmp_or_outside_repository_root");
  }
}

async function fetchJson(url: string, ms: number): Promise<unknown> {
  const res = await fetch(url, {
    signal: AbortSignal.timeout(ms),
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`http_${res.status}`);
  return (await res.json()) as unknown;
}

function delayMs(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function readYesLiquiditySides(tokenId: string): Promise<{
  bestBidBook: number | null;
  bestAskBook: number | null;
  topologyHint: ClobBookStructureHint;
}> {
  const url = `${CLOB_BOOK}?token_id=${encodeURIComponent(tokenId)}`;
  try {
    const raw = await fetchJson(url, HTTP_BOOK_MS);
    const { bid, ask } = topPricesFromBook(raw);
    return { bestBidBook: bid, bestAskBook: ask, topologyHint: bookStructure(raw) };
  } catch {
    return { bestBidBook: null, bestAskBook: null, topologyHint: "unknown" };
  }
}

async function collectSnapshotRow(tokenId: string): Promise<BookBidAskSnapshotPure> {
  const { bestBidBook, bestAskBook } = await readYesLiquiditySides(tokenId);
  return snapshotRowFromPrices({
    bestBid: bestBidBook,
    bestAsk: bestAskBook,
    collectedAtIso: new Date().toISOString(),
  });
}

function horizonSnapshotsRecord(
  sortedHorizons: readonly number[],
  snaps: readonly BookBidAskSnapshotPure[],
): Record<string, BookBidAskSnapshotPure> {
  const horizons: Record<string, BookBidAskSnapshotPure> = {
    t0: snaps[0]!,
  };
  sortedHorizons.forEach((sec, idx) => {
    horizons[`t_plus_${sec}s`] = snaps[idx + 1]!;
  });
  return horizons;
}

/** JSON seguro público para batch ou stdout — sem texto bruto de mercado Gamma. */
export async function sampleSingleTrackReadOnlyMarkoutRun(args: {
  marketId: string;
  label?: string;
  horizonsSec: readonly number[];
}): Promise<Record<string, unknown>> {
  const tStart = new Date().toISOString();
  const marketId = args.marketId.trim();
  const sorted = [...args.horizonsSec].sort((a, b) => a - b);
  const gammaUrl = buildGammaMarketByIdUrl(marketId);
  const gammaReqPath = `/markets/${marketId}`;
  let row: Record<string, unknown> | null = null;
  let tokenId: string | null = null;

  try {
    const gammaBody = await fetchJson(gammaUrl, HTTP_GAMMA_MS);
    row = unwrapGammaMarketRowById(gammaBody, marketId);
    tokenId = row && pickYesToken(row) ? pickYesToken(row)! : null;
  } catch (err) {
    return {
      sampleVerdict: "NOT_READY" as const,
      marketId,
      label: args.label,
      horizonsSecSorted: sorted,
      sampledAtUtc: tStart,
      canUseForMicrocapitalCandidate: false as const,
      reasons: [`gamma_fetch_failed:${err instanceof Error ? err.message : String(err)}`],
    };
  }

  const emptyProbe: GammaYesBookProbe = {
    bookStructure: "skipped",
    bestBidBook: null,
    bestAskBook: null,
    yesTokenPrefix: null,
  };
  let gateProbe = emptyProbe;
  if (tokenId) {
    const fb = await readYesLiquiditySides(tokenId);
    gateProbe = {
      bookStructure: fb.topologyHint,
      bestBidBook: fb.bestBidBook,
      bestAskBook: fb.bestAskBook,
      yesTokenPrefix: truncStr(tokenId, 18),
    };
  }

  const preflight = computeSingleTrackMarketPreflightPayload({
    marketId,
    nowIso: tStart,
    gammaRow: row,
    yesBookProbe: gateProbe,
    gammaRequestPath: gammaReqPath,
  });

  if (preflight.preflightVerdict !== "READY_FOR_PAPER_SHADOW_OBSERVATION" || !tokenId || !row) {
    return {
      sampleVerdict: "NOT_READY" as const,
      marketId,
      label: args.label,
      horizonsSecSorted: sorted,
      sampledAtUtc: tStart,
      canUseForMicrocapitalCandidate: false as const,
      reasons: preflight.reasons,
      gateVerbatim: preflight.preflightVerdict,
    };
  }

  const snaps: BookBidAskSnapshotPure[] = [];
  snaps.push(await collectSnapshotRow(tokenId));
  let waited = 0;
  for (const h of sorted) {
    const delta = Math.max(0, h - waited);
    await delayMs(delta * 1000);
    waited += delta;
    snaps.push(await collectSnapshotRow(tokenId));
  }

  const horizons = horizonSnapshotsRecord(sorted, snaps);

  let digest: HorizonMarkoutDigest;
  try {
    digest = computeHorizonMarkoutDigest({ snapshots: snaps, horizonsSec: sorted });
  } catch {
    return {
      sampleVerdict: "NOT_READY" as const,
      marketId,
      label: args.label,
      horizonsSecSorted: sorted,
      sampledAtUtc: tStart,
      canUseForMicrocapitalCandidate: false as const,
      reasons: ["digest_snapshot_horizon_mismatch_internal"],
      gateVerbatim: preflight.preflightVerdict,
    };
  }

  const snapKeysOrdered = ["t0", ...sorted.map(s => `t_plus_${s}s`)];
  const anyInsufficient = snapKeysOrdered.some(
    hk => horizons[hk]!.priceSource === PRICE_SOURCE_INSUFFICIENT_BID_ASK || horizons[hk]!.mid === null,
  );

  let sampleInfoVerdict: SampleInformativenessVerdict553856 =
    digest.sampleInformativenessVerdict;
  if (anyInsufficient) sampleInfoVerdict = "INSUFFICIENT_BOOK_SAMPLE";

  const legacyMarkouts = ["5", "30", "60"].map(sec => ({
    horizonSec: parseInt(sec, 10),
    markoutVsT0: digest.markoutsVsT0ByHorizonSec[sec] ?? null,
  }));

  return {
    sampleVerdict: "COMPLETE_READ_ONLY_MARKOUT_SAMPLING" as const,
    sampledAtUtc: tStart,
    finishedAtUtc: new Date().toISOString(),
    marketId,
    label: args.label,
    gammaUrlAuditPreview: truncStr(gammaUrl, 82),
    canUseForMicrocapitalCandidate: false as const,
    preflightVerdictFrozen: preflight.preflightVerdict,
    horizonsSecSorted: sorted,
    horizons,
    markoutsVsT0ByHorizonSec: digest.markoutsVsT0ByHorizonSec,
    markoutsVsT0: digest.markoutsVsT0,
    markoutLegacyTriplet: legacyMarkouts,
    nonZeroMarkoutCount: digest.nonZeroMarkoutCount,
    allFlat: digest.allFlat,
    pricePinned: digest.pricePinned,
    maxAbsMarkout: digest.maxAbsMarkout,
    averageAbsMarkout: digest.averageAbsMarkout,
    sampleInformativenessVerdict: sampleInfoVerdict,
  };
}

async function writeOutIfConfigured(outPathAbs: string, payloadText: string): Promise<void> {
  assertSamplerOutputPath(outPathAbs);
  await fs.promises.mkdir(path.dirname(path.resolve(outPathAbs)), { recursive: true });
  await fs.promises.writeFile(outPathAbs, payloadText, { encoding: "utf8" });
}

async function main(): Promise<void> {
  const cli = parseSamplingCli(process.argv);
  const payload = await sampleSingleTrackReadOnlyMarkoutRun({
    marketId: cli.marketId,
    label: cli.label,
    horizonsSec: cli.horizonsSec,
  });

  const text = `${JSON.stringify(payload, null, 2)}\n`;
  process.stdout.write(text);

  if (cli.outPath) await writeOutIfConfigured(cli.outPath, text);
}

if (require.main === module) {
  main().catch(err => {
    process.stderr.write(
      `[sampleSingleTrackReadOnlyMarkout] fatal: ${err instanceof Error ? err.message : String(err)}\n`,
    );
    process.exitCode = 1;
  });
}
