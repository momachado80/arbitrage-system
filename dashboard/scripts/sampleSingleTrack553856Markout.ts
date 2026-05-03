/**
 * Single-track markout sampler read-only (553856 Thunder Finals).
 * Apenas GET Gamma + livro público YES; delays 0/5/30/60s. Sem .paper nem worker nem watcher.
 */

import {
  compute553856PreflightPayload,
  type GammaYesBookProbe,
  bookStructure as clobSidesHint,
  buildGamma553856MarketPreflightUrl,
  unwrapGamma553856MarketRow,
  pickYesToken,
  topPricesFromBook,
  SINGLE_TRACK_553856_MARKET_ID,
} from "./preflightSingleTrack553856";
import type { ClobBookStructureHint } from "../lib/marketSuitabilityGate";
import {
  PRICE_SOURCE_INSUFFICIENT_BID_ASK,
  computeMarkoutSummary,
  snapshotRowFromPrices,
  type BookBidAskSnapshotPure,
  type SampleInformativenessVerdict553856,
} from "../lib/singleTrackMarkoutSampler";

const CLOB_BOOK = "https://clob.polymarket.com/book";
const HTTP_BOOK_MS = 5_000;
const HTTP_GAMMA_MS = 12_000;

type SampleOutcomeVerdict = "NOT_READY" | "COMPLETE_READ_ONLY_MARKOUT_SAMPLING";

function truncStr(s: string, n: number): string {
  if (s.length <= n) return s;
  return `${s.slice(0, n)}…`;
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
    return { bestBidBook: bid, bestAskBook: ask, topologyHint: clobSidesHint(raw) };
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

async function main(): Promise<void> {
  const tStart = new Date().toISOString();
  const gammaUrl = buildGamma553856MarketPreflightUrl();

  let row: Record<string, unknown> | null = null;
  let tokenId: string | null = null;

  try {
    const gammaBody = await fetchJson(gammaUrl, HTTP_GAMMA_MS);
    row = unwrapGamma553856MarketRow(gammaBody);
    tokenId = row && pickYesToken(row) ? pickYesToken(row)! : null;
  } catch (err) {
    process.stdout.write(
      `${JSON.stringify(
        {
          sampleVerdict: "NOT_READY" as const,
          marketId: SINGLE_TRACK_553856_MARKET_ID,
          sampledAtUtc: tStart,
          canUseForMicrocapitalCandidate: false as const,
          reasons: [`gamma_fetch_failed:${err instanceof Error ? err.message : String(err)}`],
        },
        null,
        2,
      )}\n`,
    );
    return;
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

  const preflight = compute553856PreflightPayload({
    nowIso: tStart,
    gammaRow: row,
    yesBookProbe: gateProbe,
  });

  if (preflight.preflightVerdict !== "READY_FOR_PAPER_SHADOW_OBSERVATION" || !tokenId || !row) {
    process.stdout.write(
      `${JSON.stringify(
        {
          sampleVerdict: "NOT_READY" as const,
          marketId: SINGLE_TRACK_553856_MARKET_ID,
          sampledAtUtc: tStart,
          canUseForMicrocapitalCandidate: false as const,
          reasons: preflight.reasons,
          gateVerbatim: preflight.preflightVerdict,
        },
        null,
        2,
      )}\n`,
    );
    return;
  }

  const horizons: Record<string, BookBidAskSnapshotPure> = {};
  horizons.t0 = await collectSnapshotRow(tokenId);
  await delayMs(5_000);
  horizons.t_plus_5s = await collectSnapshotRow(tokenId);
  await delayMs(25_000);
  horizons.t_plus_30s = await collectSnapshotRow(tokenId);
  await delayMs(30_000);
  horizons.t_plus_60s = await collectSnapshotRow(tokenId);

  const snapKeys = ["t0", "t_plus_5s", "t_plus_30s", "t_plus_60s"] as const;
  const bookTypesSnap = snapKeys.map(k => horizons[k].bookType);

  const markSummary = computeMarkoutSummary({
    mid0: horizons.t0.mid,
    mid5s: horizons.t_plus_5s.mid,
    mid30s: horizons.t_plus_30s.mid,
    mid60s: horizons.t_plus_60s.mid,
    bookTypesSnapshot: bookTypesSnap,
  });

  const anyInsufficient = snapKeys.some(
    hk => horizons[hk].priceSource === PRICE_SOURCE_INSUFFICIENT_BID_ASK || horizons[hk].mid === null,
  );

  let sampleInfoVerdict: SampleInformativenessVerdict553856 =
    markSummary.sampleInformativenessVerdict;
  if (anyInsufficient) sampleInfoVerdict = "INSUFFICIENT_BOOK_SAMPLE";

  const sampleVerdict: SampleOutcomeVerdict = "COMPLETE_READ_ONLY_MARKOUT_SAMPLING";

  const payload = {
    sampleVerdict,
    sampledAtUtc: tStart,
    finishedAtUtc: new Date().toISOString(),
    marketId: SINGLE_TRACK_553856_MARKET_ID,
    gammaUrlAuditPreview: truncStr(gammaUrl, 82),
    canUseForMicrocapitalCandidate: false as const,
    preflightVerdictFrozen: preflight.preflightVerdict,
    horizons,
    markout5s: markSummary.markout5s,
    markout30s: markSummary.markout30s,
    markout60s: markSummary.markout60s,
    nonZeroMarkoutCount: markSummary.nonZeroMarkoutCount,
    allFlat: markSummary.allFlat,
    pricePinned: markSummary.pricePinned,
    maxAbsMarkout: markSummary.maxAbsMarkout,
    averageAbsMarkout: markSummary.averageAbsMarkout,
    sampleInformativenessVerdict: sampleInfoVerdict,
  };

  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
}

main().catch(err => {
  process.stderr.write(
    `[sampleSingleTrack553856Markout] fatal: ${err instanceof Error ? err.message : String(err)}\n`,
  );
  process.exitCode = 1;
});
