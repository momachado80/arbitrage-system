/**
 * Ferramentas puras read-only para amostrar markouts em livro público bid/ask.
 * Sem rede, execução, livro nem credenciais.
 */

export const MARKOUT_NONZERO_EPS = 1e-6;

export type BookSidesLabel = "two_sided" | "not_two_sided";

export type SampleInformativenessVerdict553856 =
  | "INFORMATIVE_SAMPLE"
  | "FLAT_SAMPLE"
  | "INSUFFICIENT_BOOK_SAMPLE"
  | "PRICE_PINNED_SAMPLE";

/** Origem quando bid+ask válidos constituem média aritmética. */
export const PRICE_SOURCE_BID_ASK_MID = "bid_ask_mid";
export const PRICE_SOURCE_INSUFFICIENT_BID_ASK = "insufficient_bid_ask";

export interface BookBidAskSnapshotPure {
  bestBid: number | null;
  bestAsk: number | null;
  mid: number | null;
  spread: number | null;
  priceSource: string;
  bookType: BookSidesLabel;
  collectedAtIso: string;
}

function finPos(x: number | null): x is number {
  return x !== null && Number.isFinite(x) && x > 0;
}

export function computeMid(bestBid: number | null, bestAsk: number | null): number | null {
  if (!finPos(bestBid) || !finPos(bestAsk) || bestAsk < bestBid * 0.999) return null;
  return (bestBid + bestAsk) / 2;
}

export function computeSpread(bestBid: number | null, bestAsk: number | null): number | null {
  if (!finPos(bestBid) || !finPos(bestAsk) || bestAsk < bestBid * 0.999) return null;
  return bestAsk - bestBid;
}

export function classifyBookType(bestBid: number | null, bestAsk: number | null): BookSidesLabel {
  if (
    finPos(bestBid) &&
    finPos(bestAsk) &&
    bestAsk >= bestBid * 0.999 &&
    bestAsk <= 1 &&
    bestBid <= 1
  )
    return "two_sided";
  return "not_two_sided";
}

export interface MarkoutAccumulatorInput {
  mid0: number | null;
  mid5s: number | null;
  mid30s: number | null;
  mid60s: number | null;
  bookTypesSnapshot: readonly BookSidesLabel[];
}

export interface MarkoutSummary553856 {
  markout5s: number | null;
  markout30s: number | null;
  markout60s: number | null;
  nonZeroMarkoutCount: number;
  allFlat: boolean;
  pricePinned: boolean;
  maxAbsMarkout: number | null;
  averageAbsMarkout: number | null;
  sampleInformativenessVerdict: SampleInformativenessVerdict553856;
}

function diffOrNull(midLater: number | null, mid0: number | null): number | null {
  if (midLater === null || mid0 === null || !Number.isFinite(midLater) || !Number.isFinite(mid0))
    return null;
  return midLater - mid0;
}

function absMx(x: number | null): number | null {
  if (x === null || !Number.isFinite(x)) return null;
  return Math.abs(x);
}

/** Mids colados (~constantes nas amostras) e na cauda ~0 ou ~1 probabilística. Exige pelo menos 4 níveis estáveis para reduzir falso pinned. */
export function classifyProbabilisticPricePinned(allMids: readonly (number | null)[]): boolean {
  const v = allMids.filter((m): m is number => m !== null && Number.isFinite(m));
  if (v.length < 4) return false;
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  if (hi - lo > 1e-4) return false;
  const m = v[0]!;
  return Math.abs(m - 0.001) <= 2e-3 || Math.abs(m - 0.999) <= 2e-3;
}

export type BatchTrackReadDecision =
  | "CONTINUE_PAPER_SHADOW_CANDIDATE"
  | "RETRY_LATER"
  | "RETIRE_AS_FIRST_TRACK"
  | "RETIRE_IMMEDIATELY";

/** Heurística de decisão sobre N amostragens read-only (sem execução). */
export function decideBatchPaperShadowTrack(input: {
  sampleCount: number;
  informativeSamples: number;
  flatSamples: number;
  insufficientBookSamples: number;
  pricePinnedSamples: number;
}): BatchTrackReadDecision {
  const n = input.sampleCount;
  if (n <= 0) return "RETRY_LATER";
  const majorityInsufficient = input.insufficientBookSamples > Math.floor(n / 2);
  if (input.pricePinnedSamples > 0 || majorityInsufficient) return "RETIRE_IMMEDIATELY";
  if (input.informativeSamples >= 2) return "CONTINUE_PAPER_SHADOW_CANDIDATE";
  if (input.informativeSamples === 1) return "RETRY_LATER";
  if (input.flatSamples === n) return "RETIRE_AS_FIRST_TRACK";
  return "RETRY_LATER";
}

export function parseSamplingCli(argv: readonly string[]): {
  marketId: string;
  label?: string;
  horizonsSec: number[];
  outPath?: string;
} {
  let marketIdFlag: string | undefined;
  let labelFlag: string | undefined;
  let horizonsCsv: string | undefined;
  let outPathFlag: string | undefined;

  for (let i = 2; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === "--market-id" && argv[i + 1]) {
      marketIdFlag = argv[++i]!;
    } else if (a.startsWith("--market-id=")) {
      marketIdFlag = a.slice("--market-id=".length).trim();
    } else if (a === "--label" && argv[i + 1]) {
      labelFlag = argv[++i]!;
    } else if (a.startsWith("--label=")) {
      labelFlag = a.slice("--label=".length).trim();
    } else if (a === "--horizons" && argv[i + 1]) {
      horizonsCsv = argv[++i]!;
    } else if (a.startsWith("--horizons=")) {
      horizonsCsv = a.slice("--horizons=".length).trim();
    } else if (a === "--out" && argv[i + 1]) {
      outPathFlag = argv[++i]!;
    } else if (a.startsWith("--out=")) {
      outPathFlag = a.slice("--out=".length).trim();
    }
  }

  if (!marketIdFlag?.trim()) {
    throw new Error("missing_required_flag:--market-id");
  }

  let horizonsParsed = [5, 30, 60];
  if (horizonsCsv?.trim()) {
    horizonsParsed = horizonsCsv
      .split(",")
      .map(s => parseInt(s.trim(), 10))
      .filter(n => Number.isFinite(n) && n > 0);
    horizonsParsed = Array.from(new Set(horizonsParsed)).sort((a, b) => a - b);
  }
  if (horizonsParsed.length === 0) {
    throw new Error("horizons_must_contain_positive_integers");
  }

  return {
    marketId: marketIdFlag.trim(),
    label: labelFlag?.trim() || undefined,
    horizonsSec: horizonsParsed,
    outPath: outPathFlag?.trim() || undefined,
  };
}

export function parseBatchCli(argv: readonly string[]): {
  marketId: string;
  label?: string;
  samples: number;
  gapSeconds: number;
  outDir: string;
} {
  let marketIdFlag: string | undefined;
  let labelFlag: string | undefined;
  let samplesCsv: string | undefined;
  let gapCsv: string | undefined;
  let outDirFlag: string | undefined;

  for (let i = 2; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === "--market-id" && argv[i + 1]) marketIdFlag = argv[++i]!;
    else if (a.startsWith("--market-id=")) marketIdFlag = a.slice("--market-id=".length).trim();
    else if (a === "--label" && argv[i + 1]) labelFlag = argv[++i]!;
    else if (a.startsWith("--label=")) labelFlag = a.slice("--label=".length).trim();
    else if (a === "--samples" && argv[i + 1]) samplesCsv = argv[++i]!;
    else if (a.startsWith("--samples=")) samplesCsv = a.slice("--samples=".length).trim();
    else if (a === "--gap-seconds" && argv[i + 1]) gapCsv = argv[++i]!;
    else if (a.startsWith("--gap-seconds=")) gapCsv = a.slice("--gap-seconds=".length).trim();
    else if (a === "--out-dir" && argv[i + 1]) outDirFlag = argv[++i]!;
    else if (a.startsWith("--out-dir=")) outDirFlag = a.slice("--out-dir=".length).trim();
  }

  if (!marketIdFlag?.trim()) throw new Error("missing_required_flag:--market-id");
  if (!outDirFlag?.trim()) throw new Error("missing_required_flag:--out-dir");
  const samples = Math.max(1, parseInt(samplesCsv ?? "1", 10) || 1);
  const gapSeconds = Math.max(0, parseInt(gapCsv ?? "0", 10) || 0);

  return {
    marketId: marketIdFlag.trim(),
    label: labelFlag?.trim() || undefined,
    samples,
    gapSeconds,
    outDir: outDirFlag.trim(),
  };
}

/** URL Gamma apenas path REST; evita filtros tipo `?id=` legacy sem path. */
export function buildGammaMarketByIdUrl(marketId: string, baseUrl = "https://gamma-api.polymarket.com"): string {
  return `${baseUrl.replace(/\/$/, "")}/markets/${encodeURIComponent(marketId.trim())}`;
}

export function computeMarkoutSummary(input: MarkoutAccumulatorInput): MarkoutSummary553856 {
  const m0 = input.mid0;
  const m5 = input.mid5s;
  const m30 = input.mid30s;
  const m60 = input.mid60s;

  const markout5s = diffOrNull(m5, m0);
  const markout30s = diffOrNull(m30, m0);
  const markout60s = diffOrNull(m60, m0);

  const markouts = [markout5s, markout30s, markout60s];
  let nonZero = 0;
  for (const mk of markouts) {
    if (mk !== null && Math.abs(mk) > MARKOUT_NONZERO_EPS) nonZero++;
  }

  let allFlat = true;
  for (const mk of markouts) {
    if (mk === null) {
      allFlat = false;
      break;
    }
    if (Math.abs(mk) > MARKOUT_NONZERO_EPS) {
      allFlat = false;
      break;
    }
  }

  const absVals = markouts.map(absMx).filter((x): x is number => x !== null);
  const maxAbsMarkout = absVals.length > 0 ? Math.max(...absVals) : null;
  const averageAbsMarkout =
    absVals.length > 0 ? absVals.reduce((a, b) => a + b, 0) / absVals.length : null;

  const allFourMids = [m0, m5, m30, m60];
  const pricePinned = classifyProbabilisticPricePinned(allFourMids);

  const verdict = classifySampleInformativeness({
    bookTypesSnapshot: input.bookTypesSnapshot,
    pricePinned,
    nonZeroMarkoutCount: nonZero,
  });

  return {
    markout5s,
    markout30s,
    markout60s,
    nonZeroMarkoutCount: nonZero,
    allFlat,
    pricePinned,
    maxAbsMarkout,
    averageAbsMarkout,
    sampleInformativenessVerdict: verdict,
  };
}

export function classifySampleInformativeness(args: {
  bookTypesSnapshot: readonly BookSidesLabel[];
  pricePinned: boolean;
  nonZeroMarkoutCount: number;
}): SampleInformativenessVerdict553856 {
  const allTwo =
    args.bookTypesSnapshot.length >= 2 &&
    args.bookTypesSnapshot.every(b => b === "two_sided");
  if (!allTwo) return "INSUFFICIENT_BOOK_SAMPLE";
  if (args.pricePinned) return "PRICE_PINNED_SAMPLE";
  if (args.nonZeroMarkoutCount >= 1) return "INFORMATIVE_SAMPLE";
  return "FLAT_SAMPLE";
}

export interface HorizonMarkoutDigest {
  horizonsSec: readonly number[];
  /** Uma entrada por horizon: Δmid vs m0 */
  markoutsVsT0: readonly (number | null)[];
  markoutsVsT0ByHorizonSec: Record<string, number | null>;
  mids: readonly (number | null)[];
  spreads: readonly (number | null)[];
  bookTypesSnapshot: readonly BookSidesLabel[];
  nonZeroMarkoutCount: number;
  allFlat: boolean;
  pricePinned: boolean;
  maxAbsMarkout: number | null;
  averageAbsMarkout: number | null;
  sampleInformativenessVerdict: SampleInformativenessVerdict553856;
}

/** Markouts contra t0 para horizontes arbitrários ordenados (`snapshots[0]` = baseline). */
export function computeHorizonMarkoutDigest(args: {
  snapshots: readonly BookBidAskSnapshotPure[];
  horizonsSec: readonly number[];
}): HorizonMarkoutDigest {
  const h = [...args.horizonsSec];
  if (args.snapshots.length !== 1 + h.length)
    throw new Error("snapshot_count_must_match_horizons_plus_t0");

  const mids = args.snapshots.map(s => s.mid);
  const spreads = args.snapshots.map(s => s.spread);
  const bt = args.snapshots.map(s => s.bookType);
  const m0 = mids[0] ?? null;
  const markoutsVsT0 = h.map((_, idx) => diffOrNull(mids[idx + 1] ?? null, m0));
  const markoutsVsT0ByHorizonSec: Record<string, number | null> = {};
  for (let i = 0; i < h.length; i++) markoutsVsT0ByHorizonSec[String(h[i])] = markoutsVsT0[i] ?? null;

  let nonZero = 0;
  for (const mk of markoutsVsT0) {
    if (mk !== null && Math.abs(mk) > MARKOUT_NONZERO_EPS) nonZero++;
  }

  let allFlat = markoutsVsT0.length > 0;
  for (const mk of markoutsVsT0) {
    if (mk === null) {
      allFlat = false;
      break;
    }
    if (Math.abs(mk) > MARKOUT_NONZERO_EPS) {
      allFlat = false;
      break;
    }
  }

  const absVals = markoutsVsT0.map(absMx).filter((x): x is number => x !== null);
  const maxAbsMarkout = absVals.length > 0 ? Math.max(...absVals) : null;
  const averageAbsMarkout =
    absVals.length > 0 ? absVals.reduce((a, b) => a + b, 0) / absVals.length : null;

  const pricePinned = classifyProbabilisticPricePinned(mids);
  const sampleInformativenessVerdict = classifySampleInformativeness({
    bookTypesSnapshot: bt,
    pricePinned,
    nonZeroMarkoutCount: nonZero,
  });

  return {
    horizonsSec: h,
    markoutsVsT0,
    markoutsVsT0ByHorizonSec,
    mids,
    spreads,
    bookTypesSnapshot: bt,
    nonZeroMarkoutCount: nonZero,
    allFlat,
    pricePinned,
    maxAbsMarkout,
    averageAbsMarkout,
    sampleInformativenessVerdict,
  };
}

/** Converte topo do livro + timestamp numa linha estável ao JSON público. */
export function snapshotRowFromPrices(input: {
  bestBid: number | null;
  bestAsk: number | null;
  collectedAtIso: string;
}): BookBidAskSnapshotPure {
  const bookType = classifyBookType(input.bestBid, input.bestAsk);
  const mid = computeMid(input.bestBid, input.bestAsk);
  const spread = computeSpread(input.bestBid, input.bestAsk);
  const priceSource =
    bookType === "two_sided" && mid !== null ? PRICE_SOURCE_BID_ASK_MID : PRICE_SOURCE_INSUFFICIENT_BID_ASK;
  return {
    bestBid: input.bestBid,
    bestAsk: input.bestAsk,
    mid,
    spread,
    priceSource,
    bookType,
    collectedAtIso: input.collectedAtIso,
  };
}
