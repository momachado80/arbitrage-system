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

/** Mids colados (~constantes nas amostras) e na caixa ~0 ou ~1 probabilística. */
export function classifyProbabilisticPricePinned(allMids: readonly (number | null)[]): boolean {
  const v = allMids.filter((m): m is number => m !== null && Number.isFinite(m));
  if (v.length !== 4) return false;
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  if (hi - lo > 1e-4) return false;
  const m = v[0]!;
  return Math.abs(m - 0.001) <= 2e-3 || Math.abs(m - 0.999) <= 2e-3;
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
    args.bookTypesSnapshot.length === 4 &&
    args.bookTypesSnapshot.every(b => b === "two_sided");
  if (!allTwo) return "INSUFFICIENT_BOOK_SAMPLE";
  if (args.pricePinned) return "PRICE_PINNED_SAMPLE";
  if (args.nonZeroMarkoutCount >= 1) return "INFORMATIVE_SAMPLE";
  return "FLAT_SAMPLE";
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
