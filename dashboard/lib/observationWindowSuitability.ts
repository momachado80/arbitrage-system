/**
 * Adequação read-only de uma janela de observação (série de snapshots bid/ask).
 * Sem rede, execução, capital ou `.paper`.
 */

import { nearPinnedPrice, parsedEndUtc, parsedNowUtc, type MarketSuitabilityVerdict } from "./marketSuitabilityGate";

export const OBS_MOVE_EPS = 1e-6;

export type ObservationWindowVerdict =
  | "WINDOW_INFORMATIVE"
  | "WINDOW_QUIET_BUT_VALID"
  | "WINDOW_NOT_READY"
  | "WINDOW_INSUFFICIENT_BOOK"
  | "WINDOW_PRICE_PINNED"
  | "WINDOW_MARKET_NOT_SUITABLE"
  | "WINDOW_EXPIRED_OR_CLOSED";

export interface ObservationWindowSnapshotInput {
  collectedAtIso: string;
  bestBid: number | null;
  bestAsk: number | null;
  mid: number | null;
  spread: number | null;
  bookType: string;
  priceSource?: string | null;
}

export interface ObservationWindowEvaluateInput {
  marketId?: string | null;
  label?: string | null;
  nowIso: string;
  endDate?: string | null;
  active?: unknown;
  closed?: unknown;
  resolved?: unknown;
  snapshots: readonly ObservationWindowSnapshotInput[];
  suitabilityVerdict: MarketSuitabilityVerdict | string;
  canUseForMicrocapitalCandidate?: unknown;
}

export interface ObservationWindowEvaluateResult {
  observationWindowVerdict: ObservationWindowVerdict;
  observationWindowScore: number;
  reasons: string[];
  midRange: number | null;
  spreadRange: number | null;
  bidRange: number | null;
  askRange: number | null;
  avgMid: number | null;
  avgSpread: number | null;
  twoSidedSnapshotCount: number;
  notTwoSidedSnapshotCount: number;
  nonZeroMidMoveCount: number;
  spreadMoveCount: number;
  bookStable: boolean;
  isWindowInformative: boolean;
  canUseForPaperShadowObservation: boolean;
  canUseForMicrocapitalCandidate: false;
}

function isTrue(x: unknown): boolean {
  return x === true;
}

function rangeFinite(vals: readonly (number | null)[]): number | null {
  const v = vals.filter((x): x is number => x !== null && Number.isFinite(x));
  if (v.length === 0) return null;
  return Math.max(...v) - Math.min(...v);
}

function pairwiseDeltas(series: readonly (number | null)[]): number {
  let c = 0;
  for (let i = 1; i < series.length; i++) {
    const a = series[i - 1];
    const b = series[i];
    if (
      a !== null &&
      b !== null &&
      Number.isFinite(a) &&
      Number.isFinite(b) &&
      Math.abs(b - a) > OBS_MOVE_EPS
    ) {
      c++;
    }
  }
  return c;
}

/** Conta snapshots i≥1 com |x[i]-x[0]| > eps. */
function deviationsFromBaseline(series: readonly (number | null)[]): number {
  const x0 = series[0];
  if (x0 === null || !Number.isFinite(x0)) return 0;
  let c = 0;
  for (let i = 1; i < series.length; i++) {
    const xi = series[i];
    if (xi !== null && Number.isFinite(xi) && Math.abs(xi - x0) > OBS_MOVE_EPS) c++;
  }
  return c;
}

function bidAskPairwiseMotion(bids: (number | null)[], asks: (number | null)[]): boolean {
  for (let i = 1; i < bids.length; i++) {
    const b0 = bids[i - 1];
    const b1 = bids[i];
    const a0 = asks[i - 1];
    const a1 = asks[i];
    if (b0 !== null && b1 !== null && Number.isFinite(b0) && Number.isFinite(b1) && Math.abs(b1 - b0) > OBS_MOVE_EPS)
      return true;
    if (a0 !== null && a1 !== null && Number.isFinite(a0) && Number.isFinite(a1) && Math.abs(a1 - a0) > OBS_MOVE_EPS)
      return true;
  }
  return false;
}

function avgFinite(vals: readonly (number | null)[]): number | null {
  const v = vals.filter((x): x is number => x !== null && Number.isFinite(x));
  if (!v.length) return null;
  return v.reduce((a, b) => a + b, 0) / v.length;
}

function scoreFor(
  verdict: ObservationWindowVerdict,
  midRange: number | null,
  spreadRange: number | null,
  twoSided: number,
  n: number,
  avgSpread: number | null,
  avgMid: number | null,
): number {
  let s = 0;
  if (verdict === "WINDOW_INFORMATIVE") s += 1_500;
  else if (verdict === "WINDOW_QUIET_BUT_VALID") s += 420;
  else if (verdict === "WINDOW_INSUFFICIENT_BOOK") s += 80;
  s += (midRange ?? 0) * 2_400;
  s += (spreadRange ?? 0) * 800;
  s += Math.min(twoSided, n) * 35;
  if (avgSpread !== null && Number.isFinite(avgSpread)) s -= avgSpread * 220;
  const m = avgMid;
  if (m !== null && Number.isFinite(m)) {
    const d = Math.min(m, 1 - m);
    if (d >= 0.05 && d <= 0.65) s += 140;
    if (d < 0.05) s -= 110;
  }
  return Number.isFinite(s) ? Math.round(s * 100) / 100 : 0;
}

function pack(
  verdict: ObservationWindowVerdict,
  paperOk: boolean,
  reasons: string[],
  fields: Omit<ObservationWindowEvaluateResult, "observationWindowVerdict" | "reasons" | "canUseForPaperShadowObservation">,
): ObservationWindowEvaluateResult {
  const canPaper = paperOk && (verdict === "WINDOW_INFORMATIVE" || verdict === "WINDOW_QUIET_BUT_VALID");
  return {
    observationWindowVerdict: verdict,
    reasons,
    canUseForPaperShadowObservation: canPaper,
    ...fields,
  };
}

/**
 * Avalia se a janela (snapshots espaced) tem actividade explorável versus flat / ruído trivial.
 */
export function evaluateObservationWindow(input: ObservationWindowEvaluateInput): ObservationWindowEvaluateResult {
  const reasons: string[] = [];
  const push = (...xs: string[]): void => {
    for (const x of xs) {
      if (!reasons.includes(x)) reasons.push(x);
    }
  };

  const now = parsedNowUtc(input.nowIso);
  const snaps = [...input.snapshots];
  const mids = snaps.map(s => s.mid);
  const spreads = snaps.map(s => s.spread);
  const bids = snaps.map(s => s.bestBid);
  const asks = snaps.map(s => s.bestAsk);

  const midRange = rangeFinite(mids);
  const spreadRange = rangeFinite(spreads);
  const bidRange = rangeFinite(bids);
  const askRange = rangeFinite(asks);

  const twoSidedSnapshotCount = snaps.filter(
    s => String(s.bookType).trim().toLowerCase() === "two_sided",
  ).length;
  const notTwoSidedSnapshotCount = snaps.length - twoSidedSnapshotCount;

  const nonZeroMidMoveCount = pairwiseDeltas(mids);
  const spreadMoveCount = pairwiseDeltas(spreads);
  const midDev = deviationsFromBaseline(mids);
  const spreadDev = deviationsFromBaseline(spreads);

  const bookStable =
    pairwiseDeltas(bids) === 0 && pairwiseDeltas(asks) === 0 && nonZeroMidMoveCount === 0 && spreadMoveCount === 0;

  const avgSpread = avgFinite(spreads);
  const avgMid = avgFinite(mids);

  const midQuiet = nonZeroMidMoveCount === 0 && midDev === 0;

  const baseFields = (verdict: ObservationWindowVerdict, informative: boolean, paper: boolean): ObservationWindowEvaluateResult =>
    pack(verdict, paper, reasons, {
      observationWindowScore: scoreFor(verdict, midRange, spreadRange, twoSidedSnapshotCount, snaps.length, avgSpread, avgMid),
      midRange,
      spreadRange,
      bidRange,
      askRange,
      avgMid,
      avgSpread,
      twoSidedSnapshotCount,
      notTwoSidedSnapshotCount,
      nonZeroMidMoveCount,
      spreadMoveCount,
      bookStable,
      isWindowInformative: informative,
      canUseForMicrocapitalCandidate: false,
    });

  if (isTrue(input.closed)) {
    push("closed:true");
    return baseFields("WINDOW_EXPIRED_OR_CLOSED", false, false);
  }
  if (isTrue(input.resolved)) {
    push("resolved:true");
    return baseFields("WINDOW_EXPIRED_OR_CLOSED", false, false);
  }
  const endDt = parsedEndUtc(input.endDate ?? null);
  if (endDt && Number.isFinite(endDt.getTime()) && now.getTime() > endDt.getTime()) {
    push("endDate_passed_relative_to_nowIso");
    return baseFields("WINDOW_EXPIRED_OR_CLOSED", false, false);
  }
  if (input.active === false) {
    push("inactive_market:active_false");
    return baseFields("WINDOW_EXPIRED_OR_CLOSED", false, false);
  }

  const suit = String(input.suitabilityVerdict).trim();
  if (suit !== "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION") {
    push(`suitability_not_paper_shadow:${suit}`);
    return baseFields("WINDOW_MARKET_NOT_SUITABLE", false, false);
  }

  if (snaps.length === 0) {
    push("no_snapshots_provided");
    return baseFields("WINDOW_NOT_READY", false, false);
  }

  if (twoSidedSnapshotCount < snaps.length) {
    push(`insufficient_two_sided_book:${twoSidedSnapshotCount}_of_${snaps.length}`);
    return baseFields("WINDOW_INSUFFICIENT_BOOK", false, false);
  }

  for (let i = 0; i < snaps.length; i++) {
    const s = snaps[i]!;
    if (nearPinnedPrice(s.bestBid) || nearPinnedPrice(s.bestAsk) || nearPinnedPrice(s.mid)) {
      push(`window_price_pinned_near_extreme:snapshot_${i}`);
      return baseFields("WINDOW_PRICE_PINNED", false, false);
    }
  }

  /** Regra de actividade dentro da janela (livro válido já filtrado). */
  let informative =
    midDev >= 2 ||
    spreadDev >= 2 ||
    nonZeroMidMoveCount >= 2 ||
    spreadMoveCount >= 2 ||
    (midDev >= 1 && spreadDev >= 1) ||
    (spreadDev >= 2 && midQuiet) ||
    (spreadDev >= 1 && spreadMoveCount >= 1 && midQuiet);

  if ((midRange ?? 0) > OBS_MOVE_EPS || (spreadRange ?? 0) > OBS_MOVE_EPS) {
    informative = true;
    push("window_range_positive_mid_or_spread");
  }
  if (bidAskPairwiseMotion(bids, asks)) {
    informative = true;
    push("bid_or_ask_moved_between_snapshots");
  }

  if (informative) {
    push("window_informative_motion_in_mid_or_spread_or_quotes");
    return baseFields("WINDOW_INFORMATIVE", true, true);
  }

  push("window_quiet_but_valid_two_sided_stable_quotes");
  return baseFields("WINDOW_QUIET_BUT_VALID", false, true);
}
