/**
 * Gate read-only de adequação de mercado ao uso como candidato microcapital / observação económica viva.
 * Função pura; sem rede, sem executor, sem submissão real.
 */

export type MarketSuitabilityVerdict =
  | "SUITABLE_FOR_PAPER_OBSERVATION"
  | "UNSUITABLE_EXPIRED"
  | "UNSUITABLE_RESOLVED_OR_CLOSED"
  | "UNSUITABLE_PRICE_PINNED"
  | "UNSUITABLE_NO_INFORMATIVE_MARKOUTS"
  | "UNSUITABLE_LIQUIDITY"
  | "UNSUITABLE_MISSING_DATA";

export interface MarketSuitabilityInput {
  marketId?: string;
  title?: string;
  question?: string;
  endDate?: string | null;
  closed?: boolean;
  resolved?: boolean;
  active?: boolean;
  bestBid?: number | null;
  bestAsk?: number | null;
  spread?: number | null;
  lastPrice?: number | null;
  volume?: number | null;
  liquidity?: number | null;
  priceSource?: string | null;
  markoutInformativenessVerdict?: string | null;
  lowPricePinnedCycleCount?: number | null;
  informativeMarkoutCycleCount?: number | null;
  nowIso?: string;
}

export interface MarketSuitabilityResult {
  suitabilityVerdict: MarketSuitabilityVerdict;
  reasons: string[];
  canUseForMicrocapitalCandidate: boolean;
}

const PINNED_LOW = 0.001;
const PINNED_HIGH = 0.999;
const PRICE_PIN_TOL = 1e-5;

export function parsedNowUtc(nowIso?: string): Date {
  const d = nowIso?.trim() ? new Date(nowIso.trim()) : new Date();
  return Number.isFinite(d.getTime()) ? d : new Date();
}

export function parsedEndUtc(endDate?: string | null): Date | null {
  if (endDate === undefined || endDate === null || !String(endDate).trim()) return null;
  const d = new Date(String(endDate).trim());
  return Number.isFinite(d.getTime()) ? d : null;
}

export function nearPinnedPrice(x: unknown): boolean {
  if (typeof x !== "number" || !Number.isFinite(x)) return false;
  return Math.abs(x - PINNED_LOW) <= PRICE_PIN_TOL || Math.abs(x - PINNED_HIGH) <= PRICE_PIN_TOL;
}

/** Heurística mensal “April”: após Mai 1 UTC do ano pertinente, consideramos fora da janela. */
export function aprilNamedWindowEnded(questionOrTitle: string | undefined, now: Date): boolean {
  const t = questionOrTitle?.trim();
  if (!t || !/\bApril\b/i.test(t)) return false;
  const m = t.match(/\b(20\d{2})\b/);
  const year = m ? Number(m[1]) : now.getUTCFullYear();
  return now.getTime() >= Date.UTC(year, 4, 1);
}

const VIOLATION_ORDER: Record<MarketSuitabilityVerdict, number> = {
  UNSUITABLE_MISSING_DATA: 0,
  UNSUITABLE_RESOLVED_OR_CLOSED: 1,
  UNSUITABLE_EXPIRED: 2,
  UNSUITABLE_PRICE_PINNED: 3,
  UNSUITABLE_NO_INFORMATIVE_MARKOUTS: 4,
  UNSUITABLE_LIQUIDITY: 5,
  SUITABLE_FOR_PAPER_OBSERVATION: 99,
};

function pickWorst(xs: MarketSuitabilityVerdict[]): MarketSuitabilityVerdict {
  let out: MarketSuitabilityVerdict = "SUITABLE_FOR_PAPER_OBSERVATION";
  let bestRank = Infinity;
  for (const v of xs) {
    const r = VIOLATION_ORDER[v];
    if (r < bestRank) {
      bestRank = r;
      out = v;
    }
  }
  return out;
}

function fin(x: unknown): number | null {
  return typeof x === "number" && Number.isFinite(x) ? x : null;
}

function hasSignals(input: MarketSuitabilityInput): boolean {
  const text = `${input.title ?? ""} ${input.question ?? ""}`.trim();
  return Boolean(
    text ||
      input.marketId?.trim() ||
      input.endDate !== undefined ||
      input.closed !== undefined ||
      input.resolved !== undefined ||
      input.active !== undefined ||
      input.bestBid != null ||
      input.bestAsk != null ||
      input.spread != null ||
      input.lastPrice != null ||
      input.volume != null ||
      input.liquidity != null ||
      input.priceSource?.trim() ||
      input.markoutInformativenessVerdict !== undefined ||
      input.informativeMarkoutCycleCount !== undefined ||
      input.lowPricePinnedCycleCount !== undefined,
  );
}

/** Dois lados positivos fora das extremidades presas (~0/~1 probabilísticos). */
export function trustworthyBidAsk(
  bidRaw: unknown,
  askRaw: unknown,
): { bid: number; ask: number } | null {
  const bid = fin(bidRaw);
  const ask = fin(askRaw);
  if (bid === null || ask === null || bid <= 0 || ask <= 0 || ask < bid * 0.999) return null;
  if (nearPinnedPrice(bid) || nearPinnedPrice(ask)) return null;
  return { bid, ask };
}

/** Por defeito não promove mercado microcapital até prova suficiente. */
export function computeMarketSuitability(input: MarketSuitabilityInput): MarketSuitabilityResult {
  const reasons: string[] = [];
  const push = (s: string): void => {
    if (!reasons.includes(s)) reasons.push(s);
  };

  const now = parsedNowUtc(input.nowIso);
  const vio: MarketSuitabilityVerdict[] = [];

  if (!hasSignals(input)) {
    return {
      suitabilityVerdict: "UNSUITABLE_MISSING_DATA",
      reasons: ["no_assessable_input"],
      canUseForMicrocapitalCandidate: false,
    };
  }

  if (input.closed === true) {
    vio.push("UNSUITABLE_RESOLVED_OR_CLOSED");
    push("closed:true");
  }
  if (input.resolved === true) {
    vio.push("UNSUITABLE_RESOLVED_OR_CLOSED");
    push("resolved:true");
  }

  const endAbs = parsedEndUtc(input.endDate ?? undefined);
  const expiredByEnd = !!(endAbs && now.getTime() > endAbs.getTime());

  const qLine = input.question ?? input.title;
  let expiredTemporal = expiredByEnd;
  if (!(input.closed ?? false) && !(input.resolved ?? false) && !expiredTemporal && aprilNamedWindowEnded(qLine, now)) {
    expiredTemporal = true;
    vio.push("UNSUITABLE_EXPIRED");
    push("temporal_named_april_bucket_considered_finished");
  }

  if (!(input.closed ?? false) && !(input.resolved ?? false) && expiredByEnd) {
    vio.push("UNSUITABLE_EXPIRED");
    push("endDate_already_passed");
  }

  if (nearPinnedPrice(input.bestBid) || nearPinnedPrice(input.bestAsk) || nearPinnedPrice(input.lastPrice)) {
    vio.push("UNSUITABLE_PRICE_PINNED");
    push("extreme_quote_or_last_near_tick_floor_ceiling");
  }

  const markOutUpper = input.markoutInformativenessVerdict?.trim().toUpperCase() ?? "";
  const infoCycles = input.informativeMarkoutCycleCount;

  const infoCyclesKnown = typeof infoCycles === "number" && Number.isFinite(infoCycles);

  if (markOutUpper === "WEAK_FLAT_MARKOUTS" && infoCyclesKnown && infoCycles === 0) {
    vio.push("UNSUITABLE_NO_INFORMATIVE_MARKOUTS");
    push("weak_flat_markouts_and_zero_informative_cycles_block_microcapital_proof");
  }

  const quotedBidAsk = input.bestBid != null || input.bestAsk != null;
  const otherPriceHints =
    input.lastPrice != null || input.spread != null || Boolean(input.priceSource?.trim());

  const bids = trustworthyBidAsk(input.bestBid, input.bestAsk);

  if (quotedBidAsk) {
    if (!bids) {
      vio.push("UNSUITABLE_LIQUIDITY");
      push("dual_sided_liquidity_read_not_possible");
    }
  } else if (otherPriceHints) {
    vio.push("UNSUITABLE_LIQUIDITY");
    push("partial_price_signals_without_dual_sided_best_bid_ask");
  }

  if (vio.length > 0) {
    const primary = pickWorst(vio);
    return {
      suitabilityVerdict: primary,
      reasons,
      canUseForMicrocapitalCandidate: false,
    };
  }

  const activePositive = input.active !== false && !(input.closed ?? false) && !(input.resolved ?? false);

  let microOk =
    activePositive &&
    bids !== null &&
    !nearPinnedPrice(input.lastPrice) &&
    markOutUpper.length > 0 &&
    markOutUpper !== "WEAK_FLAT_MARKOUTS";

  microOk =
    microOk &&
    typeof infoCycles === "number" &&
    infoCycles > 0 &&
    fin(input.volume)! !== null &&
    fin(input.volume)! > 0;

  microOk =
    microOk &&
    !expiredTemporal &&
    !(endAbs && now.getTime() > endAbs.getTime()) &&
    !aprilNamedWindowEnded(qLine, now);

  if (!microOk) {
    reasons.length = 0;
    push("default_deny_or_insufficient_proof_for_positive_observation_candidate");
    if (!bids) push("dual_sided_quote_required");
    if (!(typeof infoCycles === "number" && infoCycles > 0)) push("informative_markout_positive_required");
    if (!(fin(input.volume)! !== null && fin(input.volume)! > 0)) push("liquidity_floor_not_met");
    if (expiredTemporal) push("evaluation_time_after_event_window");

    return {
      suitabilityVerdict: "UNSUITABLE_MISSING_DATA",
      reasons,
      canUseForMicrocapitalCandidate: false,
    };
  }

  return {
    suitabilityVerdict: "SUITABLE_FOR_PAPER_OBSERVATION",
    reasons: [...reasons, "minimal_paper_candidate_checks_pass"],
    canUseForMicrocapitalCandidate: true,
  };
}
