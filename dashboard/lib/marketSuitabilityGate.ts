/**
 * Gate read-only de adequação de mercado antes de worker, watcher, readiness económico ou microcapital.
 * Função pura; sem rede, sem execução, sem submissão real.
 *
 * Regra-mãe: mercado primeiro, código depois. Em dúvida, negar candidatura.
 */

export type MarketSuitabilityVerdict =
  | "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION"
  | "UNSUITABLE_EXPIRED"
  | "UNSUITABLE_CLOSED_OR_RESOLVED"
  | "UNSUITABLE_PRICE_PINNED"
  | "UNSUITABLE_NO_BOOK"
  | "UNSUITABLE_MISSING_DATA"
  | "UNSUITABLE_LOW_LIQUIDITY"
  | "UNSUITABLE_WEAK_FLAT_MARKOUTS";

/** Estrutura do livro CLOB observada (prova mínima de profundidade). */
export type ClobBookStructureHint =
  | "two_sided"
  | "asks_only"
  | "bids_only"
  | "empty"
  | "unusable"
  | "unknown"
  | "skipped";

export interface MarketSuitabilityInput {
  marketId?: string;
  title?: string;
  question?: string;
  endDate?: string | null;
  end_date?: string | null;
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
  outcomePrices?: unknown;
  clobTokenIds?: unknown;
  conditionId?: string | null;
  updatedAt?: string | null;
  category?: string | null;
  tags?: unknown;
  markoutInformativenessVerdict?: string | null;
  informativeMarkoutCycleCount?: number | null;
  nowIso?: string;
  /** Quando o script de discovery observa o livro CLOB. */
  clobBookStructure?: ClobBookStructureHint;
}

export interface MarketSuitabilityEvalResult {
  suitabilityVerdict: MarketSuitabilityVerdict;
  reasons: string[];
  canUseForPaperShadowCandidate: boolean;
  /** Fase actual: sempre false — microcapital exige prova adicional e autorização explícita. */
  canUseForMicrocapitalCandidate: boolean;
}

const PINNED_LOW = 0.001;
const PINNED_HIGH = 0.999;
const PRICE_PIN_TOL = 1.2e-4;

const MIN_VOLUME_USD = 2_500;
const MIN_LIQUIDITY_USD = 750;

const ORDER: Record<MarketSuitabilityVerdict, number> = {
  UNSUITABLE_CLOSED_OR_RESOLVED: 10,
  UNSUITABLE_EXPIRED: 20,
  UNSUITABLE_WEAK_FLAT_MARKOUTS: 30,
  UNSUITABLE_PRICE_PINNED: 40,
  UNSUITABLE_MISSING_DATA: 50,
  UNSUITABLE_NO_BOOK: 60,
  UNSUITABLE_LOW_LIQUIDITY: 70,
  SUITABLE_FOR_PAPER_SHADOW_OBSERVATION: 1000,
};

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

/** Heurística “April”: após 1 Mai UTC do ano pertinente, janela nomeada considera-se encerrada. */
export function aprilNamedWindowEnded(questionOrTitle: string | undefined, now: Date): boolean {
  const t = questionOrTitle?.trim();
  if (!t || !/\bApril\b/i.test(t)) return false;
  const m = t.match(/\b(20\d{2})\b/);
  const year = m ? Number(m[1]) : now.getUTCFullYear();
  return now.getTime() >= Date.UTC(year, 4, 1);
}

function fin(x: unknown): number | null {
  if (typeof x === "number" && Number.isFinite(x)) return x;
  if (typeof x === "string" && x.trim()) {
    const n = parseFloat(x);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function pickWorst(vs: MarketSuitabilityVerdict[]): MarketSuitabilityVerdict {
  let out: MarketSuitabilityVerdict = "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION";
  let best = Infinity;
  for (const v of vs) {
    const r = ORDER[v];
    if (r < best) {
      best = r;
      out = v;
    }
  }
  return out;
}

/** Dois lados positivos e fora das grelhas ~0/~1 quando ambos existem em número. */
export function trustworthyBidAsk(bidRaw: unknown, askRaw: unknown): { bid: number; ask: number } | null {
  const bid = fin(bidRaw);
  const ask = fin(askRaw);
  if (bid === null || ask === null || bid <= 0 || ask <= 0 || ask < bid * 0.999) return null;
  if (nearPinnedPrice(bid) || nearPinnedPrice(ask)) return null;
  return { bid, ask };
}

function hasMinimalSignals(input: MarketSuitabilityInput): boolean {
  const q = `${input.title ?? ""} ${input.question ?? ""}`.trim();
  return Boolean(
    q ||
      input.marketId?.trim() ||
      input.endDate !== undefined ||
      input.end_date !== undefined ||
      input.closed !== undefined ||
      input.resolved !== undefined ||
      input.active !== undefined ||
      input.bestBid != null ||
      input.bestAsk != null ||
      input.lastPrice != null ||
      input.volume != null ||
      input.liquidity != null ||
      input.clobBookStructure !== undefined ||
      input.markoutInformativenessVerdict !== undefined,
  );
}

function bookStructureInsufficient(struct?: ClobBookStructureHint): boolean {
  if (struct === undefined) return false;
  if (struct === "skipped") return false;
  if (struct === "two_sided") return false;
  return true;
}

/**
 * Avaliação read-only antes de pipeline operacional (worker / watcher / microcapital).
 * `canUseForMicrocapitalCandidate` permanece false nesta fase do produto.
 */
export function evaluateMarketSuitability(input: MarketSuitabilityInput): MarketSuitabilityEvalResult {
  const reasons: string[] = [];
  const push = (...xs: string[]): void => {
    for (const x of xs) {
      if (!reasons.includes(x)) reasons.push(x);
    }
  };

  const now = parsedNowUtc(input.nowIso);
  const violations: MarketSuitabilityVerdict[] = [];

  const finalize = (verdict: MarketSuitabilityVerdict, rs: string[]): MarketSuitabilityEvalResult => ({
    suitabilityVerdict: verdict,
    reasons: rs,
    canUseForPaperShadowCandidate: verdict === "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION",
    canUseForMicrocapitalCandidate: false,
  });

  if (!hasMinimalSignals(input)) {
    return finalize("UNSUITABLE_MISSING_DATA", ["no_assessable_input"]);
  }

  const endRaw = input.endDate ?? input.end_date ?? null;
  const endParsed = parsedEndUtc(endRaw);
  const expiredByEnd =
    !!(endParsed && Number.isFinite(endParsed.getTime()) && now.getTime() > endParsed.getTime());

  const ql = input.question ?? input.title;
  const aprilExpired =
    !(input.closed ?? false) && !(input.resolved ?? false) && aprilNamedWindowEnded(ql, now);

  if (input.closed === true) {
    violations.push("UNSUITABLE_CLOSED_OR_RESOLVED");
    push("closed:true");
  }
  if (input.resolved === true) {
    violations.push("UNSUITABLE_CLOSED_OR_RESOLVED");
    push("resolved:true");
  }

  const markUpper = input.markoutInformativenessVerdict?.trim().toUpperCase() ?? "";
  if (markUpper === "WEAK_FLAT_MARKOUTS") {
    violations.push("UNSUITABLE_WEAK_FLAT_MARKOUTS");
    push("markout_verdict_weak_flat_markouts_blocks_economic_suitability");
  }

  if (!(input.closed ?? false) && !(input.resolved ?? false) && expiredByEnd) {
    violations.push("UNSUITABLE_EXPIRED");
    push("endDate_passed_relative_to_nowIso");
  }
  if (!(input.closed ?? false) && !(input.resolved ?? false) && aprilExpired) {
    violations.push("UNSUITABLE_EXPIRED");
    push("temporal_named_april_bucket_considered_finished");
  }

  const endProvided = !!(endRaw && String(endRaw).trim());
  const allowMissingEndTemporal =
    aprilExpired || expiredByEnd || (input.closed ?? false) || (input.resolved ?? false);

  /** Mercado declarado vivo sem data de horizonte explícita: demasiado fragil para observação económica. */
  const shouldRequireEndWindow =
    !(input.closed ?? false) && !(input.resolved ?? false) && !(input.active === false);

  if (shouldRequireEndWindow && !endProvided && !allowMissingEndTemporal) {
    violations.push("UNSUITABLE_MISSING_DATA");
    push("missing_endDate_for_forward_horizon_proof");
  }

  if (
    nearPinnedPrice(input.bestBid) ||
    nearPinnedPrice(input.bestAsk) ||
    nearPinnedPrice(input.lastPrice)
  ) {
    violations.push("UNSUITABLE_PRICE_PINNED");
    push("quotes_or_last_near_probability_floor_ceiling");
  }

  const pair = trustworthyBidAsk(input.bestBid, input.bestAsk);
  const hasQuoteHints =
    input.bestBid != null || input.bestAsk != null || input.lastPrice != null || input.spread != null;

  if (hasQuoteHints && pair === null) {
    violations.push("UNSUITABLE_NO_BOOK");
    push("trustworthy_bid_ask_pair_missing");
  } else if (!hasQuoteHints) {
    violations.push("UNSUITABLE_NO_BOOK");
    push("missing_price_quotes_entirely");
  }

  if (bookStructureInsufficient(input.clobBookStructure)) {
    violations.push("UNSUITABLE_NO_BOOK");
    push(`insufficient_order_book_structure:${input.clobBookStructure ?? "n/a"}`);
  }

  const vol = fin(input.volume);
  const liq = fin(input.liquidity);

  if (pair !== null && vol !== null && Number.isFinite(vol) && vol < MIN_VOLUME_USD) {
    violations.push("UNSUITABLE_LOW_LIQUIDITY");
    push(`volume_below_min_usd_approx:${MIN_VOLUME_USD}`);
  }

  if (pair !== null && liq !== null && Number.isFinite(liq) && liq < MIN_LIQUIDITY_USD) {
    violations.push("UNSUITABLE_LOW_LIQUIDITY");
    push(`liquidity_below_min_usd_approx:${MIN_LIQUIDITY_USD}`);
  }

  const primaryViol = pickWorst(violations.length ? violations : ["SUITABLE_FOR_PAPER_SHADOW_OBSERVATION"]);
  const hasHardBlocks = violations.length > 0;

  if (!hasHardBlocks) {
    if (input.active !== true) {
      push("inactive_or_active_flag_missing");
      return finalize("UNSUITABLE_MISSING_DATA", reasons);
    }
    if (pair === null) {
      violations.push("UNSUITABLE_NO_BOOK");
      push("trustworthy_quotes_validation_failed_terminal");
      return finalize(pickWorst(violations), reasons);
    }
    const endFut = parsedEndUtc(endRaw);
    if (!(endFut && Number.isFinite(endFut.getTime()) && now.getTime() <= endFut.getTime())) {
      violations.push("UNSUITABLE_EXPIRED");
      push("eligible_row_requires_future_end_window");
      return finalize(pickWorst(violations), reasons);
    }
    push("paper_shadow_readonly_checks_pass_no_exec");
    return finalize("SUITABLE_FOR_PAPER_SHADOW_OBSERVATION", reasons);
  }

  return finalize(primaryViol, reasons);
}
