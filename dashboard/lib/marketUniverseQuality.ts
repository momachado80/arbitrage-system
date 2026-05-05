/**
 * Filtro read-only de qualidade do universo de mercados antes de Observation Window Scout.
 *
 * Função pura: sem rede, sem execução real, sem capital, sem persistência. `canUseForMicrocapitalCandidate`
 * permanece sempre false nesta fase.
 *
 * Regra-mãe: mercado primeiro, código depois. Para campo de prova inicial, preferir esportes objetivos
 * com horizonte próximo, mid saudável e livro two-sided. Rejeitar memes/absurdidade,
 * política/legal, cauda extrema e horizonte longo sem catalisador.
 */
import { parsedEndUtc, parsedNowUtc } from "./marketSuitabilityGate";

export type UniverseQualityVerdict =
  | "CLEAN_OBSERVATION_UNIVERSE"
  | "ACCEPTABLE_BUT_SECONDARY"
  | "REJECT_AMBIGUOUS"
  | "REJECT_MEME_OR_ABSURD"
  | "REJECT_POLITICAL_LEGAL"
  | "REJECT_LONG_HORIZON_NO_CATALYST"
  | "REJECT_TAIL_OR_TICK_FENCE"
  | "REJECT_NOT_SUITABLE";

export type EventProximityClass =
  | "very_near"
  | "near"
  | "moderate"
  | "long"
  | "very_long"
  | "none";

export interface MarketUniverseQualityInput {
  marketId?: string | null;
  question?: string | null;
  slug?: string | null;
  category?: string | null;
  tags?: unknown;
  endDate?: string | null;
  nowIso?: string | null;
  active?: unknown;
  closed?: unknown;
  resolved?: unknown;
  liquidity?: number | null;
  volume?: number | null;
  bestBid?: number | null;
  bestAsk?: number | null;
  mid?: number | null;
  spread?: number | null;
  suitabilityVerdict?: string | null;
  economicRankScore?: number | null;
  clobBookStructure?: string | null;
  strongTwoSidedBook?: boolean;
  healthyMidRange?: boolean;
  microProbabilityTail?: boolean;
  quotesNearTickFence?: boolean;
}

export interface MarketUniverseQualityResult {
  universeQualityVerdict: UniverseQualityVerdict;
  universeQualityScore: number;
  reasons: string[];
  disqualifiers: string[];
  isCleanForObservationScout: boolean;
  ambiguityRisk: number;
  memeOrAbsurdityRisk: number;
  politicalLegalRisk: number;
  longHorizonNoCatalystRisk: number;
  tailRisk: number;
  eventProximityClass: EventProximityClass;
  canUseForPaperShadowCandidate: boolean;
  canUseForMicrocapitalCandidate: false;
}

const MEME_ANCHOR_PATTERNS: { name: string; rx: RegExp }[] = [
  { name: "jesus_christ_return", rx: /\bjesus\s+christ\b/i },
  { name: "second_coming", rx: /\bsecond\s+coming\b/i },
  { name: "gta_vi_anchor", rx: /\b(gta\s*(vi|6)|grand\s+theft\s+auto\s*(vi|6))\b/i },
  { name: "bitcoin_to_one_million", rx: /\bbitcoin\b[^?]*\$?\s*1\s*(m|million|mm|mn)\b/i },
  { name: "ethereum_to_million", rx: /\bethereum\b[^?]*\$?\s*1\s*(m|million)\b/i },
  { name: "bigfoot", rx: /\bbig[\s-]?foot\b/i },
  { name: "alien_disclosure", rx: /\balien(s)?\b/i },
  { name: "ufo_disclosure", rx: /\bufo\b/i },
  { name: "world_ending", rx: /\b(end\s+of\s+the\s+world|apocalypse|rapture)\b/i },
  { name: "before_anchor_pop_culture", rx: /\bbefore\s+(taylor\s+swift|kanye|elon|musk\s+becomes)\b/i },
];

const POLITICAL_LEGAL_PATTERNS: { name: string; rx: RegExp }[] = [
  { name: "trump_named", rx: /\btrump\b/i },
  { name: "biden_named", rx: /\bbiden\b/i },
  { name: "harris_named", rx: /\bharris\b/i },
  { name: "obama_named", rx: /\bobama\b/i },
  { name: "president_term", rx: /\bpresident(ial)?\b/i },
  { name: "primary_election", rx: /\b(primaries|primary\s+election)\b/i },
  { name: "election", rx: /\belection(s)?\b/i },
  { name: "impeachment", rx: /\bimpeach(ed|ment)?\b/i },
  { name: "supreme_court", rx: /\b(supreme\s+court|scotus)\b/i },
  { name: "court_ruling", rx: /\b(court\s+ruling|appellate|federal\s+court)\b/i },
  { name: "lawsuit", rx: /\blaw\s*suit\b/i },
  { name: "indictment", rx: /\bindict(ed|ment)?\b/i },
  { name: "congress_term", rx: /\b(congress|senate|house\s+of\s+representatives|speaker\s+of\s+the\s+house)\b/i },
  { name: "war_conflict", rx: /\bwar\b/i },
  { name: "ceasefire", rx: /\bceasefire\b/i },
  { name: "invasion", rx: /\binvasion\b/i },
  { name: "putin", rx: /\bputin\b/i },
  { name: "zelensky", rx: /\bzelensky\b/i },
  { name: "kim_jong", rx: /\bkim\s+jong\b/i },
  { name: "netanyahu", rx: /\bnetanyahu\b/i },
];

const SPORT_KEYWORDS: RegExp[] = [
  /\bnba\b/i, /\bnfl\b/i, /\bmlb\b/i, /\bnhl\b/i, /\bmls\b/i, /\bncaa\b/i,
  /\bfifa\b/i, /\bworld\s+cup\b/i, /\bchampions\s+league\b/i, /\bpremier\s+league\b/i,
  /\bla\s+liga\b/i, /\bbundesliga\b/i, /\bserie\s+a\b/i, /\beuropa\s+league\b/i,
  /\bfinals\b/i, /\bchampionship\b/i, /\bsuper\s+bowl\b/i, /\bworld\s+series\b/i,
  /\bstanley\s+cup\b/i, /\bgrand\s+slam\b/i, /\bwimbledon\b/i, /\bus\s+open\b/i,
  /\bfrench\s+open\b/i, /\baustralian\s+open\b/i, /\bolympics?\b/i, /\bf1\b/i,
  /\bnascar\b/i, /\bufc\b/i, /\bcricket\b/i, /\btour\s+de\s+france\b/i,
  /\bmasters\b/i, /\bplayoffs?\b/i, /\bcup\s+final\b/i, /\bmvp\b/i,
];

const FINANCIAL_EVENT_KEYWORDS: RegExp[] = [
  /\bearnings\b/i, /\bipo\b/i, /\bfomc\b/i, /\bfederal\s+reserve\b/i,
  /\binterest\s+rate\b/i, /\brate\s+hike\b/i, /\brate\s+cut\b/i,
  /\bhalving\b/i, /\bhard\s+fork\b/i, /\bmainnet\b/i, /\blisting\b/i,
  /\binflation\s+report\b/i, /\bcpi\b/i, /\bppi\b/i, /\bnfp\b/i,
];

const MS_DAY = 86_400_000;

function fin(x: unknown): number | null {
  if (typeof x === "number" && Number.isFinite(x)) return x;
  if (typeof x === "string" && x.trim()) {
    const n = parseFloat(x);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function tagsToString(raw: unknown): string {
  if (raw == null) return "";
  if (typeof raw === "string") return raw;
  if (Array.isArray(raw)) return raw.map(v => (typeof v === "string" ? v : JSON.stringify(v))).join(" ");
  if (typeof raw === "object") {
    try {
      return JSON.stringify(raw);
    } catch {
      return "";
    }
  }
  return String(raw);
}

function composeText(input: MarketUniverseQualityInput): string {
  const parts = [
    input.question ?? "",
    input.slug ?? "",
    input.category ?? "",
    tagsToString(input.tags),
  ];
  return parts.filter(Boolean).join(" ").toLowerCase();
}

function detectMemeOrAbsurdity(text: string): { hits: string[]; risk: number } {
  const hits: string[] = [];
  for (const p of MEME_ANCHOR_PATTERNS) {
    if (p.rx.test(text)) hits.push(p.name);
  }
  const risk = hits.length === 0 ? 0 : Math.min(1, 0.55 + hits.length * 0.15);
  return { hits, risk };
}

function detectPoliticalLegal(text: string): { hits: string[]; risk: number } {
  const hits: string[] = [];
  for (const p of POLITICAL_LEGAL_PATTERNS) {
    if (p.rx.test(text)) hits.push(p.name);
  }
  const risk = hits.length === 0 ? 0 : Math.min(1, 0.6 + hits.length * 0.15);
  return { hits, risk };
}

function hasAny(text: string, patterns: RegExp[]): boolean {
  for (const rx of patterns) if (rx.test(text)) return true;
  return false;
}

function eventProximity(endDate: string | null | undefined, nowIso: string | null | undefined): {
  cls: EventProximityClass;
  daysToEnd: number | null;
} {
  const now = parsedNowUtc(nowIso ?? undefined);
  const end = parsedEndUtc(endDate ?? null);
  if (!end || !Number.isFinite(end.getTime())) return { cls: "none", daysToEnd: null };
  const diffMs = end.getTime() - now.getTime();
  const days = diffMs / MS_DAY;
  if (!Number.isFinite(days)) return { cls: "none", daysToEnd: null };
  if (days <= 14) return { cls: "very_near", daysToEnd: days };
  if (days <= 60) return { cls: "near", daysToEnd: days };
  if (days <= 180) return { cls: "moderate", daysToEnd: days };
  if (days <= 365) return { cls: "long", daysToEnd: days };
  return { cls: "very_long", daysToEnd: days };
}

function deriveMidSpread(input: MarketUniverseQualityInput): { mid: number | null; spread: number | null } {
  const explicitMid = fin(input.mid);
  const explicitSpread = fin(input.spread);
  if (explicitMid !== null) {
    return { mid: explicitMid, spread: explicitSpread };
  }
  const bid = fin(input.bestBid);
  const ask = fin(input.bestAsk);
  if (bid !== null && ask !== null && bid > 0 && ask > 0 && ask >= bid) {
    return { mid: (bid + ask) / 2, spread: ask - bid };
  }
  return { mid: null, spread: explicitSpread };
}

function nearTickFence(bid: number | null, ask: number | null): boolean {
  const b = bid;
  const a = ask;
  if (b !== null && b <= 0.003) return true;
  if (a !== null && a >= 0.997) return true;
  if (b !== null && b >= 0.997) return true;
  if (a !== null && a <= 0.003) return true;
  return false;
}

function tailRiskScore(mid: number | null, bid: number | null, ask: number | null, hintTail: boolean | undefined): number {
  if (hintTail === true) return 1;
  if (mid === null) return 0.4;
  if (mid < 0.06 || mid > 0.94) return 1;
  if (mid < 0.10 || mid > 0.90) return 0.55;
  if (nearTickFence(bid, ask)) return 0.8;
  return 0;
}

function isExpiredOrInactive(input: MarketUniverseQualityInput, nowIso: string | null | undefined): boolean {
  if (input.closed === true || input.resolved === true) return true;
  if (input.active === false) return true;
  const end = parsedEndUtc(input.endDate ?? null);
  const now = parsedNowUtc(nowIso ?? undefined);
  if (end && Number.isFinite(end.getTime()) && now.getTime() > end.getTime()) return true;
  return false;
}

function pushUnique(list: string[], v: string): void {
  if (!list.includes(v)) list.push(v);
}

/**
 * Avalia se um candidato a observação faz parte de um universo limpo para campo de prova inicial.
 * Sem rede, sem execução real, sem capital. `canUseForMicrocapitalCandidate` sempre false.
 */
export function evaluateMarketUniverseQuality(input: MarketUniverseQualityInput): MarketUniverseQualityResult {
  const reasons: string[] = [];
  const disqualifiers: string[] = [];
  const text = composeText(input);

  const { mid, spread } = deriveMidSpread(input);
  const bid = fin(input.bestBid);
  const ask = fin(input.bestAsk);
  const liquidity = fin(input.liquidity) ?? 0;
  const volume = fin(input.volume) ?? 0;
  const proximity = eventProximity(input.endDate, input.nowIso);
  const expired = isExpiredOrInactive(input, input.nowIso);

  const meme = detectMemeOrAbsurdity(text);
  const polLegal = detectPoliticalLegal(text);
  const hasSport = hasAny(text, SPORT_KEYWORDS);
  const hasFinEvent = hasAny(text, FINANCIAL_EVENT_KEYWORDS);
  const tail = tailRiskScore(mid, bid, ask, input.microProbabilityTail);

  let longHorizonNoCatalystRisk = 0;
  if (proximity.cls === "very_long") longHorizonNoCatalystRisk = 1;
  else if (proximity.cls === "long" && !hasSport && !hasFinEvent) longHorizonNoCatalystRisk = 0.85;
  else if (proximity.cls === "moderate" && !hasSport && !hasFinEvent) longHorizonNoCatalystRisk = 0.4;
  else if (proximity.cls === "none") longHorizonNoCatalystRisk = 0.7;

  const ambiguityRisk =
    proximity.cls === "none"
      ? 0.6
      : proximity.cls === "very_long" && !hasSport && !hasFinEvent
        ? 0.5
        : 0;

  const finalize = (
    verdict: UniverseQualityVerdict,
    score: number,
    isClean: boolean,
    paperOk: boolean,
  ): MarketUniverseQualityResult => ({
    universeQualityVerdict: verdict,
    universeQualityScore: Math.round(score * 100) / 100,
    reasons,
    disqualifiers,
    isCleanForObservationScout: isClean,
    ambiguityRisk: Math.round(ambiguityRisk * 100) / 100,
    memeOrAbsurdityRisk: Math.round(meme.risk * 100) / 100,
    politicalLegalRisk: Math.round(polLegal.risk * 100) / 100,
    longHorizonNoCatalystRisk: Math.round(longHorizonNoCatalystRisk * 100) / 100,
    tailRisk: Math.round(tail * 100) / 100,
    eventProximityClass: proximity.cls,
    canUseForPaperShadowCandidate: paperOk,
    canUseForMicrocapitalCandidate: false,
  });

  if (expired) {
    pushUnique(disqualifiers, "market_closed_resolved_expired_or_inactive");
    pushUnique(reasons, "market_terminal_state_blocks_observation_universe");
    return finalize("REJECT_NOT_SUITABLE", 0, false, false);
  }

  const suitVerdict = (input.suitabilityVerdict ?? "").trim();
  if (suitVerdict !== "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION") {
    pushUnique(disqualifiers, `suitability_not_paper_shadow:${suitVerdict || "missing"}`);
    pushUnique(reasons, "market_did_not_pass_suitability_gate");
    return finalize("REJECT_NOT_SUITABLE", 0, false, false);
  }

  if (polLegal.hits.length > 0) {
    for (const h of polLegal.hits) pushUnique(disqualifiers, `political_legal:${h}`);
    pushUnique(reasons, "political_legal_topic_blocks_initial_proof_universe");
    return finalize("REJECT_POLITICAL_LEGAL", 0, false, false);
  }

  if (meme.hits.length > 0) {
    for (const h of meme.hits) pushUnique(disqualifiers, `meme_or_absurd:${h}`);
    pushUnique(reasons, "meme_or_absurd_anchor_blocks_initial_proof_universe");
    return finalize("REJECT_MEME_OR_ABSURD", 0, false, false);
  }

  if (tail >= 1) {
    if (mid !== null && (mid < 0.06 || mid > 0.94)) pushUnique(disqualifiers, "tail_extreme_mid_outside_safe_band");
    if (nearTickFence(bid, ask)) pushUnique(disqualifiers, "quotes_near_tick_fence_extreme");
    if (input.microProbabilityTail === true) pushUnique(disqualifiers, "microProbabilityTail_hint_true");
    pushUnique(reasons, "tail_or_tick_fence_blocks_clean_universe");
    return finalize("REJECT_TAIL_OR_TICK_FENCE", 0, false, false);
  }

  if (proximity.cls === "very_long") {
    pushUnique(disqualifiers, "horizon_very_long_no_catalyst_proximate");
    pushUnique(reasons, "long_horizon_without_catalyst_blocks_initial_proof_universe");
    return finalize("REJECT_LONG_HORIZON_NO_CATALYST", 0, false, false);
  }
  if (proximity.cls === "long" && !hasSport && !hasFinEvent) {
    pushUnique(disqualifiers, "horizon_long_no_sport_or_financial_catalyst");
    pushUnique(reasons, "long_horizon_without_catalyst_blocks_initial_proof_universe");
    return finalize("REJECT_LONG_HORIZON_NO_CATALYST", 0, false, false);
  }
  if (proximity.cls === "none") {
    pushUnique(disqualifiers, "missing_endDate_for_proof_horizon");
    pushUnique(reasons, "no_horizon_blocks_clean_universe");
    return finalize("REJECT_AMBIGUOUS", 0, false, false);
  }

  let score = 0;
  pushUnique(reasons, "no_disqualifiers_detected");

  if (mid !== null && Number.isFinite(mid) && mid >= 0.10 && mid <= 0.80) {
    score += 220;
    pushUnique(reasons, "healthy_mid_in_band_0p10_0p80");
  } else if (mid !== null && Number.isFinite(mid) && mid >= 0.06 && mid <= 0.94) {
    score += 90;
    pushUnique(reasons, "mid_acceptable_but_secondary");
  } else {
    score += 20;
  }

  if (spread !== null && Number.isFinite(spread)) {
    if (spread <= 0.02) {
      score += 90;
      pushUnique(reasons, "narrow_spread_below_2pct");
    } else if (spread <= 0.05) {
      score += 50;
      pushUnique(reasons, "narrow_spread_below_5pct");
    } else if (spread > 0.10) {
      score -= 30;
      pushUnique(reasons, "wide_spread_above_10pct");
    }
  }

  if (input.strongTwoSidedBook === true) {
    score += 80;
    pushUnique(reasons, "strong_two_sided_book_hint");
  } else if (input.clobBookStructure === "two_sided") {
    score += 40;
    pushUnique(reasons, "two_sided_book_observed");
  }

  if (hasSport) {
    score += 110;
    pushUnique(reasons, "sport_keyword_objective_resolution");
  }
  if (hasFinEvent) {
    score += 60;
    pushUnique(reasons, "financial_event_keyword_objective_resolution");
  }

  if (proximity.cls === "very_near") {
    score += 90;
    pushUnique(reasons, "horizon_very_near_event_proximate");
  } else if (proximity.cls === "near") {
    score += 70;
    pushUnique(reasons, "horizon_near_event_proximate");
  } else if (proximity.cls === "moderate") {
    score += 30;
    pushUnique(reasons, "horizon_moderate_acceptable");
  } else if (proximity.cls === "long") {
    score += 5;
    pushUnique(reasons, "horizon_long_with_catalyst_acceptable");
  }

  if (Number.isFinite(liquidity) && liquidity > 0) score += Math.min(60, Math.log1p(Math.max(liquidity, 0)) * 6);
  if (Number.isFinite(volume) && volume > 0) score += Math.min(40, Math.log1p(Math.max(volume, 0)) * 3);

  if (typeof input.economicRankScore === "number" && Number.isFinite(input.economicRankScore)) {
    score += Math.max(-50, Math.min(80, input.economicRankScore * 0.18));
  }

  let verdict: UniverseQualityVerdict;
  let isClean: boolean;
  let paperOk: boolean;

  const passesCleanProfile =
    mid !== null &&
    mid >= 0.10 &&
    mid <= 0.80 &&
    (input.strongTwoSidedBook === true || input.clobBookStructure === "two_sided") &&
    (spread === null || spread <= 0.05) &&
    (hasSport || hasFinEvent) &&
    proximity.cls !== "long" &&
    score >= 320;

  if (passesCleanProfile) {
    verdict = "CLEAN_OBSERVATION_UNIVERSE";
    isClean = true;
    paperOk = true;
    pushUnique(reasons, "clean_for_observation_scout_objective_resolution_and_proximate");
  } else {
    verdict = "ACCEPTABLE_BUT_SECONDARY";
    isClean = false;
    paperOk = true;
    pushUnique(reasons, "acceptable_but_secondary_passes_basic_universe_filter");
  }

  return finalize(verdict, score, isClean, paperOk);
}

/** Predicado de conveniência para enriquecimento e ordenação. */
export function isCleanRow(row: Record<string, unknown>): boolean {
  const v = row.universeQualityVerdict;
  return v === "CLEAN_OBSERVATION_UNIVERSE";
}
