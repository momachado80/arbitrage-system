/**
 * Negative-Risk Conversion Pilot — eventos Gamma com negRisk ativo; economia mecânica
 * de conversão (proxy, sem trading live). Universo estreito.
 */

const GAMMA_EVENTS = "https://gamma-api.polymarket.com/events";
const FETCH_TIMEOUT_MS = 14_000;
const MAX_PAGES = 6;
const PAGE_LIMIT = 80;
const MAX_PILOT_SETS = 8;
const VIABLE_NET = 0.002;

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function num(x: unknown): number {
  const n = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(n) ? n : NaN;
}

function parseJsonStringArray(s: unknown): string[] {
  if (typeof s !== "string") return [];
  try {
    const a = JSON.parse(s) as unknown;
    if (!Array.isArray(a)) return [];
    return a.map(x => String(x).trim()).filter(Boolean);
  } catch {
    return [];
  }
}

function parseJsonNumberArray(s: unknown): number[] {
  if (typeof s !== "string") return [];
  try {
    const a = JSON.parse(s) as unknown;
    if (!Array.isArray(a)) return [];
    return a.map(x => num(x)).filter(x => Number.isFinite(x));
  } catch {
    return [];
  }
}

function isPlaceholderOutcome(name: string): boolean {
  const t = name.trim().toLowerCase();
  if (t.length < 2) return true;
  if (["tbd", "n/a", "na", "...", "—", "-"].includes(t)) return true;
  if (/^unnamed|^untitled|^placeholder|^outcome\s*\d+$/i.test(t)) return true;
  return false;
}

export type NegativeRiskPilotVerdict =
  | "no_viable_negative_risk_set"
  | "weak_negative_risk_candidate"
  | "one_viable_negative_risk_candidate"
  | "multiple_viable_negative_risk_candidates";

export type PilotVerdictPerSet = "positive_expected" | "marginal" | "not_viable";

export interface NegativeRiskConversionSetRow {
  eventId: string;
  familyId: string;
  marketIds: string[];
  negRiskAvailable: boolean;
  augmentedNegRisk: boolean;
  numberOfOutcomes: number;
  conversionStructureSummary: string;
  rawConversionEdge: number;
  estimatedEntryCost: number;
  estimatedConversionCost: number;
  estimatedExitCost: number;
  estimatedNetConversionEdge: number;
  capacityEstimate: number;
  executionFragility: "low" | "medium" | "high";
  pilotVerdictPerSet: PilotVerdictPerSet;
  supportingNote: string;
}

export interface StrongestConversionSet {
  eventId: string;
  familyId: string;
  estimatedNetConversionEdge: number;
  numberOfOutcomes: number;
  executionFragility: string;
}

export interface NegativeRiskConversionPilotDigest {
  probeVersion: "negative-risk-conversion-pilot-v1";
  readDisclaimer: string;
  negativeRiskPilotVerdict: NegativeRiskPilotVerdict;
  eventsEvaluated: number;
  eventsTradable: number;
  eventsWithPositiveNetConversionEdge: number;
  strongestConversionSets: StrongestConversionSet[];
  negativeRiskPilotSummaryLine: string;
  sets: NegativeRiskConversionSetRow[];
  computedAt: string;
}

interface GammaEventRaw {
  id?: unknown;
  slug?: unknown;
  title?: unknown;
  negRisk?: unknown;
  enableNegRisk?: unknown;
  negRiskAugmented?: unknown;
  enableOrderBook?: unknown;
  negRiskFeeBips?: unknown;
  markets?: unknown;
}

interface GammaMarketRaw {
  id?: unknown;
  question?: unknown;
  outcomes?: unknown;
  outcomePrices?: unknown;
  spread?: unknown;
  liquidity?: unknown;
  liquidityNum?: unknown;
  liquidityClob?: unknown;
  closed?: unknown;
  active?: unknown;
}

function yesPriceFromMarket(m: GammaMarketRaw): number | null {
  const prices = parseJsonNumberArray(m.outcomePrices);
  if (prices.length < 1) return null;
  return r6(Math.min(0.999, Math.max(0.0001, prices[0])));
}

function evaluateEventSet(ev: GammaEventRaw): NegativeRiskConversionSetRow | null {
  const eventId = String(ev.id ?? "");
  const familyId = typeof ev.slug === "string" && ev.slug ? ev.slug : eventId;
  const negRiskAvailable = ev.negRisk === true;
  const augmentedNegRisk = ev.enableNegRisk === true && ev.negRiskAugmented === true;
  if (!negRiskAvailable || ev.enableOrderBook !== true) return null;

  const marketsRaw = Array.isArray(ev.markets) ? (ev.markets as GammaMarketRaw[]) : [];
  const activeMarkets = marketsRaw.filter(
    m => m && m.closed !== true && m.active !== false && m.id != null,
  );
  if (activeMarkets.length < 2) return null;

  const marketIds: string[] = [];
  const yesPrices: number[] = [];
  const spreads: number[] = [];
  const liqs: number[] = [];

  for (const m of activeMarkets) {
    const oc = parseJsonStringArray(m.outcomes);
    if (oc.some(isPlaceholderOutcome)) {
      return null;
    }
    const y = yesPriceFromMarket(m);
    if (y == null) continue;
    marketIds.push(String(m.id));
    yesPrices.push(y);
    const sp = num(m.spread);
    if (Number.isFinite(sp) && sp >= 0) spreads.push(r6(Math.min(0.5, sp)));
    const L = num(m.liquidityNum ?? m.liquidityClob ?? m.liquidity);
    if (Number.isFinite(L) && L > 0) liqs.push(L);
  }

  if (marketIds.length < 2) return null;

  const sumYes = r6(yesPrices.reduce((a, b) => a + b, 0));
  const k = marketIds.length;
  const structuralSlack = r6(Math.abs(1 - sumYes));
  const rawConversionEdge = r6(Math.min(0.08, structuralSlack * 0.55));

  const meanSpread = spreads.length > 0 ? r6(spreads.reduce((a, b) => a + b, 0) / spreads.length) : 0.02;
  const estimatedEntryCost = r6(0.003 + meanSpread * 0.42);
  const feeBips = num(ev.negRiskFeeBips);
  const bips = Number.isFinite(feeBips) && feeBips > 0 ? feeBips : 40;
  const estimatedConversionCost = r6(bips / 10_000);
  const estimatedExitCost = r6(0.0032 + 0.00085 * Math.max(0, k - 2));

  const estimatedNetConversionEdge = r6(
    rawConversionEdge - estimatedEntryCost - estimatedConversionCost - estimatedExitCost,
  );

  const capacityEstimate =
    liqs.length > 0 ? r6(Math.min(...liqs)) : r6(num(activeMarkets[0]?.liquidity) || 0);

  let executionFragility: "low" | "medium" | "high" = "low";
  if (augmentedNegRisk) executionFragility = "high";
  else if (k < 4 || meanSpread > 0.055) executionFragility = "medium";

  let pilotVerdictPerSet: PilotVerdictPerSet;
  if (estimatedNetConversionEdge >= VIABLE_NET) pilotVerdictPerSet = "positive_expected";
  else if (estimatedNetConversionEdge > 0) pilotVerdictPerSet = "marginal";
  else pilotVerdictPerSet = "not_viable";

  const conversionStructureSummary = augmentedNegRisk
    ? `neg_risk_augmented|k=${k}|atomic_no_to_yes_bundle`
    : `neg_risk_standard|k=${k}|atomic_no_to_yes_bundle`;

  const supportingNote = [
    `sum_yes_implied=${sumYes}`,
    `slack_abs_1_minus_sum=${structuralSlack}`,
    `raw_edge=0.55*slack_capped_0.08`,
    `entry=0.003+0.42*mean_spread`,
    `conversion_fee_bips=${bips}`,
    `exit_proxy=${estimatedExitCost}`,
    `markets=${marketIds.join(",")}`,
  ].join("|");

  return {
    eventId,
    familyId,
    marketIds,
    negRiskAvailable,
    augmentedNegRisk,
    numberOfOutcomes: k,
    conversionStructureSummary,
    rawConversionEdge,
    estimatedEntryCost,
    estimatedConversionCost,
    estimatedExitCost,
    estimatedNetConversionEdge,
    capacityEstimate,
    executionFragility,
    pilotVerdictPerSet,
    supportingNote,
  };
}

async function fetchNegRiskEventPage(offset: number): Promise<GammaEventRaw[]> {
  const base = `${GAMMA_EVENTS}?active=true&closed=false&limit=${PAGE_LIMIT}&offset=${offset}&order=volume&ascending=false`;
  const tryUrl = async (url: string) =>
    fetch(url, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { Accept: "application/json" },
    });
  let res = await tryUrl(`${base}&negRisk=true`);
  if (!res.ok) {
    res = await tryUrl(base);
  }
  if (!res.ok) return [];
  const j = (await res.json()) as unknown;
  return Array.isArray(j) ? (j as GammaEventRaw[]) : [];
}

export async function buildNegativeRiskConversionPilotDigest(): Promise<NegativeRiskConversionPilotDigest> {
  const sets: NegativeRiskConversionSetRow[] = [];
  let eventsNegRiskSeen = 0;

  outer: for (let page = 0; page < MAX_PAGES; page++) {
    const offset = page * PAGE_LIMIT;
    const batch = await fetchNegRiskEventPage(offset);
    if (batch.length === 0) break;
    for (const ev of batch) {
      if (ev.negRisk === true) eventsNegRiskSeen++;
      if (ev.negRisk !== true) continue;
      const row = evaluateEventSet(ev);
      if (!row) continue;
      sets.push(row);
      if (sets.length >= MAX_PILOT_SETS) break outer;
    }
    await new Promise(r => setTimeout(r, 120));
  }

  const eventsTradable = sets.length;
  const eventsWithPositiveNetConversionEdge = sets.filter(s => s.estimatedNetConversionEdge > 0).length;
  const viable = sets.filter(s => s.estimatedNetConversionEdge >= VIABLE_NET).length;
  const marginal = sets.filter(s => s.pilotVerdictPerSet === "marginal").length;

  let negativeRiskPilotVerdict: NegativeRiskPilotVerdict;
  if (eventsTradable === 0) {
    negativeRiskPilotVerdict = "no_viable_negative_risk_set";
  } else if (viable >= 2) {
    negativeRiskPilotVerdict = "multiple_viable_negative_risk_candidates";
  } else if (viable === 1) {
    negativeRiskPilotVerdict = "one_viable_negative_risk_candidate";
  } else if (marginal >= 1 || eventsWithPositiveNetConversionEdge >= 1) {
    negativeRiskPilotVerdict = "weak_negative_risk_candidate";
  } else {
    negativeRiskPilotVerdict = "no_viable_negative_risk_set";
  }

  const strongestConversionSets: StrongestConversionSet[] = [...sets]
    .sort((a, b) => b.estimatedNetConversionEdge - a.estimatedNetConversionEdge)
    .slice(0, 4)
    .map(s => ({
      eventId: s.eventId,
      familyId: s.familyId,
      estimatedNetConversionEdge: s.estimatedNetConversionEdge,
      numberOfOutcomes: s.numberOfOutcomes,
      executionFragility: s.executionFragility,
    }));

  const negativeRiskPilotSummaryLine = `neg_risk_conversion_pilot: verdict=${negativeRiskPilotVerdict} | neg_risk_events_seen=${eventsNegRiskSeen} tradable_sets=${eventsTradable} pos_net_edge=${eventsWithPositiveNetConversionEdge} viable_ge_${VIABLE_NET}=${viable} | top_net=${strongestConversionSets[0]?.estimatedNetConversionEdge ?? "n/a"}`;

  return {
    probeVersion: "negative-risk-conversion-pilot-v1",
    readDisclaimer:
      "Pilot venue-native: eventos Gamma com negRisk+order book; ignora mercados com outcomes placeholder (TBD, unnamed). Borda bruta = proxy de desvio |1−Σp_yes| (mutuamente exclusivos); custos de entrada/conversão/saída são proxies fixos + negRiskFeeBips quando existir. Sem ordens live; sem busca semântica ampla.",
    negativeRiskPilotVerdict,
    eventsEvaluated: eventsNegRiskSeen,
    eventsTradable,
    eventsWithPositiveNetConversionEdge,
    strongestConversionSets,
    negativeRiskPilotSummaryLine,
    sets,
    computedAt: new Date().toISOString(),
  };
}
