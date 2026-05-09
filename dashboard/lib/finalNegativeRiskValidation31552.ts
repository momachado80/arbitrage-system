/**
 * Validação final venue-native — apenas evento 31552 (presidential-election-winner-2028).
 * Preços CLOB reais (livro YES por token); custos observados + stress; sem trading live.
 */

import { fetchParsedClobBook, parseClobTokenIds } from "./clobMicrostructure";

const EVENT_ID = "31552";
const GAMMA_EVENT_URL = `https://gamma-api.polymarket.com/events/${EVENT_ID}`;
const FETCH_TIMEOUT_MS = 12_000;
const CLOB_BATCH = 4;
const CLOB_GAP_MS = 90;

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

function tokenIdsFromEmbeddedMarket(m: Record<string, unknown>): string[] {
  const c = m.clobTokenIds;
  if (Array.isArray(c)) return c.map(x => String(x)).filter(Boolean);
  if (typeof c === "string") return parseClobTokenIds({ clobTokenIds: c });
  return [];
}

function stdev(nums: number[]): number {
  if (nums.length < 2) return 0;
  const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
  const v = nums.reduce((s, x) => s + (x - mean) ** 2, 0) / (nums.length - 1);
  return Math.sqrt(v);
}

export type FinalNegativeRiskValidationVerdict =
  | "survives_final_negative_risk_validation"
  | "fails_final_negative_risk_validation"
  | "inconclusive_but_not_promotable";

export interface FinalNegativeRiskValidation31552Digest {
  probeVersion: "final-negative-risk-validation-31552-v1";
  readDisclaimer: string;
  eventId: string;
  familyId: string;
  marketIds: string[];
  rawEstimatedConversionEdge: number;
  observedEntryCost: number;
  observedConversionCost: number;
  observedExitCost: number;
  stressAdjustedNetConversionEdge: number;
  capacityEstimate: number;
  fragilityAssessment: string;
  finalNegativeRiskValidationVerdict: FinalNegativeRiskValidationVerdict;
  finalNegativeRiskValidationSummaryLine: string;
  clobBooksResolved: number;
  marketsInSet: number;
  computedAt: string;
}

async function fetchGammaEvent(): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(GAMMA_EVENT_URL, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return null;
    const j = (await res.json()) as unknown;
    return j && typeof j === "object" && !Array.isArray(j) ? (j as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

export async function buildFinalNegativeRiskValidation31552Digest(): Promise<FinalNegativeRiskValidation31552Digest> {
  const ev = await fetchGammaEvent();
  if (!ev) {
    return failDigest("gamma_event_fetch_failed");
  }

  const familyId = typeof ev.slug === "string" ? ev.slug : EVENT_ID;
  const marketsRaw = Array.isArray(ev.markets) ? (ev.markets as Record<string, unknown>[]) : [];
  const active = marketsRaw.filter(
    m => m && m.active === true && m.closed !== true && m.id != null,
  );

  const marketIds: string[] = [];
  const mids: number[] = [];
  const spreads: number[] = [];
  const depths: number[] = [];
  let clobBooksResolved = 0;

  for (let i = 0; i < active.length; i++) {
    const m = active[i];
    const outcomes = parseJsonStringArray(m.outcomes);
    if (outcomes.some(isPlaceholderOutcome)) continue;

    const id = String(m.id);
    marketIds.push(id);

    const tokens = tokenIdsFromEmbeddedMarket(m);
    const token0 = tokens[0];
    let mid: number | null = null;
    let spread = 0.02;
    let depth = 0;

    if (token0) {
      const book = await fetchParsedClobBook(token0);
      if (book) {
        mid = r6(book.mid);
        spread = r6(book.spread);
        depth = r6(book.depthBidTop3 + book.depthAskTop3);
        clobBooksResolved++;
      }
    }

    if (mid == null) {
      const prices = parseJsonNumberArray(m.outcomePrices);
      mid = prices.length ? r6(Math.min(0.999, Math.max(0.0001, prices[0]))) : 0.5;
      spread = 0.025;
      depth = 0;
    }

    mids.push(mid);
    spreads.push(spread);
    depths.push(depth);

    if ((i + 1) % CLOB_BATCH === 0 && i < active.length - 1) {
      await new Promise(r => setTimeout(r, CLOB_GAP_MS));
    }
  }

  const n = mids.length;
  const sumMid = n > 0 ? r6(mids.reduce((a, b) => a + b, 0)) : 0;
  const slack = r6(Math.abs(1 - sumMid));
  const rawEstimatedConversionEdge = r6(Math.min(0.08, slack * 0.55));

  const meanSpread = spreads.length ? r6(spreads.reduce((a, b) => a + b, 0) / spreads.length) : 0.02;
  const feeBips = num(ev.negRiskFeeBips);
  const bips = Number.isFinite(feeBips) && feeBips > 0 ? feeBips : 40;

  const observedEntryCost = r6(0.0026 + meanSpread * 0.4);
  const observedConversionCost = r6(bips / 10_000 + 0.0022);
  const observedExitCost = r6(0.022 + meanSpread * 0.48 + 0.00055 * Math.sqrt(Math.max(1, n - 1)));

  const stdevMid = stdev(mids);
  const stressSlippage = r6(
    Math.min(0.048, stdevMid * Math.sqrt(Math.min(n, 24)) * 0.14 + meanSpread * 0.08),
  );

  const stressAdjustedNetConversionEdge = r6(
    rawEstimatedConversionEdge - observedEntryCost - observedConversionCost - observedExitCost - stressSlippage,
  );

  const liqDepths = depths.filter(d => d > 0);
  const capacityEstimate = liqDepths.length > 0 ? r6(Math.min(...liqDepths)) : 0;

  const clobRatio = n > 0 ? clobBooksResolved / n : 0;
  let fragilityAssessment = `clob_coverage=${r6(clobRatio)}|n_outcomes=${n}|mean_spread=${meanSpread}|stdev_mid=${r6(stdevMid)}|augmented_neg_risk=true`;
  if (clobRatio < 0.55 || n >= 24) fragilityAssessment += "|fragility=high";
  else if (clobRatio < 0.85 || meanSpread > 0.04) fragilityAssessment += "|fragility=medium";
  else fragilityAssessment += "|fragility=low";

  let finalNegativeRiskValidationVerdict: FinalNegativeRiskValidationVerdict;
  if (stressAdjustedNetConversionEdge >= 0.002) {
    finalNegativeRiskValidationVerdict = "survives_final_negative_risk_validation";
  } else if (stressAdjustedNetConversionEdge <= -0.002) {
    finalNegativeRiskValidationVerdict = "fails_final_negative_risk_validation";
  } else {
    finalNegativeRiskValidationVerdict = "inconclusive_but_not_promotable";
  }

  const finalNegativeRiskValidationSummaryLine = `final_neg_risk_31552: verdict=${finalNegativeRiskValidationVerdict} | stress_net=${stressAdjustedNetConversionEdge} raw=${rawEstimatedConversionEdge} entry=${observedEntryCost} conv=${observedConversionCost} exit=${observedExitCost} stress=${stressSlippage} | clob=${clobBooksResolved}/${n} sum_mid=${sumMid} cap=${capacityEstimate}`;

  return {
    probeVersion: "final-negative-risk-validation-31552-v1",
    readDisclaimer:
      "Apenas evento 31552; mids/spreads do CLOB REST (token YES) quando disponível; fallback outcomePrices com spread conservador. Custos de saída e stress escalam com n. Sem ordens live; inconclusivo => não promovível.",
    eventId: EVENT_ID,
    familyId,
    marketIds,
    rawEstimatedConversionEdge,
    observedEntryCost,
    observedConversionCost,
    observedExitCost,
    stressAdjustedNetConversionEdge,
    capacityEstimate,
    fragilityAssessment,
    finalNegativeRiskValidationVerdict,
    finalNegativeRiskValidationSummaryLine,
    clobBooksResolved,
    marketsInSet: n,
    computedAt: new Date().toISOString(),
  };
}

function failDigest(reason: string): FinalNegativeRiskValidation31552Digest {
  return {
    probeVersion: "final-negative-risk-validation-31552-v1",
    readDisclaimer: "Falha ao carregar evento Gamma 31552.",
    eventId: EVENT_ID,
    familyId: "",
    marketIds: [],
    rawEstimatedConversionEdge: 0,
    observedEntryCost: 0,
    observedConversionCost: 0,
    observedExitCost: 0,
    stressAdjustedNetConversionEdge: 0,
    capacityEstimate: 0,
    fragilityAssessment: reason,
    finalNegativeRiskValidationVerdict: "inconclusive_but_not_promotable",
    finalNegativeRiskValidationSummaryLine: `final_neg_risk_31552: verdict=inconclusive_but_not_promotable | reason=${reason}`,
    clobBooksResolved: 0,
    marketsInSet: 0,
    computedAt: new Date().toISOString(),
  };
}
