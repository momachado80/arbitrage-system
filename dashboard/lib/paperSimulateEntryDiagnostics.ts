/**
 * Diagnóstico causal de simulateEntry (filledCapital / fill ratio).
 * globalThis — mesmo padrão que paperOpenDiagnostics.
 */

const GLOBAL_KEY = "__paperSimulateEntryDiagnostics_v1";
const MAX_SAMPLES = 600;

export type SimulateEntryDiagnosticRow = {
  atIso: string;
  opportunityId: string;
  opportunityType: string;
  liquidity: number;
  spread: number;
  confidence: number;
  recommendedCapital: number;
  /** Pedido vindo do engine (options.requestedCapital), se houver */
  optionRequestedCapital: number | null;
  portfolioAvailableCapital: number;
  capByLiquidity: number;
  /** min(option|recommended, portfolio, capByLiquidity) */
  finalRequestedCapital: number;
  /** Qual tecto fixou finalRequested (o menor dos três positivos relevantes) */
  requestedCapDominant: "recommended_or_option" | "portfolio" | "liquidity_10pct";
  fillProbability: number;
  filledCapital: number;
  fillRatio: number | null;
  liqScore: number;
  spreadPenalty: number;
  sizePenalty: number;
  sizeRatio: number;
  /** Multiplicador mais baixo em prob = 0.7 * Π (corta mais o produto) */
  probDominantFactor: "liquidity_log_score" | "spread_penalty" | "size_penalty" | "confidence";
  zeroFillExplicitReason: string | null;
};

type Store = { samples: SimulateEntryDiagnosticRow[] };

function getStore(): Store {
  const g = globalThis as unknown as Record<string, Store>;
  if (!g[GLOBAL_KEY]) g[GLOBAL_KEY] = { samples: [] };
  return g[GLOBAL_KEY];
}

function dominantRequestedCap(
  rawDesired: number,
  portfolio: number,
  capLiq: number
): SimulateEntryDiagnosticRow["requestedCapDominant"] {
  type K = SimulateEntryDiagnosticRow["requestedCapDominant"];
  const a: { k: K; v: number } = { k: "recommended_or_option", v: rawDesired };
  const b: { k: K; v: number } = { k: "portfolio", v: portfolio };
  const c: { k: K; v: number } = { k: "liquidity_10pct", v: capLiq };
  let m = a;
  if (b.v < m.v) m = b;
  if (c.v < m.v) m = c;
  return m.k;
}

function dominantProbFactor(
  liqScore: number,
  spreadPenalty: number,
  sizePenalty: number,
  confidence: number
): SimulateEntryDiagnosticRow["probDominantFactor"] {
  const opts: Array<{ k: SimulateEntryDiagnosticRow["probDominantFactor"]; v: number }> = [
    { k: "liquidity_log_score", v: liqScore },
    { k: "spread_penalty", v: spreadPenalty },
    { k: "size_penalty", v: sizePenalty },
    { k: "confidence", v: confidence },
  ];
  let m = opts[0];
  for (const o of opts) if (o.v < m.v) m = o;
  return m.k;
}

export function recordSimulateEntryDiagnostic(input: {
  opportunity: { opportunityId: string; opportunityType: string; liquidity: number; spread: number; confidence: number };
  capacity: { recommendedCapital: number };
  portfolioAvailableCapital: number;
  optionRequestedCapital: number | null;
  finalRequestedCapital: number;
  fillProbability: number;
  filledCapital: number;
}): void {
  const opp = input.opportunity;
  const capByLiquidity = opp.liquidity * 0.1;
  const liq = Math.max(opp.liquidity, 1);
  const liqScore = Math.min(1, Math.log10(liq) / 5);
  const spreadPenalty = Math.max(0.3, 1 - opp.spread * 3);
  const req = input.finalRequestedCapital;
  const sizeRatio = req > 0 ? req / liq : 0;
  const sizePenalty = Math.max(0.2, 1 - sizeRatio * 2);
  const conf = opp.confidence;

  const rawDesired =
    input.optionRequestedCapital != null && input.optionRequestedCapital > 0
      ? input.optionRequestedCapital
      : input.capacity.recommendedCapital;

  const requestedCapDominant = dominantRequestedCap(
    rawDesired,
    input.portfolioAvailableCapital,
    capByLiquidity
  );

  const probDom = dominantProbFactor(liqScore, spreadPenalty, sizePenalty, conf);

  let zeroFillExplicitReason: string | null = null;
  if (req <= 0) {
    zeroFillExplicitReason = `requested_le_zero|cap_dominant=${requestedCapDominant}|rawDesired=${rawDesired.toFixed(4)}|portfolio=${input.portfolioAvailableCapital.toFixed(4)}|capLiq=${capByLiquidity.toFixed(4)}`;
  } else if (input.filledCapital <= 0) {
    zeroFillExplicitReason = `filled_zero_despite_positive_request|prob=${input.fillProbability.toFixed(6)}|prob_dominant=${probDom}`;
  }

  const row: SimulateEntryDiagnosticRow = {
    atIso: new Date().toISOString(),
    opportunityId: opp.opportunityId,
    opportunityType: opp.opportunityType,
    liquidity: opp.liquidity,
    spread: opp.spread,
    confidence: conf,
    recommendedCapital: input.capacity.recommendedCapital,
    optionRequestedCapital: input.optionRequestedCapital,
    portfolioAvailableCapital: input.portfolioAvailableCapital,
    capByLiquidity,
    finalRequestedCapital: req,
    requestedCapDominant,
    fillProbability: input.fillProbability,
    filledCapital: input.filledCapital,
    fillRatio: req > 0 ? input.filledCapital / req : null,
    liqScore,
    spreadPenalty,
    sizePenalty,
    sizeRatio,
    probDominantFactor: probDom,
    zeroFillExplicitReason: zeroFillExplicitReason,
  };

  const st = getStore();
  st.samples.push(row);
  if (st.samples.length > MAX_SAMPLES) st.samples.splice(0, st.samples.length - MAX_SAMPLES);
}

export type SimulateEntryDiagnosticsSnapshot = {
  sampleCount: number;
  filledCapitalEqZeroCount: number;
  fillRatioHistogram: { bin: string; count: number }[];
  filledCapitalHistogram: { bin: string; count: number }[];
  probDominantFactorCounts: Record<string, number>;
  requestedCapDominantCounts: Record<string, number>;
  zeroFillReasonCounts: Record<string, number>;
  recentSamples: SimulateEntryDiagnosticRow[];
  means: {
    fillRatio: number | null;
    filledCapital: number;
    finalRequestedCapital: number;
    fillProbability: number;
  };
};

function binFillRatio(r: number | null): string {
  if (r == null || !Number.isFinite(r)) return "n/a";
  if (r <= 0) return "0";
  if (r < 0.25) return "(0,0.25)";
  if (r < 0.5) return "[0.25,0.5)";
  if (r < 0.75) return "[0.5,0.75)";
  if (r < 0.9) return "[0.75,0.9)";
  return "[0.9,1]";
}

function binFilled(f: number): string {
  if (f <= 0) return "0";
  if (f < 10) return "(0,10)";
  if (f < 50) return "[10,50)";
  if (f < 100) return "[50,100)";
  if (f < 250) return "[100,250)";
  return "[250+";
}

function zeroReasonKey(reason: string | null): string {
  if (!reason) return "(filled)";
  if (reason.startsWith("requested_le_zero")) return "requested_le_zero";
  if (reason.startsWith("filled_zero_despite_positive_request")) return "filled_zero_despite_positive_request";
  return "other";
}

export function getPaperSimulateEntryDiagnostics(): SimulateEntryDiagnosticsSnapshot {
  const samples = getStore().samples;
  const n = samples.length;
  const frHist: Record<string, number> = {};
  const fcHist: Record<string, number> = {};
  const probDom: Record<string, number> = {};
  const reqDom: Record<string, number> = {};
  const zeroReason: Record<string, number> = {};
  let z = 0;
  let sumFr = 0;
  let nFr = 0;
  let sumFilled = 0;
  let sumReq = 0;
  let sumProb = 0;

  for (const s of samples) {
    const bfr = binFillRatio(s.fillRatio);
    frHist[bfr] = (frHist[bfr] || 0) + 1;
    const bfc = binFilled(s.filledCapital);
    fcHist[bfc] = (fcHist[bfc] || 0) + 1;
    probDom[s.probDominantFactor] = (probDom[s.probDominantFactor] || 0) + 1;
    reqDom[s.requestedCapDominant] = (reqDom[s.requestedCapDominant] || 0) + 1;
    const zk = zeroReasonKey(s.zeroFillExplicitReason);
    zeroReason[zk] = (zeroReason[zk] || 0) + 1;
    if (s.filledCapital <= 0) z++;
    if (s.fillRatio != null && Number.isFinite(s.fillRatio)) {
      sumFr += s.fillRatio;
      nFr++;
    }
    sumFilled += s.filledCapital;
    sumReq += s.finalRequestedCapital;
    sumProb += s.fillProbability;
  }

  const toHist = (h: Record<string, number>) =>
    Object.entries(h)
      .map(([bin, count]) => ({ bin, count }))
      .sort((a, b) => b.count - a.count);

  return {
    sampleCount: n,
    filledCapitalEqZeroCount: z,
    fillRatioHistogram: toHist(frHist),
    filledCapitalHistogram: toHist(fcHist),
    probDominantFactorCounts: probDom,
    requestedCapDominantCounts: reqDom,
    zeroFillReasonCounts: zeroReason,
    recentSamples: samples.slice(-40),
    means: {
      fillRatio: nFr > 0 ? sumFr / nFr : null,
      filledCapital: n > 0 ? sumFilled / n : 0,
      finalRequestedCapital: n > 0 ? sumReq / n : 0,
      fillProbability: n > 0 ? sumProb / n : 0,
    },
  };
}
