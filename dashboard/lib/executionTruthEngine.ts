/**
 * Autonomous Execution Viability — camada machine-native (sem trading live,
 * sem interpretação humana de eventos, sem caça semântica a anomalias).
 * Sinais: spread proxy, profundidade proxy, fila/fill, adverse selection,
 * inventário, unwind. Tudo derivado de campos observáveis do mercado normalizado.
 */

import { getAllMarkets } from "./marketDataService";
import type { NormalizedMarket } from "./polymarketClient";

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

export type ExecutionTruthVerdict =
  | "no_quoteable_robotic_edge_found"
  | "weak_quoteable_markets_only"
  | "viable_quoteable_market_present"
  | "multiple_viable_quoteable_markets_present";

export type RobotQuoteableVerdict = "not_quoteable" | "marginal_edge" | "positive_expected_cycle_net";

export interface ExecutionTruthMarketRow {
  marketId: string;
  marketTitle: string;
  observedBidAskSpread: number;
  observedDepth: number;
  fillPlausibility: number;
  adverseSelectionProxy: number;
  inventoryRiskProxy: number;
  unwindCostProxy: number;
  estimatedNetPerCycle: number;
  estimatedNetPerHour: number;
  quoteableVerdict: RobotQuoteableVerdict;
}

export interface StrongestQuoteableMarket {
  marketId: string;
  marketTitle: string;
  estimatedNetPerCycle: number;
  estimatedNetPerHour: number;
  quoteableVerdict: RobotQuoteableVerdict;
}

export type TopNetGapAuditVerdict =
  | "adverse_selection_dominant_drag"
  | "inventory_and_unwind_dominant_drag"
  | "fill_and_spread_capture_dominant_drag"
  | "mixed_balanced_drag"
  | "near_zero_reachable_by_single_proxy_lever";

export interface CounterfactualNets {
  ifSpreadImproved: number;
  ifAdverseSelectionImproved: number;
  ifInventoryDragImproved: number;
  ifFillRateImproved: number;
}

export interface TopNetGapMarketAudit {
  marketId: string;
  marketTitle: string;
  currentEstimatedNetPerCycle: number;
  gapToZero: number;
  spreadContribution: number;
  adverseSelectionContribution: number;
  inventoryDragContribution: number;
  fillPlausibilityContribution: number;
  /** Desagregação explícita de unwind+taxa (já incluídos em inventoryDragContribution para soma ao net). */
  unwindFeeDrag: number;
  counterfactualNetIfEachComponentImprovedSeparately: CounterfactualNets;
  smallestSingleComponentImprovement: {
    component: "spread" | "adverse_selection" | "inventory_drag" | "fill_rate";
    /** Unidade nativa do proxy (spread absoluto, ou factor 0–1 nos outros). */
    minimumDelta: number;
    /** Net após aplicar só esse ajuste mínimo (aprox. linear). */
    impliedNetAfterMinimum: number;
  };
  supportingNote: string;
}

export interface MinImprovementToZeroEntry {
  marketId: string;
  component: TopNetGapMarketAudit["smallestSingleComponentImprovement"]["component"];
  minimumDelta: number;
}

export interface TopNetGapAuditMeta {
  auditVersion: "top-net-gap-v1";
  marketsAudited: number;
}

export interface ExecutionTruthDigest {
  probeVersion: "execution-truth-v1";
  readDisclaimer: string;
  executionTruthVerdict: ExecutionTruthVerdict;
  marketsObserved: number;
  marketsQuoteable: number;
  marketsWithPositiveEstimatedCycleNet: number;
  strongestQuoteableMarkets: StrongestQuoteableMarket[];
  executionTruthSummaryLine: string;
  candidateMarkets: ExecutionTruthMarketRow[];
  topNetGapAudit: TopNetGapAuditMeta;
  topNetGapMarkets: TopNetGapMarketAudit[];
  avgGapToZero: number;
  dominantNegativeComponentByTopMarket: Record<string, string>;
  gapIfSpreadImproved: number;
  gapIfAdverseSelectionImproved: number;
  gapIfInventoryDragImproved: number;
  gapIfFillRateImproved: number;
  minImprovementNeededToReachZeroByMarket: MinImprovementToZeroEntry[];
  topNetGapAuditVerdict: TopNetGapAuditVerdict;
  topNetGapSummaryLine: string;
  computedAt: string;
}

const VIABLE_CYCLE_NET = 0.0045;
const MARGINAL_CYCLE_NET = 0.0008;

function queueFillStress(m: NormalizedMarket): number {
  if (m.liquidity <= 0) return 1;
  return clamp01(m.volume / (m.liquidity * 5.5 + 400));
}

export function observedSpread(m: NormalizedMarket): number {
  return r6(Math.min(0.48, m.spread));
}

export function observedDepth(m: NormalizedMarket): number {
  return r6(Math.min(24_000, m.liquidity * 0.0048));
}

export function fillPlausibilityAtSpread(m: NormalizedMarket, sEff: number): number {
  const qc = queueFillStress(m);
  return r6(
    clamp01(0.14 + 0.52 * clamp01(m.liquidity / 26_000) + 0.28 * (1 - clamp01(sEff / 0.5)) - 0.22 * qc),
  );
}

export function fillPlausibility(m: NormalizedMarket): number {
  return fillPlausibilityAtSpread(m, observedSpread(m));
}

export function adverseSelectionAtSpread(m: NormalizedMarket, sEff: number): number {
  const qc = queueFillStress(m);
  return r6(qc * 0.016 + sEff * 0.11);
}

function adverseSelectionProxy(m: NormalizedMarket): number {
  return adverseSelectionAtSpread(m, observedSpread(m));
}

export function inventoryRiskAtSpread(m: NormalizedMarket, sEff: number): number {
  return r6((1 - clamp01(m.liquidity / 42_000)) * 0.008 + sEff * 0.042);
}

function inventoryRiskProxy(m: NormalizedMarket): number {
  return inventoryRiskAtSpread(m, observedSpread(m));
}

export function unwindCostAtSpread(m: NormalizedMarket, sEff: number): number {
  const thin = 1 - clamp01(m.liquidity / 35_000);
  return r6(0.0016 + thin * 0.005 + sEff * 0.022);
}

function unwindCostProxy(m: NormalizedMarket): number {
  return unwindCostAtSpread(m, observedSpread(m));
}

function cyclesPerHourProxy(m: NormalizedMarket): number {
  const base = m.liquidity > 0 ? (m.volume / (m.liquidity + 600)) * 3.2 : 0.4;
  return r6(Math.min(14, Math.max(0.35, base)));
}

export function feeProxy(): number {
  return 0.003;
}

export function estimatedNetFromState(
  s: number,
  fill: number,
  adv: number,
  inv: number,
  unw: number,
  fee: number,
): number {
  const gross = s * 0.13 * fill;
  return r6(gross - adv - inv - unw - fee);
}

export function estimatedNetPerCycle(m: NormalizedMarket): number {
  const s = observedSpread(m);
  const fill = fillPlausibility(m);
  const adv = adverseSelectionProxy(m);
  const inv = inventoryRiskProxy(m);
  const unw = unwindCostProxy(m);
  const fee = feeProxy();
  return estimatedNetFromState(s, fill, adv, inv, unw, fee);
}

function dominantCostLabel(m: NormalizedMarket): string {
  const adv = adverseSelectionProxy(m);
  const inv = inventoryRiskProxy(m);
  const uw = unwindCostProxy(m) + feeProxy();
  if (adv >= inv && adv >= uw) return "adverse_selection_proxy";
  if (inv >= adv && inv >= uw) return "inventory_risk_proxy";
  return "unwind_and_fee";
}

function gapToZeroFromNet(net: number): number {
  return r6(net < 0 ? -net : 0);
}

function netAtSpreadEff(m: NormalizedMarket, sEff: number): number {
  const fill = fillPlausibilityAtSpread(m, sEff);
  const adv = adverseSelectionAtSpread(m, sEff);
  const inv = inventoryRiskAtSpread(m, sEff);
  const unw = unwindCostAtSpread(m, sEff);
  return estimatedNetFromState(sEff, fill, adv, inv, unw, feeProxy());
}

/** Delta de spread (absoluto) até net>=0, com todos os proxies recalculados em s. */
function bisectSpreadDeltaForZero(m: NormalizedMarket): { delta: number; netAfter: number } {
  const s0 = observedSpread(m);
  if (netAtSpreadEff(m, s0) >= 0) return { delta: 0, netAfter: netAtSpreadEff(m, s0) };
  if (netAtSpreadEff(m, 0.48) < 0) {
    return { delta: r6(0.48 - s0), netAfter: netAtSpreadEff(m, 0.48) };
  }
  let lo = s0;
  let hi = 0.48;
  for (let i = 0; i < 48; i++) {
    const mid = (lo + hi) / 2;
    if (netAtSpreadEff(m, mid) >= 0) hi = mid;
    else lo = mid;
  }
  return { delta: r6(hi - s0), netAfter: r6(netAtSpreadEff(m, hi)) };
}

function buildTopNetGapMarketAudit(m: NormalizedMarket): TopNetGapMarketAudit {
  const s = observedSpread(m);
  const fill = fillPlausibility(m);
  const adv = adverseSelectionProxy(m);
  const inv = inventoryRiskProxy(m);
  const unw = unwindCostProxy(m);
  const fee = feeProxy();
  const gross = r6(s * 0.13 * fill);
  const net = estimatedNetFromState(s, fill, adv, inv, unw, fee);
  const gapToZero = gapToZeroFromNet(net);

  const spreadContribution = r6(s * 0.13 * 0.5);
  const fillPlausibilityContribution = r6(s * 0.13 * (fill - 0.5));
  const adverseSelectionContribution = r6(-adv);
  const inventoryDragContribution = r6(-(inv + unw + fee));

  const sWide = r6(Math.min(0.48, s * 1.06));
  const fillW = fillPlausibilityAtSpread(m, sWide);
  const advW = adverseSelectionAtSpread(m, sWide);
  const invW = inventoryRiskAtSpread(m, sWide);
  const unwW = unwindCostAtSpread(m, sWide);
  const netSpread = estimatedNetFromState(sWide, fillW, advW, invW, unwW, fee);

  const advLo = r6(adv * 0.78);
  const netAdv = estimatedNetFromState(s, fill, advLo, inv, unw, fee);

  const invLo = r6(inv * 0.78);
  const unwLo = r6(unw * 0.82);
  const netInv = estimatedNetFromState(s, fill, adv, invLo, unwLo, fee);

  const fillHi = r6(Math.min(0.99, fill + 0.1));
  const netFill = estimatedNetFromState(s, fillHi, adv, inv, unw, fee);

  const counterfactualNetIfEachComponentImprovedSeparately: CounterfactualNets = {
    ifSpreadImproved: netSpread,
    ifAdverseSelectionImproved: netAdv,
    ifInventoryDragImproved: netInv,
    ifFillRateImproved: netFill,
  };

  const K = r6(adv + inv + unw + fee);
  const denomFill = s * 0.13;
  const fillNeeded = denomFill > 1e-9 ? r6(Math.min(1, K / denomFill)) : 1;
  const deltaFill = r6(Math.max(0, fillNeeded - fill));
  const netAfterFill = estimatedNetFromState(s, fillNeeded, adv, inv, unw, fee);

  const { delta: deltaS, netAfter: netAfterS } = bisectSpreadDeltaForZero(m);

  const advNeeded = r6(Math.max(0, gross - inv - unw - fee));
  const deltaAdv = r6(Math.max(0, adv - advNeeded));
  const netAfterAdv = estimatedNetFromState(s, fill, advNeeded, inv, unw, fee);

  const invOnlyNeeded = r6(Math.max(0, gross - adv - unw - fee));
  const deltaInvOnly = r6(Math.max(0, inv - invOnlyNeeded));
  const netAfterInv = estimatedNetFromState(s, fill, adv, invOnlyNeeded, unw, fee);

  type Lev = TopNetGapMarketAudit["smallestSingleComponentImprovement"]["component"];
  const candidates: { lev: Lev; delta: number; netAfter: number }[] = [
    { lev: "spread", delta: deltaS, netAfter: netAfterS },
    { lev: "adverse_selection", delta: deltaAdv, netAfter: netAfterAdv },
    { lev: "inventory_drag", delta: deltaInvOnly, netAfter: netAfterInv },
    { lev: "fill_rate", delta: deltaFill, netAfter: netAfterFill },
  ];

  const norm = (c: (typeof candidates)[0]) => {
    const w =
      c.lev === "spread"
        ? 0.48
        : c.lev === "adverse_selection"
          ? Math.max(1e-6, adv)
          : c.lev === "inventory_drag"
            ? Math.max(1e-6, inv)
            : Math.max(1e-6, 1 - fill);
    return c.delta / w;
  };

  const feasible = candidates.filter(c => c.delta < 1e5);
  const best = feasible.reduce((a, b) => (norm(a) <= norm(b) ? a : b));

  const supportingNote =
    "gap_audit: net=gross-adv-inv-unwind-fee | spread+fill=gross with fill neutral@0.5 | inventoryDragContribution bundles inv+unwind+fee | counterfactuals isolated ±6% spread / -22% adverse / -18% inv,-18% unwind / +0.10 fill cap";

  return {
    marketId: m.id,
    marketTitle: m.question.length > 130 ? `${m.question.slice(0, 127)}…` : m.question,
    currentEstimatedNetPerCycle: net,
    gapToZero,
    spreadContribution,
    adverseSelectionContribution,
    inventoryDragContribution,
    fillPlausibilityContribution,
    unwindFeeDrag: r6(-(unw + fee)),
    counterfactualNetIfEachComponentImprovedSeparately,
    smallestSingleComponentImprovement: {
      component: best.lev,
      minimumDelta: r6(best.delta),
      impliedNetAfterMinimum: r6(best.netAfter),
    },
    supportingNote,
  };
}

export function isMachineObserved(m: NormalizedMarket): boolean {
  return (
    m.active &&
    !m.closed &&
    m.outcomes.length === 2 &&
    m.prices.length === m.outcomes.length &&
    m.liquidity >= 350 &&
    m.spread > 0.001 &&
    m.spread <= 0.55
  );
}

export function isRobotQuoteableGate(m: NormalizedMarket): boolean {
  return (
    isMachineObserved(m) &&
    m.spread >= 0.012 &&
    m.spread <= 0.48 &&
    m.liquidity >= 1_800 &&
    m.volume >= 750
  );
}

function quoteableVerdictFor(m: NormalizedMarket, netCycle: number): RobotQuoteableVerdict {
  if (!isRobotQuoteableGate(m)) return "not_quoteable";
  if (netCycle >= VIABLE_CYCLE_NET) return "positive_expected_cycle_net";
  if (netCycle >= MARGINAL_CYCLE_NET) return "marginal_edge";
  return "not_quoteable";
}

function meanGapToZeroAfter(nets: number[]): number {
  if (nets.length === 0) return 0;
  return r6(nets.reduce((s, n) => s + gapToZeroFromNet(n), 0) / nets.length);
}

function resolveTopNetGapAuditVerdict(
  audits: TopNetGapMarketAudit[],
  topMarkets: NormalizedMarket[],
  leverNorms: number[],
): TopNetGapAuditVerdict {
  if (audits.length === 0 || topMarkets.length === 0) return "mixed_balanced_drag";

  const domCounts: Record<string, number> = {
    adverse_selection_proxy: 0,
    inventory_risk_proxy: 0,
    unwind_and_fee: 0,
  };
  for (const m of topMarkets) {
    const d = dominantCostLabel(m);
    domCounts[d] = (domCounts[d] ?? 0) + 1;
  }

  const captureWeak = topMarkets.filter(m => {
    const g = observedSpread(m) * 0.13 * fillPlausibility(m);
    const k =
      adverseSelectionProxy(m) + inventoryRiskProxy(m) + unwindCostProxy(m) + feeProxy();
    return k > 1e-9 && g / k < 0.55;
  }).length;

  const easyLevers = leverNorms.filter(n => n < 0.11).length;

  if (easyLevers >= 4) return "near_zero_reachable_by_single_proxy_lever";
  if (captureWeak >= 6) return "fill_and_spread_capture_dominant_drag";
  if (domCounts.adverse_selection_proxy >= 5) return "adverse_selection_dominant_drag";
  if (domCounts.inventory_risk_proxy + domCounts.unwind_and_fee >= 6) {
    return "inventory_and_unwind_dominant_drag";
  }
  if (domCounts.inventory_risk_proxy >= 4 || domCounts.unwind_and_fee >= 4) {
    return "inventory_and_unwind_dominant_drag";
  }
  return "mixed_balanced_drag";
}

function rowFor(m: NormalizedMarket): ExecutionTruthMarketRow {
  const netCycle = estimatedNetPerCycle(m);
  const cph = cyclesPerHourProxy(m);
  return {
    marketId: m.id,
    marketTitle: m.question.length > 130 ? `${m.question.slice(0, 127)}…` : m.question,
    observedBidAskSpread: observedSpread(m),
    observedDepth: observedDepth(m),
    fillPlausibility: fillPlausibility(m),
    adverseSelectionProxy: adverseSelectionProxy(m),
    inventoryRiskProxy: inventoryRiskProxy(m),
    unwindCostProxy: unwindCostProxy(m),
    estimatedNetPerCycle: netCycle,
    estimatedNetPerHour: r6(netCycle * cph),
    quoteableVerdict: quoteableVerdictFor(m, netCycle),
  };
}

export function buildExecutionTruthDigest(): ExecutionTruthDigest {
  const all = getAllMarkets();
  const observed = all.filter(isMachineObserved);
  const structurallyQuoteable = observed.filter(isRobotQuoteableGate);
  const rows = observed.map(rowFor).sort((a, b) => b.estimatedNetPerCycle - a.estimatedNetPerCycle);
  const quoteableRows = structurallyQuoteable.map(rowFor).sort((a, b) => b.estimatedNetPerCycle - a.estimatedNetPerCycle);

  const marketsObserved = observed.length;
  const marketsQuoteable = structurallyQuoteable.length;
  const marketsWithPositiveEstimatedCycleNet = quoteableRows.filter(r => r.estimatedNetPerCycle > 0).length;
  const viableByThreshold = quoteableRows.filter(r => r.estimatedNetPerCycle >= VIABLE_CYCLE_NET).length;

  const strongestQuoteableMarkets: StrongestQuoteableMarket[] = quoteableRows.slice(0, 16)
    .map(r => ({
      marketId: r.marketId,
      marketTitle: r.marketTitle,
      estimatedNetPerCycle: r.estimatedNetPerCycle,
      estimatedNetPerHour: r.estimatedNetPerHour,
      quoteableVerdict: r.quoteableVerdict,
    }));

  let executionTruthVerdict: ExecutionTruthVerdict;
  if (marketsObserved === 0 || marketsQuoteable === 0) {
    executionTruthVerdict = "no_quoteable_robotic_edge_found";
  } else if (viableByThreshold === 0) {
    executionTruthVerdict = "weak_quoteable_markets_only";
  } else if (viableByThreshold === 1) {
    executionTruthVerdict = "viable_quoteable_market_present";
  } else {
    executionTruthVerdict = "multiple_viable_quoteable_markets_present";
  }

  const executionTruthSummaryLine = `execution_truth: verdict=${executionTruthVerdict} | observed=${marketsObserved} quoteable=${marketsQuoteable} positive_cycle_net=${marketsWithPositiveEstimatedCycleNet} | top_net_cycle=${rows[0]?.estimatedNetPerCycle ?? "n/a"}`;

  const topForGap = [...structurallyQuoteable]
    .sort((a, b) => estimatedNetPerCycle(b) - estimatedNetPerCycle(a))
    .slice(0, 10);
  const topNetGapMarkets = topForGap.map(buildTopNetGapMarketAudit);
  const avgGapToZero =
    topNetGapMarkets.length > 0
      ? r6(topNetGapMarkets.reduce((s, x) => s + x.gapToZero, 0) / topNetGapMarkets.length)
      : 0;

  const dominantNegativeComponentByTopMarket: Record<string, string> = {};
  for (const m of topForGap) {
    dominantNegativeComponentByTopMarket[m.id] = dominantCostLabel(m);
  }

  const gapIfSpreadImproved = meanGapToZeroAfter(
    topNetGapMarkets.map(x => x.counterfactualNetIfEachComponentImprovedSeparately.ifSpreadImproved),
  );
  const gapIfAdverseSelectionImproved = meanGapToZeroAfter(
    topNetGapMarkets.map(x => x.counterfactualNetIfEachComponentImprovedSeparately.ifAdverseSelectionImproved),
  );
  const gapIfInventoryDragImproved = meanGapToZeroAfter(
    topNetGapMarkets.map(x => x.counterfactualNetIfEachComponentImprovedSeparately.ifInventoryDragImproved),
  );
  const gapIfFillRateImproved = meanGapToZeroAfter(
    topNetGapMarkets.map(x => x.counterfactualNetIfEachComponentImprovedSeparately.ifFillRateImproved),
  );

  const leverNorms = topForGap.map((m, i) => {
    const c = topNetGapMarkets[i].smallestSingleComponentImprovement;
    const s = observedSpread(m);
    const fill = fillPlausibility(m);
    const adv = adverseSelectionProxy(m);
    const inv = inventoryRiskProxy(m);
    const w =
      c.component === "spread"
        ? 0.48
        : c.component === "adverse_selection"
          ? Math.max(1e-6, adv)
          : c.component === "inventory_drag"
            ? Math.max(1e-6, inv)
            : Math.max(1e-6, 1 - fill);
    return c.minimumDelta / w;
  });

  const topNetGapAuditVerdict = resolveTopNetGapAuditVerdict(topNetGapMarkets, topForGap, leverNorms);

  const minImprovementNeededToReachZeroByMarket: MinImprovementToZeroEntry[] = topNetGapMarkets.map(x => ({
    marketId: x.marketId,
    component: x.smallestSingleComponentImprovement.component,
    minimumDelta: x.smallestSingleComponentImprovement.minimumDelta,
  }));

  const topNetGapSummaryLine = `top_net_gap: verdict=${topNetGapAuditVerdict} | top10_avg_gap_to_zero=${avgGapToZero} | gap_if_spread_cf=${gapIfSpreadImproved} gap_if_adverse_cf=${gapIfAdverseSelectionImproved} gap_if_inv_cf=${gapIfInventoryDragImproved} gap_if_fill_cf=${gapIfFillRateImproved}`;

  return {
    probeVersion: "execution-truth-v1",
    readDisclaimer:
      "Execution truth: métricas derivadas só de mercado normalizado (spread outcome, liquidez, volume). Não há book L2 nem fills reais; net/ciclo e /hora são proxies para viabilidade autónoma. Sem ordens, sem juízo humano sobre eventos. topNetGapAudit decompõe o défice face a zero nos top 10 quoteable; counterfactuals são isolados e ilustrativos.",
    executionTruthVerdict,
    marketsObserved,
    marketsQuoteable,
    marketsWithPositiveEstimatedCycleNet,
    strongestQuoteableMarkets,
    executionTruthSummaryLine,
    candidateMarkets: rows.slice(0, 48),
    topNetGapAudit: {
      auditVersion: "top-net-gap-v1",
      marketsAudited: topNetGapMarkets.length,
    },
    topNetGapMarkets,
    avgGapToZero,
    dominantNegativeComponentByTopMarket,
    gapIfSpreadImproved,
    gapIfAdverseSelectionImproved,
    gapIfInventoryDragImproved,
    gapIfFillRateImproved,
    minImprovementNeededToReachZeroByMarket,
    topNetGapAuditVerdict,
    topNetGapSummaryLine,
    computedAt: new Date().toISOString(),
  };
}
