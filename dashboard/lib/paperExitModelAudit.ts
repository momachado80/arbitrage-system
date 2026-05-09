/**
 * Auditoria só de leitura: modelo de preço de saída no paper (simulateExit + construção de latestState).
 * Não altera política, fechos nem PnL.
 */

import {
  effectiveGraphProvenanceForClosedAnalytics,
  PAPER_GRAPH_PROVENANCE_KEYS,
} from "./graphOpportunityPaperImpact";
import { getPaperPortfolio } from "./paperPortfolioStore";
import { isClosedTradeWithFiniteRealizedPnl } from "./paperClosedTradesMetrics";
import {
  getClosedTradeGrossRealizedPnL,
  getClosedTradeNetRealizedPnL,
  safeFeeBufferPerLeg,
} from "./paperRealizedPnlSemantics";
import { resolvePaperPolicyFromEnv } from "./paperTradeEngine";
import {
  getPaperTradeLifecycleClosesBuffer,
  type PaperTradeLifecycleClose,
} from "./paperTradeLifecycleDiagnostics";
import type { PaperGraphDiagnosticProvenance, PaperTrade } from "./paperTypes";

const EQ1_EPS = 1e-6;
const PRICE_EQ_EPS = 1e-9;
const EXTREME_SAMPLE_CAP = 10;
const TOP_LABELS_CAP = 8;
const EDGE_ZERO_EPS = 1e-12;
const EDGE_ZERO_SAMPLE_CAP = 15;

export type ExitPriceBucketKey = "lt01" | "b01_03" | "b03_05" | "b05_08" | "b08_099" | "eq1";

/** Heurística pós-fecho; ver `note` — não substitui instrumentação no fecho. */
export type ExitPriceSourceHeuristic =
  | "implied_mark_edge_near_zero_exit_approx_1"
  | "exit_price_equals_entry_possible_no_latest_or_flat"
  | "implied_mark_from_latest_other";

/** Origem do `latestState.edge` ao fecho (persistido) ou `other` (legado / indeterminado). */
export type EdgeExitSourceKey = "opp_map" | "mtm" | "fallback_no_latest" | "other";

export type EdgeZeroAtExitSample = {
  tradeId: string;
  provenance: string;
  entryPriceEstimate: number | null;
  exitPriceEstimate: number | null;
  latestOpportunityEdgeAtExit: number | null;
  exitCondition: string;
  exitPriceMarkSourceAtClose: EdgeExitSourceKey | null;
  label: string;
  ratioRealizedPnlToFilledCapital: number | null;
  edgeInstrumentation: "persisted" | "inferred_legacy_exit_approx_one";
};

export type CycleEdgeExitAudit = {
  tradeCount: number;
  edgeZeroCount: number;
  edgeZeroShare: number | null;
  edgeZeroBySource: Record<EdgeExitSourceKey, number>;
  topLabelsByEdgeZeroExtremeRatio: Array<{
    label: string;
    edgeZeroTradeCount: number;
    maxRatio: number;
  }>;
};

export type TopExtremeExitPriceTradeSample = {
  tradeId: string;
  provenance: string;
  opportunityType: string;
  entryPriceEstimate: number | null;
  exitPriceEstimate: number | null;
  /** Convénio motor: `exitPriceEstimate = 1 - edge` quando há latestState; inferido como `1 - exitPrice`. */
  latestOpportunityEdgeAtExit: number | null;
  exitCondition: string;
  /** Igual a `netRealizedPnL` (KPI líquido no store). */
  realizedPnL: number;
  netRealizedPnL: number;
  grossRealizedPnL: number;
  filledCapital: number;
  ratioRealizedPnlToFilledCapital: number | null;
  exitPriceSource: ExitPriceSourceHeuristic;
  lifecycleMarkSourceIfKnown: "opp_map" | "mtm" | "none" | null;
  lifecycleExitEqualsEntryBecauseNoLatest: boolean | null;
  warningFlags: string[];
};

export type CycleExitModelAudit = {
  tradeCount: number;
  eq1Count: number;
  eq1Share: number | null;
  avgEntryPriceEstimate: number | null;
  avgExitPriceEstimate: number | null;
  medianRatioRealizedPnlToFilledCapital: number | null;
  topLabelsByExtremeRatio: Array<{ label: string; tradeCount: number; maxRatio: number }>;
};

export type PaperExitModelAudit = {
  computedAt: string;
  note: string;
  simulateExitExitPriceRule: string;
  latestStateConstructionRule: string;
  whyExitPriceEqualsOne: string;
  closedTradesAnalyzedCount: number;
  closedTradesCountByExitPriceBucket: Record<ExitPriceBucketKey, number>;
  closedTradesCountByExitPriceSource: Record<ExitPriceSourceHeuristic, number>;
  countExitPriceExactlyOneByProvenance: Record<PaperGraphDiagnosticProvenance | "non_graph", number>;
  countExitPriceExactlyOneByExitReason: Record<string, number>;
  avgExitPriceByProvenance: Record<PaperGraphDiagnosticProvenance | "non_graph", number | null>;
  medianExitPriceByProvenance: Record<PaperGraphDiagnosticProvenance | "non_graph", number | null>;
  topExtremeExitPriceTrades: TopExtremeExitPriceTradeSample[];
  cycleExitModelAudit: CycleExitModelAudit;
  edgeAtExitInstrumentationNote: string;
  edgeClampAndNormalizationInCodebase: string;
  countEdgeZeroAtExitByProvenance: Record<PaperGraphDiagnosticProvenance | "non_graph", number>;
  countEdgeZeroAtExitByExitReason: Record<string, number>;
  countEdgeZeroAtExitBySource: Record<EdgeExitSourceKey, number>;
  avgEdgeAtExitByProvenance: Record<PaperGraphDiagnosticProvenance | "non_graph", number | null>;
  medianEdgeAtExitByProvenance: Record<PaperGraphDiagnosticProvenance | "non_graph", number | null>;
  samplesEdgeZeroAtExit: EdgeZeroAtExitSample[];
  cycleEdgeExitAudit: CycleEdgeExitAudit;
  /** Resumo da política actual de `resolveMarkPxFromTrade` para comparar auditorias antes/depois. */
  markToMarketGraphPolicy: string;
};

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function emptyProvenanceCounts(): Record<PaperGraphDiagnosticProvenance | "non_graph", number> {
  const o = {} as Record<PaperGraphDiagnosticProvenance | "non_graph", number>;
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) o[k] = 0;
  o.non_graph = 0;
  return o;
}

function emptyProvenanceNullable(): Record<PaperGraphDiagnosticProvenance | "non_graph", number | null> {
  const o = {} as Record<PaperGraphDiagnosticProvenance | "non_graph", number | null>;
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) o[k] = null;
  o.non_graph = null;
  return o;
}

function bucketExitPrice(x: number): ExitPriceBucketKey {
  if (!Number.isFinite(x)) return "lt01";
  if (x >= 1 - EQ1_EPS) return "eq1";
  if (x < 0.1) return "lt01";
  if (x < 0.3) return "b01_03";
  if (x < 0.5) return "b03_05";
  if (x < 0.8) return "b05_08";
  if (x < 1 - EQ1_EPS) return "b08_099";
  return "eq1";
}

function exitPriceSourceHeuristic(entry: number, exit: number): ExitPriceSourceHeuristic {
  if (Number.isFinite(exit) && exit >= 1 - EQ1_EPS) {
    return "implied_mark_edge_near_zero_exit_approx_1";
  }
  if (
    Number.isFinite(entry) &&
    Number.isFinite(exit) &&
    Math.abs(exit - entry) < PRICE_EQ_EPS
  ) {
    return "exit_price_equals_entry_possible_no_latest_or_flat";
  }
  return "implied_mark_from_latest_other";
}

function inferredEdgeFromExitPrice(exit: number | null): number | null {
  if (exit == null || typeof exit !== "number" || !Number.isFinite(exit)) return null;
  const e = 1 - exit;
  return Number.isFinite(e) ? round4(e) : null;
}

function tradeLabel(t: PaperTrade): string {
  const q = t.marketsInvolved?.[0]?.question;
  if (q && String(q).length > 0) return String(q).slice(0, 120);
  const oid = t.opportunityId;
  if (oid != null && String(oid).length > 0) return String(oid).slice(0, 80);
  const tid = t.tradeId;
  return tid != null && String(tid).length > 0 ? String(tid).slice(0, 80) : "(no_label)";
}

function provenanceKey(t: PaperTrade): PaperGraphDiagnosticProvenance | "non_graph" {
  if (t.sourceType !== "graph") return "non_graph";
  return effectiveGraphProvenanceForClosedAnalytics(t) ?? "unknown";
}

function medianSorted(arr: number[]): number | null {
  if (arr.length === 0) return null;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

function emptyEdgeBySource(): Record<EdgeExitSourceKey, number> {
  return { opp_map: 0, mtm: 0, fallback_no_latest: 0, other: 0 };
}

function isEdgeAtExitZero(edge: number | null | undefined): boolean {
  return typeof edge === "number" && Number.isFinite(edge) && Math.abs(edge) <= EDGE_ZERO_EPS;
}

function classifyEdgeExitSource(t: PaperTrade): EdgeExitSourceKey {
  const s = t.exitPriceMarkSourceAtClose;
  if (s === "opp_map" || s === "mtm" || s === "fallback_no_latest") return s;
  return "other";
}

/** Trade com `edge` efectivo ~0 ao fecho (exit≈1): persistido ou legado inferido por exit≈1. */
function tradeIsEdgeZeroAtExit(t: PaperTrade, exit: number): boolean {
  if (t.edgeAtExit !== undefined) {
    if (t.edgeAtExit === null) return false;
    if (typeof t.edgeAtExit === "number") return isEdgeAtExitZero(t.edgeAtExit);
  }
  return exit >= 1 - EQ1_EPS;
}

/** Edge numérico para médias: só quando faz sentido (não usa fallback exit=entry sem latest). */
function effectiveNumericEdgeForAvg(t: PaperTrade, exit: number, entry: number): number | null {
  if (t.edgeAtExit !== undefined) {
    if (t.edgeAtExit === null) return null;
    if (typeof t.edgeAtExit === "number" && Number.isFinite(t.edgeAtExit)) return t.edgeAtExit;
  }
  if (!Number.isFinite(exit)) return null;
  if (Number.isFinite(entry) && Math.abs(exit - entry) < PRICE_EQ_EPS) return null;
  return 1 - exit;
}

function edgeZeroSourceForTrade(t: PaperTrade, exit: number): EdgeExitSourceKey {
  if (t.edgeAtExit !== undefined && typeof t.edgeAtExit === "number") {
    return classifyEdgeExitSource(t);
  }
  if (exit >= 1 - EQ1_EPS) return "other";
  return "other";
}

function buildEdgeZeroSample(t: PaperTrade, feeBuf: number): EdgeZeroAtExitSample {
  const entry = typeof t.entryPriceEstimate === "number" && Number.isFinite(t.entryPriceEstimate) ? t.entryPriceEstimate : null;
  const exit =
    typeof t.exitPriceEstimate === "number" && Number.isFinite(t.exitPriceEstimate)
      ? t.exitPriceEstimate
      : null;
  const fc = typeof t.filledCapital === "number" && Number.isFinite(t.filledCapital) ? t.filledCapital : 0;
  const net = getClosedTradeNetRealizedPnL(t, feeBuf);
  const ratio = fc > 0 ? round4(net / fc) : null;
  const persisted = t.edgeAtExit !== undefined;
  const edgeVal =
    typeof t.edgeAtExit === "number" && Number.isFinite(t.edgeAtExit)
      ? t.edgeAtExit
      : exit != null
        ? inferredEdgeFromExitPrice(exit)
        : null;
  return {
    tradeId: t.tradeId,
    provenance: provenanceKey(t),
    entryPriceEstimate: entry != null ? round4(entry) : null,
    exitPriceEstimate: exit != null ? round4(exit) : null,
    latestOpportunityEdgeAtExit: edgeVal,
    exitCondition: String(t.exitCondition ?? "unknown"),
    exitPriceMarkSourceAtClose: persisted ? classifyEdgeExitSource(t) : null,
    label: tradeLabel(t),
    ratioRealizedPnlToFilledCapital: ratio,
    edgeInstrumentation: persisted ? "persisted" : "inferred_legacy_exit_approx_one",
  };
}

function buildExtremeSample(
  t: PaperTrade,
  feeBuf: number,
  lifecycleById: Map<string, PaperTradeLifecycleClose>
): TopExtremeExitPriceTradeSample {
  const entry = typeof t.entryPriceEstimate === "number" && Number.isFinite(t.entryPriceEstimate) ? t.entryPriceEstimate : null;
  const exit =
    typeof t.exitPriceEstimate === "number" && Number.isFinite(t.exitPriceEstimate)
      ? t.exitPriceEstimate
      : null;
  const fc = typeof t.filledCapital === "number" && Number.isFinite(t.filledCapital) ? t.filledCapital : 0;
  const net = getClosedTradeNetRealizedPnL(t, feeBuf);
  const gross = getClosedTradeGrossRealizedPnL(t);
  const ratio = fc > 0 ? round4(net / fc) : null;
  const src = entry != null && exit != null ? exitPriceSourceHeuristic(entry, exit) : "implied_mark_from_latest_other";
  const lc = lifecycleById.get(t.tradeId);
  const flags: string[] = [];
  if (exit != null && exit >= 1 - EQ1_EPS) flags.push("exit_implied_probability_one");
  if (ratio != null && Math.abs(ratio) > 1) flags.push("abs_net_ratio_gt_one");
  if (entry != null && entry > 0 && exit != null && exit > 1 + 1e-6) flags.push("exit_price_gt_one_unusual");
  return {
    tradeId: t.tradeId,
    provenance: provenanceKey(t),
    opportunityType: String(t.opportunityType ?? "unknown"),
    entryPriceEstimate: entry != null ? round4(entry) : null,
    exitPriceEstimate: exit != null ? round4(exit) : null,
    latestOpportunityEdgeAtExit: inferredEdgeFromExitPrice(exit),
    exitCondition: String(t.exitCondition ?? "unknown"),
    realizedPnL: round4(net),
    netRealizedPnL: round4(net),
    grossRealizedPnL: round4(gross),
    filledCapital: round4(fc),
    ratioRealizedPnlToFilledCapital: ratio,
    exitPriceSource: src,
    lifecycleMarkSourceIfKnown: lc?.markSource ?? null,
    lifecycleExitEqualsEntryBecauseNoLatest: lc != null ? lc.exitEqualsEntryBecauseNoLatest : null,
    warningFlags: flags,
  };
}

export function buildPaperExitModelAudit(): PaperExitModelAudit {
  const computedAt = new Date().toISOString();
  const feeBuf = safeFeeBufferPerLeg(resolvePaperPolicyFromEnv().feeBuffer);
  const closed = getPaperPortfolio().closedTrades.filter(isClosedTradeWithFiniteRealizedPnl);

  const lifecycleById = new Map(
    getPaperTradeLifecycleClosesBuffer().map((c) => [c.tradeId, c] as const)
  );

  const buckets: Record<ExitPriceBucketKey, number> = {
    lt01: 0,
    b01_03: 0,
    b03_05: 0,
    b05_08: 0,
    b08_099: 0,
    eq1: 0,
  };
  const sourceCounts: Record<ExitPriceSourceHeuristic, number> = {
    implied_mark_edge_near_zero_exit_approx_1: 0,
    exit_price_equals_entry_possible_no_latest_or_flat: 0,
    implied_mark_from_latest_other: 0,
  };
  const eq1ByProv = emptyProvenanceCounts();
  const eq1ByExit = new Map<string, number>();
  const exitPricesByProv = new Map<PaperGraphDiagnosticProvenance | "non_graph", number[]>();

  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) exitPricesByProv.set(k, []);
  exitPricesByProv.set("non_graph", []);

  const analyzed: PaperTrade[] = [];

  for (const t of closed) {
    const exit = t.exitPriceEstimate;
    if (typeof exit !== "number" || !Number.isFinite(exit)) continue;
    analyzed.push(t);
    buckets[bucketExitPrice(exit)] += 1;

    const entry =
      typeof t.entryPriceEstimate === "number" && Number.isFinite(t.entryPriceEstimate)
        ? t.entryPriceEstimate
        : NaN;
    const src = Number.isFinite(entry) ? exitPriceSourceHeuristic(entry, exit) : "implied_mark_from_latest_other";
    sourceCounts[src] += 1;

    const pk = provenanceKey(t);
    if (exit >= 1 - EQ1_EPS) {
      eq1ByProv[pk] += 1;
      const ex = String(t.exitCondition ?? "unknown");
      eq1ByExit.set(ex, (eq1ByExit.get(ex) ?? 0) + 1);
    }
    const arr = exitPricesByProv.get(pk);
    if (arr) arr.push(exit);
  }

  const avgExit = emptyProvenanceNullable();
  const medExit = emptyProvenanceNullable();
  for (const k of [...PAPER_GRAPH_PROVENANCE_KEYS, "non_graph" as const]) {
    const arr = exitPricesByProv.get(k) ?? [];
    if (arr.length === 0) continue;
    const sum = arr.reduce((a, b) => a + b, 0);
    avgExit[k] = round4(sum / arr.length);
    const m = medianSorted(arr);
    medExit[k] = m != null ? round4(m) : null;
  }

  const edgeZeroByProv = emptyProvenanceCounts();
  const edgeZeroByExit = new Map<string, number>();
  const edgeZeroBySource = emptyEdgeBySource();
  const edgesByProv = new Map<PaperGraphDiagnosticProvenance | "non_graph", number[]>();
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) edgesByProv.set(k, []);
  edgesByProv.set("non_graph", []);

  for (const t of analyzed) {
    const exit = t.exitPriceEstimate!;
    const entry = t.entryPriceEstimate;
    if (!tradeIsEdgeZeroAtExit(t, exit)) continue;
    const pk = provenanceKey(t);
    edgeZeroByProv[pk] += 1;
    const ex = String(t.exitCondition ?? "unknown");
    edgeZeroByExit.set(ex, (edgeZeroByExit.get(ex) ?? 0) + 1);
    edgeZeroBySource[edgeZeroSourceForTrade(t, exit)] += 1;
  }

  for (const t of analyzed) {
    const exit = t.exitPriceEstimate!;
    const entry = t.entryPriceEstimate;
    const pk = provenanceKey(t);
    const e = effectiveNumericEdgeForAvg(t, exit, entry);
    if (e == null || !Number.isFinite(e)) continue;
    const arr = edgesByProv.get(pk);
    if (arr) arr.push(e);
  }

  const avgEdgeAtExit = emptyProvenanceNullable();
  const medEdgeAtExit = emptyProvenanceNullable();
  for (const k of [...PAPER_GRAPH_PROVENANCE_KEYS, "non_graph" as const]) {
    const arr = edgesByProv.get(k) ?? [];
    if (arr.length === 0) continue;
    avgEdgeAtExit[k] = round4(arr.reduce((a, b) => a + b, 0) / arr.length);
    const md = medianSorted(arr);
    medEdgeAtExit[k] = md != null ? round4(md) : null;
  }

  const samplesEdgeZeroAtExit = analyzed
    .filter((t) => typeof t.exitPriceEstimate === "number" && tradeIsEdgeZeroAtExit(t, t.exitPriceEstimate))
    .map((t) => {
      const fc = t.filledCapital;
      const net = getClosedTradeNetRealizedPnL(t, feeBuf);
      const r = typeof fc === "number" && fc > 0 && Number.isFinite(net) ? net / fc : 0;
      return { t, r };
    })
    .sort((a, b) => Math.abs(b.r) - Math.abs(a.r))
    .slice(0, EDGE_ZERO_SAMPLE_CAP)
    .map(({ t }) => buildEdgeZeroSample(t, feeBuf));

  const withRatio = analyzed
    .map((t) => {
      const fc = t.filledCapital;
      const net = getClosedTradeNetRealizedPnL(t, feeBuf);
      const r = typeof fc === "number" && fc > 0 && Number.isFinite(net) ? net / fc : 0;
      return { t, r };
    })
    .sort((a, b) => Math.abs(b.r) - Math.abs(a.r));

  const topExtreme = withRatio.slice(0, EXTREME_SAMPLE_CAP).map(({ t }) => buildExtremeSample(t, feeBuf, lifecycleById));

  const cycleTrades = analyzed.filter(
    (t) => provenanceKey(t) === "cycle" || t.opportunityType === "graph_cycle"
  );
  const cycleEq1 = cycleTrades.filter((t) => typeof t.exitPriceEstimate === "number" && t.exitPriceEstimate >= 1 - EQ1_EPS);
  const cycleRatios = cycleTrades
    .map((t) => {
      const fc = t.filledCapital;
      const net = getClosedTradeNetRealizedPnL(t, feeBuf);
      return typeof fc === "number" && fc > 0 ? net / fc : NaN;
    })
    .filter((x) => Number.isFinite(x));

  const labelAgg = new Map<string, { count: number; maxRatio: number }>();
  for (const t of cycleTrades) {
    const lab = tradeLabel(t);
    const fc = t.filledCapital;
    const net = getClosedTradeNetRealizedPnL(t, feeBuf);
    const r = typeof fc === "number" && fc > 0 ? net / fc : 0;
    const cur = labelAgg.get(lab) ?? { count: 0, maxRatio: 0 };
    cur.count += 1;
    cur.maxRatio = Math.max(cur.maxRatio, Math.abs(r));
    labelAgg.set(lab, cur);
  }
  const topLabelsByExtremeRatio = Array.from(labelAgg.entries())
    .map(([label, v]) => ({ label, tradeCount: v.count, maxRatio: round4(v.maxRatio) }))
    .sort((a, b) => b.maxRatio - a.maxRatio)
    .slice(0, TOP_LABELS_CAP);

  const cycleEntrySum = cycleTrades.reduce((s, t) => s + (Number.isFinite(t.entryPriceEstimate) ? t.entryPriceEstimate! : 0), 0);
  const cycleExitSum = cycleTrades.reduce((s, t) => s + (Number.isFinite(t.exitPriceEstimate) ? t.exitPriceEstimate! : 0), 0);
  const cycleN = cycleTrades.length;

  const cycleExitModelAudit: CycleExitModelAudit = {
    tradeCount: cycleN,
    eq1Count: cycleEq1.length,
    eq1Share: cycleN > 0 ? round4(cycleEq1.length / cycleN) : null,
    avgEntryPriceEstimate: cycleN > 0 ? round4(cycleEntrySum / cycleN) : null,
    avgExitPriceEstimate: cycleN > 0 ? round4(cycleExitSum / cycleN) : null,
    medianRatioRealizedPnlToFilledCapital:
      cycleRatios.length > 0 ? round4(medianSorted(cycleRatios)!) : null,
    topLabelsByExtremeRatio,
  };

  const cycleEdgeZeroSafe = cycleTrades.filter(
    (t) => typeof t.exitPriceEstimate === "number" && tradeIsEdgeZeroAtExit(t, t.exitPriceEstimate)
  );
  const cycEdgeZeroBySource = emptyEdgeBySource();
  for (const t of cycleEdgeZeroSafe) {
    const ex = t.exitPriceEstimate!;
    cycEdgeZeroBySource[edgeZeroSourceForTrade(t, ex)] += 1;
  }
  const labelEdgeZero = new Map<string, { count: number; maxRatio: number }>();
  for (const t of cycleEdgeZeroSafe) {
    const lab = tradeLabel(t);
    const fc = t.filledCapital;
    const net = getClosedTradeNetRealizedPnL(t, feeBuf);
    const r = typeof fc === "number" && fc > 0 ? Math.abs(net / fc) : 0;
    const cur = labelEdgeZero.get(lab) ?? { count: 0, maxRatio: 0 };
    cur.count += 1;
    cur.maxRatio = Math.max(cur.maxRatio, r);
    labelEdgeZero.set(lab, cur);
  }
  const topLabelsByEdgeZeroExtremeRatio = Array.from(labelEdgeZero.entries())
    .map(([label, v]) => ({
      label,
      edgeZeroTradeCount: v.count,
      maxRatio: round4(v.maxRatio),
    }))
    .sort((a, b) => b.maxRatio - a.maxRatio)
    .slice(0, TOP_LABELS_CAP);

  const cycleEdgeExitAudit: CycleEdgeExitAudit = {
    tradeCount: cycleN,
    edgeZeroCount: cycleEdgeZeroSafe.length,
    edgeZeroShare: cycleN > 0 ? round4(cycleEdgeZeroSafe.length / cycleN) : null,
    edgeZeroBySource: cycEdgeZeroBySource,
    topLabelsByEdgeZeroExtremeRatio,
  };

  const countExitPriceExactlyOneByExitReason: Record<string, number> = {};
  for (const [k, v] of Array.from(eq1ByExit.entries())) countExitPriceExactlyOneByExitReason[k] = v;

  const countEdgeZeroAtExitByExitReason: Record<string, number> = {};
  for (const [k, v] of Array.from(edgeZeroByExit.entries())) countEdgeZeroAtExitByExitReason[k] = v;

  return {
    computedAt,
    note:
      "Agregados sobre fechados com PnL finito e exitPriceEstimate numérico. " +
      "Fechos novos persistem `edgeAtExit` e `exitPriceMarkSourceAtClose` no store (sem alterar fórmulas). " +
      "Legados sem esses campos: `edge≈0` inferido só quando `exitPrice≈1` (classificado como fonte `other`). " +
      "Heurística `exit_price_equals_entry_possible_no_latest_or_flat` permanece ambígua para preços.",
    simulateExitExitPriceRule:
      "executionSimulator.simulateExit: se `latestOpportunity` (latestState) não é null, `exitPrice = 1 - latestOpportunity.edge`; senão `exitPrice = entryPriceEstimate` (fallback).",
    latestStateConstructionRule:
      "paperTradeEngine processOpportunities (fecho): `latestState = { edge: opp.edge }` se a oportunidade está no oppMap deste ciclo; senão, se existe `markPxMtm` de `resolveMarkPxFromTrade`, `latestState = { edge: 1 - markPxMtm }`; caso contrário `latestState = null`. " +
      "Para `graph_complement`, `markPxMtm` alinha com `probabilityGraph` complementary: `markPx = clamp01(1 - |pA+pB-1|)` (duas pernas, `prices[0]`).",
    whyExitPriceEqualsOne:
      "Com `latestState` presente, `exitPrice = 1 - edge`. Logo `exitPrice === 1` (módulo float) quando `edge === 0` no estado latest/MTM — ou seja, o modelo interpreta probabilidade implícita 1 (100%) para o lado marcado, não um preço de mercado “intermediário”. Isto pode reflectir resolução/certeza no feed de edges ou um artefacto se `edge` vier clamped a 0.",
    closedTradesAnalyzedCount: analyzed.length,
    closedTradesCountByExitPriceBucket: buckets,
    closedTradesCountByExitPriceSource: sourceCounts,
    countExitPriceExactlyOneByProvenance: eq1ByProv,
    countExitPriceExactlyOneByExitReason,
    avgExitPriceByProvenance: avgExit,
    medianExitPriceByProvenance: medExit,
    topExtremeExitPriceTrades: topExtreme,
    cycleExitModelAudit,
    edgeAtExitInstrumentationNote:
      "Campos opcionais em `PaperTrade`: `edgeAtExit` = `latestState.edge` ao fecho se `latestState !== null`, senão `null`; " +
      "`exitPriceMarkSourceAtClose` = `opp_map` | `mtm` | `fallback_no_latest` conforme ramo em `paperTradeEngine.processOpportunities`.",
    edgeClampAndNormalizationInCodebase:
      "No caminho de fecho paper (`paperTradeEngine` → `simulateExit`), **não** há `Math.max(0, edge)` sobre `latest.opp.edge`: o valor passa directamente a `latestState.edge`. " +
      "`resolveMarkPxFromTrade`: `cross_market` e `overround`/`underround` usam fórmulas explícitas; `graph_complement` usa **duas pernas** `markPx = clamp01(1 - |pA+pB-1|)` (coerente com `entryPriceEstimate = 1 - |pA+pB-1|` da violação complementary). " +
      "Outros `graph_*` **não** usam o proxy single-market `2-(p0+p1)` (num binário, `p0+p1≈1` ⇒ mark≈1 e `edgeAtExit≈0` por **tautologia**, não por resolução real). " +
      "Noutros subsistemas (`capitalCapacityEngine`, `edgeDecayModel`, `marketImpactModel`) o `Math.max(0, edge)` aplica-se a ranking/capacidade, não ao fecho descrito.",
    markToMarketGraphPolicy:
      "graph_complement: markPx=clamp01(1-|pA+pB-1|) com dois marketIds; graph_equivalence_micro|graph_subset_micro|graph_exclusive_micro: markPx=clamp01(1-|pA-pB|) (2 pernas binárias, violação equivalence + filtros micro + split estrutural); graph_subset|graph_exclusive|graph_equivalence|graph_cycle: sem MTM (null); overround/underround/cross_market: inalterados.",
    countEdgeZeroAtExitByProvenance: edgeZeroByProv,
    countEdgeZeroAtExitByExitReason,
    countEdgeZeroAtExitBySource: edgeZeroBySource,
    avgEdgeAtExitByProvenance: avgEdgeAtExit,
    medianEdgeAtExitByProvenance: medEdgeAtExit,
    samplesEdgeZeroAtExit,
    cycleEdgeExitAudit,
  };
}
