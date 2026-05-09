/**
 * Métricas de sobrevivência de graph opportunities por `diagnosticRelationProvenance`.
 * Actualizado no ciclo paper (O(batch)); GET só lê snapshots congelados.
 */

import type { GraphOpportunity } from "./graphArbitrageEngine";
import type {
  NormalizedPaperOpportunity,
  PaperGraphDiagnosticProvenance,
  PaperOpportunityType,
  PaperTrade,
} from "./paperTypes";
import { getPaperPortfolio } from "./paperPortfolioStore";
import {
  isClosedTradeWithFiniteRealizedPnl,
  isGraphTradeForProvenanceMetrics,
} from "./paperClosedTradesMetrics";
import {
  DEFAULT_PAPER_FEE_BUFFER_PER_LEG,
  getClosedTradeEstimatedTotalFees,
  getClosedTradeGrossRealizedPnL,
  getClosedTradeNetRealizedPnL,
} from "./paperRealizedPnlSemantics";

export const PAPER_GRAPH_PROVENANCE_KEYS: readonly PaperGraphDiagnosticProvenance[] = [
  "equivalent",
  "subset",
  "exclusive",
  "complementary_strict",
  "complementary_relaxed",
  "cycle",
  "unknown",
];

export type GraphProvenanceCountRecord = Record<PaperGraphDiagnosticProvenance, number>;

export type GraphOpportunityDownstreamImpactSnapshot = {
  note: string;
  lastCycle: {
    rawOpportunityCountByProvenance: GraphProvenanceCountRecord;
    acceptedAsExtraCountByProvenance: GraphProvenanceCountRecord;
    /** Graph no merged expandido (whitelist + extras). */
    survivingToPaperCountByProvenance: GraphProvenanceCountRecord;
    /** recommendedCapital > 0 após capacity. */
    capacityPositiveCountByProvenance: GraphProvenanceCountRecord;
    /** Batch do motor pós diversity. */
    enteredEngineBatchCountByProvenance: GraphProvenanceCountRecord;
    rejectedByEconomicFiltersCountByProvenance: GraphProvenanceCountRecord;
    rejectedByPreEconomicPipelineCountByProvenance: GraphProvenanceCountRecord;
    passedEconomicEntryCountByProvenance: GraphProvenanceCountRecord;
    openedTradesThisCycleCountByProvenance: GraphProvenanceCountRecord;
    closedTradesThisCycleCountByProvenance: GraphProvenanceCountRecord;
    realizedPnLThisCycleByProvenance: GraphProvenanceCountRecord;
  };
  lifetime: {
    openedTradesCountByProvenance: GraphProvenanceCountRecord;
    closedTradesCountByProvenance: GraphProvenanceCountRecord;
    totalRealizedPnLByProvenance: GraphProvenanceCountRecord;
    avgEntryEconomicScoreByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
    avgProgressProbabilityFactorByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
  };
  provenanceComparisonLastCycle: {
    complementary_relaxed: ProvenanceComparisonRow;
    cycle: ProvenanceComparisonRow;
    equivalent: ProvenanceComparisonRow;
  };
};

export type ProvenanceComparisonRow = {
  rawInProbe: number;
  mergedExpanded: number;
  capacityPositive: number;
  engineBatch: number;
  economicRejected: number;
  economicPassed: number;
  opened: number;
  /** Taxa: economicPassed / max(engineBatch,1) */
  economicPassRate: number | null;
};

export type ComplementaryRelaxedPaperOpportunitySample = {
  opportunityId: string;
  stage: "pre_economic_reject" | "economic_reject" | "opened";
  reason: string | null;
  label: string;
};

export type ComplementaryRelaxedPaperImpactSnapshot = {
  rawProducedCount: number;
  acceptedAsExtraCount: number;
  survivingEconomicEntryCount: number;
  rejectedEconomicEntryCount: number;
  openedTradesCount: number;
  closedTradesCount: number;
  totalRealizedPnL: number;
  topRejectionReasons: Array<{ reason: string; count: number }>;
  opportunitySamples: ComplementaryRelaxedPaperOpportunitySample[];
  samplesCap: number;
};

/** Diagnóstico de integridade da propagação `graphDiagnosticProvenance` até trades (GET /api/paper/system). */
export type GraphProvenancePropagationDiagnostics = {
  openedGraphTradesWithProvenanceCount: number;
  openedGraphTradesWithoutProvenanceCount: number;
  closedGraphTradesWithProvenanceCount: number;
  closedGraphTradesWithoutProvenanceCount: number;
  /** Fechados graph sem `graphDiagnosticProvenanceAtOpen` mas classificáveis por `entryProfileKeyAtOpen` / tipo (fallback analytics). */
  fallbackClassifiedCount: number;
  topNullProvenanceEntryProfiles: Array<{ profileKey: string; count: number }>;
};

const METRICS_KEY = "__graphOppPaperImpact_v1";
const RELAXED_SAMPLES_CAP = 8;

function emptyCounts(): GraphProvenanceCountRecord {
  const o = {} as GraphProvenanceCountRecord;
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) o[k] = 0;
  return o;
}

function bump(rec: GraphProvenanceCountRecord, prov: PaperGraphDiagnosticProvenance, n = 1): void {
  rec[prov] = (rec[prov] ?? 0) + n;
}

type CycleAcc = {
  rawProbe: GraphProvenanceCountRecord;
  acceptedExtra: GraphProvenanceCountRecord;
  mergedExpanded: GraphProvenanceCountRecord;
  capacityPositive: GraphProvenanceCountRecord;
  engineBatch: GraphProvenanceCountRecord;
  rejectedEconomic: GraphProvenanceCountRecord;
  rejectedPreEconomic: GraphProvenanceCountRecord;
  passedEconomic: GraphProvenanceCountRecord;
  opened: GraphProvenanceCountRecord;
  closed: GraphProvenanceCountRecord;
  pnlThisCycle: GraphProvenanceCountRecord;
  relaxedRejectionReasons: Record<string, number>;
  relaxedSamples: ComplementaryRelaxedPaperOpportunitySample[];
};

type LifetimeAcc = {
  opened: GraphProvenanceCountRecord;
  closed: GraphProvenanceCountRecord;
  totalPnL: GraphProvenanceCountRecord;
  entryScoreSum: GraphProvenanceCountRecord;
  entryScoreN: GraphProvenanceCountRecord;
  entryProgSum: GraphProvenanceCountRecord;
  entryProgN: GraphProvenanceCountRecord;
};

type MetricsRoot = {
  cycle: CycleAcc;
  publishedLastCycle: null | {
    rawProbe: GraphProvenanceCountRecord;
    acceptedExtra: GraphProvenanceCountRecord;
    mergedExpanded: GraphProvenanceCountRecord;
    capacityPositive: GraphProvenanceCountRecord;
    engineBatch: GraphProvenanceCountRecord;
    rejectedEconomic: GraphProvenanceCountRecord;
    rejectedPreEconomic: GraphProvenanceCountRecord;
    passedEconomic: GraphProvenanceCountRecord;
    opened: GraphProvenanceCountRecord;
    closed: GraphProvenanceCountRecord;
    pnlThisCycle: GraphProvenanceCountRecord;
    relaxedRejectionReasons: Record<string, number>;
    relaxedSamples: ComplementaryRelaxedPaperOpportunitySample[];
  };
  lifetime: LifetimeAcc;
};

function newCycle(): CycleAcc {
  return {
    rawProbe: emptyCounts(),
    acceptedExtra: emptyCounts(),
    mergedExpanded: emptyCounts(),
    capacityPositive: emptyCounts(),
    engineBatch: emptyCounts(),
    rejectedEconomic: emptyCounts(),
    rejectedPreEconomic: emptyCounts(),
    passedEconomic: emptyCounts(),
    opened: emptyCounts(),
    closed: emptyCounts(),
    pnlThisCycle: emptyCounts(),
    relaxedRejectionReasons: {},
    relaxedSamples: [],
  };
}

function getRoot(): MetricsRoot {
  const g = globalThis as unknown as Record<string, MetricsRoot | undefined>;
  if (!g[METRICS_KEY]) {
    g[METRICS_KEY] = {
      cycle: newCycle(),
      publishedLastCycle: null,
      lifetime: {
        opened: emptyCounts(),
        closed: emptyCounts(),
        totalPnL: emptyCounts(),
        entryScoreSum: emptyCounts(),
        entryScoreN: emptyCounts(),
        entryProgSum: emptyCounts(),
        entryProgN: emptyCounts(),
      },
    };
  }
  return g[METRICS_KEY]!;
}

export function normalizePaperGraphProvenance(
  p: string | undefined | null
): PaperGraphDiagnosticProvenance {
  if (!p) return "unknown";
  return (PAPER_GRAPH_PROVENANCE_KEYS as readonly string[]).includes(p) ? (p as PaperGraphDiagnosticProvenance) : "unknown";
}

/**
 * Fallback diagnóstico quando `diagnosticRelationProvenance` falta no raw graph:
 * alinha tipos de oportunidade com buckets de proveniência (não distingue strict/relaxed em complement).
 */
export function inferPaperGraphProvenanceFromOpportunityType(
  opportunityType: PaperOpportunityType | string | undefined
): PaperGraphDiagnosticProvenance {
  switch (opportunityType) {
    case "graph_cycle":
      return "cycle";
    case "graph_equivalence":
    case "graph_equivalence_micro":
      return "equivalent";
    case "graph_subset_micro":
      return "subset";
    case "graph_exclusive_micro":
      return "exclusive";
    case "graph_subset":
      return "subset";
    case "graph_exclusive":
      return "exclusive";
    case "graph_complement":
      return "unknown";
    default:
      return "unknown";
  }
}

/** Normalização paper a partir do `GraphOpportunity` / ranked (sempre define um bucket). */
export function resolveGraphDiagnosticProvenanceForRawGraphOpportunity(
  opp: GraphOpportunity | (GraphOpportunity & { rank?: number })
): PaperGraphDiagnosticProvenance {
  const raw = opp.diagnosticRelationProvenance;
  if (raw) return normalizePaperGraphProvenance(raw);
  return inferPaperGraphProvenanceFromOpportunityType(opp.type);
}

/** Valor a gravar em `graphDiagnosticProvenanceAtOpen` (graph apenas). */
export function resolvedGraphDiagnosticProvenanceForNormalizedOpp(
  opp: NormalizedPaperOpportunity
): PaperGraphDiagnosticProvenance | undefined {
  if (opp.sourceType !== "graph") return undefined;
  const g = opp.graphDiagnosticProvenance;
  if (g != null && typeof g === "string" && g.length > 0) {
    return normalizePaperGraphProvenance(g);
  }
  return inferPaperGraphProvenanceFromOpportunityType(opp.opportunityType);
}

/**
 * Proveniência efectiva para analytics / fecho: campo à abertura, ou inferência por perfil (trades legados).
 */
export function effectiveGraphProvenanceForClosedAnalytics(trade: PaperTrade): PaperGraphDiagnosticProvenance | null {
  if (trade.sourceType !== "graph") return null;
  const atOpen = trade.graphDiagnosticProvenanceAtOpen;
  if (atOpen != null && typeof atOpen === "string" && atOpen.length > 0) {
    return normalizePaperGraphProvenance(atOpen);
  }
  const fromProfile = trade.entryProfileKeyAtOpen?.includes("|")
    ? trade.entryProfileKeyAtOpen.split("|")[1]
    : undefined;
  return inferPaperGraphProvenanceFromOpportunityType(fromProfile ?? trade.opportunityType);
}

export function graphProvenanceFromOpportunity(
  opp: NormalizedPaperOpportunity
): PaperGraphDiagnosticProvenance | null {
  if (opp.sourceType !== "graph") return null;
  const g = opp.graphDiagnosticProvenance;
  if (g != null && typeof g === "string" && g.length > 0) {
    return normalizePaperGraphProvenance(g);
  }
  return inferPaperGraphProvenanceFromOpportunityType(opp.opportunityType);
}

export function buildGraphProvenancePropagationDiagnostics(): GraphProvenancePropagationDiagnostics {
  const { activeTrades, closedTrades } = getPaperPortfolio();
  let openedGraphTradesWithProvenanceCount = 0;
  let openedGraphTradesWithoutProvenanceCount = 0;
  let closedGraphTradesWithProvenanceCount = 0;
  let closedGraphTradesWithoutProvenanceCount = 0;
  let fallbackClassifiedCount = 0;
  const nullProvProfile = new Map<string, number>();

  const bumpNullProfile = (t: PaperTrade): void => {
    const pk = t.entryProfileKeyAtOpen ?? "missing";
    nullProvProfile.set(pk, (nullProvProfile.get(pk) ?? 0) + 1);
  };

  for (const t of activeTrades) {
    if (t.sourceType !== "graph") continue;
    const p = t.graphDiagnosticProvenanceAtOpen;
    if (p != null && typeof p === "string" && p.length > 0) {
      openedGraphTradesWithProvenanceCount += 1;
    } else {
      openedGraphTradesWithoutProvenanceCount += 1;
      bumpNullProfile(t);
    }
  }

  for (const t of closedTrades) {
    if (t.status !== "closed" || t.sourceType !== "graph") continue;
    const p = t.graphDiagnosticProvenanceAtOpen;
    if (p != null && typeof p === "string" && p.length > 0) {
      closedGraphTradesWithProvenanceCount += 1;
    } else {
      closedGraphTradesWithoutProvenanceCount += 1;
      bumpNullProfile(t);
      const inferred = inferPaperGraphProvenanceFromOpportunityType(
        t.entryProfileKeyAtOpen?.includes("|") ? t.entryProfileKeyAtOpen.split("|")[1] : t.opportunityType
      );
      if (inferred !== "unknown") fallbackClassifiedCount += 1;
    }
  }

  const topNullProvenanceEntryProfiles = Array.from(nullProvProfile.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
    .map(([profileKey, count]) => ({ profileKey, count }));

  return {
    openedGraphTradesWithProvenanceCount,
    openedGraphTradesWithoutProvenanceCount,
    closedGraphTradesWithProvenanceCount,
    closedGraphTradesWithoutProvenanceCount,
    fallbackClassifiedCount,
    topNullProvenanceEntryProfiles,
  };
}

function pushRelaxedSample(
  cycle: CycleAcc,
  sample: ComplementaryRelaxedPaperOpportunitySample
): void {
  cycle.relaxedSamples.push(sample);
  while (cycle.relaxedSamples.length > RELAXED_SAMPLES_CAP) cycle.relaxedSamples.shift();
}

function bumpRelaxedReason(cycle: CycleAcc, reason: string): void {
  cycle.relaxedRejectionReasons[reason] = (cycle.relaxedRejectionReasons[reason] ?? 0) + 1;
}

export function resetGraphOpportunityPaperImpactCycle(): void {
  const r = getRoot();
  r.cycle = newCycle();
}

export function finalizeGraphOpportunityPaperImpactCycle(): void {
  const r = getRoot();
  const c = r.cycle;
  r.publishedLastCycle = {
    rawProbe: { ...c.rawProbe },
    acceptedExtra: { ...c.acceptedExtra },
    mergedExpanded: { ...c.mergedExpanded },
    capacityPositive: { ...c.capacityPositive },
    engineBatch: { ...c.engineBatch },
    rejectedEconomic: { ...c.rejectedEconomic },
    rejectedPreEconomic: { ...c.rejectedPreEconomic },
    passedEconomic: { ...c.passedEconomic },
    opened: { ...c.opened },
    closed: { ...c.closed },
    pnlThisCycle: { ...c.pnlThisCycle },
    relaxedRejectionReasons: { ...c.relaxedRejectionReasons },
    relaxedSamples: [...c.relaxedSamples],
  };
}

export function recordGraphRawProbeProvenance(prov: string | undefined): void {
  bump(getRoot().cycle.rawProbe, normalizePaperGraphProvenance(prov));
}

export function recordGraphAcceptedExtraProvenance(prov: string | undefined): void {
  bump(getRoot().cycle.acceptedExtra, normalizePaperGraphProvenance(prov));
}

export function recordGraphMergedExpandedIfApplicable(opp: NormalizedPaperOpportunity): void {
  const p = graphProvenanceFromOpportunity(opp);
  if (p) bump(getRoot().cycle.mergedExpanded, p);
}

export function recordGraphCapacityPositiveIfApplicable(opp: NormalizedPaperOpportunity): void {
  const p = graphProvenanceFromOpportunity(opp);
  if (p) bump(getRoot().cycle.capacityPositive, p);
}

export function recordGraphEngineBatchIfApplicable(opp: NormalizedPaperOpportunity): void {
  const p = graphProvenanceFromOpportunity(opp);
  if (p) bump(getRoot().cycle.engineBatch, p);
}

export function recordGraphRejectPreEconomic(
  opp: NormalizedPaperOpportunity,
  reason: string
): void {
  const p = graphProvenanceFromOpportunity(opp);
  if (!p) return;
  const c = getRoot().cycle;
  bump(c.rejectedPreEconomic, p);
  if (p === "complementary_relaxed") {
    bumpRelaxedReason(c, reason);
    pushRelaxedSample(c, {
      opportunityId: opp.opportunityId,
      stage: "pre_economic_reject",
      reason,
      label: (opp.marketsInvolved[0]?.question ?? "").slice(0, 100),
    });
  }
}

export function recordGraphRejectEconomic(opp: NormalizedPaperOpportunity, reason: string): void {
  const p = graphProvenanceFromOpportunity(opp);
  if (!p) return;
  const c = getRoot().cycle;
  bump(c.rejectedEconomic, p);
  if (p === "complementary_relaxed") {
    bumpRelaxedReason(c, reason);
    pushRelaxedSample(c, {
      opportunityId: opp.opportunityId,
      stage: "economic_reject",
      reason,
      label: (opp.marketsInvolved[0]?.question ?? "").slice(0, 100),
    });
  }
}

export function recordGraphPassedEconomicIfApplicable(opp: NormalizedPaperOpportunity): void {
  const p = graphProvenanceFromOpportunity(opp);
  if (p) bump(getRoot().cycle.passedEconomic, p);
}

export function recordGraphTradeOpened(
  opp: NormalizedPaperOpportunity,
  entryEconomicScore?: number,
  progressProbabilityFactor?: number
): void {
  const p = graphProvenanceFromOpportunity(opp);
  if (!p) return;
  const r = getRoot();
  bump(r.cycle.opened, p);
  bump(r.lifetime.opened, p);
  if (typeof entryEconomicScore === "number" && Number.isFinite(entryEconomicScore)) {
    r.lifetime.entryScoreSum[p] += entryEconomicScore;
    r.lifetime.entryScoreN[p] += 1;
  }
  if (typeof progressProbabilityFactor === "number" && Number.isFinite(progressProbabilityFactor)) {
    r.lifetime.entryProgSum[p] += progressProbabilityFactor;
    r.lifetime.entryProgN[p] += 1;
  }
  if (p === "complementary_relaxed") {
    pushRelaxedSample(r.cycle, {
      opportunityId: opp.opportunityId,
      stage: "opened",
      reason: null,
      label: (opp.marketsInvolved[0]?.question ?? "").slice(0, 100),
    });
  }
}

export function recordGraphTradeClosed(trade: PaperTrade): void {
  const p = effectiveGraphProvenanceForClosedAnalytics(trade);
  if (!p) return;
  const r = getRoot();
  const pnl =
    typeof trade.netRealizedPnL === "number" && Number.isFinite(trade.netRealizedPnL)
      ? trade.netRealizedPnL
      : getClosedTradeNetRealizedPnL(trade, DEFAULT_PAPER_FEE_BUFFER_PER_LEG);
  bump(r.cycle.closed, p);
  bump(r.lifetime.closed, p);
  r.lifetime.totalPnL[p] += pnl;
  r.cycle.pnlThisCycle[p] += pnl;
}

function comparisonRow(
  pub: NonNullable<MetricsRoot["publishedLastCycle"]>,
  k: PaperGraphDiagnosticProvenance
): ProvenanceComparisonRow {
  const engineBatch = pub.engineBatch[k];
  const passed = pub.passedEconomic[k];
  return {
    rawInProbe: pub.rawProbe[k],
    mergedExpanded: pub.mergedExpanded[k],
    capacityPositive: pub.capacityPositive[k],
    engineBatch,
    economicRejected: pub.rejectedEconomic[k],
    economicPassed: passed,
    opened: pub.opened[k],
    economicPassRate: engineBatch > 0 ? Math.round((passed / engineBatch) * 10000) / 10000 : null,
  };
}

function avgOrNull(sum: number, n: number): number | null {
  if (n <= 0) return null;
  return Math.round((sum / n) * 10000) / 10000;
}

export function buildGraphOpportunityDownstreamImpactSnapshot(): GraphOpportunityDownstreamImpactSnapshot {
  const r = getRoot();
  const pub = r.publishedLastCycle;
  const z = emptyCounts();
  const zf = (): GraphProvenanceCountRecord => ({ ...z });
  const nullAvgs = (): Record<PaperGraphDiagnosticProvenance, number | null> => {
    const o = {} as Record<PaperGraphDiagnosticProvenance, number | null>;
    for (const k of PAPER_GRAPH_PROVENANCE_KEYS) o[k] = null;
    return o;
  };

  if (!pub) {
    return {
    note:
      "Sem ciclo paper congelado ainda (publica após o primeiro ciclo completo). Métricas lifetime abaixo acumulam desde o arranque. rawOpportunityCount = janela `cachedGraphRaw` na expansão upstream (não o pool completo do scan).",
      lastCycle: {
        rawOpportunityCountByProvenance: zf(),
        acceptedAsExtraCountByProvenance: zf(),
        survivingToPaperCountByProvenance: zf(),
        capacityPositiveCountByProvenance: zf(),
        enteredEngineBatchCountByProvenance: zf(),
        rejectedByEconomicFiltersCountByProvenance: zf(),
        rejectedByPreEconomicPipelineCountByProvenance: zf(),
        passedEconomicEntryCountByProvenance: zf(),
        openedTradesThisCycleCountByProvenance: zf(),
        closedTradesThisCycleCountByProvenance: zf(),
        realizedPnLThisCycleByProvenance: zf(),
      },
      lifetime: {
        openedTradesCountByProvenance: { ...r.lifetime.opened },
        closedTradesCountByProvenance: { ...r.lifetime.closed },
        totalRealizedPnLByProvenance: { ...r.lifetime.totalPnL },
        avgEntryEconomicScoreByProvenance: nullAvgs(),
        avgProgressProbabilityFactorByProvenance: nullAvgs(),
      },
      provenanceComparisonLastCycle: {
        complementary_relaxed: {
          rawInProbe: 0,
          mergedExpanded: 0,
          capacityPositive: 0,
          engineBatch: 0,
          economicRejected: 0,
          economicPassed: 0,
          opened: 0,
          economicPassRate: null,
        },
        cycle: {
          rawInProbe: 0,
          mergedExpanded: 0,
          capacityPositive: 0,
          engineBatch: 0,
          economicRejected: 0,
          economicPassed: 0,
          opened: 0,
          economicPassRate: null,
        },
        equivalent: {
          rawInProbe: 0,
          mergedExpanded: 0,
          capacityPositive: 0,
          engineBatch: 0,
          economicRejected: 0,
          economicPassed: 0,
          opened: 0,
          economicPassRate: null,
        },
      },
    };
  }

  const avgScore = nullAvgs();
  const avgProg = nullAvgs();
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) {
    avgScore[k] = avgOrNull(r.lifetime.entryScoreSum[k], r.lifetime.entryScoreN[k]);
    avgProg[k] = avgOrNull(r.lifetime.entryProgSum[k], r.lifetime.entryProgN[k]);
  }

  return {
    note:
      "lastCycle = último ciclo paper congelado; lifetime = aberturas/fechos acumulados. rawOpportunity = probe `cachedGraphRaw` na expansão; survivingToPaper = graph no merged expandido.",
    lastCycle: {
      rawOpportunityCountByProvenance: { ...pub.rawProbe },
      acceptedAsExtraCountByProvenance: { ...pub.acceptedExtra },
      survivingToPaperCountByProvenance: { ...pub.mergedExpanded },
      capacityPositiveCountByProvenance: { ...pub.capacityPositive },
      enteredEngineBatchCountByProvenance: { ...pub.engineBatch },
      rejectedByEconomicFiltersCountByProvenance: { ...pub.rejectedEconomic },
      rejectedByPreEconomicPipelineCountByProvenance: { ...pub.rejectedPreEconomic },
      passedEconomicEntryCountByProvenance: { ...pub.passedEconomic },
      openedTradesThisCycleCountByProvenance: { ...pub.opened },
      closedTradesThisCycleCountByProvenance: { ...pub.closed },
      realizedPnLThisCycleByProvenance: { ...pub.pnlThisCycle },
    },
    lifetime: {
      openedTradesCountByProvenance: { ...r.lifetime.opened },
      closedTradesCountByProvenance: { ...r.lifetime.closed },
      totalRealizedPnLByProvenance: { ...r.lifetime.totalPnL },
      avgEntryEconomicScoreByProvenance: avgScore,
      avgProgressProbabilityFactorByProvenance: avgProg,
    },
    provenanceComparisonLastCycle: {
      complementary_relaxed: comparisonRow(pub, "complementary_relaxed"),
      cycle: comparisonRow(pub, "cycle"),
      equivalent: comparisonRow(pub, "equivalent"),
    },
  };
}

export function buildComplementaryRelaxedPaperImpactSnapshot(): ComplementaryRelaxedPaperImpactSnapshot {
  const r = getRoot();
  const pub = r.publishedLastCycle;
  const k = "complementary_relaxed" as const;
  const topReasons = pub
    ? Object.entries(pub.relaxedRejectionReasons)
        .map(([reason, count]) => ({ reason, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 12)
    : [];
  return {
    rawProducedCount: pub?.rawProbe[k] ?? 0,
    acceptedAsExtraCount: pub?.acceptedExtra[k] ?? 0,
    survivingEconomicEntryCount: pub?.passedEconomic[k] ?? 0,
    rejectedEconomicEntryCount: pub?.rejectedEconomic[k] ?? 0,
    openedTradesCount: r.lifetime.opened[k],
    closedTradesCount: r.lifetime.closed[k],
    totalRealizedPnL: Math.round(r.lifetime.totalPnL[k] * 10000) / 10000,
    topRejectionReasons: topReasons,
    opportunitySamples: pub ? [...pub.relaxedSamples] : [],
    samplesCap: RELAXED_SAMPLES_CAP,
  };
}

/** Agregação barata para analytics a partir de trades fechados (líquido + bruto por proveniência). */
export function aggregateClosedTradesByGraphProvenance(
  closed: PaperTrade[],
  feeBufferPerLeg: number = DEFAULT_PAPER_FEE_BUFFER_PER_LEG
): {
  closedCountByProvenance: GraphProvenanceCountRecord;
  /** Soma PnL líquido (compat: mesmo que totalNetPnLByProvenance). */
  totalPnLByProvenance: GraphProvenanceCountRecord;
  totalGrossPnLByProvenance: GraphProvenanceCountRecord;
  totalNetPnLByProvenance: GraphProvenanceCountRecord;
  avgNetPnLByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
  countNetNegativeByProvenance: GraphProvenanceCountRecord;
  countGrossPositiveNetNegativeByProvenance: GraphProvenanceCountRecord;
} {
  const closedCount = emptyCounts();
  const totalGross = emptyCounts();
  const totalNet = emptyCounts();
  const negNet = emptyCounts();
  const grossPosNetNeg = emptyCounts();
  const avgNetPnLByProvenance = {} as Record<PaperGraphDiagnosticProvenance, number | null>;
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) avgNetPnLByProvenance[k] = null;

  for (const t of closed) {
    if (!isClosedTradeWithFiniteRealizedPnl(t) || !isGraphTradeForProvenanceMetrics(t)) continue;
    const p = effectiveGraphProvenanceForClosedAnalytics(t) ?? "unknown";
    closedCount[p] += 1;
    const g = getClosedTradeGrossRealizedPnL(t);
    const n = getClosedTradeNetRealizedPnL(t, feeBufferPerLeg);
    totalGross[p] += g;
    totalNet[p] += n;
    if (n < 0) negNet[p] += 1;
    if (g > 0 && n <= 0) grossPosNetNeg[p] += 1;
  }
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) {
    totalGross[k] = Math.round(totalGross[k] * 10000) / 10000;
    totalNet[k] = Math.round(totalNet[k] * 10000) / 10000;
    avgNetPnLByProvenance[k] =
      closedCount[k] > 0 ? Math.round((totalNet[k] / closedCount[k]) * 10000) / 10000 : null;
  }
  return {
    closedCountByProvenance: closedCount,
    totalPnLByProvenance: { ...totalNet },
    totalGrossPnLByProvenance: totalGross,
    totalNetPnLByProvenance: { ...totalNet },
    avgNetPnLByProvenance,
    countNetNegativeByProvenance: negNet,
    countGrossPositiveNetNegativeByProvenance: grossPosNetNeg,
  };
}
