/**
 * Expansão disciplinada do universo de candidatos **antes** de `estimateBatchCapacity`.
 * Inclui funis de diagnóstico (último ciclo) em `setUpstreamScannerFunnelSnapshots`.
 */

import type { GraphOpportunity } from "./graphArbitrageEngine";
import type { NormalizedMarket } from "./polymarketClient";
import type { NormalizedPaperOpportunity } from "./paperTypes";
import { getUpstreamScannerExpansionPolicySnapshot } from "./paperScannerExpansionEnv";
import { getOpportunityFamilyKey } from "./paperUpstreamDiversity";
import {
  getScannerMarketBaseRecentCount,
  getScannerPairingRecentCount,
  recordScannerExpansionCoverageForOpportunities,
  setUpstreamScannerExpansionLastCycle,
  setUpstreamScannerFunnelSnapshots,
  type UpstreamScannerExpansionDiagnostics,
  type UpstreamScannerRawFunnel,
  type UpstreamScannerCrossFunnel,
  type UpstreamScannerSourceUniverse,
} from "./paperOpenDiagnostics";
import {
  recordGraphAcceptedExtraProvenance,
  recordGraphRawProbeProvenance,
} from "./graphOpportunityPaperImpact";

const GRAPH_MIN_CONF = 0.03;
const GRAPH_MIN_LIQ = 100;
const CROSS_OVERROUND = 1.02;
const CROSS_MIN_LIQ = 100;

function graphCompositeScore(o: GraphOpportunity): number {
  return o.edge * Math.log10(Math.max(1, 1 + o.liquidity)) * o.confidence;
}

function minMarketRecentForGraph(o: GraphOpportunity): number {
  if (o.marketsInvolved.length === 0) return 0;
  let m = Number.POSITIVE_INFINITY;
  for (const x of o.marketsInvolved) {
    const r = getScannerMarketBaseRecentCount(x.marketId);
    m = Math.min(m, r);
  }
  return Number.isFinite(m) ? m : 0;
}

function minMarketRecentForPair(a: NormalizedMarket, b: NormalizedMarket): number {
  return Math.min(getScannerMarketBaseRecentCount(a.id), getScannerMarketBaseRecentCount(b.id));
}

/** Família proxy para estatísticas do pool bruto (sem normalizar). */
function graphFamilyKeyFromRaw(raw: GraphOpportunity): string {
  if (raw.clusterId && String(raw.clusterId).length > 0) return `c:${raw.clusterId}`;
  const ids = raw.marketsInvolved.map((m) => m.marketId).sort();
  if (ids.length > 0) return `m:${ids.join("|")}`;
  return `o:${raw.id}`;
}

export type NormalizeGraphFn = (o: GraphOpportunity) => NormalizedPaperOpportunity;

export type NormalizeStandardCrossFn = (e: {
  marketId: string;
  question: string;
  edge: number;
  type: string;
  confidence: number;
  liquidity: number;
  outcomes?: string[];
  prices?: number[];
}) => NormalizedPaperOpportunity;

export type UpstreamScannerExpansionSourceContext = {
  scannerMode: "whitelist_scan" | "http_opportunities";
  graphScanLastScanMs: number | null;
  graphRankedOpportunitiesCount: number;
  /** Último scan de grafo (mercados → clusters → oportunidades brutas); diagnóstico quando `cachedGraphRaw` está vazio. */
  graphScanCapture?: {
    marketCount: number;
    clusterCount: number;
    rawOpportunityCount: number;
  } | null;
};

function stablePairingDiagKey(aId: string, bId: string): string {
  return aId < bId ? `p:${aId}|${bId}` : `p:${bId}|${aId}`;
}

type CrossPhysicsReject = "binary_outcomes" | "overround" | "liquidity";

function crossPhysicsRejectReason(a: NormalizedMarket, b: NormalizedMarket): CrossPhysicsReject | null {
  if (a.outcomes.length !== 2 || b.outcomes.length !== 2) return "binary_outcomes";
  const pA = a.prices[0];
  const pB = b.prices[0];
  if (pA + pB <= CROSS_OVERROUND) return "overround";
  const minLiq = Math.min(a.liquidity, b.liquidity);
  if (minLiq < CROSS_MIN_LIQ) return "liquidity";
  return null;
}

function tryCrossEdge(a: NormalizedMarket, b: NormalizedMarket): {
  marketId: string;
  question: string;
  edge: number;
  confidence: number;
  liquidity: number;
  outcomes: string[];
  prices: number[];
} | null {
  const r = crossPhysicsRejectReason(a, b);
  if (r != null) return null;
  const pA = a.prices[0];
  const pB = b.prices[0];
  const edge = pA + pB - 1;
  const minLiq = Math.min(a.liquidity, b.liquidity);
  const liquidityFactor = Math.min(1, Math.log10(Math.max(1, minLiq)) / 5);
  const spreadPenalty = Math.min(1, Math.max(0, 1 - ((a.spread + b.spread) / 2) * 2));
  const edgeFactor = Math.min(1, Math.abs(edge) * 10);
  const confidence = Math.max(0, Math.min(1, liquidityFactor * spreadPenalty * edgeFactor));
  return {
    marketId: `${a.id}+${b.id}`,
    question: `${a.question} ↔ ${b.question}`,
    edge,
    confidence,
    liquidity: minLiq,
    outcomes: [a.outcomes[0], b.outcomes[0]],
    prices: [pA, pB],
  };
}

function emptyCrossFunnel(): UpstreamScannerCrossFunnel {
  return {
    marketsConsideredForCrossExpansion: 0,
    marketsPassedToCoverageSort: 0,
    marketsAfterCoverageOrdering: 0,
    crossPairProbeWindowUsed: 0,
    rawPairingsConsideredCount: 0,
    rawPairingsRejectedByBinaryOutcomesCount: 0,
    rawPairingsRejectedByLiquidityCount: 0,
    rawPairingsRejectedByOverroundCount: 0,
    rawPairingsRejectedByRecentPenaltyCount: 0,
    rawPairingsRejectedByDuplicateOpportunityIdCount: 0,
    rawPairingsAcceptedCount: 0,
    crossPairingsDroppedByBudgetCount: 0,
  };
}

function emptyRawFunnel(): UpstreamScannerRawFunnel {
  return {
    cachedGraphRawCount: 0,
    graphRawProbeCount: 0,
    graphRawPassingMinFiltersCount: 0,
    graphRawRejectedByLowConfidenceCount: 0,
    graphRawRejectedByLowLiquidityCount: 0,
    graphRawRejectedByAlreadyMergedCount: 0,
    graphRawPreselectedCount: 0,
    graphRawNormalizationAttemptsCount: 0,
    graphRawNormalizationSuccessCount: 0,
    graphRawNormalizationFailureCount: 0,
    graphRawAcceptedAsExtraCount: 0,
    graphRawDroppedByBudgetCount: 0,
    graphRawRelaxedComplementaryAcceptedAsExtraCount: 0,
  };
}

export function applyUpstreamScannerExpansion(args: {
  merged: NormalizedPaperOpportunity[];
  graphRaw: GraphOpportunity[];
  marketsForCrossExpansion: NormalizedMarket[];
  normalizeGraph: NormalizeGraphFn;
  normalizeStandardCross: NormalizeStandardCrossFn;
  sourceContext?: UpstreamScannerExpansionSourceContext;
}): { merged: NormalizedPaperOpportunity[]; diagnostics: UpstreamScannerExpansionDiagnostics } {
  const policy = getUpstreamScannerExpansionPolicySnapshot();
  const before = args.merged.length;
  const byId = new Map(args.merged.map((o) => [o.opportunityId, o]));
  const origFamilies = new Set(args.merged.map((o) => getOpportunityFamilyKey(o)));
  const origMarketBases = new Set<string>();
  for (const o of args.merged) {
    for (const m of o.marketsInvolved) origMarketBases.add(m.marketId);
  }

  const nowMs = Date.now();
  const graphScanMs = args.sourceContext?.graphScanLastScanMs ?? null;
  const cachedGraphRawAgeMs =
    graphScanMs != null && graphScanMs > 0 ? Math.max(0, nowMs - graphScanMs) : null;

  const uniqueMarketBasesAvailable = new Set(args.marketsForCrossExpansion.map((m) => m.id)).size;

  const familyProbeKeys = new Set<string>();
  const probeLimit = Math.min(args.graphRaw.length, policy.maxGraphRawProbePerCycle);
  for (let i = 0; i < probeLimit; i++) {
    const gr = args.graphRaw[i]!;
    familyProbeKeys.add(graphFamilyKeyFromRaw(gr));
  }
  for (const fk of Array.from(origFamilies)) familyProbeKeys.add(fk);

  const cap = args.sourceContext?.graphScanCapture ?? null;
  const sourceUniverse: UpstreamScannerSourceUniverse = {
    totalMarketsAvailableForExpansion: args.marketsForCrossExpansion.length,
    uniqueMarketBasesAvailableForExpansion: uniqueMarketBasesAvailable,
    uniqueFamiliesAvailableForExpansion: familyProbeKeys.size,
    cachedGraphRawAvailable: args.graphRaw.length > 0,
    graphScanLastScanMs: graphScanMs,
    cachedGraphRawAgeMs,
    graphScanCapture: cap,
    scannerSourceSummary: [
      args.sourceContext?.scannerMode ?? "unknown",
      `merged_in=${before}`,
      `ranked_graph=${args.sourceContext?.graphRankedOpportunitiesCount ?? -1}`,
      `cached_raw=${args.graphRaw.length}`,
      cap != null
        ? `scan_in=${cap.marketCount}|clusters=${cap.clusterCount}|raw_out=${cap.rawOpportunityCount}`
        : "scan_capture=none",
      `markets=${args.marketsForCrossExpansion.length}`,
    ].join("|"),
  };

  const pushFunnel = (raw: UpstreamScannerRawFunnel, cross: UpstreamScannerCrossFunnel) => {
    setUpstreamScannerFunnelSnapshots({ raw, cross, source: sourceUniverse });
  };

  if (!policy.enabled || policy.maxExtraCandidatesPerCycle <= 0) {
    const diag: UpstreamScannerExpansionDiagnostics = {
      totalCandidatesGeneratedBeforeExpansion: before,
      totalCandidatesGeneratedAfterExpansion: before,
      additionalCandidatesIntroduced: 0,
      newFamiliesIntroduced: 0,
      newMarketBasesIntroduced: 0,
      pairingsAddedByExpansion: 0,
      candidatesDroppedByScannerBudget: 0,
      scannerBudgetUsage: {
        maxExtraCandidates: policy.maxExtraCandidatesPerCycle,
        usedExtraCandidates: 0,
        maxGraphExtras: policy.maxExtraGraphCandidatesPerCycle,
        usedGraphExtras: 0,
        maxPairings: policy.maxNewCrossPairingsPerCycle,
        usedPairings: 0,
      },
      noveltyCoverageUsage: { graphNoveltyFirstPicks: 0, crossPairNoveltyFirstPicks: 0 },
    };
    setUpstreamScannerExpansionLastCycle(diag);
    const pl = Math.min(args.graphRaw.length, policy.maxGraphRawProbePerCycle);
    pushFunnel(
      { ...emptyRawFunnel(), cachedGraphRawCount: args.graphRaw.length, graphRawProbeCount: pl },
      emptyCrossFunnel()
    );
    return { merged: args.merged, diagnostics: diag };
  }

  let usedExtra = 0;
  let usedGraph = 0;
  let usedPairings = 0;
  let pairingsAdded = 0;
  let crossDropped = 0;
  let graphNov = 0;
  let crossNov = 0;
  const extras: NormalizedPaperOpportunity[] = [];

  const rawFunnel: UpstreamScannerRawFunnel = {
    cachedGraphRawCount: args.graphRaw.length,
    graphRawProbeCount: probeLimit,
    graphRawPassingMinFiltersCount: 0,
    graphRawRejectedByLowConfidenceCount: 0,
    graphRawRejectedByLowLiquidityCount: 0,
    graphRawRejectedByAlreadyMergedCount: 0,
    graphRawPreselectedCount: 0,
    graphRawNormalizationAttemptsCount: 0,
    graphRawNormalizationSuccessCount: 0,
    graphRawNormalizationFailureCount: 0,
    graphRawAcceptedAsExtraCount: 0,
    graphRawDroppedByBudgetCount: 0,
    graphRawRelaxedComplementaryAcceptedAsExtraCount: 0,
  };

  type PreGraph = { raw: GraphOpportunity; score: number; nov: number };
  const preGraph: PreGraph[] = [];

  for (let i = 0; i < probeLimit; i++) {
    const raw = args.graphRaw[i]!;
    recordGraphRawProbeProvenance(raw.diagnosticRelationProvenance);
    if (raw.confidence < GRAPH_MIN_CONF) {
      rawFunnel.graphRawRejectedByLowConfidenceCount += 1;
      continue;
    }
    if (raw.liquidity < GRAPH_MIN_LIQ) {
      rawFunnel.graphRawRejectedByLowLiquidityCount += 1;
      continue;
    }
    rawFunnel.graphRawPassingMinFiltersCount += 1;
    if (byId.has(raw.id)) {
      rawFunnel.graphRawRejectedByAlreadyMergedCount += 1;
      continue;
    }
    preGraph.push({
      raw,
      score: graphCompositeScore(raw),
      nov: minMarketRecentForGraph(raw),
    });
  }

  rawFunnel.graphRawPreselectedCount = preGraph.length;

  preGraph.sort((a, b) => (a.nov !== b.nov ? a.nov - b.nov : b.score - a.score));

  let graphNormalizeAttempts = 0;
  for (const p of preGraph) {
    if (usedExtra >= policy.maxExtraCandidatesPerCycle) break;
    if (usedGraph >= policy.maxExtraGraphCandidatesPerCycle) break;
    if (graphNormalizeAttempts >= policy.maxGraphNormalizeAttempts) break;
    graphNormalizeAttempts += 1;
    const norm = args.normalizeGraph(p.raw);
    if (byId.has(norm.opportunityId)) {
      rawFunnel.graphRawNormalizationFailureCount += 1;
      continue;
    }
    byId.set(norm.opportunityId, norm);
    extras.push(norm);
    usedExtra += 1;
    usedGraph += 1;
    recordGraphAcceptedExtraProvenance(p.raw.diagnosticRelationProvenance);
    if (p.raw.diagnosticRelationProvenance === "complementary_relaxed") {
      rawFunnel.graphRawRelaxedComplementaryAcceptedAsExtraCount += 1;
    }
    if (p.nov < 2) graphNov += 1;
  }

  rawFunnel.graphRawNormalizationAttemptsCount = graphNormalizeAttempts;
  rawFunnel.graphRawNormalizationSuccessCount = rawFunnel.graphRawAcceptedAsExtraCount = usedGraph;

  rawFunnel.graphRawDroppedByBudgetCount =
    rawFunnel.graphRawPreselectedCount -
    rawFunnel.graphRawAcceptedAsExtraCount -
    rawFunnel.graphRawNormalizationFailureCount;

  const crossFunnel: UpstreamScannerCrossFunnel = emptyCrossFunnel();
  const marketPool = args.marketsForCrossExpansion;
  const marketCap = Math.min(marketPool.length, 600);
  crossFunnel.marketsConsideredForCrossExpansion = marketPool.length;
  crossFunnel.marketsPassedToCoverageSort = marketCap;

  const W = Math.min(marketCap, policy.crossPairingProbeWindow);
  crossFunnel.crossPairProbeWindowUsed = W;

  if (W >= 2 && usedExtra < policy.maxExtraCandidatesPerCycle && policy.maxNewCrossPairingsPerCycle > 0) {
    const scoredMk = marketPool.slice(0, marketCap).map((m) => ({
      m,
      r: getScannerMarketBaseRecentCount(m.id),
    }));
    scoredMk.sort((a, b) => (a.r !== b.r ? a.r - b.r : b.m.liquidity - a.m.liquidity));
    crossFunnel.marketsAfterCoverageOrdering = scoredMk.length;
    const window = scoredMk.slice(0, W).map((x) => x.m);

    type CrossCand = {
      norm: NormalizedPaperOpportunity;
      diversity: number;
      edge: number;
    };
    const crossList: CrossCand[] = [];
    const seenCrossOpp = new Set<string>();

    for (let i = 0; i < window.length; i++) {
      for (let j = i + 1; j < window.length; j++) {
        const a = window[i]!;
        const b = window[j]!;
        crossFunnel.rawPairingsConsideredCount += 1;
        const phys = crossPhysicsRejectReason(a, b);
        if (phys === "binary_outcomes") {
          crossFunnel.rawPairingsRejectedByBinaryOutcomesCount += 1;
          continue;
        }
        if (phys === "overround") {
          crossFunnel.rawPairingsRejectedByOverroundCount += 1;
          continue;
        }
        if (phys === "liquidity") {
          crossFunnel.rawPairingsRejectedByLiquidityCount += 1;
          continue;
        }
        const diagPk = stablePairingDiagKey(a.id, b.id);
        if (getScannerPairingRecentCount(diagPk) > 8) {
          crossFunnel.rawPairingsRejectedByRecentPenaltyCount += 1;
          continue;
        }
        const e = tryCrossEdge(a, b);
        if (!e) continue;
        const norm = args.normalizeStandardCross({
          marketId: e.marketId,
          question: e.question,
          edge: e.edge,
          type: "cross_market",
          confidence: e.confidence,
          liquidity: e.liquidity,
          outcomes: e.outcomes,
          prices: e.prices,
        });
        if (byId.has(norm.opportunityId) || seenCrossOpp.has(norm.opportunityId)) {
          crossFunnel.rawPairingsRejectedByDuplicateOpportunityIdCount += 1;
          continue;
        }
        seenCrossOpp.add(norm.opportunityId);
        crossList.push({
          norm,
          diversity: minMarketRecentForPair(a, b),
          edge: e.edge,
        });
      }
    }

    crossFunnel.rawPairingsAcceptedCount = crossList.length;

    crossList.sort((a, b) => (a.diversity !== b.diversity ? a.diversity - b.diversity : b.edge - a.edge));

    const crossEligible = crossList.length;
    const pairingsBeforeCross = usedPairings;
    for (const c of crossList) {
      if (usedExtra >= policy.maxExtraCandidatesPerCycle) break;
      if (usedPairings >= policy.maxNewCrossPairingsPerCycle) break;
      if (byId.has(c.norm.opportunityId)) continue;
      byId.set(c.norm.opportunityId, c.norm);
      extras.push(c.norm);
      usedExtra += 1;
      usedPairings += 1;
      pairingsAdded += 1;
      if (c.diversity < 2) crossNov += 1;
    }
    crossDropped = Math.max(0, crossEligible - (usedPairings - pairingsBeforeCross));
    crossFunnel.crossPairingsDroppedByBudgetCount = crossDropped;
  }

  const mergedOut = [...args.merged, ...extras];

  let newFamiliesIntroduced = 0;
  let newMarketBasesIntroduced = 0;
  const seenF = new Set<string>();
  const seenMb = new Set<string>();
  for (const o of extras) {
    const fk = getOpportunityFamilyKey(o);
    if (!origFamilies.has(fk) && !seenF.has(fk)) {
      seenF.add(fk);
      newFamiliesIntroduced += 1;
    }
    for (const m of o.marketsInvolved) {
      if (!origMarketBases.has(m.marketId) && !seenMb.has(m.marketId)) {
        seenMb.add(m.marketId);
        newMarketBasesIntroduced += 1;
      }
    }
  }

  const diagnostics: UpstreamScannerExpansionDiagnostics = {
    totalCandidatesGeneratedBeforeExpansion: before,
    totalCandidatesGeneratedAfterExpansion: mergedOut.length,
    additionalCandidatesIntroduced: extras.length,
    newFamiliesIntroduced,
    newMarketBasesIntroduced,
    pairingsAddedByExpansion: pairingsAdded,
    candidatesDroppedByScannerBudget:
      rawFunnel.graphRawDroppedByBudgetCount + crossFunnel.crossPairingsDroppedByBudgetCount,
    scannerBudgetUsage: {
      maxExtraCandidates: policy.maxExtraCandidatesPerCycle,
      usedExtraCandidates: usedExtra,
      maxGraphExtras: policy.maxExtraGraphCandidatesPerCycle,
      usedGraphExtras: usedGraph,
      maxPairings: policy.maxNewCrossPairingsPerCycle,
      usedPairings,
    },
    noveltyCoverageUsage: {
      graphNoveltyFirstPicks: graphNov,
      crossPairNoveltyFirstPicks: crossNov,
    },
  };

  if (extras.length > 0) {
    recordScannerExpansionCoverageForOpportunities(extras);
  }
  setUpstreamScannerExpansionLastCycle(diagnostics);
  pushFunnel(rawFunnel, crossFunnel);

  return { merged: mergedOut, diagnostics };
}
