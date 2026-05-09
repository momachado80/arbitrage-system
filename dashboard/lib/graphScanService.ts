import { getAllMarkets } from "./marketDataService";
import { buildClustersWithDiagnostics, type ConstraintCluster, type MarketRelation } from "./marketRelationBuilder";
import { scanClustersWithDiagnostics, type GraphOpportunity } from "./graphArbitrageEngine";
import {
  emptyClusterFormationFunnel,
  emptyGraphSourceQuality,
  emptyRelationBuilderFunnel,
  type ComplementaryRelaxedDownstreamImpact,
  type ComplementaryRelaxedOpportunitySampleRow,
  type GraphPipelineDiagnosticsSnapshot,
  type RawOpportunityProvenanceCounts,
} from "./graphPipelineDiagnostics";
import { getUpstreamScannerRawFunnelLastCycle } from "./paperOpenDiagnostics";
import {
  rankGraphOpportunities,
  computeGraphSummary,
  type RankedGraphOpportunity,
  type GraphSummary,
} from "./graphOpportunityEngine";
import { updateGraphEpisodes } from "./graphEpisodeStore";
import { getGraphScanRuntime } from "./nodeProcessRuntimeState";
import { buildStructuralMicroLaneScanSnapshot } from "./graphStructuralMicroLane";

const SCAN_INTERVAL_MS = 6_000;
const MAX_COMPLEMENTARY_RELAXED_OPPORTUNITY_SAMPLES = 10;

function emptyRawOpportunityProvenanceCounts(): RawOpportunityProvenanceCounts {
  return {
    total: 0,
    equivalent: 0,
    subset: 0,
    exclusive: 0,
    complementaryStrict: 0,
    complementaryRelaxed: 0,
    cycle: 0,
    unknown: 0,
  };
}

function bumpProvenance(acc: RawOpportunityProvenanceCounts, prov: string | undefined): void {
  acc.total += 1;
  switch (prov) {
    case "equivalent":
      acc.equivalent += 1;
      break;
    case "subset":
      acc.subset += 1;
      break;
    case "exclusive":
      acc.exclusive += 1;
      break;
    case "complementary_strict":
      acc.complementaryStrict += 1;
      break;
    case "complementary_relaxed":
      acc.complementaryRelaxed += 1;
      break;
    case "cycle":
      acc.cycle += 1;
      break;
    default:
      acc.unknown += 1;
  }
}

function buildComplementaryRelaxedDownstreamImpact(args: {
  allRelationsFlat: MarketRelation[];
  clusters: ConstraintCluster[];
  opportunities: GraphOpportunity[];
}): ComplementaryRelaxedDownstreamImpact {
  const { allRelationsFlat, clusters, opportunities } = args;
  const relaxedRels = allRelationsFlat.filter(
    (r) => r.type === "complementary" && r.complementaryInferencePath === "relaxed"
  ).length;

  let clusterContrib = 0;
  for (const c of clusters) {
    if (c.relations.some((r) => r.type === "complementary" && r.complementaryInferencePath === "relaxed")) {
      clusterContrib += 1;
    }
  }

  const provCounts = emptyRawOpportunityProvenanceCounts();
  const clusterById = new Map(clusters.map((c) => [c.clusterId, c]));
  const samples: ComplementaryRelaxedOpportunitySampleRow[] = [];

  let relaxedRawComplement = 0;
  for (const o of opportunities) {
    bumpProvenance(provCounts, o.diagnosticRelationProvenance);
    if (o.type === "graph_complement" && o.diagnosticRelationProvenance === "complementary_relaxed") {
      relaxedRawComplement += 1;
      if (samples.length < MAX_COMPLEMENTARY_RELAXED_OPPORTUNITY_SAMPLES) {
        const cl = clusterById.get(o.clusterId);
        const m0 = o.marketsInvolved[0];
        const m1 = o.marketsInvolved[1];
        samples.push({
          opportunityId: o.id,
          graphOpportunityType: o.type,
          diagnosticRelationProvenance: o.diagnosticRelationProvenance ?? "unknown",
          clusterId: o.clusterId,
          clusterMarketCount: cl ? cl.markets.length : null,
          clusterRelationCount: cl ? cl.relations.length : null,
          marketIdA: m0?.marketId ?? "",
          marketIdB: m1?.marketId ?? "",
          labelA: (m0?.question ?? "").slice(0, 80),
          labelB: (m1?.question ?? "").slice(0, 80),
        });
      }
    }
  }

  const totalRels = allRelationsFlat.length;
  const totalRaw = opportunities.length;
  const funnel = getUpstreamScannerRawFunnelLastCycle();
  const paperSurvive =
    funnel != null ? (funnel.graphRawRelaxedComplementaryAcceptedAsExtraCount ?? 0) : null;

  return {
    complementaryRelaxedRelationsAcceptedCount: relaxedRels,
    complementaryRelaxedClustersContributedCount: clusterContrib,
    complementaryRelaxedRawOpportunitiesProducedCount: relaxedRawComplement,
    complementaryRelaxedOpportunitiesSurvivingToPaperCount: paperSurvive,
    complementaryRelaxedShareOfRawOpportunities:
      totalRaw > 0 ? Math.round((relaxedRawComplement / totalRaw) * 10000) / 10000 : null,
    complementaryRelaxedShareOfAcceptedRelations:
      totalRels > 0 ? Math.round((relaxedRels / totalRels) * 10000) / 10000 : null,
    rawOpportunityProvenanceCounts: provCounts,
    complementaryRelaxedOpportunitySamples: samples,
  };
}

async function runScan(): Promise<void> {
  const st = getGraphScanRuntime();
  if (st.scanning) return;
  st.scanning = true;
  try {
    const markets = getAllMarkets();
    const marketCount = markets.length;
    if (marketCount === 0) {
      st.lastGraphScanCapture = { marketCount: 0, clusterCount: 0, rawOpportunityCount: 0 };
      st.lastGraphPipelineDiagnostics = {
        relationBuilderFunnel: emptyRelationBuilderFunnel(0),
        clusterFormationFunnel: emptyClusterFormationFunnel(),
        graphSourceQuality: emptyGraphSourceQuality(),
        capturedAtMs: Date.now(),
      };
      st.lastScanMs = Date.now();
      return;
    }

    const built = buildClustersWithDiagnostics(markets);
    const clusters = built.clusters;
    const scanOut = scanClustersWithDiagnostics(clusters);
    const opportunities = scanOut.opportunities;
    const ranked = rankGraphOpportunities(opportunities);
    const summary = computeGraphSummary(ranked, clusters.length);

    built.clusterFormationFunnel.rawOpportunitiesProducedCount = opportunities.length;
    built.clusterFormationFunnel.clustersRejectedByInvalidStructureCount =
      scanOut.diagnostics.clustersFailedInGraphScanCount;

    built.relationBuilderFunnel.complementaryRelaxedDownstreamImpact =
      buildComplementaryRelaxedDownstreamImpact({
        allRelationsFlat: built.allRelationsFlat,
        clusters,
        opportunities,
      });

    const pipeline: GraphPipelineDiagnosticsSnapshot = {
      relationBuilderFunnel: built.relationBuilderFunnel,
      clusterFormationFunnel: built.clusterFormationFunnel,
      graphSourceQuality: built.graphSourceQuality,
      capturedAtMs: Date.now(),
    };
    st.lastGraphPipelineDiagnostics = pipeline;

    st.cachedGraphRaw = opportunities;
    st.cachedRanked = ranked;
    st.cachedSummary = summary;
    st.lastScanMs = Date.now();
    st.lastStructuralMicroLaneScan = buildStructuralMicroLaneScanSnapshot(opportunities);
    st.lastGraphScanCapture = {
      marketCount,
      clusterCount: clusters.length,
      rawOpportunityCount: opportunities.length,
    };

    try {
      updateGraphEpisodes(opportunities);
    } catch {
      /* non-fatal */
    }
  } catch (err) {
    console.error("[GraphScanService] Scan failed:", err);
  } finally {
    st.scanning = false;
  }
}

function startLoop(): void {
  const st = getGraphScanRuntime();
  if (st.loopStarted) {
    console.log("[GraphScanService] loop_skip_already_active_globalThis");
    return;
  }
  st.loopStarted = true;
  console.log("[GraphScanService] Background graph scan loop started (effective; single process)");
  void runScan();
  setInterval(() => {
    void runScan();
  }, SCAN_INTERVAL_MS);
}

export function ensureGraphScanning(): void {
  startLoop();
}

export function getGraphOpportunities(): RankedGraphOpportunity[] {
  ensureGraphScanning();
  return getGraphScanRuntime().cachedRanked;
}

export function getGraphSummary(): GraphSummary {
  ensureGraphScanning();
  return getGraphScanRuntime().cachedSummary;
}

export function getGraphScanStats() {
  const st = getGraphScanRuntime();
  return {
    lastScanMs: st.lastScanMs,
    isScanning: st.scanning,
    opportunitiesCount: st.cachedRanked.length,
    rawPoolCount: st.cachedGraphRaw.length,
    lastGraphScanCapture: st.lastGraphScanCapture,
    lastGraphPipelineDiagnostics: st.lastGraphPipelineDiagnostics,
  };
}
