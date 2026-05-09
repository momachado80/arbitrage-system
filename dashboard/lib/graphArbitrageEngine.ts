import type { NormalizedMarket } from "./polymarketClient";
import type { ConstraintCluster } from "./marketRelationBuilder";
import { buildGraphFromCluster, type ConstraintViolation } from "./probabilityGraph";
import {
  classifyEquivalenceMicroStructuralLaneWithReason,
  type StructuralAssignmentReason,
} from "./graphStructuralMicroLane";

/** Proveniência da aresta que origina a violação (diagnóstico; não afecta ranking). */
export type DiagnosticRelationProvenance =
  | "equivalent"
  | "subset"
  | "exclusive"
  | "complementary_strict"
  | "complementary_relaxed"
  | "cycle"
  | "unknown";

function diagnosticProvenanceForViolation(
  cluster: ConstraintCluster,
  violation: ConstraintViolation
): DiagnosticRelationProvenance {
  if (violation.type === "cycle") return "cycle";
  if (violation.nodeIds.length < 2) return "unknown";
  const x = violation.nodeIds[0]!;
  const y = violation.nodeIds[1]!;
  const rel = cluster.relations.find(
    (r) =>
      (r.sourceMarketId === x && r.targetMarketId === y) ||
      (r.sourceMarketId === y && r.targetMarketId === x)
  );
  if (!rel) return "unknown";
  switch (rel.type) {
    case "equivalent":
      return "equivalent";
    case "subset":
      return "subset";
    case "exclusive":
      return "exclusive";
    case "complementary":
      return rel.complementaryInferencePath === "relaxed" ? "complementary_relaxed" : "complementary_strict";
    default:
      return "unknown";
  }
}

export type GraphOpportunityType =
  | "graph_subset"
  | "graph_complement"
  | "graph_exclusive"
  | "graph_equivalence"
  | "graph_equivalence_micro"
  | "graph_subset_micro"
  | "graph_exclusive_micro"
  | "graph_cycle";

export interface GraphOpportunity {
  id: string;
  type: GraphOpportunityType;
  title: string;
  description: string;
  edge: number;
  confidence: number;
  liquidity: number;
  marketsInvolved: Array<{
    marketId: string;
    question: string;
    observedProb: number;
    liquidity: number;
  }>;
  clusterId: string;
  detectedAt: string;
  /** Diagnóstico: aresta do cluster que gerou a violação (quando identificável). */
  diagnosticRelationProvenance?: DiagnosticRelationProvenance;
  /** Só micro-lanes estruturais pós-reclassificação (pureza estrutural). */
  structuralMicroLaneReason?: StructuralAssignmentReason;
}

function violationTypeToOpportunityType(v: ConstraintViolation["type"]): GraphOpportunityType {
  switch (v) {
    case "subset": return "graph_subset";
    case "complement": return "graph_complement";
    case "exclusive": return "graph_exclusive";
    case "equivalence": return "graph_equivalence";
    case "cycle": return "graph_cycle";
  }
}

function violationTitle(v: ConstraintViolation): string {
  switch (v.type) {
    case "subset": return "Violação de subconjunto";
    case "complement": return "Complementos inconsistentes";
    case "exclusive": return "Exclusividade violada";
    case "equivalence": return "Mercados equivalentes divergentes";
    case "cycle": return "Inconsistência cíclica";
  }
}

export type ScanClustersDiagnostics = {
  clustersScannedOk: number;
  clustersFailedInGraphScanCount: number;
};

export function scanClustersWithDiagnostics(clusters: ConstraintCluster[]): {
  opportunities: GraphOpportunity[];
  diagnostics: ScanClustersDiagnostics;
} {
  const t0 = Date.now();
  const opportunities: GraphOpportunity[] = [];
  let clustersScannedOk = 0;
  let clustersFailedInGraphScanCount = 0;

  for (const cluster of clusters) {
    try {
      const graph = buildGraphFromCluster(cluster);
      clustersScannedOk++;

      for (const violation of graph.violations) {
        const involvedMarkets: GraphOpportunity["marketsInvolved"] = [];
        let minLiquidity = Infinity;

        for (const nodeId of violation.nodeIds) {
          const node = graph.nodes.get(nodeId);
          if (!node) continue;
          involvedMarkets.push({
            marketId: node.marketId,
            question: node.question,
            observedProb: node.observedProb,
            liquidity: node.liquidity,
          });
          minLiquidity = Math.min(minLiquidity, node.liquidity);
        }

        if (involvedMarkets.length < 2) continue;
        if (minLiquidity === Infinity) minLiquidity = 0;

        const liquidityFactor = Math.min(1, Math.log10(Math.max(1, minLiquidity)) / 5);
        const adjustedConfidence = violation.confidence * liquidityFactor;

        if (adjustedConfidence < 0.02) continue;

        const diagnosticRelationProvenance = diagnosticProvenanceForViolation(cluster, violation);

        let resolvedType = violationTypeToOpportunityType(violation.type);

        let structuralMicroLaneReason: StructuralAssignmentReason | undefined;

        if (violation.type === "equivalence") {
          const qualifiesMicro =
            involvedMarkets.length === 2 &&
            violation.severity >= 0.08 &&
            minLiquidity >= 500 &&
            violation.nodeIds.length === 2 &&
            (() => {
              const nA = graph.nodes.get(violation.nodeIds[0]!);
              const nB = graph.nodes.get(violation.nodeIds[1]!);
              return nA != null && nB != null && nA.outcomesCount === 2 && nB.outcomesCount === 2;
            })();
          if (qualifiesMicro) {
            const mid0 = violation.nodeIds[0]!;
            const mid1 = violation.nodeIds[1]!;
            const ma = cluster.markets.find((m) => m.id === mid0);
            const mb = cluster.markets.find((m) => m.id === mid1);
            if (ma != null && mb != null) {
              const cl = classifyEquivalenceMicroStructuralLaneWithReason(ma, mb);
              resolvedType = cl.lane;
              structuralMicroLaneReason = cl.reason;
            } else {
              resolvedType = "graph_equivalence_micro";
              structuralMicroLaneReason = "residual_not_pure_equivalence_nor_monotonic_subset";
            }
          }
        }

        opportunities.push({
          id: `graph-${cluster.clusterId}-${violation.type}-${opportunities.length}`,
          type: resolvedType,
          title: violationTitle(violation),
          description: violation.description,
          edge: violation.severity,
          confidence: adjustedConfidence,
          liquidity: minLiquidity,
          marketsInvolved: involvedMarkets,
          clusterId: cluster.clusterId,
          detectedAt: new Date().toISOString(),
          diagnosticRelationProvenance,
          structuralMicroLaneReason,
        });
      }
    } catch (err) {
      clustersFailedInGraphScanCount += 1;
      console.error(`[GraphArbEngine] Cluster ${cluster.clusterId} failed:`, err);
    }
  }

  const elapsed = Date.now() - t0;
  console.log(
    `[GraphArbEngine] Scanned ${clustersScannedOk} clusters ok (${clustersFailedInGraphScanCount} failed) → ${opportunities.length} graph opportunities in ${elapsed}ms`
  );

  return {
    opportunities,
    diagnostics: { clustersScannedOk, clustersFailedInGraphScanCount },
  };
}

export function scanClusters(clusters: ConstraintCluster[]): GraphOpportunity[] {
  return scanClustersWithDiagnostics(clusters).opportunities;
}
