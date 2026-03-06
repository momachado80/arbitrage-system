import type { NormalizedMarket } from "./polymarketClient";
import type { ConstraintCluster } from "./marketRelationBuilder";
import { buildGraphFromCluster, type ConstraintViolation } from "./probabilityGraph";

export type GraphOpportunityType =
  | "graph_subset"
  | "graph_complement"
  | "graph_exclusive"
  | "graph_equivalence"
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

export function scanClusters(clusters: ConstraintCluster[]): GraphOpportunity[] {
  const t0 = Date.now();
  const opportunities: GraphOpportunity[] = [];
  let clustersScanned = 0;

  for (const cluster of clusters) {
    try {
      const graph = buildGraphFromCluster(cluster);
      clustersScanned++;

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

        opportunities.push({
          id: `graph-${cluster.clusterId}-${violation.type}-${opportunities.length}`,
          type: violationTypeToOpportunityType(violation.type),
          title: violationTitle(violation),
          description: violation.description,
          edge: violation.severity,
          confidence: adjustedConfidence,
          liquidity: minLiquidity,
          marketsInvolved: involvedMarkets,
          clusterId: cluster.clusterId,
          detectedAt: new Date().toISOString(),
        });
      }
    } catch (err) {
      console.error(`[GraphArbEngine] Cluster ${cluster.clusterId} failed:`, err);
    }
  }

  const elapsed = Date.now() - t0;
  console.log(
    `[GraphArbEngine] Scanned ${clustersScanned} clusters → ${opportunities.length} graph opportunities in ${elapsed}ms`
  );

  return opportunities;
}
