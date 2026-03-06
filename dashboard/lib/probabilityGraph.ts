import type { NormalizedMarket } from "./polymarketClient";
import type { ConstraintCluster, RelationType } from "./marketRelationBuilder";

export interface GraphNode {
  marketId: string;
  question: string;
  observedProb: number;
  liquidity: number;
  spread: number;
  impliedLowerBound: number;
  impliedUpperBound: number;
}

export interface GraphEdge {
  type: RelationType;
  sourceId: string;
  targetId: string;
  confidence: number;
  impliedConstraint: string;
}

export interface ConstraintViolation {
  type: "subset" | "complement" | "exclusive" | "equivalence" | "cycle";
  description: string;
  severity: number;
  confidence: number;
  nodeIds: string[];
}

export interface ProbabilityGraph {
  clusterId: string;
  nodes: Map<string, GraphNode>;
  edges: GraphEdge[];
  violations: ConstraintViolation[];
}

function primaryProb(m: NormalizedMarket): number {
  return m.prices.length > 0 ? m.prices[0] : 0;
}

export function buildGraphFromCluster(cluster: ConstraintCluster): ProbabilityGraph {
  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];

  for (const m of cluster.markets) {
    nodes.set(m.id, {
      marketId: m.id,
      question: m.question,
      observedProb: primaryProb(m),
      liquidity: m.liquidity,
      spread: m.spread,
      impliedLowerBound: 0,
      impliedUpperBound: 1,
    });
  }

  for (const rel of cluster.relations) {
    let constraint = "";
    switch (rel.type) {
      case "subset":
        constraint = `P(${rel.sourceMarketId}) <= P(${rel.targetMarketId})`;
        break;
      case "complementary":
        constraint = `P(${rel.sourceMarketId}) + P(${rel.targetMarketId}) ≈ 1`;
        break;
      case "exclusive":
        constraint = `P(${rel.sourceMarketId}) + P(${rel.targetMarketId}) <= 1`;
        break;
      case "equivalent":
        constraint = `P(${rel.sourceMarketId}) ≈ P(${rel.targetMarketId})`;
        break;
    }
    edges.push({
      type: rel.type,
      sourceId: rel.sourceMarketId,
      targetId: rel.targetMarketId,
      confidence: rel.confidence,
      impliedConstraint: constraint,
    });
  }

  propagateBounds(nodes, edges);

  const violations = getConstraintViolations({ clusterId: cluster.clusterId, nodes, edges, violations: [] });

  return { clusterId: cluster.clusterId, nodes, edges, violations };
}

function propagateBounds(nodes: Map<string, GraphNode>, edges: GraphEdge[]): void {
  for (const edge of edges) {
    const src = nodes.get(edge.sourceId);
    const tgt = nodes.get(edge.targetId);
    if (!src || !tgt) continue;

    switch (edge.type) {
      case "subset":
        src.impliedUpperBound = Math.min(src.impliedUpperBound, tgt.observedProb);
        tgt.impliedLowerBound = Math.max(tgt.impliedLowerBound, src.observedProb);
        break;
      case "equivalent":
        src.impliedLowerBound = Math.max(src.impliedLowerBound, tgt.observedProb - 0.03);
        src.impliedUpperBound = Math.min(src.impliedUpperBound, tgt.observedProb + 0.03);
        tgt.impliedLowerBound = Math.max(tgt.impliedLowerBound, src.observedProb - 0.03);
        tgt.impliedUpperBound = Math.min(tgt.impliedUpperBound, src.observedProb + 0.03);
        break;
      case "exclusive":
        src.impliedUpperBound = Math.min(src.impliedUpperBound, 1 - tgt.observedProb);
        tgt.impliedUpperBound = Math.min(tgt.impliedUpperBound, 1 - src.observedProb);
        break;
      case "complementary":
        src.impliedLowerBound = Math.max(src.impliedLowerBound, 1 - tgt.observedProb - 0.05);
        src.impliedUpperBound = Math.min(src.impliedUpperBound, 1 - tgt.observedProb + 0.05);
        break;
    }
  }
}

export function getConstraintViolations(graph: ProbabilityGraph): ConstraintViolation[] {
  const violations: ConstraintViolation[] = [];

  for (const edge of graph.edges) {
    const src = graph.nodes.get(edge.sourceId);
    const tgt = graph.nodes.get(edge.targetId);
    if (!src || !tgt) continue;

    const pA = src.observedProb;
    const pB = tgt.observedProb;

    switch (edge.type) {
      case "subset": {
        if (pA > pB + 0.02) {
          violations.push({
            type: "subset",
            description: `P("${src.question.slice(0, 50)}") = ${pA.toFixed(3)} > P("${tgt.question.slice(0, 50)}") = ${pB.toFixed(3)}, but A ⊆ B`,
            severity: pA - pB,
            confidence: edge.confidence * Math.min(1, (pA - pB) * 10),
            nodeIds: [src.marketId, tgt.marketId],
          });
        }
        break;
      }
      case "complementary": {
        const sum = pA + pB;
        const deviation = Math.abs(sum - 1);
        if (deviation > 0.05) {
          violations.push({
            type: "complement",
            description: `P(A) + P(B) = ${sum.toFixed(3)}, expected ≈ 1.0 (deviation: ${(deviation * 100).toFixed(1)}%)`,
            severity: deviation,
            confidence: edge.confidence * Math.min(1, deviation * 5),
            nodeIds: [src.marketId, tgt.marketId],
          });
        }
        break;
      }
      case "exclusive": {
        const sum = pA + pB;
        if (sum > 1.03) {
          violations.push({
            type: "exclusive",
            description: `Exclusive outcomes sum to ${sum.toFixed(3)}, expected <= 1.0`,
            severity: sum - 1,
            confidence: edge.confidence * Math.min(1, (sum - 1) * 5),
            nodeIds: [src.marketId, tgt.marketId],
          });
        }
        break;
      }
      case "equivalent": {
        const diff = Math.abs(pA - pB);
        if (diff > 0.05) {
          violations.push({
            type: "equivalence",
            description: `Equivalent markets differ by ${(diff * 100).toFixed(1)}%: ${pA.toFixed(3)} vs ${pB.toFixed(3)}`,
            severity: diff,
            confidence: edge.confidence * Math.min(1, diff * 5),
            nodeIds: [src.marketId, tgt.marketId],
          });
        }
        break;
      }
    }
  }

  detectCycleViolations(graph, violations);

  return violations;
}

function detectCycleViolations(graph: ProbabilityGraph, violations: ConstraintViolation[]): void {
  const adj = new Map<string, Array<{ target: string; edge: GraphEdge }>>();
  for (const e of graph.edges) {
    let list = adj.get(e.sourceId);
    if (!list) { list = []; adj.set(e.sourceId, list); }
    list.push({ target: e.targetId, edge: e });

    let list2 = adj.get(e.targetId);
    if (!list2) { list2 = []; adj.set(e.targetId, list2); }
    list2.push({ target: e.sourceId, edge: e });
  }

  const visited = new Set<string>();
  adj.forEach((neighbors, nodeA) => {
    if (visited.has(nodeA)) return;
    for (const { target: nodeB } of neighbors) {
      const neighborsB = adj.get(nodeB);
      if (!neighborsB) continue;
      for (const { target: nodeC } of neighborsB) {
        if (nodeC === nodeA) continue;
        const neighborsC = adj.get(nodeC);
        if (!neighborsC) continue;
        const closesLoop = neighborsC.some((n) => n.target === nodeA);
        if (!closesLoop) continue;

        const triKey = [nodeA, nodeB, nodeC].sort().join("|");
        if (visited.has(triKey)) continue;
        visited.add(triKey);

        const nA = graph.nodes.get(nodeA);
        const nB = graph.nodes.get(nodeB);
        const nC = graph.nodes.get(nodeC);
        if (!nA || !nB || !nC) continue;

        const sum = nA.observedProb + nB.observedProb + nC.observedProb;
        if (sum > 1.1 || sum < 0.5) {
          violations.push({
            type: "cycle",
            description: `3-node cycle sums to ${sum.toFixed(3)}: "${nA.question.slice(0, 30)}" + "${nB.question.slice(0, 30)}" + "${nC.question.slice(0, 30)}"`,
            severity: Math.abs(sum - 1),
            confidence: 0.4 * Math.min(1, Math.abs(sum - 1) * 3),
            nodeIds: [nodeA, nodeB, nodeC],
          });
        }
      }
    }
    visited.add(nodeA);
  });
}

export function computeImpliedProbabilityBounds(
  graph: ProbabilityGraph,
  marketId: string
): { lower: number; upper: number } | null {
  const node = graph.nodes.get(marketId);
  if (!node) return null;
  return { lower: node.impliedLowerBound, upper: node.impliedUpperBound };
}
