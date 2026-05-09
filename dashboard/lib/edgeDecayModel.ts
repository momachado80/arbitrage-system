/**
 * Edge Decay Model — estimates how much observed edge survives to execution.
 * Fragile opportunities decay faster; persistence and trend affect decay rate.
 */

export interface PersistenceData {
  avgActiveDurationMs?: number;
  edgeTrend?: "rising" | "stable" | "falling";
  observationCount?: number;
  longestRecentEpisodeMs?: number;
}

export interface OpportunityForDecay {
  edge: number;
  confidence: number;
  spread: number;
  liquidity: number;
  opportunityType?: string;
  sourceType?: string;
}

export type EdgeDecayResult = DecayResult;

export interface DecayResult {
  observedEdge: number;
  decayedEdge: number;
  capturableEdge: number;
  decayRate: number;
  fragilityScore: number;
  reasoning: string;
}

const GRAPH_TYPES = [
  "graph_subset", "graph_complement", "graph_exclusive",
  "graph_equivalence",
  "graph_equivalence_micro",
  "graph_subset_micro",
  "graph_exclusive_micro",
  "graph_cycle",
];

function baseDecayRate(
  latencyMs: number,
  confidence: number,
  spread: number,
  liquidity: number,
  isGraph: boolean
): number {
  const latencyFactor = Math.min(1, latencyMs / 5000) * 0.3;
  const confFactor = 1 - Math.min(0.4, confidence * 0.5);
  const spreadFactor = Math.min(0.2, spread * 2);
  const liqFactor = liquidity < 500 ? 0.15 : liquidity < 2000 ? 0.08 : 0.03;
  const graphBonus = isGraph ? 0.05 : 0;
  return Math.min(0.6, latencyFactor + confFactor + spreadFactor + liqFactor + graphBonus);
}

export function estimateDecayRate(
  opportunity: OpportunityForDecay,
  latencyMs: number,
  persistenceData?: PersistenceData | null
): number {
  const isGraph = opportunity.opportunityType
    ? GRAPH_TYPES.includes(opportunity.opportunityType)
    : opportunity.sourceType === "graph";

  let rate = baseDecayRate(
    latencyMs,
    opportunity.confidence,
    opportunity.spread,
    opportunity.liquidity,
    isGraph
  );

  if (persistenceData) {
    const avgDur = persistenceData.avgActiveDurationMs ?? 0;
    const trend = persistenceData.edgeTrend ?? "stable";
    if (avgDur < 10000) rate += 0.1;
    if (avgDur < 5000) rate += 0.08;
    if (trend === "falling") rate += 0.12;
    if (trend === "rising") rate = Math.max(0, rate - 0.05);
  }

  return Math.min(0.85, rate);
}

export function estimateCapturableEdge(
  opportunity: OpportunityForDecay,
  latencyMs: number,
  persistenceData?: PersistenceData | null
): DecayResult {
  const observed = Math.max(0, opportunity.edge);
  const decayRate = estimateDecayRate(opportunity, latencyMs, persistenceData);
  const decayedEdge = observed * (1 - decayRate);
  const capturable = Math.min(observed, decayedEdge);

  const fragilityScore = Math.min(1, decayRate * 1.5);

  let reasoning = `Decay ${(decayRate * 100).toFixed(0)}% → capturable ${(capturable * 100).toFixed(2)}%`;
  if (persistenceData?.edgeTrend === "falling") reasoning += " (falling trend)";
  if (persistenceData?.avgActiveDurationMs && persistenceData.avgActiveDurationMs < 5000) {
    reasoning += " (short persistence)";
  }

  return {
    observedEdge: observed,
    decayedEdge,
    capturableEdge: capturable,
    decayRate,
    fragilityScore,
    reasoning,
  };
}
