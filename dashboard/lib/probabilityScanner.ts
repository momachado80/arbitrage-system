import type { NormalizedMarket } from "./polymarketClient";

export type EdgeType = "overround" | "underround" | "cross_market";

export interface DetectedEdge {
  marketId: string;
  question: string;
  slug: string;
  edge: number;
  type: EdgeType;
  probSum: number;
  outcomes: string[];
  prices: number[];
  liquidity: number;
  volume: number;
  confidence: number;
}

const OVERROUND_THRESHOLD = 1.02;
const UNDERROUND_THRESHOLD = 0.97;
const MIN_OUTCOMES = 2;
const MIN_LIQUIDITY = 100;

function computeConfidence(
  liquidity: number,
  spread: number,
  edge: number
): number {
  const liquidityFactor = Math.min(1, Math.log10(Math.max(1, liquidity)) / 5);
  const spreadPenalty = Math.min(1, Math.max(0, 1 - spread * 2));
  const edgeFactor = Math.min(1, Math.abs(edge) * 10);
  return Math.max(0, Math.min(1, liquidityFactor * spreadPenalty * edgeFactor));
}

function scanSingle(market: NormalizedMarket): DetectedEdge | null {
  if (market.outcomes.length < MIN_OUTCOMES) return null;
  if (market.liquidity < MIN_LIQUIDITY) return null;

  const sum = market.probSum;

  if (sum > OVERROUND_THRESHOLD) {
    const edge = sum - 1;
    return {
      marketId: market.id,
      question: market.question,
      slug: market.slug,
      edge,
      type: "overround",
      probSum: sum,
      outcomes: market.outcomes,
      prices: market.prices,
      liquidity: market.liquidity,
      volume: market.volume,
      confidence: computeConfidence(market.liquidity, market.spread, edge),
    };
  }

  if (sum < UNDERROUND_THRESHOLD) {
    const edge = 1 - sum;
    return {
      marketId: market.id,
      question: market.question,
      slug: market.slug,
      edge,
      type: "underround",
      probSum: sum,
      outcomes: market.outcomes,
      prices: market.prices,
      liquidity: market.liquidity,
      volume: market.volume,
      confidence: computeConfidence(market.liquidity, market.spread, edge),
    };
  }

  return null;
}

function scanCrossMarket(markets: NormalizedMarket[]): DetectedEdge[] {
  const groups = new Map<string, NormalizedMarket[]>();

  for (const m of markets) {
    const key = m.category.toLowerCase().trim();
    if (!key) continue;
    const arr = groups.get(key) || [];
    arr.push(m);
    groups.set(key, arr);
  }

  const edges: DetectedEdge[] = [];

  groups.forEach((group) => {
    if (group.length < 2) return;

    for (let i = 0; i < group.length; i++) {
      for (let j = i + 1; j < group.length; j++) {
        const a = group[i];
        const b = group[j];

        if (a.outcomes.length !== 2 || b.outcomes.length !== 2) continue;

        const pA = a.prices[0];
        const pB = b.prices[0];

        if (pA + pB > OVERROUND_THRESHOLD) {
          const edge = pA + pB - 1;
          const minLiq = Math.min(a.liquidity, b.liquidity);
          if (minLiq < MIN_LIQUIDITY) continue;

          edges.push({
            marketId: `${a.id}+${b.id}`,
            question: `${a.question} ↔ ${b.question}`,
            slug: `${a.slug}+${b.slug}`,
            edge,
            type: "cross_market",
            probSum: pA + pB,
            outcomes: [a.outcomes[0], b.outcomes[0]],
            prices: [pA, pB],
            liquidity: minLiq,
            volume: a.volume + b.volume,
            confidence: computeConfidence(
              minLiq,
              (a.spread + b.spread) / 2,
              edge
            ),
          });
        }
      }
    }
  });

  return edges;
}

export function scanMarkets(markets: NormalizedMarket[]): DetectedEdge[] {
  const t0 = Date.now();
  const edges: DetectedEdge[] = [];

  for (const m of markets) {
    const e = scanSingle(m);
    if (e) edges.push(e);
  }

  const crossEdges = scanCrossMarket(markets);
  edges.push(...crossEdges);

  const elapsed = Date.now() - t0;
  console.log(
    `[ProbabilityScanner] Scanned ${markets.length} markets → ${edges.length} edges in ${elapsed}ms`
  );

  return edges;
}
