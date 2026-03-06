import type { NormalizedMarket } from "./polymarketClient";

export type RelationType =
  | "subset"
  | "exclusive"
  | "complementary"
  | "equivalent";

export interface MarketRelation {
  type: RelationType;
  sourceMarketId: string;
  targetMarketId: string;
  confidence: number;
}

export interface ConstraintCluster {
  clusterId: string;
  markets: NormalizedMarket[];
  relations: MarketRelation[];
}

const STOP_WORDS = new Set([
  "will", "the", "a", "an", "in", "on", "by", "of", "to", "be",
  "is", "at", "or", "and", "for", "with", "this", "that", "it",
  "from", "as", "are", "was", "has", "have", "do", "does", "did",
  "not", "no", "yes", "before", "after", "if", "than", "then",
]);

function tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2 && !STOP_WORDS.has(w))
  );
}

function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 0;
  let intersection = 0;
  a.forEach((w) => {
    if (b.has(w)) intersection++;
  });
  const union = a.size + b.size - intersection;
  return union > 0 ? intersection / union : 0;
}

function inferRelationType(
  a: NormalizedMarket,
  b: NormalizedMarket,
  textSim: number
): { type: RelationType; confidence: number } | null {
  const aTokens = tokenize(a.question);
  const bTokens = tokenize(b.question);

  const aIsSubsetOfB =
    aTokens.size > 0 &&
    Array.from(aTokens).every((t) => bTokens.has(t)) &&
    bTokens.size > aTokens.size;
  const bIsSubsetOfA =
    bTokens.size > 0 &&
    Array.from(bTokens).every((t) => aTokens.has(t)) &&
    aTokens.size > bTokens.size;

  if (textSim > 0.85 && a.outcomes.length === b.outcomes.length) {
    return { type: "equivalent", confidence: Math.min(0.95, textSim) };
  }

  if (aIsSubsetOfB || bIsSubsetOfA) {
    return { type: "subset", confidence: 0.7 };
  }

  if (
    a.category === b.category &&
    a.category !== "general" &&
    a.outcomes.length === 2 &&
    b.outcomes.length === 2 &&
    textSim > 0.3
  ) {
    return { type: "complementary", confidence: 0.5 + textSim * 0.3 };
  }

  return null;
}

export function buildClusters(markets: NormalizedMarket[]): ConstraintCluster[] {
  const t0 = Date.now();

  const categoryGroups = new Map<string, NormalizedMarket[]>();
  for (const m of markets) {
    const key = m.category.toLowerCase().trim() || "general";
    const arr = categoryGroups.get(key);
    if (arr) arr.push(m);
    else categoryGroups.set(key, [m]);
  }

  const tokenCache = new Map<string, Set<string>>();
  function getTokens(m: NormalizedMarket): Set<string> {
    let t = tokenCache.get(m.id);
    if (!t) {
      t = tokenize(m.question);
      tokenCache.set(m.id, t);
    }
    return t;
  }

  const clusters: ConstraintCluster[] = [];
  let totalRelations = 0;

  categoryGroups.forEach((group, category) => {
    if (group.length < 2 || group.length > 200) return;

    const relations: MarketRelation[] = [];
    const involved = new Set<string>();

    for (let i = 0; i < group.length && i < 100; i++) {
      for (let j = i + 1; j < group.length && j < 100; j++) {
        const a = group[i];
        const b = group[j];
        const sim = jaccardSimilarity(getTokens(a), getTokens(b));
        if (sim < 0.2) continue;

        const rel = inferRelationType(a, b, sim);
        if (!rel) continue;

        relations.push({
          type: rel.type,
          sourceMarketId: a.id,
          targetMarketId: b.id,
          confidence: rel.confidence,
        });
        involved.add(a.id);
        involved.add(b.id);
      }
    }

    if (relations.length > 0) {
      const clusterMarkets = group.filter((m) => involved.has(m.id));
      clusters.push({
        clusterId: `cluster-${category}-${clusters.length}`,
        markets: clusterMarkets,
        relations,
      });
      totalRelations += relations.length;
    }
  });

  if (markets.length >= 2) {
    const exclusiveGroups = new Map<string, NormalizedMarket[]>();
    for (const m of markets) {
      if (m.outcomes.length > 2 && m.liquidity >= 500) {
        const slug = m.slug.replace(/-[^-]+$/, "");
        const arr = exclusiveGroups.get(slug);
        if (arr) arr.push(m);
        else exclusiveGroups.set(slug, [m]);
      }
    }

    exclusiveGroups.forEach((group, slug) => {
      if (group.length < 2) return;
      const relations: MarketRelation[] = [];
      for (let i = 0; i < group.length; i++) {
        for (let j = i + 1; j < group.length; j++) {
          relations.push({
            type: "exclusive",
            sourceMarketId: group[i].id,
            targetMarketId: group[j].id,
            confidence: 0.6,
          });
        }
      }
      if (relations.length > 0) {
        clusters.push({
          clusterId: `cluster-excl-${slug}-${clusters.length}`,
          markets: group,
          relations,
        });
        totalRelations += relations.length;
      }
    });
  }

  const elapsed = Date.now() - t0;
  console.log(
    `[RelationBuilder] Built ${clusters.length} clusters with ${totalRelations} relations in ${elapsed}ms`
  );

  return clusters;
}
