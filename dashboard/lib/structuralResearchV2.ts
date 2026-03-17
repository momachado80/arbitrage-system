/**
 * Structural Research V2 — motor hierárquico de descoberta de subsets menos destrutivos.
 * Proteção contra fragmentação combinatória, amostra insuficiente e falso sinal.
 * Não promove challenger; não altera execução ou thresholds.
 */

import type { ClosedTradeAuditEntry } from "./shadowClosedTradeAudit";

/** Gates de amostra — configuráveis e auditáveis */
export const SAMPLE_GATE_CONFIG = {
  /** Mínimo para aparecer em pesquisa */
  minToAppear: 5,
  /** Mínimo para ranking exploratório */
  minForExploratoryRanking: 10,
  /** Mínimo para subset elegível a validação prospectiva */
  minForValidation: 20,
} as const;

const EDGE_BUCKETS = ["0-0.5%", "0.5-1%", "1-2%", "2-5%", ">5%"] as const;
const FILL_BUCKETS = ["0-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"] as const;

function edgeBucket(edge: number): string {
  if (edge < 0.005) return "0-0.5%";
  if (edge < 0.01) return "0.5-1%";
  if (edge < 0.02) return "1-2%";
  if (edge < 0.05) return "2-5%";
  return ">5%";
}

function fillBucket(r: number | null | undefined): string {
  if (r == null || r < 0) return "_unknown";
  if (r <= 0.1) return "0-0.1";
  if (r <= 0.25) return "0.1-0.25";
  if (r <= 0.5) return "0.25-0.5";
  if (r <= 0.75) return "0.5-0.75";
  return "0.75-1.0";
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const m = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[m] : (sorted[m - 1] + sorted[m]) / 2;
}

export interface SubsetMetrics {
  tradeCount: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  winRate: number;
  avgFillRatio: number;
  avgCapturableEdgeAtEntry: number;
  avgObservedEdgeAtEntry: number;
  avgHoldingTimeMs: number;
}

export interface RankedSubset extends SubsetMetrics {
  /** pairKey | pairKey|edgeBucket | pairKey|fillBucket | pairKey|edge|fill */
  subsetKey: string;
  /** global | pair | pair_x_edge | pair_x_fill | pair_x_edge_x_fill */
  level: "global" | "pair" | "pair_x_edge" | "pair_x_fill" | "pair_x_edge_x_fill";
  pairKey?: string;
  edgeBucket?: string;
  fillBucket?: string;
  /** Score de ranking: maior = menos destrutivo. Shrinkage aplicado. */
  rankingScore: number;
  /** true se usou fallback para nível pai */
  usedParentFallback: boolean;
  /** Se usedParentFallback, qual nível pai */
  parentLevel?: string;
  /** Elegível a validação prospectiva? tradeCount >= minForValidation */
  eligibleForValidation: boolean;
}

export interface StructuralResearchV2 {
  methodologySummary: string;
  sampleGateConfig: typeof SAMPLE_GATE_CONFIG;
  /** Separação rígida: só fluxo novo */
  dataScope: "new_only";
  rehydratedExcludedCount: number;
  newTradesCount: number;
  hierarchySummary: {
    global: SubsetMetrics | null;
    pairsCount: number;
    pairEdgeCount: number;
    pairFillCount: number;
    pairEdgeFillCount: number;
  };
  topLeastBadPairs: RankedSubset[];
  topLeastBadPairEdgeBuckets: RankedSubset[];
  topLeastBadPairFillBuckets: RankedSubset[];
  topLeastBadPairEdgeFillCombos: RankedSubset[];
  subsetsRejectedForLowSample: string[];
  subsetsUsingParentFallback: string[];
  rankingScoreExplanation: string;
}

function computeMetrics(entries: ClosedTradeAuditEntry[]): SubsetMetrics {
  const n = entries.length;
  if (n === 0) {
    return {
      tradeCount: 0,
      avgRealizedPnL: 0,
      medianRealizedPnL: 0,
      winRate: 0,
      avgFillRatio: 0,
      avgCapturableEdgeAtEntry: 0,
      avgObservedEdgeAtEntry: 0,
      avgHoldingTimeMs: 0,
    };
  }
  const pnls = entries.map((e) => e.realizedPnL);
  const wins = entries.filter((e) => e.realizedPnL > 0).length;
  const fillRatios = entries
    .map((e) => e.fillRatio)
    .filter((v): v is number => v != null && typeof v === "number");
  const capturable = entries.map((e) => e.capturableEdgeAtEntry ?? 0);
  const observed = entries.map((e) => e.observedEdgeAtEntry ?? 0);
  const holding = entries.map((e) => e.holdingTimeMs ?? 0);

  return {
    tradeCount: n,
    avgRealizedPnL: pnls.reduce((s, x) => s + x, 0) / n,
    medianRealizedPnL: median(pnls),
    winRate: wins / n,
    avgFillRatio: fillRatios.length ? fillRatios.reduce((a, b) => a + b, 0) / fillRatios.length : 0,
    avgCapturableEdgeAtEntry: capturable.reduce((a, b) => a + b, 0) / n,
    avgObservedEdgeAtEntry: observed.reduce((a, b) => a + b, 0) / n,
    avgHoldingTimeMs: holding.reduce((a, b) => a + b, 0) / n,
  };
}

/**
 * Shrinkage score: menos destrutivo = maior score.
 * Combina avgPnL, medianPnL, winRate com peso por sample size.
 * Shrinkage: quando n < minForExploratoryRanking, interpola com parent.
 * Heurística explícita e auditável.
 */
function computeRankingScore(
  m: SubsetMetrics,
  parent: SubsetMetrics | null,
  gate: typeof SAMPLE_GATE_CONFIG
): { score: number; usedFallback: boolean; parentLevel?: string } {
  const { minToAppear, minForExploratoryRanking } = gate;

  if (m.tradeCount < minToAppear) {
    return { score: -Infinity, usedFallback: false };
  }

  const rawScore =
    0.4 * m.avgRealizedPnL +
    0.3 * (m.winRate - 0.5) * 2 + // winRate 0.5 -> 0, 1 -> 1
    0.2 * m.medianRealizedPnL +
    0.1 * Math.log1p(m.tradeCount); // bonus por sample size

  if (m.tradeCount >= minForExploratoryRanking || !parent) {
    return { score: rawScore, usedFallback: false };
  }

  const parentScore =
    0.4 * parent.avgRealizedPnL +
    0.3 * (parent.winRate - 0.5) * 2 +
    0.2 * parent.medianRealizedPnL +
    0.1 * Math.log1p(parent.tradeCount);

  const weight = m.tradeCount / minForExploratoryRanking;
  const shrunkScore = weight * rawScore + (1 - weight) * parentScore;
  return {
    score: shrunkScore,
    usedFallback: true,
    parentLevel: "parent",
  };
}

export function runStructuralResearchV2(
  entries: ClosedTradeAuditEntry[],
  rehydratedIds: ReadonlySet<string>
): StructuralResearchV2 {
  const newEntries = entries.filter((e) => !rehydratedIds.has(e.tradeId));
  const rehydratedExcluded = entries.length - newEntries.length;

  const gate = SAMPLE_GATE_CONFIG;
  const rejected: string[] = [];
  const usedFallback: string[] = [];

  const globalMetrics = computeMetrics(newEntries);

  const byPair = new Map<string, ClosedTradeAuditEntry[]>();
  const byPairEdge = new Map<string, ClosedTradeAuditEntry[]>();
  const byPairFill = new Map<string, ClosedTradeAuditEntry[]>();
  const byPairEdgeFill = new Map<string, ClosedTradeAuditEntry[]>();

  for (const e of newEntries) {
    const pk = e.pairKey ?? "_no_pair";
    const eb = edgeBucket(e.capturableEdgeAtEntry ?? 0);
    const fb = fillBucket(e.fillRatio);

    const push = (map: Map<string, ClosedTradeAuditEntry[]>, key: string) => {
      const arr = map.get(key) ?? [];
      arr.push(e);
      map.set(key, arr);
    };

    push(byPair, pk);
    push(byPairEdge, `${pk}|${eb}`);
    push(byPairFill, `${pk}|${fb}`);
    push(byPairEdgeFill, `${pk}|${eb}|${fb}`);
  }

  const rankSubsets = (
    map: Map<string, ClosedTradeAuditEntry[]>,
    level: RankedSubset["level"],
    parentMap: Map<string, ClosedTradeAuditEntry[]> | null,
    parentKeyFn: (cellKey: string) => string | null
  ): RankedSubset[] => {
    const results: RankedSubset[] = [];
    for (const [key, cellEntries] of Array.from(map.entries())) {
      const m = computeMetrics(cellEntries);
      if (m.tradeCount < gate.minToAppear) {
        rejected.push(`${level}:${key}`);
        continue;
      }

      let parent: SubsetMetrics | null = null;
      const parentKey = parentMap ? parentKeyFn(key) : null;
      if (parentKey && parentMap) {
        const parentEntries = parentMap.get(parentKey);
        if (parentEntries) parent = computeMetrics(parentEntries);
      } else if (level !== "global") {
        parent = globalMetrics;
      }

      const { score, usedFallback: uf, parentLevel } = computeRankingScore(m, parent, gate);
      if (uf) usedFallback.push(`${level}:${key}`);

      const [pairKey, edgeBucket, fillBucket] = key.split("|");
      results.push({
        ...m,
        subsetKey: key,
        level,
        pairKey: pairKey || undefined,
        edgeBucket: edgeBucket && edgeBucket !== pairKey ? edgeBucket : undefined,
        fillBucket: fillBucket && fillBucket !== pairKey && fillBucket !== edgeBucket ? fillBucket : undefined,
        rankingScore: score,
        usedParentFallback: uf,
        parentLevel,
        eligibleForValidation: m.tradeCount >= gate.minForValidation,
      });
    }
    return results;
  };

  const pairParentKey = (key: string) => key;
  const pairEdgeParentKey = (key: string) => {
    const i = key.lastIndexOf("|");
    return i > 0 ? key.slice(0, i) : null;
  };
  const pairFillParentKey = pairEdgeParentKey;
  const pairEdgeFillParentKey = (key: string) => {
    const parts = key.split("|");
    if (parts.length >= 3) return `${parts[0]}|${parts[1]}`;
    if (parts.length === 2) return parts[0];
    return null;
  };

  const globalMap = new Map<string, ClosedTradeAuditEntry[]>();
  globalMap.set("_global", newEntries);
  const pairRanked = rankSubsets(byPair, "pair", globalMap, () => "_global");
  const pairEdgeRanked = rankSubsets(byPairEdge, "pair_x_edge", byPair, (k) => k.split("|")[0] ?? null);
  const pairFillRanked = rankSubsets(byPairFill, "pair_x_fill", byPair, (k) => k.split("|")[0] ?? null);
  const pairEdgeFillParentKeyFn = (k: string): string | null => {
    const parts = k.split("|");
    if (parts.length < 3) return parts[0] ?? null;
    const pk = parts[0];
    const eb = parts[1];
    const fb = parts[2];
    const pairEdge = `${pk}|${eb}`;
    const pairFill = `${pk}|${fb}`;
    if (byPairEdge.has(pairEdge)) return pairEdge;
    if (byPairFill.has(pairFill)) return pairFill;
    return pk;
  };
  const pairEdgeFillParentMap = new Map<string, ClosedTradeAuditEntry[]>();
  for (const [k, v] of Array.from(byPairEdge.entries())) pairEdgeFillParentMap.set(k, v);
  for (const [k, v] of Array.from(byPairFill.entries())) pairEdgeFillParentMap.set(k, v);
  for (const [k, v] of Array.from(byPair.entries())) pairEdgeFillParentMap.set(k, v);
  const pairEdgeFillRanked = rankSubsets(
    byPairEdgeFill,
    "pair_x_edge_x_fill",
    pairEdgeFillParentMap,
    pairEdgeFillParentKeyFn
  );

  const sortByScore = (a: RankedSubset[]) => [...a].sort((x, y) => y.rankingScore - x.rankingScore);

  const topPairs = sortByScore(pairRanked).slice(0, 15);
  const topPairEdge = sortByScore(pairEdgeRanked).slice(0, 15);
  const topPairFill = sortByScore(pairFillRanked).slice(0, 15);
  const topPairEdgeFill = sortByScore(pairEdgeFillRanked).slice(0, 15);

  return {
    methodologySummary:
      "Hierarchical structural research: global -> pair -> pair×edge -> pair×fill -> pair×edge×fill. " +
      "Fallback to parent when sample < minForExploratoryRanking. " +
      "Ranking score = 0.4*avgPnL + 0.3*(winRate-0.5)*2 + 0.2*medianPnL + 0.1*log1p(n). " +
      "Shrinkage: when n < 10, interpolate with parent score. Only new trades; rehydrated excluded.",
    sampleGateConfig: gate,
    dataScope: "new_only",
    rehydratedExcludedCount: rehydratedExcluded,
    newTradesCount: newEntries.length,
    hierarchySummary: {
      global: newEntries.length >= gate.minToAppear ? globalMetrics : null,
      pairsCount: byPair.size,
      pairEdgeCount: byPairEdge.size,
      pairFillCount: byPairFill.size,
      pairEdgeFillCount: byPairEdgeFill.size,
    },
    topLeastBadPairs: topPairs,
    topLeastBadPairEdgeBuckets: topPairEdge,
    topLeastBadPairFillBuckets: topPairFill,
    topLeastBadPairEdgeFillCombos: topPairEdgeFill,
    subsetsRejectedForLowSample: rejected,
    subsetsUsingParentFallback: usedFallback,
    rankingScoreExplanation:
      "Higher score = less destructive. Weighted: avgPnL 40%, winRate 30%, medianPnL 20%, sample bonus 10%. " +
      "Shrinkage toward parent when tradeCount < minForExploratoryRanking. " +
      "Subsets with tradeCount < minToAppear are excluded.",
  };
}
