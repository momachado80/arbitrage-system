/**
 * Auditoria de qualidade do desempenho por `graphDiagnosticProvenanceAtOpen` (O(n) sobre fechados).
 * Só observabilidade; não altera política nem pipeline.
 */

import {
  getClosedGraphTradesForProvenanceQualityAudit,
  isClosedTradeWithFiniteRealizedPnl,
} from "./paperClosedTradesMetrics";
import { getPaperPortfolio } from "./paperPortfolioStore";
import {
  PAPER_GRAPH_PROVENANCE_KEYS,
  effectiveGraphProvenanceForClosedAnalytics,
  type GraphProvenanceCountRecord,
} from "./graphOpportunityPaperImpact";
import { resolvePaperPolicyFromEnv } from "./paperTradeEngine";
import {
  DEFAULT_PAPER_FEE_BUFFER_PER_LEG,
  getClosedTradeEstimatedTotalFees,
  getClosedTradeGrossRealizedPnL,
  getClosedTradeNetRealizedPnL,
  getClosedTradeNetRealizedPnLOrZero,
  hasClosedPaperTradeFinitePnlSignal,
} from "./paperRealizedPnlSemantics";
import type { PaperGraphDiagnosticProvenance, PaperTrade } from "./paperTypes";

const TOP_WINNERS_PER_PROVENANCE = 3;
const TOP_LABELS_CLUSTERS = 8;
const TOP_MARKETS = 8;
const RELAXED_SAMPLE_CAP = 10;
/** Amostras no bloco de robustez estrutural (só métricas). */
const STRUCTURAL_TOP_CLUSTERS_CAP = 5;
const STRUCTURAL_TOP_LABELS_CAP = 5;
/** Máximo de fechos mais recentes na cauda temporal (vs resto). */
const STRUCTURAL_TRAILING_RECENT_CAP = 12;
/** Listas no audit intra-cluster (payload enxuto). */
const INTRA_TOKEN_TOP = 12;
const INTRA_SKELETON_TOP = 10;
const INTRA_TOPIC_TOP = 10;
const INTRA_SAMPLE_LABELS_CAP = 5;
const INTRA_TOP_SKELETONS_FOR_SAMPLES = 5;
const INTRA_TOP_TOPICS_FOR_SAMPLES = 5;

function emptyCountRecord(): GraphProvenanceCountRecord {
  const o = {} as GraphProvenanceCountRecord;
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) o[k] = 0;
  return o;
}

function emptyNestedExitCounts(): Record<PaperGraphDiagnosticProvenance, Record<string, number>> {
  const o = {} as Record<PaperGraphDiagnosticProvenance, Record<string, number>>;
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) o[k] = {};
  return o;
}

function emptyNestedExitPnL(): Record<
  PaperGraphDiagnosticProvenance,
  Record<string, { sum: number; n: number }>
> {
  const o = {} as Record<PaperGraphDiagnosticProvenance, Record<string, { sum: number; n: number }>>;
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) o[k] = {};
  return o;
}

function medianSorted(sorted: number[]): number | null {
  if (sorted.length === 0) return null;
  const m = [...sorted].sort((a, b) => a - b);
  const mid = Math.floor(m.length / 2);
  return m.length % 2 === 1 ? m[mid]! : (m[mid - 1]! + m[mid]!) / 2;
}

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function tradeLabel(t: PaperTrade): string {
  const q = t.marketsInvolved?.[0]?.question;
  if (q && String(q).length > 0) return String(q).slice(0, 100);
  const oid = t.opportunityId;
  if (oid != null && String(oid).length > 0) return String(oid).slice(0, 80);
  const tid = t.tradeId;
  return tid != null && String(tid).length > 0 ? String(tid).slice(0, 80) : "(no_label)";
}

function pnlConcentrationShares(pnls: number[]): {
  shareOfTop1: number | null;
  shareOfTop5: number | null;
  shareOfTop10: number | null;
} {
  if (pnls.length === 0) {
    return { shareOfTop1: null, shareOfTop5: null, shareOfTop10: null };
  }
  const sorted = [...pnls].sort((a, b) => b - a);
  const total = pnls.reduce((s, x) => s + x, 0);
  if (total === 0 || !Number.isFinite(total)) {
    return { shareOfTop1: null, shareOfTop5: null, shareOfTop10: null };
  }
  const sumK = (k: number) =>
    sorted.slice(0, Math.min(k, sorted.length)).reduce((s, x) => s + x, 0);
  return {
    shareOfTop1: round4(sumK(1) / total),
    shareOfTop5: round4(sumK(5) / total),
    shareOfTop10: round4(sumK(10) / total),
  };
}

type PerProvAcc = {
  /** PnL líquido por trade (0 se sem sinal finito). */
  pnls: number[];
  grossPnls: number[];
  holdingMs: number[];
  entryScores: number[];
  entryProg: number[];
  filledCaps: number[];
  netEdges: number[];
  trades: PaperTrade[];
};

function freshAcc(): PerProvAcc {
  return {
    pnls: [],
    grossPnls: [],
    holdingMs: [],
    entryScores: [],
    entryProg: [],
    filledCaps: [],
    netEdges: [],
    trades: [],
  };
}

export type PnlConcentrationRow = {
  shareOfTop1: number | null;
  shareOfTop5: number | null;
  shareOfTop10: number | null;
};

export type GraphProvenanceQualityAudit = {
  note: string;
  computedAt: string;
  closedGraphTradesAnalyzed: number;
  distribution: {
    closedTradesCountByProvenance: GraphProvenanceCountRecord;
    /** Soma PnL líquido (taxas estimadas descontadas); mesmo valor que `totalNetPnLByProvenance`. */
    totalPnLByProvenance: GraphProvenanceCountRecord;
    totalNetPnLByProvenance: GraphProvenanceCountRecord;
    totalGrossPnLByProvenance: GraphProvenanceCountRecord;
    closedWithNetNegativeByProvenance: GraphProvenanceCountRecord;
    countGrossPositiveNetNegativeByProvenance: GraphProvenanceCountRecord;
    avgPnLPerClosedTradeByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
    medianPnLPerClosedTradeByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
    topWinningTradesByProvenance: Record<
      PaperGraphDiagnosticProvenance,
      Array<{ tradeId: string; realizedPnL: number; label: string }>
    >;
    pnlConcentrationByProvenance: Record<PaperGraphDiagnosticProvenance, PnlConcentrationRow>;
  };
  complementaryRelaxedConcentration: {
    topOpportunityLabelsByClosedCount: Array<{ label: string; count: number }>;
    topOpportunityLabelsByPnL: Array<{ label: string; totalPnL: number }>;
    topClusterIdsByClosedCount: Array<{ clusterId: string; count: number }>;
    topClusterIdsByPnL: Array<{ clusterId: string; totalPnL: number }>;
    topMarketLabelsByPnLContribution: Array<{
      marketId: string;
      label: string;
      totalPnL: number;
      closedCount: number;
    }>;
  };
  holdingAndExit: {
    avgHoldingTimeMsByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
    medianHoldingTimeMsByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
    countByExitReasonByProvenance: Record<PaperGraphDiagnosticProvenance, Record<string, number>>;
    avgPnLByExitReasonByProvenance: Record<PaperGraphDiagnosticProvenance, Record<string, number | null>>;
  };
  entryQuality: {
    avgEntryEconomicScoreByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
    avgProgressProbabilityFactorByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
    avgFilledCapitalByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
    avgNetEdgeAtEntryByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
  };
};

/** Impacto das taxas estimadas (paper) por proveniência, só sobre fechados graph com sinal PnL finito. */
export type FeeImpactAudit = {
  note: string;
  computedAt: string;
  feeBufferPerLegUsed: number;
  tradeCountByProvenance: GraphProvenanceCountRecord;
  avgFeeBurdenByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
  /** Soma taxas / soma |gross|; proxy de “peso” das fees vs magnitude do PnL bruto. */
  feeAsPctOfGrossPnLByProvenance: Record<PaperGraphDiagnosticProvenance, number | null>;
  countGrossPositiveNetNegativeByProvenance: GraphProvenanceCountRecord;
};

export function buildFeeImpactAudit(
  closedGraph: PaperTrade[],
  feeBufferPerLeg: number
): FeeImpactAudit {
  const buf =
    typeof feeBufferPerLeg === "number" && Number.isFinite(feeBufferPerLeg) && feeBufferPerLeg >= 0
      ? feeBufferPerLeg
      : DEFAULT_PAPER_FEE_BUFFER_PER_LEG;
  const computedAt = new Date().toISOString();
  const byProv = new Map<PaperGraphDiagnosticProvenance, PaperTrade[]>();
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) byProv.set(k, []);
  for (const t of closedGraph) {
    if (!hasClosedPaperTradeFinitePnlSignal(t)) continue;
    const p = effectiveGraphProvenanceForClosedAnalytics(t) ?? "unknown";
    if (!byProv.has(p)) continue;
    byProv.get(p)!.push(t);
  }
  const tradeCountByProvenance = emptyCountRecord();
  const avgFeeBurdenByProvenance = {} as Record<PaperGraphDiagnosticProvenance, number | null>;
  const feeAsPctOfGrossPnLByProvenance = {} as Record<PaperGraphDiagnosticProvenance, number | null>;
  const countGrossPositiveNetNegativeByProvenance = emptyCountRecord();
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) {
    const arr = byProv.get(k) ?? [];
    tradeCountByProvenance[k] = arr.length;
    if (arr.length === 0) {
      avgFeeBurdenByProvenance[k] = null;
      feeAsPctOfGrossPnLByProvenance[k] = null;
      countGrossPositiveNetNegativeByProvenance[k] = 0;
      continue;
    }
    let feeSum = 0;
    let absGrossSum = 0;
    let flip = 0;
    for (const t of arr) {
      const fees = getClosedTradeEstimatedTotalFees(t, buf);
      const g = getClosedTradeGrossRealizedPnL(t);
      const n = getClosedTradeNetRealizedPnL(t, buf);
      feeSum += fees;
      absGrossSum += Math.abs(g);
      if (g > 0 && n <= 0) flip += 1;
    }
    avgFeeBurdenByProvenance[k] = round4(feeSum / arr.length);
    feeAsPctOfGrossPnLByProvenance[k] =
      absGrossSum > 1e-12 ? round4(feeSum / absGrossSum) : null;
    countGrossPositiveNetNegativeByProvenance[k] = flip;
  }
  return {
    note:
      "Taxas = persistidas ou filledCapital × feeBufferPerLeg × 2 (ida+volta). Gross = simulateExit.; net = gross − fees. Métricas por proveniência efectiva ao fecho.",
    computedAt,
    feeBufferPerLegUsed: buf,
    tradeCountByProvenance,
    avgFeeBurdenByProvenance,
    feeAsPctOfGrossPnLByProvenance,
    countGrossPositiveNetNegativeByProvenance,
  };
}

export type ComplementaryRelaxedQualitySample = {
  tradeId: string;
  realizedPnL: number;
  exitCondition: string;
  holdingTimeMs: number;
  entryEconomicScoreAtOpen: number | null;
  progressProbabilityFactorAtOpen: number | null;
  label: string;
  opportunityId: string;
  clusterId: string | null;
};

export type ComplementaryRelaxedQualityAudit = {
  closedTradesCount: number;
  totalPnL: number;
  avgPnL: number | null;
  pnlConcentration: PnlConcentrationRow;
  topLabels: Array<{ label: string; count: number; totalPnL: number }>;
  topClusters: Array<{ clusterId: string; count: number; totalPnL: number }>;
  countByExitReason: Record<string, number>;
  avgPnLByExitReason: Record<string, number | null>;
  avgHoldingTimeMs: number | null;
  medianHoldingTimeMs: number | null;
  sampleClosedTrades: ComplementaryRelaxedQualitySample[];
};

export type ComplementaryRelaxedStructuralRobustness = {
  note: string;
  computedAt: string;
  totalClosed: number;
  totalPnL: number;
  clusterDiversity: {
    uniqueClustersCount: number;
    shareOfTop1ClusterByClosedCount: number | null;
    shareOfTop1ClusterByPnL: number | null;
    shareOfTop3ClustersByPnL: number | null;
    /** Entropia de Shannon (bits) sobre a distribuição de contagens por cluster. */
    clusterEntropyBitsApprox: number | null;
    /** H / log2(K), K = clusters distintos; 0 se K<=1. */
    clusterEntropyNormalizedApprox: number | null;
  };
  familyTemplateDiversity: {
    uniqueOpportunityLabelsCount: number;
    uniqueMarketIdsCount: number;
    /** Proxy: prefixo de `clusterId` antes do último sufixo numérico (ex. cluster-general-0 → cluster-general). */
    uniqueUnderlyingFamiliesCount: number;
    shareOfTop1LabelByPnL: number | null;
    shareOfTop5LabelsByPnL: number | null;
  };
  timeStability: {
    /** Metade mais antiga vs mais recente por `closedAt` (ordenado cronológico). */
    olderHalfByTime: { closedCount: number; totalPnL: number; shareOfTotalPnL: number | null };
    recentHalfByTime: { closedCount: number; totalPnL: number; shareOfTotalPnL: number | null };
    /** Últimos N fechos (mais recentes); N = min(cap, max(1, ⌈n/4⌉)). */
    trailingRecentWindow: {
      windowSize: number;
      closedCount: number;
      totalPnL: number;
      shareOfTotalPnL: number | null;
    };
  };
  samples: {
    topClustersDetailed: Array<{
      clusterId: string;
      closedCount: number;
      totalPnL: number;
      shareOfClosedCount: number | null;
      shareOfTotalPnL: number | null;
    }>;
    topLabelsDetailed: Array<{
      label: string;
      closedCount: number;
      totalPnL: number;
      shareOfClosedCount: number | null;
      shareOfTotalPnL: number | null;
    }>;
  };
};

export type ComplementaryRelaxedIntraClusterAudit = {
  note: string;
  computedAt: string;
  scope: "complementary_relaxed_closed";
  totalClosed: number;
  totalPnL: number;
  /** Métricas só sobre trades com clusterId matching cluster-general-* (resto pode estar vazio). */
  clusterGeneralSlice: {
    closedCount: number;
    uniqueTemplateSkeletonCount: number;
    shareOfTop1TemplateSkeletonByPnL: number | null;
    shareOfTop5TemplateSkeletonsByPnL: number | null;
  };
  tokenTheme: {
    uniqueInformativeTokenCount: number;
    topInformativeTokensByClosedCount: Array<{ token: string; closedCount: number; totalPnL: number }>;
    topInformativeTokensByPnL: Array<{ token: string; closedCount: number; totalPnL: number }>;
    /** Parte do PnL total atribuída (split igual por token distinto no trade) aos 5 / 10 tokens com maior PnL acumulado. */
    tokenConcentrationShareTop5: number | null;
    tokenConcentrationShareTop10: number | null;
  };
  templateSkeleton: {
    uniqueTemplateSkeletonCount: number;
    topTemplateSkeletonsByClosedCount: Array<{ skeleton: string; closedCount: number; totalPnL: number }>;
    topTemplateSkeletonsByPnL: Array<{ skeleton: string; closedCount: number; totalPnL: number }>;
    shareOfTop1TemplateSkeletonByPnL: number | null;
    shareOfTop5TemplateSkeletonsByPnL: number | null;
  };
  topicBuckets: {
    uniqueTopicBucketCount: number;
    topTopicBucketsByClosedCount: Array<{ topicBucket: string; closedCount: number; totalPnL: number }>;
    topTopicBucketsByPnL: Array<{ topicBucket: string; closedCount: number; totalPnL: number }>;
  };
  samples: {
    sampleLabelsByTopTemplateSkeleton: Array<{ skeleton: string; sampleLabels: string[] }>;
    sampleLabelsByTopTopicBucket: Array<{ topicBucket: string; sampleLabels: string[] }>;
  };
};

const INTRA_STOPWORDS = new Set([
  "the",
  "and",
  "for",
  "are",
  "but",
  "not",
  "you",
  "all",
  "can",
  "was",
  "one",
  "our",
  "out",
  "day",
  "get",
  "has",
  "him",
  "his",
  "how",
  "its",
  "may",
  "new",
  "now",
  "old",
  "see",
  "two",
  "who",
  "way",
  "she",
  "use",
  "any",
  "did",
  "let",
  "put",
  "say",
  "each",
  "which",
  "their",
  "will",
  "from",
  "that",
  "this",
  "with",
  "have",
  "been",
  "were",
  "what",
  "when",
  "than",
  "then",
  "them",
  "some",
  "into",
  "more",
  "such",
  "only",
  "other",
  "time",
  "very",
  "upon",
  "about",
  "after",
  "before",
  "between",
  "through",
  "during",
  "being",
  "over",
  "would",
  "could",
  "should",
  "might",
  "must",
  "does",
  "done",
  "here",
  "there",
  "where",
  "while",
  "because",
  "until",
  "unless",
  "whether",
  "within",
  "without",
  "also",
  "just",
  "even",
  "both",
  "few",
  "most",
  "same",
  "well",
  "back",
  "much",
  "yes",
  "per",
  "pct",
]);

/** Primeira regra que casa define o bucket (heurística leve). */
const INTRA_TOPIC_RULES: Array<{ bucket: string; test: (s: string) => boolean }> = [
  {
    bucket: "sports_football_soccer",
    test: (s) =>
      /\bworld cup\b/.test(s) ||
      /\bfifa\b/.test(s) ||
      /\bepl\b/.test(s) ||
      /\buefa\b/.test(s) ||
      /\bmls\b/.test(s) ||
      /\bpremier league\b/.test(s) ||
      /\bchampions league\b/.test(s) ||
      /\bsuper bowl\b/.test(s) ||
      /\bnfl\b/.test(s) ||
      /\bnba\b/.test(s) ||
      /\bncaa\b/.test(s),
  },
  {
    bucket: "us_election_politics",
    test: (s) =>
      /\bpresident(ial)?\b/.test(s) ||
      /\belection\b/.test(s) ||
      /\bnomination\b/.test(s) ||
      /\bdemocrat(ic)?\b/.test(s) ||
      /\brepublican\b/.test(s) ||
      /\bsenate\b/.test(s) ||
      /\bcongress\b/.test(s) ||
      /\bgovernor\b/.test(s) ||
      /\btrump\b/.test(s) ||
      /\bbiden\b/.test(s),
  },
  {
    bucket: "crypto_blockchain",
    test: (s) =>
      /\bbitcoin\b/.test(s) ||
      /\bbtc\b/.test(s) ||
      /\bethereum\b/.test(s) ||
      /\beth\b/.test(s) ||
      /\bcrypto(currency)?\b/.test(s) ||
      /\bdefi\b/.test(s),
  },
  {
    bucket: "ipo_equities_macro",
    test: (s) =>
      /\bipo\b/.test(s) ||
      /\bspac\b/.test(s) ||
      /\bnasdaq\b/.test(s) ||
      /\bs&p\b/.test(s) ||
      /\bearnings\b/.test(s) ||
      /\bfed\b/.test(s) ||
      /\binterest rate\b/.test(s),
  },
  {
    bucket: "tech_ai_product",
    test: (s) =>
      /\bopenai\b/.test(s) ||
      /\bchatgpt\b/.test(s) ||
      /\bgoogle\b/.test(s) ||
      /\bapple\b/.test(s) ||
      /\bmeta\b/.test(s) ||
      /\btesla\b/.test(s) ||
      /\bai\b/.test(s),
  },
];

function normalizeLabelForTokenize(raw: string): string {
  return raw
    .toLowerCase()
    .replace(/https?:\/\/\S+/gi, " ")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function extractInformativeTokens(normalized: string): string[] {
  const parts = normalized.split(" ").filter(Boolean);
  const seen = new Set<string>();
  const out: string[] = [];
  for (const w of parts) {
    if (w.length < 3) continue;
    if (/^\d+$/.test(w)) continue;
    if (INTRA_STOPWORDS.has(w)) continue;
    if (seen.has(w)) continue;
    seen.add(w);
    out.push(w);
  }
  return out;
}

function templateSkeletonFromLabel(raw: string): string {
  let s = raw.toLowerCase();
  s = s.replace(/https?:\/\/\S+/gi, " ");
  s = s.replace(/\$[\d,]+\.?\d*\b/g, " #usd ");
  s = s.replace(/\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/g, " #date ");
  s = s.replace(/\b\d{4}\b/g, " #year ");
  s = s.replace(/\b\d+\.?\d*\b/g, " #num ");
  s = s.replace(/[^a-z0-9#]+/g, " ");
  s = s.replace(/\s+/g, " ").trim();
  s = s.replace(/(?:#num\s*)+/g, "#num ");
  s = s.replace(/(?:#year\s*)+/g, "#year ");
  s = s.replace(/(?:#usd\s*)+/g, "#usd ");
  s = s.replace(/(?:#date\s*)+/g, "#date ");
  return s.length > 0 ? s.slice(0, 100) : "_empty_";
}

function topicBucketForLabel(raw: string): string {
  const s = raw.toLowerCase();
  for (const r of INTRA_TOPIC_RULES) {
    if (r.test(s)) return r.bucket;
  }
  return "unbucketed_other";
}

function mapToSortedRows(
  m: Map<string, { count: number; pnl: number }>,
  sort: "count" | "pnl",
  limit: number
): Array<{ closedCount: number; totalPnL: number; key: string }> {
  const rows = Array.from(m.entries()).map(([key, v]) => ({
    key,
    closedCount: v.count,
    totalPnL: round4(v.pnl),
  }));
  rows.sort((a, b) => (sort === "count" ? b.closedCount - a.closedCount : b.totalPnL - a.totalPnL));
  return rows.slice(0, limit);
}

function shareTopKByPnL(
  sortedByPnlDesc: Array<{ totalPnL: number }>,
  k: number,
  totalPnl: number
): number | null {
  if (!Number.isFinite(totalPnl) || totalPnl === 0 || sortedByPnlDesc.length === 0) return null;
  const sum = sortedByPnlDesc.slice(0, k).reduce((s, r) => s + r.totalPnL, 0);
  return round4(sum / totalPnl);
}

function computeComplementaryRelaxedIntraClusterAudit(args: {
  relaxedTrades: PaperTrade[];
  totalPnL: number;
  computedAt: string;
  feeBufferPerLeg: number;
}): ComplementaryRelaxedIntraClusterAudit {
  const trades = args.relaxedTrades;
  const n = trades.length;
  const totalPnL = round4(args.totalPnL);

  const tokenMap = new Map<string, { count: number; pnl: number }>();
  const skeletonMap = new Map<string, { count: number; pnl: number }>();
  const topicMap = new Map<string, { count: number; pnl: number }>();
  const skeletonToLabels = new Map<string, Set<string>>();
  const topicToLabels = new Map<string, Set<string>>();

  const generalTrades = trades.filter((t) => {
    const cid = t.clusterId != null ? String(t.clusterId) : "";
    return /^cluster-general-/i.test(cid);
  });
  const skeletonGeneral = new Map<string, { count: number; pnl: number }>();

  for (const t of trades) {
    const label = tradeLabel(t);
    const pnl = getClosedTradeNetRealizedPnLOrZero(t, args.feeBufferPerLeg);
    const norm = normalizeLabelForTokenize(label);
    const tokens = extractInformativeTokens(norm);
    const sk = templateSkeletonFromLabel(label);
    const topic = topicBucketForLabel(label);

    bumpMap(skeletonMap, sk, pnl);
    bumpMap(topicMap, topic, pnl);

    const pushSample = (m: Map<string, Set<string>>, key: string, lab: string) => {
      let set = m.get(key);
      if (!set) {
        set = new Set<string>();
        m.set(key, set);
      }
      if (set.size < INTRA_SAMPLE_LABELS_CAP) set.add(lab.slice(0, 100));
    };
    pushSample(skeletonToLabels, sk, label);
    pushSample(topicToLabels, topic, label);

    const w = tokens.length > 0 ? pnl / tokens.length : 0;
    for (const tok of tokens) {
      bumpMap(tokenMap, tok, w);
    }

    const cid = t.clusterId != null ? String(t.clusterId) : "";
    if (/^cluster-general-/i.test(cid)) {
      bumpMap(skeletonGeneral, sk, pnl);
    }
  }

  const tokensByCount = mapToSortedRows(tokenMap, "count", INTRA_TOKEN_TOP);
  const tokensByPnl = mapToSortedRows(tokenMap, "pnl", INTRA_TOKEN_TOP);
  const allTokensByPnl = mapToSortedRows(tokenMap, "pnl", tokenMap.size);

  const skByCountFull = mapToSortedRows(skeletonMap, "count", skeletonMap.size);
  const skByPnlFull = mapToSortedRows(skeletonMap, "pnl", skeletonMap.size);
  const skByCount = skByCountFull.slice(0, INTRA_SKELETON_TOP).map((r) => ({
    skeleton: r.key,
    closedCount: r.closedCount,
    totalPnL: r.totalPnL,
  }));
  const skByPnl = skByPnlFull.slice(0, INTRA_SKELETON_TOP).map((r) => ({
    skeleton: r.key,
    closedCount: r.closedCount,
    totalPnL: r.totalPnL,
  }));

  const topicByCount = mapToSortedRows(topicMap, "count", INTRA_TOPIC_TOP).map((r) => ({
    topicBucket: r.key,
    closedCount: r.closedCount,
    totalPnL: r.totalPnL,
  }));
  const topicByPnl = mapToSortedRows(topicMap, "pnl", INTRA_TOPIC_TOP).map((r) => ({
    topicBucket: r.key,
    closedCount: r.closedCount,
    totalPnL: r.totalPnL,
  }));

  const skGeneralByPnlFull = mapToSortedRows(skeletonGeneral, "pnl", skeletonGeneral.size);

  const topicByPnlFull = mapToSortedRows(topicMap, "pnl", topicMap.size);

  const sampleLabelsByTopTemplateSkeleton = skByPnlFull
    .slice(0, INTRA_TOP_SKELETONS_FOR_SAMPLES)
    .map((r) => ({
      skeleton: r.key,
      sampleLabels: Array.from(skeletonToLabels.get(r.key) ?? []).slice(0, INTRA_SAMPLE_LABELS_CAP),
    }));

  const sampleLabelsByTopTopicBucket = topicByPnlFull
    .slice(0, INTRA_TOP_TOPICS_FOR_SAMPLES)
    .map((r) => ({
      topicBucket: r.key,
      sampleLabels: Array.from(topicToLabels.get(r.key) ?? []).slice(0, INTRA_SAMPLE_LABELS_CAP),
    }));

  return {
    note:
      "Audit intra-cluster complementary_relaxed: tokens informativos (stopwords removidas, PnL repartido por token distinto no trade), skeleton com números/datas colapsados, buckets por regex. clusterGeneralSlice = só cluster-general-*. Sem política nem I/O.",
    computedAt: args.computedAt,
    scope: "complementary_relaxed_closed",
    totalClosed: n,
    totalPnL,
    clusterGeneralSlice: {
      closedCount: generalTrades.length,
      uniqueTemplateSkeletonCount: skeletonGeneral.size,
      shareOfTop1TemplateSkeletonByPnL: shareTopKByPnL(skGeneralByPnlFull, 1, args.totalPnL),
      shareOfTop5TemplateSkeletonsByPnL: shareTopKByPnL(skGeneralByPnlFull, 5, args.totalPnL),
    },
    tokenTheme: {
      uniqueInformativeTokenCount: tokenMap.size,
      topInformativeTokensByClosedCount: tokensByCount.map((r) => ({
        token: r.key,
        closedCount: r.closedCount,
        totalPnL: r.totalPnL,
      })),
      topInformativeTokensByPnL: tokensByPnl.map((r) => ({
        token: r.key,
        closedCount: r.closedCount,
        totalPnL: r.totalPnL,
      })),
      tokenConcentrationShareTop5: shareTopKByPnL(allTokensByPnl, 5, args.totalPnL),
      tokenConcentrationShareTop10: shareTopKByPnL(allTokensByPnl, 10, args.totalPnL),
    },
    templateSkeleton: {
      uniqueTemplateSkeletonCount: skeletonMap.size,
      topTemplateSkeletonsByClosedCount: skByCount,
      topTemplateSkeletonsByPnL: skByPnl,
      shareOfTop1TemplateSkeletonByPnL: shareTopKByPnL(skByPnlFull, 1, args.totalPnL),
      shareOfTop5TemplateSkeletonsByPnL: shareTopKByPnL(skByPnlFull, 5, args.totalPnL),
    },
    topicBuckets: {
      uniqueTopicBucketCount: topicMap.size,
      topTopicBucketsByClosedCount: topicByCount,
      topTopicBucketsByPnL: topicByPnl,
    },
    samples: {
      sampleLabelsByTopTemplateSkeleton,
      sampleLabelsByTopTopicBucket,
    },
  };
}

function bumpMap(m: Map<string, { count: number; pnl: number }>, key: string, pnl: number): void {
  const e = m.get(key) ?? { count: 0, pnl: 0 };
  e.count += 1;
  e.pnl += pnl;
  m.set(key, e);
}

function topFromMap(
  m: Map<string, { count: number; pnl: number }>,
  sort: "count" | "pnl",
  limit: number
): Array<{ label?: string; clusterId?: string; marketId?: string; count: number; totalPnL: number }> {
  const rows = Array.from(m.entries()).map(([k, v]) => ({
    key: k,
    count: v.count,
    totalPnL: round4(v.pnl),
  }));
  rows.sort((a, b) => (sort === "count" ? b.count - a.count : b.totalPnL - a.totalPnL));
  return rows.slice(0, limit).map((r) => ({
    label: r.key,
    clusterId: r.key,
    marketId: r.key,
    count: r.count,
    totalPnL: r.totalPnL,
  }));
}

/** cluster-general-0 → cluster-general; sem sufixo `-digits` final → id completo. */
function coarseClusterFamilyFromId(raw: string): string {
  if (!raw || raw === "_none_") return "_none_";
  const idx = raw.lastIndexOf("-");
  if (idx <= 0) return raw;
  const tail = raw.slice(idx + 1);
  if (/^\d+$/.test(tail)) return raw.slice(0, idx);
  return raw;
}

function shannonEntropyBitsFromCounts(counts: number[], n: number): number | null {
  if (n <= 0) return null;
  let h = 0;
  for (const c of counts) {
    if (c <= 0) continue;
    const p = c / n;
    h -= p * Math.log2(p);
  }
  return round4(h);
}

function shareOfTotal(partial: number, total: number): number | null {
  if (!Number.isFinite(total) || total === 0) return null;
  return round4(partial / total);
}

function computeComplementaryRelaxedStructuralRobustness(args: {
  relaxedTrades: PaperTrade[];
  relaxedCluster: Map<string, { count: number; pnl: number }>;
  relaxedLabel: Map<string, { count: number; pnl: number }>;
  relaxedMarket: Map<string, { count: number; pnl: number; label: string }>;
  totalPnL: number;
  computedAt: string;
  feeBufferPerLeg: number;
}): ComplementaryRelaxedStructuralRobustness {
  const rn = args.relaxedTrades.length;
  const totalPnL = round4(args.totalPnL);

  const clusterEntries = Array.from(args.relaxedCluster.entries()).map(([key, v]) => ({
    key,
    count: v.count,
    pnl: v.pnl,
  }));
  const kClusters = clusterEntries.length;
  const maxCount = clusterEntries.length > 0 ? Math.max(...clusterEntries.map((c) => c.count)) : 0;
  const byPnlDesc = [...clusterEntries].sort((a, b) => b.pnl - a.pnl);
  const top1Pnl = byPnlDesc[0]?.pnl ?? 0;
  const top3PnlSum = byPnlDesc
    .slice(0, 3)
    .reduce((s, x) => s + x.pnl, 0);

  const countsForEntropy = clusterEntries.map((c) => c.count);
  const hBits = rn > 0 ? shannonEntropyBitsFromCounts(countsForEntropy, rn) : null;
  const kForNorm = kClusters;
  const maxH = kForNorm > 1 ? Math.log2(kForNorm) : 0;
  const hNorm =
    rn <= 0 || kForNorm === 0
      ? null
      : kForNorm === 1
        ? 0
        : hBits != null && maxH > 0
          ? round4(hBits / maxH)
          : null;

  const labelEntries = Array.from(args.relaxedLabel.entries()).map(([label, v]) => ({
    label,
    count: v.count,
    pnl: v.pnl,
  }));
  const labelsByPnl = [...labelEntries].sort((a, b) => b.pnl - a.pnl);
  const top1LabelPnl = labelsByPnl[0]?.pnl ?? 0;
  const top5LabelsPnl = labelsByPnl.slice(0, 5).reduce((s, x) => s + x.pnl, 0);

  const familySet = new Set<string>();
  for (const t of args.relaxedTrades) {
    const cid =
      t.clusterId != null && String(t.clusterId).length > 0 ? String(t.clusterId) : "_none_";
    familySet.add(coarseClusterFamilyFromId(cid === "_none_" ? "_none_" : cid));
  }

  const sortedByClose = [...args.relaxedTrades].sort(
    (a, b) => new Date(a.closedAt ?? 0).getTime() - new Date(b.closedAt ?? 0).getTime()
  );
  const half = Math.floor(rn / 2);
  const olderHalfTrades = sortedByClose.slice(0, half);
  const recentHalfTrades = sortedByClose.slice(half);
  const pnlOlderHalf = olderHalfTrades.reduce(
    (s, t) => s + getClosedTradeNetRealizedPnLOrZero(t, args.feeBufferPerLeg),
    0
  );
  const pnlRecentHalf = recentHalfTrades.reduce(
    (s, t) => s + getClosedTradeNetRealizedPnLOrZero(t, args.feeBufferPerLeg),
    0
  );

  const tailW =
    rn <= 0
      ? 0
      : Math.min(STRUCTURAL_TRAILING_RECENT_CAP, Math.max(1, Math.ceil(rn / 4)));
  const trailingTrades =
    tailW > 0 ? sortedByClose.slice(Math.max(0, rn - tailW)) : [];
  const pnlTrailing = trailingTrades.reduce(
    (s, t) => s + getClosedTradeNetRealizedPnLOrZero(t, args.feeBufferPerLeg),
    0
  );

  const topClustersDetailed = byPnlDesc.slice(0, STRUCTURAL_TOP_CLUSTERS_CAP).map((c) => ({
    clusterId: c.key === "_none_" ? "(none)" : c.key,
    closedCount: c.count,
    totalPnL: round4(c.pnl),
    shareOfClosedCount: rn > 0 ? round4(c.count / rn) : null,
    shareOfTotalPnL: shareOfTotal(c.pnl, args.totalPnL),
  }));

  const topLabelsDetailed = labelsByPnl.slice(0, STRUCTURAL_TOP_LABELS_CAP).map((c) => ({
    label: c.label.length > 120 ? c.label.slice(0, 120) : c.label,
    closedCount: c.count,
    totalPnL: round4(c.pnl),
    shareOfClosedCount: rn > 0 ? round4(c.count / rn) : null,
    shareOfTotalPnL: shareOfTotal(c.pnl, args.totalPnL),
  }));

  return {
    note:
      "Só observabilidade complementary_relaxed: diversidade cluster/label/mercado, proxy de família (prefixo cluster antes de sufixo numérico), estabilidade temporal (metade antiga vs recente + cauda N). Não altera política.",
    computedAt: args.computedAt,
    totalClosed: rn,
    totalPnL,
    clusterDiversity: {
      uniqueClustersCount: kClusters,
      shareOfTop1ClusterByClosedCount: rn > 0 ? round4(maxCount / rn) : null,
      shareOfTop1ClusterByPnL: shareOfTotal(top1Pnl, args.totalPnL),
      shareOfTop3ClustersByPnL: shareOfTotal(top3PnlSum, args.totalPnL),
      clusterEntropyBitsApprox: hBits,
      clusterEntropyNormalizedApprox: hNorm,
    },
    familyTemplateDiversity: {
      uniqueOpportunityLabelsCount: args.relaxedLabel.size,
      uniqueMarketIdsCount: args.relaxedMarket.size,
      uniqueUnderlyingFamiliesCount: familySet.size,
      shareOfTop1LabelByPnL: shareOfTotal(top1LabelPnl, args.totalPnL),
      shareOfTop5LabelsByPnL: shareOfTotal(top5LabelsPnl, args.totalPnL),
    },
    timeStability: {
      olderHalfByTime: {
        closedCount: olderHalfTrades.length,
        totalPnL: round4(pnlOlderHalf),
        shareOfTotalPnL: shareOfTotal(pnlOlderHalf, args.totalPnL),
      },
      recentHalfByTime: {
        closedCount: recentHalfTrades.length,
        totalPnL: round4(pnlRecentHalf),
        shareOfTotalPnL: shareOfTotal(pnlRecentHalf, args.totalPnL),
      },
      trailingRecentWindow: {
        windowSize: tailW,
        closedCount: trailingTrades.length,
        totalPnL: round4(pnlTrailing),
        shareOfTotalPnL: shareOfTotal(pnlTrailing, args.totalPnL),
      },
    },
    samples: {
      topClustersDetailed,
      topLabelsDetailed,
    },
  };
}

export type GraphQualityAuditSourceDiagnostics = {
  note: string;
  rawClosedTradesSeen: number;
  closedTradesAfterStatusClosed: number;
  closedTradesAfterGraphSourceType: number;
  closedTradesAfterFinitePnlFilter: number;
  /** Igual `closedGraphTradesWithProvenanceCount` em propagação (string não vazia). */
  closedTradesAfterProvenanceString: number;
  /** Universo efectivo de `graphProvenanceQualityAudit` (graph + fechado). */
  closedTradesForQualityAuditUniverse: number;
  complementaryRelaxedClosedTradesSeen: number;
  /** Quantos graph fechados tinham PnL não finito (excluídos pelo predicado antigo). */
  tradesExcludedByFinitePnlOnly: number;
  sampleRejectedReasons: Array<{ tradeId: string; reason: string }>;
};

export function buildGraphQualityAuditSourceDiagnostics(): GraphQualityAuditSourceDiagnostics {
  const raw = getPaperPortfolio().closedTrades;
  const statusOk = raw.filter((t) => t.status === "closed");
  const graphClosed = statusOk.filter((t) => t.sourceType === "graph");
  const finiteOnGraph = graphClosed.filter(isClosedTradeWithFiniteRealizedPnl);
  const withProvStr = graphClosed.filter(
    (t) =>
      t.graphDiagnosticProvenanceAtOpen != null &&
      typeof t.graphDiagnosticProvenanceAtOpen === "string" &&
      t.graphDiagnosticProvenanceAtOpen.length > 0
  );
  let complementaryRelaxedClosedTradesSeen = 0;
  for (const t of graphClosed) {
    if (effectiveGraphProvenanceForClosedAnalytics(t) === "complementary_relaxed") {
      complementaryRelaxedClosedTradesSeen += 1;
    }
  }
  const excluded = graphClosed.filter((t) => !isClosedTradeWithFiniteRealizedPnl(t));
  const sampleRejectedReasons = excluded.slice(0, 12).map((t) => {
    const pnl = t.realizedPnL;
    let reason = "non_finite_realized_pnl";
    if (pnl === undefined || pnl === null) reason = "missing_realized_pnl";
    else if (typeof pnl !== "number") reason = "realized_pnl_not_number";
    else if (!Number.isFinite(pnl)) reason = "realized_pnl_nan_or_inf";
    return { tradeId: t.tradeId, reason };
  });
  return {
    note:
      "Contagens sobre getPaperPortfolio().closedTrades. A auditoria de qualidade usa o mesmo universo base que propagação (status=closed, sourceType=graph). O filtro antigo getClosedTradesWithFiniteRealizedPnl excluía trades com PnL ausente/NaN e zerava closedGraphTradesAnalyzed.",
    rawClosedTradesSeen: raw.length,
    closedTradesAfterStatusClosed: statusOk.length,
    closedTradesAfterGraphSourceType: graphClosed.length,
    closedTradesAfterFinitePnlFilter: finiteOnGraph.length,
    closedTradesAfterProvenanceString: withProvStr.length,
    closedTradesForQualityAuditUniverse: graphClosed.length,
    complementaryRelaxedClosedTradesSeen,
    tradesExcludedByFinitePnlOnly: excluded.length,
    sampleRejectedReasons,
  };
}

export function buildGraphProvenanceQualityBundle(): {
  graphProvenanceQualityAudit: GraphProvenanceQualityAudit;
  complementaryRelaxedQualityAudit: ComplementaryRelaxedQualityAudit;
  complementaryRelaxedStructuralRobustness: ComplementaryRelaxedStructuralRobustness;
  complementaryRelaxedIntraClusterAudit: ComplementaryRelaxedIntraClusterAudit;
  graphQualityAuditSourceDiagnostics: GraphQualityAuditSourceDiagnostics;
  feeImpactAudit: FeeImpactAudit;
} {
  const computedAt = new Date().toISOString();
  const closed = getClosedGraphTradesForProvenanceQualityAudit();
  const feeBuf = resolvePaperPolicyFromEnv().feeBuffer;

  const byProv = new Map<PaperGraphDiagnosticProvenance, PerProvAcc>();
  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) byProv.set(k, freshAcc());

  const exitCountByProv = emptyNestedExitCounts();
  const exitPnLByProv = emptyNestedExitPnL();

  const relaxedLabel = new Map<string, { count: number; pnl: number }>();
  const relaxedCluster = new Map<string, { count: number; pnl: number }>();
  const relaxedMarket = new Map<string, { count: number; pnl: number; label: string }>();

  const relaxedTrades: PaperTrade[] = [];

  for (const t of closed) {
    const prov = effectiveGraphProvenanceForClosedAnalytics(t) ?? "unknown";
    const acc = byProv.get(prov)!;
    const pnl = getClosedTradeNetRealizedPnLOrZero(t, feeBuf);
    const gross = hasClosedPaperTradeFinitePnlSignal(t) ? getClosedTradeGrossRealizedPnL(t) : 0;
    acc.pnls.push(pnl);
    acc.grossPnls.push(gross);
    acc.trades.push(t);
    const hold = typeof t.holdingTimeMs === "number" && Number.isFinite(t.holdingTimeMs) ? t.holdingTimeMs : 0;
    acc.holdingMs.push(hold);
    if (typeof t.entryEconomicScoreAtOpen === "number" && Number.isFinite(t.entryEconomicScoreAtOpen)) {
      acc.entryScores.push(t.entryEconomicScoreAtOpen);
    }
    if (
      typeof t.progressProbabilityFactorAtOpen === "number" &&
      Number.isFinite(t.progressProbabilityFactorAtOpen)
    ) {
      acc.entryProg.push(t.progressProbabilityFactorAtOpen);
    }
    if (typeof t.filledCapital === "number" && Number.isFinite(t.filledCapital)) {
      acc.filledCaps.push(t.filledCapital);
    }
    if (typeof t.netEdgeAtEntry === "number" && Number.isFinite(t.netEdgeAtEntry)) {
      acc.netEdges.push(t.netEdgeAtEntry);
    }

    const ex = (t.exitCondition ?? "unknown") as string;
    exitCountByProv[prov][ex] = (exitCountByProv[prov][ex] ?? 0) + 1;
    const ep = exitPnLByProv[prov][ex] ?? { sum: 0, n: 0 };
    ep.sum += pnl;
    ep.n += 1;
    exitPnLByProv[prov][ex] = ep;

    if (prov === "complementary_relaxed") {
      relaxedTrades.push(t);
      const lab = tradeLabel(t);
      bumpMap(relaxedLabel, lab, pnl);
      const cid = t.clusterId != null && String(t.clusterId).length > 0 ? String(t.clusterId) : "_none_";
      bumpMap(relaxedCluster, cid, pnl);
      const mid = t.marketsInvolved?.[0]?.marketId ?? "_none_";
      const ml = t.marketsInvolved?.[0]?.question
        ? String(t.marketsInvolved[0].question).slice(0, 80)
        : mid;
      const mk = relaxedMarket.get(mid) ?? { count: 0, pnl: 0, label: ml };
      mk.count += 1;
      mk.pnl += pnl;
      if (ml.length > mk.label.length) mk.label = ml;
      relaxedMarket.set(mid, mk);
    }
  }

  const nullAvgs = (): Record<PaperGraphDiagnosticProvenance, number | null> => {
    const o = {} as Record<PaperGraphDiagnosticProvenance, number | null>;
    for (const k of PAPER_GRAPH_PROVENANCE_KEYS) o[k] = null;
    return o;
  };

  const closedCount = emptyCountRecord();
  const totalPnl = emptyCountRecord();
  const totalGrossPnl = emptyCountRecord();
  const closedNetNeg = emptyCountRecord();
  const grossPosNetNonPos = emptyCountRecord();
  const avgPnl = nullAvgs();
  const medPnl = nullAvgs();
  const avgHold = nullAvgs();
  const medHold = nullAvgs();
  const avgScore = nullAvgs();
  const avgProg = nullAvgs();
  const avgFill = nullAvgs();
  const avgNet = nullAvgs();
  const topWinners = {} as GraphProvenanceQualityAudit["distribution"]["topWinningTradesByProvenance"];
  const pnlConc = {} as Record<PaperGraphDiagnosticProvenance, PnlConcentrationRow>;
  const avgExitPnL = {} as Record<PaperGraphDiagnosticProvenance, Record<string, number | null>>;

  for (const k of PAPER_GRAPH_PROVENANCE_KEYS) {
    const acc = byProv.get(k)!;
    const n = acc.pnls.length;
    closedCount[k] = n;
    const sumP = acc.pnls.reduce((s, x) => s + x, 0);
    totalPnl[k] = round4(sumP);
    const sumGross = acc.grossPnls.reduce((s, x) => s + x, 0);
    totalGrossPnl[k] = round4(sumGross);
    let nn = 0;
    let flip = 0;
    for (let i = 0; i < acc.pnls.length; i++) {
      const net = acc.pnls[i]!;
      const g = acc.grossPnls[i]!;
      if (net < 0) nn += 1;
      if (g > 0 && net <= 0) flip += 1;
    }
    closedNetNeg[k] = nn;
    grossPosNetNonPos[k] = flip;
    avgPnl[k] = n > 0 ? round4(sumP / n) : null;
    const medP = medianSorted(acc.pnls);
    medPnl[k] = medP != null ? round4(medP) : null;
    const sumH = acc.holdingMs.reduce((s, x) => s + x, 0);
    avgHold[k] = n > 0 ? round4(sumH / n) : null;
    const medH = medianSorted(acc.holdingMs);
    medHold[k] = medH != null ? round4(medH) : null;
    avgScore[k] =
      acc.entryScores.length > 0
        ? round4(acc.entryScores.reduce((s, x) => s + x, 0) / acc.entryScores.length)
        : null;
    avgProg[k] =
      acc.entryProg.length > 0
        ? round4(acc.entryProg.reduce((s, x) => s + x, 0) / acc.entryProg.length)
        : null;
    avgFill[k] =
      acc.filledCaps.length > 0
        ? round4(acc.filledCaps.reduce((s, x) => s + x, 0) / acc.filledCaps.length)
        : null;
    avgNet[k] =
      acc.netEdges.length > 0
        ? round4(acc.netEdges.reduce((s, x) => s + x, 0) / acc.netEdges.length)
        : null;

    const sortedTrades = [...acc.trades].sort(
      (a, b) =>
        getClosedTradeNetRealizedPnLOrZero(b, feeBuf) - getClosedTradeNetRealizedPnLOrZero(a, feeBuf)
    );
    topWinners[k] = sortedTrades.slice(0, TOP_WINNERS_PER_PROVENANCE).map((t) => ({
      tradeId: t.tradeId,
      realizedPnL: round4(getClosedTradeNetRealizedPnLOrZero(t, feeBuf)),
      label: tradeLabel(t),
    }));
    pnlConc[k] = pnlConcentrationShares(acc.pnls);

    const exitAvg: Record<string, number | null> = {};
    for (const [reason, agg] of Object.entries(exitPnLByProv[k])) {
      exitAvg[reason] = agg.n > 0 ? round4(agg.sum / agg.n) : null;
    }
    avgExitPnL[k] = exitAvg;
  }

  const labelCountRows = Array.from(relaxedLabel.entries())
    .map(([label, v]) => ({ label, count: v.count, totalPnL: round4(v.pnl) }))
    .sort((a, b) => b.count - a.count)
    .slice(0, TOP_LABELS_CLUSTERS);

  const labelPnlRows = Array.from(relaxedLabel.entries())
    .map(([label, v]) => ({ label, count: v.count, totalPnL: round4(v.pnl) }))
    .sort((a, b) => b.totalPnL - a.totalPnL)
    .slice(0, TOP_LABELS_CLUSTERS);

  const clusterCountTop = topFromMap(relaxedCluster, "count", TOP_LABELS_CLUSTERS).map((r) => ({
    clusterId: r.clusterId === "_none_" ? "(none)" : r.clusterId!,
    count: r.count,
    totalPnL: r.totalPnL,
  }));

  const clusterPnlTop = topFromMap(relaxedCluster, "pnl", TOP_LABELS_CLUSTERS).map((r) => ({
    clusterId: r.clusterId === "_none_" ? "(none)" : r.clusterId!,
    count: r.count,
    totalPnL: r.totalPnL,
  }));

  const marketTop = Array.from(relaxedMarket.entries())
    .map(([marketId, v]) => ({
      marketId: marketId === "_none_" ? "(none)" : marketId,
      label: v.label.slice(0, 80),
      totalPnL: round4(v.pnl),
      closedCount: v.count,
    }))
    .sort((a, b) => b.totalPnL - a.totalPnL)
    .slice(0, TOP_MARKETS);

  const rn = relaxedTrades.length;
  const relaxedPnls = relaxedTrades.map((t) => getClosedTradeNetRealizedPnLOrZero(t, feeBuf));
  const relaxedSum = relaxedPnls.reduce((s, x) => s + x, 0);
  const relaxedExitCount: Record<string, number> = {};
  const relaxedExitPnL: Record<string, { sum: number; n: number }> = {};
  for (const t of relaxedTrades) {
    const ex = (t.exitCondition ?? "unknown") as string;
    relaxedExitCount[ex] = (relaxedExitCount[ex] ?? 0) + 1;
    const pnl = getClosedTradeNetRealizedPnLOrZero(t, feeBuf);
    const z = relaxedExitPnL[ex] ?? { sum: 0, n: 0 };
    z.sum += pnl;
    z.n += 1;
    relaxedExitPnL[ex] = z;
  }
  const relaxedAvgExit: Record<string, number | null> = {};
  for (const [ex, z] of Object.entries(relaxedExitPnL)) {
    relaxedAvgExit[ex] = z.n > 0 ? round4(z.sum / z.n) : null;
  }

  const relaxedHolds = relaxedTrades.map((t) => t.holdingTimeMs ?? 0);
  const samples = [...relaxedTrades]
    .sort((a, b) => new Date(b.closedAt ?? 0).getTime() - new Date(a.closedAt ?? 0).getTime())
    .slice(0, RELAXED_SAMPLE_CAP)
    .map(
      (t): ComplementaryRelaxedQualitySample => ({
        tradeId: t.tradeId,
        realizedPnL: round4(getClosedTradeNetRealizedPnLOrZero(t, feeBuf)),
        exitCondition: t.exitCondition ?? "unknown",
        holdingTimeMs: round4(t.holdingTimeMs ?? 0),
        entryEconomicScoreAtOpen:
          typeof t.entryEconomicScoreAtOpen === "number" && Number.isFinite(t.entryEconomicScoreAtOpen)
            ? round4(t.entryEconomicScoreAtOpen)
            : null,
        progressProbabilityFactorAtOpen:
          typeof t.progressProbabilityFactorAtOpen === "number" &&
          Number.isFinite(t.progressProbabilityFactorAtOpen)
            ? round4(t.progressProbabilityFactorAtOpen)
            : null,
        label: tradeLabel(t),
        opportunityId: t.opportunityId,
        clusterId: t.clusterId != null ? String(t.clusterId) : null,
      })
    );

  const complementaryRelaxedQualityAudit: ComplementaryRelaxedQualityAudit = {
    closedTradesCount: rn,
    totalPnL: round4(relaxedSum),
    avgPnL: rn > 0 ? round4(relaxedSum / rn) : null,
    pnlConcentration: pnlConcentrationShares(relaxedPnls),
    topLabels: labelPnlRows.map((r) => ({ label: r.label, count: r.count, totalPnL: r.totalPnL })),
    topClusters: clusterPnlTop.map((r) => ({
      clusterId: r.clusterId,
      count: r.count,
      totalPnL: r.totalPnL,
    })),
    countByExitReason: relaxedExitCount,
    avgPnLByExitReason: relaxedAvgExit,
    avgHoldingTimeMs: rn > 0 ? round4(relaxedHolds.reduce((s, x) => s + x, 0) / rn) : null,
    medianHoldingTimeMs: (() => {
      const mh = medianSorted(relaxedHolds);
      return mh != null ? round4(mh) : null;
    })(),
    sampleClosedTrades: samples,
  };

  const complementaryRelaxedStructuralRobustness = computeComplementaryRelaxedStructuralRobustness({
    relaxedTrades,
    relaxedCluster,
    relaxedLabel,
    relaxedMarket,
    totalPnL: relaxedSum,
    computedAt,
    feeBufferPerLeg: feeBuf,
  });

  const complementaryRelaxedIntraClusterAudit = computeComplementaryRelaxedIntraClusterAudit({
    relaxedTrades,
    totalPnL: relaxedSum,
    computedAt,
    feeBufferPerLeg: feeBuf,
  });

  const feeImpactAudit = buildFeeImpactAudit(closed, feeBuf);
  const totalNetPnLByProvenance = { ...totalPnl };

  const graphProvenanceQualityAudit: GraphProvenanceQualityAudit = {
    note:
      "Agregado O(n) sobre fechados graph no store (status=closed, sourceType=graph). PnL em agregados = líquido (gross simulateExit − taxas estimadas feeBuffer×2×filled salvo campos persistidos); sem sinal PnL finito → 0. Proveniência = effectiveGraphProvenanceForClosedAnalytics.",
    computedAt,
    closedGraphTradesAnalyzed: closed.length,
    distribution: {
      closedTradesCountByProvenance: closedCount,
      totalPnLByProvenance: totalPnl,
      totalNetPnLByProvenance,
      totalGrossPnLByProvenance: totalGrossPnl,
      closedWithNetNegativeByProvenance: closedNetNeg,
      countGrossPositiveNetNegativeByProvenance: grossPosNetNonPos,
      avgPnLPerClosedTradeByProvenance: avgPnl,
      medianPnLPerClosedTradeByProvenance: medPnl,
      topWinningTradesByProvenance: topWinners,
      pnlConcentrationByProvenance: pnlConc,
    },
    complementaryRelaxedConcentration: {
      topOpportunityLabelsByClosedCount: labelCountRows.map((r) => ({ label: r.label, count: r.count })),
      topOpportunityLabelsByPnL: labelPnlRows.map((r) => ({ label: r.label, totalPnL: r.totalPnL })),
      topClusterIdsByClosedCount: clusterCountTop.map((r) => ({ clusterId: r.clusterId, count: r.count })),
      topClusterIdsByPnL: clusterPnlTop.map((r) => ({ clusterId: r.clusterId, totalPnL: r.totalPnL })),
      topMarketLabelsByPnLContribution: marketTop,
    },
    holdingAndExit: {
      avgHoldingTimeMsByProvenance: avgHold,
      medianHoldingTimeMsByProvenance: medHold,
      countByExitReasonByProvenance: exitCountByProv,
      avgPnLByExitReasonByProvenance: avgExitPnL,
    },
    entryQuality: {
      avgEntryEconomicScoreByProvenance: avgScore,
      avgProgressProbabilityFactorByProvenance: avgProg,
      avgFilledCapitalByProvenance: avgFill,
      avgNetEdgeAtEntryByProvenance: avgNet,
    },
  };

  return {
    graphProvenanceQualityAudit,
    complementaryRelaxedQualityAudit,
    complementaryRelaxedStructuralRobustness,
    complementaryRelaxedIntraClusterAudit,
    graphQualityAuditSourceDiagnostics: buildGraphQualityAuditSourceDiagnostics(),
    feeImpactAudit,
  };
}
