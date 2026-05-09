import type { NormalizedMarket } from "./polymarketClient";
import {
  bumpTokenSimilarityBucket,
  countConnectedComponentsFromMarketIds,
  emptyClusterFormationFunnel,
  emptyPostPrefilterPartitionAccumulator,
  emptyRelationBuilderFunnel,
  emptyRelationsByInferencePathBreakdown,
  bumpComplementaryRelaxedRejectJaccardBucket,
  bumpComplementaryRelaxedRejectPrefixBucket,
  bumpComplementaryRelaxedRejectSharedTokenBucket,
  finalizePostPrefilterPartitionDiagnostics,
  MAX_COMPLEMENTARY_RELAXED_DIAGNOSTIC_SAMPLES,
  MAX_COMPLEMENTARY_RELAXED_POTENTIAL_RESCUE_SAMPLES,
  MAX_COMPLEMENTARY_RELAXED_REJECTED_DIAGNOSTIC_SAMPLES,
  MAX_COMPLEMENTARY_RELAXED_SHARED_PREFIX_RESCUE_SAMPLES,
  MAX_INFER_TYPE_DIAGNOSTIC_SAMPLES,
  type ClusterFormationFunnelSnapshot,
  type GraphSourceQualitySnapshot,
  type PostPrefilterPartitionAccumulator,
  type RelationBuilderFunnelSnapshot,
} from "./graphPipelineDiagnostics";

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
  /** Só `complementary`: ramo strict (categoria alinhada) vs relaxed (cross-category / general). */
  complementaryInferencePath?: "strict" | "relaxed";
}

export interface ConstraintCluster {
  clusterId: string;
  markets: NormalizedMarket[];
  relations: MarketRelation[];
}

/** Política estável do builder (diagnóstico; não altera comportamento). */
export const RELATION_BUILDER_POLICY_SNAPSHOT: Record<string, number | string | boolean> = {
  similarityThreshold: 0.2,
  equivalentTextSimilarityMin: 0.85,
  maxMarketsPerCategoryForPairing: 100,
  minCategoryGroupSize: 2,
  maxCategoryGroupSize: 200,
  exclusiveMinOutcomes: 3,
  exclusiveMinLiquidity: 500,
  /** `NormalizedMarket` não expõe datas de resolução; reserva para futuro. */
  resolutionTimeFieldPresent: false,
  /**
   * Particionamento de grupos de categoria > maxCategoryGroupSize:
   * ordenar por prefixo de texto normalizado (localeCompare) e cortar em blocos contíguos
   * de até `maxMarketsPerSubgroup` mercados, no máximo `maxSubgroupsPerLargeCategory` blocos.
   * Custo O(n log n) + O(n); sem O(n²) sobre o grupo inteiro.
   */
  largeCategoryPartitioningHeuristic: "sort_by_normalized_question_prefix_then_contiguous_chunks",
};

type PartitioningPolicy = {
  partitioningEnabled: boolean;
  maxCategoryGroupSize: number;
  maxSubgroupsPerLargeCategory: number;
  maxMarketsPerSubgroup: number;
  maxPairChecksPerLargeCategory: number;
  maxMarketsPerCategoryForPairing: number;
};

function readPartitioningPolicy(): PartitioningPolicy {
  const env = (k: string, d: string) => process.env[k]?.trim() ?? d;
  const envNum = (k: string, def: number) => {
    const n = Number(env(k, String(def)));
    return Number.isFinite(n) ? n : def;
  };
  const envBool = (k: string, def: boolean) => {
    const v = env(k, def ? "1" : "0").toLowerCase();
    if (v === "1" || v === "true" || v === "yes") return true;
    if (v === "0" || v === "false" || v === "no") return false;
    return def;
  };
  return {
    partitioningEnabled: envBool("RELATION_BUILDER_LARGE_CATEGORY_PARTITIONING_ENABLED", true),
    maxCategoryGroupSize: Math.max(2, Math.floor(envNum("RELATION_BUILDER_MAX_CATEGORY_GROUP_SIZE", 200))),
    maxSubgroupsPerLargeCategory: Math.max(1, Math.floor(envNum("RELATION_BUILDER_MAX_SUBGROUPS_PER_LARGE_CATEGORY", 12))),
    maxMarketsPerSubgroup: Math.max(8, Math.floor(envNum("RELATION_BUILDER_MAX_MARKETS_PER_SUBGROUP", 80))),
    maxPairChecksPerLargeCategory: Math.max(100, Math.floor(envNum("RELATION_BUILDER_MAX_PAIR_CHECKS_PER_LARGE_CATEGORY", 8000))),
    maxMarketsPerCategoryForPairing: Math.max(8, Math.floor(envNum("RELATION_BUILDER_MAX_MARKETS_PER_CATEGORY_FOR_PAIRING", 100))),
  };
}

function partitioningPolicySnapshot(p: PartitioningPolicy): Record<string, number | string | boolean> {
  return {
    partitioningEnabled: p.partitioningEnabled,
    maxCategoryGroupSize: p.maxCategoryGroupSize,
    maxSubgroupsPerLargeCategory: p.maxSubgroupsPerLargeCategory,
    maxMarketsPerSubgroup: p.maxMarketsPerSubgroup,
    maxPairChecksPerLargeCategory: p.maxPairChecksPerLargeCategory,
    maxMarketsPerCategoryForPairing: p.maxMarketsPerCategoryForPairing,
    heuristic: RELATION_BUILDER_POLICY_SNAPSHOT.largeCategoryPartitioningHeuristic as string,
  };
}

/** Pré-filtro lexical só no caminho particionado: barato (intersecção de tokens já usada no check completo). */
type LexicalPrefilterPolicy = {
  enabled: boolean;
  /** Mínimo de tokens informativos partilhados (após stopwords). */
  minSharedInformativeTokens: number;
  /** Se > 0, exige Jaccard sobre tokens ≥ este valor antes de gastar orçamento. */
  minTokenJaccard: number;
};

function readLexicalPrefilterPolicy(): LexicalPrefilterPolicy {
  const env = (k: string, d: string) => process.env[k]?.trim() ?? d;
  const envNum = (k: string, def: number) => {
    const n = Number(env(k, String(def)));
    return Number.isFinite(n) ? n : def;
  };
  const envBool = (k: string, def: boolean) => {
    const v = env(k, def ? "1" : "0").toLowerCase();
    if (v === "1" || v === "true" || v === "yes") return true;
    if (v === "0" || v === "false" || v === "no") return false;
    return def;
  };
  return {
    enabled: envBool("RELATION_BUILDER_LEXICAL_PREFILTER_ENABLED", true),
    minSharedInformativeTokens: Math.max(1, Math.floor(envNum("RELATION_BUILDER_LEXICAL_PREFILTER_MIN_SHARED_TOKENS", 2))),
    minTokenJaccard: Math.max(0, envNum("RELATION_BUILDER_LEXICAL_PREFILTER_MIN_TOKEN_JACCARD", 0)),
  };
}

function lexicalPrefilterPolicySnapshot(p: LexicalPrefilterPolicy): Record<string, number | string | boolean> {
  return {
    lexicalPrefilterEnabled: p.enabled,
    minSharedInformativeTokens: p.minSharedInformativeTokens,
    minTokenJaccard: p.minTokenJaccard,
    heuristic:
      "informative_token_intersection_then_optional_min_jaccard; budget_only_after_pass",
  };
}

/** Ramo extra de complementary quando o strict falha só por categoria/general; desligado por defeito (env). */
export type ComplementaryRelaxedPolicy = {
  enabled: boolean;
  /** Baseline de env `RELATION_BUILDER_COMPLEMENTARY_RELAXED_MIN_TOKEN_JACCARD` (default 0.52). */
  minTokenJaccard: number;
  /** Limiar efectivo para aceitação: override opcional ou `minTokenJaccard`. */
  effectiveMinTokenJaccard: number;
  /** `RELATION_BUILDER_COMPLEMENTARY_RELAXED_OVERRIDE_TOKEN_JACCARD` ou null se não definido. */
  overrideTokenJaccard: number | null;
  minSharedInformativeTokens: number;
  /** Baseline de env `RELATION_BUILDER_COMPLEMENTARY_RELAXED_MIN_COMMON_PREFIX_LENGTH` (default 12). */
  minCommonPrefixLength: number;
  /** Limiar efectivo de prefixo na aceitação relaxed: override opcional ou `minCommonPrefixLength`. */
  effectiveMinCommonPrefixLength: number;
  /** `RELATION_BUILDER_COMPLEMENTARY_RELAXED_OVERRIDE_COMMON_PREFIX_LENGTH` ou null. */
  overrideCommonPrefixLength: number | null;
  /** Limiares só para diagnóstico de sensibilidade (não alteram aceitação excepto comparação counterfactual). */
  sensitivityJaccard048: number;
  sensitivityJaccard044: number;
  sensitivityJaccard040: number;
  /** Contrafactuais de análise (não alteram aceitação). */
  sensitivitySharedThreshold4: number;
  sensitivitySharedThreshold3: number;
  sensitivityPrefixThreshold10: number;
  sensitivityPrefixThreshold8: number;
};

const COMPLEMENTARY_RELAXED_DISABLED: ComplementaryRelaxedPolicy = {
  enabled: false,
  minTokenJaccard: 1,
  effectiveMinTokenJaccard: 1,
  overrideTokenJaccard: null,
  minSharedInformativeTokens: 999,
  minCommonPrefixLength: 999,
  effectiveMinCommonPrefixLength: 999,
  overrideCommonPrefixLength: null,
  sensitivityJaccard048: 0.48,
  sensitivityJaccard044: 0.44,
  sensitivityJaccard040: 0.4,
  sensitivitySharedThreshold4: 4,
  sensitivitySharedThreshold3: 3,
  sensitivityPrefixThreshold10: 10,
  sensitivityPrefixThreshold8: 8,
};

function readComplementaryRelaxedPolicy(): ComplementaryRelaxedPolicy {
  const env = (k: string, d: string) => process.env[k]?.trim() ?? d;
  const envNum = (k: string, def: number) => {
    const n = Number(env(k, String(def)));
    return Number.isFinite(n) ? n : def;
  };
  const envBool = (k: string, def: boolean) => {
    const v = env(k, def ? "1" : "0").toLowerCase();
    if (v === "1" || v === "true" || v === "yes") return true;
    if (v === "0" || v === "false" || v === "no") return false;
    return def;
  };
  const minTokenJaccard = Math.max(0.2, envNum("RELATION_BUILDER_COMPLEMENTARY_RELAXED_MIN_TOKEN_JACCARD", 0.52));
  const overrideRaw = env("RELATION_BUILDER_COMPLEMENTARY_RELAXED_OVERRIDE_TOKEN_JACCARD", "");
  const overrideParsed = overrideRaw === "" ? NaN : Number(overrideRaw);
  const overrideTokenJaccard =
    Number.isFinite(overrideParsed) ? Math.max(0.2, overrideParsed) : null;
  const effectiveMinTokenJaccard = overrideTokenJaccard ?? minTokenJaccard;
  return {
    enabled: envBool("RELATION_BUILDER_COMPLEMENTARY_RELAXED_ENABLED", false),
    minTokenJaccard,
    effectiveMinTokenJaccard,
    overrideTokenJaccard,
    minSharedInformativeTokens: Math.max(
      1,
      Math.floor(envNum("RELATION_BUILDER_COMPLEMENTARY_RELAXED_MIN_SHARED_TOKENS", 5))
    ),
    ...(() => {
      const baselinePrefix = Math.max(
        0,
        Math.floor(envNum("RELATION_BUILDER_COMPLEMENTARY_RELAXED_MIN_COMMON_PREFIX_LENGTH", 12))
      );
      const overrideRaw = env(
        "RELATION_BUILDER_COMPLEMENTARY_RELAXED_OVERRIDE_COMMON_PREFIX_LENGTH",
        ""
      );
      const overrideParsed = overrideRaw === "" ? NaN : Number(overrideRaw);
      const overrideCommonPrefixLength = Number.isFinite(overrideParsed)
        ? Math.max(0, Math.floor(overrideParsed))
        : null;
      const effectiveMinCommonPrefixLength =
        overrideCommonPrefixLength ?? baselinePrefix;
      return {
        minCommonPrefixLength: baselinePrefix,
        effectiveMinCommonPrefixLength,
        overrideCommonPrefixLength,
      };
    })(),
    sensitivityJaccard048: Math.max(
      0.2,
      envNum("RELATION_BUILDER_COMPLEMENTARY_RELAXED_SENSITIVITY_JACCARD_048", 0.48)
    ),
    sensitivityJaccard044: Math.max(
      0.2,
      envNum("RELATION_BUILDER_COMPLEMENTARY_RELAXED_SENSITIVITY_JACCARD_044", 0.44)
    ),
    sensitivityJaccard040: Math.max(
      0.2,
      envNum("RELATION_BUILDER_COMPLEMENTARY_RELAXED_SENSITIVITY_JACCARD_040", 0.4)
    ),
    sensitivitySharedThreshold4: Math.max(
      1,
      Math.floor(envNum("RELATION_BUILDER_COMPLEMENTARY_RELAXED_SENSITIVITY_SHARED_4", 4))
    ),
    sensitivitySharedThreshold3: Math.max(
      1,
      Math.floor(envNum("RELATION_BUILDER_COMPLEMENTARY_RELAXED_SENSITIVITY_SHARED_3", 3))
    ),
    sensitivityPrefixThreshold10: Math.max(
      0,
      Math.floor(envNum("RELATION_BUILDER_COMPLEMENTARY_RELAXED_SENSITIVITY_PREFIX_10", 10))
    ),
    sensitivityPrefixThreshold8: Math.max(
      0,
      Math.floor(envNum("RELATION_BUILDER_COMPLEMENTARY_RELAXED_SENSITIVITY_PREFIX_8", 8))
    ),
  };
}

function complementaryRelaxedPolicySnapshot(
  p: ComplementaryRelaxedPolicy
): Record<string, number | string | boolean> {
  return {
    complementaryRelaxedEnabled: p.enabled,
    baselineMinTokenJaccard: p.minTokenJaccard,
    /** Alias de baseline para compatibilidade com leituras antigas. */
    minTokenJaccard: p.minTokenJaccard,
    effectiveMinTokenJaccard: p.effectiveMinTokenJaccard,
    overrideTokenJaccard: p.overrideTokenJaccard ?? "null",
    minSharedInformativeTokens: p.minSharedInformativeTokens,
    baselineMinCommonPrefixLength: p.minCommonPrefixLength,
    /** Alias de baseline para compatibilidade com leituras antigas. */
    minCommonPrefixLength: p.minCommonPrefixLength,
    effectiveMinCommonPrefixLength: p.effectiveMinCommonPrefixLength,
    overrideCommonPrefixLength: p.overrideCommonPrefixLength ?? "null",
    sensitivityJaccard048: p.sensitivityJaccard048,
    sensitivityJaccard044: p.sensitivityJaccard044,
    sensitivityJaccard040: p.sensitivityJaccard040,
    sensitivitySharedThreshold4: p.sensitivitySharedThreshold4,
    sensitivitySharedThreshold3: p.sensitivitySharedThreshold3,
    sensitivityPrefixThreshold10: p.sensitivityPrefixThreshold10,
    sensitivityPrefixThreshold8: p.sensitivityPrefixThreshold8,
    jaccardSensitivityAnalysisOnly: true,
    scope: "partition_path_post_cheap_lexical_only",
    strictComplementaryUnchanged: true,
    overrideNote:
      "OVERRIDE_TOKEN_JACCARD / OVERRIDE_COMMON_PREFIX_LENGTH: optional A/B; unset uses baseline Jaccard and baseline MIN_COMMON_PREFIX_LENGTH for relaxed acceptance only",
    experimentalComplementaryPrefixPhaseNote:
      "Próxima fase prolongada recomendada (só env): RELATION_BUILDER_COMPLEMENTARY_RELAXED_OVERRIDE_COMMON_PREFIX_LENGTH=10 — baseline 12 permanece default no código",
  };
}

function normalizedQuestionPrefixKey(q: string): string {
  return (q || "").toLowerCase().replace(/\s+/g, " ").trim();
}

function longestCommonPrefixLength(a: string, b: string): number {
  const max = Math.min(a.length, b.length);
  let i = 0;
  while (i < max && a.charCodeAt(i) === b.charCodeAt(i)) i += 1;
  return i;
}

function passesRelaxedAtJaccardFloor(
  textSim: number,
  sharedInformativeTokens: number,
  prefixLen: number,
  minJ: number,
  minShared: number,
  minPrefix: number
): boolean {
  return textSim >= minJ && sharedInformativeTokens >= minShared && prefixLen >= minPrefix;
}

function stableRelaxedPairKey(a: NormalizedMarket, b: NormalizedMarket): string {
  const x = a.id || "";
  const y = b.id || "";
  return x <= y ? `${x}\t${y}` : `${y}\t${x}`;
}

function thresholdLabel(t: number): string {
  return String(round4(t));
}

/**
 * Pré-filtro barato: contagens sobre os mesmos sets de `tokenize()` (stopwords já removidas).
 * Não substitui o corte 0.2 do check completo; evita gastar orçamento em pares com overlap demasiado fraco.
 */
function evaluateCheapLexicalPrefilter(
  a: NormalizedMarket,
  b: NormalizedMarket,
  getTokens: (m: NormalizedMarket) => Set<string>,
  policy: LexicalPrefilterPolicy
): { pass: boolean; tokenJaccard: number; sharedCount: number } {
  const ta = getTokens(a);
  const tb = getTokens(b);
  let shared = 0;
  ta.forEach((w) => {
    if (tb.has(w)) shared += 1;
  });
  if (shared < policy.minSharedInformativeTokens) {
    return { pass: false, tokenJaccard: 0, sharedCount: shared };
  }
  const union = ta.size + tb.size - shared;
  const j = union > 0 ? shared / union : 0;
  if (policy.minTokenJaccard > 0 && j < policy.minTokenJaccard) {
    return { pass: false, tokenJaccard: j, sharedCount: shared };
  }
    return { pass: true, tokenJaccard: j, sharedCount: shared };
}

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function countSharedInformativeTokens(
  a: NormalizedMarket,
  b: NormalizedMarket,
  getTokens: (m: NormalizedMarket) => Set<string>
): number {
  const ta = getTokens(a);
  const tb = getTokens(b);
  let s = 0;
  ta.forEach((w) => {
    if (tb.has(w)) s += 1;
  });
  return s;
}

function recordLexicalRejected(acc: PostPrefilterPartitionAccumulator, sim: number, shared: number): void {
  acc.lexical.sumOverlapRejected += sim;
  acc.lexical.countRejected += 1;
  acc.lexical.sumSharedRejected += shared;
  acc.lexical.countSharedRejected += 1;
}

function recordLexicalAccepted(acc: PostPrefilterPartitionAccumulator, sim: number, shared: number): void {
  acc.lexical.sumOverlapAccepted += sim;
  acc.lexical.countAccepted += 1;
  acc.lexical.sumSharedAccepted += shared;
  acc.lexical.countSharedAccepted += 1;
}

/** Chave estável para ordenar mercados antes de cortar em blocos (texto só). */
function partitionSortKey(m: NormalizedMarket): string {
  const q = (m.question || "").toLowerCase().replace(/\s+/g, " ").trim();
  return q.slice(0, 128);
}

/**
 * Particiona um grupo grande em subgrupos contíguos após ordenação lexicográfica por `partitionSortKey`.
 * Mercados além de `maxSubgroups * maxPerSubgroup` ficam fora (cobertura limitada).
 */
function buildSubgroupsForLargeCategory(
  group: NormalizedMarket[],
  maxSubgroups: number,
  maxPerSubgroup: number
): NormalizedMarket[][] {
  const sorted = [...group].sort((a, b) => partitionSortKey(a).localeCompare(partitionSortKey(b)));
  const subgroups: NormalizedMarket[][] = [];
  for (let i = 0; i < sorted.length && subgroups.length < maxSubgroups; i += maxPerSubgroup) {
    subgroups.push(sorted.slice(i, i + maxPerSubgroup));
  }
  return subgroups;
}

const STOP_WORDS = new Set([
  "will", "the", "a", "an", "in", "on", "by", "of", "to", "be",
  "is", "at", "or", "and", "for", "with", "this", "that", "it",
  "from", "as", "are", "was", "has", "have", "do", "does", "did",
  "not", "no", "yes", "before", "after", "if", "than", "then",
]);

/** Exportado para classificação estrutural das micro-lanes (graph) sem duplicar stopwords. */
export function tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2 && !STOP_WORDS.has(w))
  );
}

export function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 0;
  let intersection = 0;
  a.forEach((w) => {
    if (b.has(w)) intersection++;
  });
  const union = a.size + b.size - intersection;
  return union > 0 ? intersection / union : 0;
}

function cheapNormalizedPairSummary(a: NormalizedMarket, b: NormalizedMarket): string {
  const na = (a.question || "").toLowerCase().replace(/\s+/g, " ").trim().slice(0, 72);
  const nb = (b.question || "").toLowerCase().replace(/\s+/g, " ").trim().slice(0, 72);
  return `${na} || ${nb}`;
}

function recordComplementaryRelaxedRejection(
  acc: PostPrefilterPartitionAccumulator,
  a: NormalizedMarket,
  b: NormalizedMarket,
  textSim: number,
  sharedInformativeTokens: number,
  prefixLen: number,
  tokenSimForSamples: number,
  policy: ComplementaryRelaxedPolicy
): void {
  const rra = acc.complementaryRelaxedRejectionAccumulator;
  const minJ = policy.effectiveMinTokenJaccard;
  const failJ = textSim < minJ;
  const failS = sharedInformativeTokens < policy.minSharedInformativeTokens;
  const failP = prefixLen < policy.effectiveMinCommonPrefixLength;
  const nFail = (failJ ? 1 : 0) + (failS ? 1 : 0) + (failP ? 1 : 0);
  if (nFail >= 2) {
    rra.rejectedByMultipleCriteriaCount += 1;
  }
  if (failJ) {
    rra.primaryLowTokenJaccardCount += 1;
  } else if (failS) {
    rra.primaryLowSharedInformativeTokensCount += 1;
  } else if (failP) {
    rra.primaryLowCommonPrefixLengthCount += 1;
  }

  rra.countRejected += 1;
  rra.sumTokenJaccardRejected += textSim;
  rra.sumSharedTokensRejected += sharedInformativeTokens;
  rra.sumCommonPrefixLengthRejected += prefixLen;

  bumpComplementaryRelaxedRejectJaccardBucket(rra.tokenJaccardBuckets, textSim);
  bumpComplementaryRelaxedRejectSharedTokenBucket(rra.sharedTokenBuckets, sharedInformativeTokens);
  bumpComplementaryRelaxedRejectPrefixBucket(rra.commonPrefixBuckets, prefixLen);

  const reasons: string[] = [];
  if (failJ) reasons.push("low_token_jaccard");
  if (failS) reasons.push("low_shared_informative_tokens");
  if (failP) reasons.push("low_common_prefix_length");

  if (rra.rejectedSamples.length < MAX_COMPLEMENTARY_RELAXED_REJECTED_DIAGNOSTIC_SAMPLES) {
    rra.rejectedSamples.push({
      questionA: a.question || "",
      questionB: b.question || "",
      tokenSimilarity: tokenSimForSamples,
      sharedInformativeTokens,
      commonPrefixLength: prefixLen,
      rejectedReasons: reasons,
    });
  }

  const js = rra.jaccardSensitivity;
  const minS = policy.minSharedInformativeTokens;
  const minP = policy.effectiveMinCommonPrefixLength;
  const t48 = policy.sensitivityJaccard048;
  const t44 = policy.sensitivityJaccard044;
  const t40 = policy.sensitivityJaccard040;

  if (passesRelaxedAtJaccardFloor(textSim, sharedInformativeTokens, prefixLen, t48, minS, minP)) {
    js.rescueCountAt048 += 1;
    js.sumShared048 += sharedInformativeTokens;
    js.sumPrefix048 += prefixLen;
    js.count048 += 1;
  }
  if (passesRelaxedAtJaccardFloor(textSim, sharedInformativeTokens, prefixLen, t44, minS, minP)) {
    js.rescueCountAt044 += 1;
    js.sumShared044 += sharedInformativeTokens;
    js.sumPrefix044 += prefixLen;
    js.count044 += 1;
  }
  if (passesRelaxedAtJaccardFloor(textSim, sharedInformativeTokens, prefixLen, t40, minS, minP)) {
    js.rescueCountAt040 += 1;
    js.sumShared040 += sharedInformativeTokens;
    js.sumPrefix040 += prefixLen;
    js.count040 += 1;
  }

  const looseningTiers = [t48, t44, t40]
    .filter((t) => t < minJ)
    .sort((a, b) => b - a);
  let firstRescue: string | null = null;
  for (const t of looseningTiers) {
    if (passesRelaxedAtJaccardFloor(textSim, sharedInformativeTokens, prefixLen, t, minS, minP)) {
      firstRescue = thresholdLabel(t);
      break;
    }
  }
  if (
    firstRescue != null &&
    js.potentialRescueSamples.length < MAX_COMPLEMENTARY_RELAXED_POTENTIAL_RESCUE_SAMPLES
  ) {
    js.potentialRescueSamples.push({
      questionA: a.question || "",
      questionB: b.question || "",
      tokenSimilarity: tokenSimForSamples,
      sharedInformativeTokens,
      commonPrefixLength: prefixLen,
      firstThresholdThatWouldRescue: firstRescue,
    });
  }

  const ssa = rra.sharedTokenSensitivity;
  const psa = rra.commonPrefixSensitivity;
  const pairKey = stableRelaxedPairKey(a, b);
  const sh4 = policy.sensitivitySharedThreshold4;
  const sh3 = policy.sensitivitySharedThreshold3;
  const pf10 = policy.sensitivityPrefixThreshold10;
  const pf8 = policy.sensitivityPrefixThreshold8;

  if (
    sh4 < minS &&
    passesRelaxedAtJaccardFloor(textSim, sharedInformativeTokens, prefixLen, minJ, sh4, minP)
  ) {
    ssa.rescueCountIfSharedAt4 += 1;
    ssa.sumJaccard4 += textSim;
    ssa.sumPrefix4 += prefixLen;
    ssa.count4 += 1;
    ssa.pairKeysIfSharedAt4.add(pairKey);
  }
  if (
    sh3 < minS &&
    passesRelaxedAtJaccardFloor(textSim, sharedInformativeTokens, prefixLen, minJ, sh3, minP)
  ) {
    ssa.rescueCountIfSharedAt3 += 1;
    ssa.sumJaccard3 += textSim;
    ssa.sumPrefix3 += prefixLen;
    ssa.count3 += 1;
    ssa.pairKeysIfSharedAt3.add(pairKey);
  }

  if (
    pf10 < minP &&
    passesRelaxedAtJaccardFloor(textSim, sharedInformativeTokens, prefixLen, minJ, minS, pf10)
  ) {
    psa.rescueCountIfPrefixAt10 += 1;
    psa.sumJaccard10 += textSim;
    psa.sumShared10 += sharedInformativeTokens;
    psa.count10 += 1;
    psa.pairKeysIfPrefixAt10.add(pairKey);
  }
  if (
    pf8 < minP &&
    passesRelaxedAtJaccardFloor(textSim, sharedInformativeTokens, prefixLen, minJ, minS, pf8)
  ) {
    psa.rescueCountIfPrefixAt8 += 1;
    psa.sumJaccard8 += textSim;
    psa.sumShared8 += sharedInformativeTokens;
    psa.count8 += 1;
    psa.pairKeysIfPrefixAt8.add(pairKey);
  }

  const sharedAlts = Array.from(new Set([sh4, sh3]))
    .filter((x) => x < minS)
    .sort((a, b) => b - a);
  let firstSharedRescue: string | null = null;
  for (const st of sharedAlts) {
    if (passesRelaxedAtJaccardFloor(textSim, sharedInformativeTokens, prefixLen, minJ, st, minP)) {
      firstSharedRescue = `shared_${st}`;
      break;
    }
  }
  if (
    firstSharedRescue != null &&
    ssa.potentialRescuesBySharedRelaxation.length < MAX_COMPLEMENTARY_RELAXED_SHARED_PREFIX_RESCUE_SAMPLES
  ) {
    ssa.potentialRescuesBySharedRelaxation.push({
      questionA: a.question || "",
      questionB: b.question || "",
      tokenSimilarity: tokenSimForSamples,
      sharedInformativeTokens,
      commonPrefixLength: prefixLen,
      firstThresholdThatWouldRescue: firstSharedRescue,
    });
  }

  const prefixAlts = Array.from(new Set([pf10, pf8]))
    .filter((x) => x < minP)
    .sort((a, b) => b - a);
  let firstPrefixRescue: string | null = null;
  for (const pt of prefixAlts) {
    if (passesRelaxedAtJaccardFloor(textSim, sharedInformativeTokens, prefixLen, minJ, minS, pt)) {
      firstPrefixRescue = `prefix_${pt}`;
      break;
    }
  }
  if (
    firstPrefixRescue != null &&
    psa.potentialRescuesByPrefixRelaxation.length < MAX_COMPLEMENTARY_RELAXED_SHARED_PREFIX_RESCUE_SAMPLES
  ) {
    psa.potentialRescuesByPrefixRelaxation.push({
      questionA: a.question || "",
      questionB: b.question || "",
      tokenSimilarity: tokenSimForSamples,
      sharedInformativeTokens,
      commonPrefixLength: prefixLen,
      firstThresholdThatWouldRescue: firstPrefixRescue,
    });
  }
}

/**
 * Inferência de tipo (equivalent → subset → complementary strict → opcional complementary relaxado) com diagnóstico pós-prefilter.
 * `textSim` é o Jaccard sobre tokens informativos (mesmo valor usado no resto do builder).
 */
function inferRelationTypeWithDiagnostics(
  a: NormalizedMarket,
  b: NormalizedMarket,
  textSim: number,
  acc: PostPrefilterPartitionAccumulator | undefined,
  tokenSimForSamples: number,
  sharedInformativeTokens: number,
  complementaryRelaxedPolicy: ComplementaryRelaxedPolicy
): { type: RelationType; confidence: number; complementaryInferencePath?: "strict" | "relaxed" } | null {
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

  if (acc) {
    acc.typeInference.inferTypeStageDiagnostics.equivalentEvaluatedCount += 1;
  }

  if (textSim > 0.85 && a.outcomes.length === b.outcomes.length) {
    if (acc) {
      acc.typeInference.inferTypeStageDiagnostics.equivalentMatchedCount += 1;
      const s = acc.typeInference.inferTypeAcceptedSamples;
      if (s.length < MAX_INFER_TYPE_DIAGNOSTIC_SAMPLES) {
        s.push({
          questionA: a.question || "",
          questionB: b.question || "",
          inferredType: "equivalent",
          tokenSimilarity: tokenSimForSamples,
          sharedInformativeTokens,
        });
      }
    }
    return { type: "equivalent", confidence: Math.min(0.95, textSim) };
  }

  if (acc) {
    acc.typeInference.inferTypeStageDiagnostics.subsetEvaluatedCount += 1;
  }

  if (aIsSubsetOfB || bIsSubsetOfA) {
    if (acc) {
      acc.typeInference.inferTypeStageDiagnostics.subsetMatchedCount += 1;
      const s = acc.typeInference.inferTypeAcceptedSamples;
      if (s.length < MAX_INFER_TYPE_DIAGNOSTIC_SAMPLES) {
        s.push({
          questionA: a.question || "",
          questionB: b.question || "",
          inferredType: "subset",
          tokenSimilarity: tokenSimForSamples,
          sharedInformativeTokens,
        });
      }
    }
    return { type: "subset", confidence: 0.7 };
  }

  if (acc) {
    acc.typeInference.inferTypeStageDiagnostics.complementaryEvaluatedCount += 1;
  }

  if (
    a.category === b.category &&
    a.category !== "general" &&
    a.outcomes.length === 2 &&
    b.outcomes.length === 2 &&
    textSim > 0.3
  ) {
    if (acc) {
      acc.typeInference.inferTypeStageDiagnostics.complementaryMatchedCount += 1;
      const s = acc.typeInference.inferTypeAcceptedSamples;
      if (s.length < MAX_INFER_TYPE_DIAGNOSTIC_SAMPLES) {
        s.push({
          questionA: a.question || "",
          questionB: b.question || "",
          inferredType: "complementary",
          tokenSimilarity: tokenSimForSamples,
          sharedInformativeTokens,
        });
      }
    }
    return {
      type: "complementary",
      confidence: 0.5 + textSim * 0.3,
      complementaryInferencePath: "strict",
    };
  }

  const categoryBlocksStrictComplementary =
    a.category !== b.category || a.category === "general";
  const binaryOutcomesForComplementary =
    a.outcomes.length === 2 && b.outcomes.length === 2;

  if (
    acc &&
    complementaryRelaxedPolicy.enabled &&
    categoryBlocksStrictComplementary &&
    binaryOutcomesForComplementary &&
    textSim >= 0.2
  ) {
    const ti = acc.typeInference;
    ti.complementaryRelaxedAttemptedCount += 1;
    const na = normalizedQuestionPrefixKey(a.question || "");
    const nb = normalizedQuestionPrefixKey(b.question || "");
    const prefixLen = longestCommonPrefixLength(na, nb);
    const passesRelaxed =
      textSim >= complementaryRelaxedPolicy.effectiveMinTokenJaccard &&
      sharedInformativeTokens >= complementaryRelaxedPolicy.minSharedInformativeTokens &&
      prefixLen >= complementaryRelaxedPolicy.effectiveMinCommonPrefixLength;

    if (passesRelaxed) {
      ti.complementaryRelaxedAcceptedCount += 1;
      ti.nullsRescuedByComplementaryRelaxedCount += 1;
      const ar = ti.acceptedRelationCountByTypeAfterRelaxed;
      ar.complementary = (ar.complementary ?? 0) + 1;
      const rs = ti.complementaryRelaxedAcceptedSamples;
      if (rs.length < MAX_COMPLEMENTARY_RELAXED_DIAGNOSTIC_SAMPLES) {
        rs.push({
          questionA: a.question || "",
          questionB: b.question || "",
          tokenSimilarity: tokenSimForSamples,
          sharedInformativeTokens,
          categoryA: a.category,
          categoryB: b.category,
          normalizedCommonPrefixLength: prefixLen,
        });
      }
      return {
        type: "complementary",
        confidence: 0.45 + textSim * 0.25,
        complementaryInferencePath: "relaxed",
      };
    }
    ti.complementaryRelaxedRejectedCount += 1;
    recordComplementaryRelaxedRejection(
      acc,
      a,
      b,
      textSim,
      sharedInformativeTokens,
      prefixLen,
      tokenSimForSamples,
      complementaryRelaxedPolicy
    );
  }

  if (acc) {
    const br = acc.typeInference.inferTypeNullReasonBreakdown;
    let primaryLabel: string;
    if (a.category !== b.category || a.category === "general") {
      br.templateMismatchCount += 1;
      primaryLabel = "template_mismatch_category_or_general";
    } else if (a.outcomes.length !== 2 || b.outcomes.length !== 2) {
      br.incompatibleSubjectStructureCount += 1;
      primaryLabel = "complementary_requires_binary_outcomes";
    } else if (textSim <= 0.3) {
      br.insufficientSignalAfterNormalizationCount += 1;
      primaryLabel = "text_similarity_below_complementary_threshold";
    } else {
      br.otherNullReasonCount += 1;
      primaryLabel = "unexpected_null_after_complementary_checks";
    }

    br.noPatternMatchCount += 1;
    br.noTokenSubsetPatternMatchCount += 1;
    if (a.outcomes.length !== b.outcomes.length) {
      br.equivalentMissedDueToOutcomeCountMismatchCount += 1;
    } else {
      br.equivalentMissedDueToLowTextSimilarityCount += 1;
    }

    const ns = acc.typeInference.inferTypeNullSamples;
    if (ns.length < MAX_INFER_TYPE_DIAGNOSTIC_SAMPLES) {
      ns.push({
        questionA: a.question || "",
        questionB: b.question || "",
        tokenSimilarity: tokenSimForSamples,
        sharedInformativeTokens,
        inferredNullReason: primaryLabel,
        maybeNormalizedSummary: cheapNormalizedPairSummary(a, b),
      });
    }
  }

  return null;
}

function bumpRelationType(map: Partial<Record<string, number>>, t: string): void {
  map[t] = (map[t] ?? 0) + 1;
}

type TryPairOpts = {
  /** Jaccard sobre tokens informativos (reutilizado se já calculado no pré-filtro). */
  precomputedTokenSim?: number;
  /** Acumulador pós-prefilter (caminho particionado). */
  postPrefilterAcc?: PostPrefilterPartitionAccumulator;
  /** Tokens partilhados informativos (pré-filtro ou recomputado). */
  sharedInformativeTokenCount?: number;
  /** Política complementary relaxado (caminho particionado; omitir = desligado). */
  complementaryRelaxedPolicy?: ComplementaryRelaxedPolicy;
};

function tryAddRelationFromPair(
  a: NormalizedMarket,
  b: NormalizedMarket,
  rf: RelationBuilderFunnelSnapshot,
  relations: MarketRelation[],
  involved: Set<string>,
  getTokens: (m: NormalizedMarket) => Set<string>,
  partitionPath: boolean,
  opts?: TryPairOpts
): void {
  rf.candidatePairsConsideredCount += 1;
  if (partitionPath) {
    rf.candidatePairsConsideredFromPartitioningCount += 1;
  }

  const acc = opts?.postPrefilterAcc;

  if (
    !a.question?.trim() ||
    !b.question?.trim() ||
    !a.outcomes?.length ||
    !b.outcomes?.length
  ) {
    rf.candidatePairsRejectedByMissingMetadataCount += 1;
    if (acc) {
      acc.similarityAndTypeFunnel.candidatePairsRejectedByMetadataAfterPrefilterCount += 1;
    }
    return;
  }

  const sim =
    opts?.precomputedTokenSim !== undefined
      ? opts.precomputedTokenSim
      : jaccardSimilarity(getTokens(a), getTokens(b));

  const shared =
    opts?.sharedInformativeTokenCount !== undefined
      ? opts.sharedInformativeTokenCount
      : countSharedInformativeTokens(a, b, getTokens);

  if (acc) {
    bumpTokenSimilarityBucket(acc.tokenSimilarityBuckets, sim);
  }

  if (sim < 0.2) {
    rf.candidatePairsRejectedByLowSimilarityCount += 1;
    if (acc) {
      acc.similarityAndTypeFunnel.candidatePairsRejectedBySimilarityOnlyCount += 1;
      recordLexicalRejected(acc, sim, shared);
    }
    return;
  }

  if (acc) {
    acc.typeInference.inferTypeAttemptedCount += 1;
  }

  const rel = inferRelationTypeWithDiagnostics(
    a,
    b,
    sim,
    acc,
    sim,
    shared,
    opts?.complementaryRelaxedPolicy ?? COMPLEMENTARY_RELAXED_DISABLED
  );
  if (!rel) {
    if (a.outcomes.length !== b.outcomes.length) {
      rf.candidatePairsRejectedByIncompatibleResolutionRulesCount += 1;
      if (acc) {
        acc.typeInference.inferTypeNullCount += 1;
        acc.similarityAndTypeFunnel.candidatePairsRejectedBySimilarityAndTypeCount += 1;
        recordLexicalRejected(acc, sim, shared);
      }
    } else {
      rf.candidatePairsRejectedByTypeCount += 1;
      if (acc) {
        acc.typeInference.inferTypeNullCount += 1;
        acc.similarityAndTypeFunnel.candidatePairsRejectedByTypeOnlyCount += 1;
        recordLexicalRejected(acc, sim, shared);
      }
    }
    return;
  }

  if (acc) {
    const t = rel.type;
    acc.typeInference.inferTypeCountByType[t] = (acc.typeInference.inferTypeCountByType[t] ?? 0) + 1;
    acc.typeInference.acceptedRelationCountByTypeAfterPrefilter[t] =
      (acc.typeInference.acceptedRelationCountByTypeAfterPrefilter[t] ?? 0) + 1;
    acc.similarityAndTypeFunnel.candidatePairsAcceptedAfterSimilarityAndTypeCount += 1;
    recordLexicalAccepted(acc, sim, shared);
  }

  rf.candidatePairsAcceptedAsRelationsCount += 1;
  if (partitionPath) {
    rf.candidatePairsAcceptedFromPartitioningCount += 1;
  }
  bumpRelationType(rf.relationCountByType, rel.type);
  const pushed: MarketRelation = {
    type: rel.type,
    sourceMarketId: a.id,
    targetMarketId: b.id,
    confidence: rel.confidence,
  };
  if (rel.complementaryInferencePath) {
    pushed.complementaryInferencePath = rel.complementaryInferencePath;
  }
  relations.push(pushed);
  involved.add(a.id);
  involved.add(b.id);
}

function buildGraphSourceQuality(markets: NormalizedMarket[]): GraphSourceQualitySnapshot {
  const unusableReasons = new Map<string, number>();
  let usable = 0;
  for (const m of markets) {
    const idOk = Boolean(m.id?.trim());
    const qOk = Boolean(m.question?.trim());
    const oc = m.outcomes?.length ?? 0;
    if (!idOk) {
      unusableReasons.set("missing_id", (unusableReasons.get("missing_id") ?? 0) + 1);
      continue;
    }
    if (!qOk) {
      unusableReasons.set("empty_question", (unusableReasons.get("empty_question") ?? 0) + 1);
      continue;
    }
    if (oc < 2) {
      unusableReasons.set("outcomes_lt_2", (unusableReasons.get("outcomes_lt_2") ?? 0) + 1);
      continue;
    }
    usable++;
  }
  const topReasonsMarketsWereUnusable = Array.from(unusableReasons.entries())
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count);
  const n = markets.length;
  return {
    marketsWithUsableMetadataCount: usable,
    marketsWithoutUsableMetadataCount: n - usable,
    marketsWithResolvableTimeInfoCount: 0,
    marketsWithoutResolvableTimeInfoCount: n,
    topReasonsMarketsWereUnusable,
    relationBuilderPolicySnapshot: { ...RELATION_BUILDER_POLICY_SNAPSHOT },
  };
}

export type BuildClustersDiagnostics = {
  clusters: ConstraintCluster[];
  relationBuilderFunnel: RelationBuilderFunnelSnapshot;
  clusterFormationFunnel: ClusterFormationFunnelSnapshot;
  graphSourceQuality: GraphSourceQualitySnapshot;
  /** Todas as relações emitidas (categoria + exclusivo), para componentes conexos. */
  allRelationsFlat: MarketRelation[];
};

export function buildClustersWithDiagnostics(markets: NormalizedMarket[]): BuildClustersDiagnostics {
  const t0 = Date.now();
  const partitioningPolicy = readPartitioningPolicy();
  const lexicalPolicy = readLexicalPrefilterPolicy();
  const complementaryRelaxedPolicy = readComplementaryRelaxedPolicy();
  const graphSourceQuality = buildGraphSourceQuality(markets);
  graphSourceQuality.relationBuilderPolicySnapshot = {
    ...RELATION_BUILDER_POLICY_SNAPSHOT,
    ...partitioningPolicySnapshot(partitioningPolicy),
    ...lexicalPrefilterPolicySnapshot(lexicalPolicy),
  };

  const rf = emptyRelationBuilderFunnel(markets.length);
  rf.partitioningPolicySnapshot = partitioningPolicySnapshot(partitioningPolicy);
  rf.lexicalPrefilterPolicySnapshot = lexicalPrefilterPolicySnapshot(lexicalPolicy);
  const postPrefilterAcc = emptyPostPrefilterPartitionAccumulator();
  postPrefilterAcc.typeInference.complementaryRelaxedPolicySnapshot =
    complementaryRelaxedPolicySnapshot(complementaryRelaxedPolicy);
  const cf = emptyClusterFormationFunnel();
  const allRelationsFlat: MarketRelation[] = [];

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
  let clustersRejectedBySizeCount = 0;

  const maxCat = partitioningPolicy.maxCategoryGroupSize;
  const pairwiseCap = partitioningPolicy.maxMarketsPerCategoryForPairing;

  categoryGroups.forEach((group, category) => {
    if (group.length < 2) {
      rf.categoryGroupsSkippedDueToSmallCount += 1;
      clustersRejectedBySizeCount += 1;
      return;
    }

    if (group.length > maxCat) {
      if (!partitioningPolicy.partitioningEnabled) {
        rf.categoryGroupsSkippedDueToLargeCount += 1;
        clustersRejectedBySizeCount += 1;
        return;
      }

      const subgroups = buildSubgroupsForLargeCategory(
        group,
        partitioningPolicy.maxSubgroupsPerLargeCategory,
        partitioningPolicy.maxMarketsPerSubgroup
      );
      rf.largeCategoryGroupsPartitionedCount += 1;
      rf.subgroupsCreatedCount += subgroups.length;

      const maxCovered = partitioningPolicy.maxSubgroupsPerLargeCategory * partitioningPolicy.maxMarketsPerSubgroup;
      rf.largeCategoryMarketsNotCoveredByPartitionCount += Math.max(0, group.length - maxCovered);

      if (subgroups.length === 0) {
        rf.largeCategoryGroupsStillSkippedCount += 1;
        clustersRejectedBySizeCount += 1;
        return;
      }

      rf.categoryGroupsEnteredPairingLoopCount += 1;
      const relations: MarketRelation[] = [];
      const involved = new Set<string>();
      let pairBudget = partitioningPolicy.maxPairChecksPerLargeCategory;

      outer: for (const subgroup of subgroups) {
        if (subgroup.length < 2) continue;
        rf.subgroupPairingLoopsEnteredCount += 1;
        const cap = Math.min(pairwiseCap, subgroup.length);
        for (let i = 0; i < cap; i++) {
          for (let j = i + 1; j < cap; j++) {
            const a = subgroup[i]!;
            const b = subgroup[j]!;
            if (lexicalPolicy.enabled) {
              const cheap = evaluateCheapLexicalPrefilter(a, b, getTokens, lexicalPolicy);
              if (!cheap.pass) {
                rf.candidatePairsPrefilteredOutCount += 1;
                continue;
              }
              if (pairBudget <= 0) break outer;
              pairBudget -= 1;
              rf.candidatePairsPassedCheapLexicalFilterCount += 1;
              tryAddRelationFromPair(a, b, rf, relations, involved, getTokens, true, {
                precomputedTokenSim: cheap.tokenJaccard,
                sharedInformativeTokenCount: cheap.sharedCount,
                postPrefilterAcc,
                complementaryRelaxedPolicy,
              });
            } else {
              if (pairBudget <= 0) break outer;
              pairBudget -= 1;
              rf.candidatePairsPassedCheapLexicalFilterCount += 1;
              tryAddRelationFromPair(a, b, rf, relations, involved, getTokens, true, {
                postPrefilterAcc,
                complementaryRelaxedPolicy,
              });
            }
          }
        }
      }

      if (relations.length > 0) {
        const clusterMarkets = group.filter((m) => involved.has(m.id));
        const cluster: ConstraintCluster = {
          clusterId: `cluster-${category}-${clusters.length}`,
          markets: clusterMarkets,
          relations,
        };
        clusters.push(cluster);
        for (const r of relations) allRelationsFlat.push(r);
      }
      return;
    }

    rf.categoryGroupsEnteredPairingLoopCount += 1;
    const relations: MarketRelation[] = [];
    const involved = new Set<string>();

    const cap = Math.min(pairwiseCap, group.length);
    for (let i = 0; i < cap; i++) {
      for (let j = i + 1; j < cap; j++) {
        tryAddRelationFromPair(group[i]!, group[j]!, rf, relations, involved, getTokens, false, undefined);
      }
    }

    if (relations.length > 0) {
      const clusterMarkets = group.filter((m) => involved.has(m.id));
      const cluster: ConstraintCluster = {
        clusterId: `cluster-${category}-${clusters.length}`,
        markets: clusterMarkets,
        relations,
      };
      clusters.push(cluster);
      for (const r of relations) allRelationsFlat.push(r);
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
          const a = group[i]!;
          const b = group[j]!;
          rf.candidatePairsConsideredCount += 1;
          if (!a.id?.trim() || !b.id?.trim() || !a.slug?.trim() || !b.slug?.trim()) {
            rf.candidatePairsRejectedByMissingMetadataCount += 1;
            continue;
          }
          rf.candidatePairsAcceptedAsRelationsCount += 1;
          bumpRelationType(rf.relationCountByType, "exclusive");
          relations.push({
            type: "exclusive",
            sourceMarketId: a.id,
            targetMarketId: b.id,
            confidence: 0.6,
          });
        }
      }
      if (relations.length > 0) {
        const cluster: ConstraintCluster = {
          clusterId: `cluster-excl-${slug}-${clusters.length}`,
          markets: group,
          relations,
        };
        clusters.push(cluster);
        for (const r of relations) allRelationsFlat.push(r);
      }
    });
  }

  {
    const pre = rf.candidatePairsPrefilteredOutCount;
    const exp = rf.candidatePairsPassedCheapLexicalFilterCount;
    const ra = rf.candidatePairsAcceptedFromPartitioningCount;
    rf.pairBudgetEfficiencyStats = {
      pairBudgetCap: partitioningPolicy.maxPairChecksPerLargeCategory,
      expensivePartitionChecks: exp,
      relationsAcceptedInPartitionPath: ra,
      acceptanceRate: exp > 0 ? round4(ra / exp) : null,
      prefilterRejections: pre,
      cheapPrefilterPassRate: pre + exp > 0 ? round4(exp / (pre + exp)) : null,
    };
  }

  const afterCheap = rf.candidatePairsPassedCheapLexicalFilterCount;
  rf.postPrefilterPartitionDiagnostics =
    afterCheap > 0 ? finalizePostPrefilterPartitionDiagnostics(postPrefilterAcc, afterCheap) : null;
  if (rf.postPrefilterPartitionDiagnostics?.lexical.averageTokenOverlapOfAcceptedPairs != null) {
    rf.averageTokenOverlapOfAcceptedPairs =
      rf.postPrefilterPartitionDiagnostics.lexical.averageTokenOverlapOfAcceptedPairs;
  } else {
    rf.averageTokenOverlapOfAcceptedPairs = null;
  }

  const involvedIds = new Set<string>();
  for (const r of allRelationsFlat) {
    involvedIds.add(r.sourceMarketId);
    involvedIds.add(r.targetMarketId);
  }
  rf.uniqueEntitiesMatchedCount = involvedIds.size;

  const rib = emptyRelationsByInferencePathBreakdown();
  for (const r of allRelationsFlat) {
    switch (r.type) {
      case "equivalent":
        rib.equivalentCount += 1;
        break;
      case "subset":
        rib.subsetCount += 1;
        break;
      case "exclusive":
        rib.exclusiveCount += 1;
        break;
      case "complementary":
        if (r.complementaryInferencePath === "relaxed") rib.complementaryRelaxedCount += 1;
        else rib.complementaryStrictCount += 1;
        break;
      default:
        break;
    }
  }
  rf.relationsByInferencePath = rib;

  cf.relationsInputCount = allRelationsFlat.length;
  cf.connectedComponentsCount = countConnectedComponentsFromMarketIds(allRelationsFlat);
  cf.clustersBeforeFilteringCount = clusters.length;
  cf.clustersRejectedBySizeCount = clustersRejectedBySizeCount;
  cf.clustersRejectedByInvalidStructureCount = 0;
  cf.clustersAcceptedCount = clusters.length;
  cf.rawOpportunitiesProducedCount = 0;

  const elapsed = Date.now() - t0;
  console.log(
    `[RelationBuilder] Built ${clusters.length} clusters with ${allRelationsFlat.length} relations in ${elapsed}ms | pairs=${rf.candidatePairsConsideredCount} part_pairs=${rf.candidatePairsConsideredFromPartitioningCount} low_sim=${rf.candidatePairsRejectedByLowSimilarityCount} large_part=${rf.largeCategoryGroupsPartitionedCount}`
  );

  return {
    clusters,
    relationBuilderFunnel: rf,
    clusterFormationFunnel: cf,
    graphSourceQuality,
    allRelationsFlat,
  };
}

export function buildClusters(markets: NormalizedMarket[]): ConstraintCluster[] {
  return buildClustersWithDiagnostics(markets).clusters;
}
