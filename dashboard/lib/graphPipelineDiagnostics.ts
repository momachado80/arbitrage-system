/**
 * Tipos e utilitários leves para diagnóstico do pipeline grafo (relations → clusters → scan).
 * Snapshots são preenchidos no ciclo do graph scan, não em GET.
 */

/** Distribuição do Jaccard(token) para pares que já passaram no cheap prefilter (check caro). */
export type TokenSimilarityBucketsPostPrefilter = {
  lt005: number;
  b005_01: number;
  b01_02: number;
  b02_04: number;
  gte04: number;
};

/**
 * Decomposição de falhas após cheap prefilter (só caminho particionado com orçamento).
 * Buckets de outcome são mutuamente exclusivos após metadata OK.
 */
export type SimilarityAndTypeFunnelPostPrefilter = {
  candidatePairsAfterCheapPrefilterCount: number;
  candidatePairsRejectedByMetadataAfterPrefilterCount: number;
  /** sim < 0.2 (corte global inalterado). */
  candidatePairsRejectedBySimilarityOnlyCount: number;
  /** sim >= 0.2, infer null, mesmo comprimento de outcomes. */
  candidatePairsRejectedByTypeOnlyCount: number;
  /** sim >= 0.2, infer null, outcomes de comprimento diferente. */
  candidatePairsRejectedBySimilarityAndTypeCount: number;
  candidatePairsAcceptedAfterSimilarityAndTypeCount: number;
};

/**
 * Razões de `inferRelationType === null` alinhadas ao fluxo real (equivalent → subset → complementary).
 *
 * Partição A — primeira falha no ramo complementary (ordem: categoria → forma → sim): soma = inferTypeNullCount.
 * Partição B — por que o ramo equivalent não aplicou: outcomeMismatch + lowTextSim = inferTypeNullCount.
 * noTokenSubsetPatternMatchCount = inferTypeNullCount (em todo null o subset falhou).
 * noPatternMatchCount = inferTypeNullCount (nenhum dos três tipos; total explícito).
 */
export type InferTypeNullReasonBreakdown = {
  templateMismatchCount: number;
  incompatibleSubjectStructureCount: number;
  insufficientSignalAfterNormalizationCount: number;
  noPatternMatchCount: number;
  noTokenSubsetPatternMatchCount: number;
  equivalentMissedDueToOutcomeCountMismatchCount: number;
  equivalentMissedDueToLowTextSimilarityCount: number;
  normalizationMismatchCount: number;
  ambiguousDirectionCount: number;
  outcomeSemanticsUnclearCount: number;
  otherNullReasonCount: number;
};

export type InferTypeStageDiagnostics = {
  equivalentEvaluatedCount: number;
  equivalentMatchedCount: number;
  subsetEvaluatedCount: number;
  subsetMatchedCount: number;
  complementaryEvaluatedCount: number;
  complementaryMatchedCount: number;
};

export type InferTypeNullSampleRow = {
  questionA: string;
  questionB: string;
  tokenSimilarity: number;
  sharedInformativeTokens: number;
  inferredNullReason: string;
  /** Resumo barato: primeiros chars das perguntas normalizadas (só diagnóstico). */
  maybeNormalizedSummary: string | null;
};

export type InferTypeAcceptedSampleRow = {
  questionA: string;
  questionB: string;
  inferredType: string;
  tokenSimilarity: number;
  sharedInformativeTokens: number;
};

/** Amostras de relações aceites só via ramo complementary relaxado (categoria/general). */
export type InferTypeRelaxedAcceptedSampleRow = {
  questionA: string;
  questionB: string;
  tokenSimilarity: number;
  sharedInformativeTokens: number;
  categoryA: string;
  categoryB: string;
  normalizedCommonPrefixLength: number;
};

/** Funil de falhas do complementary relaxado (só pares que tentaram relaxado e foram rejeitados). */
export type ComplementaryRelaxedFailureBreakdown = {
  acceptedCount: number;
  /** Primeira condição falhada na ordem: Jaccard → shared tokens → prefixo (partição dos rejeitados). */
  rejectedByLowTokenJaccardCount: number;
  rejectedByLowSharedInformativeTokensCount: number;
  rejectedByLowCommonPrefixLengthCount: number;
  /** Rejeitados em que falharam ≥2 critérios em simultâneo (sobreposição informativa). */
  rejectedByMultipleCriteriaCount: number;
};

export type ComplementaryRelaxedRejectJaccardBuckets = {
  lt030: number;
  b030_040: number;
  b040_052: number;
  gte052: number;
};

export type ComplementaryRelaxedRejectSharedTokenBuckets = {
  lt3: number;
  eq3: number;
  eq4: number;
  gte5: number;
};

export type ComplementaryRelaxedRejectPrefixBuckets = {
  lt6: number;
  b6_11: number;
  gte12: number;
};

export type ComplementaryRelaxedRejectHistograms = {
  tokenJaccardBuckets: ComplementaryRelaxedRejectJaccardBuckets;
  sharedTokenBuckets: ComplementaryRelaxedRejectSharedTokenBuckets;
  commonPrefixBuckets: ComplementaryRelaxedRejectPrefixBuckets;
};

export type InferTypeRelaxedRejectedSampleRow = {
  questionA: string;
  questionB: string;
  tokenSimilarity: number;
  sharedInformativeTokens: number;
  commonPrefixLength: number;
  rejectedReasons: string[];
};

/** Counterfactual: rejeitados que passariam só com limiar de Jaccard mais baixo (shared/prefix inalterados). */
export type InferTypeRelaxedPotentialRescueSampleRow = {
  questionA: string;
  questionB: string;
  tokenSimilarity: number;
  sharedInformativeTokens: number;
  commonPrefixLength: number;
  /** Menor limiar alternativo (entre os de sensibilidade) que já bastaria. */
  firstThresholdThatWouldRescue: string;
};

/** Diagnóstico apenas: não altera aceitação (excepto override experimental explícito na política). */
export type ComplementaryRelaxedJaccardSensitivity = {
  currentThreshold: number;
  analysisThreshold048: number;
  analysisThreshold044: number;
  analysisThreshold040: number;
  rescueCountAt048: number;
  rescueCountAt044: number;
  rescueCountAt040: number;
  rescueRateAt048: number | null;
  rescueRateAt044: number | null;
  rescueRateAt040: number | null;
  averageSharedTokensAmongPotentialRescuesAt048: number | null;
  averageCommonPrefixAmongPotentialRescuesAt048: number | null;
  averageSharedTokensAmongPotentialRescuesAt044: number | null;
  averageCommonPrefixAmongPotentialRescuesAt044: number | null;
  averageSharedTokensAmongPotentialRescuesAt040: number | null;
  averageCommonPrefixAmongPotentialRescuesAt040: number | null;
  complementaryRelaxedPotentialRescueSamples: InferTypeRelaxedPotentialRescueSampleRow[];
};

/** Sub-acumulador: sensibilidade do Jaccard sobre rejeitados do relaxed. */
export type ComplementaryRelaxedJaccardSensitivityAccumulator = {
  rescueCountAt048: number;
  rescueCountAt044: number;
  rescueCountAt040: number;
  sumShared048: number;
  sumPrefix048: number;
  count048: number;
  sumShared044: number;
  sumPrefix044: number;
  count044: number;
  sumShared040: number;
  sumPrefix040: number;
  count040: number;
  potentialRescueSamples: InferTypeRelaxedPotentialRescueSampleRow[];
};

export type ComplementaryRelaxedSharedTokenSensitivityAccumulator = {
  rescueCountIfSharedAt4: number;
  rescueCountIfSharedAt3: number;
  sumJaccard4: number;
  sumPrefix4: number;
  count4: number;
  sumJaccard3: number;
  sumPrefix3: number;
  count3: number;
  pairKeysIfSharedAt4: Set<string>;
  pairKeysIfSharedAt3: Set<string>;
  potentialRescuesBySharedRelaxation: InferTypeRelaxedPotentialRescueSampleRow[];
};

export type ComplementaryRelaxedCommonPrefixSensitivityAccumulator = {
  rescueCountIfPrefixAt10: number;
  rescueCountIfPrefixAt8: number;
  sumJaccard10: number;
  sumShared10: number;
  count10: number;
  sumJaccard8: number;
  sumShared8: number;
  count8: number;
  pairKeysIfPrefixAt10: Set<string>;
  pairKeysIfPrefixAt8: Set<string>;
  potentialRescuesByPrefixRelaxation: InferTypeRelaxedPotentialRescueSampleRow[];
};

/** Sensibilidade contrafactual: min shared (Jaccard e prefix actuais). */
export type ComplementaryRelaxedSharedTokenSensitivity = {
  currentMinSharedInformativeTokens: number;
  analysisSharedThreshold4: number;
  analysisSharedThreshold3: number;
  rescueCountIfSharedAt4: number;
  rescueCountIfSharedAt3: number;
  rescueRateIfSharedAt4: number | null;
  rescueRateIfSharedAt3: number | null;
  averageTokenJaccardAmongPotentialRescuesIfSharedAt4: number | null;
  averageCommonPrefixAmongPotentialRescuesIfSharedAt4: number | null;
  averageTokenJaccardAmongPotentialRescuesIfSharedAt3: number | null;
  averageCommonPrefixAmongPotentialRescuesIfSharedAt3: number | null;
  potentialRescuesBySharedRelaxation: InferTypeRelaxedPotentialRescueSampleRow[];
};

/** Sensibilidade contrafactual: min prefix (Jaccard e shared actuais). */
export type ComplementaryRelaxedCommonPrefixSensitivity = {
  currentMinCommonPrefixLength: number;
  analysisPrefixThreshold10: number;
  analysisPrefixThreshold8: number;
  rescueCountIfPrefixAt10: number;
  rescueCountIfPrefixAt8: number;
  rescueRateIfPrefixAt10: number | null;
  rescueRateIfPrefixAt8: number | null;
  averageTokenJaccardAmongPotentialRescuesIfPrefixAt10: number | null;
  averageSharedTokensAmongPotentialRescuesIfPrefixAt10: number | null;
  averageTokenJaccardAmongPotentialRescuesIfPrefixAt8: number | null;
  averageSharedTokensAmongPotentialRescuesIfPrefixAt8: number | null;
  potentialRescuesByPrefixRelaxation: InferTypeRelaxedPotentialRescueSampleRow[];
};

/** Comparação de conjuntos: shared@4 vs prefix@10 (contrafactuais mais brandos listados). */
export type ComplementaryRelaxedSensitivityOverlap = {
  potentialRescuePairCountIfSharedAt4: number;
  potentialRescuePairCountIfPrefixAt10: number;
  intersectionCount: number;
  onlySharedAt4Count: number;
  onlyPrefixAt10Count: number;
  unionCount: number;
  /** |A∩B|/|A∪B|; null se união vazia. */
  jaccardCoeff: number | null;
};

export type ComplementaryRelaxedSensitivity = {
  sharedTokenRelaxation: ComplementaryRelaxedSharedTokenSensitivity;
  commonPrefixRelaxation: ComplementaryRelaxedCommonPrefixSensitivity;
  overlapBetweenPotentialRescueSets: ComplementaryRelaxedSensitivityOverlap;
};

/** Resumo A/B para experimentos do complementary relaxed (ex.: override de prefix). */
export type ComplementaryRelaxedABSnapshot = {
  mode: "baseline" | "override";
  effectivePrefixThreshold: number;
  acceptedCount: number;
  rejectedCount: number;
  acceptanceRate: number | null;
};

/** Acumulação interna (graph scan); consolidado em `finalizePostPrefilterPartitionDiagnostics`. */
export type ComplementaryRelaxedRejectionAccumulator = {
  primaryLowTokenJaccardCount: number;
  primaryLowSharedInformativeTokensCount: number;
  primaryLowCommonPrefixLengthCount: number;
  rejectedByMultipleCriteriaCount: number;
  sumTokenJaccardRejected: number;
  sumSharedTokensRejected: number;
  sumCommonPrefixLengthRejected: number;
  countRejected: number;
  tokenJaccardBuckets: ComplementaryRelaxedRejectJaccardBuckets;
  sharedTokenBuckets: ComplementaryRelaxedRejectSharedTokenBuckets;
  commonPrefixBuckets: ComplementaryRelaxedRejectPrefixBuckets;
  rejectedSamples: InferTypeRelaxedRejectedSampleRow[];
  jaccardSensitivity: ComplementaryRelaxedJaccardSensitivityAccumulator;
  sharedTokenSensitivity: ComplementaryRelaxedSharedTokenSensitivityAccumulator;
  commonPrefixSensitivity: ComplementaryRelaxedCommonPrefixSensitivityAccumulator;
};

export type TypeInferencePostPrefilterDiagnostics = {
  inferTypeAttemptedCount: number;
  inferTypeNullCount: number;
  inferTypeCountByType: Partial<Record<string, number>>;
  acceptedRelationCountByTypeAfterPrefilter: Partial<Record<string, number>>;
  inferTypeNullReasonBreakdown: InferTypeNullReasonBreakdown;
  inferTypeStageDiagnostics: InferTypeStageDiagnostics;
  inferTypeNullSamples: InferTypeNullSampleRow[];
  inferTypeAcceptedSamples: InferTypeAcceptedSampleRow[];
  /** Tentativas de complementary relaxado (só pares elegíveis: bloqueio strict = categoria/general, binários). */
  complementaryRelaxedAttemptedCount: number;
  complementaryRelaxedAcceptedCount: number;
  complementaryRelaxedRejectedCount: number;
  /** accepted / attempted; null se attempted = 0. */
  complementaryRelaxedAcceptanceRate: number | null;
  /** Igual a complementaryRelaxedAcceptedCount — nulls que deixariam de ser templateMismatch-only. */
  nullsRescuedByComplementaryRelaxedCount: number;
  complementaryRelaxedPolicySnapshot: Record<string, number | string | boolean>;
  /** Contagens por tipo apenas para relações aceites pelo ramo relaxado. */
  acceptedRelationCountByTypeAfterRelaxed: Partial<Record<string, number>>;
  complementaryRelaxedAcceptedSamples: InferTypeRelaxedAcceptedSampleRow[];
  complementaryRelaxedFailureBreakdown: ComplementaryRelaxedFailureBreakdown;
  averageTokenJaccardRejectedByRelaxed: number | null;
  averageSharedTokensRejectedByRelaxed: number | null;
  averageCommonPrefixLengthRejectedByRelaxed: number | null;
  complementaryRelaxedRejectHistograms: ComplementaryRelaxedRejectHistograms;
  complementaryRelaxedRejectedSamples: InferTypeRelaxedRejectedSampleRow[];
  complementaryRelaxedJaccardSensitivity: ComplementaryRelaxedJaccardSensitivity;
  complementaryRelaxedSensitivity: ComplementaryRelaxedSensitivity;
  /** Baseline env `MIN_COMMON_PREFIX_LENGTH` (default 12). */
  baselineMinCommonPrefixLength: number;
  /** Limiar de prefixo activo no relaxed (após override opcional). */
  effectiveMinCommonPrefixLength: number;
  /** `RELATION_BUILDER_COMPLEMENTARY_RELAXED_OVERRIDE_COMMON_PREFIX_LENGTH` ou null. */
  overrideCommonPrefixLength: number | null;
  complementaryRelaxedABSnapshot: ComplementaryRelaxedABSnapshot;
};

export type PostPrefilterLexicalStats = {
  averageTokenOverlapOfAcceptedPairs: number | null;
  averageTokenOverlapOfRejectedPairs: number | null;
  averageSharedInformativeTokensAccepted: number | null;
  averageSharedInformativeTokensRejected: number | null;
};

export type PostPrefilterPartitionDiagnostics = {
  similarityAndTypeFunnel: SimilarityAndTypeFunnelPostPrefilter;
  tokenSimilarityBuckets: TokenSimilarityBucketsPostPrefilter;
  typeInference: TypeInferencePostPrefilterDiagnostics;
  lexical: PostPrefilterLexicalStats;
};

export const MAX_INFER_TYPE_DIAGNOSTIC_SAMPLES = 10;
export const MAX_COMPLEMENTARY_RELAXED_DIAGNOSTIC_SAMPLES = 10;
export const MAX_COMPLEMENTARY_RELAXED_REJECTED_DIAGNOSTIC_SAMPLES = 10;
export const MAX_COMPLEMENTARY_RELAXED_POTENTIAL_RESCUE_SAMPLES = 10;
export const MAX_COMPLEMENTARY_RELAXED_SHARED_PREFIX_RESCUE_SAMPLES = 10;

export type PostPrefilterPartitionAccumulator = {
  similarityAndTypeFunnel: {
    candidatePairsRejectedByMetadataAfterPrefilterCount: number;
    candidatePairsRejectedBySimilarityOnlyCount: number;
    candidatePairsRejectedByTypeOnlyCount: number;
    candidatePairsRejectedBySimilarityAndTypeCount: number;
    candidatePairsAcceptedAfterSimilarityAndTypeCount: number;
  };
  tokenSimilarityBuckets: TokenSimilarityBucketsPostPrefilter;
  typeInference: TypeInferencePostPrefilterDiagnostics;
  /** Falhas do complementary relaxado; merge em finalize. */
  complementaryRelaxedRejectionAccumulator: ComplementaryRelaxedRejectionAccumulator;
  lexical: {
    sumOverlapAccepted: number;
    countAccepted: number;
    sumOverlapRejected: number;
    countRejected: number;
    sumSharedAccepted: number;
    countSharedAccepted: number;
    sumSharedRejected: number;
    countSharedRejected: number;
  };
};

export function emptyTokenSimilarityBucketsPostPrefilter(): TokenSimilarityBucketsPostPrefilter {
  return { lt005: 0, b005_01: 0, b01_02: 0, b02_04: 0, gte04: 0 };
}

function emptyInferTypeNullReasonBreakdown(): InferTypeNullReasonBreakdown {
  return {
    templateMismatchCount: 0,
    incompatibleSubjectStructureCount: 0,
    insufficientSignalAfterNormalizationCount: 0,
    noPatternMatchCount: 0,
    noTokenSubsetPatternMatchCount: 0,
    equivalentMissedDueToOutcomeCountMismatchCount: 0,
    equivalentMissedDueToLowTextSimilarityCount: 0,
    normalizationMismatchCount: 0,
    ambiguousDirectionCount: 0,
    outcomeSemanticsUnclearCount: 0,
    otherNullReasonCount: 0,
  };
}

function emptyInferTypeStageDiagnostics(): InferTypeStageDiagnostics {
  return {
    equivalentEvaluatedCount: 0,
    equivalentMatchedCount: 0,
    subsetEvaluatedCount: 0,
    subsetMatchedCount: 0,
    complementaryEvaluatedCount: 0,
    complementaryMatchedCount: 0,
  };
}

function emptyComplementaryRelaxedRejectJaccardBuckets(): ComplementaryRelaxedRejectJaccardBuckets {
  return { lt030: 0, b030_040: 0, b040_052: 0, gte052: 0 };
}

function emptyComplementaryRelaxedRejectSharedTokenBuckets(): ComplementaryRelaxedRejectSharedTokenBuckets {
  return { lt3: 0, eq3: 0, eq4: 0, gte5: 0 };
}

function emptyComplementaryRelaxedRejectPrefixBuckets(): ComplementaryRelaxedRejectPrefixBuckets {
  return { lt6: 0, b6_11: 0, gte12: 0 };
}

function emptyComplementaryRelaxedJaccardSensitivityAccumulator(): ComplementaryRelaxedJaccardSensitivityAccumulator {
  return {
    rescueCountAt048: 0,
    rescueCountAt044: 0,
    rescueCountAt040: 0,
    sumShared048: 0,
    sumPrefix048: 0,
    count048: 0,
    sumShared044: 0,
    sumPrefix044: 0,
    count044: 0,
    sumShared040: 0,
    sumPrefix040: 0,
    count040: 0,
    potentialRescueSamples: [],
  };
}

function emptyComplementaryRelaxedSharedTokenSensitivityAccumulator(): ComplementaryRelaxedSharedTokenSensitivityAccumulator {
  return {
    rescueCountIfSharedAt4: 0,
    rescueCountIfSharedAt3: 0,
    sumJaccard4: 0,
    sumPrefix4: 0,
    count4: 0,
    sumJaccard3: 0,
    sumPrefix3: 0,
    count3: 0,
    pairKeysIfSharedAt4: new Set(),
    pairKeysIfSharedAt3: new Set(),
    potentialRescuesBySharedRelaxation: [],
  };
}

function emptyComplementaryRelaxedCommonPrefixSensitivityAccumulator(): ComplementaryRelaxedCommonPrefixSensitivityAccumulator {
  return {
    rescueCountIfPrefixAt10: 0,
    rescueCountIfPrefixAt8: 0,
    sumJaccard10: 0,
    sumShared10: 0,
    count10: 0,
    sumJaccard8: 0,
    sumShared8: 0,
    count8: 0,
    pairKeysIfPrefixAt10: new Set(),
    pairKeysIfPrefixAt8: new Set(),
    potentialRescuesByPrefixRelaxation: [],
  };
}

export function emptyComplementaryRelaxedRejectionAccumulator(): ComplementaryRelaxedRejectionAccumulator {
  return {
    primaryLowTokenJaccardCount: 0,
    primaryLowSharedInformativeTokensCount: 0,
    primaryLowCommonPrefixLengthCount: 0,
    rejectedByMultipleCriteriaCount: 0,
    sumTokenJaccardRejected: 0,
    sumSharedTokensRejected: 0,
    sumCommonPrefixLengthRejected: 0,
    countRejected: 0,
    tokenJaccardBuckets: emptyComplementaryRelaxedRejectJaccardBuckets(),
    sharedTokenBuckets: emptyComplementaryRelaxedRejectSharedTokenBuckets(),
    commonPrefixBuckets: emptyComplementaryRelaxedRejectPrefixBuckets(),
    rejectedSamples: [],
    jaccardSensitivity: emptyComplementaryRelaxedJaccardSensitivityAccumulator(),
    sharedTokenSensitivity: emptyComplementaryRelaxedSharedTokenSensitivityAccumulator(),
    commonPrefixSensitivity: emptyComplementaryRelaxedCommonPrefixSensitivityAccumulator(),
  };
}

function emptyComplementaryRelaxedSensitivityDiag(): ComplementaryRelaxedSensitivity {
  const emptyShared: ComplementaryRelaxedSharedTokenSensitivity = {
    currentMinSharedInformativeTokens: 5,
    analysisSharedThreshold4: 4,
    analysisSharedThreshold3: 3,
    rescueCountIfSharedAt4: 0,
    rescueCountIfSharedAt3: 0,
    rescueRateIfSharedAt4: null,
    rescueRateIfSharedAt3: null,
    averageTokenJaccardAmongPotentialRescuesIfSharedAt4: null,
    averageCommonPrefixAmongPotentialRescuesIfSharedAt4: null,
    averageTokenJaccardAmongPotentialRescuesIfSharedAt3: null,
    averageCommonPrefixAmongPotentialRescuesIfSharedAt3: null,
    potentialRescuesBySharedRelaxation: [],
  };
  const emptyPrefix: ComplementaryRelaxedCommonPrefixSensitivity = {
    currentMinCommonPrefixLength: 12,
    analysisPrefixThreshold10: 10,
    analysisPrefixThreshold8: 8,
    rescueCountIfPrefixAt10: 0,
    rescueCountIfPrefixAt8: 0,
    rescueRateIfPrefixAt10: null,
    rescueRateIfPrefixAt8: null,
    averageTokenJaccardAmongPotentialRescuesIfPrefixAt10: null,
    averageSharedTokensAmongPotentialRescuesIfPrefixAt10: null,
    averageTokenJaccardAmongPotentialRescuesIfPrefixAt8: null,
    averageSharedTokensAmongPotentialRescuesIfPrefixAt8: null,
    potentialRescuesByPrefixRelaxation: [],
  };
  const emptyOverlap: ComplementaryRelaxedSensitivityOverlap = {
    potentialRescuePairCountIfSharedAt4: 0,
    potentialRescuePairCountIfPrefixAt10: 0,
    intersectionCount: 0,
    onlySharedAt4Count: 0,
    onlyPrefixAt10Count: 0,
    unionCount: 0,
    jaccardCoeff: null,
  };
  return {
    sharedTokenRelaxation: emptyShared,
    commonPrefixRelaxation: emptyPrefix,
    overlapBetweenPotentialRescueSets: emptyOverlap,
  };
}

function emptyComplementaryRelaxedJaccardSensitivityDiag(): ComplementaryRelaxedJaccardSensitivity {
  return {
    currentThreshold: 0.52,
    analysisThreshold048: 0.48,
    analysisThreshold044: 0.44,
    analysisThreshold040: 0.4,
    rescueCountAt048: 0,
    rescueCountAt044: 0,
    rescueCountAt040: 0,
    rescueRateAt048: null,
    rescueRateAt044: null,
    rescueRateAt040: null,
    averageSharedTokensAmongPotentialRescuesAt048: null,
    averageCommonPrefixAmongPotentialRescuesAt048: null,
    averageSharedTokensAmongPotentialRescuesAt044: null,
    averageCommonPrefixAmongPotentialRescuesAt044: null,
    averageSharedTokensAmongPotentialRescuesAt040: null,
    averageCommonPrefixAmongPotentialRescuesAt040: null,
    complementaryRelaxedPotentialRescueSamples: [],
  };
}

/** Histogramas de Jaccard(token) para pares rejeitados pelo complementary relaxado. */
export function bumpComplementaryRelaxedRejectJaccardBucket(
  b: ComplementaryRelaxedRejectJaccardBuckets,
  sim: number
): void {
  if (sim < 0.3) b.lt030 += 1;
  else if (sim < 0.4) b.b030_040 += 1;
  else if (sim < 0.52) b.b040_052 += 1;
  else b.gte052 += 1;
}

export function bumpComplementaryRelaxedRejectSharedTokenBucket(
  b: ComplementaryRelaxedRejectSharedTokenBuckets,
  n: number
): void {
  if (n < 3) b.lt3 += 1;
  else if (n === 3) b.eq3 += 1;
  else if (n === 4) b.eq4 += 1;
  else b.gte5 += 1;
}

export function bumpComplementaryRelaxedRejectPrefixBucket(
  b: ComplementaryRelaxedRejectPrefixBuckets,
  len: number
): void {
  if (len < 6) b.lt6 += 1;
  else if (len < 12) b.b6_11 += 1;
  else b.gte12 += 1;
}

function emptyComplementaryRelaxedABSnapshot(): ComplementaryRelaxedABSnapshot {
  return {
    mode: "baseline",
    effectivePrefixThreshold: 12,
    acceptedCount: 0,
    rejectedCount: 0,
    acceptanceRate: null,
  };
}

function emptyComplementaryRelaxedFailureBreakdown(): ComplementaryRelaxedFailureBreakdown {
  return {
    acceptedCount: 0,
    rejectedByLowTokenJaccardCount: 0,
    rejectedByLowSharedInformativeTokensCount: 0,
    rejectedByLowCommonPrefixLengthCount: 0,
    rejectedByMultipleCriteriaCount: 0,
  };
}

function emptyComplementaryRelaxedRejectHistograms(): ComplementaryRelaxedRejectHistograms {
  return {
    tokenJaccardBuckets: emptyComplementaryRelaxedRejectJaccardBuckets(),
    sharedTokenBuckets: emptyComplementaryRelaxedRejectSharedTokenBuckets(),
    commonPrefixBuckets: emptyComplementaryRelaxedRejectPrefixBuckets(),
  };
}

export function emptyPostPrefilterPartitionAccumulator(): PostPrefilterPartitionAccumulator {
  return {
    similarityAndTypeFunnel: {
      candidatePairsRejectedByMetadataAfterPrefilterCount: 0,
      candidatePairsRejectedBySimilarityOnlyCount: 0,
      candidatePairsRejectedByTypeOnlyCount: 0,
      candidatePairsRejectedBySimilarityAndTypeCount: 0,
      candidatePairsAcceptedAfterSimilarityAndTypeCount: 0,
    },
    tokenSimilarityBuckets: emptyTokenSimilarityBucketsPostPrefilter(),
    typeInference: {
      inferTypeAttemptedCount: 0,
      inferTypeNullCount: 0,
      inferTypeCountByType: {},
      acceptedRelationCountByTypeAfterPrefilter: {},
      inferTypeNullReasonBreakdown: emptyInferTypeNullReasonBreakdown(),
      inferTypeStageDiagnostics: emptyInferTypeStageDiagnostics(),
      inferTypeNullSamples: [],
      inferTypeAcceptedSamples: [],
      complementaryRelaxedAttemptedCount: 0,
      complementaryRelaxedAcceptedCount: 0,
      complementaryRelaxedRejectedCount: 0,
      complementaryRelaxedAcceptanceRate: null,
      nullsRescuedByComplementaryRelaxedCount: 0,
      complementaryRelaxedPolicySnapshot: {},
      acceptedRelationCountByTypeAfterRelaxed: {},
      complementaryRelaxedAcceptedSamples: [],
      complementaryRelaxedFailureBreakdown: emptyComplementaryRelaxedFailureBreakdown(),
      averageTokenJaccardRejectedByRelaxed: null,
      averageSharedTokensRejectedByRelaxed: null,
      averageCommonPrefixLengthRejectedByRelaxed: null,
      complementaryRelaxedRejectHistograms: emptyComplementaryRelaxedRejectHistograms(),
      complementaryRelaxedRejectedSamples: [],
      complementaryRelaxedJaccardSensitivity: emptyComplementaryRelaxedJaccardSensitivityDiag(),
      complementaryRelaxedSensitivity: emptyComplementaryRelaxedSensitivityDiag(),
      baselineMinCommonPrefixLength: 12,
      effectiveMinCommonPrefixLength: 12,
      overrideCommonPrefixLength: null,
      complementaryRelaxedABSnapshot: emptyComplementaryRelaxedABSnapshot(),
    },
    complementaryRelaxedRejectionAccumulator: emptyComplementaryRelaxedRejectionAccumulator(),
    lexical: {
      sumOverlapAccepted: 0,
      countAccepted: 0,
      sumOverlapRejected: 0,
      countRejected: 0,
      sumSharedAccepted: 0,
      countSharedAccepted: 0,
      sumSharedRejected: 0,
      countSharedRejected: 0,
    },
  };
}

function round4diag(n: number): number {
  return Math.round(n * 10000) / 10000;
}

export function bumpTokenSimilarityBucket(b: TokenSimilarityBucketsPostPrefilter, sim: number): void {
  if (sim < 0.05) b.lt005 += 1;
  else if (sim < 0.1) b.b005_01 += 1;
  else if (sim < 0.2) b.b01_02 += 1;
  else if (sim < 0.4) b.b02_04 += 1;
  else b.gte04 += 1;
}

export function finalizePostPrefilterPartitionDiagnostics(
  acc: PostPrefilterPartitionAccumulator,
  afterCheap: number
): PostPrefilterPartitionDiagnostics {
  const lex = acc.lexical;
  const ti = acc.typeInference;
  const relaxedAttempted = ti.complementaryRelaxedAttemptedCount;
  const relaxedAccepted = ti.complementaryRelaxedAcceptedCount;
  const relaxedRate =
    relaxedAttempted > 0 ? round4diag(relaxedAccepted / relaxedAttempted) : null;
  const rra = acc.complementaryRelaxedRejectionAccumulator;
  const rejN = rra.countRejected;
  const relaxedFailureBreakdown: ComplementaryRelaxedFailureBreakdown = {
    acceptedCount: relaxedAccepted,
    rejectedByLowTokenJaccardCount: rra.primaryLowTokenJaccardCount,
    rejectedByLowSharedInformativeTokensCount: rra.primaryLowSharedInformativeTokensCount,
    rejectedByLowCommonPrefixLengthCount: rra.primaryLowCommonPrefixLengthCount,
    rejectedByMultipleCriteriaCount: rra.rejectedByMultipleCriteriaCount,
  };
  const relaxedRejectHistograms: ComplementaryRelaxedRejectHistograms = {
    tokenJaccardBuckets: { ...rra.tokenJaccardBuckets },
    sharedTokenBuckets: { ...rra.sharedTokenBuckets },
    commonPrefixBuckets: { ...rra.commonPrefixBuckets },
  };
  const snap = ti.complementaryRelaxedPolicySnapshot;
  const parseSnapOverridePrefix = (
    v: number | string | boolean | undefined
  ): number | null => {
    if (v === undefined) return null;
    if (v === "null") return null;
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string") {
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    }
    return null;
  };
  const baselineMinCommonPrefixLength =
    typeof snap.baselineMinCommonPrefixLength === "number"
      ? (snap.baselineMinCommonPrefixLength as number)
      : typeof snap.minCommonPrefixLength === "number"
        ? (snap.minCommonPrefixLength as number)
        : 12;
  const effectiveMinCommonPrefixLength =
    typeof snap.effectiveMinCommonPrefixLength === "number"
      ? (snap.effectiveMinCommonPrefixLength as number)
      : baselineMinCommonPrefixLength;
  const overrideCommonPrefixLength = parseSnapOverridePrefix(snap.overrideCommonPrefixLength);

  const complementaryRelaxedABSnapshot: ComplementaryRelaxedABSnapshot = {
    mode: overrideCommonPrefixLength !== null ? "override" : "baseline",
    effectivePrefixThreshold: effectiveMinCommonPrefixLength,
    acceptedCount: relaxedAccepted,
    rejectedCount: ti.complementaryRelaxedRejectedCount,
    acceptanceRate: relaxedRate,
  };

  const curTh =
    typeof snap.effectiveMinTokenJaccard === "number" ? (snap.effectiveMinTokenJaccard as number) : 0.52;
  const an48 =
    typeof snap.sensitivityJaccard048 === "number" ? (snap.sensitivityJaccard048 as number) : 0.48;
  const an44 =
    typeof snap.sensitivityJaccard044 === "number" ? (snap.sensitivityJaccard044 as number) : 0.44;
  const an40 =
    typeof snap.sensitivityJaccard040 === "number" ? (snap.sensitivityJaccard040 as number) : 0.4;
  const js = rra.jaccardSensitivity;
  const jaccardSens: ComplementaryRelaxedJaccardSensitivity = {
    currentThreshold: curTh,
    analysisThreshold048: an48,
    analysisThreshold044: an44,
    analysisThreshold040: an40,
    rescueCountAt048: js.rescueCountAt048,
    rescueCountAt044: js.rescueCountAt044,
    rescueCountAt040: js.rescueCountAt040,
    rescueRateAt048: rejN > 0 ? round4diag(js.rescueCountAt048 / rejN) : null,
    rescueRateAt044: rejN > 0 ? round4diag(js.rescueCountAt044 / rejN) : null,
    rescueRateAt040: rejN > 0 ? round4diag(js.rescueCountAt040 / rejN) : null,
    averageSharedTokensAmongPotentialRescuesAt048:
      js.count048 > 0 ? round4diag(js.sumShared048 / js.count048) : null,
    averageCommonPrefixAmongPotentialRescuesAt048:
      js.count048 > 0 ? round4diag(js.sumPrefix048 / js.count048) : null,
    averageSharedTokensAmongPotentialRescuesAt044:
      js.count044 > 0 ? round4diag(js.sumShared044 / js.count044) : null,
    averageCommonPrefixAmongPotentialRescuesAt044:
      js.count044 > 0 ? round4diag(js.sumPrefix044 / js.count044) : null,
    averageSharedTokensAmongPotentialRescuesAt040:
      js.count040 > 0 ? round4diag(js.sumShared040 / js.count040) : null,
    averageCommonPrefixAmongPotentialRescuesAt040:
      js.count040 > 0 ? round4diag(js.sumPrefix040 / js.count040) : null,
    complementaryRelaxedPotentialRescueSamples: [...js.potentialRescueSamples],
  };

  const minSharedCur =
    typeof snap.minSharedInformativeTokens === "number"
      ? (snap.minSharedInformativeTokens as number)
      : 5;
  const minPrefixCur = effectiveMinCommonPrefixLength;
  const sh4 =
    typeof snap.sensitivitySharedThreshold4 === "number"
      ? (snap.sensitivitySharedThreshold4 as number)
      : 4;
  const sh3 =
    typeof snap.sensitivitySharedThreshold3 === "number"
      ? (snap.sensitivitySharedThreshold3 as number)
      : 3;
  const pf10 =
    typeof snap.sensitivityPrefixThreshold10 === "number"
      ? (snap.sensitivityPrefixThreshold10 as number)
      : 10;
  const pf8 =
    typeof snap.sensitivityPrefixThreshold8 === "number"
      ? (snap.sensitivityPrefixThreshold8 as number)
      : 8;

  const ssa = rra.sharedTokenSensitivity;
  const psa = rra.commonPrefixSensitivity;
  const sharedSens: ComplementaryRelaxedSharedTokenSensitivity = {
    currentMinSharedInformativeTokens: minSharedCur,
    analysisSharedThreshold4: sh4,
    analysisSharedThreshold3: sh3,
    rescueCountIfSharedAt4: ssa.rescueCountIfSharedAt4,
    rescueCountIfSharedAt3: ssa.rescueCountIfSharedAt3,
    rescueRateIfSharedAt4: rejN > 0 ? round4diag(ssa.rescueCountIfSharedAt4 / rejN) : null,
    rescueRateIfSharedAt3: rejN > 0 ? round4diag(ssa.rescueCountIfSharedAt3 / rejN) : null,
    averageTokenJaccardAmongPotentialRescuesIfSharedAt4:
      ssa.count4 > 0 ? round4diag(ssa.sumJaccard4 / ssa.count4) : null,
    averageCommonPrefixAmongPotentialRescuesIfSharedAt4:
      ssa.count4 > 0 ? round4diag(ssa.sumPrefix4 / ssa.count4) : null,
    averageTokenJaccardAmongPotentialRescuesIfSharedAt3:
      ssa.count3 > 0 ? round4diag(ssa.sumJaccard3 / ssa.count3) : null,
    averageCommonPrefixAmongPotentialRescuesIfSharedAt3:
      ssa.count3 > 0 ? round4diag(ssa.sumPrefix3 / ssa.count3) : null,
    potentialRescuesBySharedRelaxation: [...ssa.potentialRescuesBySharedRelaxation],
  };
  const prefixSens: ComplementaryRelaxedCommonPrefixSensitivity = {
    currentMinCommonPrefixLength: minPrefixCur,
    analysisPrefixThreshold10: pf10,
    analysisPrefixThreshold8: pf8,
    rescueCountIfPrefixAt10: psa.rescueCountIfPrefixAt10,
    rescueCountIfPrefixAt8: psa.rescueCountIfPrefixAt8,
    rescueRateIfPrefixAt10: rejN > 0 ? round4diag(psa.rescueCountIfPrefixAt10 / rejN) : null,
    rescueRateIfPrefixAt8: rejN > 0 ? round4diag(psa.rescueCountIfPrefixAt8 / rejN) : null,
    averageTokenJaccardAmongPotentialRescuesIfPrefixAt10:
      psa.count10 > 0 ? round4diag(psa.sumJaccard10 / psa.count10) : null,
    averageSharedTokensAmongPotentialRescuesIfPrefixAt10:
      psa.count10 > 0 ? round4diag(psa.sumShared10 / psa.count10) : null,
    averageTokenJaccardAmongPotentialRescuesIfPrefixAt8:
      psa.count8 > 0 ? round4diag(psa.sumJaccard8 / psa.count8) : null,
    averageSharedTokensAmongPotentialRescuesIfPrefixAt8:
      psa.count8 > 0 ? round4diag(psa.sumShared8 / psa.count8) : null,
    potentialRescuesByPrefixRelaxation: [...psa.potentialRescuesByPrefixRelaxation],
  };

  const keysShared4 = ssa.pairKeysIfSharedAt4;
  const keysPrefix10 = psa.pairKeysIfPrefixAt10;
  let intersectionCount = 0;
  for (const k of Array.from(keysShared4)) {
    if (keysPrefix10.has(k)) intersectionCount += 1;
  }
  const sizeS4 = keysShared4.size;
  const sizeP10 = keysPrefix10.size;
  const onlySharedAt4Count = sizeS4 - intersectionCount;
  const onlyPrefixAt10Count = sizeP10 - intersectionCount;
  const unionCount = sizeS4 + sizeP10 - intersectionCount;
  const relaxedOverlap: ComplementaryRelaxedSensitivityOverlap = {
    potentialRescuePairCountIfSharedAt4: sizeS4,
    potentialRescuePairCountIfPrefixAt10: sizeP10,
    intersectionCount,
    onlySharedAt4Count,
    onlyPrefixAt10Count,
    unionCount,
    jaccardCoeff:
      unionCount > 0 ? round4diag(intersectionCount / unionCount) : null,
  };
  const relaxedSensitivityFull: ComplementaryRelaxedSensitivity = {
    sharedTokenRelaxation: sharedSens,
    commonPrefixRelaxation: prefixSens,
    overlapBetweenPotentialRescueSets: relaxedOverlap,
  };

  return {
    similarityAndTypeFunnel: {
      candidatePairsAfterCheapPrefilterCount: afterCheap,
      ...acc.similarityAndTypeFunnel,
    },
    tokenSimilarityBuckets: { ...acc.tokenSimilarityBuckets },
    typeInference: {
      inferTypeAttemptedCount: ti.inferTypeAttemptedCount,
      inferTypeNullCount: ti.inferTypeNullCount,
      inferTypeCountByType: { ...ti.inferTypeCountByType },
      acceptedRelationCountByTypeAfterPrefilter: { ...ti.acceptedRelationCountByTypeAfterPrefilter },
      inferTypeNullReasonBreakdown: { ...ti.inferTypeNullReasonBreakdown },
      inferTypeStageDiagnostics: { ...ti.inferTypeStageDiagnostics },
      inferTypeNullSamples: [...ti.inferTypeNullSamples],
      inferTypeAcceptedSamples: [...ti.inferTypeAcceptedSamples],
      complementaryRelaxedAttemptedCount: ti.complementaryRelaxedAttemptedCount,
      complementaryRelaxedAcceptedCount: ti.complementaryRelaxedAcceptedCount,
      complementaryRelaxedRejectedCount: ti.complementaryRelaxedRejectedCount,
      complementaryRelaxedAcceptanceRate: relaxedRate,
      nullsRescuedByComplementaryRelaxedCount: ti.nullsRescuedByComplementaryRelaxedCount,
      complementaryRelaxedPolicySnapshot: { ...ti.complementaryRelaxedPolicySnapshot },
      acceptedRelationCountByTypeAfterRelaxed: { ...ti.acceptedRelationCountByTypeAfterRelaxed },
      complementaryRelaxedAcceptedSamples: [...ti.complementaryRelaxedAcceptedSamples],
      complementaryRelaxedFailureBreakdown: relaxedFailureBreakdown,
      averageTokenJaccardRejectedByRelaxed:
        rejN > 0 ? round4diag(rra.sumTokenJaccardRejected / rejN) : null,
      averageSharedTokensRejectedByRelaxed:
        rejN > 0 ? round4diag(rra.sumSharedTokensRejected / rejN) : null,
      averageCommonPrefixLengthRejectedByRelaxed:
        rejN > 0 ? round4diag(rra.sumCommonPrefixLengthRejected / rejN) : null,
      complementaryRelaxedRejectHistograms: relaxedRejectHistograms,
      complementaryRelaxedRejectedSamples: [...rra.rejectedSamples],
      complementaryRelaxedJaccardSensitivity: jaccardSens,
      complementaryRelaxedSensitivity: relaxedSensitivityFull,
      baselineMinCommonPrefixLength,
      effectiveMinCommonPrefixLength,
      overrideCommonPrefixLength,
      complementaryRelaxedABSnapshot,
    },
    lexical: {
      averageTokenOverlapOfAcceptedPairs:
        lex.countAccepted > 0 ? round4diag(lex.sumOverlapAccepted / lex.countAccepted) : null,
      averageTokenOverlapOfRejectedPairs:
        lex.countRejected > 0 ? round4diag(lex.sumOverlapRejected / lex.countRejected) : null,
      averageSharedInformativeTokensAccepted:
        lex.countSharedAccepted > 0 ? round4diag(lex.sumSharedAccepted / lex.countSharedAccepted) : null,
      averageSharedInformativeTokensRejected:
        lex.countSharedRejected > 0 ? round4diag(lex.sumSharedRejected / lex.countSharedRejected) : null,
    },
  };
}

/** Contagens de relações aceites por ramo de inferência (equivalent → subset → complementary strict/relaxed → exclusive). */
export type RelationsByInferencePathBreakdown = {
  equivalentCount: number;
  subsetCount: number;
  exclusiveCount: number;
  complementaryStrictCount: number;
  complementaryRelaxedCount: number;
};

/** Amostra de oportunidade bruta `graph_complement` gerada a partir de aresta complementary relaxed. */
export type ComplementaryRelaxedOpportunitySampleRow = {
  opportunityId: string;
  graphOpportunityType: string;
  diagnosticRelationProvenance: string;
  clusterId: string;
  clusterMarketCount: number | null;
  clusterRelationCount: number | null;
  marketIdA: string;
  marketIdB: string;
  labelA: string;
  labelB: string;
};

/** Impacto downstream do complementary relaxed (relações → clusters → raw opps → paper extras). */
export type ComplementaryRelaxedDownstreamImpact = {
  complementaryRelaxedRelationsAcceptedCount: number;
  complementaryRelaxedClustersContributedCount: number;
  complementaryRelaxedRawOpportunitiesProducedCount: number;
  /**
   * Oportunidades graph relaxadas que passaram normalização e entraram como extras no último ciclo
   * de `applyUpstreamScannerExpansion` (pode estar desfasado um ciclo face ao graph scan).
   */
  complementaryRelaxedOpportunitiesSurvivingToPaperCount: number | null;
  complementaryRelaxedShareOfRawOpportunities: number | null;
  complementaryRelaxedShareOfAcceptedRelations: number | null;
  /** Distribuição de oportunidades brutas por proveniência da aresta (diagnóstico). */
  rawOpportunityProvenanceCounts: RawOpportunityProvenanceCounts;
  complementaryRelaxedOpportunitySamples: ComplementaryRelaxedOpportunitySampleRow[];
};

export type RawOpportunityProvenanceCounts = {
  total: number;
  equivalent: number;
  subset: number;
  exclusive: number;
  complementaryStrict: number;
  complementaryRelaxed: number;
  cycle: number;
  unknown: number;
};

export type RelationBuilderFunnelSnapshot = {
  inputMarketsCount: number;
  /** Grupos `category` que entraram em pairing (≤ max OU particionados). */
  categoryGroupsEnteredPairingLoopCount: number;
  /** Grupos ignorados antes do loop: `len < 2`. */
  categoryGroupsSkippedDueToSmallCount: number;
  /** Particionamento desligado ou impossível: grupo grande descartado por inteiro. */
  categoryGroupsSkippedDueToLargeCount: number;
  /** Grupos > maxCategoryGroupSize tratados com subgrupos (particionamento activo). */
  largeCategoryGroupsPartitionedCount: number;
  /** Particionamento activo mas 0 subgrupos úteis (anómalo). */
  largeCategoryGroupsStillSkippedCount: number;
  /** Subgrupos criados no total (só grupos grandes). */
  subgroupsCreatedCount: number;
  /** Subgrupos com len≥2 onde correu o loop de pares. */
  subgroupPairingLoopsEnteredCount: number;
  /** Pares considerados só no caminho particionado (subgrupos). */
  candidatePairsConsideredFromPartitioningCount: number;
  /** Relações aceites só no caminho particionado. */
  candidatePairsAcceptedFromPartitioningCount: number;
  /** Mercados em grupos grandes que ficam fora do cap subgrupos×tamanho (não pareados por cobertura). */
  largeCategoryMarketsNotCoveredByPartitionCount: number;
  /** Caps efectivos + heurística (env + defaults). */
  partitioningPolicySnapshot: Record<string, number | string | boolean>;
  /** Caminho particionado: pares barrados pelo pré-filtro lexical barato (não consomem orçamento). */
  candidatePairsPrefilteredOutCount: number;
  /** Caminho particionado: pares que passaram o pré-filtro e gastaram 1 unit de pair-budget (checks “caros”). */
  candidatePairsPassedCheapLexicalFilterCount: number;
  lexicalPrefilterPolicySnapshot: Record<string, number | string | boolean>;
  /** Média do Jaccard sobre tokens informativos nos pares que geraram relação (caminho particionado). */
  averageTokenOverlapOfAcceptedPairs: number | null;
  pairBudgetEfficiencyStats: {
    pairBudgetCap: number;
    expensivePartitionChecks: number;
    relationsAcceptedInPartitionPath: number;
    acceptanceRate: number | null;
    prefilterRejections: number;
    /** expensive / (expensive + prefilterRejections) */
    cheapPrefilterPassRate: number | null;
  };
  candidatePairsConsideredCount: number;
  candidatePairsRejectedByMissingMetadataCount: number;
  candidatePairsRejectedByLowSimilarityCount: number;
  candidatePairsRejectedByIncompatibleResolutionRulesCount: number;
  candidatePairsRejectedByTimeMismatchCount: number;
  candidatePairsRejectedByTypeCount: number;
  candidatePairsAcceptedAsRelationsCount: number;
  relationCountByType: Partial<Record<string, number>>;
  uniqueEntitiesMatchedCount: number;
  /** Pós-cheap-prefilter (caminho particionado); `null` se não houve checks caros. */
  postPrefilterPartitionDiagnostics: PostPrefilterPartitionDiagnostics | null;
  /** Decomposição de todas as relações emitidas (categoria + exclusivo). */
  relationsByInferencePath: RelationsByInferencePathBreakdown;
  /**
   * Preenchido após graph scan no mesmo ciclo; `null` só antes do primeiro scan ou mercados vazios.
   */
  complementaryRelaxedDownstreamImpact: ComplementaryRelaxedDownstreamImpact | null;
};

export type ClusterFormationFunnelSnapshot = {
  relationsInputCount: number;
  connectedComponentsCount: number;
  clustersBeforeFilteringCount: number;
  clustersRejectedBySizeCount: number;
  clustersRejectedByInvalidStructureCount: number;
  clustersAcceptedCount: number;
  rawOpportunitiesProducedCount: number;
};

export type GraphSourceQualitySnapshot = {
  marketsWithUsableMetadataCount: number;
  marketsWithoutUsableMetadataCount: number;
  marketsWithResolvableTimeInfoCount: number;
  marketsWithoutResolvableTimeInfoCount: number;
  topReasonsMarketsWereUnusable: Array<{ reason: string; count: number }>;
  relationBuilderPolicySnapshot: Record<string, number | string | boolean>;
};

export type GraphPipelineDiagnosticsSnapshot = {
  relationBuilderFunnel: RelationBuilderFunnelSnapshot;
  clusterFormationFunnel: ClusterFormationFunnelSnapshot;
  graphSourceQuality: GraphSourceQualitySnapshot;
  capturedAtMs: number;
};

export function emptyRelationsByInferencePathBreakdown(): RelationsByInferencePathBreakdown {
  return {
    equivalentCount: 0,
    subsetCount: 0,
    exclusiveCount: 0,
    complementaryStrictCount: 0,
    complementaryRelaxedCount: 0,
  };
}

export function emptyRelationBuilderFunnel(inputMarketsCount: number): RelationBuilderFunnelSnapshot {
  return {
    inputMarketsCount,
    categoryGroupsEnteredPairingLoopCount: 0,
    categoryGroupsSkippedDueToSmallCount: 0,
    categoryGroupsSkippedDueToLargeCount: 0,
    largeCategoryGroupsPartitionedCount: 0,
    largeCategoryGroupsStillSkippedCount: 0,
    subgroupsCreatedCount: 0,
    subgroupPairingLoopsEnteredCount: 0,
    candidatePairsConsideredFromPartitioningCount: 0,
    candidatePairsAcceptedFromPartitioningCount: 0,
    largeCategoryMarketsNotCoveredByPartitionCount: 0,
    partitioningPolicySnapshot: {},
    candidatePairsPrefilteredOutCount: 0,
    candidatePairsPassedCheapLexicalFilterCount: 0,
    lexicalPrefilterPolicySnapshot: {},
    averageTokenOverlapOfAcceptedPairs: null,
    pairBudgetEfficiencyStats: {
      pairBudgetCap: 0,
      expensivePartitionChecks: 0,
      relationsAcceptedInPartitionPath: 0,
      acceptanceRate: null,
      prefilterRejections: 0,
      cheapPrefilterPassRate: null,
    },
    candidatePairsConsideredCount: 0,
    candidatePairsRejectedByMissingMetadataCount: 0,
    candidatePairsRejectedByLowSimilarityCount: 0,
    candidatePairsRejectedByIncompatibleResolutionRulesCount: 0,
    candidatePairsRejectedByTimeMismatchCount: 0,
    candidatePairsRejectedByTypeCount: 0,
    candidatePairsAcceptedAsRelationsCount: 0,
    relationCountByType: {},
    uniqueEntitiesMatchedCount: 0,
    postPrefilterPartitionDiagnostics: null,
    relationsByInferencePath: emptyRelationsByInferencePathBreakdown(),
    complementaryRelaxedDownstreamImpact: null,
  };
}

export function emptyClusterFormationFunnel(): ClusterFormationFunnelSnapshot {
  return {
    relationsInputCount: 0,
    connectedComponentsCount: 0,
    clustersBeforeFilteringCount: 0,
    clustersRejectedBySizeCount: 0,
    clustersRejectedByInvalidStructureCount: 0,
    clustersAcceptedCount: 0,
    rawOpportunitiesProducedCount: 0,
  };
}

export function emptyGraphSourceQuality(): GraphSourceQualitySnapshot {
  return {
    marketsWithUsableMetadataCount: 0,
    marketsWithoutUsableMetadataCount: 0,
    marketsWithResolvableTimeInfoCount: 0,
    marketsWithoutResolvableTimeInfoCount: 0,
    topReasonsMarketsWereUnusable: [],
    relationBuilderPolicySnapshot: {},
  };
}

/** Componentes conexos no grafo não dirigido definido por relações (mercados como nós). */
export function countConnectedComponentsFromMarketIds(
  relations: Array<{ sourceMarketId: string; targetMarketId: string }>
): number {
  if (relations.length === 0) return 0;
  const idToParent = new Map<string, string>();
  function find(x: string): string {
    let p = idToParent.get(x);
    if (p === undefined) {
      idToParent.set(x, x);
      return x;
    }
    if (p !== x) {
      const root = find(p);
      idToParent.set(x, root);
      return root;
    }
    return x;
  }
  function union(a: string, b: string): void {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) idToParent.set(ra, rb);
  }
  for (const r of relations) {
    union(r.sourceMarketId, r.targetMarketId);
  }
  const roots = new Set<string>();
  for (const r of relations) {
    roots.add(find(r.sourceMarketId));
    roots.add(find(r.targetMarketId));
  }
  return roots.size;
}
