/**
 * In-memory counters for paper trade *opening* funnel (diagnostics only).
 * Estado em globalThis: o Next pode empacotar o loop paper e as API routes em
 * chunks distintos; variáveis de módulo duplicadas zerariam o endpoint.
 */

const GLOBAL_KEY = "__paperOpenDiagnostics_v7";

const REJECTION_KEYS = [
  "RECOMMENDED_CAPITAL_LE_ZERO_AT_ENGINE",
  "ESTIMATED_NET_EDGE_BELOW_THRESHOLD",
  "CONFIDENCE_BELOW_THRESHOLD",
  "ACTIVE_TRADE_FOR_OPPORTUNITY",
  "EXPOSURE_CLUSTER_OR_MARKET_LIMIT",
  "REQUESTED_CAPITAL_LE_ZERO",
  "SIMULATOR_FILLED_CAPITAL_LE_ZERO",
  "SUPPRESSED_BY_ECONOMIC_COOLDOWN",
  "FILLED_CAPITAL_BELOW_MIN",
  "PROGRESS_PROBABILITY_FACTOR_BELOW_MIN",
  "ENTRY_ECONOMIC_SCORE_BELOW_MIN",
  "EXPECTED_REALIZABLE_NET_PNL_BELOW_MIN",
  "EXPECTED_NET_PNL_BELOW_MIN",
  "EXPECTED_NET_MARGIN_BELOW_MIN",
  "GROSS_TO_FEES_RATIO_BELOW_MIN",
  "CROSS_MARKET_NET_TO_GROSS_EDGE_BELOW_MIN",
  "BLOCKED_BY_SAFETY_GATE",
] as const;

export type PaperOpenRejectionKey = (typeof REJECTION_KEYS)[number];

import type { PaperEntryEconomicsMetrics, PaperEntryEconomicRejectionReason } from "./paperEntryEconomics";
import {
  getPaperEntryEconomicFilterConfig,
  STANDARD_CROSS_MARKET_PROFILE_KEY,
} from "./paperEntryEconomics";
import {
  buildEffectiveProgressGuardByProfileMap,
  getPaperExitProfileMemorySnapshot,
} from "./paperEntryProfileMemory";
import type { ExitCondition } from "./paperTypes";
import {
  getEconomicCooldownPolicySnapshot,
  getEconomicCooldownActiveSummary,
} from "./paperEconomicCooldown";
import { getExplorationPolicySnapshot, type PaperExplorationLastCycle } from "./paperExploration";
import {
  getUpstreamDiversityPolicySnapshot,
  getOpportunityFamilyKey,
  type UpstreamSelectionDiagnostics,
} from "./paperUpstreamDiversity";
import { getUpstreamScannerExpansionPolicySnapshot } from "./paperScannerExpansionEnv";
import { getGraphScanRuntime } from "./nodeProcessRuntimeState";
import type {
  ClusterFormationFunnelSnapshot,
  GraphSourceQualitySnapshot,
  RelationBuilderFunnelSnapshot,
} from "./graphPipelineDiagnostics";
import type { NormalizedPaperOpportunity } from "./paperTypes";
import {
  buildComplementaryRelaxedPaperImpactSnapshot,
  buildGraphOpportunityDownstreamImpactSnapshot,
  buildGraphProvenancePropagationDiagnostics,
  type ComplementaryRelaxedPaperImpactSnapshot,
  type GraphOpportunityDownstreamImpactSnapshot,
  type GraphProvenancePropagationDiagnostics,
} from "./graphOpportunityPaperImpact";
import {
  buildGraphProvenanceQualityBundle,
  type ComplementaryRelaxedIntraClusterAudit,
  type ComplementaryRelaxedQualityAudit,
  type ComplementaryRelaxedStructuralRobustness,
  type FeeImpactAudit,
  type GraphProvenanceQualityAudit,
  type GraphQualityAuditSourceDiagnostics,
} from "./graphProvenanceQualityAudit";
import {
  buildClosedTradesSourceDiagnostics,
  type ClosedTradesSourceDiagnostics,
} from "./paperClosedTradesMetrics";
import { buildPaperEconomicUnitAudit, type PaperEconomicUnitAudit } from "./paperEconomicUnitAudit";
import { buildPaperExitModelAudit, type PaperExitModelAudit } from "./paperExitModelAudit";
import { buildProtectedCoreDiagnostics, type ProtectedCoreDiagnostics } from "./paperProtectedCore";
import {
  buildExperimentalMicroLaneDiagnostics,
  type ExperimentalMicroLaneDiagnostics,
} from "./paperExperimentalMicroDiagnostics";

/** Último ciclo de expansão do scanner upstream (antes de capacity/exploration). */
export type UpstreamScannerExpansionDiagnostics = {
  totalCandidatesGeneratedBeforeExpansion: number;
  totalCandidatesGeneratedAfterExpansion: number;
  additionalCandidatesIntroduced: number;
  newFamiliesIntroduced: number;
  newMarketBasesIntroduced: number;
  pairingsAddedByExpansion: number;
  candidatesDroppedByScannerBudget: number;
  scannerBudgetUsage: {
    maxExtraCandidates: number;
    usedExtraCandidates: number;
    maxGraphExtras: number;
    usedGraphExtras: number;
    maxPairings: number;
    usedPairings: number;
  };
  noveltyCoverageUsage: {
    graphNoveltyFirstPicks: number;
    crossPairNoveltyFirstPicks: number;
  };
};

/** Funil do pool bruto de grafo (último ciclo paper; actualizado em `applyUpstreamScannerExpansion`). */
export type UpstreamScannerRawFunnel = {
  cachedGraphRawCount: number;
  graphRawProbeCount: number;
  graphRawPassingMinFiltersCount: number;
  graphRawRejectedByLowConfidenceCount: number;
  graphRawRejectedByLowLiquidityCount: number;
  graphRawRejectedByAlreadyMergedCount: number;
  graphRawPreselectedCount: number;
  graphRawNormalizationAttemptsCount: number;
  graphRawNormalizationSuccessCount: number;
  graphRawNormalizationFailureCount: number;
  graphRawAcceptedAsExtraCount: number;
  graphRawDroppedByBudgetCount: number;
  /** Graph raw com `diagnosticRelationProvenance === complementary_relaxed` normalizados como extras (último ciclo). */
  graphRawRelaxedComplementaryAcceptedAsExtraCount: number;
};

/** Funil da expansão cross por pares (último ciclo). */
export type UpstreamScannerCrossFunnel = {
  marketsConsideredForCrossExpansion: number;
  marketsPassedToCoverageSort: number;
  marketsAfterCoverageOrdering: number;
  crossPairProbeWindowUsed: number;
  rawPairingsConsideredCount: number;
  rawPairingsRejectedByBinaryOutcomesCount: number;
  rawPairingsRejectedByLiquidityCount: number;
  rawPairingsRejectedByOverroundCount: number;
  rawPairingsRejectedByRecentPenaltyCount: number;
  rawPairingsRejectedByDuplicateOpportunityIdCount: number;
  rawPairingsAcceptedCount: number;
  crossPairingsDroppedByBudgetCount: number;
};

/** Universo de origem visível pela expansão (último ciclo). */
export type UpstreamScannerSourceUniverse = {
  totalMarketsAvailableForExpansion: number;
  uniqueMarketBasesAvailableForExpansion: number;
  uniqueFamiliesAvailableForExpansion: number;
  cachedGraphRawAvailable: boolean;
  /** Epoch ms do último scan de grafo, se disponível. */
  graphScanLastScanMs: number | null;
  /** Idade aproximada do cache de grafo bruto (ms). */
  cachedGraphRawAgeMs: number | null;
  /** Último scan completo (explica pool bruto vazio: 0 mercados vs 0 clusters vs 0 oportunidades). */
  graphScanCapture: {
    marketCount: number;
    clusterCount: number;
    rawOpportunityCount: number;
  } | null;
  scannerSourceSummary: string;
};

/** Máximo de amostras guardadas para percentis / buckets (só rejeições net-edge). */
const NET_EDGE_REJECTION_SAMPLE_CAP = 5_000;
const ECONOMIC_PASS_SAMPLE_CAP = 300;
const ECONOMIC_ROLLING_CAP = 300;
const ALL_EVALUATED_SCORES_CAP = 400;
const CLOSE_SCORE_CAP = 200;
const NO_PROGRESS_RECENT_CAP = 40;
/** Amostra recente por candidato avaliado na etapa económica (após simulador). */
const RECENT_EVALUATED_CANDIDATES_CAP = 50;
/** Anel de famílias upstream (diversidade) — mesmo padrão globalThis que o resto do diagnóstico. */
const UPSTREAM_FAMILY_RING_CAP = 256;
/** Geração recente no scanner (mercados / pares / famílias introduzidos na expansão). */
const SCANNER_COVERAGE_RING_CAP = 384;

const REFERENCE_NET_EDGE_THRESHOLDS = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008] as const;

const ECONOMIC_REASON_TOP: PaperOpenRejectionKey[] = [
  "PROGRESS_PROBABILITY_FACTOR_BELOW_MIN",
  "ENTRY_ECONOMIC_SCORE_BELOW_MIN",
  "EXPECTED_REALIZABLE_NET_PNL_BELOW_MIN",
  "EXPECTED_NET_PNL_BELOW_MIN",
  "FILLED_CAPITAL_BELOW_MIN",
  "EXPECTED_NET_MARGIN_BELOW_MIN",
  "GROSS_TO_FEES_RATIO_BELOW_MIN",
  "CROSS_MARKET_NET_TO_GROSS_EDGE_BELOW_MIN",
];

type NoProgressAfterEntrySample = {
  tradeId: string;
  profileKey: string;
  entryEconomicScoreAtOpen: number | null;
  progressProbabilityFactorAtOpen: number | null;
  realizedPnL: number;
};

/** Amostra legível por avaliação económica (entrada final). */
export type PaperRecentEconomicCandidateRecord = {
  evaluatedAt: string;
  opportunityId: string;
  /** Título / pergunta resumida do 1.º mercado, se existir. */
  marketLabel: string | null;
  sourceType: string;
  opportunityType: string;
  profileKey: string;
  confidence: number;
  fillProbability: number;
  capacityConfidence: number;
  recommendedCapital: number;
  requestedCapital: number;
  filledCapital: number;
  estimatedGrossEdge: number;
  estimatedNetEdge: number;
  grossEdgeAtEntry: number;
  netEdgeAtEntry: number;
  expectedGrossPnlUsd: number;
  expectedFeesUsd: number;
  expectedNetPnlToOpenUsd: number;
  expectedRealizableNetPnlUsd: number;
  expectedNetProfitMargin: number;
  grossToFeesRatio: number;
  liquiditySignal: number | null;
  headroomFactor: number | null;
  monetizationFactor: number | null;
  historicalNoProgressFactor: number | null;
  progressProbabilityFactor: number;
  progressProbabilityFactorEffectiveThreshold: number | null;
  entryEconomicScore: number;
  minEntryEconomicScoreEffective: number | null;
  netToGrossEdgeRatioAtEntry: number;
  minNetToGrossEdgeRatioCrossMarket: number | null;
  passedEconomicFilters: boolean;
  /** Mesma semântica que o motor: `pass` | `fail` | `filter_disabled_pass`. */
  finalEconomicDecision: "pass" | "fail" | "filter_disabled_pass";
  rejectionReasonFinal: PaperEntryEconomicRejectionReason | null;
  rejectionReasonsAll: PaperEntryEconomicRejectionReason[];
  crossMarketGuardApplicable: boolean;
  adaptiveProgressGuardApplied: boolean;
  adaptiveProfileStress: number | null;
  historicalNoProgressRate: number | null;
  historicalTakeProfitRate: number | null;
  economicEntryFilterEnabled: boolean;
};

/** Mesma linha que `recentEvaluatedCandidates`, deduplicada por `profileKey + opportunityId` (snapshot mais recente + metadados). */
export type PaperRecentUniqueEconomicCandidateRecord = PaperRecentEconomicCandidateRecord & {
  /** Chave estável: profileKey + U+001F + opportunityId (evita ambiguidade com `|` no perfil). */
  dedupeKey: string;
  timesSeenRecent: number;
  firstSeenAt: string;
  lastSeenAt: string;
  /** Chave agregada (`finalReasonAggregationKey`) na 1.ª ocorrência na janela. */
  firstFinalReason: string;
  /** Chave agregada na última ocorrência (estado actual do candidato). */
  lastFinalReason: string;
};

/** Cache derivado de `recentEvaluatedCandidates` (actualizado no record, não em cada GET). */
type EconomicRecentWindowDerived = {
  recentAgg: {
    avgProgressProbabilityFactorByFinalReason: Record<string, number | null>;
    avgEntryEconomicScoreByFinalReason: Record<string, number | null>;
    avgNetToGrossEdgeRatioByFinalReason: Record<string, number | null>;
    countByProfileAndFinalReason: Record<string, Record<string, number>>;
  };
  uniqueDiag: {
    recentUniqueEvaluatedCandidates: PaperRecentUniqueEconomicCandidateRecord[];
    uniqueCountByFinalReason: Record<string, number>;
    uniqueCountByProfileAndFinalReason: Record<string, Record<string, number>>;
    repeatedCandidateSummary: {
      totalRecentRows: number;
      totalUniqueCandidates: number;
      repeatedRows: number;
      repeatedRowsPct: number | null;
      candidatesSeenMoreThanOnce: number;
      maxTimesSeenSingleCandidate: number;
    };
    topRepeatedOpportunityIds: Array<{
      dedupeKey: string;
      opportunityId: string;
      profileKey: string;
      timesSeenRecent: number;
      firstSeenAt: string;
      lastSeenAt: string;
      lastFinalReason: string;
    }>;
  };
};

type Store = {
  preFilterMergedTotal: number;
  preFilterEnteringEngineTotal: number;
  engineCandidatesTotal: number;
  engineReachedPreSimulateTotal: number;
  engineOpenedTotal: number;
  rejectionCounts: Record<PaperOpenRejectionKey, number>;
  netEdgeRejectionSamples: number[];
  economicEntryPassSamples: PaperEntryEconomicsMetrics[];
  economicPassScores: number[];
  economicFailScores: number[];
  economicPassProgress: number[];
  economicFailProgress: number[];
  progressFactorBuckets: {
    lt02: number;
    b02_04: number;
    b04_06: number;
    gte06: number;
  };
  noProgressAfterEntryTotal: number;
  noProgressAfterEntryRecent: NoProgressAfterEntrySample[];
  /** Todos os scores observados na avaliação (passa ou falha), para min/max/avg. */
  economicAllEvaluatedScores: number[];
  entryEconomicScoreBuckets: {
    b0_002: number;
    b002_004: number;
    b004_008: number;
    b008_plus: number;
  };
  /** `entryEconomicScoreAtOpen` em fechos take_profit (amostra recente). */
  entryScoresAtCloseTakeProfit: number[];
  /** `entryEconomicScoreAtOpen` em fechos no_progress_exit (amostra recente). */
  entryScoresAtCloseNoProgress: number[];
  recentEvaluatedCandidates: PaperRecentEconomicCandidateRecord[];
  /**
   * Pré-computado em `recordPaperEconomicCandidateEvaluation` (caminho quente do HTTP não refaz agregação/dedup).
   */
  economicRecentWindowDerived: EconomicRecentWindowDerived | null;
  suppressedByCooldownCount: number;
  suppressedByCooldownByProfile: Record<string, number>;
  suppressedCandidatesRecent: Array<{ dedupeKey: string; profileKey: string; suppressedAt: string }>;
  explorationLastCycle: PaperExplorationLastCycle | null;
  upstreamFamilyRing: string[];
  upstreamFamilyCounts: Record<string, number>;
  upstreamSelectionLastCycle: UpstreamSelectionDiagnostics | null;
  scannerMarketBaseRing: string[];
  scannerMarketBaseCounts: Record<string, number>;
  scannerPairingRing: string[];
  scannerPairingCounts: Record<string, number>;
  scannerFamilyGenRing: string[];
  scannerFamilyGenCounts: Record<string, number>;
  upstreamScannerExpansionLastCycle: UpstreamScannerExpansionDiagnostics | null;
  upstreamScannerRawFunnelLastCycle: UpstreamScannerRawFunnel | null;
  upstreamScannerCrossFunnelLastCycle: UpstreamScannerCrossFunnel | null;
  upstreamScannerSourceUniverseLastCycle: UpstreamScannerSourceUniverse | null;
};

const SUPPRESS_COOLDOWN_SAMPLE_CAP = 20;

function emptyRejections(): Record<PaperOpenRejectionKey, number> {
  const r = {} as Record<PaperOpenRejectionKey, number>;
  for (const k of REJECTION_KEYS) r[k] = 0;
  return r;
}

function getStore(): Store {
  const g = globalThis as unknown as Record<string, Store>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = {
      preFilterMergedTotal: 0,
      preFilterEnteringEngineTotal: 0,
      engineCandidatesTotal: 0,
      engineReachedPreSimulateTotal: 0,
      engineOpenedTotal: 0,
      rejectionCounts: emptyRejections(),
      netEdgeRejectionSamples: [],
      economicEntryPassSamples: [],
      economicPassScores: [],
      economicFailScores: [],
      economicPassProgress: [],
      economicFailProgress: [],
      progressFactorBuckets: { lt02: 0, b02_04: 0, b04_06: 0, gte06: 0 },
      noProgressAfterEntryTotal: 0,
      noProgressAfterEntryRecent: [],
      economicAllEvaluatedScores: [],
      entryEconomicScoreBuckets: { b0_002: 0, b002_004: 0, b004_008: 0, b008_plus: 0 },
      entryScoresAtCloseTakeProfit: [],
      entryScoresAtCloseNoProgress: [],
      recentEvaluatedCandidates: [],
      economicRecentWindowDerived: null,
      suppressedByCooldownCount: 0,
      suppressedByCooldownByProfile: {},
      suppressedCandidatesRecent: [],
      explorationLastCycle: null,
      upstreamFamilyRing: [],
      upstreamFamilyCounts: {},
      upstreamSelectionLastCycle: null,
      scannerMarketBaseRing: [],
      scannerMarketBaseCounts: {},
      scannerPairingRing: [],
      scannerPairingCounts: {},
      scannerFamilyGenRing: [],
      scannerFamilyGenCounts: {},
      upstreamScannerExpansionLastCycle: null,
      upstreamScannerRawFunnelLastCycle: null,
      upstreamScannerCrossFunnelLastCycle: null,
      upstreamScannerSourceUniverseLastCycle: null,
    };
  }
  const st = g[GLOBAL_KEY];
  if (!st.upstreamFamilyRing) st.upstreamFamilyRing = [];
  if (!st.upstreamFamilyCounts) st.upstreamFamilyCounts = {};
  if (st.upstreamSelectionLastCycle === undefined) st.upstreamSelectionLastCycle = null;
  if (!st.scannerMarketBaseRing) st.scannerMarketBaseRing = [];
  if (!st.scannerMarketBaseCounts) st.scannerMarketBaseCounts = {};
  if (!st.scannerPairingRing) st.scannerPairingRing = [];
  if (!st.scannerPairingCounts) st.scannerPairingCounts = {};
  if (!st.scannerFamilyGenRing) st.scannerFamilyGenRing = [];
  if (!st.scannerFamilyGenCounts) st.scannerFamilyGenCounts = {};
  if (st.upstreamScannerExpansionLastCycle === undefined) st.upstreamScannerExpansionLastCycle = null;
  if (st.upstreamScannerRawFunnelLastCycle === undefined) st.upstreamScannerRawFunnelLastCycle = null;
  if (st.upstreamScannerCrossFunnelLastCycle === undefined) st.upstreamScannerCrossFunnelLastCycle = null;
  if (st.upstreamScannerSourceUniverseLastCycle === undefined) st.upstreamScannerSourceUniverseLastCycle = null;
  return st;
}

function ringPushUpstreamFamily(familyKey: string): void {
  const s = getStore();
  if (s.upstreamFamilyRing.length >= UPSTREAM_FAMILY_RING_CAP) {
    const old = s.upstreamFamilyRing.shift()!;
    const c = (s.upstreamFamilyCounts[old] ?? 1) - 1;
    if (c <= 0) delete s.upstreamFamilyCounts[old];
    else s.upstreamFamilyCounts[old] = c;
  }
  s.upstreamFamilyRing.push(familyKey);
  s.upstreamFamilyCounts[familyKey] = (s.upstreamFamilyCounts[familyKey] ?? 0) + 1;
}

/** Contagem na janela recente (O(1)); usada por `applyUpstreamDiversitySelection` antes de cada batch. */
export function getFamilyRecentCount(familyKey: string): number {
  return getStore().upstreamFamilyCounts[familyKey] ?? 0;
}

/** Um tick por candidate no motor (ordem pós-diversidade). */
export function recordPaperUpstreamFamilySeen(familyKey: string): void {
  ringPushUpstreamFamily(familyKey);
}

export function setUpstreamSelectionLastCycle(d: UpstreamSelectionDiagnostics | null): void {
  getStore().upstreamSelectionLastCycle = d;
}

function ringPushScannerMarketBase(id: string): void {
  const s = getStore();
  if (s.scannerMarketBaseRing.length >= SCANNER_COVERAGE_RING_CAP) {
    const old = s.scannerMarketBaseRing.shift()!;
    const c = (s.scannerMarketBaseCounts[old] ?? 1) - 1;
    if (c <= 0) delete s.scannerMarketBaseCounts[old];
    else s.scannerMarketBaseCounts[old] = c;
  }
  s.scannerMarketBaseRing.push(id);
  s.scannerMarketBaseCounts[id] = (s.scannerMarketBaseCounts[id] ?? 0) + 1;
}

function ringPushScannerPairing(pk: string): void {
  const s = getStore();
  if (s.scannerPairingRing.length >= SCANNER_COVERAGE_RING_CAP) {
    const old = s.scannerPairingRing.shift()!;
    const c = (s.scannerPairingCounts[old] ?? 1) - 1;
    if (c <= 0) delete s.scannerPairingCounts[old];
    else s.scannerPairingCounts[old] = c;
  }
  s.scannerPairingRing.push(pk);
  s.scannerPairingCounts[pk] = (s.scannerPairingCounts[pk] ?? 0) + 1;
}

function ringPushScannerFamilyGen(fk: string): void {
  const s = getStore();
  if (s.scannerFamilyGenRing.length >= SCANNER_COVERAGE_RING_CAP) {
    const old = s.scannerFamilyGenRing.shift()!;
    const c = (s.scannerFamilyGenCounts[old] ?? 1) - 1;
    if (c <= 0) delete s.scannerFamilyGenCounts[old];
    else s.scannerFamilyGenCounts[old] = c;
  }
  s.scannerFamilyGenRing.push(fk);
  s.scannerFamilyGenCounts[fk] = (s.scannerFamilyGenCounts[fk] ?? 0) + 1;
}

export function getScannerMarketBaseRecentCount(marketId: string): number {
  return getStore().scannerMarketBaseCounts[marketId] ?? 0;
}

export function getScannerPairingRecentCount(pairingKey: string): number {
  return getStore().scannerPairingCounts[pairingKey] ?? 0;
}

export function getScannerFamilyGenRecentCount(familyKey: string): number {
  return getStore().scannerFamilyGenCounts[familyKey] ?? 0;
}

/** Regista cobertura para candidatos **novos** introduzidos pela expansão (incremental). */
export function recordScannerExpansionCoverageForOpportunities(opps: NormalizedPaperOpportunity[]): void {
  for (const opp of opps) {
    ringPushScannerFamilyGen(getOpportunityFamilyKey(opp));
    const mids = opp.marketsInvolved.map((m) => m.marketId).filter(Boolean);
    for (const mid of mids) ringPushScannerMarketBase(mid);
    if (mids.length === 2) {
      const [a, b] = [...mids].sort();
      ringPushScannerPairing(`p:${a}|${b}`);
    }
  }
}

export function setUpstreamScannerExpansionLastCycle(d: UpstreamScannerExpansionDiagnostics | null): void {
  getStore().upstreamScannerExpansionLastCycle = d;
}

export function setUpstreamScannerFunnelSnapshots(args: {
  raw: UpstreamScannerRawFunnel;
  cross: UpstreamScannerCrossFunnel;
  source: UpstreamScannerSourceUniverse;
}): void {
  const s = getStore();
  s.upstreamScannerRawFunnelLastCycle = args.raw;
  s.upstreamScannerCrossFunnelLastCycle = args.cross;
  s.upstreamScannerSourceUniverseLastCycle = args.source;
}

export function getUpstreamScannerRawFunnelLastCycle(): UpstreamScannerRawFunnel | null {
  return getStore().upstreamScannerRawFunnelLastCycle ?? null;
}

/** Contagens por dedupeKey na janela `recentEvaluatedCandidates` (O(cap); uso no sort por ciclo). */
export function getEconomicDedupeCountsFromRecentBuffer(): Map<string, number> {
  const m = new Map<string, number>();
  for (const r of getStore().recentEvaluatedCandidates) {
    const dk = `${r.profileKey}\u001f${r.opportunityId}`;
    m.set(dk, (m.get(dk) ?? 0) + 1);
  }
  return m;
}

export function recordPaperCooldownSuppress(profileKey: string, dedupeKey: string): void {
  const s = getStore();
  s.suppressedByCooldownCount += 1;
  if (!s.suppressedByCooldownByProfile[profileKey]) s.suppressedByCooldownByProfile[profileKey] = 0;
  s.suppressedByCooldownByProfile[profileKey] += 1;
  s.suppressedCandidatesRecent.push({
    dedupeKey,
    profileKey,
    suppressedAt: new Date().toISOString(),
  });
  while (s.suppressedCandidatesRecent.length > SUPPRESS_COOLDOWN_SAMPLE_CAP) {
    s.suppressedCandidatesRecent.shift();
  }
}

export function recordPaperExplorationLastCycle(c: PaperExplorationLastCycle): void {
  getStore().explorationLastCycle = c;
}

export function recordPaperPreFilterBatch(mergedLen: number, enteringEngineLen: number): void {
  const s = getStore();
  s.preFilterMergedTotal += mergedLen;
  s.preFilterEnteringEngineTotal += enteringEngineLen;
}

export function recordPaperEngineBatch(candidateCount: number): void {
  getStore().engineCandidatesTotal += candidateCount;
}

export function bumpPaperOpenRejection(reason: PaperOpenRejectionKey): void {
  getStore().rejectionCounts[reason] += 1;
}

export function recordPaperNetEdgeThresholdRejection(estimatedNetEdge: number): void {
  const s = getStore();
  const v = Number(estimatedNetEdge);
  if (!Number.isFinite(v)) return;
  s.netEdgeRejectionSamples.push(v);
  while (s.netEdgeRejectionSamples.length > NET_EDGE_REJECTION_SAMPLE_CAP) {
    s.netEdgeRejectionSamples.shift();
  }
}

export function bumpPaperReachedPreSimulate(): void {
  getStore().engineReachedPreSimulateTotal += 1;
}

export function bumpPaperOpened(): void {
  getStore().engineOpenedTotal += 1;
}

function pushCap(arr: number[], v: number, cap: number): void {
  arr.push(v);
  while (arr.length > cap) arr.shift();
}

function bumpProgressBucket(p: number): void {
  const b = getStore().progressFactorBuckets;
  if (p < 0.2) b.lt02 += 1;
  else if (p < 0.4) b.b02_04 += 1;
  else if (p < 0.6) b.b04_06 += 1;
  else b.gte06 += 1;
}

function bumpEntryEconomicScoreBucket(score: number): void {
  const b = getStore().entryEconomicScoreBuckets;
  if (score < 0.002) b.b0_002 += 1;
  else if (score < 0.004) b.b002_004 += 1;
  else if (score < 0.008) b.b004_008 += 1;
  else b.b008_plus += 1;
}

/** Amostras de avaliação económica pós-simulação (passou ou falhou o filtro). */
export function recordPaperEconomicEntryOutcome(metrics: PaperEntryEconomicsMetrics, passed: boolean): void {
  const s = getStore();
  bumpProgressBucket(metrics.progressProbabilityFactor);
  pushCap(s.economicAllEvaluatedScores, metrics.entryEconomicScore, ALL_EVALUATED_SCORES_CAP);
  bumpEntryEconomicScoreBucket(metrics.entryEconomicScore);
  if (passed) {
    s.economicEntryPassSamples.push(metrics);
    while (s.economicEntryPassSamples.length > ECONOMIC_PASS_SAMPLE_CAP) {
      s.economicEntryPassSamples.shift();
    }
    pushCap(s.economicPassScores, metrics.entryEconomicScore, ECONOMIC_ROLLING_CAP);
    pushCap(s.economicPassProgress, metrics.progressProbabilityFactor, ECONOMIC_ROLLING_CAP);
  } else {
    pushCap(s.economicFailScores, metrics.entryEconomicScore, ECONOMIC_ROLLING_CAP);
    pushCap(s.economicFailProgress, metrics.progressProbabilityFactor, ECONOMIC_ROLLING_CAP);
  }
}

/** Após fecho: compara score à entrada vs tipo de saída (take_profit vs no_progress). */
export function recordPaperEntryEconomicScoreAtClose(
  exitCondition: ExitCondition,
  entryEconomicScoreAtOpen: number | null | undefined
): void {
  const v = entryEconomicScoreAtOpen;
  if (v == null || !Number.isFinite(v)) return;
  const s = getStore();
  if (exitCondition === "take_profit") {
    pushCap(s.entryScoresAtCloseTakeProfit, v, CLOSE_SCORE_CAP);
  } else if (exitCondition === "no_progress_exit") {
    pushCap(s.entryScoresAtCloseNoProgress, v, CLOSE_SCORE_CAP);
  }
}

/** Regista um candidato avaliado na etapa económica (após `simulateEntry`). */
export function recordPaperEconomicCandidateEvaluation(row: PaperRecentEconomicCandidateRecord): void {
  const s = getStore();
  s.recentEvaluatedCandidates.push(row);
  while (s.recentEvaluatedCandidates.length > RECENT_EVALUATED_CANDIDATES_CAP) {
    s.recentEvaluatedCandidates.shift();
  }
  refreshEconomicRecentWindowDerived(s);
}

/** Trade que entrou e fechou em `no_progress_exit` (amostra recente para auditoria). */
export function recordPaperOpenedThenNoProgressExit(sample: NoProgressAfterEntrySample): void {
  const s = getStore();
  s.noProgressAfterEntryTotal += 1;
  s.noProgressAfterEntryRecent.push(sample);
  while (s.noProgressAfterEntryRecent.length > NO_PROGRESS_RECENT_CAP) {
    s.noProgressAfterEntryRecent.shift();
  }
}

export type PaperOpenDiagnosticsSnapshot = {
  preFilter: {
    mergedOpportunitiesTotal: number;
    enteringEngineTotal: number;
    droppedRecommendedCapitalLeZero: number;
    /** Não comparar directamente com `count` das rotas /api/opportunities ou /api/graph-opportunities. */
    cumulativeSemanticsNote: string;
  };
  engine: {
    candidatesEvaluatedTotal: number;
    reachedSimulateEntryTotal: number;
    openedTotal: number;
    rejectionsByReason: Record<PaperOpenRejectionKey, number>;
    rejectionTotal: number;
  };
  proportions: {
    ofEngineCandidates: Record<PaperOpenRejectionKey | "OPENED", number>;
    ofRejectionEvents: Record<PaperOpenRejectionKey, number>;
  };
  funnel: {
    stage0_mergedAllSources: number;
    stage1_afterPreFilterRecommendedCapPositive: number;
    stage2_afterNetEdgeThreshold: number;
    stage3_afterConfidenceThreshold: number;
    stage4_afterNoActiveDuplicate: number;
    stage5_afterExposureLimits: number;
    stage6_afterRequestedCapitalPositive: number;
    stage7_afterSimulatorFill: number;
    stage8_afterEconomicEntryFilters: number;
  };
  netEdgeThresholdRejections: {
    sampleCount: number;
    sampleCap: number;
    min: number | null;
    max: number | null;
    avg: number | null;
    p50: number | null;
    p90: number | null;
    p99: number | null;
    estimatedNetEdgeGteReference: Record<string, number>;
  };
  economicEntryFilter: {
    enabled: boolean;
    thresholds: ReturnType<typeof getPaperEntryEconomicFilterConfig>;
    passThrough: {
      sampleCount: number;
      avgExpectedNetPnlToOpenUsd: number | null;
      avgExpectedRealizableNetPnlUsd: number | null;
      avgFilledCapital: number | null;
      avgExpectedNetProfitMargin: number | null;
      avgGrossToFeesRatio: number | null;
      avgProgressProbabilityFactor: number | null;
      avgEntryEconomicScore: number | null;
      avgEffectiveMinProgressProbabilityFactor: number | null;
    };
  };
  economicEntryDecision: {
    avgEntryEconomicScorePass: number | null;
    avgEntryEconomicScoreFail: number | null;
    avgProgressProbabilityFactorPass: number | null;
    avgProgressProbabilityFactorFail: number | null;
    /** Rolling: todos os scores de candidatos avaliados (filtro activo). */
    recentEntryEconomicScoreEvaluated: {
      sampleCount: number;
      sampleCap: number;
      min: number | null;
      max: number | null;
      avg: number | null;
    };
    entryEconomicScoreBuckets: Store["entryEconomicScoreBuckets"];
    progressFactorBuckets: Store["progressFactorBuckets"];
    topEconomicRejectionReasons: Array<{ reason: PaperOpenRejectionKey; count: number }>;
    /** Scores à entrada vs saída (amostras recentes). */
    entryScoreAtOpenVsExit: {
      takeProfit: { n: number; avg: number | null; min: number | null; max: number | null };
      noProgressExit: { n: number; avg: number | null; min: number | null; max: number | null };
    };
    openedThenNoProgressExit: {
      total: number;
      recentSamples: NoProgressAfterEntrySample[];
      recentCap: number;
    };
    exitOutcomeByProfile: ReturnType<typeof getPaperExitProfileMemorySnapshot>;
    /** Guard mínimo de progress por perfil (global + bump adaptativo). */
    effectiveProgressGuardByProfile: ReturnType<typeof buildEffectiveProgressGuardByProfileMap>;
    profileAdaptiveDecisionSummary: {
      enabled: boolean;
      globalMin: number;
      extraMax: number;
      minSamples: number;
    };
    /** Segundo sinal `standard|cross_market`: ratio net/gross no capacity. */
    crossMarketNetGrossEntryGuard: {
      enabled: boolean;
      profileKey: string;
      minNetToGrossRatio: number;
      avgNetToGrossAmongPassesSample: number | null;
      passSampleCountCrossMarket: number;
    };
    /** Últimos candidatos avaliados na etapa económica (memória). */
    recentEvaluatedCandidates: PaperRecentEconomicCandidateRecord[];
    recentEvaluatedCandidatesCap: number;
    /** Agregados só sobre `recentEvaluatedCandidates` (janela recente). */
    avgProgressProbabilityFactorByFinalReason: Record<string, number | null>;
    avgEntryEconomicScoreByFinalReason: Record<string, number | null>;
    avgNetToGrossEdgeRatioByFinalReason: Record<string, number | null>;
    countByProfileAndFinalReason: Record<string, Record<string, number>>;
    /** Janela recente deduplicada por `profileKey + opportunityId` (derivada de `recentEvaluatedCandidates`). */
    recentUniqueEvaluatedCandidates: PaperRecentUniqueEconomicCandidateRecord[];
    /** Contagens por motivo final usando só o último estado de cada candidato único (dedupe). */
    uniqueCountByFinalReason: Record<string, number>;
    uniqueCountByProfileAndFinalReason: Record<string, Record<string, number>>;
    repeatedCandidateSummary: {
      totalRecentRows: number;
      totalUniqueCandidates: number;
      repeatedRows: number;
      /** Percentagem 0–100 das linhas que são reavaliações (duplicadas). */
      repeatedRowsPct: number | null;
      candidatesSeenMoreThanOnce: number;
      maxTimesSeenSingleCandidate: number;
    };
    /** Candidatos com mais de uma linha na janela (top 10 por `timesSeenRecent`). */
    topRepeatedOpportunityIds: Array<{
      dedupeKey: string;
      opportunityId: string;
      profileKey: string;
      timesSeenRecent: number;
      firstSeenAt: string;
      lastSeenAt: string;
      lastFinalReason: string;
    }>;
  };
  economicCooldown: {
    cooldownPolicySnapshot: ReturnType<typeof getEconomicCooldownPolicySnapshot>;
    activeCooldownSummary: ReturnType<typeof getEconomicCooldownActiveSummary>;
    suppressedByCooldownCount: number;
    suppressedByCooldownByProfile: Record<string, number>;
    suppressedCandidatesRecent: Array<{ dedupeKey: string; profileKey: string; suppressedAt: string }>;
  };
  exploration: {
    explorationPolicySnapshot: ReturnType<typeof getExplorationPolicySnapshot>;
    recentNoveltyStats: {
      batchSize: number | null;
      distinctDedupeKeysInBatch: number | null;
      avgRepeatCountInBatch: number | null;
    } | null;
    evaluatedUniqueCandidatesCount: number;
    evaluationPriorityDiagnostics: PaperExplorationLastCycle["evaluationPriorityDiagnostics"] | null;
    repeatedCandidatePenaltyAppliedCount: number;
  };
  upstreamExploration: {
    policySnapshot: ReturnType<typeof getUpstreamDiversityPolicySnapshot>;
    recentCoverageStats: {
      ringCapacity: number;
      ringUsed: number;
    };
    recentClusterStats: {
      clusterFamilyKeysRecent: number;
      marketBaseFamilyKeysRecent: number;
      opportunityFallbackFamilyKeysRecent: number;
    };
    recentFamilyConcentration: number | null;
    noveltyBudgetUsage: { reservedSlots: number; usedSlots: number } | null;
    diversityBudgetUsage: {
      maxPerFamilyInBatch: number;
      impliedAverageDominance: number | null;
    };
    averageClusterDominance: number | null;
    uniqueFamiliesSeenRecent: number;
    uniqueMarketBasesSeenRecent: number;
  };
  upstreamSelectionDiagnostics: UpstreamSelectionDiagnostics | null;
  upstreamScannerCoverage: {
    policySnapshot: ReturnType<typeof getUpstreamScannerExpansionPolicySnapshot>;
    recentScannerCoverageStats: {
      ringCapacity: number;
      marketBaseRingUsed: number;
      pairingRingUsed: number;
      familyGenRingUsed: number;
    };
    recentMarketBaseCoverage: { uniqueKeys: number; concentration: number | null };
    recentPairingCoverage: { uniqueKeys: number; concentration: number | null };
    recentFamilyCoverage: { uniqueKeys: number; concentration: number | null };
    averageCoverageBreadth: number | null;
    uniqueMarketBasesGeneratedRecent: number;
    uniqueFamiliesGeneratedRecent: number;
    uniquePairingsGeneratedRecent: number;
  };
  upstreamScannerDiagnostics: UpstreamScannerExpansionDiagnostics | null;
  upstreamScannerRawFunnel: UpstreamScannerRawFunnel | null;
  upstreamScannerCrossFunnel: UpstreamScannerCrossFunnel | null;
  upstreamScannerSourceUniverse: UpstreamScannerSourceUniverse | null;
  /** Último snapshot do pipeline grafo (relation → cluster → scan); custo O(1), preenchido no graph scan. */
  graphPipelineDiagnosticsCapturedAtMs: number | null;
  graphRelationDiagnostics: RelationBuilderFunnelSnapshot | null;
  graphClusterDiagnostics: ClusterFormationFunnelSnapshot | null;
  graphSourceQualityDiagnostics: GraphSourceQualitySnapshot | null;
  /** Sobrevivência paper por proveniência do grafo; actualizado no fim de cada ciclo paper. */
  graphOpportunityDownstreamImpact: GraphOpportunityDownstreamImpactSnapshot;
  complementaryRelaxedPaperImpact: ComplementaryRelaxedPaperImpactSnapshot;
  graphProvenancePropagationDiagnostics: GraphProvenancePropagationDiagnostics;
  /** Auditoria O(n) sobre fechados graph: distribuição PnL, concentração, exits, entrada. */
  graphProvenanceQualityAudit: GraphProvenanceQualityAudit;
  complementaryRelaxedQualityAudit: ComplementaryRelaxedQualityAudit;
  /** Diversidade estrutural / robustez por cluster, label e janela temporal (complementary_relaxed). */
  complementaryRelaxedStructuralRobustness: ComplementaryRelaxedStructuralRobustness;
  /** Diversidade intra-cluster: tokens, skeletons, buckets temáticos (complementary_relaxed). */
  complementaryRelaxedIntraClusterAudit: ComplementaryRelaxedIntraClusterAudit;
  /** Contagens por fonte (store vs snapshot API) para detectar drift entre rotas. */
  closedTradesSourceDiagnostics: ClosedTradesSourceDiagnostics;
  /** Por que o universo da quality audit coincide (ou não) com propagação / PnL finito. */
  graphQualityAuditSourceDiagnostics: GraphQualityAuditSourceDiagnostics;
  /** Carga média de taxas e trades bruto+ / líquido− por proveniência (fechados graph). */
  feeImpactAudit: FeeImpactAudit;
  /** Fórmula USD / preço implícito: realizedPnL vs filledCapital (amostras + agregados por proveniência). */
  paperEconomicUnitAudit: PaperEconomicUnitAudit;
  /** Modelo de saída: buckets de exitPrice, heurísticas de fonte, foco cycle (sem alterar PnL). */
  paperExitModelAudit: PaperExitModelAudit;
  /** Protected core invariants, operational safety gates, experimental lane status. */
  protectedCoreDiagnostics: ProtectedCoreDiagnostics;
  /** Micro-lanes experimentais isoladas (reformulação de classes bloqueadas). */
  experimentalMicroLanes: ExperimentalMicroLaneDiagnostics;
};

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

/** Deduplicação diagnóstica: mesmo par perfil+oportunidade (ids podem repetir-se noutro perfil teoricamente). */
function economicCandidateDedupeKey(r: PaperRecentEconomicCandidateRecord): string {
  return `${r.profileKey}\u001f${r.opportunityId}`;
}

function percentile(sorted: number[], p: number): number | null {
  if (sorted.length === 0) return null;
  const n = sorted.length;
  const idx = Math.min(n - 1, Math.max(0, Math.ceil(p * n) - 1));
  return round4(sorted[idx]!);
}

function avg(arr: number[]): number | null {
  if (arr.length === 0) return null;
  const sum = arr.reduce((a, b) => a + b, 0);
  return round4(sum / arr.length);
}

function minMaxAvg(arr: number[]): { min: number | null; max: number | null; avg: number | null } {
  if (arr.length === 0) return { min: null, max: null, avg: null };
  const sorted = [...arr].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    min: round4(sorted[0]!),
    max: round4(sorted[sorted.length - 1]!),
    avg: round4(sum / sorted.length),
  };
}

function finalReasonAggregationKey(r: PaperRecentEconomicCandidateRecord): string {
  if (r.finalEconomicDecision === "filter_disabled_pass") return "FILTER_DISABLED_PASS";
  if (r.finalEconomicDecision === "pass") return "PASS";
  return r.rejectionReasonFinal ?? "FAIL_UNKNOWN";
}

function buildRecentEconomicAggregates(samples: PaperRecentEconomicCandidateRecord[]): {
  avgProgressProbabilityFactorByFinalReason: Record<string, number | null>;
  avgEntryEconomicScoreByFinalReason: Record<string, number | null>;
  avgNetToGrossEdgeRatioByFinalReason: Record<string, number | null>;
  countByProfileAndFinalReason: Record<string, Record<string, number>>;
} {
  const byReason = new Map<string, PaperRecentEconomicCandidateRecord[]>();
  const countByProfileAndFinalReason: Record<string, Record<string, number>> = {};
  for (const s of samples) {
    const rk = finalReasonAggregationKey(s);
    let arr = byReason.get(rk);
    if (!arr) {
      arr = [];
      byReason.set(rk, arr);
    }
    arr.push(s);

    if (!countByProfileAndFinalReason[s.profileKey]) countByProfileAndFinalReason[s.profileKey] = {};
    const pr = countByProfileAndFinalReason[s.profileKey]!;
    pr[rk] = (pr[rk] ?? 0) + 1;
  }

  const avgProgressProbabilityFactorByFinalReason: Record<string, number | null> = {};
  const avgEntryEconomicScoreByFinalReason: Record<string, number | null> = {};
  const avgNetToGrossEdgeRatioByFinalReason: Record<string, number | null> = {};

  for (const [k, arr] of Array.from(byReason.entries())) {
    if (arr.length === 0) continue;
    const sumP = arr.reduce((a, x) => a + x.progressProbabilityFactor, 0);
    const sumS = arr.reduce((a, x) => a + x.entryEconomicScore, 0);
    const sumNg = arr.reduce((a, x) => a + x.netToGrossEdgeRatioAtEntry, 0);
    avgProgressProbabilityFactorByFinalReason[k] = round4(sumP / arr.length);
    avgEntryEconomicScoreByFinalReason[k] = round4(sumS / arr.length);
    avgNetToGrossEdgeRatioByFinalReason[k] = round4(sumNg / arr.length);
  }

  return {
    avgProgressProbabilityFactorByFinalReason,
    avgEntryEconomicScoreByFinalReason,
    avgNetToGrossEdgeRatioByFinalReason,
    countByProfileAndFinalReason,
  };
}

function buildUniqueRecentEconomicDiagnostics(samples: PaperRecentEconomicCandidateRecord[]): {
  recentUniqueEvaluatedCandidates: PaperRecentUniqueEconomicCandidateRecord[];
  uniqueCountByFinalReason: Record<string, number>;
  uniqueCountByProfileAndFinalReason: Record<string, Record<string, number>>;
  repeatedCandidateSummary: {
    totalRecentRows: number;
    totalUniqueCandidates: number;
    repeatedRows: number;
    repeatedRowsPct: number | null;
    candidatesSeenMoreThanOnce: number;
    maxTimesSeenSingleCandidate: number;
  };
  topRepeatedOpportunityIds: Array<{
    dedupeKey: string;
    opportunityId: string;
    profileKey: string;
    timesSeenRecent: number;
    firstSeenAt: string;
    lastSeenAt: string;
    lastFinalReason: string;
  }>;
} {
  const byDedupe = new Map<string, PaperRecentEconomicCandidateRecord[]>();
  for (const row of samples) {
    const k = economicCandidateDedupeKey(row);
    let arr = byDedupe.get(k);
    if (!arr) {
      arr = [];
      byDedupe.set(k, arr);
    }
    arr.push(row);
  }

  const recentUniqueEvaluatedCandidates: PaperRecentUniqueEconomicCandidateRecord[] = [];
  const uniqueCountByFinalReason: Record<string, number> = {};
  const uniqueCountByProfileAndFinalReason: Record<string, Record<string, number>> = {};

  for (const [, arr] of Array.from(byDedupe.entries())) {
    if (arr.length === 0) continue;
    const first = arr[0]!;
    const last = arr[arr.length - 1]!;
    const firstFr = finalReasonAggregationKey(first);
    const lastFr = finalReasonAggregationKey(last);
    const dk = economicCandidateDedupeKey(last);
    recentUniqueEvaluatedCandidates.push({
      ...last,
      dedupeKey: dk,
      timesSeenRecent: arr.length,
      firstSeenAt: first.evaluatedAt,
      lastSeenAt: last.evaluatedAt,
      firstFinalReason: firstFr,
      lastFinalReason: lastFr,
    });

    uniqueCountByFinalReason[lastFr] = (uniqueCountByFinalReason[lastFr] ?? 0) + 1;
    if (!uniqueCountByProfileAndFinalReason[last.profileKey]) {
      uniqueCountByProfileAndFinalReason[last.profileKey] = {};
    }
    const pr = uniqueCountByProfileAndFinalReason[last.profileKey]!;
    pr[lastFr] = (pr[lastFr] ?? 0) + 1;
  }

  recentUniqueEvaluatedCandidates.sort((a, b) => (a.lastSeenAt < b.lastSeenAt ? 1 : a.lastSeenAt > b.lastSeenAt ? -1 : 0));

  const totalRecentRows = samples.length;
  const totalUniqueCandidates = recentUniqueEvaluatedCandidates.length;
  const repeatedRows = Math.max(0, totalRecentRows - totalUniqueCandidates);
  const repeatedRowsPct =
    totalRecentRows > 0 ? round2((repeatedRows / totalRecentRows) * 100) : null;
  const candidatesSeenMoreThanOnce = recentUniqueEvaluatedCandidates.filter((r) => r.timesSeenRecent > 1).length;
  const maxTimesSeenSingleCandidate =
    recentUniqueEvaluatedCandidates.length > 0
      ? Math.max(...recentUniqueEvaluatedCandidates.map((r) => r.timesSeenRecent))
      : 0;

  const topRepeatedOpportunityIds = [...recentUniqueEvaluatedCandidates]
    .filter((r) => r.timesSeenRecent > 1)
    .sort((a, b) => b.timesSeenRecent - a.timesSeenRecent)
    .slice(0, 10)
    .map((r) => ({
      dedupeKey: r.dedupeKey,
      opportunityId: r.opportunityId,
      profileKey: r.profileKey,
      timesSeenRecent: r.timesSeenRecent,
      firstSeenAt: r.firstSeenAt,
      lastSeenAt: r.lastSeenAt,
      lastFinalReason: r.lastFinalReason,
    }));

  return {
    recentUniqueEvaluatedCandidates,
    uniqueCountByFinalReason,
    uniqueCountByProfileAndFinalReason,
    repeatedCandidateSummary: {
      totalRecentRows,
      totalUniqueCandidates,
      repeatedRows,
      repeatedRowsPct,
      candidatesSeenMoreThanOnce,
      maxTimesSeenSingleCandidate,
    },
    topRepeatedOpportunityIds,
  };
}

/** Actualiza cache derivado; chamado no record (não em cada GET). */
function refreshEconomicRecentWindowDerived(s: Store): void {
  const samples = s.recentEvaluatedCandidates;
  s.economicRecentWindowDerived = {
    recentAgg: buildRecentEconomicAggregates(samples),
    uniqueDiag: buildUniqueRecentEconomicDiagnostics(samples),
  };
}

function herfindahlFromCounts(counts: Record<string, number>, total: number): number | null {
  if (total <= 0) return null;
  let h = 0;
  for (const c of Object.values(counts)) {
    const p = c / total;
    h += p * p;
  }
  return round4(h);
}

function buildUpstreamExplorationSnapshot(st: Store): PaperOpenDiagnosticsSnapshot["upstreamExploration"] {
  const policySnapshot = getUpstreamDiversityPolicySnapshot();
  const ring = st.upstreamFamilyRing;
  const ringUsed = ring.length;
  const counts = st.upstreamFamilyCounts;
  let clusterN = 0;
  let marketBases = 0;
  let oppN = 0;
  for (const k of Object.keys(counts)) {
    if (k.startsWith("c:")) clusterN++;
    else if (k.startsWith("m:")) marketBases++;
    else if (k.startsWith("o:")) oppN++;
  }
  const uniqueFamilies = Object.keys(counts).length;
  let maxC = 0;
  for (const c of Object.values(counts)) maxC = Math.max(maxC, c);
  const impliedAvgDom = ringUsed > 0 ? round4(maxC / ringUsed) : null;
  const last = st.upstreamSelectionLastCycle;
  const noveltyBudgetUsage =
    last != null
      ? { reservedSlots: last.slotsReservedForNovelty, usedSlots: last.candidatesPromotedForNovelty }
      : null;

  return {
    policySnapshot,
    recentCoverageStats: {
      ringCapacity: UPSTREAM_FAMILY_RING_CAP,
      ringUsed,
    },
    recentClusterStats: {
      clusterFamilyKeysRecent: clusterN,
      marketBaseFamilyKeysRecent: marketBases,
      opportunityFallbackFamilyKeysRecent: oppN,
    },
    recentFamilyConcentration: herfindahlFromCounts(counts, ringUsed),
    noveltyBudgetUsage,
    diversityBudgetUsage: {
      maxPerFamilyInBatch: policySnapshot.maxPerFamilyInBatch,
      impliedAverageDominance: impliedAvgDom,
    },
    averageClusterDominance: impliedAvgDom,
    uniqueFamiliesSeenRecent: uniqueFamilies,
    uniqueMarketBasesSeenRecent: marketBases,
  };
}

function scannerHerfindahl(counts: Record<string, number>, total: number): number | null {
  if (total <= 0) return null;
  let h = 0;
  for (const c of Object.values(counts)) {
    const p = c / total;
    h += p * p;
  }
  return round4(h);
}

function buildUpstreamScannerCoverageSnapshot(st: Store): PaperOpenDiagnosticsSnapshot["upstreamScannerCoverage"] {
  const policySnapshot = getUpstreamScannerExpansionPolicySnapshot();
  const mb = st.scannerMarketBaseRing.length;
  const pk = st.scannerPairingRing.length;
  const fg = st.scannerFamilyGenRing.length;
  const umb = Object.keys(st.scannerMarketBaseCounts).length;
  const upk = Object.keys(st.scannerPairingCounts).length;
  const ufg = Object.keys(st.scannerFamilyGenCounts).length;
  const hM = scannerHerfindahl(st.scannerMarketBaseCounts, mb);
  const hP = scannerHerfindahl(st.scannerPairingCounts, pk);
  const hF = scannerHerfindahl(st.scannerFamilyGenCounts, fg);
  const breadths = [hM, hP, hF].filter((x): x is number => x != null);
  const avgBreadth =
    breadths.length > 0 ? round4(breadths.reduce((a, b) => a + b, 0) / breadths.length) : null;

  return {
    policySnapshot,
    recentScannerCoverageStats: {
      ringCapacity: SCANNER_COVERAGE_RING_CAP,
      marketBaseRingUsed: mb,
      pairingRingUsed: pk,
      familyGenRingUsed: fg,
    },
    recentMarketBaseCoverage: { uniqueKeys: umb, concentration: hM },
    recentPairingCoverage: { uniqueKeys: upk, concentration: hP },
    recentFamilyCoverage: { uniqueKeys: ufg, concentration: hF },
    averageCoverageBreadth: avgBreadth,
    uniqueMarketBasesGeneratedRecent: umb,
    uniqueFamiliesGeneratedRecent: ufg,
    uniquePairingsGeneratedRecent: upk,
  };
}

function buildNetEdgeRejectionStats(samples: number[]): PaperOpenDiagnosticsSnapshot["netEdgeThresholdRejections"] {
  const n = samples.length;
  const estimatedNetEdgeGteReference: Record<string, number> = {};
  for (const t of REFERENCE_NET_EDGE_THRESHOLDS) {
    const key = String(t);
    estimatedNetEdgeGteReference[key] = samples.filter((v) => v >= t).length;
  }
  if (n === 0) {
    return {
      sampleCount: 0,
      sampleCap: NET_EDGE_REJECTION_SAMPLE_CAP,
      min: null,
      max: null,
      avg: null,
      p50: null,
      p90: null,
      p99: null,
      estimatedNetEdgeGteReference,
    };
  }
  const sorted = [...samples].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    sampleCount: n,
    sampleCap: NET_EDGE_REJECTION_SAMPLE_CAP,
    min: round4(sorted[0]!),
    max: round4(sorted[n - 1]!),
    avg: round4(sum / n),
    p50: percentile(sorted, 0.5) ?? null,
    p90: percentile(sorted, 0.9) ?? null,
    p99: percentile(sorted, 0.99) ?? null,
    estimatedNetEdgeGteReference,
  };
}

export function getPaperOpenDiagnostics(): PaperOpenDiagnosticsSnapshot {
  const st = getStore();
  const droppedPre = st.preFilterMergedTotal - st.preFilterEnteringEngineTotal;
  const r = { ...st.rejectionCounts };
  const rejectionTotal = REJECTION_KEYS.reduce((s, k) => s + r[k], 0);

  const stage1 = st.preFilterEnteringEngineTotal;
  const stage2 = stage1 - r.ESTIMATED_NET_EDGE_BELOW_THRESHOLD - r.RECOMMENDED_CAPITAL_LE_ZERO_AT_ENGINE;
  const stage3 = stage2 - r.CONFIDENCE_BELOW_THRESHOLD;
  const stage4 = stage3 - r.ACTIVE_TRADE_FOR_OPPORTUNITY;
  const stage5 = stage4 - r.EXPOSURE_CLUSTER_OR_MARKET_LIMIT;
  const stage6 = stage5 - r.REQUESTED_CAPITAL_LE_ZERO;
  const stage7 = stage6 - r.SIMULATOR_FILLED_CAPITAL_LE_ZERO;
  const econRejects =
    r.FILLED_CAPITAL_BELOW_MIN +
    r.PROGRESS_PROBABILITY_FACTOR_BELOW_MIN +
    r.ENTRY_ECONOMIC_SCORE_BELOW_MIN +
    r.EXPECTED_REALIZABLE_NET_PNL_BELOW_MIN +
    r.EXPECTED_NET_PNL_BELOW_MIN +
    r.EXPECTED_NET_MARGIN_BELOW_MIN +
    r.GROSS_TO_FEES_RATIO_BELOW_MIN +
    r.CROSS_MARKET_NET_TO_GROSS_EDGE_BELOW_MIN;
  const stage8_afterEconomicEntryFilters = stage7 - econRejects;

  const cand = st.engineCandidatesTotal;
  const propCand: Record<string, number> = {};
  if (cand > 0) {
    for (const k of REJECTION_KEYS) {
      propCand[k] = round4(r[k] / cand);
    }
    propCand.OPENED = round4(st.engineOpenedTotal / cand);
  } else {
    for (const k of REJECTION_KEYS) propCand[k] = 0;
    propCand.OPENED = 0;
  }

  const propRej: Record<string, number> = {};
  if (rejectionTotal > 0) {
    for (const k of REJECTION_KEYS) {
      propRej[k] = round4(r[k] / rejectionTotal);
    }
  } else {
    for (const k of REJECTION_KEYS) propRej[k] = 0;
  }

  const topEconomic = ECONOMIC_REASON_TOP.map((reason) => ({ reason, count: r[reason] }))
    .filter((x) => x.count > 0)
    .sort((a, b) => b.count - a.count);

  const econCfg = getPaperEntryEconomicFilterConfig();
  const adaptiveOpts = {
    enableAdaptive: econCfg.enableAdaptiveProgressGuard,
    minSamples: econCfg.minSamplesForAdaptiveProgressGuard,
    extraMax: econCfg.adaptiveProgressGuardExtraMax,
  };

  if (!st.economicRecentWindowDerived) {
    refreshEconomicRecentWindowDerived(st);
  }
  const derived = st.economicRecentWindowDerived!;
  const recentAgg = derived.recentAgg;
  const uniqueDiag = derived.uniqueDiag;

  return {
    preFilter: {
      mergedOpportunitiesTotal: st.preFilterMergedTotal,
      enteringEngineTotal: st.preFilterEnteringEngineTotal,
      droppedRecommendedCapitalLeZero: droppedPre,
      cumulativeSemanticsNote:
        "mergedOpportunitiesTotal = soma cumulativa (desde arranque) de len(mergedExpanded) por ciclo paper completado; enteringEngineTotal = soma cumulativa de candidatos com recommendedCapital>0 antes do motor. Para o último merge/whitelist/expansão ver openUpstreamDiagnostics.lastCycle e upstreamScannerExpansionLastCycle; para ciclos iniciados vs completados ver openUpstreamDiagnostics.syncHint vs cumulative.cyclesCompleted.",
    },
    engine: {
      candidatesEvaluatedTotal: st.engineCandidatesTotal,
      reachedSimulateEntryTotal: st.engineReachedPreSimulateTotal,
      openedTotal: st.engineOpenedTotal,
      rejectionsByReason: r,
      rejectionTotal,
    },
    proportions: {
      ofEngineCandidates: propCand as Record<PaperOpenRejectionKey | "OPENED", number>,
      ofRejectionEvents: propRej as Record<PaperOpenRejectionKey, number>,
    },
    funnel: {
      stage0_mergedAllSources: st.preFilterMergedTotal,
      stage1_afterPreFilterRecommendedCapPositive: stage1,
      stage2_afterNetEdgeThreshold: stage2,
      stage3_afterConfidenceThreshold: stage3,
      stage4_afterNoActiveDuplicate: stage4,
      stage5_afterExposureLimits: stage5,
      stage6_afterRequestedCapitalPositive: stage6,
      stage7_afterSimulatorFill: stage7,
      stage8_afterEconomicEntryFilters: stage8_afterEconomicEntryFilters,
    },
    netEdgeThresholdRejections: buildNetEdgeRejectionStats(st.netEdgeRejectionSamples),
    economicEntryFilter: buildEconomicEntryFilterSnapshot(st),
    economicEntryDecision: {
      avgEntryEconomicScorePass: avg(st.economicPassScores),
      avgEntryEconomicScoreFail: avg(st.economicFailScores),
      avgProgressProbabilityFactorPass: avg(st.economicPassProgress),
      avgProgressProbabilityFactorFail: avg(st.economicFailProgress),
      recentEntryEconomicScoreEvaluated: (() => {
        const arr = st.economicAllEvaluatedScores;
        const m = minMaxAvg(arr);
        return {
          sampleCount: arr.length,
          sampleCap: ALL_EVALUATED_SCORES_CAP,
          min: m.min,
          max: m.max,
          avg: m.avg,
        };
      })(),
      entryEconomicScoreBuckets: { ...st.entryEconomicScoreBuckets },
      progressFactorBuckets: { ...st.progressFactorBuckets },
      topEconomicRejectionReasons: topEconomic,
      entryScoreAtOpenVsExit: (() => {
        const tp = st.entryScoresAtCloseTakeProfit;
        const np = st.entryScoresAtCloseNoProgress;
        const tpM = minMaxAvg(tp);
        const npM = minMaxAvg(np);
        return {
          takeProfit: { n: tp.length, avg: tpM.avg, min: tpM.min, max: tpM.max },
          noProgressExit: { n: np.length, avg: npM.avg, min: npM.min, max: npM.max },
        };
      })(),
      openedThenNoProgressExit: {
        total: st.noProgressAfterEntryTotal,
        recentSamples: [...st.noProgressAfterEntryRecent],
        recentCap: NO_PROGRESS_RECENT_CAP,
      },
      exitOutcomeByProfile: getPaperExitProfileMemorySnapshot(),
      effectiveProgressGuardByProfile: buildEffectiveProgressGuardByProfileMap(
        econCfg.minProgressProbabilityFactorToOpen,
        adaptiveOpts
      ),
      profileAdaptiveDecisionSummary: {
        enabled: econCfg.enableAdaptiveProgressGuard,
        globalMin: econCfg.minProgressProbabilityFactorToOpen,
        extraMax: econCfg.adaptiveProgressGuardExtraMax,
        minSamples: econCfg.minSamplesForAdaptiveProgressGuard,
      },
      crossMarketNetGrossEntryGuard: (() => {
        const cm = st.economicEntryPassSamples.filter(
          (m) => m.entryProfileKey === STANDARD_CROSS_MARKET_PROFILE_KEY
        );
        const n = cm.length;
        const sum = cm.reduce((a, m) => a + m.netToGrossEdgeRatioAtEntry, 0);
        return {
          enabled: econCfg.enableCrossMarketNetGrossEntryGuard,
          profileKey: STANDARD_CROSS_MARKET_PROFILE_KEY,
          minNetToGrossRatio: econCfg.minNetToGrossEdgeRatioCrossMarket,
          avgNetToGrossAmongPassesSample: n > 0 ? round4(sum / n) : null,
          passSampleCountCrossMarket: n,
        };
      })(),
      recentEvaluatedCandidates: [...st.recentEvaluatedCandidates],
      recentEvaluatedCandidatesCap: RECENT_EVALUATED_CANDIDATES_CAP,
      avgProgressProbabilityFactorByFinalReason: recentAgg.avgProgressProbabilityFactorByFinalReason,
      avgEntryEconomicScoreByFinalReason: recentAgg.avgEntryEconomicScoreByFinalReason,
      avgNetToGrossEdgeRatioByFinalReason: recentAgg.avgNetToGrossEdgeRatioByFinalReason,
      countByProfileAndFinalReason: recentAgg.countByProfileAndFinalReason,
      recentUniqueEvaluatedCandidates: uniqueDiag.recentUniqueEvaluatedCandidates,
      uniqueCountByFinalReason: uniqueDiag.uniqueCountByFinalReason,
      uniqueCountByProfileAndFinalReason: uniqueDiag.uniqueCountByProfileAndFinalReason,
      repeatedCandidateSummary: uniqueDiag.repeatedCandidateSummary,
      topRepeatedOpportunityIds: uniqueDiag.topRepeatedOpportunityIds,
    },
    economicCooldown: {
      cooldownPolicySnapshot: getEconomicCooldownPolicySnapshot(),
      activeCooldownSummary: getEconomicCooldownActiveSummary(),
      suppressedByCooldownCount: st.suppressedByCooldownCount,
      suppressedByCooldownByProfile: { ...st.suppressedByCooldownByProfile },
      suppressedCandidatesRecent: [...st.suppressedCandidatesRecent],
    },
    exploration: {
      explorationPolicySnapshot: getExplorationPolicySnapshot(),
      recentNoveltyStats: st.explorationLastCycle
        ? {
            batchSize: st.explorationLastCycle.batchSize,
            distinctDedupeKeysInBatch: st.explorationLastCycle.distinctDedupeKeysInBatch,
            avgRepeatCountInBatch: st.explorationLastCycle.avgRepeatCountInBatch,
          }
        : null,
      evaluatedUniqueCandidatesCount: uniqueDiag.repeatedCandidateSummary.totalUniqueCandidates,
      evaluationPriorityDiagnostics: st.explorationLastCycle?.evaluationPriorityDiagnostics ?? null,
      repeatedCandidatePenaltyAppliedCount: st.explorationLastCycle?.repeatedCandidatePenaltyAppliedCount ?? 0,
    },
    upstreamExploration: buildUpstreamExplorationSnapshot(st),
    upstreamSelectionDiagnostics: st.upstreamSelectionLastCycle,
    upstreamScannerCoverage: buildUpstreamScannerCoverageSnapshot(st),
    upstreamScannerDiagnostics: st.upstreamScannerExpansionLastCycle,
    upstreamScannerRawFunnel: st.upstreamScannerRawFunnelLastCycle,
    upstreamScannerCrossFunnel: st.upstreamScannerCrossFunnelLastCycle,
    upstreamScannerSourceUniverse: st.upstreamScannerSourceUniverseLastCycle,
    ...(() => {
      const gp = getGraphScanRuntime().lastGraphPipelineDiagnostics;
      if (!gp) {
        return {
          graphPipelineDiagnosticsCapturedAtMs: null,
          graphRelationDiagnostics: null,
          graphClusterDiagnostics: null,
          graphSourceQualityDiagnostics: null,
        };
      }
      return {
        graphPipelineDiagnosticsCapturedAtMs: gp.capturedAtMs,
        graphRelationDiagnostics: gp.relationBuilderFunnel,
        graphClusterDiagnostics: gp.clusterFormationFunnel,
        graphSourceQualityDiagnostics: gp.graphSourceQuality,
      };
    })(),
    graphOpportunityDownstreamImpact: buildGraphOpportunityDownstreamImpactSnapshot(),
    complementaryRelaxedPaperImpact: buildComplementaryRelaxedPaperImpactSnapshot(),
    graphProvenancePropagationDiagnostics: buildGraphProvenancePropagationDiagnostics(),
    ...(() => {
      const q = buildGraphProvenanceQualityBundle();
      return {
        graphProvenanceQualityAudit: q.graphProvenanceQualityAudit,
        complementaryRelaxedQualityAudit: q.complementaryRelaxedQualityAudit,
        complementaryRelaxedStructuralRobustness: q.complementaryRelaxedStructuralRobustness,
        complementaryRelaxedIntraClusterAudit: q.complementaryRelaxedIntraClusterAudit,
        graphQualityAuditSourceDiagnostics: q.graphQualityAuditSourceDiagnostics,
        feeImpactAudit: q.feeImpactAudit,
      };
    })(),
    closedTradesSourceDiagnostics: buildClosedTradesSourceDiagnostics(),
    paperEconomicUnitAudit: buildPaperEconomicUnitAudit(),
    paperExitModelAudit: buildPaperExitModelAudit(),
    protectedCoreDiagnostics: buildProtectedCoreDiagnostics(),
    experimentalMicroLanes: buildExperimentalMicroLaneDiagnostics(),
  };
}

function buildEconomicEntryFilterSnapshot(st: ReturnType<typeof getStore>): PaperOpenDiagnosticsSnapshot["economicEntryFilter"] {
  const samples = st.economicEntryPassSamples;
  const n = samples.length;
  const cfg = getPaperEntryEconomicFilterConfig();
  if (n === 0) {
    return {
      enabled: cfg.enabled,
      thresholds: cfg,
      passThrough: {
        sampleCount: 0,
        avgExpectedNetPnlToOpenUsd: null,
        avgExpectedRealizableNetPnlUsd: null,
        avgFilledCapital: null,
        avgExpectedNetProfitMargin: null,
        avgGrossToFeesRatio: null,
        avgProgressProbabilityFactor: null,
        avgEntryEconomicScore: null,
        avgEffectiveMinProgressProbabilityFactor: null,
      },
    };
  }
  const sumNet = samples.reduce((a, m) => a + m.expectedNetPnlToOpenUsd, 0);
  const sumReal = samples.reduce((a, m) => a + m.expectedRealizableNetPnlUsd, 0);
  const sumFill = samples.reduce((a, m) => a + m.filledCapital, 0);
  const sumMarg = samples.reduce((a, m) => a + m.expectedNetProfitMargin, 0);
  const sumGtf = samples.reduce((a, m) => a + m.grossToFeesRatio, 0);
  const sumProg = samples.reduce((a, m) => a + m.progressProbabilityFactor, 0);
  const sumScore = samples.reduce((a, m) => a + m.entryEconomicScore, 0);
  const withEff = samples.filter((m) => m.effectiveMinProgressProbabilityFactorToOpen != null);
  const sumEff = withEff.reduce((a, m) => a + (m.effectiveMinProgressProbabilityFactorToOpen ?? 0), 0);
  return {
    enabled: cfg.enabled,
    thresholds: cfg,
    passThrough: {
      sampleCount: n,
      avgExpectedNetPnlToOpenUsd: round4(sumNet / n),
      avgExpectedRealizableNetPnlUsd: round4(sumReal / n),
      avgFilledCapital: round4(sumFill / n),
      avgExpectedNetProfitMargin: round4(sumMarg / n),
      avgGrossToFeesRatio: round4(sumGtf / n),
      avgProgressProbabilityFactor: round4(sumProg / n),
      avgEntryEconomicScore: round4(sumScore / n),
      avgEffectiveMinProgressProbabilityFactor:
        withEff.length > 0 ? round4(sumEff / withEff.length) : null,
    },
  };
}
