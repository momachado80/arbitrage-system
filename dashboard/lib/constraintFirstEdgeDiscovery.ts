/**
 * Constraint First Edge Discovery — probe decisório observacional.
 * Só famílias de alta restrição (partição exaustiva binária, dominância clara, subfamílias mesmo evento).
 * Ranking por netEdgeAfterHaircut (não por anomalia bruta). Sem execução de ordens.
 */

import type { NormalizedMarket } from "./polymarketClient";
import { getAllMarkets } from "./marketDataService";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

const GLOBAL_RECURRENCE_KEY = "__constraintFirstEdgeRecurrence_v1";

type RecurrenceState = { counts: Record<string, number> };

function bumpRecurrence(familyIds: string[]): Record<string, number> {
  const g = globalThis as unknown as Record<string, RecurrenceState | undefined>;
  if (!g[GLOBAL_RECURRENCE_KEY]) g[GLOBAL_RECURRENCE_KEY] = { counts: {} };
  const st = g[GLOBAL_RECURRENCE_KEY]!;
  for (const id of familyIds) {
    st.counts[id] = (st.counts[id] ?? 0) + 1;
  }
  return { ...st.counts };
}

export type ConstraintFamilyType =
  | "exhaustive_partition_binary"
  | "strong_dominance_binary"
  | "same_event_subfamily";

export type ConstraintFirstVerdict =
  | "no_viable_family_found"
  | "weak_candidates_only"
  | "viable_candidate_present"
  | "promising_candidate_present";

export type DominantFailureComponent =
  | "raw_too_small"
  | "friction"
  | "uncertainty"
  | "model_risk"
  | "balanced_drag"
  | "insufficient_sample";

/** Decomposição do proxy `frictionCostEstimate` (v1); latência/execUncertainty reservados = 0 até haver inputs observáveis. */
export interface FrictionComponentBreakdown {
  feeComponent: number;
  spreadComponent: number;
  slippageComponent: number;
  latencyPenaltyComponent: number;
  executionUncertaintyComponent: number;
  frictionTotal: number;
}

export type FrictionAuditVerdict =
  | "insufficient_sample"
  | "spread_component_dominates_aggregate"
  | "fixed_fee_slippage_dominates_aggregate"
  | "balanced_friction_mix"
  | "best_case_min_friction_still_below_net_floor_all_types"
  | "borderline_if_friction_model_halved_observationally";

/** Coeficiente atual em `frictionCostEstimate`: fee fixo + `avgSpread * coef` + penalidade liquidez. */
export const CONSTRAINT_FIRST_SPREAD_FRICTION_COEFFICIENT_BASE = 0.35;

export type SpreadCalibrationAuditVerdict =
  | "insufficient_sample"
  | "no_family_passes_even_at_minimum_grid_multiplier"
  | "pass_only_at_multipliers_below_half"
  | "pass_at_or_above_half_multiplier_plausible_zone"
  | "pass_at_baseline_multiplier";

const ALL_FAMILY_TYPES: readonly ConstraintFamilyType[] = [
  "exhaustive_partition_binary",
  "strong_dominance_binary",
  "same_event_subfamily",
] as const;

export interface ConstraintFirstFamilyRow {
  familyId: string;
  familyType: ConstraintFamilyType;
  semanticConfidence: number;
  rawEdge: number;
  frictionCostEstimate: number;
  uncertaintyHaircut: number;
  modelRiskHaircut: number;
  netEdgeAfterHaircut: number;
  persistenceScore: number;
  recurrenceObservationCount: number;
  concentrationRisk: number;
  capacityEstimate: number | null;
  killReason: string | null;
  supportingNote: string;
  /** ids de mercados membros (auditoria) */
  memberMarketIds: string[];
}

/** Amostra observacional pré-rejeição final (auditoria de funil). */
export interface ConstraintFirstPreRejectionSample {
  familyId: string;
  familyType: ConstraintFamilyType;
  semanticConfidence: number;
  rawEdge: number;
  frictionCostEstimate: number;
  uncertaintyHaircut: number;
  modelRiskHaircut: number;
  netEdgeAfterHaircut: number;
  rejectionReason: string | null;
}

export interface ConstraintFirstStrongestRejected {
  familyId: string;
  familyType: ConstraintFamilyType;
  semanticConfidence: number;
  rawEdge: number;
  netEdgeAfterHaircut: number;
  killReason: string | null;
}

export interface IneligibleMarketBreakdown {
  closed: number;
  inactive: number;
  lowLiquidity: number;
  outcomesLt2: number;
  priceOutcomeMismatch: number;
}

export interface ConstraintFirstEdgeDiscoveryDigest {
  probeVersion: "constraint-first-edge-v1";
  readDisclaimer: string;
  constraintFirstVerdict: ConstraintFirstVerdict;
  familiesScanned: number;
  familiesPassingEdgeFloor: number;
  topFamilies: ConstraintFirstFamilyRow[];
  thresholdsUsed: Record<string, number>;
  constraintFirstSummaryLine: string;
  families: ConstraintFirstFamilyRow[];
  computedAt: string;
  /** --- Funil / diagnóstico (não altera veredito) --- */
  totalMarketsLoaded: number;
  totalMarketsEligibleAfterLiquidity: number;
  totalCandidateFamiliesBuilt: number;
  totalFamiliesRejectedSemantic: number;
  totalFamiliesRejectedConcentration: number;
  totalFamiliesRejectedHaircut: number;
  totalFamiliesRejectedNoRecurrence: number;
  totalFamiliesScored: number;
  strongestRejectedFamily: ConstraintFirstStrongestRejected | null;
  strongestRejectedReason: string | null;
  diagnosticSummaryLine: string;
  preRejectionFamilySamples: ConstraintFirstPreRejectionSample[];
  ineligibleMarketBreakdown: IneligibleMarketBreakdown;
  /** --- Viabilidade / atribuição (não altera thresholds nem veredito) --- */
  familyTypeBreakdown: Record<ConstraintFamilyType, number>;
  bestRawEdgeByFamilyType: Record<ConstraintFamilyType, number | null>;
  bestNetEdgeByFamilyType: Record<ConstraintFamilyType, number | null>;
  avgRawEdgeByFamilyType: Record<ConstraintFamilyType, number | null>;
  avgFrictionCostByFamilyType: Record<ConstraintFamilyType, number | null>;
  avgUncertaintyHaircutByFamilyType: Record<ConstraintFamilyType, number | null>;
  avgModelRiskHaircutByFamilyType: Record<ConstraintFamilyType, number | null>;
  /** Média de (passFloor + friction + uncertainty + modelRisk) por família do tipo — barreira típica de raw para net>passFloor. */
  minimumRawEdgeNeededToPassByFamilyType: Record<ConstraintFamilyType, number | null>;
  dominantFailureComponentByFamilyType: Record<ConstraintFamilyType, DominantFailureComponent>;
  economicallyImpossibleFamilyTypes: ConstraintFamilyType[];
  potentiallyViableFamilyTypes: ConstraintFamilyType[];
  feasibilitySummaryLine: string;
  /** --- Auditoria de fricção (atribuição; não altera floors nem famílias) --- */
  frictionBreakdownByFamilyType: Record<ConstraintFamilyType, FrictionComponentBreakdown | null>;
  avgFeeComponentByFamilyType: Record<ConstraintFamilyType, number | null>;
  avgSpreadComponentByFamilyType: Record<ConstraintFamilyType, number | null>;
  avgSlippageComponentByFamilyType: Record<ConstraintFamilyType, number | null>;
  avgLatencyPenaltyComponentByFamilyType: Record<ConstraintFamilyType, number | null>;
  avgExecutionUncertaintyComponentByFamilyType: Record<ConstraintFamilyType, number | null>;
  bestCaseFrictionByFamilyType: Record<ConstraintFamilyType, number | null>;
  medianFrictionByFamilyType: Record<ConstraintFamilyType, number | null>;
  frictionToRawEdgeRatioByFamilyType: Record<ConstraintFamilyType, number | null>;
  /** Melhor net hipotético por tipo: família com max rawEdge, fricção = min(friction) no tipo, mesmos unc/model dessa linha. */
  netUnderBestCaseFrictionByFamilyType: Record<ConstraintFamilyType, number | null>;
  frictionDominantComponentByFamilyType: Record<ConstraintFamilyType, "fee" | "spread" | "slippage" | "tie" | "none">;
  frictionBorderlineIfHalvedFamilyTypes: ConstraintFamilyType[];
  frictionAuditVerdict: FrictionAuditVerdict;
  frictionAuditSummaryLine: string;
  /** --- Auditoria de calibração do spread (só varia coef. do spread em friction; floors e famílias intactos) --- */
  spreadCalibrationGrid: SpreadCalibrationGridRow[];
  verdictBySpreadCoefficient: Record<string, string>;
  familiesPassingBySpreadCoefficient: Record<string, number>;
  bestNetBySpreadCoefficient: Record<string, number>;
  strongestFamilyBySpreadCoefficient: Record<
    string,
    { familyId: string; familyType: ConstraintFamilyType; netEdgeAfterHaircut: number } | null
  >;
  /** Menor coef. efectivo (avgSpread * coef) no grelha onde existe ≥1 família a passar o piso líquido; null se nenhum ponto passa. */
  minSpreadCoefficientForAnyPass: number | null;
  /** Menor coef. efectivo no grelha onde `exhaustive_partition_binary` passa; null se nunca passa no grelha. */
  minSpreadCoefficientForExhaustivePartitionBinaryPass: number | null;
  spreadCalibrationAuditVerdict: SpreadCalibrationAuditVerdict;
  spreadCalibrationSummaryLine: string;
  /** --- Auditoria de kills residuais (pós-spread-stress; não altera thresholds) --- */
  residualKillAuditGrid: ResidualKillAuditGridRow[];
  topNearPassFamilies: NearPassFamilyAudit[];
  killReasonByFamilyAtBestSpreadPoint: Record<string, string | null>;
  semanticKillCountBySpreadPoint: Record<string, number>;
  concentrationKillCountBySpreadPoint: Record<string, number>;
  mixedKillCountBySpreadPoint: Record<string, number>;
  bestNearPassFamily: BestNearPassFamilySnapshot | null;
  bestNearPassFamilyKillAttribution: ResidualKillAttributionDetail | null;
  residualKillAuditVerdict: ResidualKillAuditVerdict;
  residualKillAuditSummaryLine: string;
  /** --- Auditoria semântica de concentração (métrica vs construção; não altera thresholds nem veredito) --- */
  concentrationAuditByFamilyType: Record<ConstraintFamilyType, ConcentrationFamilyTypeAudit | null>;
  concentrationStructuralDegeneracyByFamilyType: Record<ConstraintFamilyType, boolean>;
  singleMarketFamilyTypes: ConstraintFamilyType[];
  multiMarketFamilyTypes: ConstraintFamilyType[];
  concentrationInformativeFamilyTypes: ConstraintFamilyType[];
  concentrationDegenerateFamilyTypes: ConstraintFamilyType[];
  bestFamiliesIfConcentrationIgnoredWhenStructurallyDegenerate: ConcentrationIgnoreCounterfactualRow[];
  concentrationAuditVerdict: ConcentrationAuditVerdict;
  concentrationAuditSummaryLine: string;
  /** --- Política de concentração por classe (kill stack paralelo; linhas `families` inalteradas) --- */
  concentrationPolicyByFamilyType: Record<ConstraintFamilyType, string>;
  concentrationKillModeByFamilyType: Record<
    ConstraintFamilyType,
    "hard_kill_active" | "diagnostic_no_hard_kill" | "not_applicable_no_sample"
  >;
  familiesPassingEdgeFloorUnderClassAwareConcentration: number;
  topFamiliesUnderClassAwareConcentration: ConstraintFirstFamilyRow[];
  strongestFamilyUnderClassAwareConcentration: ConstraintFirstFamilyRow | null;
  classAwareConstraintFirstVerdict: ConstraintFirstVerdict;
  classAwareConstraintFirstSummaryLine: string;
  constraintFirstVerdictComparisonLine: string;
  exhaustivePartitionBinaryHasClassAwarePass: boolean;
  sameEventSubfamilyHasClassAwarePass: boolean;
  /** --- Auditoria residual pós-concentração class-aware (atribuição; não altera regras) --- */
  postConcentrationResidualAudit: PostConcentrationFamilyResidualBreakdown[];
  topNearPassFamiliesPostConcentration: PostConcentrationNearPassSummary[];
  dominantResidualKillByFamilyType: Record<ConstraintFamilyType, string>;
  avgResidualGapToPassFloorByFamilyType: Record<ConstraintFamilyType, number | null>;
  strongestNearPassResidualBreakdown: PostConcentrationFamilyResidualBreakdown | null;
  residualKillCountsByType: Record<ConstraintFamilyType, Record<string, number>>;
  residualEconomicFailureModes: string[];
  postConcentrationResidualAuditVerdict: PostConcentrationResidualAuditVerdict;
  postConcentrationResidualSummaryLine: string;
  /** --- Auditoria de realidade do rawEdge (proxy vs observável; não altera scores nem regras) --- */
  rawEdgeRealityAudit: PartitionRawEdgeRealityAuditRow[];
  rawEdgeByFamilyType: Record<ConstraintFamilyType, RawEdgeStats | null>;
  rawEdgeVsObservedMispricingByFamilyType: Record<ConstraintFamilyType, number | null>;
  bestObservedMispricingByFamilyType: Record<ConstraintFamilyType, number | null>;
  avgObservedMispricingByFamilyType: Record<ConstraintFamilyType, number | null>;
  rawEdgeCompressionRatioByFamilyType: Record<ConstraintFamilyType, number | null>;
  strongestPartitionRealityCheck: PartitionRawEdgeRealityAuditRow | null;
  partitionRealityCheckSamples: PartitionRawEdgeRealityAuditRow[];
  rawEdgeRealityAuditVerdict: RawEdgeRealityAuditVerdict;
  rawEdgeRealitySummaryLine: string;
}

/** Uma linha do grelha: multiplicador sobre o coef. base de spread; só a parcela spread da fricção muda. */
export interface SpreadCalibrationGridRow {
  spreadCoefficientMultiplier: number;
  effectiveSpreadCoefficient: number;
  familiesPassingEdgeFloor: number;
  exhaustivePartitionBinaryPasses: boolean;
  bestNetEdge: number;
  strongestFamilyId: string | null;
  strongestFamilyType: ConstraintFamilyType | null;
  verdict: string;
}

/** Atribuição de bloqueios paralelos (semântica / concentração / só net) ao ponto de spread da grelha. */
export type ResidualKillAttributionCategory =
  | "semantic_confidence"
  | "concentration_risk"
  | "both_semantic_and_concentration"
  | "other_residual_kill"
  | "would_pass";

export type ResidualKillAuditVerdict =
  | "insufficient_sample"
  | "semantic_confidence_dominates_at_best_spread_point"
  | "concentration_risk_dominates_at_best_spread_point"
  | "mixed_semantic_and_concentration_at_best_spread_point"
  | "net_floor_only_other_residual_at_best_spread_point"
  | "balanced_residual_attribution_at_best_spread_point";

export interface NearPassFamilyAudit {
  familyId: string;
  familyType: ConstraintFamilyType;
  hypotheticalNetAtSpreadPoint: number;
  killReasonOrdered: string | null;
  residualKillAttribution: ResidualKillAttributionCategory;
  semanticConfidence: number;
  concentrationRisk: number;
}

export interface ResidualKillAttributionDetail {
  residualKillAttribution: ResidualKillAttributionCategory;
  semanticBelowKillFloor: boolean;
  concentrationAboveKillFloor: boolean;
  netAtOrBelowPassFloor: boolean;
  killReasonOrdered: string | null;
  spreadCoefficientMultiplier: number;
  effectiveSpreadCoefficient: number;
  hypotheticalNet: number;
}

export interface BestNearPassFamilySnapshot {
  familyId: string;
  familyType: ConstraintFamilyType;
  semanticConfidence: number;
  concentrationRisk: number;
  hypotheticalNetAtSpreadPoint: number;
  spreadCoefficientMultiplier: number;
  effectiveSpreadCoefficient: number;
}

export interface ResidualKillAuditGridRow {
  spreadCoefficientMultiplier: number;
  effectiveSpreadCoefficient: number;
  semanticKillCount: number;
  concentrationKillCount: number;
  mixedKillCount: number;
  otherResidualKillCount: number;
  totalResolvableFamilies: number;
  familiesPassingFullKillStack: number;
}

export type ConcentrationAuditVerdict =
  | "insufficient_sample"
  | "single_market_family_types_concentration_metric_structurally_degenerate"
  | "multi_market_family_types_concentration_informative_single_market_types_degenerate"
  | "concentration_informative_across_family_types_with_samples"
  | "mixed_or_partial_sample_concentration_semantics";

export interface ConcentrationFamilyTypeAudit {
  sampleSize: number;
  memberCountMin: number | null;
  memberCountMax: number | null;
  allFamiliesSingleMember: boolean;
  concentrationMin: number | null;
  concentrationMax: number | null;
  concentrationMean: number | null;
  concentrationRange: number | null;
  variesMeaningfullyWithinType: boolean;
  nearConstantByConstruction: boolean;
  informativeForEconomicFragility: boolean;
  structurallyDegenerateConcentrationSemantics: boolean;
  rationaleLine: string;
}

/** Counterfactual: ignorar gate de concentração só quando o tipo é degenerado por construção (não é regra viva). */
export interface ConcentrationIgnoreCounterfactualRow {
  familyId: string;
  familyType: ConstraintFamilyType;
  hypotheticalNetAtBestSpreadStress: number;
  spreadCoefficientMultiplier: number;
  effectiveSpreadCoefficient: number;
  familyTypeStructurallyDegenerateForConcentration: boolean;
  killReasonOrderedAtHypotheticalNet: string | null;
  killReasonOrderedIfConcIgnoredWhenDegenerate: string | null;
  wouldPassFullKillStackUnderCounterfactual: boolean;
}

export type PostConcentrationResidualDomCategory =
  | "semantic"
  | "net_below_pass_floor"
  | "concentration"
  | "would_pass"
  | "none";

export type PostConcentrationResidualAuditVerdict =
  | "insufficient_sample"
  | "residual_dominated_by_net_below_pass_floor"
  | "residual_semantic_floor_material_across_types"
  | "residual_mixed_net_and_semantic"
  | "exhaustive_partition_binary_economically_near_pass_under_class_aware_conc"
  | "exhaustive_partition_binary_economically_distant_dead_proxy"
  | "multi_family_types_net_haircut_stack_collapses_under_production_friction";

export interface PostConcentrationFamilyResidualBreakdown {
  familyId: string;
  familyType: ConstraintFamilyType;
  rawEdge: number;
  frictionCostEstimate: number;
  uncertaintyHaircut: number;
  modelRiskHaircut: number;
  netEdgeAfterHaircut: number;
  residualGapToPassFloor: number;
  concentrationRisk: number;
  semanticConfidence: number;
  recurrenceObservationCount: number;
  persistenceScore: number;
  classAwareKillReason: string | null;
  dominantResidualCategory: PostConcentrationResidualDomCategory;
  semanticKillContributionNote: string;
  recurrenceContributionNote: string;
  haircutStackUncertaintyPlusModel: number;
  impliedRequiredRawToClearNetPassFloor: number;
}

export interface PostConcentrationNearPassSummary {
  familyId: string;
  familyType: ConstraintFamilyType;
  netEdgeAfterHaircut: number;
  residualGapToPassFloor: number;
  classAwareKillReason: string | null;
  dominantResidualCategory: PostConcentrationResidualDomCategory;
}

export interface RawEdgeStats {
  min: number;
  max: number;
  mean: number;
  sampleSize: number;
}

export type RawEdgeRealityAuditVerdict =
  | "insufficient_sample"
  | "partition_raw_edge_systematically_below_direct_sum_deviation"
  | "partition_raw_edge_occasionally_below_direct_sum_deviation"
  | "partition_raw_edge_faithful_within_tolerance_vs_direct_sum_deviation"
  | "partition_observed_combined_mispricing_still_below_net_pass_floor_all_rows"
  | "partition_observed_combined_mispricing_clears_net_floor_for_some_rows"
  | "partition_proxy_compressed_but_observed_still_dead_vs_haircut_stack";

export interface PartitionRawEdgeRealityAuditRow {
  familyId: string;
  rawEdge: number;
  observedPartitionMispricingDirect: number;
  observedPartitionMispricingCombined: number;
  gapDirectMinusRaw: number;
  gapCombinedMinusRaw: number;
  compressionRatioCombinedOverRaw: number | null;
  impliedRequiredRawToClearNetPassFloor: number;
  netUnderObservedCombinedProxy: number;
  probSum: number;
  spread: number;
}

const AMBIGUITY_TOKENS =
  /\b(or|either|any of|multiple|combo|parlay|vs\.|versus|best of|first to|\d+\s*-\s*\d+)\b/i;

function normalizeEventStem(q: string): string {
  const s = q
    .toLowerCase()
    .replace(/\$[\d,.]+/g, "$")
    .replace(/\d{4}-\d{2}-\d{2}/g, "@")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 56);
  return s.length < 12 ? "" : s;
}

function maxPrice(m: NormalizedMarket): number {
  return m.prices.length ? Math.max(...m.prices) : 0;
}

function avgSpread(m: NormalizedMarket): number {
  return m.spread;
}

function partitionRawEdge(m: NormalizedMarket): number {
  const sumDev = Math.abs(1 - m.probSum);
  const sp = avgSpread(m);
  return r6(Math.min(0.12, sumDev * 1.8 + sp * 0.22));
}

/** Desvio directo observável |1 - probSum| (tecto), sem compressão da `partitionRawEdge`. */
function observedPartitionMispricingDirect(m: NormalizedMarket): number {
  const sumDev = Math.abs(1 - m.probSum);
  return r6(Math.min(0.15, sumDev));
}

/** Testemunho de spread (mercado); não reutiliza coeficientes da `partitionRawEdge`. */
function observedMispricingSpreadWitness(m: NormalizedMarket): number {
  return r6(Math.min(0.12, m.spread * 0.5));
}

/** Mispricing observável combinado (directo + spread), tecto 0.15 — realidade de mercado para comparar ao proxy. */
function observedPartitionMispricingCombined(m: NormalizedMarket): number {
  const a = observedPartitionMispricingDirect(m);
  const b = observedMispricingSpreadWitness(m);
  return r6(Math.min(0.15, a + b));
}

function partitionSemantic(m: NormalizedMarket): number {
  const sumDev = Math.abs(1 - m.probSum);
  return sumDev < 0.12 ? 0.9 : sumDev < 0.2 ? 0.78 : 0.62;
}

function dominanceEligible(m: NormalizedMarket): boolean {
  if (m.outcomes.length !== 2) return false;
  const mx = maxPrice(m);
  if (mx < 0.68) return false;
  if (m.spread > 0.45) return false;
  if (m.question.length > 140) return false;
  if (AMBIGUITY_TOKENS.test(m.question)) return false;
  return true;
}

function dominanceRawEdge(m: NormalizedMarket): number {
  const mx = maxPrice(m);
  return r6(Math.min(0.1, (mx - 0.5) * 0.12 + m.spread * 0.18));
}

function dominanceSemantic(m: NormalizedMarket): number {
  let s = 0.82;
  if (m.question.length > 90) s -= 0.06;
  if (m.spread > 0.25) s -= 0.05;
  return clamp01(s);
}

function familyLiquidityTotal(members: NormalizedMarket[]): number {
  return members.reduce((a, b) => a + b.liquidity, 0);
}

function concentrationRiskFromMembers(members: NormalizedMarket[]): number {
  const t = familyLiquidityTotal(members);
  if (t <= 0) return 1;
  let mx = 0;
  for (const m of members) mx = Math.max(mx, m.liquidity / t);
  return r6(mx);
}

/** Fricção com coeficiente de spread explícito (auditoria de calibração); produção usa `CONSTRAINT_FIRST_SPREAD_FRICTION_COEFFICIENT_BASE`. */
function frictionEstimateWithSpreadCoefficient(members: NormalizedMarket[], effectiveSpreadCoefficient: number): number {
  const avgLiq =
    members.length > 0 ? members.reduce((a, b) => a + b.liquidity, 0) / members.length : 0;
  const avgSp =
    members.length > 0 ? members.reduce((a, b) => a + b.spread, 0) / members.length : 0;
  const liqPen = r6((1 - clamp01(avgLiq / 25_000)) * 0.006);
  return r6(0.0065 + avgSp * effectiveSpreadCoefficient + liqPen);
}

function frictionEstimate(members: NormalizedMarket[]): number {
  return frictionEstimateWithSpreadCoefficient(members, CONSTRAINT_FIRST_SPREAD_FRICTION_COEFFICIENT_BASE);
}

/** Decomposição auditável do mesmo `frictionEstimate`; `slippage` absorve o residual para soma = total. */
function frictionComponentsFromMembers(members: NormalizedMarket[]): FrictionComponentBreakdown {
  const avgLiq =
    members.length > 0 ? members.reduce((a, b) => a + b.liquidity, 0) / members.length : 0;
  const avgSp =
    members.length > 0 ? members.reduce((a, b) => a + b.spread, 0) / members.length : 0;
  const feeComponent = 0.0065;
  const spreadComponent = r6(avgSp * CONSTRAINT_FIRST_SPREAD_FRICTION_COEFFICIENT_BASE);
  const frictionTotal = frictionEstimate(members);
  const slipResidual = r6(Math.max(0, frictionTotal - feeComponent - spreadComponent));
  return {
    feeComponent,
    spreadComponent,
    slippageComponent: slipResidual,
    latencyPenaltyComponent: 0,
    executionUncertaintyComponent: 0,
    frictionTotal,
  };
}

function uncertaintyHaircut(semantic: number, members: NormalizedMarket[]): number {
  const avgLiq =
    members.length > 0 ? members.reduce((a, b) => a + b.liquidity, 0) / members.length : 0;
  return r6((1 - semantic) * 0.045 + (1 - clamp01(avgLiq / 18_000)) * 0.018);
}

function persistenceFromSignals(recurrence: number, members: NormalizedMarket[]): number {
  const vol = members.reduce((a, b) => a + b.volume, 0);
  const liq = familyLiquidityTotal(members);
  const recNorm = clamp01(recurrence / 8);
  const volNorm = clamp01(Math.log10(vol + 10) / 6.5);
  const liqNorm = clamp01(liq / 120_000);
  return r6(recNorm * 0.45 + volNorm * 0.3 + liqNorm * 0.25);
}

function modelRiskHaircut(conc: number, persistenceScore: number): number {
  return r6(conc * 0.032 + (1 - persistenceScore) * 0.022);
}

function computeKillReasonForRow(
  semanticConfidence: number,
  netEdgeAfterHaircut: number,
  concentrationRisk: number,
  passFloor: number,
): string | null {
  if (semanticConfidence < envNum("CONSTRAINT_FIRST_SEMANTIC_KILL", 0.58)) {
    return "semantic_confidence_below_floor";
  }
  if (netEdgeAfterHaircut <= passFloor) {
    return `net_edge_after_haircut_<=${passFloor}`;
  }
  if (concentrationRisk > envNum("CONSTRAINT_FIRST_CONC_KILL", 0.82)) {
    return "concentration_too_high";
  }
  return null;
}

/** Kill stack com gate de concentração condicionado à classe: tipos degenerados usam `conc=0` só no teste do kill (valor real mantido na linha). */
function computeKillReasonForRowClassAware(
  row: ConstraintFirstFamilyRow,
  netEdgeAfterHaircut: number,
  passFloor: number,
  structuralDegeneracyByFamilyType: Record<ConstraintFamilyType, boolean>,
): string | null {
  const degenerate = structuralDegeneracyByFamilyType[row.familyType];
  const concForKillGate = degenerate ? 0 : row.concentrationRisk;
  return computeKillReasonForRow(row.semanticConfidence, netEdgeAfterHaircut, concForKillGate, passFloor);
}

function residualParallelClassification(
  row: ConstraintFirstFamilyRow,
  netHyp: number,
  passFloor: number,
): {
  semanticBlock: boolean;
  concBlock: boolean;
  netBlock: boolean;
  killOrdered: string | null;
} {
  const semTh = envNum("CONSTRAINT_FIRST_SEMANTIC_KILL", 0.58);
  const concTh = envNum("CONSTRAINT_FIRST_CONC_KILL", 0.82);
  const semanticBlock = row.semanticConfidence < semTh;
  const concBlock = row.concentrationRisk > concTh;
  const netBlock = netHyp <= passFloor;
  const killOrdered = computeKillReasonForRow(row.semanticConfidence, netHyp, row.concentrationRisk, passFloor);
  return { semanticBlock, concBlock, netBlock, killOrdered };
}

function residualKillAttributionFromBlocks(
  semanticBlock: boolean,
  concBlock: boolean,
  netBlock: boolean,
): ResidualKillAttributionCategory {
  if (!semanticBlock && !concBlock && !netBlock) return "would_pass";
  if (semanticBlock && concBlock) return "both_semantic_and_concentration";
  if (semanticBlock) return "semantic_confidence";
  if (concBlock) return "concentration_risk";
  return "other_residual_kill";
}

function capacityEstimateNotional(members: NormalizedMarket[]): number | null {
  if (members.length === 0) return null;
  const parts = members.map(m => Math.min(8000, m.liquidity * 0.015));
  return r6(parts.reduce((a, b) => a + b, 0));
}

function buildRow(
  familyId: string,
  familyType: ConstraintFamilyType,
  semanticConfidence: number,
  rawEdge: number,
  members: NormalizedMarket[],
  supportingNote: string,
  recurrenceMap: Record<string, number>,
): ConstraintFirstFamilyRow {
  const friction = frictionEstimate(members);
  const unc = uncertaintyHaircut(semanticConfidence, members);
  const conc = concentrationRiskFromMembers(members);
  const recurrence = recurrenceMap[familyId] ?? 1;
  const persistenceScore = persistenceFromSignals(recurrence, members);
  const model = modelRiskHaircut(conc, persistenceScore);
  const net = r6(rawEdge - friction - unc - model);
  const cap = capacityEstimateNotional(members);

  const passFloor = envNum("CONSTRAINT_FIRST_NET_PASS", 0.0045);
  const killReason = computeKillReasonForRow(semanticConfidence, net, conc, passFloor);

  return {
    familyId,
    familyType,
    semanticConfidence: r6(semanticConfidence),
    rawEdge,
    frictionCostEstimate: friction,
    uncertaintyHaircut: unc,
    modelRiskHaircut: model,
    netEdgeAfterHaircut: net,
    persistenceScore,
    recurrenceObservationCount: recurrence,
    concentrationRisk: conc,
    capacityEstimate: cap,
    killReason,
    supportingNote,
    memberMarketIds: members.map(m => m.id),
  };
}

function eligible(m: NormalizedMarket): boolean {
  return m.active && !m.closed && m.liquidity >= 350 && m.outcomes.length >= 2 && m.prices.length === m.outcomes.length;
}

type IneligibleKey = keyof IneligibleMarketBreakdown;

function ineligibleBucket(m: NormalizedMarket): IneligibleKey | null {
  if (m.closed) return "closed";
  if (!m.active) return "inactive";
  if (m.liquidity < 350) return "lowLiquidity";
  if (m.outcomes.length < 2) return "outcomesLt2";
  if (m.prices.length !== m.outcomes.length) return "priceOutcomeMismatch";
  return null;
}

function primaryRejectionBucket(row: ConstraintFirstFamilyRow): "semantic" | "haircut" | "concentration" | null {
  if (row.killReason === "semantic_confidence_below_floor") return "semantic";
  if (row.killReason === "concentration_too_high") return "concentration";
  if (row.killReason !== null && row.killReason.startsWith("net_edge_after_haircut")) return "haircut";
  return null;
}

function meanOrNull(nums: number[]): number | null {
  if (nums.length === 0) return null;
  return r6(nums.reduce((a, b) => a + b, 0) / nums.length);
}

function medianOrNull(nums: number[]): number | null {
  if (nums.length === 0) return null;
  const s = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? r6(s[mid]!) : r6((s[mid - 1]! + s[mid]!) / 2);
}

type FrictionAuditLayer = Pick<
  ConstraintFirstEdgeDiscoveryDigest,
  "frictionBreakdownByFamilyType" | "avgFeeComponentByFamilyType"
  | "avgSpreadComponentByFamilyType"
  | "avgSlippageComponentByFamilyType"
  | "avgLatencyPenaltyComponentByFamilyType"
  | "avgExecutionUncertaintyComponentByFamilyType"
  | "bestCaseFrictionByFamilyType"
  | "medianFrictionByFamilyType"
  | "frictionToRawEdgeRatioByFamilyType"
  | "netUnderBestCaseFrictionByFamilyType"
  | "frictionDominantComponentByFamilyType"
  | "frictionBorderlineIfHalvedFamilyTypes"
  | "frictionAuditVerdict"
  | "frictionAuditSummaryLine"
>;

function computeFrictionAuditLayer(
  rows: ConstraintFirstFamilyRow[],
  allMarkets: NormalizedMarket[],
  passFloor: number,
): FrictionAuditLayer {
  const nullNum = (): Record<ConstraintFamilyType, number | null> => ({
    exhaustive_partition_binary: null,
    strong_dominance_binary: null,
    same_event_subfamily: null,
  });
  const nullBreak = (): Record<ConstraintFamilyType, FrictionComponentBreakdown | null> => ({
    exhaustive_partition_binary: null,
    strong_dominance_binary: null,
    same_event_subfamily: null,
  });
  const nullDom = (): Record<ConstraintFamilyType, "fee" | "spread" | "slippage" | "tie" | "none"> => ({
    exhaustive_partition_binary: "none",
    strong_dominance_binary: "none",
    same_event_subfamily: "none",
  });

  const marketById = new Map<string, NormalizedMarket>();
  for (const m of allMarkets) marketById.set(m.id, m);

  interface RowBr {
    row: ConstraintFirstFamilyRow;
    br: FrictionComponentBreakdown;
  }
  const typed: Record<ConstraintFamilyType, RowBr[]> = {
    exhaustive_partition_binary: [],
    strong_dominance_binary: [],
    same_event_subfamily: [],
  };

  for (const r of rows) {
    const members: NormalizedMarket[] = [];
    for (const id of r.memberMarketIds) {
      const mm = marketById.get(id);
      if (mm) members.push(mm);
    }
    if (members.length === 0) continue;
    const br = frictionComponentsFromMembers(members);
    typed[r.familyType].push({ row: r, br });
  }

  const frictionBreakdownByFamilyType = nullBreak();
  const avgFeeComponentByFamilyType = nullNum();
  const avgSpreadComponentByFamilyType = nullNum();
  const avgSlippageComponentByFamilyType = nullNum();
  const avgLatencyPenaltyComponentByFamilyType = nullNum();
  const avgExecutionUncertaintyComponentByFamilyType = nullNum();
  const bestCaseFrictionByFamilyType = nullNum();
  const medianFrictionByFamilyType = nullNum();
  const frictionToRawEdgeRatioByFamilyType = nullNum();
  const netUnderBestCaseFrictionByFamilyType = nullNum();
  const frictionDominantComponentByFamilyType = nullDom();

  let globalFee = 0;
  let globalSpread = 0;
  let globalSlip = 0;
  let globalN = 0;

  for (const t of ALL_FAMILY_TYPES) {
    const arr = typed[t];
    if (arr.length === 0) continue;

    const fees = arr.map(x => x.br.feeComponent);
    const spreads = arr.map(x => x.br.spreadComponent);
    const slips = arr.map(x => x.br.slippageComponent);
    const totals = arr.map(x => x.br.frictionTotal);
    const frictionsRow = arr.map(x => x.row.frictionCostEstimate);

    avgFeeComponentByFamilyType[t] = meanOrNull(fees);
    avgSpreadComponentByFamilyType[t] = meanOrNull(spreads);
    avgSlippageComponentByFamilyType[t] = meanOrNull(slips);
    avgLatencyPenaltyComponentByFamilyType[t] = 0;
    avgExecutionUncertaintyComponentByFamilyType[t] = 0;

    frictionBreakdownByFamilyType[t] = {
      feeComponent: avgFeeComponentByFamilyType[t]!,
      spreadComponent: avgSpreadComponentByFamilyType[t]!,
      slippageComponent: avgSlippageComponentByFamilyType[t]!,
      latencyPenaltyComponent: 0,
      executionUncertaintyComponent: 0,
      frictionTotal: meanOrNull(totals)!,
    };

    bestCaseFrictionByFamilyType[t] = r6(Math.min(...totals));
    medianFrictionByFamilyType[t] = medianOrNull(frictionsRow);

    const ratios: number[] = [];
    for (const x of arr) {
      if (x.row.rawEdge > 1e-8) ratios.push(x.row.frictionCostEstimate / x.row.rawEdge);
    }
    frictionToRawEdgeRatioByFamilyType[t] = meanOrNull(ratios);

    const minF = Math.min(...totals);
    const bestRawEntry = arr.reduce((a, b) => (a.row.rawEdge >= b.row.rawEdge ? a : b));
    netUnderBestCaseFrictionByFamilyType[t] = r6(
      bestRawEntry.row.rawEdge -
        minF -
        bestRawEntry.row.uncertaintyHaircut -
        bestRawEntry.row.modelRiskHaircut,
    );

    const af = avgFeeComponentByFamilyType[t]!;
    const asp = avgSpreadComponentByFamilyType[t]!;
    const asl = avgSlippageComponentByFamilyType[t]!;
    const m12 = Math.max(af, asp, asl);
    const near = (a: number, b: number) => Math.abs(a - b) <= m12 * 0.08 + 1e-9;
    if (near(af, asp) && near(af, asl)) frictionDominantComponentByFamilyType[t] = "tie";
    else if (asp >= af && asp >= asl) frictionDominantComponentByFamilyType[t] = "spread";
    else if (af >= asp && af >= asl) frictionDominantComponentByFamilyType[t] = "fee";
    else frictionDominantComponentByFamilyType[t] = "slippage";

    for (const x of arr) {
      globalFee += x.br.feeComponent;
      globalSpread += x.br.spreadComponent;
      globalSlip += x.br.slippageComponent;
      globalN++;
    }
  }

  const frictionBorderlineIfHalvedFamilyTypes: ConstraintFamilyType[] = [];
  for (const t of ALL_FAMILY_TYPES) {
    const arr = typed[t];
    if (arr.length === 0) continue;
    const hit = arr.some(
      x =>
        r6(x.row.rawEdge - x.row.frictionCostEstimate * 0.5 - x.row.uncertaintyHaircut - x.row.modelRiskHaircut) >
        passFloor,
    );
    if (hit) frictionBorderlineIfHalvedFamilyTypes.push(t);
  }

  let frictionAuditVerdict: FrictionAuditVerdict = "insufficient_sample";
  let frictionAuditSummaryLine = "friction_audit: insufficient_sample";

  if (rows.length === 0 || globalN === 0) {
    return {
      frictionBreakdownByFamilyType,
      avgFeeComponentByFamilyType,
      avgSpreadComponentByFamilyType,
      avgSlippageComponentByFamilyType,
      avgLatencyPenaltyComponentByFamilyType,
      avgExecutionUncertaintyComponentByFamilyType,
      bestCaseFrictionByFamilyType,
      medianFrictionByFamilyType,
      frictionToRawEdgeRatioByFamilyType,
      netUnderBestCaseFrictionByFamilyType,
      frictionDominantComponentByFamilyType,
      frictionBorderlineIfHalvedFamilyTypes,
      frictionAuditVerdict,
      frictionAuditSummaryLine,
    };
  }

  const gTot = globalFee + globalSpread + globalSlip;
  const shareSp = gTot > 0 ? globalSpread / gTot : 0;
  const shareFeeSl = gTot > 0 ? (globalFee + globalSlip) / gTot : 0;

  let allTypesFailBestCase = true;
  let anyTypedNonEmpty = false;
  for (const t of ALL_FAMILY_TYPES) {
    if (typed[t].length === 0) continue;
    anyTypedNonEmpty = true;
    const v = netUnderBestCaseFrictionByFamilyType[t];
    if (v === null || v > passFloor) allTypesFailBestCase = false;
  }
  if (!anyTypedNonEmpty) allTypesFailBestCase = false;

  if (frictionBorderlineIfHalvedFamilyTypes.length > 0) {
    frictionAuditVerdict = "borderline_if_friction_model_halved_observationally";
    frictionAuditSummaryLine = `friction_audit: halved_friction_would_clear_floor_for_types=[${frictionBorderlineIfHalvedFamilyTypes.join(",")}] (observational; not a recommendation to loosen)`;
  } else if (allTypesFailBestCase) {
    frictionAuditVerdict = "best_case_min_friction_still_below_net_floor_all_types";
    frictionAuditSummaryLine = `friction_audit: even_best_raw_row_with_min_type_friction_net<=${passFloor} for all sampled types | aggregate_share_spread=${r6(shareSp)} fee_slip=${r6(shareFeeSl)}`;
  } else if (shareSp >= 0.48 && shareSp >= shareFeeSl) {
    frictionAuditVerdict = "spread_component_dominates_aggregate";
    frictionAuditSummaryLine = `friction_audit: aggregate_spread_share=${r6(shareSp)} of_proxy_components | model_conservative_if_spread_coef_overstates_taker_cost`;
  } else if (shareFeeSl > 0.52) {
    frictionAuditVerdict = "fixed_fee_slippage_dominates_aggregate";
    frictionAuditSummaryLine = `friction_audit: fee+slippage_share=${r6(shareFeeSl)} | fixed_fee_floor_0.0065_plus_liquidity_slippage_proxy`;
  } else {
    frictionAuditVerdict = "balanced_friction_mix";
    frictionAuditSummaryLine = `friction_audit: mixed_components spread_share=${r6(shareSp)} fee_slip_share=${r6(shareFeeSl)}`;
  }

  return {
    frictionBreakdownByFamilyType,
    avgFeeComponentByFamilyType,
    avgSpreadComponentByFamilyType,
    avgSlippageComponentByFamilyType,
    avgLatencyPenaltyComponentByFamilyType,
    avgExecutionUncertaintyComponentByFamilyType,
    bestCaseFrictionByFamilyType,
    medianFrictionByFamilyType,
    frictionToRawEdgeRatioByFamilyType,
    netUnderBestCaseFrictionByFamilyType,
    frictionDominantComponentByFamilyType,
    frictionBorderlineIfHalvedFamilyTypes,
    frictionAuditVerdict,
    frictionAuditSummaryLine,
  };
}

const SPREAD_CALIBRATION_GRID_MULTIPLIERS = [1, 0.75, 0.5, 0.35, 0.25] as const;

function spreadCalibrationGridKey(mult: number): string {
  if (mult === 1) return "1";
  if (mult === 0.75) return "0.75";
  if (mult === 0.5) return "0.5";
  if (mult === 0.35) return "0.35";
  if (mult === 0.25) return "0.25";
  return String(mult);
}

type SpreadCalibrationLayer = Pick<
  ConstraintFirstEdgeDiscoveryDigest,
  | "spreadCalibrationGrid"
  | "verdictBySpreadCoefficient"
  | "familiesPassingBySpreadCoefficient"
  | "bestNetBySpreadCoefficient"
  | "strongestFamilyBySpreadCoefficient"
  | "minSpreadCoefficientForAnyPass"
  | "minSpreadCoefficientForExhaustivePartitionBinaryPass"
  | "spreadCalibrationAuditVerdict"
  | "spreadCalibrationSummaryLine"
>;

function computeSpreadCalibrationLayer(
  rows: ConstraintFirstFamilyRow[],
  allMarkets: NormalizedMarket[],
  passFloor: number,
): SpreadCalibrationLayer {
  const empty = (): SpreadCalibrationLayer => ({
    spreadCalibrationGrid: [],
    verdictBySpreadCoefficient: {},
    familiesPassingBySpreadCoefficient: {},
    bestNetBySpreadCoefficient: {},
    strongestFamilyBySpreadCoefficient: {},
    minSpreadCoefficientForAnyPass: null,
    minSpreadCoefficientForExhaustivePartitionBinaryPass: null,
    spreadCalibrationAuditVerdict: "insufficient_sample",
    spreadCalibrationSummaryLine:
      "spread_calibration_audit: insufficient_sample (no_rows_or_no_resolvable_member_markets)",
  });

  if (rows.length === 0) return empty();

  const marketById = new Map<string, NormalizedMarket>();
  for (const m of allMarkets) marketById.set(m.id, m);

  const resolveMembers = (r: ConstraintFirstFamilyRow): NormalizedMarket[] => {
    const members: NormalizedMarket[] = [];
    for (const id of r.memberMarketIds) {
      const mm = marketById.get(id);
      if (mm) members.push(mm);
    }
    return members;
  };

  const anyResolvable = rows.some(r => resolveMembers(r).length > 0);
  if (!anyResolvable) return empty();

  const spreadCalibrationGrid: SpreadCalibrationGridRow[] = [];
  const verdictBySpreadCoefficient: Record<string, string> = {};
  const familiesPassingBySpreadCoefficient: Record<string, number> = {};
  const bestNetBySpreadCoefficient: Record<string, number> = {};
  const strongestFamilyBySpreadCoefficient: Record<
    string,
    { familyId: string; familyType: ConstraintFamilyType; netEdgeAfterHaircut: number } | null
  > = {};

  let minEffAny: number | null = null;
  let minEffExhaustive: number | null = null;

  for (const mult of SPREAD_CALIBRATION_GRID_MULTIPLIERS) {
    const effectiveSpreadCoefficient = r6(CONSTRAINT_FIRST_SPREAD_FRICTION_COEFFICIENT_BASE * mult);
    const key = spreadCalibrationGridKey(mult);

    let familiesPassingEdgeFloor = 0;
    let exhaustivePartitionBinaryPasses = false;
    let bestNetEdge = Number.NEGATIVE_INFINITY;
    let strongest: { familyId: string; familyType: ConstraintFamilyType; netEdgeAfterHaircut: number } | null = null;

    for (const r of rows) {
      const members = resolveMembers(r);
      if (members.length === 0) continue;
      const friction = frictionEstimateWithSpreadCoefficient(members, effectiveSpreadCoefficient);
      const net = r6(r.rawEdge - friction - r.uncertaintyHaircut - r.modelRiskHaircut);
      const kill = computeKillReasonForRow(r.semanticConfidence, net, r.concentrationRisk, passFloor);
      if (kill === null && net > passFloor) {
        familiesPassingEdgeFloor++;
        if (r.familyType === "exhaustive_partition_binary") exhaustivePartitionBinaryPasses = true;
      }
      if (net > bestNetEdge) {
        bestNetEdge = net;
        strongest = { familyId: r.familyId, familyType: r.familyType, netEdgeAfterHaircut: net };
      }
    }

    if (!Number.isFinite(bestNetEdge)) bestNetEdge = 0;

    let verdict: string;
    if (familiesPassingEdgeFloor > 0 && exhaustivePartitionBinaryPasses) {
      verdict = "families_pass_including_exhaustive_partition_binary";
    } else if (familiesPassingEdgeFloor > 0) {
      verdict = "families_pass_edge_floor_non_exhaustive_types_only";
    } else {
      verdict = "zero_families_pass_edge_floor_at_this_multiplier";
    }

    spreadCalibrationGrid.push({
      spreadCoefficientMultiplier: mult,
      effectiveSpreadCoefficient,
      familiesPassingEdgeFloor,
      exhaustivePartitionBinaryPasses,
      bestNetEdge: r6(bestNetEdge),
      strongestFamilyId: strongest?.familyId ?? null,
      strongestFamilyType: strongest?.familyType ?? null,
      verdict,
    });

    verdictBySpreadCoefficient[key] = verdict;
    familiesPassingBySpreadCoefficient[key] = familiesPassingEdgeFloor;
    bestNetBySpreadCoefficient[key] = r6(bestNetEdge);
    strongestFamilyBySpreadCoefficient[key] = strongest;

    if (familiesPassingEdgeFloor > 0) {
      if (minEffAny === null || effectiveSpreadCoefficient < minEffAny) {
        minEffAny = effectiveSpreadCoefficient;
      }
    }
    if (exhaustivePartitionBinaryPasses) {
      if (minEffExhaustive === null || effectiveSpreadCoefficient < minEffExhaustive) {
        minEffExhaustive = effectiveSpreadCoefficient;
      }
    }
  }

  const passAtBaseline =
    spreadCalibrationGrid.find(g => g.spreadCoefficientMultiplier === 1 && g.familiesPassingEdgeFloor > 0) !==
    undefined;
  const passAtOrAboveHalf = spreadCalibrationGrid.some(
    g => g.spreadCoefficientMultiplier >= 0.5 && g.familiesPassingEdgeFloor > 0,
  );
  const anyPass = spreadCalibrationGrid.some(g => g.familiesPassingEdgeFloor > 0);

  let spreadCalibrationAuditVerdict: SpreadCalibrationAuditVerdict;
  if (passAtBaseline) {
    spreadCalibrationAuditVerdict = "pass_at_baseline_multiplier";
  } else if (passAtOrAboveHalf) {
    spreadCalibrationAuditVerdict = "pass_at_or_above_half_multiplier_plausible_zone";
  } else if (!anyPass) {
    spreadCalibrationAuditVerdict = "no_family_passes_even_at_minimum_grid_multiplier";
  } else {
    spreadCalibrationAuditVerdict = "pass_only_at_multipliers_below_half";
  }

  const exMin = minEffExhaustive === null ? "never_on_grid" : String(r6(minEffExhaustive));
  const anyMin = minEffAny === null ? "never_on_grid" : String(r6(minEffAny));
  const bestNetAboveFloorButZeroPasses = spreadCalibrationGrid.some(
    g => g.bestNetEdge > passFloor && g.familiesPassingEdgeFloor === 0,
  );
  const gateNote = bestNetAboveFloorButZeroPasses
    ? " | note=max_net>floor_but_zero_passes=>semantic_or_concentration_kill_not_spread_only"
    : "";
  const spreadCalibrationSummaryLine = `spread_calibration_audit: verdict=${spreadCalibrationAuditVerdict} | min_effective_coef_any_pass=${anyMin} min_effective_coef_exhaustive_partition_binary=${exMin} | grid_multipliers=[${SPREAD_CALIBRATION_GRID_MULTIPLIERS.join(",")}]x base=${CONSTRAINT_FIRST_SPREAD_FRICTION_COEFFICIENT_BASE}${gateNote}`;

  return {
    spreadCalibrationGrid,
    verdictBySpreadCoefficient,
    familiesPassingBySpreadCoefficient,
    bestNetBySpreadCoefficient,
    strongestFamilyBySpreadCoefficient,
    minSpreadCoefficientForAnyPass: minEffAny === null ? null : r6(minEffAny),
    minSpreadCoefficientForExhaustivePartitionBinaryPass: minEffExhaustive === null ? null : r6(minEffExhaustive),
    spreadCalibrationAuditVerdict,
    spreadCalibrationSummaryLine,
  };
}

type ResidualKillAuditLayer = Pick<
  ConstraintFirstEdgeDiscoveryDigest,
  | "residualKillAuditGrid"
  | "topNearPassFamilies"
  | "killReasonByFamilyAtBestSpreadPoint"
  | "semanticKillCountBySpreadPoint"
  | "concentrationKillCountBySpreadPoint"
  | "mixedKillCountBySpreadPoint"
  | "bestNearPassFamily"
  | "bestNearPassFamilyKillAttribution"
  | "residualKillAuditVerdict"
  | "residualKillAuditSummaryLine"
>;

function computeResidualKillAuditLayer(
  rows: ConstraintFirstFamilyRow[],
  allMarkets: NormalizedMarket[],
  passFloor: number,
  spreadCalibration: SpreadCalibrationLayer,
): ResidualKillAuditLayer {
  const empty = (): ResidualKillAuditLayer => ({
    residualKillAuditGrid: [],
    topNearPassFamilies: [],
    killReasonByFamilyAtBestSpreadPoint: {},
    semanticKillCountBySpreadPoint: {},
    concentrationKillCountBySpreadPoint: {},
    mixedKillCountBySpreadPoint: {},
    bestNearPassFamily: null,
    bestNearPassFamilyKillAttribution: null,
    residualKillAuditVerdict: "insufficient_sample",
    residualKillAuditSummaryLine:
      "residual_kill_audit: insufficient_sample (empty_spread_grid_or_zero_resolvable_member_rows)",
  });

  if (rows.length === 0 || spreadCalibration.spreadCalibrationGrid.length === 0) return empty();

  const marketById = new Map<string, NormalizedMarket>();
  for (const m of allMarkets) marketById.set(m.id, m);
  const resolveMembers = (r: ConstraintFirstFamilyRow): NormalizedMarket[] => {
    const members: NormalizedMarket[] = [];
    for (const id of r.memberMarketIds) {
      const mm = marketById.get(id);
      if (mm) members.push(mm);
    }
    return members;
  };

  if (!rows.some(r => resolveMembers(r).length > 0)) return empty();

  const residualKillAuditGrid: ResidualKillAuditGridRow[] = [];
  const semanticKillCountBySpreadPoint: Record<string, number> = {};
  const concentrationKillCountBySpreadPoint: Record<string, number> = {};
  const mixedKillCountBySpreadPoint: Record<string, number> = {};

  for (const g of spreadCalibration.spreadCalibrationGrid) {
    const mult = g.spreadCoefficientMultiplier;
    const key = spreadCalibrationGridKey(mult);
    const eff = g.effectiveSpreadCoefficient;
    let sem = 0;
    let conc = 0;
    let mixed = 0;
    let other = 0;
    let passFull = 0;
    let totalR = 0;

    for (const r of rows) {
      const members = resolveMembers(r);
      if (members.length === 0) continue;
      totalR++;
      const frictionHyp = frictionEstimateWithSpreadCoefficient(members, eff);
      const netHyp = r6(r.rawEdge - frictionHyp - r.uncertaintyHaircut - r.modelRiskHaircut);
      const { semanticBlock, concBlock, netBlock } = residualParallelClassification(r, netHyp, passFloor);
      const cat = residualKillAttributionFromBlocks(semanticBlock, concBlock, netBlock);
      if (cat === "would_pass") passFull++;
      else if (cat === "both_semantic_and_concentration") mixed++;
      else if (cat === "semantic_confidence") sem++;
      else if (cat === "concentration_risk") conc++;
      else other++;
    }

    residualKillAuditGrid.push({
      spreadCoefficientMultiplier: mult,
      effectiveSpreadCoefficient: eff,
      semanticKillCount: sem,
      concentrationKillCount: conc,
      mixedKillCount: mixed,
      otherResidualKillCount: other,
      totalResolvableFamilies: totalR,
      familiesPassingFullKillStack: passFull,
    });
    semanticKillCountBySpreadPoint[key] = sem;
    concentrationKillCountBySpreadPoint[key] = conc;
    mixedKillCountBySpreadPoint[key] = mixed;
  }

  let bestPoint = spreadCalibration.spreadCalibrationGrid[0]!;
  for (const g of spreadCalibration.spreadCalibrationGrid) {
    if (g.bestNetEdge > bestPoint.bestNetEdge) bestPoint = g;
  }
  const bestMult = bestPoint.spreadCoefficientMultiplier;
  const bestEff = bestPoint.effectiveSpreadCoefficient;

  interface EvalRow {
    row: ConstraintFirstFamilyRow;
    netHyp: number;
    killOrdered: string | null;
    cat: ResidualKillAttributionCategory;
  }
  const evaluated: EvalRow[] = [];
  for (const r of rows) {
    const members = resolveMembers(r);
    if (members.length === 0) continue;
    const frictionHyp = frictionEstimateWithSpreadCoefficient(members, bestEff);
    const netHyp = r6(r.rawEdge - frictionHyp - r.uncertaintyHaircut - r.modelRiskHaircut);
    const { semanticBlock, concBlock, netBlock, killOrdered } = residualParallelClassification(r, netHyp, passFloor);
    const cat = residualKillAttributionFromBlocks(semanticBlock, concBlock, netBlock);
    evaluated.push({ row: r, netHyp, killOrdered, cat });
  }
  evaluated.sort((a, b) => b.netHyp - a.netHyp);

  const failing = evaluated.filter(e => e.cat !== "would_pass");
  const bestNearPassEntry = failing.length > 0 ? failing[0]! : null;

  let bestNearPassFamily: BestNearPassFamilySnapshot | null = null;
  let bestNearPassFamilyKillAttribution: ResidualKillAttributionDetail | null = null;
  if (bestNearPassEntry) {
    const r = bestNearPassEntry.row;
    const { semanticBlock, concBlock, netBlock } = residualParallelClassification(
      r,
      bestNearPassEntry.netHyp,
      passFloor,
    );
    bestNearPassFamily = {
      familyId: r.familyId,
      familyType: r.familyType,
      semanticConfidence: r.semanticConfidence,
      concentrationRisk: r.concentrationRisk,
      hypotheticalNetAtSpreadPoint: bestNearPassEntry.netHyp,
      spreadCoefficientMultiplier: bestMult,
      effectiveSpreadCoefficient: bestEff,
    };
    bestNearPassFamilyKillAttribution = {
      residualKillAttribution: bestNearPassEntry.cat,
      semanticBelowKillFloor: semanticBlock,
      concentrationAboveKillFloor: concBlock,
      netAtOrBelowPassFloor: netBlock,
      killReasonOrdered: bestNearPassEntry.killOrdered,
      spreadCoefficientMultiplier: bestMult,
      effectiveSpreadCoefficient: bestEff,
      hypotheticalNet: bestNearPassEntry.netHyp,
    };
  }

  const topNearPassFamilies: NearPassFamilyAudit[] = failing.slice(0, 15).map(e => ({
    familyId: e.row.familyId,
    familyType: e.row.familyType,
    hypotheticalNetAtSpreadPoint: e.netHyp,
    killReasonOrdered: e.killOrdered,
    residualKillAttribution: e.cat,
    semanticConfidence: e.row.semanticConfidence,
    concentrationRisk: e.row.concentrationRisk,
  }));

  const killReasonByFamilyAtBestSpreadPoint: Record<string, string | null> = {};
  for (const e of failing.slice(0, 20)) {
    killReasonByFamilyAtBestSpreadPoint[e.row.familyId] = e.killOrdered;
  }

  const snapshotRow =
    residualKillAuditGrid.find(x => x.spreadCoefficientMultiplier === bestMult) ?? residualKillAuditGrid[0]!;
  const S = snapshotRow.semanticKillCount;
  const C = snapshotRow.concentrationKillCount;
  const M = snapshotRow.mixedKillCount;
  const O = snapshotRow.otherResidualKillCount;
  const nonPass = S + C + M + O;

  let residualKillAuditVerdict: ResidualKillAuditVerdict;
  if (nonPass === 0) {
    residualKillAuditVerdict = "balanced_residual_attribution_at_best_spread_point";
  } else {
    const maxV = Math.max(S, C, M, O);
    const tops = [
      { tag: "mixed" as const, v: M },
      { tag: "semantic" as const, v: S },
      { tag: "conc" as const, v: C },
      { tag: "other" as const, v: O },
    ].filter(x => x.v === maxV && x.v > 0);
    if (tops.length !== 1) {
      residualKillAuditVerdict = "balanced_residual_attribution_at_best_spread_point";
    } else if (tops[0]!.tag === "mixed") {
      residualKillAuditVerdict = "mixed_semantic_and_concentration_at_best_spread_point";
    } else if (tops[0]!.tag === "semantic") {
      residualKillAuditVerdict = "semantic_confidence_dominates_at_best_spread_point";
    } else if (tops[0]!.tag === "conc") {
      residualKillAuditVerdict = "concentration_risk_dominates_at_best_spread_point";
    } else {
      residualKillAuditVerdict = "net_floor_only_other_residual_at_best_spread_point";
    }
  }

  const bn = bestNearPassFamily?.familyId ?? "n/a";
  const residualKillAuditSummaryLine = `residual_kill_audit: best_spread_point_mult=${bestMult} bestNetEdge=${r6(bestPoint.bestNetEdge)} | at_that_point non_pass={semantic_only=${S} conc_only=${C} both=${M} net_only=${O}} | verdict=${residualKillAuditVerdict} | strongest_near_pass=${bn}`;

  return {
    residualKillAuditGrid,
    topNearPassFamilies,
    killReasonByFamilyAtBestSpreadPoint,
    semanticKillCountBySpreadPoint,
    concentrationKillCountBySpreadPoint,
    mixedKillCountBySpreadPoint,
    bestNearPassFamily,
    bestNearPassFamilyKillAttribution,
    residualKillAuditVerdict,
    residualKillAuditSummaryLine,
  };
}

/** Variação mínima de `concentrationRisk` para considerar o proxy informativo intra-tipo (multi-membro). */
const MEANINGFUL_CONCENTRATION_RANGE = 0.015;

function bestSpreadStressPointFromGrid(grid: SpreadCalibrationGridRow[]): {
  spreadCoefficientMultiplier: number;
  effectiveSpreadCoefficient: number;
} {
  if (grid.length === 0) {
    return {
      spreadCoefficientMultiplier: 1,
      effectiveSpreadCoefficient: CONSTRAINT_FIRST_SPREAD_FRICTION_COEFFICIENT_BASE,
    };
  }
  let best = grid[0]!;
  for (const g of grid) {
    if (g.bestNetEdge > best.bestNetEdge) best = g;
  }
  return {
    spreadCoefficientMultiplier: best.spreadCoefficientMultiplier,
    effectiveSpreadCoefficient: best.effectiveSpreadCoefficient,
  };
}

type ConcentrationAuditLayer = Pick<
  ConstraintFirstEdgeDiscoveryDigest,
  | "concentrationAuditByFamilyType"
  | "concentrationStructuralDegeneracyByFamilyType"
  | "singleMarketFamilyTypes"
  | "multiMarketFamilyTypes"
  | "concentrationInformativeFamilyTypes"
  | "concentrationDegenerateFamilyTypes"
  | "bestFamiliesIfConcentrationIgnoredWhenStructurallyDegenerate"
  | "concentrationAuditVerdict"
  | "concentrationAuditSummaryLine"
>;

function computeConcentrationAuditLayer(
  rows: ConstraintFirstFamilyRow[],
  passFloor: number,
  residualKillAudit: ResidualKillAuditLayer,
  spreadCalibration: SpreadCalibrationLayer,
): ConcentrationAuditLayer {
  const nullAudit = (): Record<ConstraintFamilyType, ConcentrationFamilyTypeAudit | null> => ({
    exhaustive_partition_binary: null,
    strong_dominance_binary: null,
    same_event_subfamily: null,
  });
  const falseRec = (): Record<ConstraintFamilyType, boolean> => ({
    exhaustive_partition_binary: false,
    strong_dominance_binary: false,
    same_event_subfamily: false,
  });

  const empty = (): ConcentrationAuditLayer => ({
    concentrationAuditByFamilyType: nullAudit(),
    concentrationStructuralDegeneracyByFamilyType: falseRec(),
    singleMarketFamilyTypes: [],
    multiMarketFamilyTypes: [],
    concentrationInformativeFamilyTypes: [],
    concentrationDegenerateFamilyTypes: [],
    bestFamiliesIfConcentrationIgnoredWhenStructurallyDegenerate: [],
    concentrationAuditVerdict: "insufficient_sample",
    concentrationAuditSummaryLine:
      "concentration_semantic_audit: insufficient_sample (zero_scored_rows)",
  });

  if (rows.length === 0) return empty();

  const concentrationAuditByFamilyType = nullAudit();
  const concentrationStructuralDegeneracyByFamilyType = falseRec();
  const singleMarketFamilyTypes: ConstraintFamilyType[] = [];
  const multiMarketFamilyTypes: ConstraintFamilyType[] = [];
  const concentrationInformativeFamilyTypes: ConstraintFamilyType[] = [];
  const concentrationDegenerateFamilyTypes: ConstraintFamilyType[] = [];

  for (const t of ALL_FAMILY_TYPES) {
    const arr = rows.filter(r => r.familyType === t);
    if (arr.length === 0) continue;

    const memberCounts = arr.map(r => r.memberMarketIds.length);
    const minM = Math.min(...memberCounts);
    const maxM = Math.max(...memberCounts);
    const allFamiliesSingleMember = maxM === 1 && minM === 1;

    const concs = arr.map(r => r.concentrationRisk);
    const cmin = Math.min(...concs);
    const cmax = Math.max(...concs);
    const cmean = meanOrNull(concs);
    const concentrationRange = r6(cmax - cmin);

    const structurallyDegenerateConcentrationSemantics =
      allFamiliesSingleMember || (minM === 1 && maxM === 1 && cmin >= 1 - 1e-5 && cmax <= 1 + 1e-5);

    const nearConstantByConstruction =
      allFamiliesSingleMember || (maxM >= 2 && concentrationRange < 1e-6);
    const variesMeaningfullyWithinType =
      maxM >= 2 && concentrationRange >= MEANINGFUL_CONCENTRATION_RANGE;
    const informativeForEconomicFragility =
      !structurallyDegenerateConcentrationSemantics &&
      maxM >= 2 &&
      concentrationRange >= MEANINGFUL_CONCENTRATION_RANGE;

    const rationaleLine = structurallyDegenerateConcentrationSemantics
      ? `type=${t}: single_member_concentration=max_liquidity_share_is_1_by_definition | not_analogous_to_multi_leg_concentration`
      : !variesMeaningfullyWithinType
        ? `type=${t}: multi_member_but_concentration_range=${concentrationRange} (below_meaningful_dispersion_threshold)`
        : `type=${t}: concentration_range=${concentrationRange} | usable_fragility_proxy_within_sample`;

    concentrationAuditByFamilyType[t] = {
      sampleSize: arr.length,
      memberCountMin: minM,
      memberCountMax: maxM,
      allFamiliesSingleMember,
      concentrationMin: r6(cmin),
      concentrationMax: r6(cmax),
      concentrationMean: cmean,
      concentrationRange,
      variesMeaningfullyWithinType,
      nearConstantByConstruction,
      informativeForEconomicFragility,
      structurallyDegenerateConcentrationSemantics,
      rationaleLine,
    };
    concentrationStructuralDegeneracyByFamilyType[t] = structurallyDegenerateConcentrationSemantics;

    if (allFamiliesSingleMember) singleMarketFamilyTypes.push(t);
    else multiMarketFamilyTypes.push(t);
    if (informativeForEconomicFragility) concentrationInformativeFamilyTypes.push(t);
    if (structurallyDegenerateConcentrationSemantics) concentrationDegenerateFamilyTypes.push(t);
  }

  const stress = residualKillAudit.bestNearPassFamily
    ? {
        spreadCoefficientMultiplier: residualKillAudit.bestNearPassFamily.spreadCoefficientMultiplier,
        effectiveSpreadCoefficient: residualKillAudit.bestNearPassFamily.effectiveSpreadCoefficient,
      }
    : bestSpreadStressPointFromGrid(spreadCalibration.spreadCalibrationGrid);

  const rowById = new Map(rows.map(r => [r.familyId, r] as const));
  const bestFamiliesIfConcentrationIgnoredWhenStructurallyDegenerate: ConcentrationIgnoreCounterfactualRow[] = [];

  for (const np of residualKillAudit.topNearPassFamilies.slice(0, 15)) {
    const row = rowById.get(np.familyId);
    if (!row) continue;
    const degenerate = concentrationStructuralDegeneracyByFamilyType[row.familyType];
    const net = np.hypotheticalNetAtSpreadPoint;
    const killProd = computeKillReasonForRow(row.semanticConfidence, net, row.concentrationRisk, passFloor);
    const killIgnore = degenerate
      ? computeKillReasonForRow(row.semanticConfidence, net, 0, passFloor)
      : killProd;
    bestFamiliesIfConcentrationIgnoredWhenStructurallyDegenerate.push({
      familyId: row.familyId,
      familyType: row.familyType,
      hypotheticalNetAtBestSpreadStress: r6(net),
      spreadCoefficientMultiplier: stress.spreadCoefficientMultiplier,
      effectiveSpreadCoefficient: stress.effectiveSpreadCoefficient,
      familyTypeStructurallyDegenerateForConcentration: degenerate,
      killReasonOrderedAtHypotheticalNet: killProd,
      killReasonOrderedIfConcIgnoredWhenDegenerate: killIgnore,
      wouldPassFullKillStackUnderCounterfactual: killIgnore === null && net > passFloor,
    });
  }

  const hasDeg = concentrationDegenerateFamilyTypes.length > 0;
  const hasInf = concentrationInformativeFamilyTypes.length > 0;
  const hasMultiSamples = multiMarketFamilyTypes.length > 0;

  let concentrationAuditVerdict: ConcentrationAuditVerdict;
  if (hasDeg && hasInf) {
    concentrationAuditVerdict =
      "multi_market_family_types_concentration_informative_single_market_types_degenerate";
  } else if (hasDeg && !hasInf && hasMultiSamples) {
    concentrationAuditVerdict = "mixed_or_partial_sample_concentration_semantics";
  } else if (hasDeg && !hasInf) {
    concentrationAuditVerdict = "single_market_family_types_concentration_metric_structurally_degenerate";
  } else if (!hasDeg && hasInf) {
    concentrationAuditVerdict = "concentration_informative_across_family_types_with_samples";
  } else {
    concentrationAuditVerdict = "mixed_or_partial_sample_concentration_semantics";
  }

  const exDeg = concentrationStructuralDegeneracyByFamilyType.exhaustive_partition_binary;
  const topCf = bestFamiliesIfConcentrationIgnoredWhenStructurallyDegenerate[0];
  const topPassesCf = topCf?.wouldPassFullKillStackUnderCounterfactual === true;

  const concentrationAuditSummaryLine = `concentration_semantic_audit: verdict=${concentrationAuditVerdict} | degenerate_types=[${concentrationDegenerateFamilyTypes.join(",")}] informative_types=[${concentrationInformativeFamilyTypes.join(",")}] | exhaustive_partition_binary_degenerate=${exDeg} | top_near_pass_counterfactual_would_pass=${topPassesCf} (diagnostic_only_not_live_rule)`;

  return {
    concentrationAuditByFamilyType,
    concentrationStructuralDegeneracyByFamilyType,
    singleMarketFamilyTypes,
    multiMarketFamilyTypes,
    concentrationInformativeFamilyTypes,
    concentrationDegenerateFamilyTypes,
    bestFamiliesIfConcentrationIgnoredWhenStructurallyDegenerate,
    concentrationAuditVerdict,
    concentrationAuditSummaryLine,
  };
}

type ClassAwareConcentrationPolicyLayer = Pick<
  ConstraintFirstEdgeDiscoveryDigest,
  | "concentrationPolicyByFamilyType"
  | "concentrationKillModeByFamilyType"
  | "familiesPassingEdgeFloorUnderClassAwareConcentration"
  | "topFamiliesUnderClassAwareConcentration"
  | "strongestFamilyUnderClassAwareConcentration"
  | "classAwareConstraintFirstVerdict"
  | "classAwareConstraintFirstSummaryLine"
  | "constraintFirstVerdictComparisonLine"
  | "exhaustivePartitionBinaryHasClassAwarePass"
  | "sameEventSubfamilyHasClassAwarePass"
>;

function computeClassAwareConcentrationPolicyLayer(
  rows: ConstraintFirstFamilyRow[],
  passFloor: number,
  viableTh: number,
  promisingTh: number,
  originalVerdict: ConstraintFirstVerdict,
  originalPassCount: number,
  concentrationAudit: ConcentrationAuditLayer,
): ClassAwareConcentrationPolicyLayer {
  const defaultModes = (): Record<
    ConstraintFamilyType,
    "hard_kill_active" | "diagnostic_no_hard_kill" | "not_applicable_no_sample"
  > => ({
    exhaustive_partition_binary: "not_applicable_no_sample",
    strong_dominance_binary: "not_applicable_no_sample",
    same_event_subfamily: "not_applicable_no_sample",
  });
  const defaultPolicies = (): Record<ConstraintFamilyType, string> => ({
    exhaustive_partition_binary: "no_sample",
    strong_dominance_binary: "no_sample",
    same_event_subfamily: "no_sample",
  });

  if (rows.length === 0) {
    return {
      concentrationPolicyByFamilyType: defaultPolicies(),
      concentrationKillModeByFamilyType: defaultModes(),
      familiesPassingEdgeFloorUnderClassAwareConcentration: 0,
      topFamiliesUnderClassAwareConcentration: [],
      strongestFamilyUnderClassAwareConcentration: null,
      classAwareConstraintFirstVerdict: "no_viable_family_found",
      classAwareConstraintFirstSummaryLine:
        "class_aware_conc_policy: no_rows | pass=0 | verdict=no_viable_family_found",
      constraintFirstVerdictComparisonLine: `verdict_compare: original=${originalVerdict} class_aware_concentration=no_viable_family_found | pass_counts original=${originalPassCount} class_aware=0`,
      exhaustivePartitionBinaryHasClassAwarePass: false,
      sameEventSubfamilyHasClassAwarePass: false,
    };
  }

  const deg = concentrationAudit.concentrationStructuralDegeneracyByFamilyType;
  const byAud = concentrationAudit.concentrationAuditByFamilyType;

  const concentrationPolicyByFamilyType = defaultPolicies();
  const concentrationKillModeByFamilyType = defaultModes();

  for (const t of ALL_FAMILY_TYPES) {
    const aud = byAud[t];
    if (!aud) {
      concentrationPolicyByFamilyType[t] = "no_sample_for_type_in_this_run";
      concentrationKillModeByFamilyType[t] = "not_applicable_no_sample";
      continue;
    }
    if (deg[t]) {
      concentrationKillModeByFamilyType[t] = "diagnostic_no_hard_kill";
      concentrationPolicyByFamilyType[t] =
        "structurally_degenerate_single_market: concentrationRisk_retained_on_row_CONC_KILL_bypassed | model_haircuts_unchanged";
    } else if (aud.informativeForEconomicFragility) {
      concentrationKillModeByFamilyType[t] = "hard_kill_active";
      concentrationPolicyByFamilyType[t] =
        "multi_market_informative_dispersion: standard_CONC_KILL_after_semantic_and_net";
    } else {
      concentrationKillModeByFamilyType[t] = "hard_kill_active";
      concentrationPolicyByFamilyType[t] =
        "multi_market_default: global_CONC_KILL_semantics_thresholds_unchanged";
    }
  }

  const passingCA = rows.filter(
    r =>
      computeKillReasonForRowClassAware(r, r.netEdgeAfterHaircut, passFloor, deg) === null &&
      r.netEdgeAfterHaircut > passFloor,
  );
  passingCA.sort((a, b) => b.netEdgeAfterHaircut - a.netEdgeAfterHaircut);

  const familiesPassingEdgeFloorUnderClassAwareConcentration = passingCA.length;
  const topFamiliesUnderClassAwareConcentration = passingCA.slice(0, 8);
  const strongestFamilyUnderClassAwareConcentration = passingCA.length > 0 ? passingCA[0]! : null;

  const bestNetCA =
    passingCA.length > 0 ? passingCA[0]!.netEdgeAfterHaircut : rows[0]!.netEdgeAfterHaircut;

  let classAwareConstraintFirstVerdict: ConstraintFirstVerdict = "no_viable_family_found";
  if (passingCA.length === 0) {
    classAwareConstraintFirstVerdict = bestNetCA > 0 ? "weak_candidates_only" : "no_viable_family_found";
  } else {
    const top = passingCA[0]!;
    if (
      top.netEdgeAfterHaircut >= promisingTh &&
      top.recurrenceObservationCount >= 2 &&
      top.concentrationRisk < 0.48
    ) {
      classAwareConstraintFirstVerdict = "promising_candidate_present";
    } else if (top.netEdgeAfterHaircut >= viableTh) {
      classAwareConstraintFirstVerdict = "viable_candidate_present";
    } else {
      classAwareConstraintFirstVerdict = "weak_candidates_only";
    }
  }

  const exhaustivePartitionBinaryHasClassAwarePass = passingCA.some(
    x => x.familyType === "exhaustive_partition_binary",
  );
  const sameEventSubfamilyHasClassAwarePass = passingCA.some(x => x.familyType === "same_event_subfamily");

  const classAwareConstraintFirstSummaryLine = `class_aware_conc_policy: verdict=${classAwareConstraintFirstVerdict} | pass_edge_floor=${familiesPassingEdgeFloorUnderClassAwareConcentration} (vs_original=${originalPassCount}) | exhaustive_partition_pass=${exhaustivePartitionBinaryHasClassAwarePass} same_event_pass=${sameEventSubfamilyHasClassAwarePass}`;

  const constraintFirstVerdictComparisonLine = `verdict_compare: original=${originalVerdict} class_aware_concentration=${classAwareConstraintFirstVerdict} | pass_counts original=${originalPassCount} class_aware=${familiesPassingEdgeFloorUnderClassAwareConcentration}`;

  return {
    concentrationPolicyByFamilyType,
    concentrationKillModeByFamilyType,
    familiesPassingEdgeFloorUnderClassAwareConcentration,
    topFamiliesUnderClassAwareConcentration,
    strongestFamilyUnderClassAwareConcentration,
    classAwareConstraintFirstVerdict,
    classAwareConstraintFirstSummaryLine,
    constraintFirstVerdictComparisonLine,
    exhaustivePartitionBinaryHasClassAwarePass,
    sameEventSubfamilyHasClassAwarePass,
  };
}

function postConcentrationDominantCategory(
  row: ConstraintFirstFamilyRow,
  passFloor: number,
  deg: Record<ConstraintFamilyType, boolean>,
): PostConcentrationResidualDomCategory {
  const kill = computeKillReasonForRowClassAware(row, row.netEdgeAfterHaircut, passFloor, deg);
  if (kill === null && row.netEdgeAfterHaircut > passFloor) return "would_pass";
  if (kill === "semantic_confidence_below_floor") return "semantic";
  if (kill === "concentration_too_high") return "concentration";
  if (kill !== null && kill.startsWith("net_edge")) return "net_below_pass_floor";
  if (kill === null && row.netEdgeAfterHaircut <= passFloor) return "net_below_pass_floor";
  return "none";
}

function buildPostConcentrationBreakdown(
  row: ConstraintFirstFamilyRow,
  passFloor: number,
  deg: Record<ConstraintFamilyType, boolean>,
): PostConcentrationFamilyResidualBreakdown {
  const kill = computeKillReasonForRowClassAware(row, row.netEdgeAfterHaircut, passFloor, deg);
  const cat = postConcentrationDominantCategory(row, passFloor, deg);
  const semTh = envNum("CONSTRAINT_FIRST_SEMANTIC_KILL", 0.58);
  const semanticKillContributionNote =
    row.semanticConfidence >= semTh
      ? `semantic_above_kill_floor(>=${semTh})`
      : `semantic_below_kill_floor(<${semTh})`;
  const recurrenceContributionNote = `recurrenceObservationCount=${row.recurrenceObservationCount} persistenceScore=${r6(row.persistenceScore)} | paper_promising_band_also_requires_recurrence_ge_2`;
  return {
    familyId: row.familyId,
    familyType: row.familyType,
    rawEdge: row.rawEdge,
    frictionCostEstimate: row.frictionCostEstimate,
    uncertaintyHaircut: row.uncertaintyHaircut,
    modelRiskHaircut: row.modelRiskHaircut,
    netEdgeAfterHaircut: row.netEdgeAfterHaircut,
    residualGapToPassFloor: r6(Math.max(0, passFloor - row.netEdgeAfterHaircut)),
    concentrationRisk: row.concentrationRisk,
    semanticConfidence: row.semanticConfidence,
    recurrenceObservationCount: row.recurrenceObservationCount,
    persistenceScore: row.persistenceScore,
    classAwareKillReason: kill,
    dominantResidualCategory: cat,
    semanticKillContributionNote,
    recurrenceContributionNote,
    haircutStackUncertaintyPlusModel: r6(row.uncertaintyHaircut + row.modelRiskHaircut),
    impliedRequiredRawToClearNetPassFloor: r6(
      passFloor + row.frictionCostEstimate + row.uncertaintyHaircut + row.modelRiskHaircut,
    ),
  };
}

type PostConcentrationResidualFeasibilityLayer = Pick<
  ConstraintFirstEdgeDiscoveryDigest,
  | "postConcentrationResidualAudit"
  | "topNearPassFamiliesPostConcentration"
  | "dominantResidualKillByFamilyType"
  | "avgResidualGapToPassFloorByFamilyType"
  | "strongestNearPassResidualBreakdown"
  | "residualKillCountsByType"
  | "residualEconomicFailureModes"
  | "postConcentrationResidualAuditVerdict"
  | "postConcentrationResidualSummaryLine"
>;

function computePostConcentrationResidualFeasibilityLayer(
  rows: ConstraintFirstFamilyRow[],
  passFloor: number,
  concentrationAudit: ConcentrationAuditLayer,
  classAwarePassCount: number,
): PostConcentrationResidualFeasibilityLayer {
  const nullGapRec = (): Record<ConstraintFamilyType, number | null> => ({
    exhaustive_partition_binary: null,
    strong_dominance_binary: null,
    same_event_subfamily: null,
  });
  const emptyCounts = (): Record<string, number> => ({
    semantic: 0,
    net_below_pass_floor: 0,
    concentration: 0,
    would_pass: 0,
    none: 0,
  });
  const empty = (): PostConcentrationResidualFeasibilityLayer => ({
    postConcentrationResidualAudit: [],
    topNearPassFamiliesPostConcentration: [],
    dominantResidualKillByFamilyType: {
      exhaustive_partition_binary: "no_sample",
      strong_dominance_binary: "no_sample",
      same_event_subfamily: "no_sample",
    },
    avgResidualGapToPassFloorByFamilyType: nullGapRec(),
    strongestNearPassResidualBreakdown: null,
    residualKillCountsByType: {
      exhaustive_partition_binary: emptyCounts(),
      strong_dominance_binary: emptyCounts(),
      same_event_subfamily: emptyCounts(),
    },
    residualEconomicFailureModes: [],
    postConcentrationResidualAuditVerdict: "insufficient_sample",
    postConcentrationResidualSummaryLine:
      "post_conc_residual: insufficient_sample (zero_rows) | class_aware_pass=0",
  });

  if (rows.length === 0) return empty();

  const deg = concentrationAudit.concentrationStructuralDegeneracyByFamilyType;

  const passesClassAware = (r: ConstraintFirstFamilyRow) =>
    computeKillReasonForRowClassAware(r, r.netEdgeAfterHaircut, passFloor, deg) === null &&
    r.netEdgeAfterHaircut > passFloor;

  const failing = rows.filter(r => !passesClassAware(r));
  failing.sort((a, b) => b.netEdgeAfterHaircut - a.netEdgeAfterHaircut);

  const postConcentrationResidualAudit: PostConcentrationFamilyResidualBreakdown[] = failing
    .slice(0, 18)
    .map(r => buildPostConcentrationBreakdown(r, passFloor, deg));

  const topNearPassFamiliesPostConcentration: PostConcentrationNearPassSummary[] = postConcentrationResidualAudit
    .slice(0, 15)
    .map(x => ({
      familyId: x.familyId,
      familyType: x.familyType,
      netEdgeAfterHaircut: x.netEdgeAfterHaircut,
      residualGapToPassFloor: x.residualGapToPassFloor,
      classAwareKillReason: x.classAwareKillReason,
      dominantResidualCategory: x.dominantResidualCategory,
    }));

  const strongestNearPassResidualBreakdown =
    postConcentrationResidualAudit.length > 0 ? postConcentrationResidualAudit[0]! : null;

  const residualKillCountsByType: Record<ConstraintFamilyType, Record<string, number>> = {
    exhaustive_partition_binary: emptyCounts(),
    strong_dominance_binary: emptyCounts(),
    same_event_subfamily: emptyCounts(),
  };

  const dominantResidualKillByFamilyType: Record<ConstraintFamilyType, string> = {
    exhaustive_partition_binary: "no_sample",
    strong_dominance_binary: "no_sample",
    same_event_subfamily: "no_sample",
  };

  const avgResidualGapToPassFloorByFamilyType = nullGapRec();

  for (const t of ALL_FAMILY_TYPES) {
    const arr = rows.filter(r => r.familyType === t);
    if (arr.length === 0) continue;
    const gaps = arr.map(r => Math.max(0, passFloor - r.netEdgeAfterHaircut));
    avgResidualGapToPassFloorByFamilyType[t] = meanOrNull(gaps);

    for (const r of arr) {
      const c = postConcentrationDominantCategory(r, passFloor, deg);
      residualKillCountsByType[t][c] = (residualKillCountsByType[t][c] ?? 0) + 1;
    }
    const tc = residualKillCountsByType[t];
    const order = ["net_below_pass_floor", "semantic", "concentration", "none", "would_pass"];
    let bestK = "none";
    let bestV = -1;
    for (const k of order) {
      const v = tc[k] ?? 0;
      if (v > bestV) {
        bestV = v;
        bestK = k;
      }
    }
    dominantResidualKillByFamilyType[t] = bestK;
  }

  let netBelow = 0;
  let semBelow = 0;
  let concKill = 0;
  for (const r of rows) {
    if (passesClassAware(r)) continue;
    const kill = computeKillReasonForRowClassAware(r, r.netEdgeAfterHaircut, passFloor, deg);
    if (kill !== null && kill.startsWith("net_edge")) netBelow++;
    else if (kill === "semantic_confidence_below_floor") semBelow++;
    else if (kill === "concentration_too_high") concKill++;
    else if (kill === null && r.netEdgeAfterHaircut <= passFloor) netBelow++;
  }
  const failN = netBelow + semBelow + concKill;

  const partRows = rows.filter(r => r.familyType === "exhaustive_partition_binary");
  const bestPartNet = partRows.length ? Math.max(...partRows.map(r => r.netEdgeAfterHaircut)) : -999;
  const NEAR_BAND = 0.003;
  const DEAD_BAND = 0.025;

  const borderlinePaper = rows.some(r => {
    const kill = computeKillReasonForRowClassAware(r, r.netEdgeAfterHaircut, passFloor, deg);
    return (
      r.netEdgeAfterHaircut >= passFloor - 0.002 &&
      r.netEdgeAfterHaircut < passFloor &&
      kill !== null &&
      kill.startsWith("net_edge")
    );
  });

  const modes: string[] = [];
  if (failN > 0) {
    modes.push(`class_aware_failures_total=${failN}`);
    modes.push(`net_below_pass_floor=${netBelow}`);
    modes.push(`semantic_floor=${semBelow}`);
    modes.push(`concentration_kill=${concKill}`);
  }
  if (borderlinePaper) modes.push("borderline_slice_net_only_within_2bp_of_floor_observed");
  if (classAwarePassCount > 0) modes.push(`class_aware_pass_count=${classAwarePassCount}`);
  const residualEconomicFailureModes = modes;

  let postConcentrationResidualAuditVerdict: PostConcentrationResidualAuditVerdict;
  if (partRows.length > 0 && bestPartNet >= passFloor - NEAR_BAND && bestPartNet <= passFloor) {
    postConcentrationResidualAuditVerdict =
      "exhaustive_partition_binary_economically_near_pass_under_class_aware_conc";
  } else if (partRows.length > 0 && bestPartNet < passFloor - DEAD_BAND) {
    postConcentrationResidualAuditVerdict = "exhaustive_partition_binary_economically_distant_dead_proxy";
  } else if (failN > 0 && netBelow >= semBelow * 4 && netBelow >= concKill * 4) {
    postConcentrationResidualAuditVerdict = "residual_dominated_by_net_below_pass_floor";
  } else if (failN > 0 && semBelow >= netBelow * 0.35 && semBelow >= 8) {
    postConcentrationResidualAuditVerdict = "residual_semantic_floor_material_across_types";
  } else if (failN > 0 && semBelow > 0 && netBelow > 0) {
    postConcentrationResidualAuditVerdict = "residual_mixed_net_and_semantic";
  } else {
    postConcentrationResidualAuditVerdict =
      "multi_family_types_net_haircut_stack_collapses_under_production_friction";
  }

  const topDom = strongestNearPassResidualBreakdown?.dominantResidualCategory ?? "n/a";
  const postConcentrationResidualSummaryLine = `post_conc_residual: verdict=${postConcentrationResidualAuditVerdict} | strongest_near_pass=${strongestNearPassResidualBreakdown?.familyId ?? "n/a"} dom=${topDom} gap=${strongestNearPassResidualBreakdown?.residualGapToPassFloor ?? "n/a"} | best_partition_net=${r6(bestPartNet)} class_aware_pass=${classAwarePassCount} | modes=[${residualEconomicFailureModes.join(";")}]`;

  return {
    postConcentrationResidualAudit,
    topNearPassFamiliesPostConcentration,
    dominantResidualKillByFamilyType,
    avgResidualGapToPassFloorByFamilyType,
    strongestNearPassResidualBreakdown,
    residualKillCountsByType,
    residualEconomicFailureModes,
    postConcentrationResidualAuditVerdict,
    postConcentrationResidualSummaryLine,
  };
}

const PARTITION_REALITY_AUDIT_JSON_CAP = 48;

function buildPartitionRealityRow(
  r: ConstraintFirstFamilyRow,
  m: NormalizedMarket,
  passFloor: number,
): PartitionRawEdgeRealityAuditRow {
  const obsDir = observedPartitionMispricingDirect(m);
  const obsComb = observedPartitionMispricingCombined(m);
  const gapD = r6(obsDir - r.rawEdge);
  const gapC = r6(obsComb - r.rawEdge);
  const ratio = r.rawEdge > 1e-9 ? r6(obsComb / r.rawEdge) : null;
  const implied = r6(passFloor + r.frictionCostEstimate + r.uncertaintyHaircut + r.modelRiskHaircut);
  const netObs = r6(obsComb - r.frictionCostEstimate - r.uncertaintyHaircut - r.modelRiskHaircut);
  return {
    familyId: r.familyId,
    rawEdge: r.rawEdge,
    observedPartitionMispricingDirect: obsDir,
    observedPartitionMispricingCombined: obsComb,
    gapDirectMinusRaw: gapD,
    gapCombinedMinusRaw: gapC,
    compressionRatioCombinedOverRaw: ratio,
    impliedRequiredRawToClearNetPassFloor: implied,
    netUnderObservedCombinedProxy: netObs,
    probSum: r6(m.probSum),
    spread: r6(m.spread),
  };
}

function observedMispricingCombinedForRowMembers(
  r: ConstraintFirstFamilyRow,
  marketById: Map<string, NormalizedMarket>,
): number | null {
  const ms: NormalizedMarket[] = [];
  for (const id of r.memberMarketIds) {
    const mm = marketById.get(id);
    if (mm) ms.push(mm);
  }
  if (ms.length === 0) return null;
  if (r.familyType === "exhaustive_partition_binary" || r.familyType === "strong_dominance_binary") {
    return observedPartitionMispricingCombined(ms[0]!);
  }
  const vals = ms.map(observedPartitionMispricingCombined);
  return r6(vals.reduce((a, b) => a + b, 0) / vals.length);
}

type RawEdgeRealityAuditLayer = Pick<
  ConstraintFirstEdgeDiscoveryDigest,
  | "rawEdgeRealityAudit"
  | "rawEdgeByFamilyType"
  | "rawEdgeVsObservedMispricingByFamilyType"
  | "bestObservedMispricingByFamilyType"
  | "avgObservedMispricingByFamilyType"
  | "rawEdgeCompressionRatioByFamilyType"
  | "strongestPartitionRealityCheck"
  | "partitionRealityCheckSamples"
  | "rawEdgeRealityAuditVerdict"
  | "rawEdgeRealitySummaryLine"
>;

function computeRawEdgeRealityAuditLayer(
  rows: ConstraintFirstFamilyRow[],
  allMarkets: NormalizedMarket[],
  passFloor: number,
  strongestResidual: PostConcentrationFamilyResidualBreakdown | null,
): RawEdgeRealityAuditLayer {
  const nullStats = (): Record<ConstraintFamilyType, RawEdgeStats | null> => ({
    exhaustive_partition_binary: null,
    strong_dominance_binary: null,
    same_event_subfamily: null,
  });
  const nullNum = (): Record<ConstraintFamilyType, number | null> => ({
    exhaustive_partition_binary: null,
    strong_dominance_binary: null,
    same_event_subfamily: null,
  });

  const empty = (): RawEdgeRealityAuditLayer => ({
    rawEdgeRealityAudit: [],
    rawEdgeByFamilyType: nullStats(),
    rawEdgeVsObservedMispricingByFamilyType: nullNum(),
    bestObservedMispricingByFamilyType: nullNum(),
    avgObservedMispricingByFamilyType: nullNum(),
    rawEdgeCompressionRatioByFamilyType: nullNum(),
    strongestPartitionRealityCheck: null,
    partitionRealityCheckSamples: [],
    rawEdgeRealityAuditVerdict: "insufficient_sample",
    rawEdgeRealitySummaryLine:
      "raw_edge_reality: insufficient_sample (zero_rows) | partition_audit_rows=0",
  });

  if (rows.length === 0) return empty();

  const marketById = new Map<string, NormalizedMarket>();
  for (const m of allMarkets) marketById.set(m.id, m);

  const rawEdgeByFamilyType = nullStats();
  const rawEdgeVsObservedMispricingByFamilyType = nullNum();
  const bestObservedMispricingByFamilyType = nullNum();
  const avgObservedMispricingByFamilyType = nullNum();
  const rawEdgeCompressionRatioByFamilyType = nullNum();

  for (const t of ALL_FAMILY_TYPES) {
    const arr = rows.filter(r => r.familyType === t);
    if (arr.length === 0) continue;
    const raws = arr.map(x => x.rawEdge);
    rawEdgeByFamilyType[t] = {
      min: r6(Math.min(...raws)),
      max: r6(Math.max(...raws)),
      mean: meanOrNull(raws)!,
      sampleSize: arr.length,
    };
    const gaps: number[] = [];
    const obsList: number[] = [];
    const ratios: number[] = [];
    for (const r of arr) {
      const obs = observedMispricingCombinedForRowMembers(r, marketById);
      if (obs === null) continue;
      obsList.push(obs);
      gaps.push(r6(obs - r.rawEdge));
      if (r.rawEdge > 1e-9) ratios.push(obs / r.rawEdge);
    }
    if (obsList.length > 0) {
      rawEdgeVsObservedMispricingByFamilyType[t] = meanOrNull(gaps);
      bestObservedMispricingByFamilyType[t] = r6(Math.max(...obsList));
      avgObservedMispricingByFamilyType[t] = meanOrNull(obsList);
      rawEdgeCompressionRatioByFamilyType[t] = ratios.length ? meanOrNull(ratios.map(r6)) : null;
    }
  }

  const partitionRows = rows.filter(r => r.familyType === "exhaustive_partition_binary");
  const partitionAuditsFull: PartitionRawEdgeRealityAuditRow[] = [];
  for (const r of partitionRows) {
    const m = marketById.get(r.memberMarketIds[0] ?? "");
    if (!m) continue;
    partitionAuditsFull.push(buildPartitionRealityRow(r, m, passFloor));
  }

  if (partitionAuditsFull.length === 0) {
    return {
      rawEdgeRealityAudit: [],
      rawEdgeByFamilyType,
      rawEdgeVsObservedMispricingByFamilyType,
      bestObservedMispricingByFamilyType,
      avgObservedMispricingByFamilyType,
      rawEdgeCompressionRatioByFamilyType,
      strongestPartitionRealityCheck: null,
      partitionRealityCheckSamples: [],
      rawEdgeRealityAuditVerdict: "insufficient_sample",
      rawEdgeRealitySummaryLine:
        "raw_edge_reality: insufficient_sample (no_resolvable_partition_member_markets) | partition_audit_rows=0",
    };
  }

  const maxNetObs = Math.max(...partitionAuditsFull.map(x => x.netUnderObservedCombinedProxy));
  const meanRatio =
    meanOrNull(
      partitionAuditsFull
        .map(x => x.compressionRatioCombinedOverRaw)
        .filter((x): x is number => x !== null)
        .map(r6),
    ) ?? 1;
  const fracGap = partitionAuditsFull.filter(x => x.gapCombinedMinusRaw > 0.003).length / partitionAuditsFull.length;
  const fracGapLoose = partitionAuditsFull.filter(x => x.gapCombinedMinusRaw > 0.001).length / partitionAuditsFull.length;

  const sortedByGap = [...partitionAuditsFull].sort((a, b) => b.gapCombinedMinusRaw - a.gapCombinedMinusRaw);
  const rawEdgeRealityAudit = sortedByGap.slice(0, PARTITION_REALITY_AUDIT_JSON_CAP);

  const closestToPass = [...partitionAuditsFull].sort(
    (a, b) => b.netUnderObservedCombinedProxy - a.netUnderObservedCombinedProxy,
  );
  const partitionRealityCheckSamples = closestToPass.slice(0, 12);

  let strongestPartitionRealityCheck: PartitionRawEdgeRealityAuditRow | null = null;
  if (strongestResidual?.familyType === "exhaustive_partition_binary") {
    const hit = partitionAuditsFull.find(x => x.familyId === strongestResidual.familyId);
    if (hit) strongestPartitionRealityCheck = hit;
  }
  if (!strongestPartitionRealityCheck) {
    const topRawPart = partitionRows.reduce((a, b) => (a.rawEdge >= b.rawEdge ? a : b));
    const m0 = marketById.get(topRawPart.memberMarketIds[0] ?? "");
    if (m0) strongestPartitionRealityCheck = buildPartitionRealityRow(topRawPart, m0, passFloor);
  }

  const paperSlice = partitionAuditsFull.some(x => x.netUnderObservedCombinedProxy >= passFloor - 0.002);

  let rawEdgeRealityAuditVerdict: RawEdgeRealityAuditVerdict;
  if (maxNetObs > passFloor) {
    rawEdgeRealityAuditVerdict = "partition_observed_combined_mispricing_clears_net_floor_for_some_rows";
  } else if (meanRatio >= 1.22 && fracGap >= 0.52) {
    rawEdgeRealityAuditVerdict = "partition_raw_edge_systematically_below_direct_sum_deviation";
  } else if (meanRatio >= 1.08 && fracGapLoose >= 0.38) {
    rawEdgeRealityAuditVerdict = "partition_raw_edge_occasionally_below_direct_sum_deviation";
  } else if (Math.abs(meanRatio - 1) <= 0.05 && fracGap < 0.2) {
    rawEdgeRealityAuditVerdict = "partition_raw_edge_faithful_within_tolerance_vs_direct_sum_deviation";
  } else if (meanRatio >= 1.12) {
    rawEdgeRealityAuditVerdict = "partition_proxy_compressed_but_observed_still_dead_vs_haircut_stack";
  } else {
    rawEdgeRealityAuditVerdict = "partition_observed_combined_mispricing_still_below_net_pass_floor_all_rows";
  }

  const rawEdgeRealitySummaryLine = `raw_edge_reality: verdict=${rawEdgeRealityAuditVerdict} | partition_n=${partitionAuditsFull.length} max_net_under_observed_combined=${r6(maxNetObs)} mean_compression_ratio=${r6(meanRatio)} frac_gap_gt_0.003=${r6(fracGap)} paper_slice_observed_net_within_2bp=${paperSlice} | strongest_partition_row=${strongestPartitionRealityCheck?.familyId ?? "n/a"}`;

  return {
    rawEdgeRealityAudit,
    rawEdgeByFamilyType,
    rawEdgeVsObservedMispricingByFamilyType,
    bestObservedMispricingByFamilyType,
    avgObservedMispricingByFamilyType,
    rawEdgeCompressionRatioByFamilyType,
    strongestPartitionRealityCheck,
    partitionRealityCheckSamples,
    rawEdgeRealityAuditVerdict,
    rawEdgeRealitySummaryLine,
  };
}

function rowDominantFailureComponent(
  r: ConstraintFirstFamilyRow,
  passFloor: number,
): DominantFailureComponent {
  const f = r.frictionCostEstimate;
  const u = r.uncertaintyHaircut;
  const m = r.modelRiskHaircut;
  const hair = f + u + m;
  if (hair <= 0) return "balanced_drag";
  if (r.rawEdge < passFloor && r.rawEdge < hair * 0.35) return "raw_too_small";
  if (f >= u && f >= m) return "friction";
  if (u >= f && u >= m) return "uncertainty";
  return "model_risk";
}

function tallyMode(votes: DominantFailureComponent[]): DominantFailureComponent {
  if (votes.length === 0) return "insufficient_sample";
  const order: DominantFailureComponent[] = [
    "friction",
    "uncertainty",
    "model_risk",
    "raw_too_small",
    "balanced_drag",
  ];
  const counts: Partial<Record<DominantFailureComponent, number>> = {};
  for (const v of votes) counts[v] = (counts[v] ?? 0) + 1;
  let best: DominantFailureComponent = votes[0]!;
  let bestC = -1;
  for (const k of order) {
    const c = counts[k] ?? 0;
    if (c > bestC) {
      bestC = c;
      best = k;
    }
  }
  return best;
}

type FeasibilityLayer = Pick<
  ConstraintFirstEdgeDiscoveryDigest,
  "familyTypeBreakdown" | "bestRawEdgeByFamilyType"
  | "bestNetEdgeByFamilyType"
  | "avgRawEdgeByFamilyType"
  | "avgFrictionCostByFamilyType"
  | "avgUncertaintyHaircutByFamilyType"
  | "avgModelRiskHaircutByFamilyType"
  | "minimumRawEdgeNeededToPassByFamilyType"
  | "dominantFailureComponentByFamilyType"
  | "economicallyImpossibleFamilyTypes"
  | "potentiallyViableFamilyTypes"
  | "feasibilitySummaryLine"
>;

function computeFeasibilityLayer(rows: ConstraintFirstFamilyRow[], passFloor: number): FeasibilityLayer {
  const familyTypeBreakdown: Record<ConstraintFamilyType, number> = {
    exhaustive_partition_binary: 0,
    strong_dominance_binary: 0,
    same_event_subfamily: 0,
  };
  for (const r of rows) familyTypeBreakdown[r.familyType]++;

  const byType: Record<ConstraintFamilyType, ConstraintFirstFamilyRow[]> = {
    exhaustive_partition_binary: [],
    strong_dominance_binary: [],
    same_event_subfamily: [],
  };
  for (const r of rows) byType[r.familyType].push(r);

  const nullRec = (): Record<ConstraintFamilyType, number | null> => ({
    exhaustive_partition_binary: null,
    strong_dominance_binary: null,
    same_event_subfamily: null,
  });

  const bestRawEdgeByFamilyType = nullRec();
  const bestNetEdgeByFamilyType = nullRec();
  const avgRawEdgeByFamilyType = nullRec();
  const avgFrictionCostByFamilyType = nullRec();
  const avgUncertaintyHaircutByFamilyType = nullRec();
  const avgModelRiskHaircutByFamilyType = nullRec();
  const minimumRawEdgeNeededToPassByFamilyType = nullRec();

  const dominantFailureComponentByFamilyType: Record<ConstraintFamilyType, DominantFailureComponent> = {
    exhaustive_partition_binary: "insufficient_sample",
    strong_dominance_binary: "insufficient_sample",
    same_event_subfamily: "insufficient_sample",
  };

  for (const t of ALL_FAMILY_TYPES) {
    const arr = byType[t];
    if (arr.length === 0) continue;
    bestRawEdgeByFamilyType[t] = r6(Math.max(...arr.map(x => x.rawEdge)));
    bestNetEdgeByFamilyType[t] = r6(Math.max(...arr.map(x => x.netEdgeAfterHaircut)));
    avgRawEdgeByFamilyType[t] = meanOrNull(arr.map(x => x.rawEdge));
    avgFrictionCostByFamilyType[t] = meanOrNull(arr.map(x => x.frictionCostEstimate));
    avgUncertaintyHaircutByFamilyType[t] = meanOrNull(arr.map(x => x.uncertaintyHaircut));
    avgModelRiskHaircutByFamilyType[t] = meanOrNull(arr.map(x => x.modelRiskHaircut));
    const requiredRaw = arr.map(x => r6(passFloor + x.frictionCostEstimate + x.uncertaintyHaircut + x.modelRiskHaircut));
    minimumRawEdgeNeededToPassByFamilyType[t] = meanOrNull(requiredRaw);
    dominantFailureComponentByFamilyType[t] = tallyMode(arr.map(r => rowDominantFailureComponent(r, passFloor)));
  }

  const economicallyImpossibleFamilyTypes: ConstraintFamilyType[] = [];
  const potentiallyViableFamilyTypes: ConstraintFamilyType[] = [];

  for (const t of ALL_FAMILY_TYPES) {
    const n = familyTypeBreakdown[t];
    if (n === 0) continue;
    const bestRaw = bestRawEdgeByFamilyType[t];
    const bestNet = bestNetEdgeByFamilyType[t];
    const minNeed = minimumRawEdgeNeededToPassByFamilyType[t];
    const impossible =
      bestNet !== null &&
      bestNet <= passFloor &&
      bestRaw !== null &&
      minNeed !== null &&
      bestRaw < minNeed;
    if (impossible) economicallyImpossibleFamilyTypes.push(t);
    else if (bestNet !== null && bestNet > passFloor) potentiallyViableFamilyTypes.push(t);
    else if (bestRaw !== null && minNeed !== null && bestRaw >= minNeed) potentiallyViableFamilyTypes.push(t);
  }

  const impS = economicallyImpossibleFamilyTypes.join(",") || "none";
  const viaS = potentiallyViableFamilyTypes.join(",") || "none";
  const feasibilitySummaryLine = `feasibility: impossible_types=[${impS}] potentially_viable=[${viaS}] | attribution uses row-level friction/uncertainty/model vs raw (see dominantFailureComponentByFamilyType)`;

  return {
    familyTypeBreakdown,
    bestRawEdgeByFamilyType,
    bestNetEdgeByFamilyType,
    avgRawEdgeByFamilyType,
    avgFrictionCostByFamilyType,
    avgUncertaintyHaircutByFamilyType,
    avgModelRiskHaircutByFamilyType,
    minimumRawEdgeNeededToPassByFamilyType,
    dominantFailureComponentByFamilyType,
    economicallyImpossibleFamilyTypes,
    potentiallyViableFamilyTypes,
    feasibilitySummaryLine,
  };
}

function buildDiagnosticSummaryLine(args: {
  loaded: number;
  eligible: number;
  built: number;
  scored: number;
  rejSem: number;
  rejHair: number;
  rejConc: number;
  rejRec: number;
  ineligible: IneligibleMarketBreakdown;
  strongestNet: number | null;
  strongestReason: string | null;
}): string {
  const b = args.ineligible;
  if (args.loaded === 0) {
    return "funnel_collapse: empty_market_universe | getAllMarkets()=0 (bootstrap/refresh não publicou mercados neste processo)";
  }
  if (args.eligible === 0) {
    const parts = [
      `closed=${b.closed}`,
      `inactive=${b.inactive}`,
      `lowLiq(<350)=${b.lowLiquidity}`,
      `outcomes<2=${b.outcomesLt2}`,
      `price≠outcomes=${b.priceOutcomeMismatch}`,
    ];
    return `funnel_collapse: zero_eligible_after_gate | loaded=${args.loaded} | ineligible{${parts.join(" ")}}`;
  }
  if (args.built === 0) {
    return `funnel_collapse: zero_candidate_families_built | eligible=${args.eligible} (provável: sem mercados binários outcomes=2 na bolsa elegível, ou stems<12 para clusters)`;
  }
  if (args.scored === 0) {
    return "funnel_collapse: zero_families_scored (inconsistência: built>0 mas rows=0 — não esperado)";
  }
  const parts = `rejected_semantic=${args.rejSem} rejected_haircut_net=${args.rejHair} rejected_conc=${args.rejConc} edge_pass_but_recurrence_lt2=${args.rejRec}`;
  const tail =
    args.strongestReason !== null
      ? `strongest_rejected_net=${args.strongestNet ?? "n/a"} reason=${args.strongestReason}`
      : "strongest_rejected=n/a";
  return `funnel: built=${args.built} scored=${args.scored} | ${parts} | ${tail}`;
}

export function buildConstraintFirstEdgeDiscoveryDigest(): ConstraintFirstEdgeDiscoveryDigest {
  const allMarkets = getAllMarkets();
  const ineligible: IneligibleMarketBreakdown = {
    closed: 0,
    inactive: 0,
    lowLiquidity: 0,
    outcomesLt2: 0,
    priceOutcomeMismatch: 0,
  };
  for (const m of allMarkets) {
    const k = ineligibleBucket(m);
    if (k) ineligible[k]++;
  }

  const markets = allMarkets.filter(eligible);
  const stemMap = new Map<string, NormalizedMarket[]>();

  for (const m of markets) {
    if (m.outcomes.length !== 2) continue;
    const stem = normalizeEventStem(m.question);
    if (stem.length < 12) continue;
    const arr = stemMap.get(stem) ?? [];
    arr.push(m);
    stemMap.set(stem, arr);
  }

  const candidateFamilies: Array<{
    id: string;
    type: ConstraintFamilyType;
    members: NormalizedMarket[];
    raw: number;
    sem: number;
    note: string;
  }> = [];

  for (const m of markets) {
    if (m.outcomes.length !== 2) continue;
    const raw = partitionRawEdge(m);
    const sem = partitionSemantic(m);
    candidateFamilies.push({
      id: `partition:${m.id}`,
      type: "exhaustive_partition_binary",
      members: [m],
      raw,
      sem,
      note: `Binário ME: probSum=${r6(m.probSum)} spread=${r6(m.spread)}`,
    });
    if (dominanceEligible(m)) {
      candidateFamilies.push({
        id: `dominance:${m.id}`,
        type: "strong_dominance_binary",
        members: [m],
        raw: dominanceRawEdge(m),
        sem: dominanceSemantic(m),
        note: `Dominância: maxP=${r6(maxPrice(m))} spread=${r6(m.spread)}`,
      });
    }
  }

  for (const [, group] of Array.from(stemMap.entries())) {
    if (group.length < 2 || group.length > 6) continue;
    const allBinary = group.every(x => x.outcomes.length === 2);
    if (!allBinary) continue;
    const stemKey = group[0]!.question.slice(0, 48);
    let h = 0;
    for (let i = 0; i < stemKey.length; i++) h = ((h << 5) - h + stemKey.charCodeAt(i)!) | 0;
    const id = `same_event:${Math.abs(h)}`;
    const raw = r6(
      Math.min(
        0.11,
        group.reduce((s, m) => s + partitionRawEdge(m), 0) / group.length + 0.01 * (group.length - 1),
      ),
    );
    const sem = r6(0.55 + Math.min(0.25, group.length * 0.04));
    candidateFamilies.push({
      id,
      type: "same_event_subfamily",
      members: group,
      raw,
      sem,
      note: `Subfamília mesmo evento: n=${group.length} stem≈${stemKey.slice(0, 40)}…`,
    });
  }

  const familyIds = candidateFamilies.map(f => f.id);
  const recurrenceMap = bumpRecurrence(familyIds);

  const rows: ConstraintFirstFamilyRow[] = candidateFamilies.map(f =>
    buildRow(f.id, f.type, f.sem, f.raw, f.members, f.note, recurrenceMap),
  );

  rows.sort((a, b) => b.netEdgeAfterHaircut - a.netEdgeAfterHaircut);

  const passFloor = envNum("CONSTRAINT_FIRST_NET_PASS", 0.0045);
  const passing = rows.filter(r => r.killReason === null && r.netEdgeAfterHaircut > passFloor);

  let totalFamiliesRejectedSemantic = 0;
  let totalFamiliesRejectedConcentration = 0;
  let totalFamiliesRejectedHaircut = 0;
  for (const r of rows) {
    const b = primaryRejectionBucket(r);
    if (b === "semantic") totalFamiliesRejectedSemantic++;
    else if (b === "haircut") totalFamiliesRejectedHaircut++;
    else if (b === "concentration") totalFamiliesRejectedConcentration++;
  }

  const totalFamiliesRejectedNoRecurrence = rows.filter(
    r => r.killReason === null && r.netEdgeAfterHaircut > passFloor && r.recurrenceObservationCount < 2,
  ).length;

  const rejectedForStrongest = rows.filter(r => !(r.killReason === null && r.netEdgeAfterHaircut > passFloor));
  let strongestRejectedFamily: ConstraintFirstStrongestRejected | null = null;
  let strongestRejectedReason: string | null = null;
  if (rejectedForStrongest.length > 0) {
    const sr = rejectedForStrongest.reduce((a, b) => (a.netEdgeAfterHaircut >= b.netEdgeAfterHaircut ? a : b));
    strongestRejectedFamily = {
      familyId: sr.familyId,
      familyType: sr.familyType,
      semanticConfidence: sr.semanticConfidence,
      rawEdge: sr.rawEdge,
      netEdgeAfterHaircut: sr.netEdgeAfterHaircut,
      killReason: sr.killReason,
    };
    strongestRejectedReason =
      sr.killReason ??
      (sr.netEdgeAfterHaircut <= passFloor ? `below_net_pass_floor(<=${passFloor})` : "unknown_rejection_path");
  }

  const preRejectionFamilySamples: ConstraintFirstPreRejectionSample[] = rows.slice(0, 5).map(r => ({
    familyId: r.familyId,
    familyType: r.familyType,
    semanticConfidence: r.semanticConfidence,
    rawEdge: r.rawEdge,
    frictionCostEstimate: r.frictionCostEstimate,
    uncertaintyHaircut: r.uncertaintyHaircut,
    modelRiskHaircut: r.modelRiskHaircut,
    netEdgeAfterHaircut: r.netEdgeAfterHaircut,
    rejectionReason:
      r.killReason ??
      (r.netEdgeAfterHaircut > passFloor ? "passed_kill_gates_and_edge_floor" : `below_net_edge_pass_floor(<=${passFloor})`),
  }));

  const diagnosticSummaryLine = buildDiagnosticSummaryLine({
    loaded: allMarkets.length,
    eligible: markets.length,
    built: candidateFamilies.length,
    scored: rows.length,
    rejSem: totalFamiliesRejectedSemantic,
    rejHair: totalFamiliesRejectedHaircut,
    rejConc: totalFamiliesRejectedConcentration,
    rejRec: totalFamiliesRejectedNoRecurrence,
    ineligible,
    strongestNet: strongestRejectedFamily?.netEdgeAfterHaircut ?? null,
    strongestReason: strongestRejectedReason,
  });

  const bestNet = rows.length ? rows[0]!.netEdgeAfterHaircut : 0;
  const viableTh = envNum("CONSTRAINT_FIRST_VIABLE", 0.0065);
  const promisingTh = envNum("CONSTRAINT_FIRST_PROMISING", 0.011);

  let verdict: ConstraintFirstVerdict = "no_viable_family_found";
  if (rows.length === 0) {
    verdict = "no_viable_family_found";
  } else if (passing.length === 0) {
    verdict = bestNet > 0 ? "weak_candidates_only" : "no_viable_family_found";
  } else {
    const top = passing[0]!;
    if (top.netEdgeAfterHaircut >= promisingTh && top.recurrenceObservationCount >= 2 && top.concentrationRisk < 0.48) {
      verdict = "promising_candidate_present";
    } else if (top.netEdgeAfterHaircut >= viableTh) {
      verdict = "viable_candidate_present";
    } else {
      verdict = "weak_candidates_only";
    }
  }

  const thresholdsUsed: Record<string, number> = {
    CONSTRAINT_FIRST_NET_PASS: passFloor,
    CONSTRAINT_FIRST_VIABLE: viableTh,
    CONSTRAINT_FIRST_PROMISING: promisingTh,
    CONSTRAINT_FIRST_SEMANTIC_KILL: envNum("CONSTRAINT_FIRST_SEMANTIC_KILL", 0.58),
    CONSTRAINT_FIRST_CONC_KILL: envNum("CONSTRAINT_FIRST_CONC_KILL", 0.82),
    MIN_LIQUIDITY_ELIGIBLE: 350,
  };

  const topFamilies = rows.slice(0, 8);
  const summary = `constraint_first: ${verdict} | scanned=${rows.length} pass=${passing.length} | bestNet=${rows[0] ? r6(rows[0].netEdgeAfterHaircut) : 0}`;

  const feasibility = computeFeasibilityLayer(rows, passFloor);
  const frictionAudit = computeFrictionAuditLayer(rows, allMarkets, passFloor);
  const spreadCalibration = computeSpreadCalibrationLayer(rows, allMarkets, passFloor);
  const residualKillAudit = computeResidualKillAuditLayer(rows, allMarkets, passFloor, spreadCalibration);
  const concentrationAudit = computeConcentrationAuditLayer(rows, passFloor, residualKillAudit, spreadCalibration);
  const classAwarePolicy = computeClassAwareConcentrationPolicyLayer(
    rows,
    passFloor,
    viableTh,
    promisingTh,
    verdict,
    passing.length,
    concentrationAudit,
  );
  const postConcentrationResidual = computePostConcentrationResidualFeasibilityLayer(
    rows,
    passFloor,
    concentrationAudit,
    classAwarePolicy.familiesPassingEdgeFloorUnderClassAwareConcentration,
  );
  const rawEdgeReality = computeRawEdgeRealityAuditLayer(
    rows,
    allMarkets,
    passFloor,
    postConcentrationResidual.strongestNearPassResidualBreakdown,
  );

  return {
    probeVersion: "constraint-first-edge-v1",
    readDisclaimer:
      "Constraint-first: métricas observacionais com haircuts conservadores. netEdgeAfterHaircut>0 não implica lucro realizável nem fill garantido. Paper só após validação independente.",
    constraintFirstVerdict: verdict,
    familiesScanned: rows.length,
    familiesPassingEdgeFloor: passing.length,
    topFamilies,
    thresholdsUsed,
    constraintFirstSummaryLine: summary,
    families: rows,
    computedAt: new Date().toISOString(),
    totalMarketsLoaded: allMarkets.length,
    totalMarketsEligibleAfterLiquidity: markets.length,
    totalCandidateFamiliesBuilt: candidateFamilies.length,
    totalFamiliesRejectedSemantic,
    totalFamiliesRejectedConcentration,
    totalFamiliesRejectedHaircut,
    totalFamiliesRejectedNoRecurrence,
    totalFamiliesScored: rows.length,
    strongestRejectedFamily,
    strongestRejectedReason,
    diagnosticSummaryLine,
    preRejectionFamilySamples,
    ineligibleMarketBreakdown: ineligible,
    ...feasibility,
    ...frictionAudit,
    ...spreadCalibration,
    ...residualKillAudit,
    ...concentrationAudit,
    ...classAwarePolicy,
    ...postConcentrationResidual,
    ...rawEdgeReality,
  };
}
