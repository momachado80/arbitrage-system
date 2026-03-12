/**
 * Adaptive Calibration Engine v1 — recommendation and challenger generation only.
 * Does NOT change baseline profiles, execution, or production behavior.
 * Reads audit data, produces structured recommendations with confidence.
 */

import type {
  ClosedTradeAuditResult,
  ProfileComparisonEntry,
} from "./shadowClosedTradeAudit";
import type { ShadowProfileConfig } from "./shadowSimulationProfiles";
import { getProfileById } from "./shadowSimulationProfiles";

// ─── Experimentation thresholds (for challenger spec generation) ───────────────
// Lower bars: enough to justify a shadow experiment, not promotion.
const MIN_CLOSED_FOR_EXPERIMENT = 20;
const MIN_CLOSED_FOR_EDGE_EXPERIMENT = 20;
const MIN_CLOSED_FOR_PAIR_PENALTY_EXPERIMENT = 20;
const MIN_BASE_CLOSED_FOR_EXPERIMENT = 10;
const MIN_PAIR_SAMPLE_EXPERIMENT = 3;
const PAIR_AVG_PNL_FOR_PENALTY_EXPERIMENT = -0.2;

// ─── Promotion / stronger recommendation thresholds (unchanged, strict) ───────
const MIN_CLOSED_FOR_HOLD_REC = 30;
const MIN_CLOSED_FOR_PAIR_REC = 50;
const MIN_CLOSED_FOR_FILL_REC = 40;
const MIN_CLOSED_FOR_EDGE_REC = 40;
const MIN_CLOSED_FOR_PAIR_PENALTY_REC = 35;
const MIN_PAIR_SAMPLE = 5;
const MIN_TOTAL_CLOSED_FOR_ADAPTIVE = 20;
const PAIR_LOSS_SHARE_FOR_REC = 0.4;
const FILL_BUCKET_IMPROVEMENT_THRESHOLD = 0.003;
const EDGE_DECILE_IMPROVEMENT_THRESHOLD = 0.002;
const PAIR_AVG_PNL_FOR_PENALTY = -0.3;

// ─── Edgegate v2: refinement when v1 showed directional improvement but stayed negative ───
const MIN_CLOSED_FOR_EDGEGATE_V2 = 10;
const EDGEGATE_V2_THRESHOLD_UPLIFT = 0.003;

// ─── Types ──────────────────────────────────────────────────────────────────

export type RecommendationType =
  | "maxHoldingTimeMs"
  | "pairKeyExclusion"
  | "minFillRatioThreshold"
  | "minCapturableEdgeThreshold"
  | "minCapturableEdgeThresholdV2"
  | "entryScorePenaltyByPair"
  | "maxCapitalPerTrade";

export interface CalibrationSignal {
  profileId: string;
  signalType: string;
  value: number;
  threshold?: number;
  evidence: string;
}

export interface CalibrationRecommendation {
  profileId: string;
  recommendationType: RecommendationType;
  confidence: number;
  currentValue: unknown;
  recommendedValue: unknown;
  reason: string;
  supportingEvidenceSummary: string;
  /** When true: worth experimenting, but not yet promotion-grade evidence */
  forExperimentationOnly?: boolean;
  /** Why this spec was generated (even when promotionReadiness is false) */
  experimentationRationale?: string;
  /** Promotion-grade: strong enough evidence for promotion consideration */
  promotionEligible?: boolean;
}

export interface AdaptiveProfileSpec {
  profileId: string;
  baseProfileId: string;
  label: string;
  status: "proposed" | "spec_only";
  changes: Record<string, unknown>;
  fullConfig: ShadowProfileConfig;
  /** Human-readable hypothesis this challenger tests */
  hypothesis?: string;
  /** Entry threshold (e.g. minCapturableEdgeToTrade) when applicable */
  entryThreshold?: number;
  /** Pair penalty config when applicable */
  pairPenaltyConfig?: Record<string, number>;
  /** maxCapitalPerTrade override when applicable */
  maxCapitalPerTradeOverride?: number;
  /** How entry gating works */
  expectedMechanism?: string;
  /** Why this spec was generated even when promotionReadiness is false (experimentation) */
  whyGenerated?: string;
  /** True when generated from experimentation-only recommendation */
  forExperimentationOnly?: boolean;
}

export interface PromotionReadiness {
  profileId: string;
  ready: boolean;
  reason: string;
  /** 0–1 confidence that evidence supports promotion consideration */
  promotionConfidence?: number;
  /** Detailed reason (alias for reason, for API clarity) */
  promotionReason?: string;
  summaryMetrics?: {
    totalClosed?: number;
    avgRealizedPnL?: number;
    winRate?: number;
    avgHoldingTimeMs?: number;
  };
}

export interface AdaptiveCalibrationResult {
  status: "ok" | "insufficient_data";
  generatedAt: string;
  enoughData: boolean;
  recommendations: CalibrationRecommendation[];
  adaptiveChallengers: AdaptiveProfileSpec[];
  promotionReadiness: PromotionReadiness[];
  /** True when enough evidence exists to justify experimentation (challenger specs) */
  experimentationThresholdsMet?: boolean;
  /** True when at least one baseline challenger meets promotion criteria */
  promotionThresholdsMet?: boolean;
  /** Human-readable explanation when challengers exist but promotionReadiness is false */
  whyChallengersGenerated?: string;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function confidenceFromSampleSize(n: number, min: number): number {
  if (n < min) return 0;
  const excess = n - min;
  return Math.min(0.95, 0.3 + excess * 0.02);
}

function confidenceFromEffectSize(delta: number, threshold: number): number {
  if (delta <= 0) return 0;
  const ratio = delta / Math.max(threshold, 0.001);
  return Math.min(0.9, 0.2 + ratio * 0.2);
}

// ─── Recommendation logic ───────────────────────────────────────────────────

/**
 * 1. Holding time recommendation
 * Reduce maxHoldingTimeMs when: enough sample, max_holding_time exits drive losses,
 * shorter-hold comparison profiles perform materially better.
 */
function recommendHoldingTime(
  audit: ClosedTradeAuditResult,
  profileComparison: ProfileComparisonEntry[],
  baseProfileIds: string[]
): CalibrationRecommendation[] {
  const recs: CalibrationRecommendation[] = [];
  const totalClosed = audit.dataSufficiency.totalClosed;
  if (totalClosed < MIN_CLOSED_FOR_HOLD_REC) return recs;

  const baseline300k = profileComparison.filter(
    (p) => p.maxHoldingTimeMs === 300_000 && baseProfileIds.includes(p.profileId)
  );
  const shorterProfiles = profileComparison.filter(
    (p) => p.maxHoldingTimeMs < 300_000 && p.totalClosed >= 10
  );

  for (const base of baseline300k) {
    if (base.totalClosed < MIN_CLOSED_FOR_HOLD_REC) continue;

    const cfg = getProfileById(base.profileId);
    if (!cfg) continue;

    const exitDiag = audit.exitReasonDiagnostics["max_holding_time"];
    const exitShare = exitDiag
      ? exitDiag.tradeCount / totalClosed
      : 0;
    const exitDrivesLoss =
      exitDiag && exitDiag.avgRealizedPnL < -0.001 && exitShare >= 0.2;

    let bestShorter: ProfileComparisonEntry | null = null;
    let improvement = 0;
    for (const s of shorterProfiles) {
      if (s.avgRealizedPnL > base.avgRealizedPnL + FILL_BUCKET_IMPROVEMENT_THRESHOLD) {
        const delta = s.avgRealizedPnL - base.avgRealizedPnL;
        if (delta > improvement) {
          improvement = delta;
          bestShorter = s;
        }
      }
    }

    if (!exitDrivesLoss && !bestShorter) continue;

    const targetHoldMs = bestShorter?.maxHoldingTimeMs ?? 60_000;
    const confSample = confidenceFromSampleSize(base.totalClosed, MIN_CLOSED_FOR_HOLD_REC);
    const confEffect = confidenceFromEffectSize(
      improvement || (exitDrivesLoss ? 0.01 : 0),
      FILL_BUCKET_IMPROVEMENT_THRESHOLD
    );
    const confidence = Math.min(0.85, (confSample + confEffect) / 2);

    recs.push({
      profileId: base.profileId,
      recommendationType: "maxHoldingTimeMs",
      confidence,
      currentValue: cfg.maxHoldingTimeMs,
      recommendedValue: targetHoldMs,
      reason: bestShorter
        ? `Shorter-hold profile (${bestShorter.profileId}, ${bestShorter.maxHoldingTimeMs / 1000}s) has avgRealizedPnL ${bestShorter.avgRealizedPnL.toFixed(4)} vs baseline ${base.avgRealizedPnL.toFixed(4)}.`
        : `max_holding_time exits (${(exitShare * 100).toFixed(0)}% of trades) drive losses; avg PnL ${exitDiag?.avgRealizedPnL.toFixed(4) ?? "N/A"}.`,
      supportingEvidenceSummary: `totalClosed=${base.totalClosed}, exitShare=${(exitShare * 100).toFixed(0)}%, shorterImprovement=${improvement.toFixed(4)}`,
    });
  }
  return recs;
}

/**
 * 2. Pair penalty / exclusion recommendation
 * When minority of pairKeys explains disproportionate losses; each flagged pair has minimum sample.
 */
function recommendPairExclusion(
  audit: ClosedTradeAuditResult,
  profileComparison: ProfileComparisonEntry[],
  baseProfileIds: string[]
): CalibrationRecommendation[] {
  const recs: CalibrationRecommendation[] = [];
  const totalClosed = audit.dataSufficiency.totalClosed;
  if (totalClosed < MIN_CLOSED_FOR_PAIR_REC) return recs;
  if (!audit.causalReadout.pairSelectionLikelyPrimaryDriver) return recs;

  const worstPairs = audit.worstPairs.filter((p) => p.tradeCount >= MIN_PAIR_SAMPLE);
  if (worstPairs.length === 0) return recs;

  const totalLoss = profileComparison
    .filter((p) => baseProfileIds.includes(p.profileId))
    .reduce((s, p) => s + p.totalClosed * Math.min(0, p.avgRealizedPnL), 0);
  const pairLossShare = worstPairs
    .slice(0, 5)
    .reduce((s, p) => s + p.tradeCount * Math.max(0, -p.avgRealizedPnL), 0);

  if (Math.abs(totalLoss) < 1e-6 || pairLossShare / Math.abs(totalLoss) < PAIR_LOSS_SHARE_FOR_REC) {
    return recs;
  }

  for (const base of profileComparison.filter((p) => baseProfileIds.includes(p.profileId))) {
    if (base.totalClosed < MIN_CLOSED_FOR_PAIR_REC) continue;

    const toExclude = worstPairs.slice(0, 3).map((p) => p.pairKey);
    const confSample = confidenceFromSampleSize(base.totalClosed, MIN_CLOSED_FOR_PAIR_REC);
    const confidence = Math.min(0.8, confSample * 1.2);

    recs.push({
      profileId: base.profileId,
      recommendationType: "pairKeyExclusion",
      confidence,
      currentValue: null,
      recommendedValue: toExclude,
      reason: `Top ${toExclude.length} pairKeys explain ${(pairLossShare / Math.abs(totalLoss) * 100).toFixed(0)}% of losses. Each has ≥${MIN_PAIR_SAMPLE} trades.`,
      supportingEvidenceSummary: `worstPairs=${worstPairs.slice(0, 3).map((p) => p.pairKey).join(",")}, pairLossShare=${(pairLossShare / Math.abs(totalLoss) * 100).toFixed(0)}%`,
    });
  }
  return recs;
}

/**
 * 3. Fill ratio threshold recommendation
 * When poor fill-ratio buckets materially underperform better buckets; enough sample.
 */
function recommendFillRatioThreshold(
  audit: ClosedTradeAuditResult,
  profileComparison: ProfileComparisonEntry[],
  baseProfileIds: string[]
): CalibrationRecommendation[] {
  const recs: CalibrationRecommendation[] = [];
  const totalClosed = audit.dataSufficiency.totalClosed;
  if (totalClosed < MIN_CLOSED_FOR_FILL_REC) return recs;
  if (!audit.causalReadout.fillQualityLikelyPrimaryDriver) return recs;

  const buckets = audit.byFillRatioBucket;
  const low = buckets["0-0.1"];
  const high = buckets["0.75-1.0"];
  if (!low || !high || low.tradeCount < 5 || high.tradeCount < 5) return recs;

  const improvement = high.avgRealizedPnL - low.avgRealizedPnL;
  if (improvement < FILL_BUCKET_IMPROVEMENT_THRESHOLD) return recs;

  for (const base of profileComparison.filter((p) => baseProfileIds.includes(p.profileId))) {
    if (base.totalClosed < MIN_CLOSED_FOR_FILL_REC) continue;

    const recommendedMin = 0.25;
    const confSample = confidenceFromSampleSize(base.totalClosed, MIN_CLOSED_FOR_FILL_REC);
    const confEffect = confidenceFromEffectSize(improvement, FILL_BUCKET_IMPROVEMENT_THRESHOLD);
    const confidence = Math.min(0.8, (confSample + confEffect) / 2);

    recs.push({
      profileId: base.profileId,
      recommendationType: "minFillRatioThreshold",
      confidence,
      currentValue: null,
      recommendedValue: recommendedMin,
      reason: `Low fill-ratio bucket (0-0.1) avgPnL ${low.avgRealizedPnL.toFixed(4)} vs high (0.75-1.0) ${high.avgRealizedPnL.toFixed(4)}. Filtering low-fill trades may help.`,
      supportingEvidenceSummary: `lowBucketAvg=${low.avgRealizedPnL.toFixed(4)}, highBucketAvg=${high.avgRealizedPnL.toFixed(4)}, improvement=${improvement.toFixed(4)}`,
    });
  }
  return recs;
}

/**
 * 4. Min capturable edge threshold recommendation (entry calibration)
 * When edge monotonicity is unhealthy: higher capturable-edge deciles less bad than lower.
 * Recommend stricter minCapturableEdgeToTrade to filter marginal entries.
 *
 * Two paths:
 * - Promotion-grade: totalClosed>=40, decileKeys>=3, improvement>=0.002, base>=40
 * - Experimentation: totalClosed>=20, enoughDataForCausalReadout, !edgeMonotonicityHealthy, decileKeys>=2, base>=15
 */
function recommendMinCapturableEdgeThreshold(
  audit: ClosedTradeAuditResult,
  profileComparison: ProfileComparisonEntry[],
  baseProfileIds: string[]
): CalibrationRecommendation[] {
  const recs: CalibrationRecommendation[] = [];
  const totalClosed = audit.dataSufficiency.totalClosed;
  const { enoughDataForCausalReadout, edgeMonotonicityHealthy } = audit.causalReadout;
  if (edgeMonotonicityHealthy) return recs;

  const deciles = audit.byCapturableEdgeDecile;
  const decileKeys = Object.keys(deciles).sort();

  // ─── Promotion-grade path (strict) ───
  if (
    totalClosed >= MIN_CLOSED_FOR_EDGE_REC &&
    decileKeys.length >= 3
  ) {
    let improvement = 0;
    for (let i = 1; i < decileKeys.length; i++) {
      const high = deciles[decileKeys[i]];
      const low = deciles[decileKeys[i - 1]];
      if (high && low && high.avgRealizedPnL > low.avgRealizedPnL + EDGE_DECILE_IMPROVEMENT_THRESHOLD) {
        improvement = Math.max(improvement, high.avgRealizedPnL - low.avgRealizedPnL);
      }
    }
    if (improvement >= EDGE_DECILE_IMPROVEMENT_THRESHOLD) {
      for (const base of profileComparison.filter((p) => baseProfileIds.includes(p.profileId))) {
        if (base.totalClosed < MIN_CLOSED_FOR_EDGE_REC) continue;

        const cfg = getProfileById(base.profileId);
        if (!cfg) continue;

        const currentMin = cfg.minNetCapturableEdgeToTrade;
        const recommended = Math.min(0.02, Math.max(currentMin * 1.5, currentMin + 0.005, 0.01));

        const confSample = confidenceFromSampleSize(base.totalClosed, MIN_CLOSED_FOR_EDGE_REC);
        const confEffect = confidenceFromEffectSize(improvement, EDGE_DECILE_IMPROVEMENT_THRESHOLD);
        const confidence = Math.min(0.8, (confSample + confEffect) / 2);

        recs.push({
          profileId: base.profileId,
          recommendationType: "minCapturableEdgeThreshold",
          confidence,
          currentValue: currentMin,
          recommendedValue: recommended,
          reason: `Edge monotonicity unhealthy; higher capturable-edge deciles show better outcomes. Stricter threshold may filter marginal entries.`,
          supportingEvidenceSummary: `totalClosed=${base.totalClosed}, decileImprovement=${improvement.toFixed(4)}, recommendedThreshold=${recommended.toFixed(4)}`,
          promotionEligible: true,
        });
      }
      return recs;
    }
  }

  // ─── Experimentation path (lower bar: worth testing, not promotion) ───
  if (
    totalClosed >= MIN_CLOSED_FOR_EDGE_EXPERIMENT &&
    enoughDataForCausalReadout &&
    decileKeys.length >= 2
  ) {
    for (const base of profileComparison.filter((p) => baseProfileIds.includes(p.profileId))) {
      if (base.totalClosed < MIN_BASE_CLOSED_FOR_EXPERIMENT) continue;

      const cfg = getProfileById(base.profileId);
      if (!cfg) continue;

      const currentMin = cfg.minNetCapturableEdgeToTrade ?? 0;
      const recommended = Math.min(0.02, Math.max(currentMin * 1.5, currentMin + 0.005, 0.01));

      const confSample = confidenceFromSampleSize(base.totalClosed, MIN_CLOSED_FOR_EDGE_EXPERIMENT);
      const confidence = Math.min(0.6, confSample * 0.9);

      recs.push({
        profileId: base.profileId,
        recommendationType: "minCapturableEdgeThreshold",
        confidence,
        currentValue: currentMin,
        recommendedValue: recommended,
        reason: `Edge monotonicity unhealthy; higher capturable-edge buckets may perform differently. Conservative experiment: stricter threshold.`,
        supportingEvidenceSummary: `totalClosed=${base.totalClosed}, decileKeys=${decileKeys.length}, recommendedThreshold=${recommended.toFixed(4)}`,
        forExperimentationOnly: true,
        experimentationRationale: "Enough data for causal readout and unhealthy edge monotonicity. Worth testing a stricter minCapturableEdge challenger before promotion-grade evidence.",
        promotionEligible: false,
      });
    }
  }
  return recs;
}

/**
 * 5. Entry score penalty by pair recommendation
 * When some pairKeys consistently underperform but not enough for full exclusion.
 * Apply penalty to entry qualification rather than excluding.
 *
 * Two paths:
 * - Promotion-grade: totalClosed>=35, worstPairs with count>=5, pnl<-0.3, base>=35
 * - Experimentation: totalClosed>=20, worstPairs with count>=3, pnl<-0.2, base>=15
 */
function recommendEntryScorePenaltyByPair(
  audit: ClosedTradeAuditResult,
  profileComparison: ProfileComparisonEntry[],
  baseProfileIds: string[]
): CalibrationRecommendation[] {
  const recs: CalibrationRecommendation[] = [];
  const totalClosed = audit.dataSufficiency.totalClosed;

  // ─── Promotion-grade path (strict) ───
  const worstPairsPromo = audit.worstPairs.filter(
    (p) => p.tradeCount >= MIN_PAIR_SAMPLE && p.avgRealizedPnL < PAIR_AVG_PNL_FOR_PENALTY
  );
  if (
    totalClosed >= MIN_CLOSED_FOR_PAIR_PENALTY_REC &&
    worstPairsPromo.length > 0
  ) {
    const pairPenalties: Record<string, number> = {};
    for (const p of worstPairsPromo.slice(0, 5)) {
      const penalty = Math.min(0.03, Math.max(0.005, -p.avgRealizedPnL * 0.02));
      pairPenalties[p.pairKey] = penalty;
    }

    for (const base of profileComparison.filter((p) => baseProfileIds.includes(p.profileId))) {
      if (base.totalClosed < MIN_CLOSED_FOR_PAIR_PENALTY_REC) continue;

      const confSample = confidenceFromSampleSize(base.totalClosed, MIN_CLOSED_FOR_PAIR_PENALTY_REC);
      const confidence = Math.min(0.75, confSample);

      recs.push({
        profileId: base.profileId,
        recommendationType: "entryScorePenaltyByPair",
        confidence,
        currentValue: null,
        recommendedValue: pairPenalties,
        reason: `Top ${Object.keys(pairPenalties).length} pairKeys show avgRealizedPnL < ${PAIR_AVG_PNL_FOR_PENALTY}. Penalize entry for these pairs rather than full exclusion.`,
        supportingEvidenceSummary: `pairPenalties=${JSON.stringify(pairPenalties)}, pairCount=${Object.keys(pairPenalties).length}`,
        promotionEligible: true,
      });
    }
    return recs;
  }

  // ─── Experimentation path (lower bar: worth testing) ───
  // Use top 3 toxic pairs only: small, conservative set per post-mortem (edgegate v1).
  const worstPairsExp = audit.worstPairs.filter(
    (p) =>
      p.tradeCount >= MIN_PAIR_SAMPLE_EXPERIMENT &&
      p.avgRealizedPnL < PAIR_AVG_PNL_FOR_PENALTY_EXPERIMENT
  );
  if (
    totalClosed >= MIN_CLOSED_FOR_PAIR_PENALTY_EXPERIMENT &&
    worstPairsExp.length > 0
  ) {
    const pairPenalties: Record<string, number> = {};
    for (const p of worstPairsExp.slice(0, 3)) {
      const penalty = Math.min(0.03, Math.max(0.005, -p.avgRealizedPnL * 0.02));
      pairPenalties[p.pairKey] = penalty;
    }

    for (const base of profileComparison.filter((p) => baseProfileIds.includes(p.profileId))) {
      if (base.totalClosed < MIN_BASE_CLOSED_FOR_EXPERIMENT) continue;

      const confSample = confidenceFromSampleSize(
        base.totalClosed,
        MIN_CLOSED_FOR_PAIR_PENALTY_EXPERIMENT
      );
      const confidence = Math.min(0.6, confSample * 0.9);

      recs.push({
        profileId: base.profileId,
        recommendationType: "entryScorePenaltyByPair",
        confidence,
        currentValue: null,
        recommendedValue: pairPenalties,
        reason: `Some pairKeys show avgRealizedPnL < ${PAIR_AVG_PNL_FOR_PENALTY_EXPERIMENT}. Worth testing penalty challenger before promotion-grade evidence.`,
        supportingEvidenceSummary: `pairPenalties=${JSON.stringify(pairPenalties)}, pairCount=${Object.keys(pairPenalties).length}`,
        forExperimentationOnly: true,
        experimentationRationale: "Consistently underperforming pairKeys with sufficient sample. Conservative experiment: apply entry penalty to test impact.",
        promotionEligible: false,
      });
    }
  }
  return recs;
}

/**
 * 6. Edgegate v2 recommendation (refinement of v1)
 * When edgegate_v1 showed directional improvement vs base (shadow_1000) but remained negative,
 * recommend v2 with stricter threshold = v1 + uplift. Experimentation only, not promotion.
 */
function recommendEdgegateV2(
  audit: ClosedTradeAuditResult,
  profileComparison: ProfileComparisonEntry[]
): CalibrationRecommendation[] {
  const recs: CalibrationRecommendation[] = [];
  if (!audit.causalReadout.enoughDataForCausalReadout) return recs;

  const base = profileComparison.find((p) => p.profileId === "shadow_1000");
  const v1 = profileComparison.find((p) => p.profileId === "shadow_1000_adapt_edgegate_v1");
  if (!base || !v1 || v1.totalClosed < MIN_CLOSED_FOR_EDGEGATE_V2) return recs;

  const directionalImprovement = v1.avgRealizedPnL > base.avgRealizedPnL;
  const stillNegative = v1.avgRealizedPnL < 0;
  if (!directionalImprovement || !stillNegative) return recs;

  const cfg = getProfileById("shadow_1000");
  if (!cfg) return recs;

  const baseMin = cfg.minNetCapturableEdgeToTrade ?? 0.007;
  const v1Threshold = Math.min(
    0.02,
    Math.max(baseMin * 1.5, baseMin + 0.005, 0.01)
  );
  const v2Threshold = Math.min(0.02, v1Threshold + EDGEGATE_V2_THRESHOLD_UPLIFT);

  const confSample = confidenceFromSampleSize(v1.totalClosed, MIN_CLOSED_FOR_EDGEGATE_V2);
  const confidence = Math.min(0.55, confSample * 0.9);

  recs.push({
    profileId: "shadow_1000",
    recommendationType: "minCapturableEdgeThresholdV2",
    confidence,
    currentValue: v1Threshold,
    recommendedValue: v2Threshold,
    reason: `Edgegate v1 showed directional improvement vs shadow_1000 (${v1.avgRealizedPnL.toFixed(4)} > ${base.avgRealizedPnL.toFixed(4)}) but remained negative. V2 tests stricter threshold (${v2Threshold.toFixed(4)}) to continue refinement.`,
    supportingEvidenceSummary: `v1AvgPnL=${v1.avgRealizedPnL.toFixed(4)}, baseAvgPnL=${base.avgRealizedPnL.toFixed(4)}, v1Threshold=${v1Threshold.toFixed(4)}, v2Threshold=${v2Threshold.toFixed(4)}`,
    forExperimentationOnly: true,
    experimentationRationale: "Edgegate v1 showed directional improvement; v2 tests stricter entry gate as next iteration.",
    promotionEligible: false,
  });
  return recs;
}

/**
 * 7. Max capital per trade recommendation (sizing isolation)
 * When edgegate v1 and pairpenalty v1 both showed larger avgFilledCapital and worse PnL than baseline.
 * Test: reduce maxCapitalPerTrade to limit capital concentration per trade.
 * Single-variable experiment: no entry/exit/hold changes.
 */
const CAPITAL_PER_TRADE_REDUCTION_RATIO = 0.5; // 50% of baseline
const BASELINE_SHADOW_1000_MAX_CAPITAL_PER_TRADE = 150;

function recommendMaxCapitalPerTrade(
  audit: ClosedTradeAuditResult,
  profileComparison: ProfileComparisonEntry[],
  baseProfileIds: string[]
): CalibrationRecommendation[] {
  if (audit.dataSufficiency.totalClosed < MIN_TOTAL_CLOSED_FOR_ADAPTIVE) return [];
  const base = profileComparison.find((p) => p.profileId === "shadow_1000");
  if (!base || !baseProfileIds.includes("shadow_1000")) return [];

  const cfg = getProfileById("shadow_1000");
  if (!cfg) return [];

  const recommended = Math.round(
    (cfg.maxCapitalPerTrade ?? BASELINE_SHADOW_1000_MAX_CAPITAL_PER_TRADE) * CAPITAL_PER_TRADE_REDUCTION_RATIO
  );
  // Ensure meaningful reduction and minimum viable cap
  const capped = Math.max(25, Math.min(recommended, cfg.maxCapitalPerTrade - 25));

  return [
    {
      profileId: "shadow_1000",
      recommendationType: "maxCapitalPerTrade",
      confidence: 0.6,
      currentValue: cfg.maxCapitalPerTrade,
      recommendedValue: capped,
      reason: "Edgegate v1 and pairpenalty v1 both showed larger avgFilledCapital and worse PnL. Hypothesis: capital concentration amplifies losses. Test reduced maxCapitalPerTrade as single-variable experiment.",
      supportingEvidenceSummary: `baselineMaxCap=${cfg.maxCapitalPerTrade}, challengerMaxCap=${capped}`,
      forExperimentationOnly: true,
      experimentationRationale: "Post-mortem: both entry-filtering challengers underperformed with larger filled capital. Sizing reduction isolates capital-concentration hypothesis.",
      promotionEligible: false,
    },
  ];
}

// ─── Challenger spec generation ─────────────────────────────────────────────

function buildChallengerFromRecommendation(
  rec: CalibrationRecommendation,
  baseConfig: ShadowProfileConfig
): AdaptiveProfileSpec | null {
  if (rec.recommendationType === "maxHoldingTimeMs") {
    const holdMs = rec.recommendedValue as number;
    const spec: ShadowProfileConfig = {
      ...baseConfig,
      profileId: `${baseConfig.profileId}_adapt_hold_${holdMs / 1000}s`,
      label: `${baseConfig.label} (adapt hold ${holdMs / 1000}s)`,
      maxHoldingTimeMs: holdMs,
      enabled: false,
      baseProfileId: baseConfig.profileId,
      isAdaptive: true,
    };
    return {
      profileId: spec.profileId,
      baseProfileId: baseConfig.profileId,
      label: spec.label,
      status: "proposed",
      changes: { maxHoldingTimeMs: holdMs },
      fullConfig: spec,
    };
  }
  if (rec.recommendationType === "pairKeyExclusion") {
    const excluded = rec.recommendedValue as string[];
    const fullConfig: ShadowProfileConfig = {
      ...baseConfig,
      profileId: `${baseConfig.profileId}_adapt_pairfilter_v1`,
      label: `${baseConfig.label} (adapt pair filter)`,
      enabled: false,
      excludedPairKeys: excluded,
      baseProfileId: baseConfig.profileId,
      isAdaptive: true,
    };
    return {
      profileId: fullConfig.profileId,
      baseProfileId: baseConfig.profileId,
      label: fullConfig.label,
      status: "spec_only",
      changes: { excludedPairKeys: excluded },
      fullConfig,
    };
  }
  if (rec.recommendationType === "minFillRatioThreshold") {
    const minFill = rec.recommendedValue as number;
    const fullConfig: ShadowProfileConfig = {
      ...baseConfig,
      profileId: `${baseConfig.profileId}_adapt_fillfilter_v1`,
      label: `${baseConfig.label} (adapt min fill)`,
      enabled: false,
      minFillRatioToTrade: minFill,
      baseProfileId: baseConfig.profileId,
      isAdaptive: true,
    };
    return {
      profileId: fullConfig.profileId,
      baseProfileId: baseConfig.profileId,
      label: fullConfig.label,
      status: "spec_only",
      changes: { minFillRatioToTrade: minFill },
      fullConfig,
    };
  }
  if (rec.recommendationType === "minCapturableEdgeThreshold") {
    const threshold = rec.recommendedValue as number;
    const fullConfig: ShadowProfileConfig = {
      ...baseConfig,
      profileId: `${baseConfig.profileId}_adapt_edgegate_v1`,
      label: `${baseConfig.label} (adapt edge gate v1)`,
      enabled: false,
      minCapturableEdgeToTrade: threshold,
      baseProfileId: baseConfig.profileId,
      isAdaptive: true,
    };
    return {
      profileId: fullConfig.profileId,
      baseProfileId: baseConfig.profileId,
      label: fullConfig.label,
      status: "proposed",
      changes: { minCapturableEdgeToTrade: threshold },
      fullConfig,
      hypothesis: "Entry miscalibration: entering with marginal capturable edge leads to losses. Higher edge threshold may improve realized outcomes.",
      entryThreshold: threshold,
      expectedMechanism: "Reject entry if capturableEdgeAtEntry < minCapturableEdgeToTrade",
      whyGenerated: rec.experimentationRationale,
      forExperimentationOnly: rec.forExperimentationOnly ?? false,
    };
  }
  if (rec.recommendationType === "minCapturableEdgeThresholdV2") {
    const threshold = rec.recommendedValue as number;
    const fullConfig: ShadowProfileConfig = {
      ...baseConfig,
      profileId: `${baseConfig.profileId}_adapt_edgegate_v2`,
      label: `${baseConfig.label} (adapt edge gate v2)`,
      enabled: false,
      minCapturableEdgeToTrade: threshold,
      baseProfileId: baseConfig.profileId,
      isAdaptive: true,
    };
    return {
      profileId: fullConfig.profileId,
      baseProfileId: baseConfig.profileId,
      label: fullConfig.label,
      status: "spec_only",
      changes: { minCapturableEdgeToTrade: threshold },
      fullConfig,
      hypothesis: "Edgegate v1 showed directional improvement; v2 tests stricter threshold (v1 + 0.003) to continue entry-calibration refinement.",
      entryThreshold: threshold,
      expectedMechanism: "Reject entry if capturableEdgeAtEntry < minCapturableEdgeToTrade",
      whyGenerated: rec.experimentationRationale,
      forExperimentationOnly: true,
    };
  }
  if (rec.recommendationType === "maxCapitalPerTrade") {
    const maxCap = rec.recommendedValue as number;
    const fullConfig: ShadowProfileConfig = {
      ...baseConfig,
      profileId: `${baseConfig.profileId}_adapt_captrade_v1`,
      label: `${baseConfig.label} (adapt cap per trade v1)`,
      enabled: false,
      maxCapitalPerTrade: maxCap,
      baseProfileId: baseConfig.profileId,
      isAdaptive: true,
    };
    return {
      profileId: fullConfig.profileId,
      baseProfileId: baseConfig.profileId,
      label: fullConfig.label,
      status: "spec_only",
      changes: { maxCapitalPerTrade: maxCap },
      fullConfig,
      hypothesis: "Post-mortem: edgegate v1 and pairpenalty v1 both showed larger avgFilledCapital and worse PnL. Hypothesis: capital concentration amplifies losses. Single-variable test: reduce maxCapitalPerTrade only.",
      maxCapitalPerTradeOverride: maxCap,
      expectedMechanism: "Cap requested capital per trade at maxCapitalPerTrade; no changes to entry, exit, hold, or fill logic.",
      whyGenerated: rec.experimentationRationale,
      forExperimentationOnly: true,
    };
  }
  if (rec.recommendationType === "entryScorePenaltyByPair") {
    const penalties = rec.recommendedValue as Record<string, number>;
    const fullConfig: ShadowProfileConfig = {
      ...baseConfig,
      profileId: `${baseConfig.profileId}_adapt_pairpenalty_v1`,
      label: `${baseConfig.label} (adapt pair penalty)`,
      enabled: false,
      entryPairPenalties: penalties,
      baseProfileId: baseConfig.profileId,
      isAdaptive: true,
    };
    return {
      profileId: fullConfig.profileId,
      baseProfileId: baseConfig.profileId,
      label: fullConfig.label,
      status: "spec_only",
      changes: { entryPairPenalties: penalties },
      fullConfig,
      hypothesis: "Post-mortem: edgegate v1 failed; global edge threshold concentrated capital into toxic trades. Pair-aware penalties target worst pairs (poor realized PnL + meaningful count) without changing sizing or exit logic.",
      pairPenaltyConfig: penalties,
      expectedMechanism: "effectiveEdge = capturableEdgeAtEntry - penaltyForPair; reject if effectiveEdge < minNetCapturableEdgeToTrade",
      whyGenerated: rec.experimentationRationale,
      forExperimentationOnly: rec.forExperimentationOnly ?? false,
    };
  }
  return null;
}

function getAdaptiveChallengerSpecs(recommendations: CalibrationRecommendation[]): AdaptiveProfileSpec[] {
  const seen = new Set<string>();
  const specs: AdaptiveProfileSpec[] = [];
  for (const rec of recommendations) {
    const cfg = getProfileById(rec.profileId);
    if (!cfg) continue;
    const spec = buildChallengerFromRecommendation(rec, cfg);
    if (spec && !seen.has(spec.profileId)) {
      seen.add(spec.profileId);
      specs.push(spec);
    }
  }
  return specs;
}

// ─── Promotion readiness ────────────────────────────────────────────────────

function computePromotionReadiness(
  profileIds: string[],
  profileComparison: ProfileComparisonEntry[]
): PromotionReadiness[] {
  const result: PromotionReadiness[] = [];
  const minClosed = 50;
  const minWinRate = 0.45;
  const minAvgPnL = -0.5;

  for (const pid of profileIds) {
    const pc = profileComparison.find((p) => p.profileId === pid);
    const ready =
      !!pc &&
      pc.totalClosed >= minClosed &&
      pc.avgRealizedPnL > minAvgPnL &&
      pc.winRate >= minWinRate;

    const reason = !pc
      ? "No audit data for profile"
      : pc.totalClosed < minClosed
        ? `Insufficient trades (${pc.totalClosed} < ${minClosed})`
        : pc.avgRealizedPnL <= minAvgPnL
          ? `Negative avg PnL (${pc.avgRealizedPnL.toFixed(4)})`
          : pc.winRate < minWinRate
            ? `Win rate below threshold (${(pc.winRate * 100).toFixed(0)}% < ${minWinRate * 100}%)`
            : "Meets minimum evidence for consideration";

    const promotionConfidence = pc
      ? confidenceFromSampleSize(pc.totalClosed, minClosed) *
        (pc.avgRealizedPnL > minAvgPnL ? 1 : 0.3) *
        (pc.winRate >= minWinRate ? 1 : 0.5)
      : 0;

    result.push({
      profileId: pid,
      ready,
      reason,
      promotionConfidence: ready ? Math.min(0.95, promotionConfidence) : promotionConfidence,
      promotionReason: reason,
      summaryMetrics: pc
        ? {
            totalClosed: pc.totalClosed,
            avgRealizedPnL: pc.avgRealizedPnL,
            winRate: pc.winRate,
            avgHoldingTimeMs: pc.avgHoldingTimeMs,
          }
        : undefined,
    });
  }
  return result;
}

// ─── Main entry ─────────────────────────────────────────────────────────────

const BASE_PROFILE_IDS = ["shadow_100", "shadow_1000"];

export function computeAdaptiveCalibration(audit: ClosedTradeAuditResult): AdaptiveCalibrationResult {
  const enoughData = audit.dataSufficiency.totalClosed >= MIN_TOTAL_CLOSED_FOR_ADAPTIVE;
  const profileComparison = audit.profileComparison ?? [];

  const recommendations: CalibrationRecommendation[] = [];
  if (enoughData) {
    recommendations.push(...recommendHoldingTime(audit, profileComparison, BASE_PROFILE_IDS));
    recommendations.push(...recommendPairExclusion(audit, profileComparison, BASE_PROFILE_IDS));
    recommendations.push(...recommendFillRatioThreshold(audit, profileComparison, BASE_PROFILE_IDS));
    recommendations.push(...recommendMinCapturableEdgeThreshold(audit, profileComparison, BASE_PROFILE_IDS));
    recommendations.push(...recommendEdgegateV2(audit, profileComparison));
    recommendations.push(...recommendEntryScorePenaltyByPair(audit, profileComparison, BASE_PROFILE_IDS));
    recommendations.push(...recommendMaxCapitalPerTrade(audit, profileComparison, BASE_PROFILE_IDS));
  }

  const adaptiveChallengers = getAdaptiveChallengerSpecs(recommendations);
  const allProfileIds = [
    ...BASE_PROFILE_IDS,
    ...profileComparison.map((p) => p.profileId),
    ...adaptiveChallengers.map((c) => c.profileId),
  ];
  const uniqueIds = Array.from(new Set(allProfileIds));
  const promotionReadiness = computePromotionReadiness(uniqueIds, profileComparison);

  const promotionThresholdsMet = promotionReadiness.some((p) => p.ready);
  const experimentationThresholdsMet =
    enoughData && (recommendations.length > 0 || adaptiveChallengers.length > 0);
  const whyChallengersGenerated =
    adaptiveChallengers.length > 0 && !promotionThresholdsMet
      ? "Challenger specs generated for experimentation: evidence sufficient to justify shadow experiments but not yet promotion-grade. Use ENABLED_ADAPTIVE_CHALLENGERS to enable manually."
      : undefined;

  return {
    status: enoughData ? "ok" : "insufficient_data",
    generatedAt: new Date().toISOString(),
    enoughData,
    recommendations,
    adaptiveChallengers,
    promotionReadiness,
    experimentationThresholdsMet,
    promotionThresholdsMet,
    whyChallengersGenerated,
  };
}
