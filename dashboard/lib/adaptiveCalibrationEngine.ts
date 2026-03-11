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

// ─── Minimum evidence thresholds (guardrails) ────────────────────────────────
const MIN_CLOSED_FOR_HOLD_REC = 30;
const MIN_CLOSED_FOR_PAIR_REC = 50;
const MIN_CLOSED_FOR_FILL_REC = 40;
const MIN_PAIR_SAMPLE = 5;
const MIN_TOTAL_CLOSED_FOR_ADAPTIVE = 20;
const PAIR_LOSS_SHARE_FOR_REC = 0.4;
const FILL_BUCKET_IMPROVEMENT_THRESHOLD = 0.003;

// ─── Types ──────────────────────────────────────────────────────────────────

export type RecommendationType = "maxHoldingTimeMs" | "pairKeyExclusion" | "minFillRatioThreshold";

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
}

export interface AdaptiveProfileSpec {
  profileId: string;
  baseProfileId: string;
  label: string;
  status: "proposed" | "spec_only";
  changes: Record<string, unknown>;
  fullConfig: ShadowProfileConfig;
}

export interface PromotionReadiness {
  profileId: string;
  ready: boolean;
  reason: string;
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
  for (const pid of profileIds) {
    const pc = profileComparison.find((p) => p.profileId === pid);
    const minClosed = 50;
    const minWinRate = 0.45;
    const ready =
      !!pc &&
      pc.totalClosed >= minClosed &&
      pc.avgRealizedPnL > -0.5 &&
      pc.winRate >= minWinRate;

    result.push({
      profileId: pid,
      ready,
      reason: !pc
        ? "No audit data for profile"
        : pc.totalClosed < minClosed
          ? `Insufficient trades (${pc.totalClosed} < ${minClosed})`
          : pc.avgRealizedPnL <= -0.5
            ? `Negative avg PnL (${pc.avgRealizedPnL.toFixed(4)})`
            : pc.winRate < minWinRate
              ? `Win rate below threshold (${(pc.winRate * 100).toFixed(0)}% < ${minWinRate * 100}%)`
              : "Meets minimum evidence for consideration",
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
  }

  const adaptiveChallengers = getAdaptiveChallengerSpecs(recommendations);
  const allProfileIds = [
    ...BASE_PROFILE_IDS,
    ...profileComparison.map((p) => p.profileId),
    ...adaptiveChallengers.map((c) => c.profileId),
  ];
  const uniqueIds = Array.from(new Set(allProfileIds));
  const promotionReadiness = computePromotionReadiness(uniqueIds, profileComparison);

  return {
    status: enoughData ? "ok" : "insufficient_data",
    generatedAt: new Date().toISOString(),
    enoughData,
    recommendations,
    adaptiveChallengers,
    promotionReadiness,
  };
}
