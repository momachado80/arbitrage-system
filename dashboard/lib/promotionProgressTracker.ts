/**
 * Promotion Progress Tracker — lightweight longitudinal monitor.
 * Measures how close the system is to the "robust" confidence tier
 * based on current promotion readiness, robustness, and ops data.
 * No new alpha, no new rules, no optimization — governance only.
 */

import type { OperationalizationAssessment } from "./momentumOperationalization";
import type { OperationalizationRobustnessAssessment } from "./operationalizationRobustness";
import type {
  PromotionReadinessAssessment,
  PromotionConfidenceTier,
  MaintenanceVerdict,
} from "./operationalizationPromotionReadiness";

function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

type TrendLabel = "improving" | "holding" | "degrading" | "unknown";

interface DimensionProgress {
  current: number;
  target: number;
  pct: number;
  met: boolean;
}

export interface PromotionProgressAssessment {
  progressToRobustPct: number;
  currentTier: PromotionConfidenceTier;
  targetTier: "robust";
  passingEventsProgress: DimensionProgress;
  distinctMarketsProgress: DimensionProgress;
  persistenceTrend: TrendLabel;
  sensitivityTrend: TrendLabel;
  concentrationTrend: TrendLabel;
  overfitTrend: TrendLabel;
  robustnessTrend: TrendLabel;
  blockersToRobust: string[];
  nearestPromotionMilestone: string;
  readDisclaimer: string;
}

const ROBUST_MIN_PASSING = 10;
const ROBUST_MIN_MARKETS = 5;

function dimProgress(current: number, target: number): DimensionProgress {
  const pct = target > 0 ? r4(Math.min(1, current / target)) : current > 0 ? 1 : 0;
  return { current, target, pct, met: current >= target };
}

function maintenanceToTrend(v: MaintenanceVerdict): TrendLabel {
  if (v === "improving") return "improving";
  if (v === "holding") return "holding";
  if (v === "degrading") return "degrading";
  return "unknown";
}

export function buildPromotionProgress(
  ops: OperationalizationAssessment,
  robustness: OperationalizationRobustnessAssessment,
  promo: PromotionReadinessAssessment,
): PromotionProgressAssessment {
  const evProg = dimProgress(promo.passingEventCount, ROBUST_MIN_PASSING);
  const mktProg = dimProgress(promo.distinctPassingMarkets, ROBUST_MIN_MARKETS);

  const persistTrend = maintenanceToTrend(promo.persistenceMaintenanceVerdict);
  const sensTrend = maintenanceToTrend(promo.sensitivityMaintenanceVerdict);
  const concTrend = maintenanceToTrend(promo.concentrationMaintenanceVerdict);

  const overfitTrend: TrendLabel =
    robustness.overfitRiskVerdict === "low" ? "improving" :
    robustness.overfitRiskVerdict === "moderate" ? "holding" : "degrading";

  const robTrend: TrendLabel =
    robustness.robustnessVerdict === "robust" ? "improving" :
    robustness.robustnessVerdict === "stable" ? "holding" :
    robustness.robustnessVerdict === "weak_but_persistent" ? "holding" :
    robustness.robustnessVerdict === "fragile" ? "degrading" : "unknown";

  const quantDims = [evProg.pct, mktProg.pct];
  const qualDims: number[] = [];
  const trendScore = (t: TrendLabel): number =>
    t === "improving" ? 1 : t === "holding" ? 0.6 : t === "degrading" ? 0.2 : 0;
  qualDims.push(trendScore(persistTrend));
  qualDims.push(trendScore(sensTrend));
  qualDims.push(trendScore(concTrend));
  qualDims.push(trendScore(overfitTrend));
  qualDims.push(trendScore(robTrend));

  const quantAvg = quantDims.reduce((a, b) => a + b, 0) / quantDims.length;
  const qualAvg = qualDims.reduce((a, b) => a + b, 0) / qualDims.length;
  const progressPct = r4(Math.min(1, quantAvg * 0.5 + qualAvg * 0.5));

  const blockers: string[] = [];
  if (!evProg.met) blockers.push(`Passing events: ${evProg.current}/${evProg.target}`);
  if (!mktProg.met) blockers.push(`Distinct markets: ${mktProg.current}/${mktProg.target}`);
  if (persistTrend !== "improving") blockers.push(`Persistence: ${persistTrend} (need improving)`);
  if (sensTrend !== "improving") blockers.push(`Sensitivity: ${sensTrend} (need improving)`);
  if (concTrend === "degrading" || concTrend === "unknown") blockers.push(`Concentration: ${concTrend} (need holding+)`);
  if (overfitTrend !== "improving") blockers.push(`Overfit: ${overfitTrend} (need low/improving)`);

  let milestone: string;
  if (promo.currentConfidenceTier === "robust") {
    milestone = "Robust atingido. Monitorar estabilidade contínua.";
  } else if (promo.currentConfidenceTier === "stable") {
    const evRemain = Math.max(0, ROBUST_MIN_PASSING - promo.passingEventCount);
    const mktRemain = Math.max(0, ROBUST_MIN_MARKETS - promo.distinctPassingMarkets);
    const parts: string[] = [];
    if (evRemain > 0) parts.push(`+${evRemain} passing events`);
    if (mktRemain > 0) parts.push(`+${mktRemain} distinct markets`);
    const qualBlockers = blockers.filter(b => !b.startsWith("Passing") && !b.startsWith("Distinct"));
    if (qualBlockers.length > 0) parts.push(`fix: ${qualBlockers.join(", ")}`);
    milestone = parts.length > 0
      ? `stable→robust: ${parts.join("; ")}`
      : "stable→robust: all quantitative met, pending qualitative confirmation";
  } else {
    milestone = `${promo.currentConfidenceTier}→${nextTierUp(promo.currentConfidenceTier)}: ver promotionReadiness.nextPromotionTarget`;
  }

  return {
    progressToRobustPct: progressPct,
    currentTier: promo.currentConfidenceTier,
    targetTier: "robust",
    passingEventsProgress: evProg,
    distinctMarketsProgress: mktProg,
    persistenceTrend: persistTrend,
    sensitivityTrend: sensTrend,
    concentrationTrend: concTrend,
    overfitTrend: overfitTrend,
    robustnessTrend: robTrend,
    blockersToRobust: blockers,
    nearestPromotionMilestone: milestone,
    readDisclaimer:
      "Progress tracker puramente observacional. Percentual não é probabilidade de lucro; é grau de completude dos critérios de governance para tier 'robust'.",
  };
}

function nextTierUp(tier: PromotionConfidenceTier): string {
  switch (tier) {
    case "too_early": return "fragile";
    case "fragile": return "weak_but_persistent";
    case "weak_but_persistent": return "stable";
    case "stable": return "robust";
    case "robust": return "robust (max)";
  }
}

export function buildProgressSummaryLine(a: PromotionProgressAssessment): string {
  const pct = Math.round(a.progressToRobustPct * 100);
  return `progress: ${a.currentTier}→robust ${pct}% | ev=${a.passingEventsProgress.current}/${a.passingEventsProgress.target} mkts=${a.distinctMarketsProgress.current}/${a.distinctMarketsProgress.target} blockers=${a.blockersToRobust.length}`;
}
