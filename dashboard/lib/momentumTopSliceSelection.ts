/**
 * Top-Slice Selection Assessment — camada analítica pura sobre MomentumEvent[].
 * Compara distribuições de atributos entre o top quartile e o universo completo
 * para identificar quais características estão sobre-representadas nos melhores
 * eventos. Propõe regras de filtro candidatas com estimativa conservadora de
 * melhoria, cobertura e risco de concentração. Não altera eventos nem probe state.
 */

import type { MomentumEvent, MomentumEventType } from "./momentumSnipingProbe";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}
function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

const MIN_EVENTS_FOR_SELECTION = () =>
  Math.max(8, Math.floor(envNum("MOMENTUM_SELECTION_MIN_EVENTS", 12)));
const OVERFIT_MAX_RULES = () =>
  Math.max(1, Math.floor(envNum("MOMENTUM_SELECTION_MAX_RULES", 4)));

export type SelectionAssessmentVerdict =
  | "insufficient_sample"
  | "no_clear_filter_advantage"
  | "weak_filter_opportunity"
  | "promising_filter_opportunity"
  | "overfit_risk";

interface BucketComparison {
  bucket: string;
  countAll: number;
  shareAll: number;
  countTop: number;
  shareTop: number;
  enrichment: number;
  avgProxyAll: number | null;
  avgProxyTop: number | null;
}

interface CandidateFilterRule {
  ruleId: string;
  description: string;
  rationale: string;
  eventsPassingInAll: number;
  eventsPassingInTop: number;
  coverageAll: number;
  coverageTop: number;
  avgProxyPassing: number | null;
  avgProxyFailing: number | null;
  capturableRatePassing: number;
  concentrationRiskPassing: number;
  improvementVsAll: number | null;
  verdict: "beneficial" | "marginal" | "harmful" | "inconclusive";
}

export interface TopSliceSelectionAssessment {
  totalEventsAnalyzed: number;
  topSliceEventsAnalyzed: number;
  featureComparisonSummary: string;
  topVsAllByEventType: BucketComparison[];
  topVsAllByLiquidityBucket: BucketComparison[];
  topVsAllByMagnitudeBucket: BucketComparison[];
  topVsAllByCaptureProxyBucket: BucketComparison[];
  topVsAllByMarketCategoryIfAvailable: BucketComparison[];
  commonPatternsInTopSlice: string[];
  weakPatternsInTopSlice: string[];
  candidateFilterRules: CandidateFilterRule[];
  candidateFilterRulesBacktestLikeRead: string;
  filterImprovementEstimate: number | null;
  filterCoverageEstimate: number | null;
  filterConcentrationRisk: number | null;
  selectionAssessmentVerdict: SelectionAssessmentVerdict;
  supportingReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: Record<string, number>;
}

function scoreEvent(e: MomentumEvent): number {
  return e.conservativeCaptureProxy + e.magnitude * 0.15 + (e.liquidityAtDetection >= 10_000 ? 0.001 : 0);
}

function avg(nums: number[]): number | null {
  if (nums.length === 0) return null;
  return r4(nums.reduce((a, b) => a + b, 0) / nums.length);
}

function capRate(events: readonly MomentumEvent[]): number {
  if (events.length === 0) return 0;
  return r4(events.filter(e => e.capturable).length / events.length);
}

function concRisk(events: readonly MomentumEvent[]): number {
  if (events.length === 0) return 0;
  const counts: Record<string, number> = {};
  for (const e of events) counts[e.marketId] = (counts[e.marketId] ?? 0) + 1;
  return r4(Math.max(...Object.values(counts)) / events.length);
}

function liqBucket(liq: number): string {
  if (liq < 1_000) return "<1k";
  if (liq < 10_000) return "1k-10k";
  if (liq < 100_000) return "10k-100k";
  return "≥100k";
}

function magBucket(mag: number): string {
  if (mag < 0.005) return "<0.5%";
  if (mag < 0.01) return "0.5-1%";
  if (mag < 0.02) return "1-2%";
  if (mag < 0.05) return "2-5%";
  return "≥5%";
}

function proxyBucket(p: number): string {
  if (p <= -0.005) return "very_neg";
  if (p < 0) return "slight_neg";
  if (p === 0) return "zero";
  if (p < 0.005) return "slight_pos";
  return "good_pos";
}

function buildBucketComparison(
  all: readonly MomentumEvent[],
  top: readonly MomentumEvent[],
  keyFn: (e: MomentumEvent) => string,
): BucketComparison[] {
  const allBuckets: Record<string, MomentumEvent[]> = {};
  const topBuckets: Record<string, MomentumEvent[]> = {};
  for (const e of all) {
    const k = keyFn(e);
    (allBuckets[k] ??= []).push(e);
  }
  for (const e of top) {
    const k = keyFn(e);
    (topBuckets[k] ??= []).push(e);
  }
  const keys = Array.from(new Set([...Object.keys(allBuckets), ...Object.keys(topBuckets)]));
  return keys.map(bucket => {
    const aEvts = allBuckets[bucket] ?? [];
    const tEvts = topBuckets[bucket] ?? [];
    const shareAll = all.length > 0 ? r4(aEvts.length / all.length) : 0;
    const shareTop = top.length > 0 ? r4(tEvts.length / top.length) : 0;
    return {
      bucket,
      countAll: aEvts.length,
      shareAll,
      countTop: tEvts.length,
      shareTop,
      enrichment: shareAll > 0 ? r4(shareTop / shareAll) : (shareTop > 0 ? 999 : 0),
      avgProxyAll: avg(aEvts.map(e => e.conservativeCaptureProxy)),
      avgProxyTop: avg(tEvts.map(e => e.conservativeCaptureProxy)),
    };
  }).sort((a, b) => b.enrichment - a.enrichment);
}

function deriveRules(
  all: readonly MomentumEvent[],
  top: readonly MomentumEvent[],
  byType: BucketComparison[],
  byLiq: BucketComparison[],
  byMag: BucketComparison[],
): CandidateFilterRule[] {
  const rules: CandidateFilterRule[] = [];
  const maxRules = OVERFIT_MAX_RULES();
  const avgAll = avg(all.map(e => e.conservativeCaptureProxy));

  const enrichedTypes = byType
    .filter(b => b.enrichment >= 1.3 && b.countTop >= 2)
    .map(b => b.bucket);

  if (enrichedTypes.length > 0 && enrichedTypes.length < byType.length) {
    const passing = all.filter(e => enrichedTypes.includes(e.eventType));
    const failing = all.filter(e => !enrichedTypes.includes(e.eventType));
    const avgP = avg(passing.map(e => e.conservativeCaptureProxy));
    const avgF = avg(failing.map(e => e.conservativeCaptureProxy));
    const imp = avgP !== null && avgAll !== null ? r4(avgP - avgAll) : null;
    rules.push({
      ruleId: "prefer_event_types",
      description: `Preferir eventTypes: ${enrichedTypes.join(", ")}`,
      rationale: `Sobre-representados no top slice (enrichment ≥1.3, count≥2). avgProxy passing=${avgP} vs all=${avgAll}.`,
      eventsPassingInAll: passing.length,
      eventsPassingInTop: top.filter(e => enrichedTypes.includes(e.eventType)).length,
      coverageAll: all.length > 0 ? r4(passing.length / all.length) : 0,
      coverageTop: top.length > 0 ? r4(top.filter(e => enrichedTypes.includes(e.eventType)).length / top.length) : 0,
      avgProxyPassing: avgP,
      avgProxyFailing: avgF,
      capturableRatePassing: capRate(passing),
      concentrationRiskPassing: concRisk(passing),
      improvementVsAll: imp,
      verdict: imp !== null && imp > 0.002 ? "beneficial" : imp !== null && imp > 0 ? "marginal" : "inconclusive",
    });
  }

  const goodLiqBuckets = byLiq
    .filter(b => b.enrichment >= 1.2 && b.countTop >= 2 && b.avgProxyTop !== null && (b.avgProxyAll === null || (b.avgProxyTop ?? 0) > (b.avgProxyAll ?? 0)))
    .map(b => b.bucket);

  if (goodLiqBuckets.length > 0 && rules.length < maxRules) {
    const passing = all.filter(e => goodLiqBuckets.includes(liqBucket(e.liquidityAtDetection)));
    const failing = all.filter(e => !goodLiqBuckets.includes(liqBucket(e.liquidityAtDetection)));
    const avgP = avg(passing.map(e => e.conservativeCaptureProxy));
    const avgF = avg(failing.map(e => e.conservativeCaptureProxy));
    const imp = avgP !== null && avgAll !== null ? r4(avgP - avgAll) : null;
    rules.push({
      ruleId: "prefer_liquidity_buckets",
      description: `Preferir liquidez: ${goodLiqBuckets.join(", ")}`,
      rationale: `Faixas sobre-representadas no top slice com melhor avgProxy.`,
      eventsPassingInAll: passing.length,
      eventsPassingInTop: top.filter(e => goodLiqBuckets.includes(liqBucket(e.liquidityAtDetection))).length,
      coverageAll: all.length > 0 ? r4(passing.length / all.length) : 0,
      coverageTop: top.length > 0 ? r4(top.filter(e => goodLiqBuckets.includes(liqBucket(e.liquidityAtDetection))).length / top.length) : 0,
      avgProxyPassing: avgP,
      avgProxyFailing: avgF,
      capturableRatePassing: capRate(passing),
      concentrationRiskPassing: concRisk(passing),
      improvementVsAll: imp,
      verdict: imp !== null && imp > 0.002 ? "beneficial" : imp !== null && imp > 0 ? "marginal" : "inconclusive",
    });
  }

  const goodMagBuckets = byMag
    .filter(b => b.enrichment >= 1.2 && b.countTop >= 2 && (b.avgProxyTop ?? -1) > (b.avgProxyAll ?? -1))
    .map(b => b.bucket);

  if (goodMagBuckets.length > 0 && rules.length < maxRules) {
    const passing = all.filter(e => goodMagBuckets.includes(magBucket(e.magnitude)));
    const failing = all.filter(e => !goodMagBuckets.includes(magBucket(e.magnitude)));
    const avgP = avg(passing.map(e => e.conservativeCaptureProxy));
    const avgF = avg(failing.map(e => e.conservativeCaptureProxy));
    const imp = avgP !== null && avgAll !== null ? r4(avgP - avgAll) : null;
    rules.push({
      ruleId: "prefer_magnitude_buckets",
      description: `Preferir magnitude: ${goodMagBuckets.join(", ")}`,
      rationale: `Faixas sobre-representadas no top slice com melhor avgProxy.`,
      eventsPassingInAll: passing.length,
      eventsPassingInTop: top.filter(e => goodMagBuckets.includes(magBucket(e.magnitude))).length,
      coverageAll: all.length > 0 ? r4(passing.length / all.length) : 0,
      coverageTop: top.length > 0 ? r4(top.filter(e => goodMagBuckets.includes(magBucket(e.magnitude))).length / top.length) : 0,
      avgProxyPassing: avgP,
      avgProxyFailing: avgF,
      capturableRatePassing: capRate(passing),
      concentrationRiskPassing: concRisk(passing),
      improvementVsAll: imp,
      verdict: imp !== null && imp > 0.002 ? "beneficial" : imp !== null && imp > 0 ? "marginal" : "inconclusive",
    });
  }

  const negTypes = byType.filter(b => b.enrichment < 0.5 && b.countAll >= 3);
  if (negTypes.length > 0 && rules.length < maxRules) {
    const excludeTypes = negTypes.map(b => b.bucket);
    const passing = all.filter(e => !excludeTypes.includes(e.eventType));
    const failing = all.filter(e => excludeTypes.includes(e.eventType));
    const avgP = avg(passing.map(e => e.conservativeCaptureProxy));
    const avgF = avg(failing.map(e => e.conservativeCaptureProxy));
    const imp = avgP !== null && avgAll !== null ? r4(avgP - avgAll) : null;
    if (imp !== null && imp > 0) {
      rules.push({
        ruleId: "exclude_weak_types",
        description: `Excluir eventTypes: ${excludeTypes.join(", ")}`,
        rationale: `Sub-representados no top slice (enrichment <0.5, count≥3 no all). avgProxy excluídos=${avgF}.`,
        eventsPassingInAll: passing.length,
        eventsPassingInTop: top.filter(e => !excludeTypes.includes(e.eventType)).length,
        coverageAll: all.length > 0 ? r4(passing.length / all.length) : 0,
        coverageTop: top.length > 0 ? r4(top.filter(e => !excludeTypes.includes(e.eventType)).length / top.length) : 0,
        avgProxyPassing: avgP,
        avgProxyFailing: avgF,
        capturableRatePassing: capRate(passing),
        concentrationRiskPassing: concRisk(passing),
        improvementVsAll: imp,
        verdict: imp > 0.002 ? "beneficial" : "marginal",
      });
    }
  }

  return rules;
}

export function buildTopSliceSelectionAssessment(
  allEvents: readonly MomentumEvent[],
): TopSliceSelectionAssessment {
  const minEv = MIN_EVENTS_FOR_SELECTION();
  const total = allEvents.length;

  const ranked = [...allEvents].sort((a, b) => scoreEvent(b) - scoreEvent(a));
  const topN = Math.max(1, Math.floor(total * 0.25));
  const topSlice = ranked.slice(0, topN);

  const byType = buildBucketComparison(allEvents, topSlice, e => e.eventType);
  const byLiq = buildBucketComparison(allEvents, topSlice, e => liqBucket(e.liquidityAtDetection));
  const byMag = buildBucketComparison(allEvents, topSlice, e => magBucket(e.magnitude));
  const byProxy = buildBucketComparison(allEvents, topSlice, e => proxyBucket(e.conservativeCaptureProxy));
  const byCategory: BucketComparison[] = [];

  const commonPatterns: string[] = [];
  const weakPatterns: string[] = [];

  for (const b of byType) {
    if (b.enrichment >= 1.5 && b.countTop >= 2) {
      commonPatterns.push(`eventType "${b.bucket}" sobre-representado no topo (enrichment=${b.enrichment}, share top=${b.shareTop} vs all=${b.shareAll})`);
    }
    if (b.enrichment < 0.5 && b.countAll >= 3) {
      weakPatterns.push(`eventType "${b.bucket}" sub-representado no topo (enrichment=${b.enrichment}, count all=${b.countAll})`);
    }
  }
  for (const b of byLiq) {
    if (b.enrichment >= 1.5 && b.countTop >= 2) {
      commonPatterns.push(`liquidez ${b.bucket} sobre-representada no topo (enrichment=${b.enrichment})`);
    }
    if (b.enrichment < 0.5 && b.countAll >= 3) {
      weakPatterns.push(`liquidez ${b.bucket} sub-representada no topo (enrichment=${b.enrichment})`);
    }
  }
  for (const b of byMag) {
    if (b.enrichment >= 1.5 && b.countTop >= 2) {
      commonPatterns.push(`magnitude ${b.bucket} sobre-representada no topo (enrichment=${b.enrichment})`);
    }
  }

  const rules = total >= minEv ? deriveRules(allEvents, topSlice, byType, byLiq, byMag) : [];

  const beneficialRules = rules.filter(r => r.verdict === "beneficial");
  const marginalRules = rules.filter(r => r.verdict === "marginal");

  let combinedImprovementEst: number | null = null;
  let combinedCoverageEst: number | null = null;
  let combinedConcRisk: number | null = null;

  if (beneficialRules.length > 0) {
    const bestRule = beneficialRules.sort(
      (a, b) => (b.improvementVsAll ?? 0) - (a.improvementVsAll ?? 0),
    )[0]!;
    combinedImprovementEst = bestRule.improvementVsAll;
    combinedCoverageEst = bestRule.coverageAll;
    combinedConcRisk = bestRule.concentrationRiskPassing;
  } else if (marginalRules.length > 0) {
    const bestRule = marginalRules.sort(
      (a, b) => (b.improvementVsAll ?? 0) - (a.improvementVsAll ?? 0),
    )[0]!;
    combinedImprovementEst = bestRule.improvementVsAll;
    combinedCoverageEst = bestRule.coverageAll;
    combinedConcRisk = bestRule.concentrationRiskPassing;
  }

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];

  let verdict: SelectionAssessmentVerdict;

  if (total < minEv) {
    verdict = "insufficient_sample";
    blockingReasons.push(`Eventos ${total} < mínimo ${minEv}.`);
  } else if (rules.length === 0) {
    verdict = "no_clear_filter_advantage";
    blockingReasons.push("Nenhum bucket com enrichment suficiente para propor regra. Distribuição uniforme entre top e all.");
  } else if (rules.length > OVERFIT_MAX_RULES()) {
    verdict = "overfit_risk";
    blockingReasons.push(`${rules.length} regras propostas > max seguro ${OVERFIT_MAX_RULES()}. Muitas dimensões com sinal = risco de overfit à amostra.`);
  } else if (beneficialRules.length >= 1) {
    const bestImp = beneficialRules[0]!.improvementVsAll ?? 0;
    const bestCov = beneficialRules[0]!.coverageAll;
    const bestConc = beneficialRules[0]!.concentrationRiskPassing;
    if (bestCov < 0.1) {
      verdict = "overfit_risk";
      blockingReasons.push(`Melhor regra cobre apenas ${r4(bestCov * 100)}% do universo. Filtro demasiado restritivo.`);
    } else if (bestConc > 0.6) {
      verdict = "weak_filter_opportunity";
      supportingReasons.push(`Regra beneficial (imp=${bestImp}) mas concentração=${r4(bestConc * 100)}%. Precisa diversificar.`);
    } else {
      verdict = "promising_filter_opportunity";
      supportingReasons.push(
        `${beneficialRules.length} regra(s) beneficial. Melhor: imp=${bestImp}, coverage=${r4(bestCov * 100)}%, conc=${r4(bestConc * 100)}%.`,
      );
    }
  } else if (marginalRules.length >= 1) {
    verdict = "weak_filter_opportunity";
    supportingReasons.push(`${marginalRules.length} regra(s) marginal; melhoria positiva mas abaixo de 0.002.`);
  } else {
    verdict = "no_clear_filter_advantage";
    blockingReasons.push("Regras derivadas são inconclusivas ou harmful. Sem filtro vantajoso observado.");
  }

  const featureSummaryParts: string[] = [];
  if (commonPatterns.length > 0) featureSummaryParts.push(`${commonPatterns.length} padrão(ões) sobre-representado(s) no topo`);
  if (weakPatterns.length > 0) featureSummaryParts.push(`${weakPatterns.length} padrão(ões) sub-representado(s)`);
  if (rules.length > 0) featureSummaryParts.push(`${rules.length} regra(s) candidata(s) derivada(s)`);
  const featureComparisonSummary = featureSummaryParts.length > 0
    ? featureSummaryParts.join("; ") + "."
    : "Sem diferença relevante entre top slice e universo geral.";

  return {
    totalEventsAnalyzed: total,
    topSliceEventsAnalyzed: topN,
    featureComparisonSummary,
    topVsAllByEventType: byType,
    topVsAllByLiquidityBucket: byLiq,
    topVsAllByMagnitudeBucket: byMag,
    topVsAllByCaptureProxyBucket: byProxy,
    topVsAllByMarketCategoryIfAvailable: byCategory,
    commonPatternsInTopSlice: commonPatterns,
    weakPatternsInTopSlice: weakPatterns,
    candidateFilterRules: rules,
    candidateFilterRulesBacktestLikeRead:
      "Leitura comparativa observacional sobre amostra in-sample. Não é backtest real — apenas compara distribuições de atributos entre top slice e universo completo. Qualquer regra proposta precisa de validação out-of-sample antes de uso.",
    filterImprovementEstimate: combinedImprovementEst,
    filterCoverageEstimate: combinedCoverageEst,
    filterConcentrationRisk: combinedConcRisk,
    selectionAssessmentVerdict: verdict,
    supportingReasons,
    blockingReasons,
    thresholdsUsed: {
      MOMENTUM_SELECTION_MIN_EVENTS: minEv,
      MOMENTUM_SELECTION_MAX_RULES: OVERFIT_MAX_RULES(),
    },
  };
}

export function buildSelectionSummaryLine(a: TopSliceSelectionAssessment): string {
  if (a.totalEventsAnalyzed < MIN_EVENTS_FOR_SELECTION()) {
    return `selection: insufficient_sample (${a.totalEventsAnalyzed} events)`;
  }
  const ruleCount = a.candidateFilterRules.length;
  const beneficial = a.candidateFilterRules.filter(r => r.verdict === "beneficial").length;
  const imp = a.filterImprovementEstimate !== null
    ? (a.filterImprovementEstimate > 0 ? "+" : "") + String(a.filterImprovementEstimate)
    : "n/a";
  return `selection: ${a.selectionAssessmentVerdict} | rules=${ruleCount} beneficial=${beneficial} bestImp=${imp} cov=${a.filterCoverageEstimate ?? "n/a"}`;
}
