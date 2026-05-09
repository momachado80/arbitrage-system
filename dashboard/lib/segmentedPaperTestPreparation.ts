/**
 * Segmented paper test preparation — wave-1 single segment only:
 * eventType == spread_spike. No liquidity/magnitude filters yet.
 * Readiness only; does not execute paper or change the winning base rule.
 */

import type { MomentumEvent } from "./momentumSnipingProbe";
import type { RankedEventAssessment } from "./momentumRankedEventAssessment";
import type { TopSliceRobustnessAssessment } from "./momentumTopSliceRobustness";
import type { TopSliceSelectionAssessment } from "./momentumTopSliceSelection";
import type { OperationalizationAssessment } from "./momentumOperationalization";
import type { OperationalizationRobustnessAssessment } from "./operationalizationRobustness";
import type { PromotionReadinessAssessment } from "./operationalizationPromotionReadiness";
import type { PromotionProgressAssessment } from "./promotionProgressTracker";
import type { RealisticPaperExecutionAssessment } from "./realisticPaperExecutionAssessment";
import type { ExecutionSurvivabilitySegmentation } from "./executionSurvivabilitySegmentation";

function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}
function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}
function avg(nums: number[]): number | null {
  if (nums.length === 0) return null;
  return r4(nums.reduce((a, b) => a + b, 0) / nums.length);
}

const MIN_UNIVERSE_EVENTS = () =>
  Math.max(8, Math.floor(envNum("MOMENTUM_SEG_PAPER_MIN_UNIVERSE", 12)));
const MIN_SEGMENT_EVENTS = () =>
  Math.max(4, Math.floor(envNum("MOMENTUM_SEG_PAPER_MIN_SEGMENT_EVENTS", 6)));
const MIN_SEGMENT_MARKETS = () =>
  Math.max(2, Math.floor(envNum("MOMENTUM_SEG_PAPER_MIN_SEGMENT_MARKETS", 2)));
const MAX_SEGMENT_CONC = () => envNum("MOMENTUM_SEG_PAPER_MAX_CONCENTRATION", 0.45);

export type SegmentedPaperPreparationVerdict =
  | "too_early_for_segmented_paper"
  | "candidate_segment_ready_for_paper"
  | "candidate_segment_not_ready";

export interface SegmentedPaperTestPreparation {
  segmentedPaperPreparationVerdict: SegmentedPaperPreparationVerdict;
  targetSegmentDefinition: string;
  targetSegmentEventCount: number;
  targetSegmentCoverage: number;
  targetSegmentDistinctMarkets: number;
  targetSegmentAvgProxy: number | null;
  targetSegmentImprovementVsAll: number | null;
  targetSegmentConcentrationRisk: number;
  targetSegmentTemporalFootprint: {
    firstDetectedAt: string | null;
    lastDetectedAt: string | null;
    spanMs: number | null;
  };
  targetSegmentWindowDistribution: Array<{ windowLabel: string; eventCount: number; share: number }>;
  targetSegmentReadinessRisks: string[];
  targetSegmentReadinessReasons: string[];
  nextEscalationRule: string;
  segmentedPaperPreparationSummaryLine: string;
  thresholdsUsed: Record<string, number | string>;
  readDisclaimer: string;
}

function concRisk(events: readonly MomentumEvent[]): number {
  if (events.length === 0) return 0;
  const c: Record<string, number> = {};
  for (const e of events) c[e.marketId] = (c[e.marketId] ?? 0) + 1;
  return r4(Math.max(...Object.values(c)) / events.length);
}

function temporalFootprint(segment: readonly MomentumEvent[]): {
  firstDetectedAt: string | null;
  lastDetectedAt: string | null;
  spanMs: number | null;
} {
  if (segment.length === 0) {
    return { firstDetectedAt: null, lastDetectedAt: null, spanMs: null };
  }
  const ts = segment.map(e => Date.parse(e.detectedAt)).filter(t => Number.isFinite(t));
  if (ts.length === 0) {
    return { firstDetectedAt: null, lastDetectedAt: null, spanMs: null };
  }
  const minT = Math.min(...ts);
  const maxT = Math.max(...ts);
  const first = segment.find(e => Date.parse(e.detectedAt) === minT)?.detectedAt ?? null;
  const last = segment.find(e => Date.parse(e.detectedAt) === maxT)?.detectedAt ?? null;
  return {
    firstDetectedAt: first,
    lastDetectedAt: last,
    spanMs: maxT >= minT ? maxT - minT : null,
  };
}

function windowDistribution(segment: readonly MomentumEvent[]): Array<{
  windowLabel: string;
  eventCount: number;
  share: number;
}> {
  if (segment.length === 0) return [];
  const ts = segment.map(e => ({ e, t: Date.parse(e.detectedAt) }));
  const valid = ts.filter(x => Number.isFinite(x.t));
  if (valid.length === 0) {
    return [{ windowLabel: "unknown_time", eventCount: segment.length, share: 1 }];
  }
  const tMin = Math.min(...valid.map(x => x.t));
  const tMax = Math.max(...valid.map(x => x.t));
  const span = Math.max(1, tMax - tMin);
  const n = 4;
  const counts = new Array(n).fill(0) as number[];
  for (const { t } of valid) {
    const b = Math.min(n - 1, Math.floor(((t - tMin) / span) * n));
    counts[b]!++;
  }
  const total = segment.length;
  return counts.map((eventCount, i) => ({
    windowLabel: `quartile_${i + 1}_of_span`,
    eventCount,
    share: total > 0 ? r4(eventCount / total) : 0,
  }));
}

function spreadSpikeBeneficialRule(selection: TopSliceSelectionAssessment): boolean {
  return selection.candidateFilterRules.some(
    r =>
      r.verdict === "beneficial" &&
      (r.ruleId.includes("spread_spike") || r.description.toLowerCase().includes("spread_spike")),
  );
}

export function buildSegmentedPaperTestPreparation(
  allEvents: readonly MomentumEvent[],
  ranked: RankedEventAssessment,
  robustness: TopSliceRobustnessAssessment,
  selection: TopSliceSelectionAssessment,
  ops: OperationalizationAssessment,
  opsRobustness: OperationalizationRobustnessAssessment,
  promo: PromotionReadinessAssessment,
  progress: PromotionProgressAssessment,
  paperExec: RealisticPaperExecutionAssessment,
  survivability: ExecutionSurvivabilitySegmentation,
): SegmentedPaperTestPreparation {
  const minUniverse = MIN_UNIVERSE_EVENTS();
  const minSegEv = MIN_SEGMENT_EVENTS();
  const minSegMk = MIN_SEGMENT_MARKETS();
  const maxConc = MAX_SEGMENT_CONC();

  const segment = allEvents.filter(e => e.eventType === "spread_spike");
  const nAll = allEvents.length;
  const nSeg = segment.length;

  const coverage = nAll > 0 ? r4(nSeg / nAll) : 0;
  const distinctMkts = new Set(segment.map(e => e.marketId)).size;
  const avgAll = avg(allEvents.map(e => e.conservativeCaptureProxy));
  const avgSeg = avg(segment.map(e => e.conservativeCaptureProxy));
  const improvementVsAll =
    avgAll !== null && avgSeg !== null ? r4(avgSeg - avgAll) : null;
  const segConc = concRisk(segment);
  const footprint = temporalFootprint(segment);
  const winDist = windowDistribution(segment);

  const risks: string[] = [];
  const reasons: string[] = [];

  if (opsRobustness.overfitRiskVerdict === "high") {
    risks.push("Overfit operacional alto no histórico global; paper segmentado é leitura frágil.");
  }
  if (paperExec.realisticPaperExecutionVerdict === "edge_destroyed_by_friction") {
    risks.push("Paper realista global com edge destruído por fricção; segmento pode não sobreviver a execução.");
  }
  if (survivability.executionSurvivabilityVerdict === "no_viable_subset_found") {
    risks.push("Survivability não encontrou subset viável sob fricções; spread_spike isolado ainda pode falhar no paper.");
  }
  if (promo.distinctPassingMarkets < 2) {
    risks.push("Pouca amplitude global (mercados passing na promo); cautela ao generalizar.");
  }
  if (progress.progressToRobustPct < 0.5) {
    risks.push("Progresso para robust operacional <50%; disciplina de promo ainda incompleta.");
  }
  const rankedOk =
    ranked.rankedSignalVerdict === "promising_top_slice_signal" ||
    ranked.rankedSignalVerdict === "concentrated_but_interesting";
  const robustOk =
    robustness.topSliceRobustnessVerdict === "improving_and_diversifying" ||
    robustness.topSliceRobustnessVerdict === "robust_top_slice_signal" ||
    robustness.topSliceRobustnessVerdict === "weak_but_persistent";
  const selectionOk =
    selection.selectionAssessmentVerdict === "promising_filter_opportunity" ||
    selection.selectionAssessmentVerdict === "weak_filter_opportunity";
  const spikeRuleAligned = spreadSpikeBeneficialRule(selection);

  let verdict: SegmentedPaperPreparationVerdict;

  if (
    ranked.rankedSignalVerdict === "insufficient_sample" ||
    selection.selectionAssessmentVerdict === "insufficient_sample" ||
    nAll < minUniverse
  ) {
    verdict = "too_early_for_segmented_paper";
    reasons.push(
      `Universo ou camadas base ainda cedo (ranked/selection insufficient ou N=${nAll}<${minUniverse}).`,
    );
  } else if (
    nSeg < minSegEv ||
    distinctMkts < minSegMk ||
    segConc > maxConc ||
    !rankedOk ||
    !robustOk ||
    !selectionOk
  ) {
    verdict = "candidate_segment_not_ready";
    if (nSeg < minSegEv) reasons.push(`spread_spike count ${nSeg} < ${minSegEv}.`);
    if (distinctMkts < minSegMk) reasons.push(`Mercados distintos no segmento ${distinctMkts} < ${minSegMk}.`);
    if (segConc > maxConc) reasons.push(`Concentração no segmento ${segConc} > ${maxConc}.`);
    if (!rankedOk) reasons.push(`Ranked verdict não sustenta segmento (${ranked.rankedSignalVerdict}).`);
    if (!robustOk) reasons.push(`Robustez não sustenta paper (${robustness.topSliceRobustnessVerdict}).`);
    if (!selectionOk) reasons.push(`Seleção não sustenta (${selection.selectionAssessmentVerdict}).`);
  } else {
    verdict = "candidate_segment_ready_for_paper";
    reasons.push(
      `spread_spike: N=${nSeg}, coverage=${coverage}, mercados=${distinctMkts}, conc=${segConc}; ranked/robust/selection alinhados.`,
    );
    if (!spikeRuleAligned) {
      risks.push("Regra beneficial explícita para spread_spike não encontrada na lista de candidatos; segmento ainda coerente com tipo de evento.");
    }
  }

  const nextEscalationRule =
    "Onda 2 (só após leitura estável da onda 1): acrescentar magnitude >= 5% sobre spread_spike; onda 3: depois considerar liquidez 1k–10k. Não combinar antes da validação da onda 1.";

  const thresholdsUsed: Record<string, number | string> = {
    MOMENTUM_SEG_PAPER_MIN_UNIVERSE: minUniverse,
    MOMENTUM_SEG_PAPER_MIN_SEGMENT_EVENTS: minSegEv,
    MOMENTUM_SEG_PAPER_MIN_SEGMENT_MARKETS: minSegMk,
    MOMENTUM_SEG_PAPER_MAX_CONCENTRATION: maxConc,
    context_bestOperationalRuleSet: ops.bestOperationalRuleSet?.ruleSetLabel ?? "n/a",
  };

  const readDisclaimer =
    "Preparação para paper segmentado observacional (wave-1: só spread_spike). Não altera a regra base nem executa ordens. Onda 2/3 (magnitude, liquidez) ficam deferidas até leitura da onda 1.";

  const base: SegmentedPaperTestPreparation = {
    segmentedPaperPreparationVerdict: verdict,
    targetSegmentDefinition:
      "eventType == spread_spike (wave-1 única; sem filtro de liquidez 1k–10k nem magnitude >=5%)",
    targetSegmentEventCount: nSeg,
    targetSegmentCoverage: coverage,
    targetSegmentDistinctMarkets: distinctMkts,
    targetSegmentAvgProxy: avgSeg,
    targetSegmentImprovementVsAll: improvementVsAll,
    targetSegmentConcentrationRisk: segConc,
    targetSegmentTemporalFootprint: footprint,
    targetSegmentWindowDistribution: winDist,
    targetSegmentReadinessRisks: risks,
    targetSegmentReadinessReasons: reasons,
    nextEscalationRule,
    segmentedPaperPreparationSummaryLine: "",
    thresholdsUsed,
    readDisclaimer,
  };

  base.segmentedPaperPreparationSummaryLine = buildSegmentedPaperPreparationSummaryLine(base);
  return base;
}

export function buildSegmentedPaperPreparationSummaryLine(
  a: SegmentedPaperTestPreparation,
): string {
  return `segPaper: ${a.segmentedPaperPreparationVerdict} | spread_spike n=${a.targetSegmentEventCount} cov=${a.targetSegmentCoverage} mkts=${a.targetSegmentDistinctMarkets} conc=${a.targetSegmentConcentrationRisk}`;
}
