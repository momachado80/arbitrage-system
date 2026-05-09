/**
 * Segmented paper execution assessment — wave-1 only (eventType == spread_spike).
 * Same friction model as realisticPaperExecutionAssessment (simulateTrade); comparable read.
 * Does not execute live orders or change the base operational rule.
 */

import type { MomentumEvent } from "./momentumSnipingProbe";
import type { OperationalizationAssessment } from "./momentumOperationalization";
import type { OperationalizationRobustnessAssessment } from "./operationalizationRobustness";
import type { PromotionReadinessAssessment } from "./operationalizationPromotionReadiness";
import type { PromotionProgressAssessment } from "./promotionProgressTracker";
import type { RealisticPaperExecutionAssessment } from "./realisticPaperExecutionAssessment";
import type { ExecutionSurvivabilitySegmentation } from "./executionSurvivabilitySegmentation";
import type { SegmentedPaperTestPreparation } from "./segmentedPaperTestPreparation";
import {
  simulateTrade,
  type SimulatedTrade,
  type ExecutionFragilityVerdict,
} from "./realisticPaperExecutionAssessment";

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

const MIN_SEGMENT_FOR_EXEC = () =>
  Math.max(4, Math.floor(envNum("MOMENTUM_SEG_EXEC_MIN_EVENTS", 6)));

export type SegmentedPaperExecutionVerdict =
  | "insufficient_sample"
  | "segment_fails_under_friction"
  | "segment_survives_but_weak"
  | "segment_survives_friction";

export interface SegmentedDegradationBucket {
  label: string;
  count: number;
  avgOriginalProxy: number | null;
  avgNetProxy: number | null;
  degradationPct: number | null;
}

export interface SegmentedPaperExecutionAssessment {
  segmentedPaperExecutionVerdict: SegmentedPaperExecutionVerdict;
  targetSegmentDefinition: string;
  segmentedSimulatedTradeCount: number;
  segmentedSimulatedFillRate: number;
  segmentedAverageSlippage: number | null;
  segmentedLatencyPenalty: number | null;
  segmentedExitPenalty: number | null;
  segmentedNetImprovementVsAll: number | null;
  segmentedNetImprovementVsBaseline: number | null;
  segmentedPnLProxy: number;
  segmentedExecutionDegradationPct: number | null;
  segmentedDegradationByExitReason: SegmentedDegradationBucket[];
  segmentedDegradationByMarket: SegmentedDegradationBucket[];
  segmentedExecutionFragilityVerdict: ExecutionFragilityVerdict;
  segmentedExecutionReasons: string[];
  segmentedExecutionRisks: string[];
  nextEscalationRule: string;
  segmentedPaperExecutionSummaryLine: string;
  thresholdsUsed: Record<string, number | string>;
  readDisclaimer: string;
}

function buildDegradationByKey(
  trades: SimulatedTrade[],
  keyFn: (t: SimulatedTrade) => string,
): SegmentedDegradationBucket[] {
  const groups: Record<string, SimulatedTrade[]> = {};
  for (const t of trades) {
    const k = keyFn(t);
    (groups[k] ??= []).push(t);
  }
  return Object.entries(groups)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 10)
    .map(([label, ts]) => {
      const origAvg = avg(ts.map(t => t.originalProxy));
      const netAvg = avg(ts.filter(t => t.filled).map(t => t.netProxy));
      const deg =
        origAvg !== null && origAvg > 0 && netAvg !== null
          ? r4(1 - netAvg / origAvg)
          : null;
      return { label, count: ts.length, avgOriginalProxy: origAvg, avgNetProxy: netAvg, degradationPct: deg };
    });
}

function fragilityFrom(fillRate: number, degradationPct: number | null): ExecutionFragilityVerdict {
  if (fillRate >= 0.6 && (degradationPct === null || degradationPct < 0.5)) {
    return "robust_execution";
  }
  if (fillRate >= 0.4 && (degradationPct === null || degradationPct < 0.8)) {
    return "moderate_fragility";
  }
  return "high_fragility";
}

export function buildSegmentedPaperExecutionAssessment(
  allEvents: readonly MomentumEvent[],
  ops: OperationalizationAssessment,
  opsRobustness: OperationalizationRobustnessAssessment,
  promo: PromotionReadinessAssessment,
  progress: PromotionProgressAssessment,
  paperExec: RealisticPaperExecutionAssessment,
  survivability: ExecutionSurvivabilitySegmentation,
  preparation: SegmentedPaperTestPreparation,
): SegmentedPaperExecutionAssessment {
  const minSeg = MIN_SEGMENT_FOR_EXEC();
  const segment = allEvents.filter(e => e.eventType === "spread_spike");
  const n = segment.length;

  const targetSegmentDefinition =
    "eventType == spread_spike (wave-1; mesmas fricções que realisticPaper; sem magnitude/liquidez)";

  const reasons: string[] = [];
  const risks: string[] = [];

  if (preparation.segmentedPaperPreparationVerdict === "too_early_for_segmented_paper") {
    const empty: SegmentedPaperExecutionAssessment = {
      segmentedPaperExecutionVerdict: "insufficient_sample",
      targetSegmentDefinition,
      segmentedSimulatedTradeCount: 0,
      segmentedSimulatedFillRate: 0,
      segmentedAverageSlippage: null,
      segmentedLatencyPenalty: null,
      segmentedExitPenalty: null,
      segmentedNetImprovementVsAll: null,
      segmentedNetImprovementVsBaseline: null,
      segmentedPnLProxy: 0,
      segmentedExecutionDegradationPct: null,
      segmentedDegradationByExitReason: [],
      segmentedDegradationByMarket: [],
      segmentedExecutionFragilityVerdict: "high_fragility",
      segmentedExecutionReasons: [
        "Preparação wave-1 em too_early_for_segmented_paper; execução segmentada não avaliada.",
      ],
      segmentedExecutionRisks: [],
      nextEscalationRule: preparation.nextEscalationRule,
      segmentedPaperExecutionSummaryLine: "",
      thresholdsUsed: {
        MOMENTUM_SEG_EXEC_MIN_EVENTS: minSeg,
        friction_model: "same_as_realisticPaper",
        context_bestOperationalRuleSet: ops.bestOperationalRuleSet?.ruleSetLabel ?? "n/a",
      },
      readDisclaimer:
        "Simulação paper segmentada wave-1; fricções idênticas ao módulo global. Não é PnL real.",
    };
    empty.segmentedPaperExecutionSummaryLine = buildSegmentedPaperExecutionSummaryLine(empty);
    return empty;
  }

  if (n < minSeg) {
    const empty: SegmentedPaperExecutionAssessment = {
      segmentedPaperExecutionVerdict: "insufficient_sample",
      targetSegmentDefinition,
      segmentedSimulatedTradeCount: 0,
      segmentedSimulatedFillRate: 0,
      segmentedAverageSlippage: null,
      segmentedLatencyPenalty: null,
      segmentedExitPenalty: null,
      segmentedNetImprovementVsAll: null,
      segmentedNetImprovementVsBaseline: null,
      segmentedPnLProxy: 0,
      segmentedExecutionDegradationPct: null,
      segmentedDegradationByExitReason: [],
      segmentedDegradationByMarket: [],
      segmentedExecutionFragilityVerdict: "high_fragility",
      segmentedExecutionReasons: [`spread_spike N=${n} < mínimo ${minSeg} para leitura de execução.`],
      segmentedExecutionRisks: [],
      nextEscalationRule: preparation.nextEscalationRule,
      segmentedPaperExecutionSummaryLine: "",
      thresholdsUsed: {
        MOMENTUM_SEG_EXEC_MIN_EVENTS: minSeg,
        friction_model: "same_as_realisticPaper",
        context_bestOperationalRuleSet: ops.bestOperationalRuleSet?.ruleSetLabel ?? "n/a",
      },
      readDisclaimer:
        "Simulação paper segmentada wave-1; fricções idênticas ao módulo global. Não é PnL real.",
    };
    empty.segmentedPaperExecutionSummaryLine = buildSegmentedPaperExecutionSummaryLine(empty);
    return empty;
  }

  const trades = segment.map(e => simulateTrade(e));
  const filled = trades.filter(t => t.filled);
  const fillRate = n > 0 ? r4(filled.length / n) : 0;

  const avgSlip = avg(filled.map(t => t.exitSlippage));
  const avgLat = avg(filled.map(t => t.latencyDecay));
  const avgExit = avg(filled.map(t => t.exitSlippage));

  const baselineUniverse = avg(allEvents.map(e => e.conservativeCaptureProxy));
  const baselineSegmentPreFriction = avg(segment.map(e => e.conservativeCaptureProxy));
  const netAvg = avg(filled.map(t => t.netProxy));

  const impVsAll =
    netAvg !== null && baselineUniverse !== null ? r4(netAvg - baselineUniverse) : null;
  const impVsBaseline =
    netAvg !== null && baselineSegmentPreFriction !== null
      ? r4(netAvg - baselineSegmentPreFriction)
      : null;

  const cumPnL = r4(filled.reduce((s, t) => s + t.netProxy, 0));
  const degradationPct =
    baselineSegmentPreFriction !== null &&
    baselineSegmentPreFriction > 0 &&
    netAvg !== null
      ? r4(1 - netAvg / baselineSegmentPreFriction)
      : null;

  const byExit = buildDegradationByKey(filled, t => t.exitReason);
  const byMarket = buildDegradationByKey(filled, t => t.marketId.slice(0, 20));

  const frag = fragilityFrom(fillRate, degradationPct);

  const globalFilledAvg =
    paperExec.simulatedTradeCount > 0
      ? r4(paperExec.simulatedPnLProxy / paperExec.simulatedTradeCount)
      : null;

  if (opsRobustness.overfitRiskVerdict === "high") {
    risks.push("Overfit operacional alto; leitura segmentada frágil.");
  }
  if (paperExec.realisticPaperExecutionVerdict === "edge_destroyed_by_friction") {
    risks.push("Paper global com edge destruído; comparar segmento com cautela extrema.");
  }
  if (survivability.executionSurvivabilityVerdict === "no_viable_subset_found") {
    risks.push("Survivability global sem subset viável; segmento spread_spike pode ser excepção pontual.");
  }
  if (promo.distinctPassingMarkets < 2) {
    risks.push("Amplitude promo baixa.");
  }
  if (progress.progressToRobustPct < 0.5) {
    risks.push("Progresso a robust <50%.");
  }

  reasons.push(
    `spread_spike: simulados N=${n}, fills=${filled.length}, fillRate=${fillRate}, avgNet=${netAvg ?? "n/a"}.`,
  );
  if (globalFilledAvg !== null && netAvg !== null) {
    reasons.push(`Média net filled global=${globalFilledAvg} vs segmento=${netAvg}.`);
  }
  if (impVsAll !== null && paperExec.simulatedNetImprovementVsBaseline !== null) {
    reasons.push(
      `NetImp vs universo segment=${impVsAll} vs global paper netImp=${paperExec.simulatedNetImprovementVsBaseline}.`,
    );
  }

  let verdict: SegmentedPaperExecutionVerdict;

  if (netAvg === null || filled.length === 0) {
    verdict = "segment_fails_under_friction";
    reasons.push("Nenhum fill simulado ou net médio indefinido.");
  } else if (netAvg <= 0) {
    verdict = "segment_fails_under_friction";
    reasons.push("Média net após fricções <= 0 no segmento.");
  } else if (
    netAvg > 0.002 &&
    fillRate >= 0.4 &&
    (degradationPct === null || degradationPct < 0.85) &&
    (globalFilledAvg === null || netAvg >= globalFilledAvg - 0.001)
  ) {
    verdict = "segment_survives_friction";
    reasons.push("Segmento mantém net positivo com fill e degradação compatíveis; comparável ou melhor que global filled avg.");
  } else if (netAvg > 0) {
    verdict = "segment_survives_but_weak";
    reasons.push("Net positivo mas fraco, fill baixo, degradação elevada ou abaixo do global filled avg.");
  } else {
    verdict = "segment_fails_under_friction";
  }

  const nextEscalationRule = preparation.nextEscalationRule;

  const thresholdsUsed: Record<string, number | string> = {
    MOMENTUM_SEG_EXEC_MIN_EVENTS: minSeg,
    friction_model: "same_as_realisticPaper_MOMENTUM_PAPER_*",
    preparation_verdict: preparation.segmentedPaperPreparationVerdict,
    context_bestOperationalRuleSet: ops.bestOperationalRuleSet?.ruleSetLabel ?? "n/a",
  };

  const base: SegmentedPaperExecutionAssessment = {
    segmentedPaperExecutionVerdict: verdict,
    targetSegmentDefinition,
    segmentedSimulatedTradeCount: filled.length,
    segmentedSimulatedFillRate: fillRate,
    segmentedAverageSlippage: avgSlip,
    segmentedLatencyPenalty: avgLat,
    segmentedExitPenalty: avgExit,
    segmentedNetImprovementVsAll: impVsAll,
    segmentedNetImprovementVsBaseline: impVsBaseline,
    segmentedPnLProxy: cumPnL,
    segmentedExecutionDegradationPct: degradationPct,
    segmentedDegradationByExitReason: byExit,
    segmentedDegradationByMarket: byMarket,
    segmentedExecutionFragilityVerdict: frag,
    segmentedExecutionReasons: reasons,
    segmentedExecutionRisks: risks,
    nextEscalationRule,
    segmentedPaperExecutionSummaryLine: "",
    thresholdsUsed,
    readDisclaimer:
      "Execução paper segmentada wave-1 (spread_spike) com fricções iguais ao realisticPaper. Não altera regra base; não é exchange. wave-2/3 deferidas até leitura da wave-1.",
  };

  base.segmentedPaperExecutionSummaryLine = buildSegmentedPaperExecutionSummaryLine(base);
  return base;
}

export function buildSegmentedPaperExecutionSummaryLine(
  a: SegmentedPaperExecutionAssessment,
): string {
  const v = a.segmentedPaperExecutionVerdict;
  const n = a.segmentedSimulatedTradeCount;
  const fr = a.segmentedSimulatedFillRate;
  const imp = a.segmentedNetImprovementVsAll;
  const impS = imp !== null ? (imp > 0 ? "+" : "") + String(imp) : "n/a";
  const deg =
    a.segmentedExecutionDegradationPct !== null
      ? String(r4(a.segmentedExecutionDegradationPct * 100)) + "%"
      : "n/a";
  return `segExec: ${v} | fills=${n} rate=${fr} netImpAll=${impS} deg=${deg} frag=${a.segmentedExecutionFragilityVerdict}`;
}
