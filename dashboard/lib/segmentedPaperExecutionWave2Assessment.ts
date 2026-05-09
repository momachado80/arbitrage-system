/**
 * Segmented paper execution assessment — wave-2 only:
 * eventType == spread_spike AND magnitude >= 5% (configurable).
 * Same simulateTrade frictions as global and wave-1. Liquidez 1k–10k fica wave-3.
 */

import type { MomentumEvent } from "./momentumSnipingProbe";
import type { OperationalizationAssessment } from "./momentumOperationalization";
import type { OperationalizationRobustnessAssessment } from "./operationalizationRobustness";
import type { PromotionReadinessAssessment } from "./operationalizationPromotionReadiness";
import type { PromotionProgressAssessment } from "./promotionProgressTracker";
import type { RealisticPaperExecutionAssessment } from "./realisticPaperExecutionAssessment";
import type { ExecutionSurvivabilitySegmentation } from "./executionSurvivabilitySegmentation";
import type { SegmentedPaperTestPreparation } from "./segmentedPaperTestPreparation";
import type { SegmentedPaperExecutionAssessment } from "./segmentedPaperExecutionAssessment";
import {
  simulateTrade,
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

const MIN_W2_SEGMENT = () =>
  Math.max(4, Math.floor(envNum("MOMENTUM_SEG_W2_EXEC_MIN_EVENTS", 6)));
const W2_MAGNITUDE_MIN = () => envNum("MOMENTUM_SEG_W2_MAGNITUDE_MIN", 0.05);

export type SegmentedWave2ExecutionVerdict =
  | "insufficient_sample"
  | "wave2_fails_under_friction"
  | "wave2_survives_but_weak"
  | "wave2_survives_friction";

export interface SegmentedPaperExecutionWave2Assessment {
  segmentedWave2ExecutionVerdict: SegmentedWave2ExecutionVerdict;
  targetSegmentDefinition: string;
  segmentedWave2SimulatedTradeCount: number;
  segmentedWave2SimulatedFillRate: number;
  segmentedWave2AverageSlippage: number | null;
  segmentedWave2LatencyPenalty: number | null;
  segmentedWave2ExitPenalty: number | null;
  segmentedWave2NetImprovementVsAll: number | null;
  segmentedWave2NetImprovementVsBaseline: number | null;
  segmentedWave2PnLProxy: number;
  segmentedWave2ExecutionDegradationPct: number | null;
  segmentedWave2ExecutionFragilityVerdict: ExecutionFragilityVerdict;
  segmentedWave2ExecutionReasons: string[];
  segmentedWave2ExecutionRisks: string[];
  nextEscalationRule: string;
  segmentedWave2ExecutionSummaryLine: string;
  thresholdsUsed: Record<string, number | string>;
  readDisclaimer: string;
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

function wave1FilledAvg(w1: SegmentedPaperExecutionAssessment): number | null {
  const c = w1.segmentedSimulatedTradeCount;
  if (c <= 0) return null;
  return r4(w1.segmentedPnLProxy / c);
}

export function buildSegmentedPaperExecutionWave2Assessment(
  allEvents: readonly MomentumEvent[],
  ops: OperationalizationAssessment,
  opsRobustness: OperationalizationRobustnessAssessment,
  promo: PromotionReadinessAssessment,
  progress: PromotionProgressAssessment,
  paperExec: RealisticPaperExecutionAssessment,
  survivability: ExecutionSurvivabilitySegmentation,
  preparation: SegmentedPaperTestPreparation,
  wave1Exec: SegmentedPaperExecutionAssessment,
): SegmentedPaperExecutionWave2Assessment {
  const minSeg = MIN_W2_SEGMENT();
  const magMin = W2_MAGNITUDE_MIN();
  const segment = allEvents.filter(
    e => e.eventType === "spread_spike" && e.magnitude >= magMin,
  );
  const n = segment.length;

  const targetSegmentDefinition = `eventType == spread_spike AND magnitude >= ${magMin} (wave-2; fricções idênticas a realisticPaper; sem filtro liquidez)`;

  const nextEscalationRule =
    "wave-3 (só se wave-2 justificar): acrescentar liquidez 1k–10k sobre spread_spike + magnitude >= limiar wave-2; não combinar antes de leitura estável da wave-2.";

  const emptyBase = (
    verdict: SegmentedWave2ExecutionVerdict,
    reasons: string[],
    risks: string[],
  ): SegmentedPaperExecutionWave2Assessment => {
    const out: SegmentedPaperExecutionWave2Assessment = {
      segmentedWave2ExecutionVerdict: verdict,
      targetSegmentDefinition,
      segmentedWave2SimulatedTradeCount: 0,
      segmentedWave2SimulatedFillRate: 0,
      segmentedWave2AverageSlippage: null,
      segmentedWave2LatencyPenalty: null,
      segmentedWave2ExitPenalty: null,
      segmentedWave2NetImprovementVsAll: null,
      segmentedWave2NetImprovementVsBaseline: null,
      segmentedWave2PnLProxy: 0,
      segmentedWave2ExecutionDegradationPct: null,
      segmentedWave2ExecutionFragilityVerdict: "high_fragility",
      segmentedWave2ExecutionReasons: reasons,
      segmentedWave2ExecutionRisks: risks,
      nextEscalationRule,
      segmentedWave2ExecutionSummaryLine: "",
      thresholdsUsed: {
        MOMENTUM_SEG_W2_EXEC_MIN_EVENTS: minSeg,
        MOMENTUM_SEG_W2_MAGNITUDE_MIN: magMin,
        friction_model: "same_as_realisticPaper",
        context_bestOperationalRuleSet: ops.bestOperationalRuleSet?.ruleSetLabel ?? "n/a",
        preparation_verdict: preparation.segmentedPaperPreparationVerdict,
      },
      readDisclaimer:
        "Paper segmentado wave-2 (spread_spike + magnitude). Fricções iguais ao módulo global. wave-3 (liquidez) deferida. Não altera regra base.",
    };
    out.segmentedWave2ExecutionSummaryLine = buildSegmentedWave2ExecutionSummaryLine(out);
    return out;
  };

  if (preparation.segmentedPaperPreparationVerdict === "too_early_for_segmented_paper") {
    return emptyBase("insufficient_sample", [
      "Preparação wave-1 em too_early; wave-2 exec não avaliada (governação alinhada à wave-1).",
    ], []);
  }

  if (n < minSeg) {
    return emptyBase("insufficient_sample", [
      `wave-2 segmento N=${n} < mínimo ${minSeg} (spread_spike + magnitude>=${magMin}).`,
    ], []);
  }

  const trades = segment.map(e => simulateTrade(e));
  const filled = trades.filter(t => t.filled);
  const fillRate = n > 0 ? r4(filled.length / n) : 0;

  const avgSlip = avg(filled.map(t => t.exitSlippage));
  const avgLat = avg(filled.map(t => t.latencyDecay));
  const avgExit = avg(filled.map(t => t.exitSlippage));

  const baselineUniverse = avg(allEvents.map(e => e.conservativeCaptureProxy));
  const baselineSegmentPre = avg(segment.map(e => e.conservativeCaptureProxy));
  const netAvg = avg(filled.map(t => t.netProxy));

  const impVsAll =
    netAvg !== null && baselineUniverse !== null ? r4(netAvg - baselineUniverse) : null;
  const impVsBaseline =
    netAvg !== null && baselineSegmentPre !== null ? r4(netAvg - baselineSegmentPre) : null;

  const cumPnL = r4(filled.reduce((s, t) => s + t.netProxy, 0));
  const degradationPct =
    baselineSegmentPre !== null && baselineSegmentPre > 0 && netAvg !== null
      ? r4(1 - netAvg / baselineSegmentPre)
      : null;

  const frag = fragilityFrom(fillRate, degradationPct);

  const globalFilledAvg =
    paperExec.simulatedTradeCount > 0
      ? r4(paperExec.simulatedPnLProxy / paperExec.simulatedTradeCount)
      : null;

  const reasons: string[] = [];
  const risks: string[] = [];

  if (opsRobustness.overfitRiskVerdict === "high") {
    risks.push("Overfit operacional alto; wave-2 frágil.");
  }
  if (paperExec.realisticPaperExecutionVerdict === "edge_destroyed_by_friction") {
    risks.push("Paper global edge destruído; comparar wave-2 com cautela.");
  }
  if (survivability.executionSurvivabilityVerdict === "no_viable_subset_found") {
    risks.push("Survivability sem subset viável globalmente.");
  }
  if (promo.distinctPassingMarkets < 2) {
    risks.push("Amplitude promo baixa.");
  }
  if (progress.progressToRobustPct < 0.5) {
    risks.push("Progresso a robust <50%.");
  }

  reasons.push(
    `wave-2: N=${n}, fills=${filled.length}, fillRate=${fillRate}, avgNet=${netAvg ?? "n/a"} (spread_spike + mag>=${magMin}).`,
  );
  if (globalFilledAvg !== null && netAvg !== null) {
    reasons.push(`Média net filled global=${globalFilledAvg} vs wave-2=${netAvg}.`);
  }

  const w1Avg = wave1FilledAvg(wave1Exec);
  if (w1Avg !== null && netAvg !== null) {
    const delta = r4(netAvg - w1Avg);
    reasons.push(`Comparação wave-1 filled avg net=${w1Avg} vs wave-2=${netAvg} (delta=${delta}).`);
    if (delta > 0.002) {
      reasons.push("wave-2 melhora material vs wave-1 em média net filled (limiar conservador +0.002).");
    } else if (delta < -0.002) {
      risks.push("wave-2 média net filled abaixo de wave-1 (delta negativo).");
    }
  }

  let verdict: SegmentedWave2ExecutionVerdict;

  if (netAvg === null || filled.length === 0) {
    verdict = "wave2_fails_under_friction";
    reasons.push("Sem fills ou net médio indefinido.");
  } else if (netAvg <= 0) {
    verdict = "wave2_fails_under_friction";
    reasons.push("Média net após fricções <= 0.");
  } else if (
    netAvg > 0.002 &&
    fillRate >= 0.4 &&
    (degradationPct === null || degradationPct < 0.85) &&
    (globalFilledAvg === null || netAvg >= globalFilledAvg - 0.001)
  ) {
    verdict = "wave2_survives_friction";
    reasons.push("Net positivo, fill e degradação OK vs global filled avg.");
  } else if (netAvg > 0) {
    verdict = "wave2_survives_but_weak";
    reasons.push("Net positivo mas fraco vs critérios strict (fill/degradação/global).");
  } else {
    verdict = "wave2_fails_under_friction";
  }

  const out: SegmentedPaperExecutionWave2Assessment = {
    segmentedWave2ExecutionVerdict: verdict,
    targetSegmentDefinition,
    segmentedWave2SimulatedTradeCount: filled.length,
    segmentedWave2SimulatedFillRate: fillRate,
    segmentedWave2AverageSlippage: avgSlip,
    segmentedWave2LatencyPenalty: avgLat,
    segmentedWave2ExitPenalty: avgExit,
    segmentedWave2NetImprovementVsAll: impVsAll,
    segmentedWave2NetImprovementVsBaseline: impVsBaseline,
    segmentedWave2PnLProxy: cumPnL,
    segmentedWave2ExecutionDegradationPct: degradationPct,
    segmentedWave2ExecutionFragilityVerdict: frag,
    segmentedWave2ExecutionReasons: reasons,
    segmentedWave2ExecutionRisks: risks,
    nextEscalationRule,
    segmentedWave2ExecutionSummaryLine: "",
    thresholdsUsed: {
      MOMENTUM_SEG_W2_EXEC_MIN_EVENTS: minSeg,
      MOMENTUM_SEG_W2_MAGNITUDE_MIN: magMin,
      friction_model: "same_as_realisticPaper_MOMENTUM_PAPER_*",
      preparation_verdict: preparation.segmentedPaperPreparationVerdict,
      context_bestOperationalRuleSet: ops.bestOperationalRuleSet?.ruleSetLabel ?? "n/a",
    },
    readDisclaimer:
      "Execução paper segmentada wave-2. wave-3 (liquidez 1k–10k) deferida até leitura estável desta camada.",
  };
  out.segmentedWave2ExecutionSummaryLine = buildSegmentedWave2ExecutionSummaryLine(out);
  return out;
}

export function buildSegmentedWave2ExecutionSummaryLine(
  a: SegmentedPaperExecutionWave2Assessment,
): string {
  const imp = a.segmentedWave2NetImprovementVsAll;
  const impS = imp !== null ? (imp > 0 ? "+" : "") + String(imp) : "n/a";
  const deg =
    a.segmentedWave2ExecutionDegradationPct !== null
      ? String(r4(a.segmentedWave2ExecutionDegradationPct * 100)) + "%"
      : "n/a";
  return `segW2: ${a.segmentedWave2ExecutionVerdict} | fills=${a.segmentedWave2SimulatedTradeCount} rate=${a.segmentedWave2SimulatedFillRate} netImpAll=${impS} deg=${deg} frag=${a.segmentedWave2ExecutionFragilityVerdict}`;
}
