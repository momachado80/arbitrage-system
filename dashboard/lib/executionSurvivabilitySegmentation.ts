/**
 * Execution Survivability Segmentation — diagnosis layer.
 * Segments passing events of the winning conservative rule into stronger vs weaker
 * subsets under realistic paper frictions. Does not invent new alpha or replace the rule.
 *
 * Nota operacional: interrupções longas (ex.: falta de energia) não invalidam o histórico
 * acumulado; leituras que assumem continuidade fina entre snapshots devem tratar o pós-restart
 * como continuação após gap conhecido, não série perfeitamente contínua.
 */

import type { MomentumEvent } from "./momentumSnipingProbe";
import type { OperationalizationAssessment } from "./momentumOperationalization";
import type { OperationalizationRobustnessAssessment } from "./operationalizationRobustness";
import type { PromotionReadinessAssessment } from "./operationalizationPromotionReadiness";
import type { PromotionProgressAssessment } from "./promotionProgressTracker";
import type { RealisticPaperExecutionAssessment } from "./realisticPaperExecutionAssessment";
import {
  applyConservativeFilter,
  simulateTrade,
  holdingWindowBucket,
  type SimulatedTrade,
} from "./realisticPaperExecutionAssessment";

function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}
function avg(nums: number[]): number | null {
  if (nums.length === 0) return null;
  return r4(nums.reduce((a, b) => a + b, 0) / nums.length);
}
function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}

const MIN_SEGMENT_SIZE = () =>
  Math.max(2, Math.floor(envNum("MOMENTUM_SEG_MIN_SIZE", 3)));
const MIN_EVENTS_FOR_SEGMENTATION = () =>
  Math.max(3, Math.floor(envNum("MOMENTUM_SEG_MIN_EVENTS", 6)));

export type ExecutionSurvivabilityVerdict =
  | "insufficient_sample"
  | "no_viable_subset_found"
  | "weak_candidate_subset_found"
  | "promising_candidate_subset_found";

interface SegmentProfile {
  segmentId: string;
  axis: string;
  label: string;
  eventCount: number;
  filledCount: number;
  fillRate: number;
  avgOriginalProxy: number | null;
  avgNetProxy: number | null;
  cumNetProxy: number;
  degradationPct: number | null;
  distinctMarkets: number;
  survivable: boolean;
}

interface SegmentCandidate {
  segmentId: string;
  axis: string;
  label: string;
  eventCount: number;
  distinctMarkets: number;
  avgNetProxy: number | null;
  cumNetProxy: number;
  coveragePct: number;
  netImprovementVsBaseline: number | null;
  risk: string;
}

export interface ExecutionSurvivabilitySegmentation {
  totalPassingEvents: number;
  minimumRequired: number;
  hasEnoughSample: boolean;
  executionSurvivabilityVerdict: ExecutionSurvivabilityVerdict;
  survivableSubsetCount: number;
  survivableSubsetCoverage: number;
  survivableSubsetDistinctMarkets: number;
  bestMagnitudeBucket: SegmentProfile | null;
  bestMarketSubset: SegmentProfile | null;
  bestExitReasonProfile: SegmentProfile | null;
  bestHoldingWindowProfile: SegmentProfile | null;
  degradationByMagnitudeBucket: SegmentProfile[];
  degradationByMarketSubset: SegmentProfile[];
  degradationByExitProfile: SegmentProfile[];
  degradationByHoldingWindowProfile: SegmentProfile[];
  netImprovementBySegment: Array<{ segmentId: string; netImprovement: number | null }>;
  pnlProxyBySegment: Array<{ segmentId: string; cumPnlProxy: number }>;
  segmentSelectionCandidates: SegmentCandidate[];
  segmentSelectionRisks: string[];
  supportingReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: Record<string, number | string>;
  readDisclaimer: string;
  executionSurvivabilitySummaryLine: string;
}

function magnitudeBucket(mag: number): string {
  if (mag < 0.01) return "mag_<0.01";
  if (mag < 0.03) return "mag_0.01-0.03";
  if (mag < 0.06) return "mag_0.03-0.06";
  return "mag_>=0.06";
}

interface EventTradePair {
  event: MomentumEvent;
  trade: SimulatedTrade;
}

function segmentSurvivable(
  axis: string,
  pairCount: number,
  avgNet: number | null,
  distinctMarkets: number,
  minSize: number,
): boolean {
  if (pairCount < minSize || avgNet === null || avgNet <= 0) return false;
  if (axis === "market") {
    return true;
  }
  if (distinctMarkets >= 2) return true;
  return pairCount >= Math.max(minSize, 6);
}

function buildSegmentProfiles(
  pairs: EventTradePair[],
  axis: string,
  keyFn: (p: EventTradePair) => string,
  minSize: number,
): SegmentProfile[] {
  const groups: Record<string, EventTradePair[]> = {};
  for (const p of pairs) {
    const k = keyFn(p);
    (groups[k] ??= []).push(p);
  }

  return Object.entries(groups)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 12)
    .map(([label, ps]) => {
      const filled = ps.filter(p => p.trade.filled);
      const avgOrig = avg(ps.map(p => p.trade.originalProxy));
      const avgNet = avg(filled.map(p => p.trade.netProxy));
      const cumNet = r4(filled.reduce((s, p) => s + p.trade.netProxy, 0));
      const deg =
        avgOrig !== null && avgOrig > 0 && avgNet !== null
          ? r4(1 - avgNet / avgOrig)
          : null;
      const markets = new Set(ps.map(p => p.event.marketId));
      const survivable = segmentSurvivable(axis, ps.length, avgNet, markets.size, minSize);

      return {
        segmentId: `${axis}:${label}`,
        axis,
        label,
        eventCount: ps.length,
        filledCount: filled.length,
        fillRate: ps.length > 0 ? r4(filled.length / ps.length) : 0,
        avgOriginalProxy: avgOrig,
        avgNetProxy: avgNet,
        cumNetProxy: cumNet,
        degradationPct: deg,
        distinctMarkets: markets.size,
        survivable,
      };
    });
}

function pickBest(profiles: SegmentProfile[], minSize: number): SegmentProfile | null {
  const viable = profiles.filter(
    p => p.eventCount >= minSize && p.avgNetProxy !== null && p.avgNetProxy > 0,
  );
  if (viable.length === 0) return null;
  viable.sort((a, b) => (b.avgNetProxy ?? 0) - (a.avgNetProxy ?? 0));
  return viable[0]!;
}

function buildCombinedProfile(
  pairs: EventTradePair[],
  magLabel: string | undefined,
  minSize: number,
): SegmentProfile | null {
  if (!magLabel) return null;
  const ps = pairs.filter(
    p =>
      magnitudeBucket(p.event.magnitude) === magLabel && p.trade.exitReason === "clean",
  );
  if (ps.length < minSize) return null;
  const filled = ps.filter(p => p.trade.filled);
  const avgOrig = avg(ps.map(p => p.trade.originalProxy));
  const avgNet = avg(filled.map(p => p.trade.netProxy));
  if (avgNet === null || avgNet <= 0) return null;
  const cumNet = r4(filled.reduce((s, p) => s + p.trade.netProxy, 0));
  const deg =
    avgOrig !== null && avgOrig > 0 && avgNet !== null
      ? r4(1 - avgNet / avgOrig)
      : null;
  const markets = new Set(ps.map(p => p.event.marketId));
  const survivable = segmentSurvivable("combined", ps.length, avgNet, markets.size, minSize);

  return {
    segmentId: `combined:${magLabel}+exit_clean`,
    axis: "combined",
    label: `${magLabel}+exit_clean`,
    eventCount: ps.length,
    filledCount: filled.length,
    fillRate: ps.length > 0 ? r4(filled.length / ps.length) : 0,
    avgOriginalProxy: avgOrig,
    avgNetProxy: avgNet,
    cumNetProxy: cumNet,
    degradationPct: deg,
    distinctMarkets: markets.size,
    survivable,
  };
}

function dedupeProfiles(all: SegmentProfile[]): SegmentProfile[] {
  const seen = new Set<string>();
  const out: SegmentProfile[] = [];
  for (const p of all) {
    if (seen.has(p.segmentId)) continue;
    seen.add(p.segmentId);
    out.push(p);
  }
  return out;
}

export function buildExecutionSurvivabilitySegmentation(
  allEvents: readonly MomentumEvent[],
  ops: OperationalizationAssessment,
  robustness: OperationalizationRobustnessAssessment,
  promo: PromotionReadinessAssessment,
  progress: PromotionProgressAssessment,
  paperExec: RealisticPaperExecutionAssessment,
): ExecutionSurvivabilitySegmentation {
  const minEv = MIN_EVENTS_FOR_SEGMENTATION();
  const minSeg = MIN_SEGMENT_SIZE();

  const mags = allEvents.map(e => e.magnitude).sort((a, b) => a - b);
  const p25 =
    mags.length >= 4
      ? mags[Math.floor(mags.length * 0.25)]!
      : (mags[Math.floor(mags.length / 2)] ?? 0.005);
  const magFloor = r4(Math.max(p25, 0.005));

  const passing = applyConservativeFilter(allEvents, magFloor);
  const hasEnough = passing.length >= minEv;

  const pairs: EventTradePair[] = passing.map(e => ({
    event: e,
    trade: simulateTrade(e),
  }));

  const baselineAvg = avg(allEvents.map(e => e.conservativeCaptureProxy));

  const byMag = buildSegmentProfiles(pairs, "magnitude", p => magnitudeBucket(p.event.magnitude), minSeg);
  const byMarket = buildSegmentProfiles(pairs, "market", p => p.event.marketId.slice(0, 20), minSeg);
  const byExit = buildSegmentProfiles(pairs, "exit_reason", p => p.trade.exitReason, minSeg);
  const byWindow = buildSegmentProfiles(
    pairs,
    "holding_window",
    p => holdingWindowBucket(p.event.durationMs),
    minSeg,
  );

  const bestMag = pickBest(byMag, minSeg);
  const bestMarket = pickBest(byMarket, minSeg);
  const bestExit = pickBest(byExit, minSeg);
  const bestWindow = pickBest(byWindow, minSeg);

  const combined = buildCombinedProfile(pairs, bestMag?.label, minSeg);
  const allProfiles = dedupeProfiles([
    ...byMag,
    ...byMarket,
    ...byExit,
    ...byWindow,
    ...(combined ? [combined] : []),
  ]);

  const survivableProfiles = allProfiles.filter(p => p.survivable);

  const survivableEvents = new Set<string>();
  const survivableMarkets = new Set<string>();
  for (const prof of survivableProfiles) {
    for (const p of pairs) {
      let match = false;
      if (prof.axis === "magnitude") {
        match = magnitudeBucket(p.event.magnitude) === prof.label;
      } else if (prof.axis === "market") {
        match = p.event.marketId.slice(0, 20) === prof.label;
      } else if (prof.axis === "exit_reason") {
        match = p.trade.exitReason === prof.label;
      } else if (prof.axis === "holding_window") {
        match = holdingWindowBucket(p.event.durationMs) === prof.label;
      } else if (prof.axis === "combined") {
        match =
          bestMag != null &&
          magnitudeBucket(p.event.magnitude) === bestMag.label &&
          p.trade.exitReason === "clean";
      }
      if (match) {
        survivableEvents.add(p.event.detectedAt + p.event.marketId);
        survivableMarkets.add(p.event.marketId);
      }
    }
  }

  const survivableCount = survivableEvents.size;
  const survivableCoverage = passing.length > 0 ? r4(survivableCount / passing.length) : 0;
  const survivableDistinctMkts = survivableMarkets.size;

  const sortedForImp = [...allProfiles].sort((a, b) => b.eventCount - a.eventCount);
  const netImprovementBySegment = sortedForImp.slice(0, 20).map(p => ({
    segmentId: p.segmentId,
    netImprovement:
      p.avgNetProxy !== null && baselineAvg !== null ? r4(p.avgNetProxy - baselineAvg) : null,
  }));
  const pnlProxyBySegment = sortedForImp.slice(0, 20).map(p => ({
    segmentId: p.segmentId,
    cumPnlProxy: p.cumNetProxy,
  }));

  const candidates: SegmentCandidate[] = [];
  const risks: string[] = [];

  for (const prof of survivableProfiles) {
    const coveragePct = passing.length > 0 ? r4(prof.eventCount / passing.length) : 0;
    const netImp =
      prof.avgNetProxy !== null && baselineAvg !== null
        ? r4(prof.avgNetProxy - baselineAvg)
        : null;

    let risk = "none";
    if (prof.eventCount < 4) risk = "very_small_sample";
    else if (prof.axis === "market") risk = "single_market_concentration";
    else if (prof.distinctMarkets < 2 && prof.eventCount < 6) risk = "low_market_diversity";
    else if (coveragePct < 0.12) risk = "narrow_coverage";

    candidates.push({
      segmentId: prof.segmentId,
      axis: prof.axis,
      label: prof.label,
      eventCount: prof.eventCount,
      distinctMarkets: prof.distinctMarkets,
      avgNetProxy: prof.avgNetProxy,
      cumNetProxy: prof.cumNetProxy,
      coveragePct,
      netImprovementVsBaseline: netImp,
      risk,
    });
  }

  candidates.sort((a, b) => (b.avgNetProxy ?? 0) - (a.avgNetProxy ?? 0));

  if (robustness.overfitRiskVerdict === "high") {
    risks.push("Robustez operacional reportou overfit alto; segmentos são leitura frágil.");
  }
  if (paperExec.realisticPaperExecutionVerdict === "edge_destroyed_by_friction") {
    risks.push(
      "Paper realista global: edge destruído por fricção; subconjuntos positivos podem ser ruído ou concentração.",
    );
  }
  if (promo.distinctPassingMarkets < 3) {
    risks.push(
      `Amplitude global baixa (distinctPassingMarkets=${promo.distinctPassingMarkets}); cautela ao estreitar.`,
    );
  }
  if (survivableDistinctMkts < 2 && survivableCount > 0) {
    risks.push("Subconjuntos survivable concentrados em poucos mercados.");
  }
  if (survivableCount > 0 && survivableCoverage < 0.15) {
    risks.push("Coverage do subconjunto survivable < 15% do total passing.");
  }
  const tinySegments = candidates.filter(c => c.eventCount < 4);
  if (tinySegments.length > candidates.length / 2 && candidates.length > 0) {
    risks.push("Maioria dos candidatos survivable têm amostra muito pequena.");
  }

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];
  let verdict: ExecutionSurvivabilityVerdict;

  const bestRuleLabel = ops.bestOperationalRuleSet?.ruleSetLabel ?? "n/a";

  if (!paperExec.hasEnoughSample) {
    verdict = "insufficient_sample";
    blockingReasons.push(
      `Paper execution: amostra insuficiente (${paperExec.totalEventsConsidered}/${paperExec.minimumRequired}).`,
    );
  } else if (!hasEnough) {
    verdict = "insufficient_sample";
    blockingReasons.push(`Passing events ${passing.length} < mínimo ${minEv}.`);
  } else if (candidates.length === 0) {
    verdict = "no_viable_subset_found";
    blockingReasons.push(
      "Nenhum segmento com net proxy médio positivo, tamanho mínimo e critérios de survivability.",
    );
  } else {
    const breadthOk = promo.distinctPassingMarkets >= 3 || progress.progressToRobustPct >= 0.85;
    const strongCandidates = candidates.filter(c => {
      const netOk = (c.avgNetProxy ?? 0) > 0.001;
      const sizeOk = c.eventCount >= 4;
      const covOk = c.coveragePct >= 0.1;
      const mktOk =
        c.axis === "market"
          ? true
          : c.distinctMarkets >= 2 || c.eventCount >= 8;
      return netOk && sizeOk && covOk && mktOk && breadthOk;
    });

    if (strongCandidates.length > 0) {
      verdict = "promising_candidate_subset_found";
      const best = strongCandidates[0]!;
      supportingReasons.push(
        `Segmento ${best.segmentId}: ${best.eventCount} eventos, ${best.distinctMarkets} mercados no bucket, avgNet=${best.avgNetProxy}, coverage=${r4(best.coveragePct * 100)}%. Rule=${bestRuleLabel}.`,
      );
    } else {
      verdict = "weak_candidate_subset_found";
      const best = candidates[0]!;
      supportingReasons.push(
        `Melhor candidato ${best.segmentId}: ${best.eventCount} eventos, avgNet=${best.avgNetProxy}. Risco: ${best.risk}.`,
      );
      if (!breadthOk) {
        blockingReasons.push(
          "Amplitude global ou progresso a robust ainda não sustentam subset 'promising' sem cautela extra.",
        );
      }
    }
  }

  const thresholdsUsed: Record<string, number | string> = {
    MOMENTUM_SEG_MIN_SIZE: minSeg,
    MOMENTUM_SEG_MIN_EVENTS: minEv,
    context_bestOperationalRuleSet: bestRuleLabel,
    context_promo_passingEvents: promo.passingEventCount,
    context_promo_distinctMarkets: promo.distinctPassingMarkets,
    context_paper_verdict: paperExec.realisticPaperExecutionVerdict,
    context_progressToRobustPct: r4(progress.progressToRobustPct),
    context_overfitRisk: robustness.overfitRiskVerdict,
  };

  const readDisclaimer =
    "Segmentação diagnóstica sobre eventos passing da regra conservadora com as mesmas fricções paper do módulo realisticPaper. Subconjuntos com net positivo não substituem validação out-of-sample. Interrupções operacionais longas não apagam histórico, mas leituras dependentes de continuidade temporal fina devem considerar o pós-restart como continuação após gap conhecido.";

  const base: ExecutionSurvivabilitySegmentation = {
    totalPassingEvents: passing.length,
    minimumRequired: minEv,
    hasEnoughSample: hasEnough && paperExec.hasEnoughSample,
    executionSurvivabilityVerdict: verdict,
    survivableSubsetCount: survivableCount,
    survivableSubsetCoverage: survivableCoverage,
    survivableSubsetDistinctMarkets: survivableDistinctMkts,
    bestMagnitudeBucket: bestMag,
    bestMarketSubset: bestMarket,
    bestExitReasonProfile: bestExit,
    bestHoldingWindowProfile: bestWindow,
    degradationByMagnitudeBucket: byMag,
    degradationByMarketSubset: byMarket,
    degradationByExitProfile: byExit,
    degradationByHoldingWindowProfile: byWindow,
    netImprovementBySegment,
    pnlProxyBySegment,
    segmentSelectionCandidates: candidates,
    segmentSelectionRisks: risks,
    supportingReasons,
    blockingReasons,
    thresholdsUsed,
    readDisclaimer,
    executionSurvivabilitySummaryLine: "",
  };

  base.executionSurvivabilitySummaryLine = buildSurvivabilitySummaryLine(base);

  return base;
}

export function buildSurvivabilitySummaryLine(
  a: ExecutionSurvivabilitySegmentation,
): string {
  if (!a.hasEnoughSample) {
    return `survivability: insufficient_sample (${a.totalPassingEvents}/${a.minimumRequired})`;
  }
  const best = a.segmentSelectionCandidates[0];
  const bestLabel = best ? best.segmentId : "none";
  const bestNet = best?.avgNetProxy ?? "n/a";
  return `survivability: ${a.executionSurvivabilityVerdict} | subsets=${a.survivableSubsetCount} coverage=${r4(a.survivableSubsetCoverage * 100)}% mkts=${a.survivableSubsetDistinctMarkets} best=${bestLabel}(net=${bestNet})`;
}
