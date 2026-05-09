/**
 * Realistic Paper Execution Assessment — degradation-measurement layer.
 * Estimates how much of the robust observational edge survives once
 * conservative execution frictions (latency, fill rate, slippage,
 * exit penalty) are applied. Still simulated; no live orders.
 */

import type { MomentumEvent, MomentumEventType } from "./momentumSnipingProbe";
import type { OperationalizationAssessment } from "./momentumOperationalization";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}
function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}
function avg(nums: number[]): number | null {
  if (nums.length === 0) return null;
  return r4(nums.reduce((a, b) => a + b, 0) / nums.length);
}

const MIN_EVENTS_FOR_PAPER = () =>
  Math.max(1, Math.floor(envNum("MOMENTUM_PAPER_MIN_EVENTS", 8)));

const LATENCY_MS = () => envNum("MOMENTUM_PAPER_LATENCY_MS", 2000);
const EDGE_DECAY_PER_SEC = () => envNum("MOMENTUM_PAPER_EDGE_DECAY_PER_SEC", 0.0008);
const BASE_FILL_RATE = () => envNum("MOMENTUM_PAPER_BASE_FILL_RATE", 0.75);
const EXIT_SLIPPAGE_RATE = () => envNum("MOMENTUM_PAPER_EXIT_SLIPPAGE", 0.15);
const TIMEOUT_PENALTY_RATE = () => envNum("MOMENTUM_PAPER_TIMEOUT_PENALTY", 0.25);
const FEE_PER_SIDE = () => envNum("MOMENTUM_PAPER_FEE_PER_SIDE", 0.005);

export type RealisticPaperVerdict =
  | "insufficient_sample"
  | "edge_destroyed_by_friction"
  | "marginal_after_friction"
  | "viable_after_friction"
  | "strong_after_friction";

export type ExecutionFragilityVerdict =
  | "robust_execution"
  | "moderate_fragility"
  | "high_fragility";

export interface SimulatedTrade {
  eventType: MomentumEventType;
  marketId: string;
  originalProxy: number;
  latencyDecay: number;
  filled: boolean;
  fillFailureReason: string | null;
  exitSlippage: number;
  exitReason: "clean" | "timeout" | "deterioration";
  netProxy: number;
  holdingWindowMs: number;
}

interface DegradationBucket {
  label: string;
  count: number;
  avgOriginalProxy: number | null;
  avgNetProxy: number | null;
  degradationPct: number | null;
}

export interface RealisticPaperExecutionAssessment {
  totalEventsConsidered: number;
  minimumRequired: number;
  hasEnoughSample: boolean;
  realisticPaperExecutionVerdict: RealisticPaperVerdict;
  simulatedTradeCount: number;
  simulatedFillRate: number;
  simulatedAverageSlippage: number | null;
  simulatedLatencyPenalty: number | null;
  simulatedExitPenalty: number | null;
  simulatedNetImprovementVsBaseline: number | null;
  simulatedPnLProxy: number;
  executionDegradationPct: number | null;
  degradationByExitReason: DegradationBucket[];
  degradationByMarket: DegradationBucket[];
  degradationByHoldingWindow: DegradationBucket[];
  fillFailureReasons: Array<{ reason: string; count: number; share: number }>;
  executionFragilityVerdict: ExecutionFragilityVerdict;
  supportingReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: Record<string, number>;
  readDisclaimer: string;
}

export function applyConservativeFilter(
  events: readonly MomentumEvent[],
  magFloor: number,
): MomentumEvent[] {
  return events.filter(
    e =>
      e.capturable &&
      e.magnitude >= magFloor &&
      !(e.magnitude < 0.003 || e.conservativeCaptureProxy <= -0.003),
  );
}

export function simulateTrade(e: MomentumEvent): SimulatedTrade {
  const latencyMs = LATENCY_MS();
  const decayPerSec = EDGE_DECAY_PER_SEC();
  const baseFill = BASE_FILL_RATE();
  const exitSlip = EXIT_SLIPPAGE_RATE();
  const timeoutPen = TIMEOUT_PENALTY_RATE();
  const fee = FEE_PER_SIDE();

  const latencyDecay = r4(decayPerSec * (latencyMs / 1000));

  const proxyAfterLatency = e.conservativeCaptureProxy - latencyDecay;

  const liqFactor = e.liquidityAtDetection >= 5000 ? 1.0
    : e.liquidityAtDetection >= 1000 ? 0.9
    : e.liquidityAtDetection >= 500 ? 0.75
    : 0.5;
  const fillProb = baseFill * liqFactor;

  const rng = simpleHash(e.marketId + e.detectedAt);
  const filled = rng < fillProb;

  let fillFailureReason: string | null = null;
  if (!filled) {
    if (liqFactor < 0.75) fillFailureReason = "low_liquidity";
    else if (proxyAfterLatency <= 0) fillFailureReason = "edge_decayed_before_fill";
    else fillFailureReason = "missed_fill";
  }

  let exitReason: "clean" | "timeout" | "deterioration";
  let exitPenalty: number;
  const exitRng = simpleHash(e.detectedAt + e.marketId);
  if (exitRng < 0.6) {
    exitReason = "clean";
    exitPenalty = r4(exitSlip * e.magnitude);
  } else if (exitRng < 0.85) {
    exitReason = "timeout";
    exitPenalty = r4(timeoutPen * e.magnitude);
  } else {
    exitReason = "deterioration";
    exitPenalty = r4(timeoutPen * e.magnitude * 1.5);
  }

  const totalFees = fee * 2;
  const netProxy = filled
    ? r4(proxyAfterLatency - exitPenalty - totalFees)
    : 0;

  return {
    eventType: e.eventType,
    marketId: e.marketId,
    originalProxy: e.conservativeCaptureProxy,
    latencyDecay,
    filled,
    fillFailureReason,
    exitSlippage: exitPenalty,
    exitReason,
    netProxy,
    holdingWindowMs: e.durationMs,
  };
}

function simpleHash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h % 1000) / 1000;
}

function buildDegradationByKey(
  trades: SimulatedTrade[],
  keyFn: (t: SimulatedTrade) => string,
): DegradationBucket[] {
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
      const netAvg = avg(ts.map(t => t.netProxy));
      const deg = origAvg !== null && origAvg > 0 && netAvg !== null
        ? r4(1 - netAvg / origAvg)
        : null;
      return { label, count: ts.length, avgOriginalProxy: origAvg, avgNetProxy: netAvg, degradationPct: deg };
    });
}

export function holdingWindowBucket(ms: number): string {
  if (ms < 5000) return "<5s";
  if (ms < 15000) return "5-15s";
  if (ms < 30000) return "15-30s";
  return ">30s";
}

export function buildRealisticPaperExecution(
  allEvents: readonly MomentumEvent[],
  ops: OperationalizationAssessment,
): RealisticPaperExecutionAssessment {
  const minEv = MIN_EVENTS_FOR_PAPER();
  const mags = allEvents.map(e => e.magnitude).sort((a, b) => a - b);
  const p25 = mags.length >= 4
    ? mags[Math.floor(mags.length * 0.25)]!
    : (mags[Math.floor(mags.length / 2)] ?? 0.005);
  const magFloor = r4(Math.max(p25, 0.005));

  const passing = applyConservativeFilter(allEvents, magFloor);
  const hasEnough = passing.length >= minEv;

  const trades = passing.map(simulateTrade);
  const filled = trades.filter(t => t.filled);
  const unfilled = trades.filter(t => !t.filled);

  const fillRate = trades.length > 0 ? r4(filled.length / trades.length) : 0;

  const avgSlippage = avg(filled.map(t => t.exitSlippage));
  const avgLatencyPen = avg(filled.map(t => t.latencyDecay));
  const avgExitPen = avg(filled.map(t => t.exitSlippage));

  const baselineAvg = avg(allEvents.map(e => e.conservativeCaptureProxy));
  const netAvg = avg(filled.map(t => t.netProxy));
  const netImprovement = baselineAvg !== null && netAvg !== null
    ? r4(netAvg - baselineAvg) : null;

  const cumPnL = r4(filled.reduce((s, t) => s + t.netProxy, 0));

  const origAvgPassing = avg(passing.map(e => e.conservativeCaptureProxy));
  const degradationPct = origAvgPassing !== null && origAvgPassing > 0 && netAvg !== null
    ? r4(1 - (netAvg / origAvgPassing))
    : null;

  const failReasons: Record<string, number> = {};
  for (const t of unfilled) {
    const r = t.fillFailureReason ?? "unknown";
    failReasons[r] = (failReasons[r] ?? 0) + 1;
  }
  const failReasonsArr = Object.entries(failReasons)
    .sort((a, b) => b[1] - a[1])
    .map(([reason, count]) => ({
      reason,
      count,
      share: trades.length > 0 ? r4(count / trades.length) : 0,
    }));

  const degByExit = buildDegradationByKey(filled, t => t.exitReason);
  const degByMarket = buildDegradationByKey(filled, t => t.marketId.slice(0, 20));
  const degByWindow = buildDegradationByKey(filled, t => holdingWindowBucket(t.holdingWindowMs));

  let fragilityVerdict: ExecutionFragilityVerdict;
  if (fillRate >= 0.6 && (degradationPct === null || degradationPct < 0.5)) {
    fragilityVerdict = "robust_execution";
  } else if (fillRate >= 0.4 && (degradationPct === null || degradationPct < 0.8)) {
    fragilityVerdict = "moderate_fragility";
  } else {
    fragilityVerdict = "high_fragility";
  }

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];
  let verdict: RealisticPaperVerdict;

  if (!hasEnough) {
    verdict = "insufficient_sample";
    blockingReasons.push(`Passing events ${passing.length} < mínimo ${minEv}.`);
  } else if (netImprovement !== null && netImprovement > 0.002 && fillRate >= 0.5) {
    verdict = "strong_after_friction";
    supportingReasons.push(
      `Net improvement +${netImprovement} após fricções; fillRate=${fillRate}; degradation=${degradationPct !== null ? r4(degradationPct * 100) + "%" : "n/a"}.`,
    );
  } else if (netImprovement !== null && netImprovement > 0 && fillRate >= 0.4) {
    verdict = "viable_after_friction";
    supportingReasons.push(
      `Net improvement +${netImprovement} positivo após fricções; fillRate=${fillRate}.`,
    );
  } else if (netAvg !== null && netAvg > 0) {
    verdict = "marginal_after_friction";
    supportingReasons.push(
      `Net proxy médio positivo (${netAvg}) mas improvement vs baseline fraco ou fillRate baixo (${fillRate}).`,
    );
  } else {
    verdict = "edge_destroyed_by_friction";
    blockingReasons.push(
      `Edge observacional destruído pelas fricções de execução. netAvg=${netAvg ?? "n/a"}, fillRate=${fillRate}.`,
    );
  }

  return {
    totalEventsConsidered: passing.length,
    minimumRequired: minEv,
    hasEnoughSample: hasEnough,
    realisticPaperExecutionVerdict: verdict,
    simulatedTradeCount: filled.length,
    simulatedFillRate: fillRate,
    simulatedAverageSlippage: avgSlippage,
    simulatedLatencyPenalty: avgLatencyPen,
    simulatedExitPenalty: avgExitPen,
    simulatedNetImprovementVsBaseline: netImprovement,
    simulatedPnLProxy: cumPnL,
    executionDegradationPct: degradationPct,
    degradationByExitReason: degByExit,
    degradationByMarket: degByMarket,
    degradationByHoldingWindow: degByWindow,
    fillFailureReasons: failReasonsArr,
    executionFragilityVerdict: fragilityVerdict,
    supportingReasons,
    blockingReasons,
    thresholdsUsed: {
      MOMENTUM_PAPER_MIN_EVENTS: minEv,
      MOMENTUM_PAPER_LATENCY_MS: LATENCY_MS(),
      MOMENTUM_PAPER_EDGE_DECAY_PER_SEC: EDGE_DECAY_PER_SEC(),
      MOMENTUM_PAPER_BASE_FILL_RATE: BASE_FILL_RATE(),
      MOMENTUM_PAPER_EXIT_SLIPPAGE: EXIT_SLIPPAGE_RATE(),
      MOMENTUM_PAPER_TIMEOUT_PENALTY: TIMEOUT_PENALTY_RATE(),
      MOMENTUM_PAPER_FEE_PER_SIDE: FEE_PER_SIDE(),
    },
    readDisclaimer:
      "Simulação paper com fricções conservadoras (latência, fill rate, slippage, fees). Não é PnL real. Resultado positivo ≠ lucro garantido; resultado negativo indica que fricções excedem edge observacional.",
  };
}

export function buildRealisticPaperSummaryLine(
  a: RealisticPaperExecutionAssessment,
): string {
  if (!a.hasEnoughSample) {
    return `paper: insufficient_sample (${a.totalEventsConsidered}/${a.minimumRequired})`;
  }
  const netImp = a.simulatedNetImprovementVsBaseline;
  const impS = netImp !== null ? (netImp > 0 ? "+" : "") + String(netImp) : "n/a";
  const deg = a.executionDegradationPct !== null
    ? String(r4(a.executionDegradationPct * 100)) + "%"
    : "n/a";
  return `paper: ${a.realisticPaperExecutionVerdict} | trades=${a.simulatedTradeCount} fill=${a.simulatedFillRate} netImp=${impS} deg=${deg} fragility=${a.executionFragilityVerdict}`;
}
