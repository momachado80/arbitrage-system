/**
 * Momentum / Sniping Probe — observacional.
 * Detecta microdesalinhamentos temporários (spread spikes, gap entre legs, spread
 * recovery, quote velocity) usando snapshots sequenciais do marketDataService.
 * Não executa trades nem altera probes estruturais.
 */

import { getAllMarkets } from "./marketDataService";
import type { NormalizedMarket } from "./polymarketClient";
import {
  buildRankedEventAssessment,
  type RankedEventAssessment,
} from "./momentumRankedEventAssessment";
import {
  buildTopSliceRobustnessAssessment,
  type TopSliceRobustnessAssessment,
} from "./momentumTopSliceRobustness";
import {
  buildTopSliceSelectionAssessment,
  type TopSliceSelectionAssessment,
} from "./momentumTopSliceSelection";
import {
  buildOperationalizationAssessment,
  type OperationalizationAssessment,
} from "./momentumOperationalization";
import {
  buildOperationalizationRobustness,
  type OperationalizationRobustnessAssessment,
} from "./operationalizationRobustness";
import {
  buildPromotionReadiness,
  type PromotionReadinessAssessment,
} from "./operationalizationPromotionReadiness";
import {
  buildPromotionProgress,
  type PromotionProgressAssessment,
} from "./promotionProgressTracker";
import {
  buildRealisticPaperExecution,
  type RealisticPaperExecutionAssessment,
} from "./realisticPaperExecutionAssessment";
import {
  buildExecutionSurvivabilitySegmentation,
  type ExecutionSurvivabilitySegmentation,
} from "./executionSurvivabilitySegmentation";
import {
  buildSegmentedPaperTestPreparation,
  type SegmentedPaperTestPreparation,
} from "./segmentedPaperTestPreparation";
import {
  buildSegmentedPaperExecutionAssessment,
  type SegmentedPaperExecutionAssessment,
} from "./segmentedPaperExecutionAssessment";
import {
  buildSegmentedPaperExecutionWave2Assessment,
  type SegmentedPaperExecutionWave2Assessment,
} from "./segmentedPaperExecutionWave2Assessment";
import {
  buildOperationalizationAssessmentV2,
  type OperationalizationAssessmentV2,
} from "./momentumOperationalizationV2";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}
function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}
function median(nums: number[]): number | null {
  if (nums.length === 0) return null;
  const s = [...nums].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : r4((s[m - 1]! + s[m]!) / 2);
}

const GLOBAL_KEY = "__momentumSnipingProbe_v1";

const SCAN_INTERVAL_MS = () => envNum("MOMENTUM_SCAN_INTERVAL_MS", 10_000);
const BOOT_DELAY_MS = () => envNum("MOMENTUM_BOOT_DELAY_MS", 30_000);
const MAX_EVENTS = () => envNum("MOMENTUM_MAX_EVENTS", 200);
const SPREAD_SPIKE_THRESHOLD = () => envNum("MOMENTUM_SPREAD_SPIKE_THRESHOLD", 0.06);
const SPREAD_RECOVERY_THRESHOLD = () => envNum("MOMENTUM_SPREAD_RECOVERY_THRESHOLD", 0.03);
const PRICE_VELOCITY_THRESHOLD = () => envNum("MOMENTUM_PRICE_VELOCITY_THRESHOLD", 0.03);
const MIN_LIQUIDITY = () => envNum("MOMENTUM_MIN_LIQUIDITY", 200);
const FEE_PROXY = () => envNum("MOMENTUM_FEE_PROXY", 0.005);
const MIN_EVENTS_FOR_READ = () => Math.max(1, Math.floor(envNum("MOMENTUM_MIN_EVENTS_FOR_READ", 8)));

export type MomentumEventType =
  | "spread_spike"
  | "spread_recovery"
  | "price_velocity_spike"
  | "leg_gap_anomaly";

export interface MomentumEvent {
  eventType: MomentumEventType;
  detectedAt: string;
  marketId: string;
  marketQuestion: string;
  durationMs: number;
  magnitude: number;
  spreadBefore: number;
  spreadAfter: number;
  pricesBefore: number[];
  pricesAfter: number[];
  liquidityAtDetection: number;
  conservativeCaptureProxy: number;
  capturable: boolean;
}

export type MomentumSnipingVerdict =
  | "insufficient_sample"
  | "mostly_noise"
  | "weak_executable_signal"
  | "promising_microstructure_signal"
  | "unstable_or_negative";

export interface MomentumSnipingDigest {
  computedAt: string;
  probeVersion: "momentum-sniping-v1";
  readNature: "observational_microstructure_probe";
  readDisclaimer: string;
  scannerRunning: boolean;
  snapshotsTaken: number;
  candidateMomentumEventsCount: number;
  candidateSnipingEventsCount: number;
  averageEventDurationMs: number | null;
  medianEventDurationMs: number | null;
  averageObservedDislocation: number | null;
  maxObservedDislocation: number | null;
  averageConservativeCapturableProxy: number | null;
  cumulativeConservativeCapturableProxy: number;
  eventFrequencyPerHour: number | null;
  repeatedPositiveEventPattern: boolean;
  repeatedNegativeEventPattern: boolean;
  concentrationByMarketOrBucket: Record<string, { count: number; share: number }>;
  enoughSampleForRead: boolean;
  momentumSnipingVerdict: MomentumSnipingVerdict;
  supportingReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: Record<string, number>;
  rankedEventAssessment: RankedEventAssessment;
  topSliceRobustnessAssessment: TopSliceRobustnessAssessment;
  topSliceSelectionAssessment: TopSliceSelectionAssessment;
  operationalizationAssessmentV2: OperationalizationAssessmentV2;
  operationalizationAssessment: OperationalizationAssessment;
  operationalizationRobustnessAssessment: OperationalizationRobustnessAssessment;
  promotionReadinessAssessment: PromotionReadinessAssessment;
  promotionProgressAssessment: PromotionProgressAssessment;
  realisticPaperExecutionAssessment: RealisticPaperExecutionAssessment;
  executionSurvivabilitySegmentation: ExecutionSurvivabilitySegmentation;
  segmentedPaperTestPreparation: SegmentedPaperTestPreparation;
  segmentedPaperExecutionAssessment: SegmentedPaperExecutionAssessment;
  segmentedPaperExecutionWave2Assessment: SegmentedPaperExecutionWave2Assessment;
  recentEvents: MomentumEvent[];
}

interface MarketSnapshot {
  id: string;
  question: string;
  spread: number;
  prices: number[];
  liquidity: number;
  ts: number;
}

interface ProbeState {
  loopStarted: boolean;
  scheduledTimeoutId: ReturnType<typeof setTimeout> | null;
  snapshotsTaken: number;
  previousSnap: Map<string, MarketSnapshot>;
  events: MomentumEvent[];
  firstScanAt: number | null;
  lastScanAt: number | null;
}

function getState(): ProbeState {
  const g = globalThis as unknown as Record<string, ProbeState | undefined>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = {
      loopStarted: false,
      scheduledTimeoutId: null,
      snapshotsTaken: 0,
      previousSnap: new Map(),
      events: [],
      firstScanAt: null,
      lastScanAt: null,
    };
  }
  return g[GLOBAL_KEY]!;
}

function takeSnapshot(): Map<string, MarketSnapshot> {
  const markets = getAllMarkets();
  const snap = new Map<string, MarketSnapshot>();
  const minLiq = MIN_LIQUIDITY();
  const now = Date.now();
  for (const m of markets) {
    if (m.closed || !m.active) continue;
    if (m.liquidity < minLiq) continue;
    if (m.prices.length < 2) continue;
    snap.set(m.id, {
      id: m.id,
      question: m.question.slice(0, 120),
      spread: m.spread,
      prices: m.prices.map(p => r4(p)),
      liquidity: m.liquidity,
      ts: now,
    });
  }
  return snap;
}

function detectEvents(prev: Map<string, MarketSnapshot>, curr: Map<string, MarketSnapshot>): MomentumEvent[] {
  const events: MomentumEvent[] = [];
  const spikeT = SPREAD_SPIKE_THRESHOLD();
  const recoveryT = SPREAD_RECOVERY_THRESHOLD();
  const velT = PRICE_VELOCITY_THRESHOLD();
  const fee = FEE_PROXY();
  const now = new Date().toISOString();

  for (const [id, c] of Array.from(curr.entries())) {
    const p = prev.get(id);
    if (!p) continue;
    const dt = c.ts - p.ts;
    if (dt <= 0) continue;

    const spreadDelta = c.spread - p.spread;

    if (spreadDelta >= spikeT) {
      const mag = r4(spreadDelta);
      const captureProxy = r4(mag * 0.35 - fee);
      events.push({
        eventType: "spread_spike",
        detectedAt: now,
        marketId: id,
        marketQuestion: c.question,
        durationMs: dt,
        magnitude: mag,
        spreadBefore: r4(p.spread),
        spreadAfter: r4(c.spread),
        pricesBefore: p.prices,
        pricesAfter: c.prices,
        liquidityAtDetection: c.liquidity,
        conservativeCaptureProxy: captureProxy,
        capturable: captureProxy > 0,
      });
    }

    if (spreadDelta <= -recoveryT && p.spread >= spikeT) {
      const mag = r4(Math.abs(spreadDelta));
      const captureProxy = r4(mag * 0.3 - fee);
      events.push({
        eventType: "spread_recovery",
        detectedAt: now,
        marketId: id,
        marketQuestion: c.question,
        durationMs: dt,
        magnitude: mag,
        spreadBefore: r4(p.spread),
        spreadAfter: r4(c.spread),
        pricesBefore: p.prices,
        pricesAfter: c.prices,
        liquidityAtDetection: c.liquidity,
        conservativeCaptureProxy: captureProxy,
        capturable: captureProxy > 0,
      });
    }

    if (p.prices.length === c.prices.length) {
      let maxPriceDelta = 0;
      for (let i = 0; i < p.prices.length; i++) {
        maxPriceDelta = Math.max(maxPriceDelta, Math.abs(c.prices[i]! - p.prices[i]!));
      }
      if (maxPriceDelta >= velT) {
        const mag = r4(maxPriceDelta);
        const captureProxy = r4(mag * 0.25 - fee);
        events.push({
          eventType: "price_velocity_spike",
          detectedAt: now,
          marketId: id,
          marketQuestion: c.question,
          durationMs: dt,
          magnitude: mag,
          spreadBefore: r4(p.spread),
          spreadAfter: r4(c.spread),
          pricesBefore: p.prices,
          pricesAfter: c.prices,
          liquidityAtDetection: c.liquidity,
          conservativeCaptureProxy: captureProxy,
          capturable: captureProxy > 0,
        });
      }

      if (c.prices.length >= 2) {
        const probSum = c.prices.reduce((s, x) => s + x, 0);
        const gap = Math.abs(probSum - 1.0);
        const prevProbSum = p.prices.reduce((s, x) => s + x, 0);
        const prevGap = Math.abs(prevProbSum - 1.0);
        if (gap >= 0.04 && gap > prevGap + 0.015) {
          const mag = r4(gap);
          const captureProxy = r4(gap * 0.4 - fee);
          events.push({
            eventType: "leg_gap_anomaly",
            detectedAt: now,
            marketId: id,
            marketQuestion: c.question,
            durationMs: dt,
            magnitude: mag,
            spreadBefore: r4(p.spread),
            spreadAfter: r4(c.spread),
            pricesBefore: p.prices,
            pricesAfter: c.prices,
            liquidityAtDetection: c.liquidity,
            conservativeCaptureProxy: captureProxy,
            capturable: captureProxy > 0,
          });
        }
      }
    }
  }
  return events;
}

function runScan(): void {
  const st = getState();
  const curr = takeSnapshot();
  st.snapshotsTaken++;
  if (!st.firstScanAt) st.firstScanAt = Date.now();
  st.lastScanAt = Date.now();

  const prevSize = st.previousSnap.size;
  if (prevSize > 0) {
    const newEvents = detectEvents(st.previousSnap, curr);
    for (const e of newEvents) {
      st.events.push(e);
    }
    const max = MAX_EVENTS();
    if (st.events.length > max) {
      st.events = st.events.slice(-max);
    }
    if (st.snapshotsTaken <= 5 || newEvents.length > 0) {
      console.log(
        `[MomentumSniping] scan #${st.snapshotsTaken} markets=${curr.size} prev=${prevSize} newEvents=${newEvents.length} totalEvents=${st.events.length}`,
      );
    }
  } else {
    console.log(`[MomentumSniping] scan #${st.snapshotsTaken} baseline markets=${curr.size}`);
  }
  st.previousSnap = curr;
}

function scheduleNext(delayMs: number): void {
  const st = getState();
  if (st.scheduledTimeoutId !== null) {
    clearTimeout(st.scheduledTimeoutId);
    st.scheduledTimeoutId = null;
  }
  st.scheduledTimeoutId = setTimeout(() => {
    st.scheduledTimeoutId = null;
    try { runScan(); } catch (e) {
      console.error("[MomentumSniping] scan error", e instanceof Error ? e.message : e);
    }
    scheduleNext(SCAN_INTERVAL_MS());
  }, delayMs);
}

export function ensureMomentumSnipingProbe(): void {
  const st = getState();
  if (st.loopStarted) return;
  st.loopStarted = true;
  console.log("[MomentumSniping] Scheduler started");
  scheduleNext(BOOT_DELAY_MS());
}

export function buildMomentumSnipingDigest(): MomentumSnipingDigest {
  const st = getState();
  const events = st.events;
  const fee = FEE_PROXY();
  const minEv = MIN_EVENTS_FOR_READ();

  const momentumEvents = events.filter(e => e.eventType === "spread_spike" || e.eventType === "price_velocity_spike");
  const snipingEvents = events.filter(e => e.eventType === "spread_recovery" || e.eventType === "leg_gap_anomaly");
  const allEvents = events;

  const durations = allEvents.map(e => e.durationMs);
  const magnitudes = allEvents.map(e => e.magnitude);
  const proxies = allEvents.map(e => e.conservativeCaptureProxy);
  const capturableCount = allEvents.filter(e => e.capturable).length;

  const elapsed = st.firstScanAt && st.lastScanAt ? st.lastScanAt - st.firstScanAt : 0;
  const hours = elapsed / 3_600_000;

  const byMarket: Record<string, number> = {};
  for (const e of allEvents) {
    byMarket[e.marketId] = (byMarket[e.marketId] ?? 0) + 1;
  }
  const concentration: Record<string, { count: number; share: number }> = {};
  const sorted = Object.entries(byMarket).sort((a, b) => b[1] - a[1]).slice(0, 12);
  for (const [mid, cnt] of sorted) {
    const q = allEvents.find(e => e.marketId === mid)?.marketQuestion ?? mid;
    concentration[q.slice(0, 100)] = { count: cnt, share: allEvents.length > 0 ? r4(cnt / allEvents.length) : 0 };
  }

  const posProxies = proxies.filter(p => p > 0);
  const negProxies = proxies.filter(p => p < 0);
  const repeatedPositiveEventPattern = posProxies.length >= 3 && posProxies.length > negProxies.length;
  const repeatedNegativeEventPattern = negProxies.length >= 3 && negProxies.length > posProxies.length * 2;

  const enoughSampleForRead = allEvents.length >= minEv;

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];

  let verdict: MomentumSnipingVerdict;
  if (!enoughSampleForRead) {
    verdict = "insufficient_sample";
    blockingReasons.push(`Eventos detectados ${allEvents.length} < mínimo ${minEv}. Scanner a ${SCAN_INTERVAL_MS() / 1000}s.`);
  } else if (repeatedNegativeEventPattern) {
    verdict = "unstable_or_negative";
    blockingReasons.push(
      `Padrão negativo: ${negProxies.length} eventos com capture proxy negativo vs ${posProxies.length} positivo.`,
    );
  } else if (capturableCount === 0) {
    verdict = "mostly_noise";
    supportingReasons.push("Todos os eventos têm capture proxy ≤ 0 após fee — microdesalinhamentos abaixo do custo mínimo.");
  } else {
    const posRate = allEvents.length > 0 ? capturableCount / allEvents.length : 0;
    const avgProxy = proxies.length > 0 ? proxies.reduce((a, b) => a + b, 0) / proxies.length : 0;

    if (
      posRate >= 0.4 &&
      capturableCount >= 5 &&
      avgProxy > 0 &&
      repeatedPositiveEventPattern
    ) {
      verdict = "promising_microstructure_signal";
      supportingReasons.push(
        `Taxa capturable ${r4(posRate)} com ${capturableCount} eventos positivos; padrão repetido; avgProxy=${r4(avgProxy)}.`,
      );
    } else if (capturableCount >= 2 && posRate >= 0.15) {
      verdict = "weak_executable_signal";
      supportingReasons.push(
        `${capturableCount} evento(s) capturable (posRate=${r4(posRate)}); sinal fraco mas existente.`,
      );
    } else {
      verdict = "mostly_noise";
      supportingReasons.push(
        `Eventos capturable ${capturableCount}/${allEvents.length}; insuficiente para sinal executável.`,
      );
    }
  }

  supportingReasons.push(
    "Eventos: spread_spike (alargamento súbito), spread_recovery (recuperação após spike), price_velocity_spike (movimento rápido de quote), leg_gap_anomaly (probSum desalinha de 1.0). captureProxy = magnitude × haircut conservador − fee.",
  );

  const thresholdsUsed: Record<string, number> = {
    MOMENTUM_SPREAD_SPIKE_THRESHOLD: SPREAD_SPIKE_THRESHOLD(),
    MOMENTUM_SPREAD_RECOVERY_THRESHOLD: SPREAD_RECOVERY_THRESHOLD(),
    MOMENTUM_PRICE_VELOCITY_THRESHOLD: PRICE_VELOCITY_THRESHOLD(),
    MOMENTUM_MIN_LIQUIDITY: MIN_LIQUIDITY(),
    MOMENTUM_FEE_PROXY: fee,
    MOMENTUM_MIN_EVENTS_FOR_READ: minEv,
    MOMENTUM_SCAN_INTERVAL_MS: SCAN_INTERVAL_MS(),
    MOMENTUM_MAX_EVENTS: MAX_EVENTS(),
  };

  const rankedAssessment = buildRankedEventAssessment(allEvents);
  const topSliceRobustnessAssessment = buildTopSliceRobustnessAssessment(rankedAssessment);
  const topSliceSelectionAssessment = buildTopSliceSelectionAssessment(allEvents);
  const opsAssessmentV2 = buildOperationalizationAssessmentV2(
    allEvents,
    rankedAssessment,
    topSliceRobustnessAssessment,
    topSliceSelectionAssessment,
  );
  const opsAssessment = buildOperationalizationAssessment(allEvents);
  const opsRobustness = buildOperationalizationRobustness(allEvents, opsAssessment);
  const promoReadiness = buildPromotionReadiness(allEvents, opsAssessment, opsRobustness);
  const promoProgress = buildPromotionProgress(opsAssessment, opsRobustness, promoReadiness);
  const paperExec = buildRealisticPaperExecution(allEvents, opsAssessment);
  const executionSurvivabilitySegmentation = buildExecutionSurvivabilitySegmentation(
    allEvents,
    opsAssessment,
    opsRobustness,
    promoReadiness,
    promoProgress,
    paperExec,
  );
  const segmentedPaperTestPreparation = buildSegmentedPaperTestPreparation(
    allEvents,
    rankedAssessment,
    topSliceRobustnessAssessment,
    topSliceSelectionAssessment,
    opsAssessment,
    opsRobustness,
    promoReadiness,
    promoProgress,
    paperExec,
    executionSurvivabilitySegmentation,
  );
  const segmentedPaperExecutionAssessment = buildSegmentedPaperExecutionAssessment(
    allEvents,
    opsAssessment,
    opsRobustness,
    promoReadiness,
    promoProgress,
    paperExec,
    executionSurvivabilitySegmentation,
    segmentedPaperTestPreparation,
  );
  const segmentedPaperExecutionWave2Assessment = buildSegmentedPaperExecutionWave2Assessment(
    allEvents,
    opsAssessment,
    opsRobustness,
    promoReadiness,
    promoProgress,
    paperExec,
    executionSurvivabilitySegmentation,
    segmentedPaperTestPreparation,
    segmentedPaperExecutionAssessment,
  );

  return {
    computedAt: new Date().toISOString(),
    probeVersion: "momentum-sniping-v1",
    readNature: "observational_microstructure_probe",
    readDisclaimer:
      "Probe observacional de microestrutura. Não executa trades; capture proxy é estimativa conservadora (magnitude × haircut − fee). promising ≠ lucro real; weak_signal ≠ estratégia viável sem execução real testada.",
    scannerRunning: st.loopStarted,
    snapshotsTaken: st.snapshotsTaken,
    candidateMomentumEventsCount: momentumEvents.length,
    candidateSnipingEventsCount: snipingEvents.length,
    averageEventDurationMs: durations.length > 0 ? Math.round(durations.reduce((a, b) => a + b, 0) / durations.length) : null,
    medianEventDurationMs: median(durations) != null ? Math.round(median(durations)!) : null,
    averageObservedDislocation: magnitudes.length > 0 ? r4(magnitudes.reduce((a, b) => a + b, 0) / magnitudes.length) : null,
    maxObservedDislocation: magnitudes.length > 0 ? r4(Math.max(...magnitudes)) : null,
    averageConservativeCapturableProxy: proxies.length > 0 ? r4(proxies.reduce((a, b) => a + b, 0) / proxies.length) : null,
    cumulativeConservativeCapturableProxy: r4(proxies.reduce((a, b) => a + b, 0)),
    eventFrequencyPerHour: hours > 0.01 ? r4(allEvents.length / hours) : null,
    repeatedPositiveEventPattern,
    repeatedNegativeEventPattern,
    concentrationByMarketOrBucket: concentration,
    enoughSampleForRead,
    momentumSnipingVerdict: verdict,
    supportingReasons,
    blockingReasons,
    thresholdsUsed,
    rankedEventAssessment: rankedAssessment,
    topSliceRobustnessAssessment,
    topSliceSelectionAssessment,
    operationalizationAssessmentV2: opsAssessmentV2,
    operationalizationAssessment: opsAssessment,
    operationalizationRobustnessAssessment: opsRobustness,
    promotionReadinessAssessment: promoReadiness,
    promotionProgressAssessment: promoProgress,
    realisticPaperExecutionAssessment: paperExec,
    executionSurvivabilitySegmentation,
    segmentedPaperTestPreparation,
    segmentedPaperExecutionAssessment,
    segmentedPaperExecutionWave2Assessment,
    recentEvents: allEvents.slice(-20),
  };
}
