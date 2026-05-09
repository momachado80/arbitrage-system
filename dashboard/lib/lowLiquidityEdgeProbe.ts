/**
 * Low-Liquidity Edge Probe — observação pura de micro-ineficiências em mercados
 * negligenciados (baixa liquidez, outcomes binários ou ternários).
 *
 * FASE 1: sem trades. Detecta candidatos, mede persistência temporal do edge,
 * estima executabilidade e custo, e reporta diagnósticos.
 *
 * Regime default de limiares probSum: over > 1.03 ou under < 0.96 (env PROBE_LOW_LIQ_OVERROUND / UNDERROUND).
 * Regime sensitivity (opcional, env): over > 1.01 ou under < 0.99 — PROBE_LOW_LIQ_THRESHOLD_REGIME=sensitivity
 * ou PROBE_LOW_LIQ_SENSITIVITY_MODE=1; overrides PROBE_LOW_LIQ_SENSITIVITY_OVERROUND / _UNDERROUND.
 *
 * Hipótese operacional:
 *   Mercados com $100–$2 000 de liquidez e probSum fora da banda neutra (conforme regime activo) podem conter
 *   micro-oportunidades ignoradas por humanos (ticket pequeno, custo de atenção alto).
 *   Se estas oportunidades forem recorrentes, duráveis (>30s) e sobreviverem a
 *   spread+fees estimados, podem ser capturáveis de forma escalável por volume de
 *   tentativas, não por ticket individual.
 *
 * Separação:
 *   - NÃO partilha tipos/paper/safety com graph, cross_market ou standard.
 *   - Estado em globalThis com chave própria.
 *   - Loop próprio (intervalo configurável; default 12s).
 *   - Endpoint próprio /api/probe/low-liquidity.
 */

import type { NormalizedMarket } from "./polymarketClient";
import { getAllMarkets } from "./marketDataService";

// ---------------------------------------------------------------------------
// Config (env-overridable; sem default agressivo)
// ---------------------------------------------------------------------------

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}

function envBool(k: string, def: boolean): boolean {
  const raw = process.env[k]?.trim().toLowerCase();
  if (!raw) return def;
  return raw === "1" || raw === "true" || raw === "yes";
}

/** Regime de limiares probSum: default (produção) vs passo de sensibilidade controlado (env apenas). */
export type ProbeThresholdRegime = "default" | "sensitivity";

/**
 * `default`: PROBE_LOW_LIQ_OVERROUND (default 1.03), PROBE_LOW_LIQ_UNDERROUND (default 0.96).
 * `sensitivity`: PROBE_LOW_LIQ_SENSITIVITY_OVERROUND (default 1.01), PROBE_LOW_LIQ_SENSITIVITY_UNDERROUND (default 0.99).
 * Activar sensibilidade: PROBE_LOW_LIQ_THRESHOLD_REGIME=sensitivity OU PROBE_LOW_LIQ_SENSITIVITY_MODE=1
 */
export function getProbeThresholdRegime(): ProbeThresholdRegime {
  const r = process.env.PROBE_LOW_LIQ_THRESHOLD_REGIME?.trim().toLowerCase();
  if (r === "sensitivity") return "sensitivity";
  if (envBool("PROBE_LOW_LIQ_SENSITIVITY_MODE", false)) return "sensitivity";
  return "default";
}

function activeOverroundThreshold(): number {
  if (getProbeThresholdRegime() === "sensitivity") {
    return envNum("PROBE_LOW_LIQ_SENSITIVITY_OVERROUND", 1.01);
  }
  return envNum("PROBE_LOW_LIQ_OVERROUND", 1.03);
}

function activeUnderroundThreshold(): number {
  if (getProbeThresholdRegime() === "sensitivity") {
    return envNum("PROBE_LOW_LIQ_SENSITIVITY_UNDERROUND", 0.99);
  }
  return envNum("PROBE_LOW_LIQ_UNDERROUND", 0.96);
}

const MIN_LIQ = () => envNum("PROBE_LOW_LIQ_MIN", 100);
const MAX_LIQ = () => envNum("PROBE_LOW_LIQ_MAX", 2000);
const MAX_OUTCOMES = () => envNum("PROBE_LOW_LIQ_MAX_OUTCOMES", 4);
const FEE_PER_LEG = () => envNum("PROBE_LOW_LIQ_FEE_PER_LEG", 0.02);
const ASSUMED_FILL_RATIO = () => envNum("PROBE_LOW_LIQ_FILL_RATIO", 0.25);
const MAX_EXECUTABLE_CAPITAL_SHARE = () => envNum("PROBE_LOW_LIQ_MAX_CAP_SHARE", 0.05);
const PROBE_CYCLE_MS = () => envNum("PROBE_LOW_LIQ_CYCLE_MS", 12_000);

const MAX_TRACKED_CANDIDATES = 200;
const MAX_SAMPLE_OUTPUT = 20;
const MAX_HISTORY_PER_CANDIDATE = 60;
/** Margem para classificar near-miss em relação aos limiares de probSum (não relaxa thresholds). */
const NEAR_MISS_DELTA = () => envNum("PROBE_LOW_LIQ_NEAR_MISS_DELTA", 0.01);
/** Mínimo de capital executável (USD) para contar `fails_executable_capital`; 0 = desactivado. */
const MIN_EXECUTABLE_USD_FOR_FAIL = () => envNum("PROBE_LOW_LIQ_MIN_EXECUTABLE_USD", 0);
const MAX_NEAR_MISS_SAMPLES = 15;

function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ProbeCandidateEdgeType = "overround" | "underround";

export interface ProbeCandidateSnapshot {
  marketId: string;
  question: string;
  category: string;
  outcomes: string[];
  prices: number[];
  liquidity: number;
  volume: number;
  probSum: number;
  spread: number;
  edgeType: ProbeCandidateEdgeType;
  grossEdge: number;
  estimatedSpreadCost: number;
  estimatedRoundTripFees: number;
  estimatedNetEdge: number;
  estimatedExecutableCapital: number;
  estimatedFillCapital: number;
  netEdgeSurvivesAfterCosts: boolean;
  detectedAt: number;
}

export interface ProbeCandidateTrack {
  marketId: string;
  question: string;
  category: string;
  firstSeenAt: number;
  lastSeenAt: number;
  sightings: number;
  edgeType: ProbeCandidateEdgeType;
  edgeHistory: Array<{
    ts: number;
    grossEdge: number;
    netEdge: number;
    liquidity: number;
    probSum: number;
  }>;
  peakGrossEdge: number;
  peakNetEdge: number;
  avgGrossEdge: number;
  avgNetEdge: number;
  avgLiquidity: number;
  edgePersistenceMs: number;
  lastNetEdgeSurvived: boolean;
}

export type ProbeVerdict =
  | "insufficient_data"
  | "promising"
  | "weak"
  | "destructive";

export interface LowLiquidityProbeDigest {
  computedAt: string;
  /** Regime activo de limiares probSum (default inalterado; sensitivity = passo opcional). */
  thresholdRegime: ProbeThresholdRegime;
  /** Como interpretar e alternar regime (env). */
  thresholdRegimeNote: string;
  probeStatus: "observing" | "idle";
  cycleCount: number;
  lastCycleAt: string | null;
  lastCycleDurationMs: number | null;
  marketsScanned: number;
  lowLiqMarketCount: number;
  config: {
    minLiquidity: number;
    maxLiquidity: number;
    /** Limiares efectivamente usados neste ciclo (reflectem o regime activo). */
    overroundThreshold: number;
    underroundThreshold: number;
    maxOutcomes: number;
    feePerLeg: number;
    assumedFillRatio: number;
    cyclePeriodMs: number;
  };
  candidateCountCurrentCycle: number;
  trackedCandidateCount: number;
  recurringCandidateCount: number;
  avgObservedLiquidity: number | null;
  avgEstimatedExecutableCapital: number | null;
  avgEstimatedNetEdgeAfterCosts: number | null;
  avgEdgePersistenceMs: number | null;
  netEdgeSurvivalRate: number | null;
  dominantExitProblem: string;
  topSamples: ProbeSampleRow[];
  reasonCounts: Record<string, number>;
  probeVerdict: ProbeVerdict;
  probeVerdictExplanation: string;
  /** Funil do último ciclo completado (candidatos = 0 → ver neutral band + failureReasonCounts). */
  funnelDiagnostics: ProbeFunnelDiagnostics | null;
}

export interface ProbeSampleRow {
  marketId: string;
  question: string;
  edgeType: ProbeCandidateEdgeType;
  sightings: number;
  avgGrossEdge: number;
  avgNetEdge: number;
  avgLiquidity: number;
  edgePersistenceMs: number;
  netEdgeSurvived: boolean;
}

/** Funil do último ciclo — onde os candidatos são filtrados (diagnóstico; não altera thresholds). */
export interface ProbeFunnelDiagnostics {
  computedAtCycle: number;
  thresholdRegime: ProbeThresholdRegime;
  activeOverroundThreshold: number;
  activeUnderroundThreshold: number;
  marketsScanned: number;
  /** Activo, não fechado, 2 ≤ outcomes ≤ maxOutcomes (antes do filtro de liquidez). */
  validOutcomeCountMarkets: number;
  /** Passa outcomes válidos e banda de liquidez [min, max]. */
  lowLiquidityMarkets: number;
  marketsWithProbSumOverOverroundThreshold: number;
  marketsWithProbSumUnderUnderroundThreshold: number;
  marketsWithPositiveGrossEdge: number;
  marketsWithPositiveNetEdgeAfterSpread: number;
  marketsWithPositiveNetEdgeAfterFees: number;
  nearMissCandidates: number;
  nearMissOverround: number;
  nearMissUnderround: number;
  failureReasonCounts: Record<string, number>;
  topNearMissSamples: ProbeNearMissSampleRow[];
  funnelNote: string;
}

export interface ProbeNearMissSampleRow {
  marketId: string;
  question: string;
  probSum: number;
  liquidity: number;
  spread: number;
  grossEdge: number;
  netAfterSpread: number;
  netAfterFees: number;
  distanceToNearestThreshold: number;
  failureReasons: string[];
}

// ---------------------------------------------------------------------------
// Runtime state (globalThis, separado de tudo o resto)
// ---------------------------------------------------------------------------

const GLOBAL_KEY = "__lowLiquidityEdgeProbe_v1";

interface ProbeRuntimeState {
  loopStarted: boolean;
  cycleCount: number;
  lastCycleAt: number | null;
  lastCycleDurationMs: number | null;
  lastMarketsScanned: number;
  lastLowLiqCount: number;
  lastCandidateCount: number;
  lastFunnel: ProbeFunnelDiagnostics | null;
  tracks: Map<string, ProbeCandidateTrack>;
}

function getProbeState(): ProbeRuntimeState {
  const g = globalThis as unknown as Record<string, ProbeRuntimeState | undefined>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = {
      loopStarted: false,
      cycleCount: 0,
      lastCycleAt: null,
      lastCycleDurationMs: null,
      lastMarketsScanned: 0,
      lastLowLiqCount: 0,
      lastCandidateCount: 0,
      lastFunnel: null,
      tracks: new Map(),
    };
  }
  return g[GLOBAL_KEY]!;
}

// ---------------------------------------------------------------------------
// Scanner (pura observação, sem side-effects)
// ---------------------------------------------------------------------------

function isValidOutcomeCount(m: NormalizedMarket): boolean {
  if (m.closed || !m.active) return false;
  return m.outcomes.length >= 2 && m.outcomes.length <= MAX_OUTCOMES();
}

function isLowLiqCandidate(m: NormalizedMarket): boolean {
  if (!isValidOutcomeCount(m)) return false;
  if (m.liquidity < MIN_LIQ() || m.liquidity > MAX_LIQ()) return false;
  return true;
}

function computeCosts(m: NormalizedMarket, grossEdge: number) {
  const spreadCost = m.spread * 0.5;
  const roundTripFees = FEE_PER_LEG() * 2;
  const netAfterSpread = grossEdge - spreadCost;
  const netAfterFees = grossEdge - spreadCost - roundTripFees;
  const maxCapShare = MAX_EXECUTABLE_CAPITAL_SHARE();
  const executableCapital = m.liquidity * maxCapShare;
  const fillCapital = executableCapital * ASSUMED_FILL_RATIO();
  return { spreadCost, roundTripFees, netAfterSpread, netAfterFees, executableCapital, fillCapital };
}

type NearMissAgg = {
  row: ProbeNearMissSampleRow;
  sortKey: number;
};

function buildFailureReasonsForMarket(
  m: NormalizedMarket,
  sum: number,
  overT: number,
  underT: number,
  grossEdge: number,
  netAfterSpread: number,
  netAfterFees: number,
  executableCapital: number,
  fillCapital: number,
  isOver: boolean,
  isUnder: boolean
): string[] {
  const reasons: string[] = [];
  if (!isOver && !isUnder) {
    reasons.push("neutral_probsum_band");
  }
  if (sum <= overT) reasons.push("fails_overround_threshold");
  if (sum >= underT) reasons.push("fails_underround_threshold");
  if ((isOver || isUnder) && grossEdge > 0 && netAfterSpread <= 0) {
    reasons.push("fails_spread_cost");
  }
  if ((isOver || isUnder) && grossEdge > 0 && netAfterSpread > 0 && netAfterFees <= 0) {
    reasons.push("fails_fee_cost");
  }
  const minExe = MIN_EXECUTABLE_USD_FOR_FAIL();
  if (minExe > 0 && (isOver || isUnder) && grossEdge > 0 && executableCapital < minExe) {
    reasons.push("fails_executable_capital");
  }
  if ((isOver || isUnder) && grossEdge > 0 && fillCapital <= 1e-9) {
    reasons.push("fails_fill_assumption");
  }
  return reasons;
}

function computeProbeFunnel(markets: NormalizedMarket[], cycleIndex: number): ProbeFunnelDiagnostics {
  const overT = activeOverroundThreshold();
  const underT = activeUnderroundThreshold();
  const delta = NEAR_MISS_DELTA();

  let validOutcomeCountMarkets = 0;
  for (const m of markets) {
    if (isValidOutcomeCount(m)) validOutcomeCountMarkets++;
  }

  const lowLiq = markets.filter(isLowLiqCandidate);
  const nLow = lowLiq.length;

  let marketsWithProbSumOverOverroundThreshold = 0;
  let marketsWithProbSumUnderUnderroundThreshold = 0;
  let marketsWithPositiveGrossEdge = 0;
  let marketsWithPositiveNetEdgeAfterSpread = 0;
  let marketsWithPositiveNetEdgeAfterFees = 0;

  const failureReasonCounts: Record<string, number> = {
    fails_overround_threshold: 0,
    fails_underround_threshold: 0,
    fails_spread_cost: 0,
    fails_fee_cost: 0,
    fails_executable_capital: 0,
    fails_fill_assumption: 0,
    neutral_probsum_band: 0,
    near_miss_overround: 0,
    near_miss_underround: 0,
  };

  const nearMissPool: NearMissAgg[] = [];

  for (const m of lowLiq) {
    const sum = m.probSum;
    const isOver = sum > overT;
    const isUnder = sum < underT;
    const inNeutral = !isOver && !isUnder;

    if (isOver) marketsWithProbSumOverOverroundThreshold++;
    if (isUnder) marketsWithProbSumUnderUnderroundThreshold++;

    let grossEdge = 0;
    if (isOver) grossEdge = sum - 1;
    else if (isUnder) grossEdge = 1 - sum;

    if (isOver || isUnder) marketsWithPositiveGrossEdge++;

    const { netAfterSpread, netAfterFees, executableCapital, fillCapital } = computeCosts(m, grossEdge);

    if ((isOver || isUnder) && grossEdge > 0 && netAfterSpread > 0) {
      marketsWithPositiveNetEdgeAfterSpread++;
    }
    if ((isOver || isUnder) && grossEdge > 0 && netAfterFees > 0) {
      marketsWithPositiveNetEdgeAfterFees++;
    }

    if (inNeutral) {
      failureReasonCounts["neutral_probsum_band"] =
        (failureReasonCounts["neutral_probsum_band"] ?? 0) + 1;
    }
    if (sum <= overT) {
      failureReasonCounts["fails_overround_threshold"] =
        (failureReasonCounts["fails_overround_threshold"] ?? 0) + 1;
    }
    if (sum >= underT) {
      failureReasonCounts["fails_underround_threshold"] =
        (failureReasonCounts["fails_underround_threshold"] ?? 0) + 1;
    }

    if ((isOver || isUnder) && grossEdge > 0 && netAfterSpread <= 0) {
      failureReasonCounts["fails_spread_cost"] = (failureReasonCounts["fails_spread_cost"] ?? 0) + 1;
    }
    if ((isOver || isUnder) && grossEdge > 0 && netAfterSpread > 0 && netAfterFees <= 0) {
      failureReasonCounts["fails_fee_cost"] = (failureReasonCounts["fails_fee_cost"] ?? 0) + 1;
    }
    const minExe = MIN_EXECUTABLE_USD_FOR_FAIL();
    if (minExe > 0 && (isOver || isUnder) && grossEdge > 0 && executableCapital < minExe) {
      failureReasonCounts["fails_executable_capital"] =
        (failureReasonCounts["fails_executable_capital"] ?? 0) + 1;
    }
    if ((isOver || isUnder) && grossEdge > 0 && fillCapital <= 1e-9) {
      failureReasonCounts["fails_fill_assumption"] =
        (failureReasonCounts["fails_fill_assumption"] ?? 0) + 1;
    }

    if (sum > overT - delta && sum <= overT) {
      failureReasonCounts["near_miss_overround"] = (failureReasonCounts["near_miss_overround"] ?? 0) + 1;
    }
    if (sum >= underT && sum < underT + delta) {
      failureReasonCounts["near_miss_underround"] = (failureReasonCounts["near_miss_underround"] ?? 0) + 1;
    }

    const reasons = buildFailureReasonsForMarket(
      m,
      sum,
      overT,
      underT,
      grossEdge,
      netAfterSpread,
      netAfterFees,
      executableCapital,
      fillCapital,
      isOver,
      isUnder
    );

    if (inNeutral) {
      const distNeutral = Math.min(overT - sum, sum - underT);
      nearMissPool.push({
        sortKey: distNeutral,
        row: {
          marketId: m.id,
          question: m.question.slice(0, 140),
          probSum: r4(sum),
          liquidity: r4(m.liquidity),
          spread: r4(m.spread),
          grossEdge: r4(grossEdge),
          netAfterSpread: r4(netAfterSpread),
          netAfterFees: r4(netAfterFees),
          distanceToNearestThreshold: r4(distNeutral),
          failureReasons: reasons,
        },
      });
    } else if ((isOver || isUnder) && grossEdge > 0 && netAfterFees <= 0) {
      nearMissPool.push({
        sortKey: -netAfterFees,
        row: {
          marketId: m.id,
          question: m.question.slice(0, 140),
          probSum: r4(sum),
          liquidity: r4(m.liquidity),
          spread: r4(m.spread),
          grossEdge: r4(grossEdge),
          netAfterSpread: r4(netAfterSpread),
          netAfterFees: r4(netAfterFees),
          distanceToNearestThreshold: r4(Math.abs(netAfterFees)),
          failureReasons: reasons,
        },
      });
    }
  }

  nearMissPool.sort((a, b) => {
    const aN = a.row.failureReasons.includes("neutral_probsum_band");
    const bN = b.row.failureReasons.includes("neutral_probsum_band");
    if (aN && !bN) return -1;
    if (!aN && bN) return 1;
    return a.sortKey - b.sortKey;
  });

  const topNearMissSamples = nearMissPool.slice(0, MAX_NEAR_MISS_SAMPLES).map((x) => x.row);

  const nearMissCandidates =
    (failureReasonCounts["near_miss_overround"] ?? 0) + (failureReasonCounts["near_miss_underround"] ?? 0);

  const funnelNote =
    "Counts are from the latest probe cycle over low-liquidity markets only (except marketsScanned and validOutcomeCountMarkets). " +
    "neutral_probsum_band counts markets where underT ≤ probSum ≤ overT — the usual reason for zero candidates when thresholds are tight. " +
    "fails_overround_threshold and fails_underround_threshold count how many low-liq markets do NOT exceed each side (neutral markets increment both).";

  return {
    computedAtCycle: cycleIndex,
    thresholdRegime: getProbeThresholdRegime(),
    activeOverroundThreshold: r4(overT),
    activeUnderroundThreshold: r4(underT),
    marketsScanned: markets.length,
    validOutcomeCountMarkets,
    lowLiquidityMarkets: nLow,
    marketsWithProbSumOverOverroundThreshold: marketsWithProbSumOverOverroundThreshold,
    marketsWithProbSumUnderUnderroundThreshold: marketsWithProbSumUnderUnderroundThreshold,
    marketsWithPositiveGrossEdge,
    marketsWithPositiveNetEdgeAfterSpread,
    marketsWithPositiveNetEdgeAfterFees,
    nearMissCandidates,
    nearMissOverround: failureReasonCounts["near_miss_overround"] ?? 0,
    nearMissUnderround: failureReasonCounts["near_miss_underround"] ?? 0,
    failureReasonCounts,
    topNearMissSamples,
    funnelNote,
  };
}

function detectEdge(m: NormalizedMarket): ProbeCandidateSnapshot | null {
  const sum = m.probSum;
  const overT = activeOverroundThreshold();
  const underT = activeUnderroundThreshold();

  let edgeType: ProbeCandidateEdgeType;
  let grossEdge: number;

  if (sum > overT) {
    edgeType = "overround";
    grossEdge = sum - 1;
  } else if (sum < underT) {
    edgeType = "underround";
    grossEdge = 1 - sum;
  } else {
    return null;
  }

  const spreadCost = m.spread * 0.5;
  const feeLeg = FEE_PER_LEG();
  const roundTripFees = feeLeg * 2;
  const netEdge = grossEdge - spreadCost - roundTripFees;

  const maxCapShare = MAX_EXECUTABLE_CAPITAL_SHARE();
  const executableCapital = m.liquidity * maxCapShare;
  const fillRatio = ASSUMED_FILL_RATIO();
  const fillCapital = executableCapital * fillRatio;

  return {
    marketId: m.id,
    question: m.question,
    category: m.category,
    outcomes: m.outcomes,
    prices: m.prices,
    liquidity: m.liquidity,
    volume: m.volume,
    probSum: sum,
    spread: m.spread,
    edgeType,
    grossEdge,
    estimatedSpreadCost: spreadCost,
    estimatedRoundTripFees: roundTripFees,
    estimatedNetEdge: netEdge,
    estimatedExecutableCapital: executableCapital,
    estimatedFillCapital: fillCapital,
    netEdgeSurvivesAfterCosts: netEdge > 0,
    detectedAt: Date.now(),
  };
}

function updateTrack(st: ProbeRuntimeState, snap: ProbeCandidateSnapshot): void {
  let track = st.tracks.get(snap.marketId);
  const now = snap.detectedAt;

  if (!track) {
    if (st.tracks.size >= MAX_TRACKED_CANDIDATES) {
      let oldest: string | null = null;
      let oldestTs = Infinity;
      for (const [k, v] of Array.from(st.tracks.entries())) {
        if (v.lastSeenAt < oldestTs) {
          oldestTs = v.lastSeenAt;
          oldest = k;
        }
      }
      if (oldest) st.tracks.delete(oldest);
    }

    track = {
      marketId: snap.marketId,
      question: snap.question,
      category: snap.category,
      firstSeenAt: now,
      lastSeenAt: now,
      sightings: 0,
      edgeType: snap.edgeType,
      edgeHistory: [],
      peakGrossEdge: 0,
      peakNetEdge: -Infinity,
      avgGrossEdge: 0,
      avgNetEdge: 0,
      avgLiquidity: 0,
      edgePersistenceMs: 0,
      lastNetEdgeSurvived: false,
    };
    st.tracks.set(snap.marketId, track);
  }

  track.lastSeenAt = now;
  track.sightings += 1;
  track.edgeType = snap.edgeType;
  track.lastNetEdgeSurvived = snap.netEdgeSurvivesAfterCosts;

  if (track.edgeHistory.length >= MAX_HISTORY_PER_CANDIDATE) {
    track.edgeHistory.shift();
  }
  track.edgeHistory.push({
    ts: now,
    grossEdge: snap.grossEdge,
    netEdge: snap.estimatedNetEdge,
    liquidity: snap.liquidity,
    probSum: snap.probSum,
  });

  let sumGross = 0;
  let sumNet = 0;
  let sumLiq = 0;
  for (const h of track.edgeHistory) {
    sumGross += h.grossEdge;
    sumNet += h.netEdge;
    sumLiq += h.liquidity;
  }
  const n = track.edgeHistory.length;
  track.avgGrossEdge = sumGross / n;
  track.avgNetEdge = sumNet / n;
  track.avgLiquidity = sumLiq / n;
  track.peakGrossEdge = Math.max(track.peakGrossEdge, snap.grossEdge);
  track.peakNetEdge = Math.max(track.peakNetEdge, snap.estimatedNetEdge);
  track.edgePersistenceMs = track.lastSeenAt - track.firstSeenAt;
}

// ---------------------------------------------------------------------------
// Cycle
// ---------------------------------------------------------------------------

function runProbeCycle(): void {
  const st = getProbeState();
  const t0 = Date.now();

  try {
    const markets = getAllMarkets();
    st.lastMarketsScanned = markets.length;

    const lowLiq = markets.filter(isLowLiqCandidate);
    st.lastLowLiqCount = lowLiq.length;

    const candidates: ProbeCandidateSnapshot[] = [];
    for (const m of lowLiq) {
      const snap = detectEdge(m);
      if (snap) candidates.push(snap);
    }
    st.lastCandidateCount = candidates.length;

    for (const c of candidates) {
      updateTrack(st, c);
    }

    st.cycleCount += 1;
    st.lastFunnel = computeProbeFunnel(markets, st.cycleCount);
    st.lastCycleAt = Date.now();
    st.lastCycleDurationMs = Date.now() - t0;

    if (st.cycleCount <= 3 || st.cycleCount % 50 === 0) {
      console.log(
        `[LowLiqProbe] cycle=${st.cycleCount} scanned=${markets.length} lowLiq=${lowLiq.length} candidates=${candidates.length} tracked=${st.tracks.size} ms=${st.lastCycleDurationMs}`
      );
    }
  } catch (err) {
    st.lastCycleDurationMs = Date.now() - t0;
    console.error("[LowLiqProbe] cycle error:", err instanceof Error ? err.message : err);
  }
}

// ---------------------------------------------------------------------------
// Loop startup (idempotente via globalThis)
// ---------------------------------------------------------------------------

export function ensureLowLiquidityProbe(): void {
  const st = getProbeState();
  if (st.loopStarted) return;
  st.loopStarted = true;
  console.log("[LowLiqProbe] Observation loop started (pure observation, no trades)");
  setTimeout(runProbeCycle, 8_000);
  setInterval(runProbeCycle, PROBE_CYCLE_MS());
}

// ---------------------------------------------------------------------------
// Digest builder (leitura pura do estado)
// ---------------------------------------------------------------------------

function avg(arr: number[]): number | null {
  if (arr.length === 0) return null;
  return r4(arr.reduce((a, b) => a + b, 0) / arr.length);
}

export function buildLowLiquidityProbeDigest(): LowLiquidityProbeDigest {
  const st = getProbeState();
  const tracks = Array.from(st.tracks.values());

  const recurring = tracks.filter((t) => t.sightings >= 3);
  const withNetSurvived = tracks.filter((t) => t.avgNetEdge > 0);

  const allLiqs = tracks.map((t) => t.avgLiquidity);
  const allExecCaps = tracks.map((t) => t.avgLiquidity * MAX_EXECUTABLE_CAPITAL_SHARE());
  const allNetEdges = tracks.map((t) => t.avgNetEdge);
  const allPersistence = tracks.filter((t) => t.sightings >= 2).map((t) => t.edgePersistenceMs);

  const netSurvivalRate =
    tracks.length > 0 ? r4(withNetSurvived.length / tracks.length) : null;

  const reasonCounts: Record<string, number> = {
    overround_gross_positive: 0,
    underround_gross_positive: 0,
    net_survives_after_costs: 0,
    net_destroyed_by_spread: 0,
    net_destroyed_by_fees: 0,
    recurring_3plus_sightings: recurring.length,
    single_sighting_only: tracks.filter((t) => t.sightings === 1).length,
  };
  for (const t of tracks) {
    if (t.edgeType === "overround") reasonCounts["overround_gross_positive"]!++;
    else reasonCounts["underround_gross_positive"]!++;
    if (t.avgNetEdge > 0) {
      reasonCounts["net_survives_after_costs"]!++;
    } else {
      const feeCost = FEE_PER_LEG() * 2;
      const spreadGross = t.avgGrossEdge - t.avgNetEdge - feeCost;
      if (spreadGross > feeCost) {
        reasonCounts["net_destroyed_by_spread"]!++;
      } else {
        reasonCounts["net_destroyed_by_fees"]!++;
      }
    }
  }

  let dominantExitProblem = "insufficient_data";
  if (tracks.length > 0) {
    const spreadKills = reasonCounts["net_destroyed_by_spread"] ?? 0;
    const feeKills = reasonCounts["net_destroyed_by_fees"] ?? 0;
    const survives = reasonCounts["net_survives_after_costs"] ?? 0;
    if (survives > spreadKills && survives > feeKills) {
      dominantExitProblem = "none_so_far";
    } else if (spreadKills >= feeKills) {
      dominantExitProblem = "spread_dominates_edge";
    } else {
      dominantExitProblem = "fees_dominate_edge";
    }
  }

  const topSorted = [...tracks]
    .sort((a, b) => {
      const aScore = a.avgNetEdge * a.sightings;
      const bScore = b.avgNetEdge * b.sightings;
      return bScore - aScore;
    })
    .slice(0, MAX_SAMPLE_OUTPUT);

  const topSamples: ProbeSampleRow[] = topSorted.map((t) => ({
    marketId: t.marketId,
    question: t.question.slice(0, 120),
    edgeType: t.edgeType,
    sightings: t.sightings,
    avgGrossEdge: r4(t.avgGrossEdge),
    avgNetEdge: r4(t.avgNetEdge),
    avgLiquidity: r4(t.avgLiquidity),
    edgePersistenceMs: t.edgePersistenceMs,
    netEdgeSurvived: t.avgNetEdge > 0,
  }));

  let probeVerdict: ProbeVerdict = "insufficient_data";
  let probeVerdictExplanation =
    "Not enough observation cycles or tracked candidates to draw a conclusion.";

  if (st.cycleCount >= 10 && tracks.length >= 5) {
    const survivalRate = netSurvivalRate ?? 0;
    const recurringRate = tracks.length > 0 ? recurring.length / tracks.length : 0;
    const avgPersist = avg(allPersistence) ?? 0;

    if (survivalRate >= 0.3 && recurringRate >= 0.2 && avgPersist >= 30_000) {
      probeVerdict = "promising";
      probeVerdictExplanation =
        `Net edge survives in ${r4(survivalRate * 100)}% of tracked candidates, ` +
        `${r4(recurringRate * 100)}% are recurring (3+ sightings), ` +
        `avg persistence ${Math.round(avgPersist / 1000)}s. ` +
        "Consider advancing to Phase 2 (ultra-small paper probe).";
    } else if (survivalRate >= 0.1 || recurringRate >= 0.1) {
      probeVerdict = "weak";
      probeVerdictExplanation =
        `Net edge survival ${r4(survivalRate * 100)}%, ` +
        `recurring rate ${r4(recurringRate * 100)}%, ` +
        `avg persistence ${Math.round(avgPersist / 1000)}s. ` +
        "Signal exists but too weak for Phase 2 without further filtering.";
    } else {
      probeVerdict = "destructive";
      probeVerdictExplanation =
        `Net edge survival ${r4(survivalRate * 100)}%, ` +
        `recurring rate ${r4(recurringRate * 100)}%. ` +
        "Edges are consistently destroyed by costs. No basis for Phase 2.";
    }
  }

  const regime = getProbeThresholdRegime();
  const thresholdRegimeNote =
    regime === "default"
      ? "Active: default probSum thresholds (PROBE_LOW_LIQ_OVERROUND default 1.03, PROBE_LOW_LIQ_UNDERROUND default 0.96). For controlled sensitivity pass: PROBE_LOW_LIQ_THRESHOLD_REGIME=sensitivity or PROBE_LOW_LIQ_SENSITIVITY_MODE=1 (uses PROBE_LOW_LIQ_SENSITIVITY_OVERROUND default 1.01, PROBE_LOW_LIQ_SENSITIVITY_UNDERROUND default 0.99)."
      : "Active: sensitivity probSum thresholds (PROBE_LOW_LIQ_SENSITIVITY_OVERROUND default 1.01, PROBE_LOW_LIQ_SENSITIVITY_UNDERROUND default 0.99). Revert to default: unset PROBE_LOW_LIQ_THRESHOLD_REGIME and PROBE_LOW_LIQ_SENSITIVITY_MODE.";

  return {
    computedAt: new Date().toISOString(),
    thresholdRegime: regime,
    thresholdRegimeNote,
    probeStatus: st.loopStarted ? "observing" : "idle",
    cycleCount: st.cycleCount,
    lastCycleAt: st.lastCycleAt ? new Date(st.lastCycleAt).toISOString() : null,
    lastCycleDurationMs: st.lastCycleDurationMs,
    marketsScanned: st.lastMarketsScanned,
    lowLiqMarketCount: st.lastLowLiqCount,
    config: {
      minLiquidity: MIN_LIQ(),
      maxLiquidity: MAX_LIQ(),
      overroundThreshold: activeOverroundThreshold(),
      underroundThreshold: activeUnderroundThreshold(),
      maxOutcomes: MAX_OUTCOMES(),
      feePerLeg: FEE_PER_LEG(),
      assumedFillRatio: ASSUMED_FILL_RATIO(),
      cyclePeriodMs: PROBE_CYCLE_MS(),
    },
    candidateCountCurrentCycle: st.lastCandidateCount,
    trackedCandidateCount: tracks.length,
    recurringCandidateCount: recurring.length,
    avgObservedLiquidity: avg(allLiqs),
    avgEstimatedExecutableCapital: avg(allExecCaps),
    avgEstimatedNetEdgeAfterCosts: avg(allNetEdges),
    avgEdgePersistenceMs: avg(allPersistence),
    netEdgeSurvivalRate: netSurvivalRate,
    dominantExitProblem,
    topSamples,
    reasonCounts,
    probeVerdict,
    probeVerdictExplanation,
    funnelDiagnostics: st.lastFunnel,
  };
}
