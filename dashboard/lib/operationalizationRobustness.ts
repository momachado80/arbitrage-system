/**
 * Operationalization Robustness Assessment — avalia se o rule set vencedor
 * (conservative) permanece estável à medida que a amostra cresce e em
 * diferentes janelas temporais. Inclui sensibilidade a thresholds e risco
 * de overfit. Puramente observacional; não executa trades.
 */

import type { MomentumEvent } from "./momentumSnipingProbe";
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

const MIN_EVENTS_FOR_ROBUSTNESS = () =>
  Math.max(8, Math.floor(envNum("MOMENTUM_OPS_ROBUSTNESS_MIN_EVENTS", 12)));
const MIN_WINDOW_SIZE = 4;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type OpsRobustnessVerdict =
  | "insufficient_sample_for_full_robustness"
  | "unstable"
  | "fragile"
  | "weak_but_persistent"
  | "stable"
  | "robust";

export type OverfitRisk = "low" | "moderate" | "high";

export type ThresholdSensitivity = "stable" | "fragile" | "highly_threshold_dependent";

interface WindowMetrics {
  windowId: string;
  eventCount: number;
  passingCount: number;
  coverage: number;
  avgProxyFiltered: number | null;
  improvementVsBaseline: number | null;
  capturableRate: number;
  concentrationRisk: number;
  distinctMarkets: number;
}

interface SensitivityPoint {
  magnitudeFloor: number;
  proxyNegativeCutoff: number;
  passingCount: number;
  coverage: number;
  avgProxyFiltered: number | null;
  improvementVsBaseline: number | null;
}

export interface OperationalizationRobustnessAssessment {
  totalEventsForRobustness: number;
  minimumRequired: number;
  hasFullRobustness: boolean;
  robustnessVerdict: OpsRobustnessVerdict;
  bestRuleSetStabilityScore: number | null;
  improvementPersistenceRate: number | null;
  rollingWindowImprovementSeries: WindowMetrics[];
  rollingWindowCoverageSeries: WindowMetrics[];
  rollingWindowConcentrationSeries: WindowMetrics[];
  cumulativeGrowthWindows: WindowMetrics[];
  fixedRecentWindows: WindowMetrics[];
  equalBucketWindows: WindowMetrics[];
  thresholdSensitivitySummary: {
    sensitivity: ThresholdSensitivity;
    grid: SensitivityPoint[];
    stableCount: number;
    fragileCount: number;
    explanation: string;
  };
  dominantPassReasons: string[];
  dominantExitReasons: string[];
  discardRuleEffectivenessSummary: Array<{
    ruleId: string;
    totalDiscarded: number;
    avgProxyDiscarded: number | null;
    effectivenessLabel: string;
  }>;
  overfitRiskVerdict: OverfitRisk;
  overfitRiskFactors: string[];
  supportingReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: Record<string, number>;
  readDisclaimer: string;
}

// ---------------------------------------------------------------------------
// Conservative rule filter — replicated here to avoid coupling
// ---------------------------------------------------------------------------

function applyConservativeRule(
  events: readonly MomentumEvent[],
  magFloor: number,
  proxyNegCutoff: number,
): { passing: MomentumEvent[]; discarded: MomentumEvent[] } {
  const passing: MomentumEvent[] = [];
  const discarded: MomentumEvent[] = [];
  for (const e of events) {
    const entryOk = e.capturable && e.magnitude >= magFloor;
    const discardHit = e.magnitude < 0.003 || e.conservativeCaptureProxy <= proxyNegCutoff;
    if (entryOk && !discardHit) passing.push(e);
    else discarded.push(e);
  }
  return { passing, discarded };
}

function windowMetrics(
  windowId: string,
  events: readonly MomentumEvent[],
  magFloor: number,
  proxyNegCutoff: number,
): WindowMetrics {
  const baseAvg = avg(events.map(e => e.conservativeCaptureProxy));
  const { passing } = applyConservativeRule(events, magFloor, proxyNegCutoff);
  const filtAvg = avg(passing.map(e => e.conservativeCaptureProxy));
  const mktSet = new Set(passing.map(e => e.marketId));
  const mktCounts: Record<string, number> = {};
  for (const e of passing) mktCounts[e.marketId] = (mktCounts[e.marketId] ?? 0) + 1;
  const maxConc = passing.length > 0
    ? r4(Math.max(...Object.values(mktCounts)) / passing.length)
    : 0;

  return {
    windowId,
    eventCount: events.length,
    passingCount: passing.length,
    coverage: events.length > 0 ? r4(passing.length / events.length) : 0,
    avgProxyFiltered: filtAvg,
    improvementVsBaseline:
      baseAvg !== null && filtAvg !== null ? r4(filtAvg - baseAvg) : null,
    capturableRate: passing.length > 0
      ? r4(passing.filter(e => e.capturable).length / passing.length)
      : 0,
    concentrationRisk: maxConc,
    distinctMarkets: mktSet.size,
  };
}

// ---------------------------------------------------------------------------
// Window builders
// ---------------------------------------------------------------------------

function buildCumulativeGrowthWindows(
  events: readonly MomentumEvent[],
  magFloor: number,
  proxyNegCutoff: number,
): WindowMetrics[] {
  const n = events.length;
  if (n < MIN_WINDOW_SIZE) return [];
  const steps = [
    Math.min(n, Math.max(MIN_WINDOW_SIZE, Math.floor(n * 0.25))),
    Math.min(n, Math.max(MIN_WINDOW_SIZE, Math.floor(n * 0.5))),
    Math.min(n, Math.max(MIN_WINDOW_SIZE, Math.floor(n * 0.75))),
    n,
  ];
  const unique = Array.from(new Set(steps)).filter(s => s >= MIN_WINDOW_SIZE);
  return unique.map((s, i) =>
    windowMetrics(`cumulative_${i + 1}_of_${unique.length}`, events.slice(0, s), magFloor, proxyNegCutoff),
  );
}

function buildFixedRecentWindows(
  events: readonly MomentumEvent[],
  magFloor: number,
  proxyNegCutoff: number,
): WindowMetrics[] {
  const n = events.length;
  if (n < MIN_WINDOW_SIZE) return [];
  const sizes = [
    Math.min(n, Math.max(MIN_WINDOW_SIZE, Math.floor(n * 0.33))),
    Math.min(n, Math.max(MIN_WINDOW_SIZE, Math.floor(n * 0.5))),
    Math.min(n, n),
  ];
  const unique = Array.from(new Set(sizes)).filter(s => s >= MIN_WINDOW_SIZE);
  return unique.map((s, i) =>
    windowMetrics(`recent_last_${s}`, events.slice(-s), magFloor, proxyNegCutoff),
  );
}

function buildEqualBucketWindows(
  events: readonly MomentumEvent[],
  magFloor: number,
  proxyNegCutoff: number,
): WindowMetrics[] {
  const n = events.length;
  const buckets = Math.min(3, Math.floor(n / MIN_WINDOW_SIZE));
  if (buckets < 2) return [];
  const size = Math.floor(n / buckets);
  const windows: WindowMetrics[] = [];
  for (let i = 0; i < buckets; i++) {
    const start = i * size;
    const end = i === buckets - 1 ? n : start + size;
    windows.push(
      windowMetrics(`bucket_${i + 1}_of_${buckets}`, events.slice(start, end), magFloor, proxyNegCutoff),
    );
  }
  return windows;
}

// ---------------------------------------------------------------------------
// Threshold sensitivity grid (small, controlled)
// ---------------------------------------------------------------------------

function buildSensitivityGrid(
  events: readonly MomentumEvent[],
): { grid: SensitivityPoint[]; sensitivity: ThresholdSensitivity; stableCount: number; fragileCount: number; explanation: string } {
  const magFloors = [0.003, 0.005, 0.008];
  const proxyCutoffs = [-0.001, -0.003, -0.005];
  const baseAvg = avg(events.map(e => e.conservativeCaptureProxy));

  const grid: SensitivityPoint[] = [];
  for (const mf of magFloors) {
    for (const pc of proxyCutoffs) {
      const { passing } = applyConservativeRule(events, mf, pc);
      const filtAvg = avg(passing.map(e => e.conservativeCaptureProxy));
      grid.push({
        magnitudeFloor: mf,
        proxyNegativeCutoff: pc,
        passingCount: passing.length,
        coverage: events.length > 0 ? r4(passing.length / events.length) : 0,
        avgProxyFiltered: filtAvg,
        improvementVsBaseline:
          baseAvg !== null && filtAvg !== null ? r4(filtAvg - baseAvg) : null,
      });
    }
  }

  const improvements = grid
    .map(p => p.improvementVsBaseline)
    .filter((v): v is number => v !== null);
  const posCount = improvements.filter(i => i > 0).length;
  const total = improvements.length;
  const stableCount = posCount;
  const fragileCount = total - posCount;

  let sensitivity: ThresholdSensitivity;
  let explanation: string;
  if (total === 0) {
    sensitivity = "fragile";
    explanation = "Nenhum ponto da grid produziu improvement calculável.";
  } else if (posCount / total >= 0.7) {
    sensitivity = "stable";
    explanation = `${posCount}/${total} pontos com improvement > 0 — resultado robusto a variações de threshold.`;
  } else if (posCount / total >= 0.4) {
    sensitivity = "fragile";
    explanation = `${posCount}/${total} pontos com improvement > 0 — resultado sensível a escolha de threshold.`;
  } else {
    sensitivity = "highly_threshold_dependent";
    explanation = `Apenas ${posCount}/${total} pontos com improvement > 0 — resultado depende criticamente dos thresholds escolhidos.`;
  }

  return { grid, sensitivity, stableCount, fragileCount, explanation };
}

// ---------------------------------------------------------------------------
// Discard rule effectiveness
// ---------------------------------------------------------------------------

function discardEffectiveness(
  events: readonly MomentumEvent[],
): OperationalizationRobustnessAssessment["discardRuleEffectivenessSummary"] {
  const rules: Array<{ ruleId: string; test: (e: MomentumEvent) => boolean }> = [
    { ruleId: "discard_low_magnitude", test: e => e.magnitude < 0.003 },
    { ruleId: "discard_negative_proxy", test: e => e.conservativeCaptureProxy <= -0.003 },
  ];

  const mktCounts: Record<string, number> = {};
  for (const e of events) mktCounts[e.marketId] = (mktCounts[e.marketId] ?? 0) + 1;
  const repeating = new Set(Object.entries(mktCounts).filter(([, c]) => c >= 2).map(([k]) => k));
  if (repeating.size > 0 && events.length >= 8) {
    rules.push({ ruleId: "discard_non_repeating_markets", test: e => !repeating.has(e.marketId) });
  }

  return rules.map(r => {
    const discarded = events.filter(r.test);
    const avgP = avg(discarded.map(e => e.conservativeCaptureProxy));
    let label: string;
    if (discarded.length === 0) label = "no_impact";
    else if (avgP !== null && avgP < -0.001) label = "effective_removes_bad";
    else if (avgP !== null && avgP > 0.001) label = "counterproductive_removes_good";
    else label = "marginal";
    return {
      ruleId: r.ruleId,
      totalDiscarded: discarded.length,
      avgProxyDiscarded: avgP,
      effectivenessLabel: label,
    };
  });
}

// ---------------------------------------------------------------------------
// Dominant reasons
// ---------------------------------------------------------------------------

function dominantPassReasons(events: readonly MomentumEvent[], magFloor: number, proxyNegCutoff: number): string[] {
  const { passing } = applyConservativeRule(events, magFloor, proxyNegCutoff);
  if (passing.length === 0) return ["Nenhum evento passa o filtro conservador."];
  const reasons: string[] = [];
  const types: Record<string, number> = {};
  for (const e of passing) types[e.eventType] = (types[e.eventType] ?? 0) + 1;
  const sorted = Object.entries(types).sort((a, b) => b[1] - a[1]);
  if (sorted.length > 0) {
    reasons.push(`Tipo dominante: ${sorted[0]![0]} (${sorted[0]![1]}/${passing.length}).`);
  }
  const avgMag = avg(passing.map(e => e.magnitude));
  if (avgMag !== null) reasons.push(`Magnitude média dos aceitos: ${avgMag}.`);
  const avgProxy = avg(passing.map(e => e.conservativeCaptureProxy));
  if (avgProxy !== null) reasons.push(`CaptureProxy médio dos aceitos: ${avgProxy}.`);
  return reasons;
}

// ---------------------------------------------------------------------------
// Stability score & persistence
// ---------------------------------------------------------------------------

function stabilityScore(windows: WindowMetrics[]): number | null {
  const imps = windows.map(w => w.improvementVsBaseline).filter((v): v is number => v !== null);
  if (imps.length < 2) return null;
  const mu = imps.reduce((a, b) => a + b, 0) / imps.length;
  if (Math.abs(mu) < 1e-8) return 0;
  const variance = imps.reduce((a, v) => a + (v - mu) ** 2, 0) / imps.length;
  const cv = Math.sqrt(variance) / Math.abs(mu);
  return r4(Math.max(0, Math.min(1, 1 - cv)));
}

function improvementPersistence(windows: WindowMetrics[]): number | null {
  const imps = windows.map(w => w.improvementVsBaseline).filter((v): v is number => v !== null);
  if (imps.length === 0) return null;
  return r4(imps.filter(i => i > 0).length / imps.length);
}

// ---------------------------------------------------------------------------
// Overfit risk
// ---------------------------------------------------------------------------

function classifyOverfitRisk(
  allWindows: WindowMetrics[],
  sensitivity: ThresholdSensitivity,
  totalEvents: number,
  opsAssessment: OperationalizationAssessment,
): { risk: OverfitRisk; factors: string[] } {
  const factors: string[] = [];
  let score = 0;

  const best = opsAssessment.bestOperationalRuleSet;
  if (best && best.concentrationRiskFiltered > 0.5) {
    factors.push(`Concentração alta no rule set vencedor (${r4(best.concentrationRiskFiltered * 100)}%).`);
    score += 2;
  }
  if (best && best.distinctMarketsFiltered < 3) {
    factors.push(`Poucos mercados distintos no filtrado (${best.distinctMarketsFiltered}).`);
    score += 2;
  }
  if (best && best.eventsAfterFilter < 5) {
    factors.push(`Poucos eventos passam o filtro (${best.eventsAfterFilter}).`);
    score += 1;
  }

  const imps = allWindows.map(w => w.improvementVsBaseline).filter((v): v is number => v !== null);
  const negCount = imps.filter(i => i <= 0).length;
  if (imps.length >= 3 && negCount / imps.length > 0.5) {
    factors.push(`Maioria das janelas (${negCount}/${imps.length}) sem improvement positivo.`);
    score += 2;
  }

  if (sensitivity === "highly_threshold_dependent") {
    factors.push("Resultado altamente dependente dos thresholds escolhidos.");
    score += 2;
  } else if (sensitivity === "fragile") {
    factors.push("Resultado sensível a thresholds.");
    score += 1;
  }

  if (totalEvents < 15) {
    factors.push(`Amostra total pequena (${totalEvents}).`);
    score += 1;
  }

  if (score >= 4) return { risk: "high", factors };
  if (score >= 2) return { risk: "moderate", factors };
  return { risk: "low", factors };
}

// ---------------------------------------------------------------------------
// Main builder
// ---------------------------------------------------------------------------

export function buildOperationalizationRobustness(
  allEvents: readonly MomentumEvent[],
  opsAssessment: OperationalizationAssessment,
): OperationalizationRobustnessAssessment {
  const minEv = MIN_EVENTS_FOR_ROBUSTNESS();
  const total = allEvents.length;
  const hasFullRobustness = total >= minEv;

  const best = opsAssessment.bestOperationalRuleSet;
  const mags = allEvents.map(e => e.magnitude).sort((a, b) => a - b);
  const p25 = mags.length >= 4 ? mags[Math.floor(mags.length * 0.25)]! : (mags[Math.floor(mags.length / 2)] ?? 0.005);
  const magFloor = r4(Math.max(p25, 0.005));
  const proxyNegCutoff = -0.003;

  const cumulative = hasFullRobustness ? buildCumulativeGrowthWindows(allEvents, magFloor, proxyNegCutoff) : [];
  const recent = hasFullRobustness ? buildFixedRecentWindows(allEvents, magFloor, proxyNegCutoff) : [];
  const buckets = hasFullRobustness ? buildEqualBucketWindows(allEvents, magFloor, proxyNegCutoff) : [];
  const allWindows = [...cumulative, ...recent, ...buckets];

  const sensResult = hasFullRobustness
    ? buildSensitivityGrid(allEvents)
    : { grid: [], sensitivity: "fragile" as ThresholdSensitivity, stableCount: 0, fragileCount: 0, explanation: "Amostra insuficiente para grid de sensibilidade." };

  const stabScore = stabilityScore(allWindows);
  const impPersist = improvementPersistence(allWindows);

  const { risk: overfitRisk, factors: overfitFactors } = hasFullRobustness
    ? classifyOverfitRisk(allWindows, sensResult.sensitivity, total, opsAssessment)
    : { risk: "moderate" as OverfitRisk, factors: [`Amostra insuficiente (${total} < ${minEv}).`] };

  const domPass = hasFullRobustness
    ? dominantPassReasons(allEvents, magFloor, proxyNegCutoff)
    : ["Amostra insuficiente."];
  const domExit = [
    "exit_recovery_observed: spread recovery no mesmo mercado.",
    "exit_deterioration: proxy deteriora para negativo.",
    "exit_timeout: sem melhoria em 3 scans consecutivos.",
  ];
  const discardEff = hasFullRobustness ? discardEffectiveness(allEvents) : [];

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];
  let verdict: OpsRobustnessVerdict;

  if (!hasFullRobustness) {
    verdict = "insufficient_sample_for_full_robustness";
    blockingReasons.push(`Eventos ${total} < mínimo ${minEv} para robustez completa.`);
  } else if (!best) {
    verdict = "unstable";
    blockingReasons.push("Sem rule set vencedor no operationalization assessment.");
  } else {
    const persistence = impPersist ?? 0;
    const stability = stabScore ?? 0;
    const sens = sensResult.sensitivity;

    if (persistence >= 0.8 && stability >= 0.6 && sens === "stable" && overfitRisk === "low") {
      verdict = "robust";
      supportingReasons.push(
        `Persistência ${persistence}, estabilidade ${stability}, sensibilidade ${sens}, overfit ${overfitRisk}.`,
      );
    } else if (persistence >= 0.6 && stability >= 0.4 && overfitRisk !== "high") {
      verdict = "stable";
      supportingReasons.push(
        `Persistência ${persistence}, estabilidade ${stability}, overfit ${overfitRisk}.`,
      );
    } else if (persistence >= 0.4 && overfitRisk !== "high") {
      verdict = "weak_but_persistent";
      supportingReasons.push(
        `Persistência ${persistence} — sinal fraco mas presente em múltiplas janelas.`,
      );
    } else if (sens === "highly_threshold_dependent" || overfitRisk === "high") {
      verdict = "fragile";
      blockingReasons.push(
        `Sensibilidade=${sens}, overfit=${overfitRisk} — resultado depende demais de parametrização ou amostra.`,
      );
    } else {
      verdict = "unstable";
      blockingReasons.push(
        `Persistência ${persistence}, estabilidade ${stability} — sinal não se mantém entre janelas.`,
      );
    }
  }

  return {
    totalEventsForRobustness: total,
    minimumRequired: minEv,
    hasFullRobustness,
    robustnessVerdict: verdict,
    bestRuleSetStabilityScore: stabScore,
    improvementPersistenceRate: impPersist,
    rollingWindowImprovementSeries: allWindows.map(w => ({
      ...w,
      avgProxyFiltered: w.avgProxyFiltered,
    })),
    rollingWindowCoverageSeries: allWindows.map(w => ({
      ...w,
    })),
    rollingWindowConcentrationSeries: allWindows.map(w => ({
      ...w,
    })),
    cumulativeGrowthWindows: cumulative,
    fixedRecentWindows: recent,
    equalBucketWindows: buckets,
    thresholdSensitivitySummary: sensResult,
    dominantPassReasons: domPass,
    dominantExitReasons: domExit,
    discardRuleEffectivenessSummary: discardEff,
    overfitRiskVerdict: overfitRisk,
    overfitRiskFactors: overfitFactors,
    supportingReasons,
    blockingReasons,
    thresholdsUsed: {
      MOMENTUM_OPS_ROBUSTNESS_MIN_EVENTS: minEv,
      MIN_WINDOW_SIZE,
      CONSERVATIVE_MAG_FLOOR: magFloor,
      CONSERVATIVE_PROXY_NEG_CUTOFF: proxyNegCutoff,
      SENSITIVITY_MAG_FLOORS: 3,
      SENSITIVITY_PROXY_CUTOFFS: 3,
    },
    readDisclaimer:
      "Avaliação de robustez in-sample sobre rule set conservador. Não é backtest nem validação out-of-sample. Qualquer rule set precisa de paper testing real antes de uso com capital.",
  };
}

export function buildOpsRobustnessSummaryLine(
  a: OperationalizationRobustnessAssessment,
): string {
  if (!a.hasFullRobustness) {
    return `opsRobust: insufficient_sample (${a.totalEventsForRobustness}/${a.minimumRequired})`;
  }
  const stab = a.bestRuleSetStabilityScore !== null ? String(a.bestRuleSetStabilityScore) : "n/a";
  const persist = a.improvementPersistenceRate !== null ? String(a.improvementPersistenceRate) : "n/a";
  return `opsRobust: ${a.robustnessVerdict} | stab=${stab} persist=${persist} sens=${a.thresholdSensitivitySummary.sensitivity} overfit=${a.overfitRiskVerdict}`;
}
