/**
 * Operationalization Assessment — transforma sinal observacional do momentum probe
 * em candidate rule sets auditáveis (entry/exit/discard). Compara até 3 rule sets
 * simples entre si e avalia cobertura, melhoria e concentração. Não executa trades.
 */

import type { MomentumEvent, MomentumEventType } from "./momentumSnipingProbe";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}
function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

const MIN_EVENTS_FOR_OPS = () =>
  Math.max(8, Math.floor(envNum("MOMENTUM_OPS_MIN_EVENTS", 12)));
const MAX_RULE_SETS = 3;

export type OperationalizationVerdict =
  | "insufficient_sample"
  | "no_clear_operational_rule"
  | "weak_operational_rule"
  | "promising_operational_rule"
  | "overfit_risk";

interface OperationalRule {
  ruleId: string;
  description: string;
  rationale: string;
}

interface RuleSetEvaluation {
  ruleSetId: string;
  ruleSetLabel: string;
  entryRules: OperationalRule[];
  exitRules: OperationalRule[];
  discardRules: OperationalRule[];
  eventsPassingEntry: number;
  eventsDiscarded: number;
  eventsAfterFilter: number;
  coverage: number;
  avgProxyFiltered: number | null;
  avgProxyDiscarded: number | null;
  capturableRateFiltered: number;
  concentrationRiskFiltered: number;
  improvementVsBaseline: number | null;
  distinctMarketsFiltered: number;
}

export interface OperationalizationAssessment {
  totalEventsEligibleForOperationalization: number;
  candidateEntryRules: OperationalRule[];
  candidateExitRules: OperationalRule[];
  candidateDiscardRules: OperationalRule[];
  entryRuleComparisons: Array<{ ruleId: string; passing: number; avgProxy: number | null }>;
  exitRuleComparisons: Array<{ ruleId: string; description: string }>;
  discardRuleComparisons: Array<{ ruleId: string; discarded: number; avgProxyDiscarded: number | null }>;
  bestOperationalRuleSet: RuleSetEvaluation | null;
  bestOperationalRuleSetRationale: string;
  ruleSetCoverage: number | null;
  ruleSetConcentrationRisk: number | null;
  ruleSetEstimatedImprovementVsBaseline: number | null;
  ruleSetEstimatedImprovementVsTopDecile: number | null;
  ruleSetStabilityRead: string;
  operationalizationVerdict: OperationalizationVerdict;
  supportingReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: Record<string, number>;
  readDisclaimer: string;
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
  const c: Record<string, number> = {};
  for (const e of events) c[e.marketId] = (c[e.marketId] ?? 0) + 1;
  return r4(Math.max(...Object.values(c)) / events.length);
}

function distinctMarkets(events: readonly MomentumEvent[]): number {
  const s = new Set<string>();
  for (const e of events) s.add(e.marketId);
  return s.size;
}

function medianMagnitude(events: readonly MomentumEvent[]): number {
  if (events.length === 0) return 0;
  const s = events.map(e => e.magnitude).sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

function deriveEntryRules(all: readonly MomentumEvent[]): OperationalRule[] {
  const rules: OperationalRule[] = [];
  const mags = all.map(e => e.magnitude).sort((a, b) => a - b);
  const medMag = mags.length > 0 ? mags[Math.floor(mags.length / 2)]! : 0;
  const p25Mag = mags.length >= 4 ? mags[Math.floor(mags.length * 0.25)]! : medMag;

  const typeCounts: Record<string, { total: number; capturable: number }> = {};
  for (const e of all) {
    const t = typeCounts[e.eventType] ??= { total: 0, capturable: 0 };
    t.total++;
    if (e.capturable) t.capturable++;
  }
  const bestTypes = Object.entries(typeCounts)
    .filter(([, v]) => v.total >= 3 && v.capturable / v.total > 0)
    .sort((a, b) => b[1].capturable / b[1].total - a[1].capturable / a[1].total)
    .map(([k]) => k);

  if (bestTypes.length > 0 && bestTypes.length < Object.keys(typeCounts).length) {
    rules.push({
      ruleId: "entry_prefer_types",
      description: `Aceitar eventTypes: ${bestTypes.join(", ")}`,
      rationale: `Tipos com maior capturable rate observada (≥3 ocorrências e rate>0).`,
    });
  }

  const magThreshold = r4(Math.max(p25Mag, 0.005));
  rules.push({
    ruleId: "entry_min_magnitude",
    description: `Magnitude mínima ≥ ${magThreshold}`,
    rationale: `Baseado em p25 das magnitudes observadas (${r4(p25Mag)}), com piso de 0.005. Eventos abaixo tendem a não cobrir fee.`,
  });

  rules.push({
    ruleId: "entry_min_capture_proxy",
    description: "captureProxy > 0 (capturable=true)",
    rationale: "Filtro básico: só considerar eventos onde proxy já supera fee estimada.",
  });

  return rules;
}

function deriveExitRules(): OperationalRule[] {
  return [
    {
      ruleId: "exit_recovery_observed",
      description: "Sair se spread_recovery detectado no mesmo mercado",
      rationale: "Recovery indica que a janela de oportunidade fechou; continuar exposto é risco.",
    },
    {
      ruleId: "exit_deterioration",
      description: "Sair se captureProxy deteriora para < 0 em snapshot seguinte",
      rationale: "Proxy negativo no follow-up invalida o sinal de entrada.",
    },
    {
      ruleId: "exit_timeout",
      description: "Timeout observacional: 3 scans sem melhoria → descartar",
      rationale: "Eventos que não evoluem em ~30s (3×10s) provavelmente são ruído absorvido.",
    },
  ];
}

function deriveDiscardRules(all: readonly MomentumEvent[]): OperationalRule[] {
  const rules: OperationalRule[] = [];

  rules.push({
    ruleId: "discard_excessive_concentration",
    description: "Descartar se mercado representa >40% dos eventos filtrados",
    rationale: "Concentração excessiva indica artefato de um único mercado, não padrão sistemático.",
  });

  rules.push({
    ruleId: "discard_low_magnitude",
    description: "Descartar magnitude < 0.003",
    rationale: "Movimentos < 0.3% são microrruído indistinguível de latência/jitter da API.",
  });

  rules.push({
    ruleId: "discard_negative_proxy",
    description: "Descartar captureProxy ≤ -0.003",
    rationale: "Proxy muito negativo — custo supera magnitude observada por margem significativa.",
  });

  const mktCounts: Record<string, number> = {};
  for (const e of all) mktCounts[e.marketId] = (mktCounts[e.marketId] ?? 0) + 1;
  const singletons = Object.values(mktCounts).filter(c => c === 1).length;
  if (singletons > 0 && all.length >= 8) {
    rules.push({
      ruleId: "discard_non_repeating_markets",
      description: "Descartar eventos de mercados com apenas 1 ocorrência",
      rationale: `${singletons} mercados com evento único — sem repetição, sem confiança.`,
    });
  }

  return rules;
}

function evaluateRuleSet(
  setId: string,
  label: string,
  all: readonly MomentumEvent[],
  entryFilter: (e: MomentumEvent) => boolean,
  discardFilter: (e: MomentumEvent) => boolean,
  entryRules: OperationalRule[],
  exitRules: OperationalRule[],
  discardRules: OperationalRule[],
  baselineAvg: number | null,
): RuleSetEvaluation {
  const passing = all.filter(e => entryFilter(e) && !discardFilter(e));
  const discarded = all.filter(e => !entryFilter(e) || discardFilter(e));

  return {
    ruleSetId: setId,
    ruleSetLabel: label,
    entryRules,
    exitRules,
    discardRules,
    eventsPassingEntry: all.filter(entryFilter).length,
    eventsDiscarded: discarded.length,
    eventsAfterFilter: passing.length,
    coverage: all.length > 0 ? r4(passing.length / all.length) : 0,
    avgProxyFiltered: avg(passing.map(e => e.conservativeCaptureProxy)),
    avgProxyDiscarded: avg(discarded.map(e => e.conservativeCaptureProxy)),
    capturableRateFiltered: capRate(passing),
    concentrationRiskFiltered: concRisk(passing),
    improvementVsBaseline:
      baselineAvg !== null && passing.length > 0
        ? r4((avg(passing.map(e => e.conservativeCaptureProxy)) ?? 0) - baselineAvg)
        : null,
    distinctMarketsFiltered: distinctMarkets(passing),
  };
}

export function buildOperationalizationAssessment(
  allEvents: readonly MomentumEvent[],
): OperationalizationAssessment {
  const minEv = MIN_EVENTS_FOR_OPS();
  const total = allEvents.length;
  const baselineAvg = avg(allEvents.map(e => e.conservativeCaptureProxy));

  const topDecileN = Math.max(1, Math.floor(total * 0.1));
  const ranked = [...allEvents].sort(
    (a, b) =>
      b.conservativeCaptureProxy + b.magnitude * 0.15 -
      (a.conservativeCaptureProxy + a.magnitude * 0.15),
  );
  const topDecile = ranked.slice(0, topDecileN);
  const topDecileAvg = avg(topDecile.map(e => e.conservativeCaptureProxy));

  const entryRules = total >= minEv ? deriveEntryRules(allEvents) : [];
  const exitRules = deriveExitRules();
  const discardRules = total >= minEv ? deriveDiscardRules(allEvents) : [];

  const medMag = medianMagnitude(allEvents);
  const p25Mag = total >= 4
    ? allEvents.map(e => e.magnitude).sort((a, b) => a - b)[Math.floor(total * 0.25)]!
    : medMag;
  const magThreshold = r4(Math.max(p25Mag, 0.005));

  const typeCounts: Record<string, { total: number; capturable: number }> = {};
  for (const e of allEvents) {
    const t = typeCounts[e.eventType] ??= { total: 0, capturable: 0 };
    t.total++;
    if (e.capturable) t.capturable++;
  }
  const bestTypes = new Set(
    Object.entries(typeCounts)
      .filter(([, v]) => v.total >= 3 && v.capturable / v.total > 0)
      .map(([k]) => k),
  );
  const hasTypeFilter = bestTypes.size > 0 && bestTypes.size < Object.keys(typeCounts).length;

  const mktCounts: Record<string, number> = {};
  for (const e of allEvents) mktCounts[e.marketId] = (mktCounts[e.marketId] ?? 0) + 1;
  const repeatingMkts = new Set(
    Object.entries(mktCounts).filter(([, c]) => c >= 2).map(([k]) => k),
  );

  const conservativeEntry = (e: MomentumEvent) =>
    e.capturable && e.magnitude >= magThreshold;
  const conservativeDiscard = (e: MomentumEvent) =>
    e.magnitude < 0.003 || e.conservativeCaptureProxy <= -0.003;

  const moderateEntry = (e: MomentumEvent) =>
    e.capturable &&
    e.magnitude >= magThreshold &&
    (!hasTypeFilter || bestTypes.has(e.eventType));
  const moderateDiscard = (e: MomentumEvent) =>
    conservativeDiscard(e) ||
    (total >= 8 && repeatingMkts.size > 0 && !repeatingMkts.has(e.marketId));

  const aggressiveEntry = (e: MomentumEvent) =>
    e.capturable &&
    e.magnitude >= magThreshold * 1.5 &&
    e.liquidityAtDetection >= 1000;
  const aggressiveDiscard = (e: MomentumEvent) =>
    moderateDiscard(e);

  const sets: RuleSetEvaluation[] = [];

  if (total >= minEv) {
    sets.push(
      evaluateRuleSet(
        "conservative", "Conservador: capturable + magnitude≥p25 − ruído",
        allEvents, conservativeEntry, conservativeDiscard,
        entryRules.filter(r => r.ruleId !== "entry_prefer_types"),
        exitRules,
        discardRules.filter(r => r.ruleId !== "discard_non_repeating_markets"),
        baselineAvg,
      ),
    );
    sets.push(
      evaluateRuleSet(
        "moderate", "Moderado: + tipo preferido + mercados repetidos",
        allEvents, moderateEntry, moderateDiscard,
        entryRules,
        exitRules,
        discardRules,
        baselineAvg,
      ),
    );
    sets.push(
      evaluateRuleSet(
        "selective", "Seletivo: magnitude alta + liquidez≥1k + filtros completos",
        allEvents, aggressiveEntry, aggressiveDiscard,
        [...entryRules, {
          ruleId: "entry_min_liquidity",
          description: "Liquidez ≥ 1000",
          rationale: "Mercados com menos de $1k de liquidez têm fill incerto.",
        }],
        exitRules,
        discardRules,
        baselineAvg,
      ),
    );
  }

  const entryComparisons = entryRules.map(r => {
    let fn: (e: MomentumEvent) => boolean;
    if (r.ruleId === "entry_prefer_types") fn = e => bestTypes.has(e.eventType);
    else if (r.ruleId === "entry_min_magnitude") fn = e => e.magnitude >= magThreshold;
    else fn = e => e.capturable;
    const passing = allEvents.filter(fn);
    return { ruleId: r.ruleId, passing: passing.length, avgProxy: avg(passing.map(e => e.conservativeCaptureProxy)) };
  });

  const discardComparisons = discardRules.map(r => {
    let fn: (e: MomentumEvent) => boolean;
    if (r.ruleId === "discard_low_magnitude") fn = e => e.magnitude < 0.003;
    else if (r.ruleId === "discard_negative_proxy") fn = e => e.conservativeCaptureProxy <= -0.003;
    else if (r.ruleId === "discard_non_repeating_markets") fn = e => !repeatingMkts.has(e.marketId);
    else fn = () => false;
    const discarded = allEvents.filter(fn);
    return { ruleId: r.ruleId, discarded: discarded.length, avgProxyDiscarded: avg(discarded.map(e => e.conservativeCaptureProxy)) };
  });

  const viable = sets.filter(s => s.eventsAfterFilter >= 2 && s.coverage >= 0.05);
  const best = viable.sort((a, b) => {
    const impA = a.improvementVsBaseline ?? -999;
    const impB = b.improvementVsBaseline ?? -999;
    const concPenA = a.concentrationRiskFiltered > 0.5 ? -0.01 : 0;
    const concPenB = b.concentrationRiskFiltered > 0.5 ? -0.01 : 0;
    return (impB + concPenB) - (impA + concPenA);
  })[0] ?? null;

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];
  let verdict: OperationalizationVerdict;

  if (total < minEv) {
    verdict = "insufficient_sample";
    blockingReasons.push(`Eventos ${total} < mínimo ${minEv}.`);
  } else if (!best) {
    verdict = "no_clear_operational_rule";
    blockingReasons.push("Nenhum rule set viável (cobertura ≥5% e ≥2 eventos após filtro).");
  } else {
    const imp = best.improvementVsBaseline ?? 0;
    const conc = best.concentrationRiskFiltered;
    const cov = best.coverage;

    if (cov < 0.05) {
      verdict = "overfit_risk";
      blockingReasons.push(`Melhor rule set cobre apenas ${r4(cov * 100)}%.`);
    } else if (imp <= 0) {
      verdict = "no_clear_operational_rule";
      blockingReasons.push(`Melhor rule set não melhora baseline (imp=${imp}).`);
    } else if (conc > 0.6 && best.distinctMarketsFiltered < 3) {
      verdict = "overfit_risk";
      blockingReasons.push(`Concentração ${r4(conc * 100)}% com apenas ${best.distinctMarketsFiltered} mercados — overfit provável.`);
    } else if (imp > 0.003 && conc <= 0.5 && cov >= 0.1 && best.capturableRateFiltered >= 0.5) {
      verdict = "promising_operational_rule";
      supportingReasons.push(
        `Rule set "${best.ruleSetLabel}": imp=${imp}, cov=${r4(cov * 100)}%, conc=${r4(conc * 100)}%, capRate=${best.capturableRateFiltered}, ${best.distinctMarketsFiltered} mercados.`,
      );
    } else if (imp > 0) {
      verdict = "weak_operational_rule";
      supportingReasons.push(
        `Rule set "${best.ruleSetLabel}": imp=${imp} positivo mas fraco. cov=${r4(cov * 100)}%, conc=${r4(conc * 100)}%.`,
      );
    } else {
      verdict = "no_clear_operational_rule";
      blockingReasons.push("Nenhum rule set produz melhoria convincente.");
    }
  }

  const improvVsTopDecile =
    best && topDecileAvg !== null && best.avgProxyFiltered !== null
      ? r4(best.avgProxyFiltered - topDecileAvg)
      : null;

  let stabilityRead: string;
  if (!best) {
    stabilityRead = "Sem rule set viável para avaliar estabilidade.";
  } else if (best.eventsAfterFilter < 5) {
    stabilityRead = `Amostra filtrada pequena (${best.eventsAfterFilter}) — estabilidade não avaliável.`;
  } else if (best.concentrationRiskFiltered > 0.5) {
    stabilityRead = "Concentração alta no filtrado — estabilidade comprometida por dependência de poucos mercados.";
  } else {
    stabilityRead = `Amostra filtrada razoável (${best.eventsAfterFilter}, ${best.distinctMarketsFiltered} mercados) — monitorar persistência em janelas futuras.`;
  }

  let bestRationale: string;
  if (!best) {
    bestRationale = "Nenhum rule set viável derivado.";
  } else {
    bestRationale = `"${best.ruleSetLabel}" selecionado por melhor combinação de melhoria vs baseline (${best.improvementVsBaseline}), cobertura (${r4(best.coverage * 100)}%) e concentração controlada (${r4(best.concentrationRiskFiltered * 100)}%).`;
  }

  return {
    totalEventsEligibleForOperationalization: total,
    candidateEntryRules: entryRules,
    candidateExitRules: exitRules,
    candidateDiscardRules: discardRules,
    entryRuleComparisons: entryComparisons,
    exitRuleComparisons: exitRules.map(r => ({ ruleId: r.ruleId, description: r.description })),
    discardRuleComparisons: discardComparisons,
    bestOperationalRuleSet: best,
    bestOperationalRuleSetRationale: bestRationale,
    ruleSetCoverage: best?.coverage ?? null,
    ruleSetConcentrationRisk: best?.concentrationRiskFiltered ?? null,
    ruleSetEstimatedImprovementVsBaseline: best?.improvementVsBaseline ?? null,
    ruleSetEstimatedImprovementVsTopDecile: improvVsTopDecile,
    ruleSetStabilityRead: stabilityRead,
    operationalizationVerdict: verdict,
    supportingReasons,
    blockingReasons,
    thresholdsUsed: {
      MOMENTUM_OPS_MIN_EVENTS: minEv,
      MAX_RULE_SETS,
    },
    readDisclaimer:
      "Regras operacionais observacionais derivadas in-sample. Não são recomendações de trading. Qualquer rule set precisa de validação out-of-sample e paper testing antes de uso com capital real.",
  };
}

export function buildOperationalizationSummaryLine(a: OperationalizationAssessment): string {
  if (a.totalEventsEligibleForOperationalization < MIN_EVENTS_FOR_OPS()) {
    return `ops: insufficient_sample (${a.totalEventsEligibleForOperationalization} events)`;
  }
  const best = a.bestOperationalRuleSet;
  if (!best) return `ops: ${a.operationalizationVerdict} | no viable ruleset`;
  const imp = a.ruleSetEstimatedImprovementVsBaseline;
  const impS = imp !== null ? (imp > 0 ? "+" : "") + String(imp) : "n/a";
  return `ops: ${a.operationalizationVerdict} | best="${best.ruleSetId}" imp=${impS} cov=${r4(best.coverage * 100)}% conc=${r4(best.concentrationRiskFiltered * 100)}% mkts=${best.distinctMarketsFiltered}`;
}
