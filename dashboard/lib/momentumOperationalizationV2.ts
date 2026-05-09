/**
 * Operationalization Assessment V2 — regra operacional conservadora e auditável
 * derivada do conjunto ranking + robustez + seleção do top slice.
 * Observacional apenas; não executa ordens; evita empilhar filtros e penaliza combinações estreitas.
 */

import type { MomentumEvent, MomentumEventType } from "./momentumSnipingProbe";
import type { RankedEventAssessment } from "./momentumRankedEventAssessment";
import type { TopSliceRobustnessAssessment } from "./momentumTopSliceRobustness";
import type { TopSliceSelectionAssessment } from "./momentumTopSliceSelection";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}
function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

const MIN_EVENTS = () =>
  Math.max(8, Math.floor(envNum("MOMENTUM_OPS_V2_MIN_EVENTS", 12)));

export type OperationalizationV2Verdict =
  | "insufficient_sample"
  | "weak_candidate_rule"
  | "promising_operational_rule"
  | "overfit_prone_rule";

export type RuleSimplicityVerdict = "simple" | "moderate" | "complex";
export type OverfitRiskVerdict = "low" | "medium" | "high";

export interface OperationalizationAssessmentV2 {
  operationalizationVerdict: OperationalizationV2Verdict;
  candidateRuleSetName: string;
  candidateEntryRules: string[];
  candidateExitRules: string[];
  candidateDiscardRules: string[];
  coverageEstimate: number | null;
  concentrationRiskEstimate: number | null;
  improvementVsBaselineEstimate: number | null;
  improvementVsTopDecileEstimate: number | null;
  ruleSimplicityVerdict: RuleSimplicityVerdict;
  overfitRiskVerdict: OverfitRiskVerdict;
  operationalizationReasons: string[];
  operationalizationRisks: string[];
  operationalizationSummaryLine: string;
  thresholdsUsed: Record<string, number>;
  readDisclaimer: string;
}

function avg(nums: number[]): number | null {
  if (nums.length === 0) return null;
  return r4(nums.reduce((a, b) => a + b, 0) / nums.length);
}

function concRisk(events: readonly MomentumEvent[]): number {
  if (events.length === 0) return 0;
  const c: Record<string, number> = {};
  for (const e of events) c[e.marketId] = (c[e.marketId] ?? 0) + 1;
  return r4(Math.max(...Object.values(c)) / events.length);
}

function liqMidBand(e: MomentumEvent): boolean {
  const L = e.liquidityAtDetection;
  return L >= 1_000 && L < 10_000;
}

function magAtLeast5pct(e: MomentumEvent): boolean {
  return e.magnitude >= 0.05;
}

function typeIs(e: MomentumEvent, t: MomentumEventType): boolean {
  return e.eventType === t;
}

type UnaryPredicate = { id: string; describe: string; fn: (e: MomentumEvent) => boolean };

function buildUnaryCandidates(selection: TopSliceSelectionAssessment): UnaryPredicate[] {
  const out: UnaryPredicate[] = [];
  const enrichedSpike = selection.topVsAllByEventType.some(
    b => b.bucket === "spread_spike" && b.enrichment >= 1.25 && b.countTop >= 2,
  );
  if (enrichedSpike) {
    out.push({
      id: "spread_spike",
      describe: "Entrada observacional: eventType = spread_spike (sobre-representado no top slice).",
      fn: e => typeIs(e, "spread_spike"),
    });
  }
  const liqOk = selection.topVsAllByLiquidityBucket.some(
    b =>
      b.bucket === "1k-10k" &&
      b.enrichment >= 1.15 &&
      b.countTop >= 2 &&
      (b.avgProxyTop ?? -1) > (b.avgProxyAll ?? -1),
  );
  if (liqOk) {
    out.push({
      id: "liquidity_1k_10k",
      describe: "Entrada observacional: liquidez em [1k, 10k) USD no momento da deteção.",
      fn: liqMidBand,
    });
  }
  const magOk = selection.topVsAllByMagnitudeBucket.some(
    b =>
      b.bucket === "≥5%" &&
      b.enrichment >= 1.15 &&
      b.countTop >= 2 &&
      (b.avgProxyTop ?? -1) > (b.avgProxyAll ?? -1),
  );
  if (magOk) {
    out.push({
      id: "magnitude_ge_5pct",
      describe: "Entrada observacional: magnitude ≥ 5% (deslocamento observado).",
      fn: magAtLeast5pct,
    });
  }
  return out;
}

function robustnessSupports(rob: TopSliceRobustnessAssessment): boolean {
  const v = rob.topSliceRobustnessVerdict;
  return (
    v === "improving_and_diversifying" ||
    v === "robust_top_slice_signal" ||
    v === "weak_but_persistent"
  );
}

function robustnessToxic(rob: TopSliceRobustnessAssessment): boolean {
  const v = rob.topSliceRobustnessVerdict;
  return v === "concentrated_artifact_risk" || v === "unstable";
}

export function buildOperationalizationAssessmentV2(
  events: readonly MomentumEvent[],
  ranked: RankedEventAssessment,
  robustness: TopSliceRobustnessAssessment,
  selection: TopSliceSelectionAssessment,
): OperationalizationAssessmentV2 {
  const minEv = MIN_EVENTS();
  const total = events.length;
  const baselineAvg = ranked.averageCapturableProxyAll ?? avg(events.map(e => e.conservativeCaptureProxy));
  const topDecileAvg =
    ranked.averageCapturableProxyTopDecile ??
    avg(
      [...events]
        .sort(
          (a, b) =>
            b.conservativeCaptureProxy +
            b.magnitude * 0.15 +
            (b.liquidityAtDetection >= 10_000 ? 0.001 : 0) -
            (a.conservativeCaptureProxy + a.magnitude * 0.15 + (a.liquidityAtDetection >= 10_000 ? 0.001 : 0)),
        )
        .slice(0, Math.max(1, Math.floor(total * 0.1)))
        .map(e => e.conservativeCaptureProxy),
    );

  const reasons: string[] = [];
  const risks: string[] = [];
  const readDisclaimer =
    "V2: operacionalização observacional in-sample a partir do top slice; não é execução nem promessa de PnL. Requer confirmação out-of-sample.";

  const empty: OperationalizationAssessmentV2 = {
    operationalizationVerdict: "insufficient_sample",
    candidateRuleSetName: "none",
    candidateEntryRules: [],
    candidateExitRules: [],
    candidateDiscardRules: [],
    coverageEstimate: null,
    concentrationRiskEstimate: null,
    improvementVsBaselineEstimate: null,
    improvementVsTopDecileEstimate: null,
    ruleSimplicityVerdict: "simple",
    overfitRiskVerdict: "high",
    operationalizationReasons: [],
    operationalizationRisks: [],
    operationalizationSummaryLine: "",
    thresholdsUsed: { MOMENTUM_OPS_V2_MIN_EVENTS: minEv },
    readDisclaimer,
  };

  if (total < minEv || ranked.eventsEligibleForRanking < minEv) {
    return {
      ...empty,
      operationalizationRisks: [
        `Amostra ${total} < ${minEv} ou ranking inelegível — não operacionalizar além de leitura exploratória.`,
      ],
      operationalizationSummaryLine: `opsV2: insufficient_sample (n=${total})`,
    };
  }

  const unaries = buildUnaryCandidates(selection);
  if (unaries.length === 0) {
    return {
      ...empty,
      operationalizationVerdict: "weak_candidate_rule",
      candidateRuleSetName: "microstructure_v2_no_unary_signal",
      candidateExitRules: defaultExitRules(),
      candidateDiscardRules: defaultDiscardRules(),
      operationalizationRisks: [
        "Nenhum ingrediente spread_spike / liquidez 1k-10k / magnitude ≥5% com suporte claro no topVsAll — não forçar regra.",
      ],
      operationalizationSummaryLine: "opsV2: weak_candidate_rule | no unary pattern from selection",
    };
  }

  interface Scored {
    pred: UnaryPredicate;
    matched: MomentumEvent[];
    improvement: number | null;
    coverage: number;
    conc: number;
  }

  const scored: Scored[] = [];
  for (const pred of unaries) {
    const matched = events.filter(pred.fn);
    const imp =
      baselineAvg !== null && matched.length > 0
        ? r4((avg(matched.map(e => e.conservativeCaptureProxy)) ?? 0) - baselineAvg)
        : null;
    scored.push({
      pred,
      matched,
      improvement: imp,
      coverage: total > 0 ? r4(matched.length / total) : 0,
      conc: concRisk(matched),
    });
  }

  scored.sort((a, b) => {
    const ia = a.improvement ?? -999;
    const ib = b.improvement ?? -999;
    if (ib !== ia) return ib - ia;
    return b.coverage - a.coverage;
  });

  const bestUnary = scored[0]!;
  let usePair = false;
  let predA = bestUnary.pred;
  let predB: UnaryPredicate | null = null;
  let ruleSimplicity: RuleSimplicityVerdict = "simple";
  let overfit: OverfitRiskVerdict = "low";

  const second = scored[1];
  if (
    second &&
    bestUnary.matched.length >= 3 &&
    (bestUnary.improvement ?? 0) < 0.002 &&
    selection.selectionAssessmentVerdict !== "overfit_risk"
  ) {
    const andRaw = events.filter(e => bestUnary.pred.fn(e) && second.pred.fn(e));
    if (andRaw.length >= 3) {
      const impAnd =
        baselineAvg !== null
          ? r4((avg(andRaw.map(e => e.conservativeCaptureProxy)) ?? 0) - baselineAvg)
          : null;
      const covAnd = total > 0 ? r4(andRaw.length / total) : 0;
      if (
        impAnd !== null &&
        impAnd > (bestUnary.improvement ?? -999) + 0.0005 &&
        covAnd >= 0.06
      ) {
        usePair = true;
        predB = second.pred;
        ruleSimplicity = "moderate";
        overfit = "medium";
        reasons.push(
          `Combinação AND conservadora: ${predA.id} ∧ ${predB.id} (melhor imp vs baseline que unário isolado).`,
        );
      }
    }
  }

  if (!usePair) {
    reasons.push(
      `Regra unária: ${predA.id} — maior melhoria vs baseline entre candidatos alinhados ao top slice.`,
    );
  }

  const passEntry = (e: MomentumEvent) =>
    usePair && predB ? predA.fn(e) && predB.fn(e) : predA.fn(e);
  const matched = events.filter(e => passEntry(e) && e.capturable);

  const entryRules: string[] =
    usePair && predB
      ? [
          `${predA.describe} E ${predB.describe.replace(/^Entrada observacional: /, "")}`,
          "Gate: capturable=true (captureProxy > 0 após fee proxy), coerente com o momentum probe.",
        ]
      : [
          predA.describe,
          "Gate: capturable=true (captureProxy > 0 após fee proxy), coerente com o momentum probe.",
        ];

  const finalImp =
    baselineAvg !== null && matched.length > 0
      ? r4((avg(matched.map(e => e.conservativeCaptureProxy)) ?? 0) - baselineAvg)
      : null;
  const finalCov = total > 0 ? r4(matched.length / total) : 0;
  const finalConc = matched.length > 0 ? concRisk(matched) : 0;
  const finalImpVsTop =
    matched.length > 0 && topDecileAvg !== null
      ? r4((avg(matched.map(e => e.conservativeCaptureProxy)) ?? 0) - topDecileAvg)
      : null;

  if (selection.selectionAssessmentVerdict === "overfit_risk") {
    overfit = "high";
    risks.push("Seleção top-slice reportou overfit_risk — regra V2 tratada como frágil.");
  }
  if (usePair) {
    overfit = overfit === "low" ? "medium" : overfit;
    risks.push("Duas condições AND reduzem cobertura e aumentam risco de especificidade à amostra.");
  }
  if (robustnessToxic(robustness)) {
    overfit = "high";
    risks.push(`Robustez temporal: ${robustness.topSliceRobustnessVerdict} — artefato ou instabilidade provável.`);
  }
  if (finalConc > 0.55) {
    overfit = overfit === "low" ? "medium" : overfit;
    risks.push(`Concentração estimada ${r4(finalConc * 100)}% no universo filtrado.`);
  }
  if (!ranked.topSliceShowsImprovement) {
    risks.push("Ranked: topSliceShowsImprovement=false — melhoria do topo vs média não está demonstrada nesta janela.");
  }

  let verdict: OperationalizationV2Verdict = "weak_candidate_rule";

  if (matched.length < 3 || finalCov < 0.06) {
    verdict = "insufficient_sample";
    risks.push("Menos de 3 eventos após gate capturable ou cobertura <6% — apenas candidato fraco.");
  } else if (
    overfit === "high" ||
    finalConc > 0.65 ||
    (selection.selectionAssessmentVerdict === "overfit_risk" && finalCov < 0.12)
  ) {
    verdict = "overfit_prone_rule";
  } else if (
    finalImp !== null &&
    finalImp >= 0.002 &&
    finalCov >= 0.1 &&
    finalConc <= 0.5 &&
    robustnessSupports(robustness) &&
    ranked.topSliceShowsImprovement &&
    (overfit === "low" || overfit === "medium")
  ) {
    verdict = "promising_operational_rule";
    reasons.push(
      `Melhoria vs baseline ${finalImp}, cobertura ${r4(finalCov * 100)}%, concentração ${r4(finalConc * 100)}%, robustez favorável.`,
    );
  } else if (finalImp !== null && finalImp > 0) {
    verdict = "weak_candidate_rule";
    reasons.push("Melhoria positiva mas abaixo dos limiares V2 para promoção a promissora.");
  } else {
    verdict = "weak_candidate_rule";
    risks.push("Melhoria vs baseline não positiva após gate capturable.");
  }

  const ruleSetName =
    verdict === "insufficient_sample"
      ? "microstructure_v2_insufficient"
      : `microstructure_v2_${usePair && predB ? `pair_${predA.id}_${predB.id}` : predA.id}`;

  const summary = `opsV2: ${verdict} | set=${ruleSetName} cov=${finalCov} conc=${finalConc} impΔ=${finalImp ?? "n/a"} impVsD10=${finalImpVsTop ?? "n/a"} simp=${ruleSimplicity} overfit=${overfit}`;

  return {
    operationalizationVerdict: verdict,
    candidateRuleSetName: ruleSetName,
    candidateEntryRules: entryRules,
    candidateExitRules: defaultExitRules(),
    candidateDiscardRules: defaultDiscardRules(),
    coverageEstimate: finalCov,
    concentrationRiskEstimate: finalConc,
    improvementVsBaselineEstimate: finalImp,
    improvementVsTopDecileEstimate: finalImpVsTop,
    ruleSimplicityVerdict: ruleSimplicity,
    overfitRiskVerdict: overfit,
    operationalizationReasons: reasons,
    operationalizationRisks: risks,
    operationalizationSummaryLine: summary,
    thresholdsUsed: { MOMENTUM_OPS_V2_MIN_EVENTS: minEv },
    readDisclaimer,
  };
}

function defaultExitRules(): string[] {
  return [
    "Saída observacional: spread_recovery no mesmo mercado ou compressão de spread abaixo do limiar de spike.",
    "Saída observacional: timeout ~30s (3× intervalo de scan) sem confirmação de proxy positivo em snapshot seguinte.",
    "Saída observacional: captureProxy em snapshot seguinte ≤ 0.",
  ];
}

function defaultDiscardRules(): string[] {
  return [
    "Descartar magnitude < 0.3% (ruído de cotação).",
    "Descartar captureProxy ≤ −0.003 (custo domina).",
    "Descartar se um único mercado > 45% dos eventos candidatos após filtro (concentração operacional inaceitável).",
  ];
}

export function buildOperationalizationV2SummaryLine(a: OperationalizationAssessmentV2): string {
  return a.operationalizationSummaryLine;
}
