/**
 * Operationalization Promotion Readiness — meta-evaluation layer.
 * Determines whether the winning conservative rule set has earned
 * a higher confidence tier as evidence accumulates. Not an alpha
 * discovery module; purely promotion discipline.
 */

import type { MomentumEvent } from "./momentumSnipingProbe";
import type { OperationalizationAssessment } from "./momentumOperationalization";
import type {
  OperationalizationRobustnessAssessment,
  OpsRobustnessVerdict,
  OverfitRisk,
  ThresholdSensitivity,
} from "./operationalizationRobustness";

function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

export type PromotionConfidenceTier =
  | "too_early"
  | "fragile"
  | "weak_but_persistent"
  | "stable"
  | "robust";

export type EvidenceBreadthVerdict =
  | "insufficient"
  | "narrow"
  | "moderate"
  | "broad";

export type MaintenanceVerdict = "failing" | "degrading" | "holding" | "improving";

interface PromotionGate {
  gateId: string;
  description: string;
  required: boolean;
  passed: boolean;
  currentValue: string;
  threshold: string;
}

export interface PromotionReadinessAssessment {
  promotionReadinessVerdict: PromotionConfidenceTier;
  currentConfidenceTier: PromotionConfidenceTier;
  passingEventCount: number;
  distinctPassingMarkets: number;
  evidenceBreadthVerdict: EvidenceBreadthVerdict;
  persistenceMaintenanceVerdict: MaintenanceVerdict;
  sensitivityMaintenanceVerdict: MaintenanceVerdict;
  concentrationMaintenanceVerdict: MaintenanceVerdict;
  minimumPromotionGates: PromotionGate[];
  gatesPassed: number;
  gatesFailed: number;
  promotionBlockers: string[];
  nextPromotionTarget: string;
  supportingReasons: string[];
  readDisclaimer: string;
}

function applyConservativeFilter(
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

function evidenceBreadth(
  passingCount: number,
  distinctMarkets: number,
): EvidenceBreadthVerdict {
  if (passingCount < 3 || distinctMarkets < 2) return "insufficient";
  if (passingCount < 5 || distinctMarkets < 3) return "narrow";
  if (passingCount < 10 || distinctMarkets < 5) return "moderate";
  return "broad";
}

function persistenceMaintenance(
  robustness: OperationalizationRobustnessAssessment,
): MaintenanceVerdict {
  const p = robustness.improvementPersistenceRate;
  if (p === null) return "failing";
  if (p >= 0.8) return "improving";
  if (p >= 0.6) return "holding";
  if (p >= 0.3) return "degrading";
  return "failing";
}

function sensitivityMaintenance(
  sensitivity: ThresholdSensitivity,
): MaintenanceVerdict {
  if (sensitivity === "stable") return "improving";
  if (sensitivity === "fragile") return "degrading";
  return "failing";
}

function concentrationMaintenance(
  robustness: OperationalizationRobustnessAssessment,
  ops: OperationalizationAssessment,
): MaintenanceVerdict {
  const best = ops.bestOperationalRuleSet;
  if (!best) return "failing";
  const conc = best.concentrationRiskFiltered;
  if (conc <= 0.3) return "improving";
  if (conc <= 0.5) return "holding";
  if (conc <= 0.7) return "degrading";
  return "failing";
}

export function buildPromotionReadiness(
  allEvents: readonly MomentumEvent[],
  ops: OperationalizationAssessment,
  robustness: OperationalizationRobustnessAssessment,
): PromotionReadinessAssessment {
  const best = ops.bestOperationalRuleSet;

  const mags = allEvents.map(e => e.magnitude).sort((a, b) => a - b);
  const p25 = mags.length >= 4
    ? mags[Math.floor(mags.length * 0.25)]!
    : (mags[Math.floor(mags.length / 2)] ?? 0.005);
  const magFloor = r4(Math.max(p25, 0.005));

  const passing = applyConservativeFilter(allEvents, magFloor);
  const passingCount = passing.length;
  const passingMkts = new Set(passing.map(e => e.marketId));
  const distinctMarkets = passingMkts.size;

  const evBreadth = evidenceBreadth(passingCount, distinctMarkets);
  const persistMaint = persistenceMaintenance(robustness);
  const sensMaint = sensitivityMaintenance(robustness.thresholdSensitivitySummary.sensitivity);
  const concMaint = concentrationMaintenance(robustness, ops);

  const improvPositive = (robustness.improvementPersistenceRate ?? 0) > 0;
  const impVsBaseline = best?.improvementVsBaseline ?? null;

  const gates: PromotionGate[] = [
    {
      gateId: "min_passing_events",
      description: "Mínimo 3 eventos passam o filtro conservador",
      required: true,
      passed: passingCount >= 3,
      currentValue: String(passingCount),
      threshold: "≥ 3",
    },
    {
      gateId: "min_distinct_markets",
      description: "Mínimo 2 mercados distintos no filtrado",
      required: true,
      passed: distinctMarkets >= 2,
      currentValue: String(distinctMarkets),
      threshold: "≥ 2",
    },
    {
      gateId: "positive_improvement",
      description: "Improvement vs baseline positivo",
      required: true,
      passed: impVsBaseline !== null && impVsBaseline > 0,
      currentValue: impVsBaseline !== null ? String(impVsBaseline) : "n/a",
      threshold: "> 0",
    },
    {
      gateId: "persistence_not_failing",
      description: "Persistência de improvement não em failing",
      required: true,
      passed: persistMaint !== "failing",
      currentValue: persistMaint,
      threshold: "≠ failing",
    },
    {
      gateId: "sensitivity_not_failing",
      description: "Sensibilidade a thresholds não em failing",
      required: true,
      passed: sensMaint !== "failing",
      currentValue: sensMaint,
      threshold: "≠ failing",
    },
    {
      gateId: "overfit_not_high",
      description: "Overfit risk não é high",
      required: false,
      passed: robustness.overfitRiskVerdict !== "high",
      currentValue: robustness.overfitRiskVerdict,
      threshold: "≠ high",
    },
    {
      gateId: "concentration_not_failing",
      description: "Concentração não em failing (< 70%)",
      required: false,
      passed: concMaint !== "failing",
      currentValue: concMaint,
      threshold: "≠ failing",
    },
    {
      gateId: "robustness_not_unstable",
      description: "Robustness verdict não é unstable",
      required: false,
      passed: robustness.robustnessVerdict !== "unstable",
      currentValue: robustness.robustnessVerdict,
      threshold: "≠ unstable",
    },
  ];

  const requiredGates = gates.filter(g => g.required);
  const allRequiredPassed = requiredGates.every(g => g.passed);
  const gatesPassed = gates.filter(g => g.passed).length;
  const gatesFailed = gates.length - gatesPassed;

  const blockers: string[] = [];
  for (const g of gates) {
    if (!g.passed) {
      blockers.push(`${g.gateId}: ${g.description} (current=${g.currentValue}, need=${g.threshold})`);
    }
  }

  let tier: PromotionConfidenceTier;
  if (!allRequiredPassed || !robustness.hasFullRobustness) {
    tier = "too_early";
  } else if (robustness.overfitRiskVerdict === "high" || evBreadth === "insufficient") {
    tier = "fragile";
  } else if (
    evBreadth === "narrow" ||
    persistMaint === "degrading" ||
    concMaint === "failing"
  ) {
    tier = "weak_but_persistent";
  } else if (
    evBreadth === "broad" &&
    persistMaint === "improving" &&
    sensMaint === "improving" &&
    concMaint !== "degrading" &&
    robustness.overfitRiskVerdict === "low"
  ) {
    tier = "robust";
  } else if (
    (evBreadth === "moderate" || evBreadth === "broad") &&
    persistMaint !== "failing" &&
    sensMaint !== "failing"
  ) {
    tier = "stable";
  } else {
    tier = "weak_but_persistent";
  }

  let nextTarget: string;
  if (tier === "too_early") {
    nextTarget = "Acumular mais eventos para satisfazer gates obrigatórios (≥3 passing, ≥2 mercados, improvement > 0).";
  } else if (tier === "fragile") {
    nextTarget = "Reduzir overfit risk (mais mercados, mais eventos) e alargar evidence breadth para 'narrow' ou superior.";
  } else if (tier === "weak_but_persistent") {
    nextTarget = "Alargar breadth para 'moderate' (≥5 passing, ≥3 mercados) e manter persistence ≥ holding.";
  } else if (tier === "stable") {
    nextTarget = "Alargar breadth para 'broad' (≥10 passing, ≥5 mercados), manter persistence improving e overfit low.";
  } else {
    nextTarget = "Tier máximo atingido. Monitorar estabilidade contínua; considerar paper testing real.";
  }

  const supportingReasons: string[] = [];
  if (gatesPassed > 0) {
    supportingReasons.push(`${gatesPassed}/${gates.length} gates passados.`);
  }
  if (evBreadth !== "insufficient") {
    supportingReasons.push(`Evidence breadth: ${evBreadth} (${passingCount} eventos, ${distinctMarkets} mercados).`);
  }
  if (persistMaint === "improving" || persistMaint === "holding") {
    supportingReasons.push(`Persistence maintenance: ${persistMaint}.`);
  }

  return {
    promotionReadinessVerdict: tier,
    currentConfidenceTier: tier,
    passingEventCount: passingCount,
    distinctPassingMarkets: distinctMarkets,
    evidenceBreadthVerdict: evBreadth,
    persistenceMaintenanceVerdict: persistMaint,
    sensitivityMaintenanceVerdict: sensMaint,
    concentrationMaintenanceVerdict: concMaint,
    minimumPromotionGates: gates,
    gatesPassed,
    gatesFailed,
    promotionBlockers: blockers,
    nextPromotionTarget: nextTarget,
    supportingReasons,
    readDisclaimer:
      "Meta-avaliação de disciplina de promoção. Não é sinal de trading. Tier acima de 'too_early' significa que evidência observacional justifica continuar monitorando; não significa que a regra é lucrativa em produção.",
  };
}

export function buildPromotionReadinessSummaryLine(
  a: PromotionReadinessAssessment,
): string {
  return `promo: ${a.promotionReadinessVerdict} | gates=${a.gatesPassed}/${a.gatesPassed + a.gatesFailed} pass=${a.passingEventCount} mkts=${a.distinctPassingMarkets} breadth=${a.evidenceBreadthVerdict}`;
}
