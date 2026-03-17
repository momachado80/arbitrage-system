/**
 * Experiment Registry — structured registry of allowed experiment families and variants.
 * Defines which hypotheses are allowed, base profiles, and single-variable changes.
 * Does NOT auto-activate anything.
 */

export type ExperimentStatus =
  | "proposed"
  | "available"
  | "active"
  | "under_observation"
  | "successful"
  | "failed"
  | "inconclusive"
  | "archived"
  | "superseded";

export interface ExperimentFamilyDefinition {
  familyId: string;
  label: string;
  allowedBaseProfiles: string[];
  isolatedVariable: string;
  safetyConstraints: string[];
  eligibleForRecommendation: boolean;
  allowedForAutomatedActivation: boolean;
  /** Challenger profileId pattern; e.g. shadow_1000_adapt_captrade_v1 */
  challengerPattern?: string;
}

export interface ExperimentVariantDefinition extends ExperimentFamilyDefinition {
  variantId: string;
  challengerProfileId: string;
}

/** Experiment families and their constraints. allowedForAutomatedActivation is false for all in v1. */
export const EXPERIMENT_FAMILIES: ExperimentFamilyDefinition[] = [
  {
    familyId: "captrade",
    label: "Max capital per trade reduction",
    allowedBaseProfiles: ["shadow_1000"],
    isolatedVariable: "maxCapitalPerTrade",
    safetyConstraints: ["single-variable change only", "no exit/hold/fill changes"],
    eligibleForRecommendation: true,
    allowedForAutomatedActivation: false,
    challengerPattern: "shadow_1000_adapt_captrade_v1",
  },
  {
    familyId: "entryfloor",
    label: "Entry floor (minNetCapturableEdgeToTrade) on captrade",
    allowedBaseProfiles: ["shadow_1000_adapt_captrade_v1"],
    isolatedVariable: "minNetCapturableEdgeToTrade",
    safetyConstraints: ["single-variable on top of captrade", "inherits maxCapitalPerTrade=75"],
    eligibleForRecommendation: true,
    allowedForAutomatedActivation: false,
    challengerPattern: "shadow_1000_adapt_captrade_entryfloor_v1",
  },
  {
    familyId: "pairpenalty",
    label: "Pair-level entry penalties",
    allowedBaseProfiles: ["shadow_1000", "shadow_1000_adapt_captrade_entryfloor_v1"],
    isolatedVariable: "entryPairPenalties",
    safetyConstraints: ["pair-specific penalties from worstPairs", "effectiveEdge = capturableEdge - penalty"],
    eligibleForRecommendation: true,
    allowedForAutomatedActivation: false,
    challengerPattern: "shadow_1000_adapt_pairpenalty_v1",
  },
  {
    familyId: "entryfloor_pairpenalty",
    label: "Pair penalties on top of entryfloor (winning branch refinement)",
    allowedBaseProfiles: ["shadow_1000_adapt_captrade_entryfloor_v1"],
    isolatedVariable: "entryPairPenalties",
    safetyConstraints: ["layered on winning branch", "single new variable"],
    eligibleForRecommendation: true,
    allowedForAutomatedActivation: false,
    challengerPattern: "shadow_1000_adapt_captrade_entryfloor_pairpenalty_v1",
  },
  {
    familyId: "edgegate",
    label: "Min capturable edge threshold (global gate)",
    allowedBaseProfiles: ["shadow_1000"],
    isolatedVariable: "minCapturableEdgeToTrade",
    safetyConstraints: ["single-variable change"],
    eligibleForRecommendation: true,
    allowedForAutomatedActivation: false,
    challengerPattern: "shadow_1000_adapt_edgegate_v1",
  },
  {
    familyId: "exit_refinement_placeholder",
    label: "Exit refinement (placeholder for future)",
    allowedBaseProfiles: ["shadow_1000"],
    isolatedVariable: "maxHoldingTimeMs_or_exit_logic",
    safetyConstraints: ["not yet implemented"],
    eligibleForRecommendation: false,
    allowedForAutomatedActivation: false,
  },
  {
    familyId: "exit_refinement",
    label: "Exit refinement on captrade (shorter max hold)",
    allowedBaseProfiles: ["shadow_1000_adapt_captrade_v1"],
    isolatedVariable: "maxHoldingTimeMs",
    safetyConstraints: ["single-variable on top of captrade", "inherits maxCapitalPerTrade=75", "exit timing only"],
    eligibleForRecommendation: true,
    allowedForAutomatedActivation: false,
    challengerPattern: "shadow_1000_adapt_captrade_exitrefine_v1",
  },
  {
    familyId: "fillguard_refinement",
    label: "Fill-quality / adverse-selection guard on captrade+exitrefine",
    allowedBaseProfiles: ["shadow_1000_adapt_captrade_exitrefine_v1"],
    isolatedVariable: "minFillRatioToTrade",
    safetyConstraints: ["single-variable on top of captrade+exitrefine", "inherits maxCapitalPerTrade=75, maxHoldingTimeMs=60s", "fill-quality guard only"],
    eligibleForRecommendation: true,
    allowedForAutomatedActivation: false,
    challengerPattern: "shadow_1000_adapt_captrade_exitrefine_fillguard_v1",
  },
];

export function getFamilyByChallengerId(challengerProfileId: string): ExperimentFamilyDefinition | undefined {
  return EXPERIMENT_FAMILIES.find(
    (f) => f.challengerPattern === challengerProfileId || challengerProfileId.includes(f.familyId)
  );
}

export function getFamilyById(familyId: string): ExperimentFamilyDefinition | undefined {
  return EXPERIMENT_FAMILIES.find((f) => f.familyId === familyId);
}
