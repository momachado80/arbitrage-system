/**
 * Structural Evidence Readiness — legibilidade operacional para Branch A.
 * Expõe quanto falta para suficiência sem alterar economia, thresholds ou decisões.
 */

import type { ProfileEligibilityDiagnostics } from "./profileEligibilityFunnel";
import type { StructuralRecalibrationReview } from "./structuralRecalibrationReview";

/** Mesmos thresholds do review — não alterar */
const MIN_CYCLES = 10;
const MIN_RAW_OPPS = 50;
const MIN_PROFILES = 2;

/** Ciclos abaixo do mínimo para considerar NEAR_READY */
const NEAR_READY_CYCLES_THRESHOLD = 7;

export type ReadinessStatus = "NOT_READY" | "NEAR_READY" | "READY";

export type NextDecisionBranchWhenReady =
  | "likely_fill_gate_if_pattern_persists"
  | "likely_post_open_if_pattern_changes"
  | "likely_pair_gate_if_pattern_persists"
  | "likely_capfloor_or_degratio_if_pattern_persists"
  | "undetermined";

export interface StructuralEvidenceReadiness {
  evidenceSufficient: boolean;
  currentCycles: number;
  minimumCycles: number;
  cyclesRemaining: number;
  currentRawOpportunities: number;
  minimumRawOpportunities: number;
  rawOpportunitiesRemaining: number;
  currentProfilesReviewed: number;
  minimumProfilesReviewed: number;
  profilesRemaining: number;
  readinessStatus: ReadinessStatus;
  nextDecisionBranchWhenReady: NextDecisionBranchWhenReady;
  provisionalSignal: string;
  provisionalSignalIsDecisive: false;
  summary: string;
}

const STRUCTURAL_PROFILE_IDS = new Set([
  "shadow_1000_structural_riskmanaged_v1",
  "shadow_1000_structural_fillrelax_v1",
  "shadow_1000_structural_exitkill_v1",
  "shadow_1000_structural_exitkill_window180_v1",
  "shadow_1000_structural_lateexit_nonreversion_v1",
  "shadow_1000_structural_lateexit_tighter_v1",
  "shadow_1000_structural_narrow_gt5_fill10to25_v1",
  "shadow_1000_structural_narrow_fill10to25_v1",
]);

function inferProvisionalSignal(diagnostics: Record<string, ProfileEligibilityDiagnostics>): {
  signal: string;
  nextBranch: NextDecisionBranchWhenReady;
} {
  const structuralIds = Object.keys(diagnostics).filter((id) => STRUCTURAL_PROFILE_IDS.has(id));
  if (structuralIds.length === 0) return { signal: "no_data", nextBranch: "undetermined" };

  let fillChokeCount = 0;
  let pairChokeCount = 0;
  let postOpenNegativeCount = 0;
  let capfloorDegratioCount = 0;

  for (const id of structuralIds) {
    const d = diagnostics[id];
    if (!d) continue;
    const cs = d.chokePointSummary?.toLowerCase() ?? "";
    const { funnel } = d;
    const pnl = d.economics.realizedPnlTotal ?? 0;
    const closed = funnel.closedTradeCount ?? 0;

    if (cs.includes("fill") || cs.includes("fill bucket")) fillChokeCount++;
    if (cs.includes("pair") && !cs.includes("fill")) pairChokeCount++;
    if (closed >= 3 && pnl < 0) postOpenNegativeCount++;
    if (cs.includes("capfloor") || cs.includes("degratio")) capfloorDegratioCount++;
  }

  if (postOpenNegativeCount >= structuralIds.length * 0.5) {
    return {
      signal: "post_open_negative_economics_preview",
      nextBranch: "likely_post_open_if_pattern_changes",
    };
  }
  if (fillChokeCount >= structuralIds.length * 0.5) {
    return {
      signal: "fill_gate_preview",
      nextBranch: "likely_fill_gate_if_pattern_persists",
    };
  }
  if (pairChokeCount >= structuralIds.length * 0.5) {
    return { signal: "pair_gate_preview", nextBranch: "likely_pair_gate_if_pattern_persists" };
  }
  if (capfloorDegratioCount >= structuralIds.length * 0.5) {
    return {
      signal: "capfloor_degratio_preview",
      nextBranch: "likely_capfloor_or_degratio_if_pattern_persists",
    };
  }
  return { signal: "undetermined_preview", nextBranch: "undetermined" };
}

export function getStructuralEvidenceReadiness(
  review: StructuralRecalibrationReview,
  diagnostics: Record<string, ProfileEligibilityDiagnostics>
): StructuralEvidenceReadiness {
  const reviewed = review.reviewedProfiles;
  const currentProfiles = reviewed.length;
  const currentRaw = reviewed.reduce(
    (s, id) => s + (diagnostics[id]?.rawOpportunitiesSeen ?? 0),
    0
  );
  const currentCycles =
    reviewed.length > 0
      ? Math.max(...reviewed.map((id) => diagnostics[id]?.cyclesProcessed ?? 0))
      : 0;

  const cyclesRemaining = Math.max(0, MIN_CYCLES - currentCycles);
  const rawRemaining = Math.max(0, MIN_RAW_OPPS - currentRaw);
  const profilesRemaining = Math.max(0, MIN_PROFILES - currentProfiles);

  let readinessStatus: ReadinessStatus = "NOT_READY";
  if (review.evidenceSufficient) {
    readinessStatus = "READY";
  } else if (
    currentCycles >= NEAR_READY_CYCLES_THRESHOLD &&
    currentRaw >= MIN_RAW_OPPS &&
    currentProfiles >= MIN_PROFILES
  ) {
    readinessStatus = "NEAR_READY";
  } else if (cyclesRemaining <= 3 && currentRaw >= MIN_RAW_OPPS && currentProfiles >= MIN_PROFILES) {
    readinessStatus = "NEAR_READY";
  }

  const { signal, nextBranch } = inferProvisionalSignal(diagnostics);

  const missing: string[] = [];
  if (cyclesRemaining > 0) missing.push(`${cyclesRemaining} ciclos`);
  if (rawRemaining > 0) missing.push(`${rawRemaining} opps brutas`);
  if (profilesRemaining > 0) missing.push(`${profilesRemaining} profiles`);

  const summary =
    missing.length === 0
      ? "Evidência suficiente. Pronto para próxima decisão."
      : `Evidência insuficiente. Falta: ${missing.join(", ")}. ` +
        `Padrão preliminar: ${signal} (não decisivo). ` +
        `Nenhuma modificação econômica deve ser feita ainda.`;

  return {
    evidenceSufficient: review.evidenceSufficient,
    currentCycles,
    minimumCycles: MIN_CYCLES,
    cyclesRemaining,
    currentRawOpportunities: currentRaw,
    minimumRawOpportunities: MIN_RAW_OPPS,
    rawOpportunitiesRemaining: rawRemaining,
    currentProfilesReviewed: currentProfiles,
    minimumProfilesReviewed: MIN_PROFILES,
    profilesRemaining,
    readinessStatus,
    nextDecisionBranchWhenReady: nextBranch,
    provisionalSignal: signal,
    provisionalSignalIsDecisive: false,
    summary,
  };
}
