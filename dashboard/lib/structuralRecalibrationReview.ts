/**
 * Structural Recalibration Review — consolida diagnósticos em shortlist objetiva.
 * Transforma profileEligibilityDiagnostics + profileEligibilityJudgements em prioridade
 * de recalibração estrutural. Uma única hipótese recomendada. Sem alterar lógica econômica.
 */

import type { ProfileEligibilityDiagnostics } from "./profileEligibilityFunnel";
import type { ProfileEligibilityJudgement, DominantChokeStage } from "./profileEligibilityJudge";

export type StructuralGate =
  | "pair_gate"
  | "fill_gate"
  | "capfloor_gate"
  | "degratio_gate"
  | "candidate_to_open"
  | "post_open_negative_economics";

const STRUCTURAL_PROFILE_IDS = new Set([
  "shadow_1000_structural_riskmanaged_v1",
  "shadow_1000_structural_exitkill_v1",
  "shadow_1000_structural_exitkill_window180_v1",
  "shadow_1000_structural_lateexit_nonreversion_v1",
  "shadow_1000_structural_lateexit_tighter_v1",
  "shadow_1000_structural_narrow_gt5_fill10to25_v1",
  "shadow_1000_structural_narrow_fill10to25_v1",
]);

const PRE_OPEN_GATES: StructuralGate[] = [
  "pair_gate",
  "fill_gate",
  "capfloor_gate",
  "degratio_gate",
  "candidate_to_open",
];

const POST_OPEN_GATES: StructuralGate[] = ["post_open_negative_economics"];

const MIN_STRUCTURAL_PROFILES_FOR_REVIEW = 2;
const MIN_TOTAL_RAW_OPPS_FOR_REVIEW = 50;
const MIN_CYCLES_FOR_REVIEW = 10;

export interface GatePressureEntry {
  gate: StructuralGate;
  affectedProfilesCount: number;
  totalRawOpportunitiesSeen: number;
  totalFinalCandidates: number;
  totalOpenedTrades: number;
  totalClosedTrades: number;
  chokeFrequency: number;
  relativePressure: number;
  supportingProfiles: string[];
  evidenceGrade: "WEAK" | "MODERATE" | "STRONG";
}

export interface StructuralRecalibrationReview {
  reviewedProfiles: string[];
  dominantStructuralChokesRanked: StructuralGate[];
  gatePressureRanking: GatePressureEntry[];
  preOpenVsPostOpenBreakdown: {
    preOpen: { totalAffected: number; gates: StructuralGate[]; pressure: number };
    postOpen: { totalAffected: number; gates: StructuralGate[]; pressure: number };
  };
  recalibrationCandidates: StructuralGate[];
  blockedRecalibrations: StructuralGate[];
  recommendedNextSingleHypothesis: {
    gate: StructuralGate | null;
    action: string;
    reason: string;
  };
  executiveSummary: string;
  evidenceSufficient: boolean;
}

function isStructuralProfile(profileId: string): boolean {
  return STRUCTURAL_PROFILE_IDS.has(profileId);
}

function getGateFromChokeStage(stage: DominantChokeStage): StructuralGate | null {
  switch (stage) {
    case "pair_gate":
    case "fill_gate":
    case "capfloor_gate":
    case "degratio_gate":
    case "candidate_to_open":
    case "post_open_negative_economics":
      return stage;
    default:
      return null;
  }
}

export function getStructuralRecalibrationReview(
  diagnostics: Record<string, ProfileEligibilityDiagnostics>,
  judgements: Record<string, ProfileEligibilityJudgement>
): StructuralRecalibrationReview {
  const structuralProfileIds = Object.keys(diagnostics).filter(isStructuralProfile);
  const reviewedProfiles = structuralProfileIds.filter((id) => diagnostics[id] != null);

  if (reviewedProfiles.length === 0) {
    return {
      reviewedProfiles: [],
      dominantStructuralChokesRanked: [],
      gatePressureRanking: [],
      preOpenVsPostOpenBreakdown: {
        preOpen: { totalAffected: 0, gates: [], pressure: 0 },
        postOpen: { totalAffected: 0, gates: [], pressure: 0 },
      },
      recalibrationCandidates: [],
      blockedRecalibrations: PRE_OPEN_GATES.concat(POST_OPEN_GATES),
      recommendedNextSingleHypothesis: {
        gate: null,
        action: "keep_collecting",
        reason: "Nenhum profile estrutural materializado para revisão.",
      },
      executiveSummary:
        "Nenhum profile estrutural disponível para revisão. Manter coleta de dados.",
      evidenceSufficient: false,
    };
  }

  const totalRaw = reviewedProfiles.reduce(
    (s, id) => s + (diagnostics[id]?.rawOpportunitiesSeen ?? 0),
    0
  );
  const totalCycles = Math.max(
    ...reviewedProfiles.map((id) => diagnostics[id]?.cyclesProcessed ?? 0)
  );

  const insufficientEvidence =
    reviewedProfiles.length < MIN_STRUCTURAL_PROFILES_FOR_REVIEW ||
    totalRaw < MIN_TOTAL_RAW_OPPS_FOR_REVIEW ||
    totalCycles < MIN_CYCLES_FOR_REVIEW;

  const gateData: Record<StructuralGate, GatePressureEntry> = {
    pair_gate: {
      gate: "pair_gate",
      affectedProfilesCount: 0,
      totalRawOpportunitiesSeen: 0,
      totalFinalCandidates: 0,
      totalOpenedTrades: 0,
      totalClosedTrades: 0,
      chokeFrequency: 0,
      relativePressure: 0,
      supportingProfiles: [],
      evidenceGrade: "WEAK",
    },
    fill_gate: {
      gate: "fill_gate",
      affectedProfilesCount: 0,
      totalRawOpportunitiesSeen: 0,
      totalFinalCandidates: 0,
      totalOpenedTrades: 0,
      totalClosedTrades: 0,
      chokeFrequency: 0,
      relativePressure: 0,
      supportingProfiles: [],
      evidenceGrade: "WEAK",
    },
    capfloor_gate: {
      gate: "capfloor_gate",
      affectedProfilesCount: 0,
      totalRawOpportunitiesSeen: 0,
      totalFinalCandidates: 0,
      totalOpenedTrades: 0,
      totalClosedTrades: 0,
      chokeFrequency: 0,
      relativePressure: 0,
      supportingProfiles: [],
      evidenceGrade: "WEAK",
    },
    degratio_gate: {
      gate: "degratio_gate",
      affectedProfilesCount: 0,
      totalRawOpportunitiesSeen: 0,
      totalFinalCandidates: 0,
      totalOpenedTrades: 0,
      totalClosedTrades: 0,
      chokeFrequency: 0,
      relativePressure: 0,
      supportingProfiles: [],
      evidenceGrade: "WEAK",
    },
    candidate_to_open: {
      gate: "candidate_to_open",
      affectedProfilesCount: 0,
      totalRawOpportunitiesSeen: 0,
      totalFinalCandidates: 0,
      totalOpenedTrades: 0,
      totalClosedTrades: 0,
      chokeFrequency: 0,
      relativePressure: 0,
      supportingProfiles: [],
      evidenceGrade: "WEAK",
    },
    post_open_negative_economics: {
      gate: "post_open_negative_economics",
      affectedProfilesCount: 0,
      totalRawOpportunitiesSeen: 0,
      totalFinalCandidates: 0,
      totalOpenedTrades: 0,
      totalClosedTrades: 0,
      chokeFrequency: 0,
      relativePressure: 0,
      supportingProfiles: [],
      evidenceGrade: "WEAK",
    },
  };

  for (const profileId of reviewedProfiles) {
    const d = diagnostics[profileId];
    const j = judgements[profileId];
    if (!d || !j) continue;

    const gate = getGateFromChokeStage(j.dominantChokeStage);
    if (!gate) continue;

    const entry = gateData[gate];
    entry.affectedProfilesCount++;
    entry.supportingProfiles.push(profileId);
    entry.totalRawOpportunitiesSeen += d.rawOpportunitiesSeen;
    entry.totalFinalCandidates += d.funnel.finalCandidateCount;
    entry.totalOpenedTrades += d.funnel.openedTradeCount;
    entry.totalClosedTrades += d.funnel.closedTradeCount;

    if (j.evidenceGrade === "STRONG") {
      entry.evidenceGrade = "STRONG";
    } else if (j.evidenceGrade === "MODERATE" && entry.evidenceGrade === "WEAK") {
      entry.evidenceGrade = "MODERATE";
    }
  }

  const totalChokedProfiles = Object.values(gateData).reduce(
    (s, e) => s + e.affectedProfilesCount,
    0
  );
  const maxAffected = Math.max(
    ...Object.values(gateData).map((e) => e.affectedProfilesCount),
    1
  );

  for (const gate of Object.keys(gateData) as StructuralGate[]) {
    const e = gateData[gate];
    e.chokeFrequency =
      reviewedProfiles.length > 0 ? e.affectedProfilesCount / reviewedProfiles.length : 0;
    e.relativePressure =
      totalChokedProfiles > 0 ? e.affectedProfilesCount / totalChokedProfiles : 0;
  }

  const gatePressureRanking = Object.values(gateData)
    .filter((e) => e.affectedProfilesCount > 0)
    .sort((a, b) => {
      if (b.affectedProfilesCount !== a.affectedProfilesCount) {
        return b.affectedProfilesCount - a.affectedProfilesCount;
      }
      return b.totalRawOpportunitiesSeen - a.totalRawOpportunitiesSeen;
    });

  const dominantStructuralChokesRanked = gatePressureRanking.map((e) => e.gate);

  const preOpenAffected = PRE_OPEN_GATES.reduce(
    (s, g) => s + gateData[g].affectedProfilesCount,
    0
  );
  const postOpenAffected = gateData["post_open_negative_economics"].affectedProfilesCount;
  const preOpenPressure = reviewedProfiles.length > 0 ? preOpenAffected / reviewedProfiles.length : 0;
  const postOpenPressure = reviewedProfiles.length > 0 ? postOpenAffected / reviewedProfiles.length : 0;

  const preOpenGatesWithPressure = PRE_OPEN_GATES.filter(
    (g) => gateData[g].affectedProfilesCount > 0
  );
  const postOpenGatesWithPressure = POST_OPEN_GATES.filter(
    (g) => gateData[g].affectedProfilesCount > 0
  );

  let recalibrationCandidates: StructuralGate[] = [];
  let blockedRecalibrations: StructuralGate[] = [];
  let recommendedNextSingleHypothesis: StructuralRecalibrationReview["recommendedNextSingleHypothesis"] = {
    gate: null,
    action: "keep_collecting",
    reason: "Evidência insuficiente.",
  };
  let executiveSummary = "";

  if (insufficientEvidence) {
    blockedRecalibrations = [...PRE_OPEN_GATES, ...POST_OPEN_GATES];
    recommendedNextSingleHypothesis = {
      gate: null,
      action: "keep_collecting",
      reason: `Amostra insuficiente: ${reviewedProfiles.length} profiles, ${totalRaw} raw opps, ${totalCycles} ciclos. Mínimos: ${MIN_STRUCTURAL_PROFILES_FOR_REVIEW} profiles, ${MIN_TOTAL_RAW_OPPS_FOR_REVIEW} raw opps, ${MIN_CYCLES_FOR_REVIEW} ciclos.`,
    };
    executiveSummary =
      "Sem evidência suficiente ainda para priorizar recalibração. Continuar coleta de dados.";
  } else {
    const postOpenDominant =
      postOpenAffected > 0 && postOpenPressure >= preOpenPressure;

    if (postOpenDominant && postOpenAffected >= reviewedProfiles.length * 0.5) {
      blockedRecalibrations = [...PRE_OPEN_GATES, ...POST_OPEN_GATES];
      recalibrationCandidates = [];
      recommendedNextSingleHypothesis = {
        gate: "post_open_negative_economics",
        action: "inspect_exit_economics",
        reason: "Problema dominante após abertura. Não recalibrar entrada.",
      };
      executiveSummary =
        "Os gates pré-open não são o principal choke; o problema dominante aparece após abertura, com destruição econômica no close. Não recalibrar entrada ainda.";
    } else if (gatePressureRanking.length > 0) {
      const top = gatePressureRanking[0];
      const hasStrongEvidence =
        top.evidenceGrade === "STRONG" || top.evidenceGrade === "MODERATE";

      if (hasStrongEvidence && top.gate !== "post_open_negative_economics") {
        recalibrationCandidates = [top.gate];
        blockedRecalibrations = (PRE_OPEN_GATES.concat(POST_OPEN_GATES) as StructuralGate[]).filter(
          (g) => g !== top.gate
        );

        const actionMap: Record<StructuralGate, string> = {
          pair_gate: "relaxamento controlado do pair set",
          fill_gate: "relaxamento controlado do fill bucket",
          capfloor_gate: "revisão do capfloor",
          degratio_gate: "revisão do degratio",
          candidate_to_open: "inspecionar transição open",
          post_open_negative_economics: "inspecionar exit economics",
        };

        recommendedNextSingleHypothesis = {
          gate: top.gate,
          action: actionMap[top.gate] ?? "inspect",
          reason: `${top.affectedProfilesCount} profile(s) estrangulado(s) em ${top.gate}.`,
        };

        const gateLabelMap: Record<StructuralGate, string> = {
          pair_gate: "pair",
          fill_gate: "fill",
          capfloor_gate: "capfloor",
          degratio_gate: "degratio",
          candidate_to_open: "candidate_to_open",
          post_open_negative_economics: "exit",
        };

        executiveSummary = `A família estrutural é majoritariamente estrangulada no gate de ${gateLabelMap[top.gate]} antes da formação de candidates. Próxima hipótese única recomendada: ${actionMap[top.gate]}.`;
      } else {
        blockedRecalibrations = [...PRE_OPEN_GATES, ...POST_OPEN_GATES];
        recommendedNextSingleHypothesis = {
          gate: null,
          action: "keep_collecting",
          reason: `Gate dominante ${top.gate} com evidência ${top.evidenceGrade}. Aguardar mais dados.`,
        };
        executiveSummary =
          "Sem evidência suficiente ainda para recalibração. Manter coleta de dados.";
      }
    } else {
      blockedRecalibrations = [...PRE_OPEN_GATES, ...POST_OPEN_GATES];
      recommendedNextSingleHypothesis = {
        gate: null,
        action: "do_not_modify_yet",
        reason: "Nenhum choke estrutural dominante identificado.",
      };
      executiveSummary =
        "Nenhum choke estrutural dominante nos profiles revisados. Manter monitoramento.";
    }
  }

  return {
    reviewedProfiles,
    dominantStructuralChokesRanked,
    gatePressureRanking,
    preOpenVsPostOpenBreakdown: {
      preOpen: {
        totalAffected: preOpenAffected,
        gates: preOpenGatesWithPressure,
        pressure: preOpenPressure,
      },
      postOpen: {
        totalAffected: postOpenAffected,
        gates: postOpenGatesWithPressure,
        pressure: postOpenPressure,
      },
    },
    recalibrationCandidates,
    blockedRecalibrations,
    recommendedNextSingleHypothesis,
    executiveSummary,
    evidenceSufficient: !insufficientEvidence,
  };
}
