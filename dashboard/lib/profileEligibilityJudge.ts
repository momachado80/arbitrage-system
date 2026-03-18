/**
 * Profile Eligibility Judge — transforma o funil de elegibilidade em decisão operacional.
 * Regras causais, sem alterar lógica econômica. Apenas observabilidade → decisão.
 */

import type { ProfileEligibilityDiagnostics } from "./profileEligibilityFunnel";

export type DominantChokeStage =
  | "upstream_no_opportunities"
  | "pair_gate"
  | "fill_gate"
  | "capfloor_gate"
  | "degratio_gate"
  | "candidate_to_open"
  | "post_open_negative_economics"
  | "insufficient_sample"
  | "no_clear_choke";

export type FunnelHealthStatus =
  | "HEALTHY_FLOW"
  | "STARVED_UPSTREAM"
  | "OVERFILTERED_EARLY"
  | "OVERFILTERED_LATE"
  | "OPENING_BROKEN"
  | "ECONOMICALLY_NEGATIVE_AFTER_OPEN"
  | "INCONCLUSIVE";

export type EvidenceGrade = "WEAK" | "MODERATE" | "STRONG";

export type RecommendedAction =
  | "keep_collecting"
  | "inspect_upstream_supply"
  | "inspect_pair_set"
  | "inspect_fill_bucket"
  | "inspect_capfloor"
  | "inspect_degratio"
  | "inspect_open_transition"
  | "inspect_exit_economics"
  | "eligible_for_recalibration_review"
  | "do_not_modify_yet";

export interface ProfileEligibilityJudgement {
  profileId: string;
  label?: string;
  dominantChokeStage: DominantChokeStage;
  dominantChokeReason: string;
  funnelHealthStatus: FunnelHealthStatus;
  evidenceGrade: EvidenceGrade;
  recommendedAction: RecommendedAction;
  summary: string;
}

/** Mínimo de ciclos processados para considerar amostra útil */
const MIN_CYCLES_FOR_OPINION = 5;

/** Mínimo de opps brutas para avaliar gates estruturais */
const MIN_RAW_OPPS_FOR_GATES = 20;

/** Mínimo de closes para avaliar economia pós-open */
const MIN_CLOSES_FOR_ECONOMICS = 3;

/** Razão abaixo da qual consideramos um gate estrangulando (ex: rawToPair < 0.05 = pair choke) */
const CHOKE_CONVERSION_THRESHOLD = 0.05;

function getTopDiscardReason(discardReasonCounts: Record<string, number>): { reason: string; count: number } | null {
  const entries = Object.entries(discardReasonCounts).filter(([, c]) => c > 0);
  if (entries.length === 0) return null;
  entries.sort((a, b) => b[1] - a[1]);
  return { reason: entries[0][0], count: entries[0][1] };
}

function computeJudgement(d: ProfileEligibilityDiagnostics): ProfileEligibilityJudgement {
  const { funnel, conversions, economics, discardReasonCounts } = d;
  const raw = d.rawOpportunitiesSeen;
  const cycles = d.cyclesProcessed;
  const pair = funnel.pairEligibleCount;
  const fill = funnel.fillEligibleCount;
  const capfloor = funnel.capfloorEligibleCount;
  const degratio = funnel.degratioEligibleCount;
  const candidate = funnel.finalCandidateCount;
  const opened = funnel.openedTradeCount;
  const closed = funnel.closedTradeCount;
  const pnlTotal = economics.realizedPnlTotal;

  const topDiscard = getTopDiscardReason(discardReasonCounts);
  const totalDiscards = Object.values(discardReasonCounts).reduce((a, b) => a + b, 0);

  const hasStructuralGates =
    (discardReasonCounts["structural_risk_pair_mismatch"] ?? 0) > 0 ||
    (discardReasonCounts["structural_risk_fill_bucket_mismatch"] ?? 0) > 0 ||
    (discardReasonCounts["structural_risk_capfloor"] ?? 0) > 0 ||
    (discardReasonCounts["structural_risk_degratio"] ?? 0) > 0;

  /** 1. Insufficient sample — não dá para opinar */
  if (cycles < MIN_CYCLES_FOR_OPINION) {
    return {
      profileId: d.profileId,
      label: d.label,
      dominantChokeStage: "insufficient_sample",
      dominantChokeReason: cycles < MIN_CYCLES_FOR_OPINION ? "poucos ciclos" : "amostra de opps insuficiente",
      funnelHealthStatus: "INCONCLUSIVE",
      evidenceGrade: "WEAK",
      recommendedAction: "keep_collecting",
      summary: "Sem evidência suficiente ainda para recalibração.",
    };
  }

  /** 2. Upstream no opportunities — processado mas sem opps */
  if (cycles >= MIN_CYCLES_FOR_OPINION && raw < 5) {
    return {
      profileId: d.profileId,
      label: d.label,
      dominantChokeStage: "upstream_no_opportunities",
      dominantChokeReason: "merge/graph não entregando opps ao profile",
      funnelHealthStatus: "STARVED_UPSTREAM",
      evidenceGrade: raw === 0 ? "STRONG" : "MODERATE",
      recommendedAction: "inspect_upstream_supply",
      summary: "Profile processado normalmente, mas sem oportunidades brutas chegando.",
    };
  }

  /** 3. Candidate to open — passou tudo mas não abriu (possível bug) */
  if (candidate > 0 && opened === 0) {
    return {
      profileId: d.profileId,
      label: d.label,
      dominantChokeStage: "candidate_to_open",
      dominantChokeReason: "addShadowTrade não foi chamado ou falhou",
      funnelHealthStatus: "OPENING_BROKEN",
      evidenceGrade: "STRONG",
      recommendedAction: "inspect_open_transition",
      summary: "Profile chega a candidate mas não abre trades. Verificar transição open.",
    };
  }

  /** 4. Post-open negative economics — abre e fecha com PnL negativo */
  if (opened > 0 && closed >= MIN_CLOSES_FOR_ECONOMICS && pnlTotal < 0) {
    return {
      profileId: d.profileId,
      label: d.label,
      dominantChokeStage: "post_open_negative_economics",
      dominantChokeReason: "realizedPnL total negativo após closes",
      funnelHealthStatus: "ECONOMICALLY_NEGATIVE_AFTER_OPEN",
      evidenceGrade: closed >= 5 ? "STRONG" : "MODERATE",
      recommendedAction: "inspect_exit_economics",
      summary: "Profile abre trades, porém continua destruindo economia no close.",
    };
  }

  /** 5–8. Structural gates (só aplicável quando profile tem esses filtros) */
  if (hasStructuralGates && raw >= MIN_RAW_OPPS_FOR_GATES) {
    /** Pair gate: opps brutas mas morre antes de pair */
    if (pair === 0 && (discardReasonCounts["structural_risk_pair_mismatch"] ?? 0) > totalDiscards * 0.5) {
      return {
        profileId: d.profileId,
        label: d.label,
        dominantChokeStage: "pair_gate",
        dominantChokeReason: "structural_risk_pair_mismatch dominante",
        funnelHealthStatus: "OVERFILTERED_EARLY",
        evidenceGrade: raw >= 50 ? "STRONG" : "MODERATE",
        recommendedAction: "inspect_pair_set",
        summary: "Profile processado normalmente, mas estrangulado majoritariamente no gate de pair set.",
      };
    }

    /** Fill gate: passa pair mas morre em fill */
    if (pair > 0 && fill === 0 && (discardReasonCounts["structural_risk_fill_bucket_mismatch"] ?? 0) > totalDiscards * 0.3) {
      return {
        profileId: d.profileId,
        label: d.label,
        dominantChokeStage: "fill_gate",
        dominantChokeReason: "structural_risk_fill_bucket_mismatch dominante",
        funnelHealthStatus: "OVERFILTERED_EARLY",
        evidenceGrade: pair >= 10 ? "STRONG" : "MODERATE",
        recommendedAction: "inspect_fill_bucket",
        summary: "Profile processado normalmente, mas estrangulado majoritariamente no gate de fill.",
      };
    }

    /** Capfloor gate: passa fill mas morre em capfloor */
    if (fill > 0 && capfloor === 0 && (discardReasonCounts["structural_risk_capfloor"] ?? 0) > 0) {
      return {
        profileId: d.profileId,
        label: d.label,
        dominantChokeStage: "capfloor_gate",
        dominantChokeReason: "structural_risk_capfloor dominante",
        funnelHealthStatus: "OVERFILTERED_LATE",
        evidenceGrade: fill >= 10 ? "STRONG" : "MODERATE",
        recommendedAction: "inspect_capfloor",
        summary: "Profile processado normalmente, mas estrangulado majoritariamente no gate de capfloor.",
      };
    }

    /** Degratio gate: passa capfloor mas morre em degratio */
    if (capfloor > 0 && degratio === 0 && (discardReasonCounts["structural_risk_degratio"] ?? 0) > 0) {
      return {
        profileId: d.profileId,
        label: d.label,
        dominantChokeStage: "degratio_gate",
        dominantChokeReason: "structural_risk_degratio dominante",
        funnelHealthStatus: "OVERFILTERED_LATE",
        evidenceGrade: capfloor >= 10 ? "STRONG" : "MODERATE",
        recommendedAction: "inspect_degratio",
        summary: "Profile processado normalmente, mas estrangulado majoritariamente no gate de degratio.",
      };
    }

    /** Choke por conversão: usa ratios quando os counts absolutos não batem */
    if (pair === 0 && raw > 0 && conversions.rawToPair < CHOKE_CONVERSION_THRESHOLD) {
      return {
        profileId: d.profileId,
        label: d.label,
        dominantChokeStage: "pair_gate",
        dominantChokeReason: topDiscard ? topDiscard.reason : "conversão raw→pair muito baixa",
        funnelHealthStatus: "OVERFILTERED_EARLY",
        evidenceGrade: "MODERATE",
        recommendedAction: "inspect_pair_set",
        summary: "Profile processado normalmente, mas estrangulado majoritariamente no gate de pair.",
      };
    }
    if (pair > 0 && fill === 0 && conversions.pairToFill < CHOKE_CONVERSION_THRESHOLD) {
      return {
        profileId: d.profileId,
        label: d.label,
        dominantChokeStage: "fill_gate",
        dominantChokeReason: topDiscard ? topDiscard.reason : "conversão pair→fill muito baixa",
        funnelHealthStatus: "OVERFILTERED_EARLY",
        evidenceGrade: "MODERATE",
        recommendedAction: "inspect_fill_bucket",
        summary: "Profile processado normalmente, mas estrangulado majoritariamente no gate de fill.",
      };
    }
    if (fill > 0 && capfloor === 0 && conversions.fillToCapfloor < CHOKE_CONVERSION_THRESHOLD) {
      return {
        profileId: d.profileId,
        label: d.label,
        dominantChokeStage: "capfloor_gate",
        dominantChokeReason: topDiscard ? topDiscard.reason : "conversão fill→capfloor muito baixa",
        funnelHealthStatus: "OVERFILTERED_LATE",
        evidenceGrade: "MODERATE",
        recommendedAction: "inspect_capfloor",
        summary: "Profile processado normalmente, mas estrangulado majoritariamente no gate de capfloor.",
      };
    }
    if (capfloor > 0 && degratio === 0 && conversions.capfloorToDegratio < CHOKE_CONVERSION_THRESHOLD) {
      return {
        profileId: d.profileId,
        label: d.label,
        dominantChokeStage: "degratio_gate",
        dominantChokeReason: topDiscard ? topDiscard.reason : "conversão capfloor→degratio muito baixa",
        funnelHealthStatus: "OVERFILTERED_LATE",
        evidenceGrade: "MODERATE",
        recommendedAction: "inspect_degratio",
        summary: "Profile processado normalmente, mas estrangulado majoritariamente no gate de degratio.",
      };
    }
  }

  /** 9. Healthy flow ou inconclusivo — fluxo ativo, PnL ok ou amostra pequena para economia */
  if (opened > 0 && closed > 0 && pnlTotal >= 0) {
    return {
      profileId: d.profileId,
      label: d.label,
      dominantChokeStage: "no_clear_choke",
      dominantChokeReason: "fluxo ativo, PnL realizado não negativo",
      funnelHealthStatus: "HEALTHY_FLOW",
      evidenceGrade: closed >= MIN_CLOSES_FOR_ECONOMICS ? "STRONG" : "MODERATE",
      recommendedAction: "do_not_modify_yet",
      summary: "Fluxo ativo, sem choke dominante identificado. Manter monitoramento.",
    };
  }

  /** 10. Aberto mas sem closes suficientes para avaliar economia */
  if (opened > 0 && closed < MIN_CLOSES_FOR_ECONOMICS) {
    return {
      profileId: d.profileId,
      label: d.label,
      dominantChokeStage: "no_clear_choke",
      dominantChokeReason: "poucos closes para avaliar economia",
      funnelHealthStatus: "INCONCLUSIVE",
      evidenceGrade: "WEAK",
      recommendedAction: "keep_collecting",
      summary: "Profile abre e fecha, mas amostra de closes insuficiente para concluir sobre economia.",
    };
  }

  /** 11. Discard dominante mas não mapeável a gate estrutural — perfil sem structural ou choke genérico */
  if (topDiscard && topDiscard.count > candidate && totalDiscards > 0) {
    return {
      profileId: d.profileId,
      label: d.label,
      dominantChokeStage: "no_clear_choke",
      dominantChokeReason: `principal descarte: ${topDiscard.reason}`,
      funnelHealthStatus: "INCONCLUSIVE",
      evidenceGrade: "MODERATE",
      recommendedAction: "eligible_for_recalibration_review",
      summary: `Principal descarte observado: ${topDiscard.reason}. Revisar critérios de entrada.`,
    };
  }

  /** Fallback */
  return {
    profileId: d.profileId,
    label: d.label,
    dominantChokeStage: "no_clear_choke",
    dominantChokeReason: "sem choke dominante identificado",
    funnelHealthStatus: "INCONCLUSIVE",
    evidenceGrade: "WEAK",
    recommendedAction: "do_not_modify_yet",
    summary: "Sem evidência suficiente ainda para recalibração.",
  };
}

/** Produz julgamentos por profile a partir dos diagnósticos do funil */
export function getProfileEligibilityJudgements(
  diagnostics: Record<string, ProfileEligibilityDiagnostics>
): Record<string, ProfileEligibilityJudgement> {
  const out: Record<string, ProfileEligibilityJudgement> = {};
  for (const [profileId, d] of Object.entries(diagnostics)) {
    out[profileId] = computeJudgement(d);
  }
  return out;
}
