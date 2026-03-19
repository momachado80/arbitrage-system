/**
 * Profile Observability Consistency — audita divergência entre funil, economia e heartbeat.
 * Descobre por que um profile pode aparecer como "não processado" enquanto exibe closes e PnL.
 */

import type { ProfileEligibilityDiagnostics } from "./profileEligibilityFunnel";

export type ConsistencyStatus =
  | "CONSISTENT"
  | "FUNNEL_MISSING_BUT_ECONOMICS_PRESENT"
  | "HEARTBEAT_PRESENT_BUT_FUNNEL_EMPTY"
  | "NEW_PROFILE_NO_DATA_YET"
  | "INCONCLUSIVE";

export interface ProfileConsistencyEntry {
  profileId: string;
  label?: string;
  hasHeartbeat: boolean;
  hasEligibilityCounters: boolean;
  hasEconomicHistory: boolean;
  hasClosedTrades: boolean;
  consistencyStatus: ConsistencyStatus;
  inconsistencyReasons: string[];
  summary: string;
}

export interface ProfileObservabilityConsistency {
  byProfile: Record<string, ProfileConsistencyEntry>;
  generatedAt: string;
}

interface ProfileInput {
  profileId: string;
  label?: string;
  lastCycleProcessedAt: string | null;
  closedTradesCount: number;
  realizedPnL: number;
}

function computeConsistency(
  profile: ProfileInput,
  diagnostics: ProfileEligibilityDiagnostics | null
): ProfileConsistencyEntry {
  const hasClosedTrades = profile.closedTradesCount > 0;
  const hasEconomicHistory = hasClosedTrades || profile.realizedPnL !== 0;
  const hasHeartbeat = profile.lastCycleProcessedAt != null && profile.lastCycleProcessedAt !== "";
  const d = diagnostics;

  const funnelAllZero =
    !d ||
    (d.cyclesProcessed === 0 &&
      d.rawOpportunitiesSeen === 0 &&
      d.funnel.pairEligibleCount === 0 &&
      d.funnel.fillEligibleCount === 0 &&
      d.funnel.finalCandidateCount === 0 &&
      d.funnel.openedTradeCount === 0);

  const hasEligibilityCounters = d != null && (d.cyclesProcessed > 0 || d.rawOpportunitiesSeen > 0);

  const reasons: string[] = [];
  let status: ConsistencyStatus = "CONSISTENT";

  if (!d) {
    status = "NEW_PROFILE_NO_DATA_YET";
    reasons.push("profile sem entrada em eligibilityDiagnostics");
  } else if (funnelAllZero && hasEconomicHistory) {
    status = "FUNNEL_MISSING_BUT_ECONOMICS_PRESENT";
    reasons.push("contadores de funil zerados (pós-deploy) com histórico econômico reidratado");
  } else if (hasHeartbeat && funnelAllZero) {
    status = "HEARTBEAT_PRESENT_BUT_FUNNEL_EMPTY";
    reasons.push("heartbeat presente mas funnel vazio — possível reset pós-deploy ou profile novo");
  } else if (funnelAllZero && !hasEconomicHistory) {
    status = "NEW_PROFILE_NO_DATA_YET";
    reasons.push("profile novo ou sem runtime suficiente — sem dados de funil nem economia");
  } else if (!hasEligibilityCounters && hasEconomicHistory) {
    status = "FUNNEL_MISSING_BUT_ECONOMICS_PRESENT";
    reasons.push("economia presente (reidratada) mas funil não materializado neste runtime");
  }

  if (status === "CONSISTENT" && funnelAllZero && !hasEconomicHistory) {
    status = "INCONCLUSIVE";
    reasons.push("dados insuficientes para conclusão");
  }

  let summary: string;
  switch (status) {
    case "FUNNEL_MISSING_BUT_ECONOMICS_PRESENT":
      summary =
        "Histórico econômico reidratado presente; contadores de funil zerados (reset pós-deploy). Comparação pai vs challenger requer ciclos suficientes após deploy.";
      break;
    case "HEARTBEAT_PRESENT_BUT_FUNNEL_EMPTY":
      summary =
        "Heartbeat ativo mas funnel vazio. Pode indicar profile processado mas sem opps matching, ou reset recente.";
      break;
    case "NEW_PROFILE_NO_DATA_YET":
      summary = "Profile novo ou sem runtime suficiente. Aguardar ciclos para materialização do funil.";
      break;
    case "INCONCLUSIVE":
      summary = "Dados insuficientes para determinar consistência.";
      break;
    default:
      summary = "Funil, economia e heartbeat consistentes.";
  }

  return {
    profileId: profile.profileId,
    label: profile.label,
    hasHeartbeat,
    hasEligibilityCounters,
    hasEconomicHistory,
    hasClosedTrades,
    consistencyStatus: status,
    inconsistencyReasons: reasons,
    summary,
  };
}

export function getProfileObservabilityConsistency(
  profiles: ProfileInput[],
  diagnostics: Record<string, ProfileEligibilityDiagnostics>
): ProfileObservabilityConsistency {
  const byProfile: Record<string, ProfileConsistencyEntry> = {};
  const allIds = new Set([
    ...profiles.map((p) => p.profileId),
    ...Object.keys(diagnostics),
  ]);

  for (const pid of Array.from(allIds)) {
    const profile = profiles.find((p) => p.profileId === pid);
    const input: ProfileInput = profile ?? {
      profileId: pid,
      lastCycleProcessedAt: null,
      closedTradesCount: 0,
      realizedPnL: 0,
    };
    if (profile) {
      input.label = profile.label;
    } else {
      const d = diagnostics[pid];
      if (d) {
        input.closedTradesCount = d.funnel.closedTradeCount ?? 0;
        input.realizedPnL = d.economics.realizedPnlTotal ?? 0;
      }
    }
    byProfile[pid] = computeConsistency(input, diagnostics[pid] ?? null);
  }

  return {
    byProfile,
    generatedAt: new Date().toISOString(),
  };
}
