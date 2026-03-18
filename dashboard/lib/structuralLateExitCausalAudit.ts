/**
 * Structural Late Exit Causal Audit — shadow_1000_structural_lateexit_nonreversion_v1.
 * Evidência causal: por que late exit não dispara. Proximidade aos thresholds no momento do close.
 */

import type { ShadowProfileState } from "./shadowSimulationStore";
import { STRUCTURAL_LATE_EXIT_PROFILE_ID } from "./structuralLateExitDiagnostics";

const MIN_OBSERVATION_MS = 90_000;
const MAX_HOLDING_MS = 300_000;
const REVERSION_MIN_FRACTION = 0.5;
const STAGNANT_EDGE_FLOOR = 0.03;
const NET_EDGE_PROLONGED_FLOOR = 0.02;

/** "Perto" = dentro de banda do threshold para diagnóstico */
const NEAR_NON_REVERSION_RATIO_LO = 0.45;
const NEAR_NON_REVERSION_RATIO_HI = 0.6;
const NEAR_NET_EDGE_LO = 0.015;
const NEAR_NET_EDGE_HI = 0.035;
const NEAR_STAGNANT_LO = 0.025;
const NEAR_STAGNANT_HI = 0.045;

export interface LateExitPerTradeEntry {
  tradeId: string;
  profileId: string;
  holdingTimeMs: number;
  exitReason: string;
  capturableAtEntry: number;
  observedAtEntry: number;
  observedAtClose: number;
  capturableProxyAtClose: number;
  reversionThreshold: number;
  inLateExitWindow: boolean;
  distToNonReversion: number;
  distToNetEdgeFloor: number;
  distToStagnantFloor: number;
  nearNonReversion: boolean;
  nearNetEdgeFloor: boolean;
  nearStagnantFloor: boolean;
  wouldHaveTriggeredNonReversion: boolean;
  wouldHaveTriggeredNetEdge: boolean;
  wouldHaveTriggeredStagnant: boolean;
  closestCriterion: "non_reversion" | "net_edge_prolonged_low" | "stagnant_edge" | "none";
  edgeAtExitAvailable: boolean;
}

export interface LateExitCausalAggregate {
  profileId: string;
  totalClosed: number;
  closedInLateExitWindow: number;
  closedByMaxHolding: number;
  closedByOther: number;
  nearNonReversionCount: number;
  nearNetEdgeFloorCount: number;
  nearStagnantFloorCount: number;
  wouldTriggerNonReversionCount: number;
  wouldTriggerNetEdgeCount: number;
  wouldTriggerStagnantCount: number;
  avgObservedAtClose: number;
  avgObservedAtEntry: number;
  avgDistToNonReversion: number;
  avgDistToNetEdgeFloor: number;
  avgDistToStagnantFloor: number;
  closestCriterionCounts: Record<string, number>;
  edgeAtExitNullCount: number;
}

export interface StructuralLateExitCausalAuditBlock {
  perTrade: LateExitPerTradeEntry[];
  aggregate: LateExitCausalAggregate;
  conclusion: {
    causeMostLikely: string;
    evidenceSummary: string;
    thresholdsFrouxos: boolean;
    ordemAvaliacaoProblema: boolean;
    sinaisErrados: boolean;
    camposDesatualizados: boolean;
    lateExitRaramenteAplicavel: boolean;
    recomendacao: string;
  };
}

export interface LateExitCausalAuditOpts {
  profileId: string;
  stagnantEdgeFloor: number;
  netEdgeProlongedFloor: number;
}

function computePerTrade(
  t: {
    tradeId: string;
    holdingTimeMs?: number | null;
    exitReason?: string | null;
    capturableEdgeAtEntry?: number | null;
    observedEdgeAtEntry?: number | null;
    edgeAtExit?: number | null;
  },
  profileId: string,
  opts: LateExitCausalAuditOpts
): LateExitPerTradeEntry {
  const capEntry = t.capturableEdgeAtEntry ?? 0;
  const obsEntry = t.observedEdgeAtEntry ?? 0;
  const edgeExit = t.edgeAtExit ?? null;
  const holdingMs = t.holdingTimeMs ?? 0;
  const exitReason = t.exitReason ?? "unknown";

  const edgeAtExitAvailable = edgeExit != null && typeof edgeExit === "number";
  const observedAtClose = edgeAtExitAvailable ? edgeExit : 0;
  const capturableProxyAtClose =
    obsEntry > 0.0001 ? capEntry * (observedAtClose / obsEntry) : observedAtClose;

  const reversionThreshold = REVERSION_MIN_FRACTION * obsEntry;
  const inLateExitWindow =
    holdingMs >= MIN_OBSERVATION_MS && holdingMs < MAX_HOLDING_MS;

  const distToNonReversion = observedAtClose - reversionThreshold;
  const distToNetEdgeFloor = observedAtClose - opts.netEdgeProlongedFloor;
  const distToStagnantFloor = observedAtClose - opts.stagnantEdgeFloor;

  const ratioToEntry = obsEntry > 0.0001 ? observedAtClose / obsEntry : 1;
  const nearNonReversion =
    ratioToEntry >= NEAR_NON_REVERSION_RATIO_LO &&
    ratioToEntry <= NEAR_NON_REVERSION_RATIO_HI;
  const nearNetEdgeFloor =
    observedAtClose >= NEAR_NET_EDGE_LO && observedAtClose <= NEAR_NET_EDGE_HI;
  const nearStagnantFloor =
    observedAtClose >= NEAR_STAGNANT_LO && observedAtClose <= NEAR_STAGNANT_HI;

  const wouldHaveTriggeredNonReversion =
    obsEntry > 0.0001 && observedAtClose < REVERSION_MIN_FRACTION * obsEntry;
  const wouldHaveTriggeredNetEdge = observedAtClose <= opts.netEdgeProlongedFloor;
  const wouldHaveTriggeredStagnant =
    observedAtClose <= opts.stagnantEdgeFloor;

  const absDist = [
    { k: "non_reversion" as const, v: Math.abs(distToNonReversion) },
    { k: "net_edge_prolonged_low" as const, v: Math.abs(distToNetEdgeFloor) },
    { k: "stagnant_edge" as const, v: Math.abs(distToStagnantFloor) },
  ];
  absDist.sort((a, b) => a.v - b.v);
  const closestCriterion =
    wouldHaveTriggeredNonReversion || wouldHaveTriggeredNetEdge || wouldHaveTriggeredStagnant
      ? wouldHaveTriggeredNonReversion
        ? "non_reversion"
        : wouldHaveTriggeredNetEdge
          ? "net_edge_prolonged_low"
          : "stagnant_edge"
      : absDist[0]!.v < 0.01
        ? absDist[0]!.k
        : "none";

  return {
    tradeId: t.tradeId,
    profileId,
    holdingTimeMs: holdingMs,
    exitReason,
    capturableAtEntry: capEntry,
    observedAtEntry: obsEntry,
    observedAtClose,
    capturableProxyAtClose,
    reversionThreshold,
    inLateExitWindow,
    distToNonReversion,
    distToNetEdgeFloor,
    distToStagnantFloor,
    nearNonReversion,
    nearNetEdgeFloor,
    nearStagnantFloor,
    wouldHaveTriggeredNonReversion,
    wouldHaveTriggeredNetEdge,
    wouldHaveTriggeredStagnant,
    closestCriterion,
    edgeAtExitAvailable,
  };
}

export function getStructuralLateExitCausalAudit(
  profile: ShadowProfileState | undefined,
  opts?: LateExitCausalAuditOpts
): StructuralLateExitCausalAuditBlock {
  const effectiveOpts: LateExitCausalAuditOpts = opts ?? {
    profileId: STRUCTURAL_LATE_EXIT_PROFILE_ID,
    stagnantEdgeFloor: STAGNANT_EDGE_FLOOR,
    netEdgeProlongedFloor: NET_EDGE_PROLONGED_FLOOR,
  };

  const closed = (profile?.closedTrades ?? []).filter(
    (t) =>
      t.status === "closed" &&
      t.closedAt &&
      t.structuralRiskFilterMatchAtOpen !== false
  );

  const perTrade = closed.map((t) =>
    computePerTrade(t, effectiveOpts.profileId, effectiveOpts)
  );

  const closedInWindow = perTrade.filter((e) => e.inLateExitWindow).length;
  const closedByMaxHolding = perTrade.filter(
    (e) => e.exitReason === "max_holding_time"
  ).length;
  const closedByOther = perTrade.length - closedByMaxHolding;

  const nearNonReversion = perTrade.filter((e) => e.nearNonReversion).length;
  const nearNetEdgeFloor = perTrade.filter((e) => e.nearNetEdgeFloor).length;
  const nearStagnantFloor = perTrade.filter((e) => e.nearStagnantFloor).length;
  const wouldTriggerNonReversion = perTrade.filter(
    (e) => e.wouldHaveTriggeredNonReversion
  ).length;
  const wouldTriggerNetEdge = perTrade.filter(
    (e) => e.wouldHaveTriggeredNetEdge
  ).length;
  const wouldTriggerStagnant = perTrade.filter(
    (e) => e.wouldHaveTriggeredStagnant
  ).length;

  const withEdge = perTrade.filter((e) => e.edgeAtExitAvailable);
  const avgObservedAtClose =
    withEdge.length > 0
      ? withEdge.reduce((s, e) => s + e.observedAtClose, 0) / withEdge.length
      : 0;
  const avgObservedAtEntry =
    perTrade.length > 0
      ? perTrade.reduce((s, e) => s + e.observedAtEntry, 0) / perTrade.length
      : 0;

  const avgDistNonRev =
    perTrade.length > 0
      ? perTrade.reduce((s, e) => s + e.distToNonReversion, 0) / perTrade.length
      : 0;
  const avgDistNet =
    perTrade.length > 0
      ? perTrade.reduce((s, e) => s + e.distToNetEdgeFloor, 0) / perTrade.length
      : 0;
  const avgDistStag =
    perTrade.length > 0
      ? perTrade.reduce((s, e) => s + e.distToStagnantFloor, 0) / perTrade.length
      : 0;

  const closestCounts: Record<string, number> = {};
  for (const e of perTrade) {
    closestCounts[e.closestCriterion] =
      (closestCounts[e.closestCriterion] ?? 0) + 1;
  }

  const edgeAtExitNullCount = perTrade.filter(
    (e) => !e.edgeAtExitAvailable
  ).length;

  const aggregate: LateExitCausalAggregate = {
    profileId: effectiveOpts.profileId,
    totalClosed: perTrade.length,
    closedInLateExitWindow: closedInWindow,
    closedByMaxHolding,
    closedByOther,
    nearNonReversionCount: nearNonReversion,
    nearNetEdgeFloorCount: nearNetEdgeFloor,
    nearStagnantFloorCount: nearStagnantFloor,
    wouldTriggerNonReversionCount: wouldTriggerNonReversion,
    wouldTriggerNetEdgeCount: wouldTriggerNetEdge,
    wouldTriggerStagnantCount: wouldTriggerStagnant,
    avgObservedAtClose,
    avgObservedAtEntry,
    avgDistToNonReversion: avgDistNonRev,
    avgDistToNetEdgeFloor: avgDistNet,
    avgDistToStagnantFloor: avgDistStag,
    closestCriterionCounts: closestCounts,
    edgeAtExitNullCount,
  };

  let causeMostLikely = "unknown";
  let evidenceSummary = "";
  let thresholdsFrouxos = false;
  let ordemAvaliacaoProblema = false;
  let sinaisErrados = false;
  let camposDesatualizados = false;
  let lateExitRaramenteAplicavel = false;
  let recomendacao = "";

  if (perTrade.length === 0) {
    causeMostLikely = "sem_dados";
    evidenceSummary = "Nenhum trade fechado ainda.";
  } else if (closedInWindow === 0 && closedByMaxHolding >= Math.ceil(perTrade.length * 0.5)) {
    causeMostLikely = "maioria_max_holding_fora_janela";
    evidenceSummary = `${closedByMaxHolding} de ${perTrade.length} trades fecharam por max_holding_time (holding ~300s). Nenhum fechou por late exit dentro da janela (90-300s). O late exit é avaliado apenas quando 90s <= holding < 300s.`;
    lateExitRaramenteAplicavel = true;

    if (withEdge.length > 0) {
      const wouldTriggerExcludingNull =
        perTrade.filter(
          (e) =>
            e.edgeAtExitAvailable &&
            (e.wouldHaveTriggeredNonReversion ||
              e.wouldHaveTriggeredNetEdge ||
              e.wouldHaveTriggeredStagnant)
        ).length;
      if (wouldTriggerExcludingNull > 0) {
        causeMostLikely = "thresholds_satisfeitos_no_close_mas_janela_pulada";
        evidenceSummary += ` No momento do close, ${wouldTriggerExcludingNull} trades (com edge disponível) satisfariam algum critério. Possível: edge só cai no final.`;
      } else if (avgObservedAtClose > effectiveOpts.stagnantEdgeFloor + 0.02) {
        causeMostLikely = "observed_edge_mantem_alto";
        evidenceSummary += ` avgObservedAtClose=${(avgObservedAtClose * 100).toFixed(2)}% >> stagnantFloor 3%. Edge permanece alto durante 90-300s. Thresholds frouxos para o regime.`;
        thresholdsFrouxos = true;
        recomendacao =
          "Considerar apertar thresholds: reversionMinFraction (ex. 0.6), stagnantEdgeFloor (ex. 0.045), netEdgeProlongedFloor (ex. 0.03). Ou reduzir minObservationMs para antecipar.";
      } else {
        evidenceSummary += ` avgObservedAtClose=${(avgObservedAtClose * 100).toFixed(2)}%. Proximidade aos thresholds: ver perTrade.`;
      }
    } else {
      causeMostLikely = "edge_at_exit_indisponivel";
      evidenceSummary += ` edgeAtExit null em ${edgeAtExitNullCount} trades (opp desapareceu antes do close). Sem dados de proximidade.`;
      camposDesatualizados = edgeAtExitNullCount === perTrade.length;
    }
  } else if (closedInWindow > 0) {
    causeMostLikely = "alguns_na_janela";
    evidenceSummary = `${closedInWindow} trades fecharam dentro da janela. Ver perTrade para detalhes.`;
  }

  return {
    perTrade,
    aggregate,
    conclusion: {
      causeMostLikely,
      evidenceSummary,
      thresholdsFrouxos,
      ordemAvaliacaoProblema,
      sinaisErrados,
      camposDesatualizados,
      lateExitRaramenteAplicavel,
      recomendacao,
    },
  };
}
