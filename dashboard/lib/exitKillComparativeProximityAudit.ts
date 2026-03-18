/**
 * Exit Kill Comparative Proximity Audit — shadow_1000_structural_exitkill_v1 vs window180_v1.
 * Evidência causal: valores no momento mais próximo do fechamento e distância aos thresholds.
 * Caveat: para trades que fecharam fora da janela (ex. max_holding), os valores são no close,
 * não no momento em que o kill teria sido avaliado (90s/180s).
 */

import type { ShadowProfileState } from "./shadowSimulationStore";
import { STRUCTURAL_EXIT_KILL_PROFILE_ID } from "./structuralExitKillDiagnostics";
import { STRUCTURAL_EXIT_KILL_WINDOW180_PROFILE_ID } from "./structuralExitKillWindow180Diagnostics";

const KILL_CAPTABLE_DECAY = 0.5;
const KILL_OBSERVED_DECAY = 0.5;
const KILL_NET_EDGE_FLOOR = 0.02;

/** "Perto" = dentro de 15% do threshold */
const NEAR_CAPTABLE_RATIO_LO = 0.5;
const NEAR_CAPTABLE_RATIO_HI = 0.6;
const NEAR_OBSERVED_RATIO_LO = 0.5;
const NEAR_OBSERVED_RATIO_HI = 0.6;
const NEAR_NET_EDGE_LO = 0.02;
const NEAR_NET_EDGE_HI = 0.035;

export interface PerTradeProximityEntry {
  tradeId: string;
  profileId: string;
  holdingTimeMs: number;
  exitReason: string;
  capturableAtEntry: number;
  observedAtEntry: number;
  capturableProxyAtClose: number;
  observedAtClose: number;
  netEdgeAtClose: number;
  degRatioAtClose: number;
  absentCyclesAtClose: number;
  distToCapturableThreshold: number;
  distToObservedThreshold: number;
  distToNetEdgeThreshold: number;
  nearCapturable: boolean;
  nearObserved: boolean;
  nearNetEdge: boolean;
  closestCriterion: "capturable_decayed" | "observed_decayed" | "net_edge_floor" | "none";
  inKillWindow: boolean;
  killWindowMs: number;
}

export interface ChallengerProximityAggregate {
  profileId: string;
  killWindowMs: number;
  totalClosed: number;
  closedInKillWindow: number;
  nearCapturableCount: number;
  nearObservedCount: number;
  nearNetEdgeCount: number;
  avgDistToCapturable: number;
  avgDistToObserved: number;
  avgDistToNetEdge: number;
  closestCriterionCounts: Record<string, number>;
}

export interface ExitKillComparativeProximityAuditBlock {
  exitkill_v1: {
    perTrade: PerTradeProximityEntry[];
    aggregate: ChallengerProximityAggregate;
  };
  exitkill_window180_v1: {
    perTrade: PerTradeProximityEntry[];
    aggregate: ChallengerProximityAggregate;
  };
  conclusion: {
    evidenceThresholdsFrouxos: boolean;
    evidenceSinaisErrados: boolean;
    evidenceHipoteseDesalinhada: boolean;
    recomendacaoAmpliarJanela: boolean;
    recomendacaoApertarThresholds: boolean;
    faltaInstrumentacao: boolean;
    summary: string;
  };
}

function computePerTrade(
  t: {
    tradeId: string;
    holdingTimeMs?: number | null;
    exitReason?: string | null;
    capturableEdgeAtEntry?: number | null;
    observedEdgeAtEntry?: number | null;
    edgeAtExit?: number | null;
    opportunityAbsentCyclesAtKill?: number | null;
  },
  profileId: string,
  killWindowMs: number
): PerTradeProximityEntry {
  const capEntry = t.capturableEdgeAtEntry ?? 0;
  const obsEntry = t.observedEdgeAtEntry ?? 0;
  const edgeExit = t.edgeAtExit ?? 0;
  const holdingMs = t.holdingTimeMs ?? 0;

  const capturableProxy =
    obsEntry > 0.0001 ? capEntry * (edgeExit / obsEntry) : edgeExit;
  const observedAtClose = edgeExit;
  const netEdgeAtClose = edgeExit;
  const degRatioAtClose =
    observedAtClose > 0.0001 ? capturableProxy / observedAtClose : 0;
  const absentCyclesAtClose = t.opportunityAbsentCyclesAtKill ?? 0;

  const capturableThreshold = 0.5 * capEntry;
  const observedThreshold = 0.5 * obsEntry;

  const distCapturable = capturableProxy - capturableThreshold;
  const distObserved = observedAtClose - observedThreshold;
  const distNetEdge = observedAtClose - KILL_NET_EDGE_FLOOR;

  const ratioCapturable = capEntry > 0.0001 ? capturableProxy / capEntry : 1;
  const ratioObserved = obsEntry > 0.0001 ? observedAtClose / obsEntry : 1;

  const nearCapturable =
    ratioCapturable >= NEAR_CAPTABLE_RATIO_LO &&
    ratioCapturable <= NEAR_CAPTABLE_RATIO_HI;
  const nearObserved =
    ratioObserved >= NEAR_OBSERVED_RATIO_LO &&
    ratioObserved <= NEAR_OBSERVED_RATIO_HI;
  const nearNetEdge =
    netEdgeAtClose >= NEAR_NET_EDGE_LO && netEdgeAtClose <= NEAR_NET_EDGE_HI;

  const absDist = [
    { k: "capturable_decayed" as const, v: Math.abs(distCapturable) },
    { k: "observed_decayed" as const, v: Math.abs(distObserved) },
    { k: "net_edge_floor" as const, v: Math.abs(distNetEdge) },
  ];
  absDist.sort((a, b) => a.v - b.v);
  const closestCriterion =
    distCapturable <= 0 || distObserved <= 0 || distNetEdge <= 0
      ? absDist[0]!.k
      : absDist[0]!.v < 0.01
        ? absDist[0]!.k
        : "none";

  return {
    tradeId: t.tradeId,
    profileId,
    holdingTimeMs: holdingMs,
    exitReason: t.exitReason ?? "unknown",
    capturableAtEntry: capEntry,
    observedAtEntry: obsEntry,
    capturableProxyAtClose: capturableProxy,
    observedAtClose,
    netEdgeAtClose,
    degRatioAtClose,
    absentCyclesAtClose,
    distToCapturableThreshold: distCapturable,
    distToObservedThreshold: distObserved,
    distToNetEdgeThreshold: distNetEdge,
    nearCapturable,
    nearObserved,
    nearNetEdge,
    closestCriterion,
    inKillWindow: holdingMs < killWindowMs,
    killWindowMs,
  };
}

function aggregatePerChallenger(
  entries: PerTradeProximityEntry[],
  profileId: string,
  killWindowMs: number
): ChallengerProximityAggregate {
  const closedInWindow = entries.filter((e) => e.inKillWindow).length;
  const nearCapturable = entries.filter((e) => e.nearCapturable).length;
  const nearObserved = entries.filter((e) => e.nearObserved).length;
  const nearNetEdge = entries.filter((e) => e.nearNetEdge).length;

  const avgDistCap =
    entries.length > 0
      ? entries.reduce((s, e) => s + e.distToCapturableThreshold, 0) /
        entries.length
      : 0;
  const avgDistObs =
    entries.length > 0
      ? entries.reduce((s, e) => s + e.distToObservedThreshold, 0) /
        entries.length
      : 0;
  const avgDistNet =
    entries.length > 0
      ? entries.reduce((s, e) => s + e.distToNetEdgeThreshold, 0) / entries.length
      : 0;

  const closestCounts: Record<string, number> = {};
  for (const e of entries) {
    closestCounts[e.closestCriterion] =
      (closestCounts[e.closestCriterion] ?? 0) + 1;
  }

  return {
    profileId,
    killWindowMs,
    totalClosed: entries.length,
    closedInKillWindow: closedInWindow,
    nearCapturableCount: nearCapturable,
    nearObservedCount: nearObserved,
    nearNetEdgeCount: nearNetEdge,
    avgDistToCapturable: avgDistCap,
    avgDistToObserved: avgDistObs,
    avgDistToNetEdge: avgDistNet,
    closestCriterionCounts: closestCounts,
  };
}

export function getExitKillComparativeProximityAudit(
  profiles: ShadowProfileState[]
): ExitKillComparativeProximityAuditBlock {
  const exitkill = profiles.find((p) => p.profileId === STRUCTURAL_EXIT_KILL_PROFILE_ID);
  const window180 = profiles.find(
    (p) => p.profileId === STRUCTURAL_EXIT_KILL_WINDOW180_PROFILE_ID
  );

  const closedExitkill = (exitkill?.closedTrades ?? []).filter(
    (t) =>
      t.status === "closed" &&
      t.closedAt &&
      t.structuralRiskFilterMatchAtOpen !== false
  );
  const closedWindow180 = (window180?.closedTrades ?? []).filter(
    (t) =>
      t.status === "closed" &&
      t.closedAt &&
      t.structuralRiskFilterMatchAtOpen !== false
  );

  const perTradeExitkill = closedExitkill.map((t) =>
    computePerTrade(t, STRUCTURAL_EXIT_KILL_PROFILE_ID, 90_000)
  );
  const perTradeWindow180 = closedWindow180.map((t) =>
    computePerTrade(t, STRUCTURAL_EXIT_KILL_WINDOW180_PROFILE_ID, 180_000)
  );

  const aggExitkill = aggregatePerChallenger(
    perTradeExitkill,
    STRUCTURAL_EXIT_KILL_PROFILE_ID,
    90_000
  );
  const aggWindow180 = aggregatePerChallenger(
    perTradeWindow180,
    STRUCTURAL_EXIT_KILL_WINDOW180_PROFILE_ID,
    180_000
  );

  const allEntries = [...perTradeExitkill, ...perTradeWindow180];
  const anyNearCapturable = allEntries.some((e) => e.nearCapturable);
  const anyNearObserved = allEntries.some((e) => e.nearObserved);
  const anyNearNetEdge = allEntries.some((e) => e.nearNetEdge);
  const totalNear = allEntries.filter(
    (e) => e.nearCapturable || e.nearObserved || e.nearNetEdge
  ).length;
  const avgHoldingExitkill =
    perTradeExitkill.length > 0
      ? perTradeExitkill.reduce((s, e) => s + e.holdingTimeMs, 0) /
        perTradeExitkill.length
      : 0;
  const avgHoldingWindow180 =
    perTradeWindow180.length > 0
      ? perTradeWindow180.reduce((s, e) => s + e.holdingTimeMs, 0) /
        perTradeWindow180.length
      : 0;

  let evidenceThresholdsFrouxos = false;
  let evidenceSinaisErrados = false;
  let evidenceHipoteseDesalinhada = false;
  let recomendacaoAmpliarJanela = false;
  let recomendacaoApertarThresholds = false;
  let faltaInstrumentacao = false;
  let summary = "";

  if (aggExitkill.closedInKillWindow === 0 && aggWindow180.closedInKillWindow === 0) {
    if (avgHoldingExitkill > 90_000 && avgHoldingWindow180 > 180_000) {
      evidenceHipoteseDesalinhada = true;
      recomendacaoAmpliarJanela = true;
      summary =
        "Trades fecham fora da janela (avg hold >> window). Kill nunca avaliado. Ampliar janela ou reduzir maxHolding daria chance ao kill.";
    } else {
      summary = "Nenhum trade fechou dentro da janela. Dados insuficientes para proximidade.";
    }
  } else if (totalNear === 0 && allEntries.length > 0) {
    evidenceHipoteseDesalinhada = true;
    summary =
      "Nenhum trade perto dos thresholds no momento do close. Valores no close distantes. Hipótese de kill precoce pode estar desalinhada com regime.";
  } else if (totalNear > 0) {
    if (
      (anyNearCapturable || anyNearObserved || anyNearNetEdge) &&
      aggExitkill.closedInKillWindow + aggWindow180.closedInKillWindow > 0
    ) {
      const netFloorCount =
        (aggExitkill.closestCriterionCounts["net_edge_floor"] ?? 0) +
        (aggWindow180.closestCriterionCounts["net_edge_floor"] ?? 0);
      if (netFloorCount > (allEntries.length * 0.5)) {
        recomendacaoApertarThresholds = true;
        recomendacaoApertarThresholds = true;
        summary =
          "net_edge_floor foi o critério mais próximo na maioria dos trades. Considerar apertar killNetEdgeFloor.";
      } else {
        summary =
          "Alguns trades perto dos thresholds. Critério dominante: ver closestCriterionCounts.";
      }
    }
  }

  return {
    exitkill_v1: { perTrade: perTradeExitkill, aggregate: aggExitkill },
    exitkill_window180_v1: {
      perTrade: perTradeWindow180,
      aggregate: aggWindow180,
    },
    conclusion: {
      evidenceThresholdsFrouxos,
      evidenceSinaisErrados,
      evidenceHipoteseDesalinhada,
      recomendacaoAmpliarJanela,
      recomendacaoApertarThresholds,
      faltaInstrumentacao,
      summary: summary || "Análise inconclusiva. Revisar perTrade e aggregate.",
    },
  };
}
