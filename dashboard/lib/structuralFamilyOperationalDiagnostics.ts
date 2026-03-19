/**
 * Structural Family Operational Diagnostics — auditoria operacional.
 * Expõe heartbeat, lastCycleProcessedAt e evidência de freeze por família.
 */

import type { ShadowProfileState } from "./shadowSimulationStore";

const STRUCTURAL_PROFILE_IDS = new Set([
  "shadow_1000_structural_riskmanaged_v1",
  "shadow_1000_structural_fillrelax_v1",
  "shadow_1000_structural_fillrelax_capfloorrelax_v1",
  "shadow_1000_structural_exitkill_v1",
  "shadow_1000_structural_exitkill_window180_v1",
  "shadow_1000_structural_lateexit_nonreversion_v1",
  "shadow_1000_structural_lateexit_tighter_v1",
  "shadow_1000_structural_narrow_gt5_fill10to25_v1",
  "shadow_1000_structural_narrow_fill10to25_v1",
]);

export interface ProfileHeartbeatEntry {
  profileId: string;
  isStructural: boolean;
  lastUpdate: string | null;
  lastCycleProcessedAt: string | null;
  activeTradesCount: number;
  closedTradesCount: number;
  staleSeconds: number | null;
  appearsFrozen: boolean;
}

export interface StructuralFamilyOperationalBlock {
  perProfile: ProfileHeartbeatEntry[];
  structuralFamily: {
    profileIds: string[];
    allStale: boolean;
    minLastCycleProcessedAt: string | null;
    maxStaleSeconds: number | null;
  };
  nonStructuralFamily: {
    profileIds: string[];
    anyRecentHeartbeat: boolean;
  };
  conclusion: {
    structuralFamilyFrozen: boolean;
    evidence: string;
    recomendacao: string;
  };
}

const STALE_THRESHOLD_SECONDS = 120;

export function getStructuralFamilyOperationalDiagnostics(
  profiles: ShadowProfileState[]
): StructuralFamilyOperationalBlock {
  const now = Date.now();
  const perProfile: ProfileHeartbeatEntry[] = [];

  for (const p of profiles) {
    const isStructural = STRUCTURAL_PROFILE_IDS.has(p.profileId);
    const lastCycle = (p as { lastCycleProcessedAt?: string | null }).lastCycleProcessedAt ?? null;
    const lastUpdate = p.lastUpdate ?? null;

    const refTs = lastCycle ?? lastUpdate;
    const staleSeconds =
      refTs != null ? Math.max(0, Math.round((now - new Date(refTs).getTime()) / 1000)) : null;
    const appearsFrozen =
      staleSeconds != null && staleSeconds > STALE_THRESHOLD_SECONDS;

    perProfile.push({
      profileId: p.profileId,
      isStructural,
      lastUpdate,
      lastCycleProcessedAt: lastCycle,
      activeTradesCount: p.activeTrades.length,
      closedTradesCount: p.closedTrades.length,
      staleSeconds,
      appearsFrozen,
    });
  }

  const structuralProfiles = perProfile.filter((e) => e.isStructural);
  const nonStructuralProfiles = perProfile.filter((e) => !e.isStructural);

  const structuralLastCycles = structuralProfiles
    .map((e) => e.lastCycleProcessedAt)
    .filter((t): t is string => t != null);
  const minLastCycleStructural =
    structuralLastCycles.length > 0
      ? structuralLastCycles.reduce((a, b) => (a < b ? a : b))
      : null;
  const maxStaleStructural =
    structuralProfiles.length > 0
      ? Math.max(
          ...structuralProfiles.map((e) => e.staleSeconds ?? 0)
        )
      : null;
  const allStructuralStale =
    structuralProfiles.length > 0 &&
    structuralProfiles.every((e) => e.appearsFrozen);

  const anyNonStructuralRecent =
    nonStructuralProfiles.some(
      (e) => e.staleSeconds != null && e.staleSeconds < STALE_THRESHOLD_SECONDS
    );

  let structuralFamilyFrozen = false;
  let evidence = "";
  let recomendacao = "";

  if (structuralProfiles.length === 0) {
    evidence = "Nenhum profile estrutural materializado.";
  } else if (allStructuralStale && !anyNonStructuralRecent) {
    evidence = "Todos os perfis parecem stale; loop global pode estar parado.";
    recomendacao = "Verificar shadowLoopHeartbeatCount, marketBootstrapStatus.";
  } else if (allStructuralStale && anyNonStructuralRecent) {
    structuralFamilyFrozen = true;
    evidence = `Perfis estruturais com lastCycleProcessedAt stale (threshold ${STALE_THRESHOLD_SECONDS}s) enquanto perfis não-estruturais continuam ativos. Causa provável: lastUpdate/lastCycle só atualizam quando há open/close; estruturais não tiveram opps matching. Com updateProfileHeartbeat, lastCycleProcessedAt deve atualizar todo ciclo. Se ainda stale após deploy, perfis estruturais podem estar falhando no try/catch.`;
    recomendacao =
      "Após deploy com updateProfileHeartbeat: se lastCycleProcessedAt continua stale para estruturais, inspecionar logs [ShadowSim] profile X failed.";
  } else if (!allStructuralStale) {
    evidence = "Alguns perfis estruturais com heartbeat recente. Família operacional.";
    recomendacao = "Monitorar perProfile.appearsFrozen.";
  }

  return {
    perProfile,
    structuralFamily: {
      profileIds: structuralProfiles.map((e) => e.profileId),
      allStale: allStructuralStale,
      minLastCycleProcessedAt: minLastCycleStructural,
      maxStaleSeconds: maxStaleStructural,
    },
    nonStructuralFamily: {
      profileIds: nonStructuralProfiles.map((e) => e.profileId),
      anyRecentHeartbeat: anyNonStructuralRecent,
    },
    conclusion: {
      structuralFamilyFrozen,
      evidence,
      recomendacao,
    },
  };
}
