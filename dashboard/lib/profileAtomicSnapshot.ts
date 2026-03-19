/**
 * Profile Atomic Snapshot — snapshot atômico por profile para legibilidade confiável.
 * Todos os campos derivados da mesma leitura de profile/store.
 */

import type { ProfileEligibilityDiagnostics } from "./profileEligibilityFunnel";
import type { ProfileConsistencyEntry } from "./profileObservabilityConsistency";
import type { ProfileActiveTradeAgingDiagnostics } from "./activeTradeAgingDiagnostics";

export interface ProfileAtomicSnapshotEntry {
  profileId: string;
  label?: string;
  consistency: ProfileConsistencyEntry | null;
  funnel: ProfileEligibilityDiagnostics["funnel"] | null;
  conversions: ProfileEligibilityDiagnostics["conversions"] | null;
  economics: ProfileEligibilityDiagnostics["economics"] | null;
  activeTradesSummary: {
    count: number;
    oldestMs: number;
    avgMs: number;
    summary: string;
  } | null;
  activeTradesAging: ProfileActiveTradeAgingDiagnostics["activeTradesAging"];
  lastCycleProcessedAt: string | null;
  summary: string;
}

export interface ProfileAtomicSnapshot {
  byProfile: Record<string, ProfileAtomicSnapshotEntry>;
  generatedAt: string;
}

function buildAtomicSummary(
  consistency: ProfileConsistencyEntry | null,
  elig: ProfileEligibilityDiagnostics | null,
  aging: ProfileActiveTradeAgingDiagnostics | null
): string {
  const parts: string[] = [];
  if (consistency?.summary) parts.push(`[Consistência] ${consistency.summary}`);
  if (elig?.chokePointSummary) parts.push(`[Funil] ${elig.chokePointSummary}`);
  if ((aging?.activeTradesCount ?? 0) > 0 && aging?.summary) parts.push(`[Aging] ${aging.summary}`);
  if (parts.length === 0) return "Sem dados suficientes para resumo.";
  return parts.join(" ");
}

export function buildProfileAtomicSnapshot(
  profileIds: string[],
  getConsistency: (pid: string) => ProfileConsistencyEntry | null,
  getEligibility: (pid: string) => ProfileEligibilityDiagnostics | null,
  getAging: (pid: string) => ProfileActiveTradeAgingDiagnostics | null,
  getLastCycleProcessedAt: (pid: string) => string | null,
  getLabel: (pid: string) => string | undefined
): ProfileAtomicSnapshot {
  const byProfile: Record<string, ProfileAtomicSnapshotEntry> = {};
  const now = new Date().toISOString();

  for (const pid of profileIds) {
    const consistency = getConsistency(pid);
    const elig = getEligibility(pid);
    const aging = getAging(pid);

    const activeTradesSummary =
      aging && aging.activeTradesCount > 0
        ? {
            count: aging.activeTradesCount,
            oldestMs: aging.oldestActiveTradeMs,
            avgMs: aging.avgActiveTradeMs,
            summary: aging.summary,
          }
        : null;

    byProfile[pid] = {
      profileId: pid,
      label: getLabel(pid) ?? elig?.label,
      consistency: consistency ?? null,
      funnel: elig?.funnel ?? null,
      conversions: elig?.conversions ?? null,
      economics: elig?.economics ?? null,
      activeTradesSummary,
      activeTradesAging: aging?.activeTradesAging ?? [],
      lastCycleProcessedAt: getLastCycleProcessedAt(pid),
      summary: buildAtomicSummary(consistency ?? null, elig ?? null, aging ?? null),
    };
  }

  return {
    byProfile,
    generatedAt: now,
  };
}
