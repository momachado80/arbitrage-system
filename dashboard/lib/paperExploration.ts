/**
 * Ordenação leve por diversidade: penaliza dedupeKeys frequentes na janela recente.
 * Não altera política económica — só ordem de avaliação no batch.
 */

import type { NormalizedPaperOpportunity } from "./paperTypes";
import type { CapacityResult } from "./capitalCapacityEngine";
import { entryProfileKey } from "./paperEntryProfileMemory";
import { makeEconomicDedupeKey } from "./paperEconomicCooldown";

function envBool(name: string, defaultValue: boolean): boolean {
  const raw = process.env[name]?.trim().toLowerCase();
  if (!raw) return defaultValue;
  return raw === "1" || raw === "true" || raw === "yes";
}

function envNum(name: string, defaultValue: number): number {
  const raw = process.env[name]?.trim();
  if (!raw) return defaultValue;
  const n = Number(raw);
  return Number.isFinite(n) ? n : defaultValue;
}

export type PaperExplorationPolicySnapshot = {
  enabled: boolean;
  repeatPenaltyWeight: number;
};

export function getExplorationPolicySnapshot(): PaperExplorationPolicySnapshot {
  return {
    enabled: envBool("PAPER_EXPLORATION_ENABLED", true),
    repeatPenaltyWeight: Math.max(0, envNum("PAPER_EXPLORATION_REPEAT_PENALTY_WEIGHT", 0.002)),
  };
}

export type PaperExplorationLastCycle = {
  at: string;
  batchSize: number;
  repeatedCandidatePenaltyAppliedCount: number;
  distinctDedupeKeysInBatch: number;
  avgRepeatCountInBatch: number | null;
  evaluationPriorityDiagnostics: Array<{
    opportunityId: string;
    profileKey: string;
    estimatedNetEdge: number;
    repeatCount: number;
    sortKey: number;
  }>;
};

/**
 * `dedupeCounts`: contagens na janela `recentEvaluatedCandidates` (já computado em O(cap)).
 */
export function sortOpportunitiesForExploration(
  items: Array<{ opp: NormalizedPaperOpportunity; capacity: CapacityResult }>,
  dedupeCounts: Map<string, number>
): { sorted: Array<{ opp: NormalizedPaperOpportunity; capacity: CapacityResult }>; lastCycle: PaperExplorationLastCycle } {
  const cfg = getExplorationPolicySnapshot();
  const at = new Date().toISOString();
  if (!cfg.enabled || items.length <= 1) {
    return {
      sorted: items,
      lastCycle: {
        at,
        batchSize: items.length,
        repeatedCandidatePenaltyAppliedCount: 0,
        distinctDedupeKeysInBatch: new Set(
          items.map((x) => makeEconomicDedupeKey(entryProfileKey(x.opp.sourceType, x.opp.opportunityType), x.opp.opportunityId))
        ).size,
        avgRepeatCountInBatch: null,
        evaluationPriorityDiagnostics: [],
      },
    };
  }

  const enriched = items.map((item, idx) => {
    const profileKey = entryProfileKey(item.opp.sourceType, item.opp.opportunityType);
    const dk = makeEconomicDedupeKey(profileKey, item.opp.opportunityId);
    const repeatCount = dedupeCounts.get(dk) ?? 0;
    const base = item.capacity.estimatedNetEdge;
    const penalty = cfg.repeatPenaltyWeight * Math.log(1 + repeatCount);
    const sortKey = base - penalty;
    return { item, sortKey, repeatCount, dk, profileKey, idx, base };
  });

  enriched.sort((a, b) => {
    if (b.sortKey !== a.sortKey) return b.sortKey - a.sortKey;
    return a.idx - b.idx;
  });

  const penaltyApplied = enriched.filter((e) => e.repeatCount > 0).length;
  const sumR = enriched.reduce((s, e) => s + e.repeatCount, 0);
  const avgR = enriched.length > 0 ? sumR / enriched.length : null;
  const distinct = new Set(enriched.map((e) => e.dk)).size;

  const priorityDiagnostics = enriched.slice(0, 8).map((e) => ({
    opportunityId: e.item.opp.opportunityId,
    profileKey: e.profileKey,
    estimatedNetEdge: round6(e.base),
    repeatCount: e.repeatCount,
    sortKey: round6(e.sortKey),
  }));

  return {
    sorted: enriched.map((e) => e.item),
    lastCycle: {
      at,
      batchSize: items.length,
      repeatedCandidatePenaltyAppliedCount: penaltyApplied,
      distinctDedupeKeysInBatch: distinct,
      avgRepeatCountInBatch: avgR != null ? round6(avgR) : null,
      evaluationPriorityDiagnostics: priorityDiagnostics,
    },
  };
}

function round6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}
