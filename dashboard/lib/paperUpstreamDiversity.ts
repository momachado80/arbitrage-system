/**
 * Selecção upstream diversity-aware sobre o batch já filtrado (sem expandir universo).
 *
 * **Family key (proxy):** `clusterId` do grafo quando existe; senão `m:<marketIds ordenados>`;
 * fallback `o:<opportunityId>`. Agrupa cross-market pelo par de mercados de forma estável.
 */

import type { NormalizedPaperOpportunity } from "./paperTypes";
import type { CapacityResult } from "./capitalCapacityEngine";

export type PaperUpstreamDiversityPolicySnapshot = {
  enabled: boolean;
  maxPerFamilyInBatch: number;
  noveltyFraction: number;
};

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

export function getUpstreamDiversityPolicySnapshot(): PaperUpstreamDiversityPolicySnapshot {
  return {
    enabled: envBool("PAPER_UPSTREAM_DIVERSITY_ENABLED", true),
    maxPerFamilyInBatch: Math.max(1, Math.floor(envNum("PAPER_UPSTREAM_MAX_PER_FAMILY", 4))),
    noveltyFraction: Math.max(0, Math.min(1, envNum("PAPER_UPSTREAM_NOVELTY_FRACTION", 0.25))),
  };
}

export function getOpportunityFamilyKey(opp: NormalizedPaperOpportunity): string {
  if (opp.clusterId != null && String(opp.clusterId).length > 0) {
    return `c:${opp.clusterId}`;
  }
  const ids = opp.marketsInvolved.map((m) => m.marketId).sort();
  if (ids.length > 0) return `m:${ids.join("|")}`;
  return `o:${opp.opportunityId}`;
}

export type UpstreamSelectionDiagnostics = {
  totalCandidatesBeforeDiversitySelection: number;
  totalCandidatesAfterDiversitySelection: number;
  slotsReservedForNovelty: number;
  slotsReservedForMerit: number;
  clustersTrimmedByBudget: number;
  candidatesPromotedForNovelty: number;
  candidatesDroppedForClusterDominance: number;
  uniqueFamiliesInInput: number;
  uniqueFamiliesInOutput: number;
};

type Item = { opp: NormalizedPaperOpportunity; capacity: CapacityResult };

/**
 * 1) Round-robin por família com ordem de famílias por novidade (menos vistas primeiro).
 * 2) Cada família contribui no máximo `maxPerFamilyInBatch` até esgotar ou encher n.
 * 3) Backfill na ordem de exploração original para garantir comprimento n (relaxa cap residual).
 */
export function applyUpstreamDiversitySelection(
  explorationOrdered: Item[],
  familyRecentCount: (familyKey: string) => number
): { selected: Item[]; diagnostics: UpstreamSelectionDiagnostics } {
  const policy = getUpstreamDiversityPolicySnapshot();
  const n = explorationOrdered.length;

  const baseDiag = (inputFamilies: number, outputFamilies: number): UpstreamSelectionDiagnostics => ({
    totalCandidatesBeforeDiversitySelection: n,
    totalCandidatesAfterDiversitySelection: n,
    slotsReservedForNovelty: Math.floor(n * policy.noveltyFraction),
    slotsReservedForMerit: n - Math.floor(n * policy.noveltyFraction),
    clustersTrimmedByBudget: 0,
    candidatesPromotedForNovelty: 0,
    candidatesDroppedForClusterDominance: 0,
    uniqueFamiliesInInput: inputFamilies,
    uniqueFamiliesInOutput: outputFamilies,
  });

  if (!policy.enabled || n <= 1) {
    const uf = new Set(explorationOrdered.map((x) => getOpportunityFamilyKey(x.opp)));
    return {
      selected: explorationOrdered,
      diagnostics: { ...baseDiag(uf.size, uf.size) },
    };
  }

  const noveltySlots = Math.floor(n * policy.noveltyFraction);
  const meritSlots = n - noveltySlots;

  const groupOrder: string[] = [];
  const groups = new Map<string, Item[]>();
  for (const it of explorationOrdered) {
    const fk = getOpportunityFamilyKey(it.opp);
    if (!groups.has(fk)) {
      groups.set(fk, []);
      groupOrder.push(fk);
    }
    groups.get(fk)!.push(it);
  }

  const sortedKeys = [...groupOrder].sort(
    (a, b) => familyRecentCount(a) - familyRecentCount(b) || a.localeCompare(b)
  );

  const out: Item[] = [];
  const takenPerFamily = new Map<string, number>();
  const posInGroup = new Map<string, number>();
  for (const fk of groupOrder) posInGroup.set(fk, 0);

  let progressed = true;
  while (progressed && out.length < n) {
    progressed = false;
    for (const fk of sortedKeys) {
      if (out.length >= n) break;
      if ((takenPerFamily.get(fk) ?? 0) >= policy.maxPerFamilyInBatch) continue;
      const arr = groups.get(fk)!;
      const pos = posInGroup.get(fk) ?? 0;
      if (pos >= arr.length) continue;
      out.push(arr[pos]!);
      posInGroup.set(fk, pos + 1);
      takenPerFamily.set(fk, (takenPerFamily.get(fk) ?? 0) + 1);
      progressed = true;
    }
  }

  const afterRR = out.length;
  const droppedDominance = n - afterRR;

  const picked = new Set(out.map((x) => x.opp.opportunityId));
  let trimmed = 0;
  for (const it of explorationOrdered) {
    if (out.length >= n) break;
    if (picked.has(it.opp.opportunityId)) continue;
    const fk = getOpportunityFamilyKey(it.opp);
    if ((takenPerFamily.get(fk) ?? 0) >= policy.maxPerFamilyInBatch) {
      trimmed++;
      continue;
    }
    out.push(it);
    picked.add(it.opp.opportunityId);
    takenPerFamily.set(fk, (takenPerFamily.get(fk) ?? 0) + 1);
  }

  for (const it of explorationOrdered) {
    if (out.length >= n) break;
    if (picked.has(it.opp.opportunityId)) continue;
    out.push(it);
    picked.add(it.opp.opportunityId);
  }

  const uniqueOut = new Set(out.map((x) => getOpportunityFamilyKey(x.opp)));
  const lowSeen = sortedKeys.length > 0 ? familyRecentCount(sortedKeys[0]!) : 0;
  let promoted = 0;
  for (let i = 0; i < Math.min(out.length, noveltySlots + 5); i++) {
    if (familyRecentCount(getOpportunityFamilyKey(out[i]!.opp)) <= lowSeen) promoted++;
  }

  return {
    selected: out.slice(0, n),
    diagnostics: {
      totalCandidatesBeforeDiversitySelection: n,
      totalCandidatesAfterDiversitySelection: Math.min(out.length, n),
      slotsReservedForNovelty: noveltySlots,
      slotsReservedForMerit: meritSlots,
      clustersTrimmedByBudget: trimmed,
      candidatesPromotedForNovelty: promoted,
      candidatesDroppedForClusterDominance: droppedDominance,
      uniqueFamiliesInInput: groupOrder.length,
      uniqueFamiliesInOutput: uniqueOut.size,
    },
  };
}
