/**
 * Narrow Challenger Helpers — filtros explícitos e auditáveis para o subset-alvo.
 * Mesma semântica de buckets do structuralPersistenceValidation (audit).
 * Hipótese: narrow challenger criado para testar persistência local do subset
 * 562794+567561 + edge 2-5% + fill 0.25-0.5, sem alterar regime global.
 */

/** Edge bucket 2-5%: >= 0.02 e < 0.05 (decimal) */
export function isEdgeBucketMatch(capturableEdgeAtEntry: number, targetBucket: "2-5%"): boolean {
  if (targetBucket !== "2-5%") return false;
  return capturableEdgeAtEntry >= 0.02 && capturableEdgeAtEntry < 0.05;
}

/** Fill bucket 0.25-0.5: >= 0.25 e < 0.5. Usa fillRatio no momento da decisão. */
export function isFillBucketMatch(fillRatio: number | null | undefined, targetBucket: "0.25-0.5"): boolean {
  if (targetBucket !== "0.25-0.5") return false;
  if (fillRatio == null || fillRatio < 0) return false;
  return fillRatio >= 0.25 && fillRatio < 0.5;
}

/** Pair key exato */
export function isPairMatch(pairKey: string | null | undefined, targetPairKey: string): boolean {
  if (!pairKey) return false;
  return pairKey === targetPairKey;
}

export interface NarrowChallengerTarget {
  pairKey: string;
  capturableEdgeBucket: "2-5%";
  fillRatioBucket: "0.25-0.5";
}

/** Target do narrow challenger 562794+567561 edge 2-5% fill 0.25-0.5 */
export const NARROW_CHALLENGER_TARGET: NarrowChallengerTarget = {
  pairKey: "562794+567561",
  capturableEdgeBucket: "2-5%",
  fillRatioBucket: "0.25-0.5",
};
