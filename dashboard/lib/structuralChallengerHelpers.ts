/**
 * Structural Challenger Helpers — filtros explícitos pair × fill × edge.
 * Mesma semântica de buckets do structuralPersistenceValidation / audit.
 */

/** Conjunto de pares alvo para challengers estruturais */
export const STRUCTURAL_PAIR_SET = new Set([
  "540817+565065",
  "540817+562187",
  "540817+573647",
  "540817+540818",
  "556108+567561",
  "556108+562187",
  "540818+556108",
]);

export function isStructuralPairMatch(pairKey: string | null | undefined): boolean {
  if (!pairKey) return false;
  return STRUCTURAL_PAIR_SET.has(pairKey);
}

/** Fill bucket 0.1-0.25: > 0.1 e <= 0.25 (mesma lógica do audit) */
export function isStructuralFillBucketMatch(fillRatio: number | null | undefined): boolean {
  if (fillRatio == null || fillRatio < 0) return false;
  return fillRatio > 0.1 && fillRatio <= 0.25;
}

/** Edge bucket >5%: >= 0.05 (decimal) */
export function isStructuralEdgeBucketGt5Match(capturableEdge: number): boolean {
  return capturableEdge >= 0.05;
}

/** Bucket de edge para registro (mesma lógica do structuralPersistenceValidation) */
export function getCapturableEdgeBucket(edge: number): string {
  if (edge < 0.005) return "0-0.5%";
  if (edge < 0.01) return "0.5-1%";
  if (edge < 0.02) return "1-2%";
  if (edge < 0.05) return "2-5%";
  return ">5%";
}

/** Bucket de fill para registro */
export function getFillRatioBucket(r: number | null | undefined): string {
  if (r == null || r < 0) return "_unknown";
  if (r <= 0.1) return "0-0.1";
  if (r <= 0.25) return "0.1-0.25";
  if (r <= 0.5) return "0.25-0.5";
  if (r <= 0.75) return "0.5-0.75";
  return "0.75-1.0";
}

export interface StructuralChallengerTarget {
  pairKeys: readonly string[];
  fillRatioBucket: "0.1-0.25";
  capturableEdgeBucket: ">5%" | null;
}
