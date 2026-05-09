/** Evita ciclos de import: invalidação do cache de perfis de safety sem depender do portfolio. */

let epoch = 0;

export function bumpPaperSafetyProfileCacheEpoch(): void {
  epoch += 1;
}

export function getPaperSafetyProfileCacheEpoch(): number {
  return epoch;
}
