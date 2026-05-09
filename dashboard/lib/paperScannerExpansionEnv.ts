/**
 * Política de expansão do scanner upstream (só env + tipos; sem dependências de diagnóstico).
 */

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

export type PaperUpstreamScannerExpansionPolicySnapshot = {
  enabled: boolean;
  /** Máximo de candidatos adicionais por ciclo (graph + cross combinados). */
  maxExtraCandidatesPerCycle: number;
  /** Tecto de extras vindos só do pool de grafo (além do top-K global). */
  maxExtraGraphCandidatesPerCycle: number;
  /** Máximo de novos pares cross (janela limitada) por ciclo. */
  maxNewCrossPairingsPerCycle: number;
  /** Tamanho da janela de mercados a considerar para pares (custo O(W²)). */
  crossPairingProbeWindow: number;
  /** Limite de linhas do `cachedGraphRaw` a percorrer por ciclo (sem normalizar todas). */
  maxGraphRawProbePerCycle: number;
  /** Tecto de chamadas `normalizeGraph` na fase de expansão (após pré-rank). */
  maxGraphNormalizeAttempts: number;
};

export function getUpstreamScannerExpansionPolicySnapshot(): PaperUpstreamScannerExpansionPolicySnapshot {
  return {
    enabled: envBool("PAPER_SCANNER_EXPANSION_ENABLED", true),
    maxExtraCandidatesPerCycle: Math.max(0, Math.floor(envNum("PAPER_SCANNER_EXPANSION_MAX_EXTRA", 24))),
    maxExtraGraphCandidatesPerCycle: Math.max(0, Math.floor(envNum("PAPER_SCANNER_EXPANSION_MAX_GRAPH_EXTRA", 16))),
    maxNewCrossPairingsPerCycle: Math.max(0, Math.floor(envNum("PAPER_SCANNER_EXPANSION_MAX_PAIRINGS", 8))),
    crossPairingProbeWindow: Math.max(4, Math.floor(envNum("PAPER_SCANNER_EXPANSION_PAIR_WINDOW", 18))),
    maxGraphRawProbePerCycle: Math.max(50, Math.floor(envNum("PAPER_SCANNER_EXPANSION_MAX_GRAPH_RAW_PROBE", 1200))),
    maxGraphNormalizeAttempts: Math.max(8, Math.floor(envNum("PAPER_SCANNER_EXPANSION_MAX_GRAPH_NORMALIZE", 64))),
  };
}
