/**
 * Diretório base para estados JSON da trilha de observação (probes + ladder history).
 *
 * Local (default): `<cwd>/.paper/<ficheiro>.json`
 * Railway / volume: definir `PAPER_STATE_DIR=/data` (montar volume nesse path) → ficheiros em `/data/<ficheiro>.json`
 *
 * Overrides por ficheiro (caminho absoluto) mantêm precedência: POCKET_ECON_STATE_PATH,
 * POCKET_EXEC_STATE_PATH, MINIMAL_PAPER_STATE_PATH, SYSTEM_LADDER_HISTORY_PATH.
 */

import path from "path";

export const PAPER_TRAIL_FILENAMES = {
  pocketEconomics: "pocket-economics-state.json",
  pocketExecution: "pocket-execution-state.json",
  minimalPaperExecution: "minimal-paper-execution-state.json",
  systemLadderHistory: "system-ladder-history.json",
  /** Histórico append-only (JSONL) — validação final venue-native evento 31552. */
  finalNegativeRisk31552History: "final-negative-risk-31552-history.jsonl",
  /** Histórico append-only (JSONL) — monitor BTC narrow reaction 1823774 / 1823775. */
  reachAprilBtcReactionMonitorHistory: "reach-april-btc-reaction-monitor-history.jsonl",
  /** Histórico append-only (JSONL) — monitor cross-venue anchor refinado só mercado 1823789. */
  crossVenueAnchor1823789MonitorHistory: "cross-venue-anchor-1823789-monitor-history.jsonl",
  /** Metadados operacionais (JSON, não append-only) — worker final-neg-risk-31552. */
  finalNegativeRisk31552Meta: "final-negative-risk-31552-meta.json",
  /** Metadados operacionais — worker reach-april-btc-monitor. */
  reachAprilBtcReactionMonitorMeta: "reach-april-btc-reaction-monitor-meta.json",
  /** Metadados operacionais — worker cross-venue-anchor-1823789-monitor. */
  crossVenueAnchor1823789MonitorMeta: "cross-venue-anchor-1823789-monitor-meta.json",
  /** Histórico append-only (JSONL) — paper execution narrow só mercado 1823789 (gated). */
  crossVenueAnchor1823789PaperExecutionHistory: "cross-venue-anchor-1823789-paper-execution-history.jsonl",
  /** Metadados operacionais — worker paper execution 1823789. */
  crossVenueAnchor1823789PaperExecutionMeta: "cross-venue-anchor-1823789-paper-execution-meta.json",
  /** Diagnóstico append-only (JSONL) — conectividade CLOB no runtime do paper-exec 1823789. */
  crossVenueAnchor1823789ClobConnectivityHistory:
    "cross-venue-anchor-1823789-clob-connectivity-history.jsonl",
  /** Append-only (JSONL) — auditoria shadow/micro-live 1823789 (sem submissão real). */
  crossVenueAnchor1823789MicroLiveShadowAudit:
    "cross-venue-anchor-1823789-micro-live-shadow-audit.jsonl",
  /** Resumo compacto agregado no worker — consumido pelo dashboard scoreboard. */
  finalNegativeRisk31552Summary: "final-negative-risk-31552-summary.json",
  reachAprilBtcReactionMonitorSummary: "reach-april-btc-reaction-monitor-summary.json",
  crossVenueAnchor1823789MonitorSummary: "cross-venue-anchor-1823789-monitor-summary.json",
} as const;

export function resolvePaperStateDir(cwd: string = process.cwd()): string {
  const raw = process.env.PAPER_STATE_DIR?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return path.join(cwd, ".paper");
}

export function defaultPaperTrailStatePath(
  filename: string,
  cwd: string = process.cwd(),
): string {
  return path.join(resolvePaperStateDir(cwd), filename);
}
