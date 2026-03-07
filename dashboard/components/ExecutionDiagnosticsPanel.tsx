"use client";

import type { AnalyticsData } from "@/lib/api";
import { num } from "@/lib/format";

interface Props {
  data: AnalyticsData | null;
}

const REASON_LABELS: Record<string, string> = {
  INSUFFICIENT_LIQUIDITY: "Liquidez insuficiente",
  SLIPPAGE_TOO_HIGH: "Slippage alto",
  EDGE_BELOW_THRESHOLD: "Edge abaixo do limite",
  LATENCY_RISK: "Risco de latência",
  TRADE_SIZE_TOO_SMALL: "Tamanho muito pequeno",
  EXECUTION_ENGINE_DISABLED: "Motor desativado",
  UNKNOWN: "Desconhecido",
};

export default function ExecutionDiagnosticsPanel({ data }: Props) {
  const stats = data?.rejection_stats;
  const hasData = stats != null && stats.totalRejected > 0;

  const totalRejected = stats?.totalRejected ?? 0;
  const countsByReason = stats?.countsByReason ?? {};
  const paperExecuted =
    (data?.paper_trading?.closedTrades ?? 0) + (data?.paper_trading?.activeTrades ?? 0);
  const shadowExecuted = (data?.shadow_profiles ?? []).reduce(
    (s, p) => s + (p.activeTrades ?? 0) + (p.closedTrades ?? 0),
    0
  );
  const totalExecuted = paperExecuted + shadowExecuted;
  const totalAttempts = totalRejected + totalExecuted;
  const successRatio =
    totalAttempts > 0 ? (totalExecuted / totalAttempts) * 100 : 0;

  const sortedReasons = Object.entries(countsByReason)
    .filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className="terminal-panel">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-terminal-cyan text-xs font-bold tracking-widest uppercase">
          Diagnóstico de Execução
        </h2>
      </div>

      {!hasData && totalAttempts === 0 ? (
        <div className="text-terminal-muted text-xs">
          Nenhuma rejeição registrada. Aguardando dados do pipeline.
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <div>
              <div className="text-terminal-muted text-[10px] mb-0.5">
                Total rejeitados
              </div>
              <div className="text-terminal-text text-sm font-bold">
                {num(totalRejected)}
              </div>
            </div>
            <div>
              <div className="text-terminal-muted text-[10px] mb-0.5">
                Execuções bem-sucedidas
              </div>
              <div className="text-terminal-green text-sm font-bold">
                {num(totalExecuted)}
              </div>
            </div>
            <div>
              <div className="text-terminal-muted text-[10px] mb-0.5">
                Taxa de sucesso
              </div>
              <div
                className={`text-sm font-bold ${
                  successRatio >= 50
                    ? "text-terminal-green"
                    : successRatio >= 20
                      ? "text-terminal-yellow"
                      : "text-terminal-red"
                }`}
              >
                {totalAttempts > 0 ? successRatio.toFixed(1) : "—"}%
              </div>
            </div>
          </div>

          {sortedReasons.length > 0 && (
            <div>
              <div className="text-terminal-muted text-[10px] mb-2">
                Rejeições por motivo
              </div>
              <div className="space-y-1.5">
                {sortedReasons.map(([reason, count]) => {
                  const pct =
                    totalRejected > 0 ? (count / totalRejected) * 100 : 0;
                  return (
                    <div
                      key={reason}
                      className="flex items-center justify-between text-xs"
                    >
                      <span className="text-terminal-text">
                        {REASON_LABELS[reason] ?? reason}
                      </span>
                      <span className="text-terminal-muted">
                        {num(count)} ({pct.toFixed(0)}%)
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {stats?.timestamp && (
            <div className="text-terminal-muted text-[10px] pt-2 border-t border-terminal-border">
              Última atualização: {new Date(stats.timestamp).toLocaleTimeString()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
