"use client";

import { useState } from "react";
import type { AnalyticsData } from "@/lib/api";
import { num } from "@/lib/format";
import ExplainModal from "./ExplainModal";

interface Props {
  data: AnalyticsData | null;
}

export default function RiskPanel({ data }: Props) {
  const [showModal, setShowModal] = useState(false);
  const r = data?.risk_engine_state;
  const hasData = r != null;

  const drawdown = r?.drawdown_remaining ?? 0;
  const isSafe = drawdown > 200;
  const isWarning = drawdown > 0 && drawdown <= 200;

  const statusColor = isSafe
    ? "text-terminal-green"
    : isWarning
      ? "text-terminal-yellow"
      : "text-terminal-red";

  const statusLabel = isSafe
    ? "SEGURO"
    : isWarning
      ? "ATENÇÃO"
      : "CRÍTICO";

  const statusDot = isSafe
    ? "pulse-dot pulse-dot-green"
    : isWarning
      ? "pulse-dot pulse-dot-yellow"
      : "pulse-dot pulse-dot-red";

  return (
    <>
      <div className="terminal-panel">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-terminal-cyan text-xs font-bold tracking-widest uppercase">
            Risco
          </h2>
          <div className="flex items-center gap-2">
            <span className={statusDot} />
            <span className={`text-xs font-bold ${statusColor}`}>
              {hasData ? statusLabel : "—"}
            </span>
            <button
              onClick={() => setShowModal(true)}
              className="text-[10px] text-terminal-muted border border-terminal-border rounded px-2 py-0.5 hover:text-terminal-cyan hover:border-terminal-cyan transition-colors ml-2"
            >
              Explicar
            </button>
          </div>
        </div>

        {!hasData ? (
          <div className="text-terminal-muted text-xs">
            Motor de risco não ativo.
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <div className="text-terminal-muted text-[10px] mb-0.5">
                Capital em uso
              </div>
              <div className="text-terminal-text text-sm font-bold">
                {num(r.total_inventory)}
              </div>
            </div>
            <div>
              <div className="text-terminal-muted text-[10px] mb-0.5">
                Mercados ativos
              </div>
              <div className="text-terminal-text text-sm font-bold">
                {r.markets_with_position}
              </div>
            </div>
            <div>
              <div className="text-terminal-muted text-[10px] mb-0.5">
                PnL acumulado
              </div>
              <div
                className={`text-sm font-bold ${
                  r.pnl >= 0 ? "text-terminal-green" : "text-terminal-red"
                }`}
              >
                {num(r.pnl)} bps
              </div>
            </div>
            <div>
              <div className="text-terminal-muted text-[10px] mb-0.5">
                Drawdown restante
              </div>
              <div className={`text-sm font-bold ${statusColor}`}>
                {num(drawdown)} bps
              </div>
            </div>
          </div>
        )}
      </div>

      <ExplainModal
        open={showModal}
        onClose={() => setShowModal(false)}
        title="O que são os controles de risco?"
      >
        <p>
          O motor de risco impede que o sistema assuma posições excessivas que
          poderiam causar perdas grandes.
        </p>
        <p className="text-terminal-muted">
          <span className="text-terminal-text font-semibold">
            Capital em uso
          </span>{" "}
          — valor total das posições abertas em todos os mercados.
        </p>
        <p className="text-terminal-muted">
          <span className="text-terminal-text font-semibold">
            Drawdown restante
          </span>{" "}
          — quanto o sistema ainda pode perder antes de parar automaticamente.
          Se chegar a zero, nenhum trade novo é permitido.
        </p>
        <div className="flex items-center gap-2">
          <span className="pulse-dot pulse-dot-green" />
          <span className="text-terminal-muted">
            <span className="text-terminal-green font-semibold">SEGURO</span> — drawdown
            restante {'>'} 200 bps
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="pulse-dot pulse-dot-yellow" />
          <span className="text-terminal-muted">
            <span className="text-terminal-yellow font-semibold">ATENÇÃO</span>{" "}
            — drawdown se aproximando do limite
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="pulse-dot pulse-dot-red" />
          <span className="text-terminal-muted">
            <span className="text-terminal-red font-semibold">CRÍTICO</span> —
            limite atingido, trades bloqueados
          </span>
        </div>
      </ExplainModal>
    </>
  );
}
