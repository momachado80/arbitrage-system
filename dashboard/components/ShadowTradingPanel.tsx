"use client";

import { useState } from "react";
import type { AnalyticsData } from "@/lib/api";
import { num, pct, bps } from "@/lib/format";
import ExplainModal from "./ExplainModal";

interface Props {
  data: AnalyticsData | null;
}

export default function ShadowTradingPanel({ data }: Props) {
  const [showModal, setShowModal] = useState(false);
  const s = data?.shadow_trading_summary;
  const hasData = s && s.shadow_trades > 0;

  return (
    <>
      <div className="terminal-panel h-full flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-terminal-cyan text-xs font-bold tracking-widest uppercase">
            Validação do Sistema
          </h2>
          <button
            onClick={() => setShowModal(true)}
            className="text-[10px] text-terminal-muted border border-terminal-border rounded px-2 py-0.5 hover:text-terminal-cyan hover:border-terminal-cyan transition-colors"
          >
            Explicar
          </button>
        </div>

        {!hasData ? (
          <div className="flex-1 flex items-center justify-center text-terminal-muted text-xs">
            Nenhum trade simulado ainda. O sistema está coletando dados.
          </div>
        ) : (
          <div className="space-y-4 flex-1">
            <div className="grid grid-cols-2 gap-3">
              <Metric label="Trades simulados" value={String(s.shadow_trades)} />
              <Metric
                label="Taxa de execução"
                value={pct(s.mean_fill_rate)}
                color="text-terminal-green"
              />
              <Metric
                label="Trades lucrativos"
                value={pct(s.profitable_pct)}
                color={
                  s.profitable_pct >= 0.6
                    ? "text-terminal-green"
                    : s.profitable_pct >= 0.4
                      ? "text-terminal-yellow"
                      : "text-terminal-red"
                }
              />
              <Metric
                label="PnL médio"
                value={`${num(s.mean_expected_pnl)} bps`}
                color={
                  s.mean_expected_pnl > 0
                    ? "text-terminal-green"
                    : "text-terminal-red"
                }
              />
            </div>

            <div className="border-t border-terminal-border pt-3">
              <div className="text-terminal-muted text-[10px] mb-1">
                Lucro esperado total
              </div>
              <div
                className={`text-2xl font-bold ${
                  s.total_expected_pnl > 0
                    ? "text-terminal-green terminal-glow-green"
                    : "text-terminal-red"
                }`}
              >
                {num(s.total_expected_pnl)} bps
              </div>
            </div>
          </div>
        )}
      </div>

      <ExplainModal
        open={showModal}
        onClose={() => setShowModal(false)}
        title="O que é a validação do sistema?"
      >
        <p>
          O sistema executa trades simulados (paper trading) para validar se as
          oportunidades detectadas seriam lucrativas na prática.
        </p>
        <p className="text-terminal-muted">
          <span className="text-terminal-text font-semibold">
            Taxa de execução
          </span>{" "}
          — percentual dos trades que seriam preenchidos com sucesso.
        </p>
        <p className="text-terminal-muted">
          <span className="text-terminal-text font-semibold">
            Trades lucrativos
          </span>{" "}
          — percentual dos trades simulados que tiveram lucro positivo.
        </p>
        <p className="text-terminal-muted">
          <span className="text-terminal-text font-semibold">
            PnL esperado
          </span>{" "}
          — lucro total acumulado em basis points (1 bps = 0.01%).
        </p>
        <p>
          Quanto mais altos esses números, mais confiável é o sistema.
        </p>
      </ExplainModal>
    </>
  );
}

function Metric({
  label,
  value,
  color = "text-terminal-text",
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div>
      <div className="text-terminal-muted text-[10px] mb-0.5">{label}</div>
      <div className={`text-sm font-bold ${color}`}>{value}</div>
    </div>
  );
}
