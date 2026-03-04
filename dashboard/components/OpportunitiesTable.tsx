"use client";

import { useState } from "react";
import type { AnalyticsData } from "@/lib/api";
import { num, pct, truncateId } from "@/lib/format";
import ExplainModal from "./ExplainModal";

interface Props {
  data: AnalyticsData | null;
}

export default function OpportunitiesTable({ data }: Props) {
  const [showModal, setShowModal] = useState(false);
  const rows = data?.opportunity_ranking ?? [];

  return (
    <>
      <div className="terminal-panel h-full flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-terminal-cyan text-xs font-bold tracking-widest uppercase">
            Melhores Oportunidades
          </h2>
          <button
            onClick={() => setShowModal(true)}
            className="text-[10px] text-terminal-muted border border-terminal-border rounded px-2 py-0.5 hover:text-terminal-cyan hover:border-terminal-cyan transition-colors"
          >
            Explicar
          </button>
        </div>

        {rows.length === 0 ? (
          <div className="flex-1 flex items-center justify-center text-terminal-muted text-xs">
            Nenhuma oportunidade detectada no momento.
          </div>
        ) : (
          <div className="overflow-auto flex-1 -mx-4 px-4">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-terminal-muted border-b border-terminal-border">
                  <th className="text-left py-2 font-medium">Mercado</th>
                  <th className="text-right py-2 font-medium">Lucro/h</th>
                  <th className="text-right py-2 font-medium">Confiança</th>
                  <th className="text-right py-2 font-medium">Score</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 10).map((row, i) => {
                  const score = row.score ?? 0;
                  const isHigh = score > 5;
                  return (
                    <tr
                      key={i}
                      className={`border-b border-terminal-border/50 ${
                        isHigh ? "bg-terminal-green/5" : ""
                      }`}
                    >
                      <td className="py-2 text-terminal-text font-medium">
                        {truncateId(row.market_id, 20)}
                      </td>
                      <td className="py-2 text-right text-terminal-green">
                        {num(row.expected_pnl_per_hour)} bps/h
                      </td>
                      <td className="py-2 text-right text-terminal-text">
                        {pct(row.fill_rate)}
                      </td>
                      <td className="py-2 text-right">
                        <span
                          className={`font-bold ${isHigh ? "text-terminal-green" : "text-terminal-text"}`}
                        >
                          {num(score)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <ExplainModal
        open={showModal}
        onClose={() => setShowModal(false)}
        title="O que são as oportunidades?"
      >
        <p>
          O sistema analisa centenas de mercados e identifica onde os preços
          estão incorretos em relação ao valor real.
        </p>
        <p className="text-terminal-muted">
          <span className="text-terminal-text font-semibold">Lucro/h</span> —
          estimativa de ganho por hora se executar a estratégia neste mercado.
        </p>
        <p className="text-terminal-muted">
          <span className="text-terminal-text font-semibold">Confiança</span> —
          probabilidade de conseguir executar o trade com sucesso.
        </p>
        <p className="text-terminal-muted">
          <span className="text-terminal-text font-semibold">Score</span> —
          nota final combinando lucro, confiança e liquidez. Quanto maior,
          melhor.
        </p>
      </ExplainModal>
    </>
  );
}
