"use client";

import { useState } from "react";
import type { AnalyticsData } from "@/lib/api";
import { num, pct, bps, truncateId } from "@/lib/format";
import ExplainModal from "./ExplainModal";

interface Props {
  data: AnalyticsData | null;
}

export default function ArbitrageTable({ data }: Props) {
  const [showModal, setShowModal] = useState(false);
  const [expanded, setExpanded] = useState<number | null>(null);
  const rows = data?.cross_market_trade_candidates ?? [];

  return (
    <>
      <div className="terminal-panel h-full flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-terminal-cyan text-xs font-bold tracking-widest uppercase">
            Arbitragens Cross-Market
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
            Nenhuma arbitragem cross-market ativa.
          </div>
        ) : (
          <div className="overflow-auto flex-1 -mx-4 px-4">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-terminal-muted border-b border-terminal-border">
                  <th className="text-left py-2 font-medium">Basket</th>
                  <th className="text-right py-2 font-medium">Trades</th>
                  <th className="text-right py-2 font-medium">Edge</th>
                  <th className="text-right py-2 font-medium">Lucro</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 10).map((row, i) => (
                  <>
                    <tr
                      key={`row-${i}`}
                      className="border-b border-terminal-border/50 cursor-pointer hover:bg-terminal-border/20 transition-colors"
                      onClick={() => setExpanded(expanded === i ? null : i)}
                    >
                      <td className="py-2 text-terminal-text font-mono">
                        <span className="text-terminal-muted mr-1">
                          {expanded === i ? "▾" : "▸"}
                        </span>
                        {truncateId(row.basket_id, 12)}
                      </td>
                      <td className="py-2 text-right text-terminal-text">
                        {row.n_legs} trades
                      </td>
                      <td className="py-2 text-right text-terminal-yellow font-semibold">
                        {num(row.edge_bps)} bps
                      </td>
                      <td className="py-2 text-right text-terminal-green">
                        {num(row.basket_expected_pnl)} bps
                      </td>
                    </tr>
                    {expanded === i && (
                      <tr key={`detail-${i}`}>
                        <td
                          colSpan={4}
                          className="py-3 px-4 bg-terminal-bg/50 border-b border-terminal-border/30"
                        >
                          <div className="text-xs space-y-2">
                            <p className="text-terminal-cyan font-semibold">
                              Por que esta arbitragem existe?
                            </p>
                            <p className="text-terminal-muted">
                              {row.type === "exclusivity" ? (
                                <>
                                  Esses mercados deveriam somar 100%, mas
                                  atualmente somam{" "}
                                  <span className="text-terminal-yellow font-semibold">
                                    {num((row.edge_bps / 100) + 100)}%
                                  </span>
                                  . A estratégia é vender todos os outcomes
                                  proporcionalmente.
                                </>
                              ) : (
                                <>
                                  Mispricing detectado entre mercados
                                  complementares. Edge de{" "}
                                  <span className="text-terminal-yellow">
                                    {bps(row.edge_bps)}
                                  </span>
                                  .
                                </>
                              )}
                            </p>
                            <div className="flex gap-4 text-terminal-muted">
                              <span>
                                Fill rate:{" "}
                                <span className="text-terminal-text">
                                  {pct(row.basket_fill_rate)}
                                </span>
                              </span>
                              <span>
                                Alpha captura:{" "}
                                <span className="text-terminal-text">
                                  {pct(row.basket_alpha_capture_ratio)}
                                </span>
                              </span>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <ExplainModal
        open={showModal}
        onClose={() => setShowModal(false)}
        title="O que é arbitragem cross-market?"
      >
        <p>
          Arbitragem acontece quando mercados relacionados têm preços
          inconsistentes entre si.
        </p>
        <p>
          <span className="text-terminal-text font-semibold">Exemplo:</span> Se
          um mercado de eleição tem 3 candidatos, as probabilidades deveriam
          somar exatamente 100%. Se somam 107%, há 7% de lucro garantido
          vendendo todos os outcomes.
        </p>
        <p className="text-terminal-muted">
          <span className="text-terminal-text font-semibold">Basket</span> — um
          grupo de trades coordenados que formam a estratégia.
        </p>
        <p className="text-terminal-muted">
          <span className="text-terminal-text font-semibold">Edge</span> — o
          tamanho da ineficiência de preço detectada.
        </p>
        <p className="text-terminal-muted">
          <span className="text-terminal-text font-semibold">Lucro</span> —
          estimativa de lucro após custos de execução (slippage + taxas).
        </p>
      </ExplainModal>
    </>
  );
}
