"use client";

import { useState } from "react";
import type { AnalyticsData } from "@/lib/api";
import { num, pct } from "@/lib/format";
import ExplainModal from "./ExplainModal";

interface Props {
  data: AnalyticsData | null;
}

const SHADOW_PROFILE_IDS = ["shadow_100", "shadow_1000"] as const;

function getDisplayProfiles(data: AnalyticsData | null) {
  const fromApi = data?.shadow_profiles ?? [];
  return SHADOW_PROFILE_IDS.map((id) => {
    const p = fromApi.find((x) => x.profileId === id);
    return p ?? {
      profileId: id,
      label: id === "shadow_100" ? "Shadow 100 USD" : "Shadow 1000 USD",
      startingCapital: id === "shadow_100" ? 100 : 1000,
      currentEquity: id === "shadow_100" ? 100 : 1000,
      realizedPnL: 0,
      activeTrades: 0,
      closedTrades: 0,
    };
  });
}

export default function ShadowTradingPanel({ data }: Props) {
  const [showModal, setShowModal] = useState(false);
  const s = data?.shadow_trading_summary;
  const paper = s?.paper_trading ?? data?.paper_trading;
  const shadowProfiles = getDisplayProfiles(data);
  const hasShadow = shadowProfiles.some((p) => p.activeTrades > 0 || p.closedTrades > 0);
  const hasData =
    s &&
    (s.shadow_trades > 0 ||
      (paper && (paper.activeTrades > 0 || paper.closedTrades > 0)) ||
      hasShadow ||
      shadowProfiles.length > 0);

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
          <div className="flex-1 flex flex-col justify-center text-terminal-muted text-xs space-y-3">
            <div className="text-center">Nenhum trade simulado ainda. O sistema está coletando dados.</div>
            <div className="border-t border-terminal-border pt-3">
              <div className="text-terminal-muted text-[10px] mb-2">Shadow Simulation</div>
              <div className="space-y-2">
                {shadowProfiles.map((p) => (
                  <div key={p.profileId} className="flex flex-wrap gap-3 text-xs">
                    <span className="font-medium">{p.label}</span>
                    <span>({p.profileId})</span>
                    <span>Capital: {num(p.startingCapital)}</span>
                    <span>Equity: {num(p.currentEquity)}</span>
                    <span>PnL: {num(p.realizedPnL)}</span>
                    <span>{p.activeTrades} ativos / {p.closedTrades} fechados</span>
                  </div>
                ))}
              </div>
            </div>
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

            {paper && (paper.closedTrades > 0 || paper.activeTrades > 0) && (
              <div className="border-t border-terminal-border pt-3">
                <div className="text-terminal-muted text-[10px] mb-1">Paper Trading</div>
                <div className="flex flex-wrap gap-3 text-xs">
                  <span>PnL: <span className={paper.realizedPnL >= 0 ? "text-terminal-green" : "text-terminal-red"}>{num(paper.realizedPnL)}</span></span>
                  <span>Equity: {num(paper.currentEquity)}</span>
                  <span>Trades: {paper.activeTrades} ativos / {paper.closedTrades} fechados</span>
                  {paper.closedTrades > 0 && <span>Win rate: {pct(paper.winRate)}</span>}
                </div>
              </div>
            )}

            <div className="border-t border-terminal-border pt-3">
              <div className="text-terminal-muted text-[10px] mb-2">Shadow Simulation</div>
              <div className="space-y-2">
                {shadowProfiles.map((p) => (
                  <div key={p.profileId} className="flex flex-wrap gap-3 text-xs">
                    <span className="font-medium">{p.label}</span>
                    <span className="text-terminal-muted">({p.profileId})</span>
                    <span>Capital: {num(p.startingCapital)}</span>
                    <span>Equity: {num(p.currentEquity)}</span>
                    <span>PnL: <span className={p.realizedPnL >= 0 ? "text-terminal-green" : "text-terminal-red"}>{num(p.realizedPnL)}</span></span>
                    <span>{p.activeTrades} ativos / {p.closedTrades} fechados</span>
                  </div>
                ))}
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
