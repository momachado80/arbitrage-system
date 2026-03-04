"use client";

import type { AnalyticsData } from "@/lib/api";
import { regimeLabel, regimeColor, regimeDot, pct } from "@/lib/format";

interface Props {
  data: AnalyticsData | null;
  lastUpdate: Date | null;
  error: string | null;
}

export default function Header({ data, lastUpdate, error }: Props) {
  const regime = data?.market_regime;
  const state = regime?.state;
  const score = regime?.regime_score ?? 0;

  const descriptions: Record<string, string> = {
    high_edge: "O sistema detectou boas oportunidades de trading agora.",
    neutral: "Condições de mercado moderadas. Oportunidades limitadas.",
    hostile: "Condições desfavoráveis. Poucas oportunidades detectadas.",
  };

  return (
    <header className="terminal-panel flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
      <div className="flex items-center gap-4">
        <div>
          <h1 className="text-terminal-text text-lg font-bold tracking-widest">
            POLYMARKET ALPHA TERMINAL
          </h1>
          <div className="flex items-center gap-2 mt-1">
            <span className={regimeDot(state)} />
            <span className={`text-sm font-semibold ${regimeColor(state)}`}>
              Estado do mercado: {regimeLabel(state)}
            </span>
          </div>
        </div>
      </div>

      <div className="text-right space-y-1">
        <div className="text-terminal-muted text-xs">
          Qualidade do mercado:{" "}
          <span className={`font-bold text-sm ${regimeColor(state)}`}>
            {pct(score)}
          </span>
        </div>
        <p className="text-terminal-muted text-xs max-w-xs">
          {descriptions[state ?? ""] ?? "Aguardando dados..."}
        </p>
        <div className="text-terminal-muted text-[10px]">
          {error ? (
            <span className="text-terminal-red">Erro: {error}</span>
          ) : lastUpdate ? (
            <>Atualizado: {lastUpdate.toLocaleTimeString("pt-BR")}</>
          ) : (
            "Conectando..."
          )}
        </div>
      </div>
    </header>
  );
}
