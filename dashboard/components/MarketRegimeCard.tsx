"use client";

import { useState } from "react";
import type { AnalyticsData } from "@/lib/api";
import { pct, regimeColor, qualityLevel, qualityColor } from "@/lib/format";
import ExplainModal from "./ExplainModal";

interface Props {
  data: AnalyticsData | null;
}

export default function MarketRegimeCard({ data }: Props) {
  const [showModal, setShowModal] = useState(false);

  const regime = data?.market_regime;
  const score = regime?.regime_score ?? 0;
  const state = regime?.state;
  const c = regime?.components;

  return (
    <>
      <div className="terminal-panel h-full flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-terminal-cyan text-xs font-bold tracking-widest uppercase">
            Qualidade do Mercado
          </h2>
          <button
            onClick={() => setShowModal(true)}
            className="text-[10px] text-terminal-muted border border-terminal-border rounded px-2 py-0.5 hover:text-terminal-cyan hover:border-terminal-cyan transition-colors"
          >
            Explicar
          </button>
        </div>

        <div className={`text-5xl font-bold mb-3 ${regimeColor(state)}`}>
          {pct(score)}
        </div>

        <p className="text-terminal-muted text-xs mb-4 leading-relaxed">
          {score >= 0.65
            ? "Erros de preço estão durando tempo suficiente para executar trades com sucesso."
            : score >= 0.35
              ? "Condições moderadas. Alguns erros de preço detectados."
              : "Erros de preço desaparecem rápido demais. Momento desfavorável."}
        </p>

        <div className="mt-auto space-y-2 text-xs">
          <Row
            label="Persistência dos erros"
            value={c?.edge_persistence}
          />
          <Row label="Liquidez" value={c?.liquidity_score} />
          <Row
            label="Oportunidades"
            value={c?.opportunity_density}
          />
          <Row
            label="Competição"
            value={c?.competition}
            invert
          />
        </div>
      </div>

      <ExplainModal
        open={showModal}
        onClose={() => setShowModal(false)}
        title="O que significa qualidade do mercado?"
      >
        <p>
          Este indicador estima se as condições atuais são boas para encontrar
          e executar trades lucrativos.
        </p>
        <p>Ele considera:</p>
        <ul className="list-disc list-inside space-y-1 text-terminal-muted">
          <li>
            <span className="text-terminal-text">Persistência dos erros</span>{" "}
            — quanto tempo os erros de preço duram antes de serem corrigidos
          </li>
          <li>
            <span className="text-terminal-text">Liquidez</span> — se há
            volume suficiente para executar trades
          </li>
          <li>
            <span className="text-terminal-text">Oportunidades</span> — quantos
            erros de preço aparecem por hora
          </li>
          <li>
            <span className="text-terminal-text">Competição</span> — nível de
            competição entre bots (quanto maior, pior)
          </li>
        </ul>
        <p className="text-terminal-muted">
          Score acima de 65% = boas condições. Abaixo de 35% = desfavorável.
        </p>
      </ExplainModal>
    </>
  );
}

function Row({
  label,
  value,
  invert = false,
}: {
  label: string;
  value?: number;
  invert?: boolean;
}) {
  const v = value ?? 0;
  const display = qualityLevel(invert ? 1 - v : v);
  const color = qualityColor(invert ? 1 - v : v);
  return (
    <div className="flex justify-between">
      <span className="text-terminal-muted">{label}</span>
      <span className={`font-semibold ${color}`}>{display}</span>
    </div>
  );
}
