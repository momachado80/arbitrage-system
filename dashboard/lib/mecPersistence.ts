/**
 * MEC Persistence — sumarização pura das re-observações de uma cesta flagged.
 *
 * Pergunta que responde: um gap observado num snapshot SOBREVIVE minutos depois
 * (capturável por operador pequeno) ou evapora (comido por bot / ruído de book)?
 * É a variável que faltava para precificar "micro-edge recorrente": sem
 * frequência × duração × capacidade, qualquer estimativa de renda é chute.
 *
 * Funções puras, sem rede, sem I/O, sem execução, sem .paper, sem microcapital.
 */

export interface PersistenceObservation {
  /** Segundos desde a descoberta (0 = re-observação imediata). */
  offsetSec: number;
  /** Gap no melhor nível: buy ⇒ 1 − Σ best_ask; sell ⇒ Σ best_bid − 1. */
  grossAtBest: number;
  /** Gap no VWAP por tamanho-alvo em USD (null = profundidade não preenche). */
  grossAtVwapBySize: Record<string, number | null>;
  /** Capacidade aproximada preenchível (USD) no top do livro. */
  fillableUsdApprox: number;
}

export type PersistenceVerdict =
  | "insufficient_observations"
  | "persistent"
  | "decayed"
  | "transient";

export interface PersistenceSummary {
  nObservations: number;
  positiveAtBestCount: number;
  /** Fração das observações com grossAtBest > minGross. */
  persistenceScore: number;
  firstPositive: boolean;
  lastPositive: boolean;
  /** Maior capacidade aproximada vista (USD) entre as observações positivas. */
  maxFillableUsdWhilePositive: number;
  verdict: PersistenceVerdict;
}

/**
 * Regras:
 *  - < 2 observações ⇒ insufficient_observations
 *  - persistence ≥ 0.75 E última positiva ⇒ persistent
 *  - primeira positiva E última não-positiva ⇒ decayed (era real, evaporou)
 *  - resto ⇒ transient (ruído/flicker)
 */
export function summarizePersistence(
  observations: PersistenceObservation[],
  minGross: number,
): PersistenceSummary {
  const obs = [...observations].sort((a, b) => a.offsetSec - b.offsetSec);
  const n = obs.length;
  const isPos = (o: PersistenceObservation): boolean => o.grossAtBest > minGross;
  const positives = obs.filter(isPos);
  const positiveAtBestCount = positives.length;
  const persistenceScore = n > 0 ? positiveAtBestCount / n : 0;
  const firstPositive = n > 0 ? isPos(obs[0]!) : false;
  const lastPositive = n > 0 ? isPos(obs[n - 1]!) : false;
  const maxFillableUsdWhilePositive = positives.reduce(
    (m, o) => Math.max(m, o.fillableUsdApprox),
    0,
  );

  let verdict: PersistenceVerdict;
  if (n < 2) verdict = "insufficient_observations";
  else if (persistenceScore >= 0.75 && lastPositive) verdict = "persistent";
  else if (firstPositive && !lastPositive) verdict = "decayed";
  else verdict = "transient";

  return {
    nObservations: n,
    positiveAtBestCount,
    persistenceScore,
    firstPositive,
    lastPositive,
    maxFillableUsdWhilePositive,
    verdict,
  };
}
