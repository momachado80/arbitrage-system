/**
 * Poisson Goals Model — motor de probabilidade puro para partidas de futebol.
 *
 * Modelo baseline: gols do mandante ~ Poisson(λH), gols do visitante ~ Poisson(λA),
 * INDEPENDENTES. É o ponto de partida clássico (Maher 1982). A correlação leve entre
 * placares (Dixon-Coles) é um refinamento posterior — documentado como limitação.
 *
 * Funções puras, sem rede, sem I/O, sem execução, sem .paper, sem microcapital.
 * Apenas estatística: dado λH e λA, calcula distribuições de placar/total/resultado.
 */

const MAX_GOALS_DEFAULT = 12;

/** PMF de Poisson: P(X=k) = e^{-λ} λ^k / k!  (iterativo, estável para k pequeno). */
export function poissonPmf(lambda: number, k: number): number {
  if (!(lambda >= 0) || !Number.isInteger(k) || k < 0) return 0;
  if (lambda === 0) return k === 0 ? 1 : 0;
  // log-space para estabilidade: ln P = -λ + k·ln λ − ln(k!)
  let lnFact = 0;
  for (let i = 2; i <= k; i++) lnFact += Math.log(i);
  const lnP = -lambda + k * Math.log(lambda) - lnFact;
  return Math.exp(lnP);
}

/** CDF de Poisson: P(X ≤ k). */
export function poissonCdf(lambda: number, k: number): number {
  if (k < 0) return 0;
  let acc = 0;
  for (let i = 0; i <= k; i++) acc += poissonPmf(lambda, i);
  return Math.min(1, acc);
}

/** P(placar exato i-j) = P(H=i)·P(A=j) sob independência. */
export function pExactScore(lambdaHome: number, lambdaAway: number, i: number, j: number): number {
  return poissonPmf(lambdaHome, i) * poissonPmf(lambdaAway, j);
}

/**
 * P(total de gols over/under da linha). Soma de dois Poisson independentes é
 * Poisson(λH+λA), então o total usa um único λ. Linha típica: 0.5, 1.5, 2.5, 5.5.
 */
export function pOver(lambdaHome: number, lambdaAway: number, line: number): number {
  const lambdaTotal = lambdaHome + lambdaAway;
  /** over N.5 ⇔ total ≥ ⌈line⌉ = floor(line)+1. */
  const threshold = Math.floor(line) + 1;
  return Math.max(0, 1 - poissonCdf(lambdaTotal, threshold - 1));
}

export function pUnder(lambdaHome: number, lambdaAway: number, line: number): number {
  return Math.max(0, 1 - pOver(lambdaHome, lambdaAway, line));
}

export interface MatchResultProbs {
  homeWin: number;
  draw: number;
  awayWin: number;
}

/** P(vitória mandante / empate / vitória visitante) somando a grade de placares. */
export function matchResultProbs(
  lambdaHome: number,
  lambdaAway: number,
  maxGoals = MAX_GOALS_DEFAULT,
): MatchResultProbs {
  let homeWin = 0;
  let draw = 0;
  let awayWin = 0;
  for (let i = 0; i <= maxGoals; i++) {
    const pi = poissonPmf(lambdaHome, i);
    for (let j = 0; j <= maxGoals; j++) {
      const p = pi * poissonPmf(lambdaAway, j);
      if (i > j) homeWin += p;
      else if (i === j) draw += p;
      else awayWin += p;
    }
  }
  return { homeWin, draw, awayWin };
}

/**
 * P(handicap asiático/europeu inteiro-e-meio): "home (−line)" = mandante vence por
 * mais de `line` gols. Ex.: home (−1.5) ⇒ P(H − A ≥ 2).
 */
export function pSpreadCover(
  lambdaHome: number,
  lambdaAway: number,
  favored: "home" | "away",
  line: number,
  maxGoals = MAX_GOALS_DEFAULT,
): number {
  const margin = Math.floor(line) + 1; // −1.5 ⇒ cobre se margem ≥ 2
  let p = 0;
  for (let i = 0; i <= maxGoals; i++) {
    const pi = poissonPmf(lambdaHome, i);
    for (let j = 0; j <= maxGoals; j++) {
      const diff = favored === "home" ? i - j : j - i;
      if (diff >= margin) p += pi * poissonPmf(lambdaAway, j);
    }
  }
  return p;
}

/** P(ambos marcam) = P(H≥1)·P(A≥1). */
export function pBothTeamsScore(lambdaHome: number, lambdaAway: number): number {
  return (1 - poissonPmf(lambdaHome, 0)) * (1 - poissonPmf(lambdaAway, 0));
}

/** Massa de probabilidade total da grade (sanidade: deve ≈ 1 para maxGoals alto). */
export function gridMass(lambdaHome: number, lambdaAway: number, maxGoals = MAX_GOALS_DEFAULT): number {
  let m = 0;
  for (let i = 0; i <= maxGoals; i++) {
    const pi = poissonPmf(lambdaHome, i);
    for (let j = 0; j <= maxGoals; j++) m += pi * poissonPmf(lambdaAway, j);
  }
  return m;
}
