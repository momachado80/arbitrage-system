/**
 * Rolling ELO para seleções (puro) — estilo World Football Elo Ratings.
 *
 * Atualização zero-soma com multiplicador de goleada. Calibra os ratings que
 * alimentam soccerEloModel → poissonGoalsModel. Sem rede, sem I/O, sem execução.
 */

export interface EloUpdateParams {
  /** Fator K (peso do ajuste por jogo). */
  k: number;
  /** Vantagem de mando em pontos ELO (0 em campo neutro). */
  homeAdvantageElo: number;
}

export const DEFAULT_ROLLING_ELO: EloUpdateParams = {
  k: 40,
  homeAdvantageElo: 65,
};

export const ELO_INITIAL = 1500;

/** Expectativa de resultado (prob. de vitória do mandante, empate diluído). */
export function expectedScore(homeElo: number, awayElo: number, homeAdv: number): number {
  return 1 / (1 + 10 ** (-((homeElo + homeAdv) - awayElo) / 400));
}

/** Multiplicador de goleada (World Football Elo). */
export function goalDiffMultiplier(goalDiff: number): number {
  const a = Math.abs(goalDiff);
  if (a <= 1) return 1;
  if (a === 2) return 1.5;
  return (11 + a) / 8;
}

export interface EloUpdate {
  home: number;
  away: number;
  delta: number;
}

/** Atualiza ELO de ambos os times após uma partida (zero-soma). */
export function updateElo(
  homeElo: number,
  awayElo: number,
  homeGoals: number,
  awayGoals: number,
  neutral: boolean,
  params: EloUpdateParams = DEFAULT_ROLLING_ELO,
): EloUpdate {
  const homeAdv = neutral ? 0 : params.homeAdvantageElo;
  const we = expectedScore(homeElo, awayElo, homeAdv);
  const w = homeGoals > awayGoals ? 1 : homeGoals === awayGoals ? 0.5 : 0;
  const g = goalDiffMultiplier(homeGoals - awayGoals);
  const delta = params.k * g * (w - we);
  return { home: homeElo + delta, away: awayElo - delta, delta };
}
