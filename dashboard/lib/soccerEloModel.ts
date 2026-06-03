/**
 * Soccer ELO → expected goals (puro).
 *
 * Mapeia ratings ELO de duas seleções + vantagem de mando para os λ (gols
 * esperados) que alimentam o Poisson model. Modelo linear de supremacia:
 *   supremacy_goals = (eloHome − eloAway + homeAdvElo) / eloPerGoalSupremacy
 *   λH = (avgTotalGoals + supremacy) / 2
 *   λA = (avgTotalGoals − supremacy) / 2   (com piso positivo)
 *
 * TODOS os parâmetros são calibráveis — a calibração É o jogo. Defaults razoáveis
 * para futebol de seleções, mas devem ser ajustados por backtest antes de qualquer
 * uso com capital. Sem rede, sem I/O, sem execução.
 */

export interface EloModelParams {
  /** Vantagem de mando em pontos ELO (0 para jogo neutro, ex. Copa em sede única). */
  homeAdvantageElo: number;
  /** Pontos ELO equivalentes a 1 gol de supremacia esperada. */
  eloPerGoalSupremacy: number;
  /** Média de gols totais da liga/competição (~2.6 seleções, ~2.7 clubes top). */
  avgTotalGoals: number;
  /** Piso para cada λ (evita 0; nenhum time tem expectativa de 0 gols). */
  lambdaFloor: number;
}

export const DEFAULT_ELO_PARAMS: EloModelParams = {
  homeAdvantageElo: 65,
  eloPerGoalSupremacy: 120,
  avgTotalGoals: 2.6,
  lambdaFloor: 0.15,
};

export interface ExpectedGoals {
  lambdaHome: number;
  lambdaAway: number;
  supremacyGoals: number;
}

export function expectedGoalsFromElo(
  homeElo: number,
  awayElo: number,
  neutralVenue: boolean,
  params: EloModelParams = DEFAULT_ELO_PARAMS,
): ExpectedGoals {
  const homeAdv = neutralVenue ? 0 : params.homeAdvantageElo;
  const supremacyGoals = (homeElo - awayElo + homeAdv) / params.eloPerGoalSupremacy;
  const lambdaHome = Math.max(params.lambdaFloor, (params.avgTotalGoals + supremacyGoals) / 2);
  const lambdaAway = Math.max(params.lambdaFloor, (params.avgTotalGoals - supremacyGoals) / 2);
  return {
    lambdaHome: Math.round(lambdaHome * 1e6) / 1e6,
    lambdaAway: Math.round(lambdaAway * 1e6) / 1e6,
    supremacyGoals: Math.round(supremacyGoals * 1e6) / 1e6,
  };
}
