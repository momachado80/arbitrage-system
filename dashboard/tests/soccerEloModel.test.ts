import {
  expectedGoalsFromElo,
  DEFAULT_ELO_PARAMS,
} from "../lib/soccerEloModel";
import { describe, test, assertTrue, assertEqual } from "./_assert";

function near(a: number, b: number, tol = 1e-4): boolean {
  return Math.abs(a - b) < tol;
}

describe("tests/soccerEloModel.test.ts", () => {
  test("times iguais em campo neutro → λ iguais = avgTotal/2", () => {
    const g = expectedGoalsFromElo(1800, 1800, true);
    assertTrue(near(g.supremacyGoals, 0), "sem supremacia");
    assertTrue(near(g.lambdaHome, 1.3), "λH = 2.6/2");
    assertTrue(near(g.lambdaAway, 1.3), "λA = 2.6/2");
  });

  test("vantagem de mando desloca λ a favor do mandante", () => {
    const g = expectedGoalsFromElo(1800, 1800, false);
    /** supremacy = 65/120 = 0.541667; λH=(2.6+0.5417)/2=1.5708, λA=1.0292 */
    assertTrue(near(g.supremacyGoals, 65 / 120), "supremacia do mando");
    assertTrue(g.lambdaHome > g.lambdaAway, "mandante favorecido");
    assertTrue(near(g.lambdaHome, (2.6 + 65 / 120) / 2), "λH");
  });

  test("campo neutro zera vantagem de mando", () => {
    const home = expectedGoalsFromElo(1900, 1700, false);
    const neutral = expectedGoalsFromElo(1900, 1700, true);
    assertTrue(home.supremacyGoals > neutral.supremacyGoals, "mando soma supremacia");
    assertTrue(near(neutral.supremacyGoals, 200 / 120), "neutro = só diff ELO");
  });

  test("favorito extremo → λA bate o piso", () => {
    const g = expectedGoalsFromElo(2200, 1500, true);
    assertEqual(g.lambdaAway, DEFAULT_ELO_PARAMS.lambdaFloor, "λA no piso");
    assertTrue(g.lambdaHome > 2.5, "λH alto");
  });

  test("params custom alteram o mapeamento", () => {
    const tight = expectedGoalsFromElo(1900, 1700, true, {
      ...DEFAULT_ELO_PARAMS,
      eloPerGoalSupremacy: 400,
    });
    const loose = expectedGoalsFromElo(1900, 1700, true, {
      ...DEFAULT_ELO_PARAMS,
      eloPerGoalSupremacy: 80,
    });
    assertTrue(loose.supremacyGoals > tight.supremacyGoals, "menos ELO/gol → mais supremacia");
  });
});
