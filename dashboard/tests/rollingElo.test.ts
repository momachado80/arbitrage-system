import {
  expectedScore,
  goalDiffMultiplier,
  updateElo,
  DEFAULT_ROLLING_ELO,
  ELO_INITIAL,
} from "../lib/rollingElo";
import { describe, test, assertTrue, assertEqual } from "./_assert";

function near(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) < tol;
}

describe("tests/rollingElo.test.ts", () => {
  test("expectedScore: times iguais sem mando = 0.5", () => {
    assertTrue(near(expectedScore(1500, 1500, 0), 0.5), "0.5");
  });

  test("expectedScore: mando aumenta expectativa do mandante", () => {
    assertTrue(expectedScore(1500, 1500, 65) > 0.5, "mando > 0.5");
  });

  test("expectedScore: +400 ELO ≈ 0.909", () => {
    assertTrue(near(expectedScore(1900, 1500, 0), 1 / (1 + 10 ** -1)), "10:1 odds");
  });

  test("goalDiffMultiplier: 1→1, 2→1.5, 3→1.75, 4→1.875", () => {
    assertEqual(goalDiffMultiplier(1), 1, "gd1");
    assertEqual(goalDiffMultiplier(-1), 1, "gd-1 simétrico");
    assertEqual(goalDiffMultiplier(2), 1.5, "gd2");
    assertTrue(near(goalDiffMultiplier(3), 14 / 8), "gd3");
    assertTrue(near(goalDiffMultiplier(4), 15 / 8), "gd4");
  });

  test("updateElo zero-soma: ganho do mandante = perda do visitante", () => {
    const u = updateElo(1500, 1500, 2, 0, true);
    assertTrue(near(u.home - 1500, -(u.away - 1500)), "zero-soma");
    assertTrue(u.home > 1500, "vencedor sobe");
    assertTrue(u.away < 1500, "perdedor cai");
  });

  test("updateElo: empate entre iguais (neutro) ≈ sem mudança", () => {
    const u = updateElo(1500, 1500, 1, 1, true);
    assertTrue(near(u.delta, 0), "empate iguais = delta 0");
  });

  test("updateElo: goleada move mais que vitória magra", () => {
    const narrow = updateElo(1500, 1500, 1, 0, true).delta;
    const blowout = updateElo(1500, 1500, 4, 0, true).delta;
    assertTrue(blowout > narrow, "goleada pesa mais");
  });

  test("updateElo: favorito que vence ganha menos que zebra que vence", () => {
    const favWins = updateElo(1900, 1500, 1, 0, true).delta;
    const dogWins = updateElo(1500, 1900, 1, 0, true).delta;
    assertTrue(dogWins > favWins, "zebra ganha mais pontos");
  });

  test("constantes expostas", () => {
    assertEqual(ELO_INITIAL, 1500, "inicial 1500");
    assertEqual(DEFAULT_ROLLING_ELO.k, 40, "K 40");
  });
});
