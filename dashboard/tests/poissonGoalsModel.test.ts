import {
  poissonPmf,
  poissonCdf,
  pExactScore,
  pOver,
  pUnder,
  matchResultProbs,
  pSpreadCover,
  pBothTeamsScore,
  gridMass,
} from "../lib/poissonGoalsModel";
import { describe, test, assertTrue, assertEqual } from "./_assert";

function near(a: number, b: number, tol = 1e-5): boolean {
  return Math.abs(a - b) < tol;
}

describe("tests/poissonGoalsModel.test.ts", () => {
  test("poissonPmf valores conhecidos (λ=1)", () => {
    assertTrue(near(poissonPmf(1, 0), Math.exp(-1)), "P(0)=e^-1");
    assertTrue(near(poissonPmf(1, 1), Math.exp(-1)), "P(1)=e^-1");
    assertTrue(near(poissonPmf(1, 2), Math.exp(-1) / 2), "P(2)=e^-1/2");
  });

  test("poissonPmf λ=2.6 P(0)=e^-2.6", () => {
    assertTrue(near(poissonPmf(2.6, 0), Math.exp(-2.6)), "P(0)");
  });

  test("poissonPmf defensivo: k negativo ou não-inteiro → 0; λ=0 → indicador em 0", () => {
    assertEqual(poissonPmf(2, -1), 0, "k<0");
    assertEqual(poissonPmf(2, 1.5), 0, "k não inteiro");
    assertEqual(poissonPmf(0, 0), 1, "λ=0,k=0");
    assertEqual(poissonPmf(0, 1), 0, "λ=0,k=1");
  });

  test("poissonCdf soma a PMF", () => {
    const c = poissonCdf(2.6, 2);
    const manual = poissonPmf(2.6, 0) + poissonPmf(2.6, 1) + poissonPmf(2.6, 2);
    assertTrue(near(c, manual), "cdf=Σpmf");
  });

  test("pOver 0.5 = 1 − P(0 gols)", () => {
    /** λtotal = 1.6+1.0 = 2.6 → over0.5 = 1−e^-2.6 = 0.925726 */
    assertTrue(near(pOver(1.6, 1.0, 0.5), 1 - Math.exp(-2.6)), "over 0.5");
  });

  test("pOver 2.5 com λtotal=2.6 ≈ 0.481569", () => {
    assertTrue(near(pOver(1.6, 1.0, 2.5), 0.481569, 1e-4), "over 2.5");
  });

  test("pUnder = 1 − pOver", () => {
    assertTrue(near(pUnder(1.6, 1.0, 2.5) + pOver(1.6, 1.0, 2.5), 1), "complementares");
  });

  test("pExactScore(1.5,1.0,1,1) = P(H=1)·P(A=1)", () => {
    const expected = poissonPmf(1.5, 1) * poissonPmf(1.0, 1);
    assertTrue(near(pExactScore(1.5, 1.0, 1, 1), expected), "placar exato");
  });

  test("matchResultProbs soma ≈ 1 e empate maior entre iguais", () => {
    const r = matchResultProbs(1.3, 1.3);
    assertTrue(near(r.homeWin + r.draw + r.awayWin, 1, 1e-6), "soma 1");
    assertTrue(near(r.homeWin, r.awayWin, 1e-6), "simétrico entre iguais");
  });

  test("matchResultProbs favorito forte → homeWin domina", () => {
    const r = matchResultProbs(2.8, 0.6);
    assertTrue(r.homeWin > 0.7, "mandante forte vence muito");
    assertTrue(r.homeWin > r.awayWin, "home > away");
  });

  test("pSpreadCover home (−1.5) = P(H−A ≥ 2)", () => {
    /** brute force pequeno para conferir. */
    let manual = 0;
    for (let i = 0; i <= 12; i++)
      for (let j = 0; j <= 12; j++) if (i - j >= 2) manual += poissonPmf(2.0, i) * poissonPmf(0.5, j);
    assertTrue(near(pSpreadCover(2.0, 0.5, "home", 1.5), manual), "spread home -1.5");
  });

  test("pBothTeamsScore = P(H≥1)·P(A≥1)", () => {
    const expected = (1 - poissonPmf(1.5, 0)) * (1 - poissonPmf(1.0, 0));
    assertTrue(near(pBothTeamsScore(1.5, 1.0), expected), "btts");
  });

  test("gridMass ≈ 1 (sanidade da grade)", () => {
    assertTrue(near(gridMass(2.0, 1.5), 1, 1e-6), "massa total ~1");
  });
});
