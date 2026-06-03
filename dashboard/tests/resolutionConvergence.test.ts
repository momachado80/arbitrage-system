import {
  scoreConvergenceCandidate,
  CONV_NEAR_EXTREME_MID,
  CONV_NEAR_RESOLUTION_DAYS,
  type ConvergenceCostModel,
} from "../lib/resolutionConvergence";
import { describe, test, assertEqual, assertTrue } from "./_assert";

const COSTS: ConvergenceCostModel = {
  costOfCapitalAnnual: 0.1,
  gasPerTxUsd: 0.03,
  targetSizeUsd: 100,
  umaHaircutByCategory: { crypto_feed: 0.001, sports: 0.003, electoral: 0.006, unknown: 0.01 },
};

function near(a: number, b: number, tol = 1e-5): boolean {
  return Math.abs(a - b) < tol;
}

describe("tests/resolutionConvergence.test.ts", () => {
  test("constantes expostas", () => {
    assertEqual(CONV_NEAR_EXTREME_MID, 0.92, "extremo 0.92");
    assertEqual(CONV_NEAR_RESOLUTION_DAYS, 7, "resolução 7d");
  });

  test("YES perto de 1 + perto da resolução → convergence_candidate_needs_fair_value", () => {
    const r = scoreConvergenceCandidate(
      { yesMid: 0.96, yesAsk: 0.97, noAsk: 0.05, daysToResolution: 2, category: "crypto_feed" },
      COSTS,
    );
    assertEqual(r.favoredSide, "YES", "favorito YES");
    assertTrue(near(r.discount, 0.03), "desconto 1−0.97");
    assertEqual(r.verdict, "convergence_candidate_needs_fair_value", "candidato (precisa fair value)");
    assertTrue(r.anchorable, "crypto é ancorável");
  });

  test("NO perto de 1 (YES perto de 0) → favorito NO", () => {
    const r = scoreConvergenceCandidate(
      { yesMid: 0.04, yesAsk: 0.06, noAsk: 0.97, daysToResolution: 3, category: "sports" },
      COSTS,
    );
    assertEqual(r.favoredSide, "NO", "favorito NO");
    assertTrue(near(r.favoredAsk ?? -1, 0.97), "compra NO ask");
    assertTrue(!r.anchorable, "sports não ancorável");
  });

  test("mid no meio (0.55) → not_near_extreme", () => {
    const r = scoreConvergenceCandidate(
      { yesMid: 0.55, yesAsk: 0.56, noAsk: 0.46, daysToResolution: 2, category: "crypto_feed" },
      COSTS,
    );
    assertEqual(r.verdict, "not_near_extreme", "não está em extremo");
    assertEqual(r.favoredSide, null, "sem favorito");
  });

  test("extremo mas longe da resolução (60d) → not_near_resolution", () => {
    const r = scoreConvergenceCandidate(
      { yesMid: 0.96, yesAsk: 0.97, noAsk: 0.05, daysToResolution: 60, category: "crypto_feed" },
      COSTS,
    );
    assertEqual(r.verdict, "not_near_resolution", "longe da resolução");
  });

  test("desconto menor que custos mecânicos → costs_exceed_discount", () => {
    /** ask 0.995 → desconto 0.5%; UMA unknown 1% já supera. */
    const r = scoreConvergenceCandidate(
      { yesMid: 0.99, yesAsk: 0.995, noAsk: 0.02, daysToResolution: 2, category: "unknown" },
      COSTS,
    );
    assertTrue(r.discount < 0.01, "desconto pequeno");
    assertEqual(r.verdict, "costs_exceed_discount", "custos comem o desconto");
  });

  test("yield anualizado: desconto 3% em 2 dias é alto, mas é descritivo (não risk-free)", () => {
    const r = scoreConvergenceCandidate(
      { yesMid: 0.96, yesAsk: 0.97, noAsk: 0.05, daysToResolution: 2, category: "crypto_feed" },
      COSTS,
    );
    /** (0.03/0.97)*(365/2) ≈ 5.64 (564% a.a.) — reflete o risco residual, não almoço grátis. */
    assertTrue(r.convergenceYieldAnnualized > 5, "yield anualizado alto");
  });

  test("verdict nunca é 'viable' — convergência exige fair value", () => {
    const r = scoreConvergenceCandidate(
      { yesMid: 0.97, yesAsk: 0.975, noAsk: 0.04, daysToResolution: 1, category: "crypto_feed" },
      COSTS,
    );
    assertTrue(
      r.verdict === "convergence_candidate_needs_fair_value" ||
        r.verdict === "costs_exceed_discount",
      "nunca declara edge mecânico",
    );
  });

  test("defensivo: preços fora de (0,1) → invalid_input", () => {
    assertEqual(
      scoreConvergenceCandidate(
        { yesMid: 1.2, yesAsk: 0.97, noAsk: 0.05, daysToResolution: 2, category: "crypto_feed" },
        COSTS,
      ).verdict,
      "invalid_input",
      "mid inválido",
    );
  });

  test("lockup escala com dias até resolução", () => {
    const short = scoreConvergenceCandidate(
      { yesMid: 0.96, yesAsk: 0.97, noAsk: 0.05, daysToResolution: 1, category: "crypto_feed" },
      COSTS,
    );
    const long = scoreConvergenceCandidate(
      { yesMid: 0.96, yesAsk: 0.97, noAsk: 0.05, daysToResolution: 7, category: "crypto_feed" },
      COSTS,
    );
    assertTrue(long.netDiscountAfterCosts < short.netDiscountAfterCosts, "mais dias, mais lockup, menos net");
  });
});
