import {
  brierScore,
  logLoss,
  calibrationBins,
  baseRateBrier,
  brierSkillScore,
  type ForecastSample,
} from "../lib/forecastCalibration";
import { describe, test, assertTrue, assertEqual } from "./_assert";

function near(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) < tol;
}

describe("tests/forecastCalibration.test.ts", () => {
  test("brierScore conhecido", () => {
    const s: ForecastSample[] = [
      { predicted: 0.9, outcome: 1 },
      { predicted: 0.1, outcome: 0 },
    ];
    assertTrue(near(brierScore(s) ?? -1, (0.01 + 0.01) / 2), "brier 0.01");
  });

  test("brierScore perfeito = 0; chute 50/50 = 0.25", () => {
    assertEqual(brierScore([{ predicted: 1, outcome: 1 }, { predicted: 0, outcome: 0 }]), 0, "perfeito");
    assertTrue(
      near(brierScore([{ predicted: 0.5, outcome: 1 }, { predicted: 0.5, outcome: 0 }]) ?? -1, 0.25),
      "chute",
    );
  });

  test("brierScore vazio → null", () => {
    assertEqual(brierScore([]), null, "vazio");
  });

  test("logLoss penaliza confiança errada e é finito em p=0/1 (clamp)", () => {
    const good = logLoss([{ predicted: 0.99, outcome: 1 }]) ?? Infinity;
    const bad = logLoss([{ predicted: 0.01, outcome: 1 }]) ?? -1;
    assertTrue(bad > good, "errar com confiança custa mais");
    assertTrue(Number.isFinite(logLoss([{ predicted: 1, outcome: 0 }]) ?? Infinity), "clamp evita infinito");
  });

  test("baseRateBrier = brier do preditor de frequência constante", () => {
    /** 3 eventos, base rate 2/3. brier = média (2/3 − y)². */
    const s: ForecastSample[] = [
      { predicted: 0.4, outcome: 1 },
      { predicted: 0.4, outcome: 1 },
      { predicted: 0.4, outcome: 0 },
    ];
    const base = 2 / 3;
    const manual = ((base - 1) ** 2 + (base - 1) ** 2 + (base - 0) ** 2) / 3;
    assertTrue(near(baseRateBrier(s) ?? -1, manual), "base rate brier");
  });

  test("brierSkillScore: modelo = base rate → skill 0; modelo melhor → >0", () => {
    /** Modelo que prevê exatamente a base rate não agrega → skill ~0. */
    const flat: ForecastSample[] = [
      { predicted: 0.5, outcome: 1 },
      { predicted: 0.5, outcome: 0 },
    ];
    assertTrue(near(brierSkillScore(flat) ?? -99, 0), "skill 0 quando = base rate");

    const skilled: ForecastSample[] = [
      { predicted: 0.95, outcome: 1 },
      { predicted: 0.05, outcome: 0 },
      { predicted: 0.9, outcome: 1 },
      { predicted: 0.1, outcome: 0 },
    ];
    assertTrue((brierSkillScore(skilled) ?? -1) > 0.5, "modelo bom tem skill positivo");
  });

  test("calibrationBins agrupa e compara previsto vs empírico", () => {
    const s: ForecastSample[] = [
      { predicted: 0.05, outcome: 0 },
      { predicted: 0.07, outcome: 0 },
      { predicted: 0.95, outcome: 1 },
      { predicted: 0.92, outcome: 1 },
    ];
    const bins = calibrationBins(s, 10);
    assertEqual(bins.length, 10, "10 bins");
    assertEqual(bins[0]!.count, 2, "2 no bin baixo");
    assertEqual(bins[9]!.count, 2, "2 no bin alto");
    assertTrue((bins[0]!.empiricalRate ?? -1) === 0, "bin baixo empírico 0");
    assertTrue((bins[9]!.empiricalRate ?? -1) === 1, "bin alto empírico 1");
  });
});
