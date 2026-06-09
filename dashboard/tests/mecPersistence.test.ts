import {
  summarizePersistence,
  type PersistenceObservation,
} from "../lib/mecPersistence";
import { describe, test, assertEqual, assertTrue } from "./_assert";

const MIN_GROSS = 0.003;

function obs(offsetSec: number, grossAtBest: number, fillable = 50): PersistenceObservation {
  return {
    offsetSec,
    grossAtBest,
    grossAtVwapBySize: { "10": grossAtBest, "30": null, "100": null },
    fillableUsdApprox: fillable,
  };
}

describe("tests/mecPersistence.test.ts", () => {
  test("menos de 2 observações → insufficient_observations", () => {
    assertEqual(summarizePersistence([], MIN_GROSS).verdict, "insufficient_observations", "vazio");
    assertEqual(
      summarizePersistence([obs(0, 0.02)], MIN_GROSS).verdict,
      "insufficient_observations",
      "1 obs",
    );
  });

  test("gap positivo em todas as observações → persistent", () => {
    const s = summarizePersistence(
      [obs(0, 0.02), obs(60, 0.018), obs(300, 0.015), obs(900, 0.012)],
      MIN_GROSS,
    );
    assertEqual(s.verdict, "persistent", "persistente");
    assertEqual(s.persistenceScore, 1, "score 1");
    assertTrue(s.lastPositive, "última positiva");
  });

  test("gap positivo no início que evapora → decayed (comido por bot)", () => {
    const s = summarizePersistence(
      [obs(0, 0.02), obs(60, 0.001), obs(300, -0.005), obs(900, -0.01)],
      MIN_GROSS,
    );
    assertEqual(s.verdict, "decayed", "decaiu");
    assertTrue(s.firstPositive && !s.lastPositive, "começou positivo, terminou não");
  });

  test("gap intermitente (flicker) → transient", () => {
    const s = summarizePersistence(
      [obs(0, -0.001), obs(60, 0.02), obs(300, -0.002), obs(900, 0.015)],
      MIN_GROSS,
    );
    assertEqual(s.verdict, "transient", "intermitente");
    assertEqual(s.persistenceScore, 0.5, "metade positiva");
  });

  test("persistent exige última positiva: 3/4 positivas mas termina negativa → decayed", () => {
    const s = summarizePersistence(
      [obs(0, 0.02), obs(60, 0.02), obs(300, 0.02), obs(900, -0.01)],
      MIN_GROSS,
    );
    assertEqual(s.persistenceScore, 0.75, "score 0.75");
    assertEqual(s.verdict, "decayed", "termina negativa → não é persistente");
  });

  test("positivo só acima de minGross: 0.002 < 0.003 não conta", () => {
    const s = summarizePersistence([obs(0, 0.002), obs(60, 0.002)], MIN_GROSS);
    assertEqual(s.positiveAtBestCount, 0, "abaixo do piso não conta");
    assertEqual(s.verdict, "transient", "sem positivos → transient");
  });

  test("maxFillableUsdWhilePositive só considera observações positivas", () => {
    const s = summarizePersistence(
      [obs(0, 0.02, 80), obs(60, -0.01, 500), obs(300, 0.015, 120)],
      MIN_GROSS,
    );
    assertEqual(s.maxFillableUsdWhilePositive, 120, "ignora fillable da obs negativa");
  });

  test("observações fora de ordem são ordenadas por offset", () => {
    const s = summarizePersistence(
      [obs(900, -0.01), obs(0, 0.02), obs(60, 0.02)],
      MIN_GROSS,
    );
    assertTrue(s.firstPositive, "primeira (t=0) positiva");
    assertTrue(!s.lastPositive, "última (t=900) negativa");
    assertEqual(s.verdict, "decayed", "ordem temporal respeitada");
  });
});
