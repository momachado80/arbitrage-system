import {
  OBS_MOVE_EPS,
  evaluateObservationWindow,
  type ObservationWindowSnapshotInput,
} from "../lib/observationWindowSuitability";
import fs from "fs";
import path from "path";
import { describe, test, assertEqual, assertTrue } from "./_assert";

function snap(
  overrides: Partial<ObservationWindowSnapshotInput> & Pick<ObservationWindowSnapshotInput, "bestBid" | "bestAsk" | "mid" | "spread">,
): ObservationWindowSnapshotInput {
  return {
    collectedAtIso: "2026-05-06T12:00:00.000Z",
    priceSource: "bid_ask_mid",
    bookType: "two_sided",
    ...overrides,
  };
}

describe("tests/observationWindowSuitability.test.ts", () => {
  const baseMarket = {
    nowIso: "2026-05-06T14:00:00.000Z",
    suitabilityVerdict: "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION" as const,
    active: true,
    closed: false,
    resolved: false,
    endDate: "2035-01-01T00:00:00.000Z",
  };

  test("mercado vencido gera WINDOW_EXPIRED_OR_CLOSED", () => {
    const r = evaluateObservationWindow({
      ...baseMarket,
      snapshots: [],
      endDate: "2020-01-01T00:00:00.000Z",
    });
    assertEqual(r.observationWindowVerdict, "WINDOW_EXPIRED_OR_CLOSED", "expirado");
    assertEqual(r.canUseForMicrocapitalCandidate, false, "micro false");
    assertEqual(r.canUseForPaperShadowObservation, false, "sem papel");
  });

  test("mercado closed/resolved gera WINDOW_EXPIRED_OR_CLOSED", () => {
    assertEqual(
      evaluateObservationWindow({ ...baseMarket, closed: true, snapshots: [] }).observationWindowVerdict,
      "WINDOW_EXPIRED_OR_CLOSED",
      "closed",
    );
    assertEqual(
      evaluateObservationWindow({ ...baseMarket, resolved: true, snapshots: [] }).observationWindowVerdict,
      "WINDOW_EXPIRED_OR_CLOSED",
      "resolved",
    );
  });

  test("suitability ruim gera WINDOW_MARKET_NOT_SUITABLE", () => {
    const r = evaluateObservationWindow({
      ...baseMarket,
      suitabilityVerdict: "UNSUITABLE_PRICE_PINNED",
      snapshots: [snap({ bestBid: 0.4, bestAsk: 0.41, mid: 0.405, spread: 0.01 })],
    });
    assertEqual(r.observationWindowVerdict, "WINDOW_MARKET_NOT_SUITABLE", "gate");
    assertEqual(r.canUseForMicrocapitalCandidate, false, "micro false");
  });

  test("snapshots sem livro bilateral geram WINDOW_INSUFFICIENT_BOOK", () => {
    const snaps: ObservationWindowSnapshotInput[] = [
      { collectedAtIso: "t1", bookType: "not_two_sided", bestBid: 0.4, bestAsk: null, mid: null, spread: null },
      { collectedAtIso: "t2", bookType: "not_two_sided", bestBid: null, bestAsk: 0.41, mid: null, spread: null },
    ];
    const r = evaluateObservationWindow({ ...baseMarket, snapshots: snaps });
    assertEqual(r.observationWindowVerdict, "WINDOW_INSUFFICIENT_BOOK", "topologia");
  });

  test("preço perto de 0.001 ou 0.999 gera WINDOW_PRICE_PINNED", () => {
    const pinnedLow = evaluateObservationWindow({
      ...baseMarket,
      snapshots: [
        snap({ bestBid: 0.001, bestAsk: 0.003, mid: (0.001 + 0.003) / 2, spread: 0.002 }),
      ],
    });
    assertEqual(pinnedLow.observationWindowVerdict, "WINDOW_PRICE_PINNED", "andar baixo");

    const pinnedHigh = evaluateObservationWindow({
      ...baseMarket,
      snapshots: [
        snap({ bestBid: 0.996, bestAsk: 0.999, mid: (0.996 + 0.999) / 2, spread: 0.003 }),
      ],
    });
    assertEqual(pinnedHigh.observationWindowVerdict, "WINDOW_PRICE_PINNED", "andar alto");
  });

  test("livro bilateral mas mid/spread planos ⇒ WINDOW_QUIET_BUT_VALID", () => {
    const r = evaluateObservationWindow({
      ...baseMarket,
      snapshots: [
        snap({ bestBid: 0.52, bestAsk: 0.53, mid: 0.525, spread: 0.01 }),
        snap({ collectedAtIso: "t2", bookType: "two_sided", bestBid: 0.52, bestAsk: 0.53, mid: 0.525, spread: 0.01 }),
      ],
    });
    assertEqual(r.observationWindowVerdict, "WINDOW_QUIET_BUT_VALID", "quieto válido");
    assertEqual(r.isWindowInformative, false, "não é informativo");
    assertTrue(r.observationWindowScore < 9999, "score moderado");
  });

  test("mid variando ao longo de ≥2 snapshots gera WINDOW_INFORMATIVE", () => {
    const r = evaluateObservationWindow({
      ...baseMarket,
      snapshots: [
        snap({ bestBid: 0.5, bestAsk: 0.51, mid: 0.505, spread: 0.01 }),
        snap({ collectedAtIso: "t2", bookType: "two_sided", bestBid: 0.53, bestAsk: 0.54, mid: 0.535, spread: 0.01 }),
        snap({ collectedAtIso: "t3", bookType: "two_sided", bestBid: 0.55, bestAsk: 0.56, mid: 0.555, spread: 0.01 }),
      ],
    });
    assertEqual(r.observationWindowVerdict, "WINDOW_INFORMATIVE", "movimento mid");
    assertEqual(r.canUseForMicrocapitalCandidate, false, "micro false");
    assertEqual(r.canUseForPaperShadowObservation, true, "papel ok");
    assertEqual(r.isWindowInformative, true, "informative flag");
    assertTrue((r.midRange ?? 0) > OBS_MOVE_EPS, "tem faixa mid");
  });

  test("spread mudando com mid estável ⇒ WINDOW_INFORMATIVE", () => {
    const r = evaluateObservationWindow({
      ...baseMarket,
      snapshots: [
        snap({ bestBid: 0.5, bestAsk: 0.51, mid: 0.505, spread: 0.01 }),
        snap({ collectedAtIso: "t2", bookType: "two_sided", bestBid: 0.49, bestAsk: 0.52, mid: 0.505, spread: 0.03 }),
        snap({
          collectedAtIso: "t3",
          bookType: "two_sided",
          bestBid: 0.475,
          bestAsk: 0.535,
          mid: 0.505,
          spread: 0.06,
        }),
      ],
    });
    assertEqual(r.observationWindowVerdict, "WINDOW_INFORMATIVE", "spread varia mid fixo");
  });

  test("evaluateObservationWindow força microcapital false", () => {
    const r = evaluateObservationWindow({
      ...baseMarket,
      canUseForMicrocapitalCandidate: true,
      snapshots: [snap({ bestBid: 0.4, bestAsk: 0.41, mid: 0.405, spread: 0.01 })],
    });
    assertEqual(r.canUseForMicrocapitalCandidate, false, "sempre false");
  });

  test("lib sem termos de execução reservados no ficheiro (regex palavra)", () => {
    const libPath = path.join(__dirname, "../lib/observationWindowSuitability.ts");
    const testPath = path.join(__dirname, "observationWindowSuitability.test.ts");
    const patterns: RegExp[] = [
      /\border\b/i,
      /\bsubmit\b/i,
      /\bwallet\b/i,
      /\bsigner\b/i,
      /\bprivatekey\b/i,
      /\bmnemonic\b/i,
      /\bseed\s+phrase\b/i,
    ];
    const hayLib = fs.readFileSync(libPath, "utf8");
    const hayTest = fs.readFileSync(testPath, "utf8");
    for (const re of patterns) {
      assertTrue(!re.test(hayLib), `lib observação evita ${re}`);
      assertTrue(!re.test(hayTest), `teste observação evita ${re}`);
    }
  });
});
