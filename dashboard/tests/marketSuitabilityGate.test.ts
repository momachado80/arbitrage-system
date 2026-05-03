import { computeMarketSuitability } from "../lib/marketSuitabilityGate";
import { describe, test, assertEqual, assertTrue } from "./_assert";

describe("tests/marketSuitabilityGate.test.ts", () => {
  const nowPastApril = "2026-05-03T15:00:00.000Z";

  test('"Will Ethereum reach $4,000 in April?" bloqueado em 2026-05-03 por expiração temporal', () => {
    const r = computeMarketSuitability({
      question: "Will Ethereum reach $4,000 in April?",
      marketId: "1823789",
      nowIso: nowPastApril,
      bestBid: 0.004,
      bestAsk: 0.006,
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_EXPIRED", "verdict expired");
    assertTrue(!r.canUseForMicrocapitalCandidate, "microcapital veto");
    assertTrue(r.reasons.some(x => x.includes("april") || x.includes("temporal")), "reason hint");
  });

  test("closed=true bloqueia", () => {
    const r = computeMarketSuitability({
      marketId: "x",
      closed: true,
      bestBid: 0.51,
      bestAsk: 0.53,
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_RESOLVED_OR_CLOSED", "closed verdict");
    assertTrue(!r.canUseForMicrocapitalCandidate, "no micro");
  });

  test("resolved=true bloqueia", () => {
    const r = computeMarketSuitability({
      marketId: "x",
      resolved: true,
      bestBid: 0.41,
      bestAsk: 0.43,
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_RESOLVED_OR_CLOSED", "resolved verdict");
    assertTrue(!r.canUseForMicrocapitalCandidate, "no micro");
  });

  test("quotes presos perto de 0.001 bloqueiam", () => {
    const r = computeMarketSuitability({
      marketId: "x",
      bestBid: null,
      bestAsk: 0.001,
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_PRICE_PINNED", "pinned");
    assertTrue(!r.canUseForMicrocapitalCandidate, "no micro");
  });

  test("WEAK_FLAT_MARKOUTS + informativeMarkoutCycleCount=0 bloqueia microcapital", () => {
    const r = computeMarketSuitability({
      question: "Active illiquid?",
      marketId: "999",
      nowIso: "2026-02-01T12:00:00.000Z",
      bestBid: 0.2,
      bestAsk: 0.22,
      markoutInformativenessVerdict: "WEAK_FLAT_MARKOUTS",
      informativeMarkoutCycleCount: 0,
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_NO_INFORMATIVE_MARKOUTS", "flat markouts");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro veto");
  });

  test("sem bid/ask confiável bloqueia", () => {
    const r = computeMarketSuitability({
      marketId: "x",
      priceSource: "gamma_hint",
      lastPrice: 0.31,
      nowIso: "2026-02-01T12:00:00.000Z",
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_LIQUIDITY", "liquidity");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro veto");
  });

  test("mercado bem formado pode ser SUITABLE_FOR_PAPER_OBSERVATION", () => {
    const r = computeMarketSuitability({
      question: "Will XYZ reach target in December 2027?",
      marketId: "new-candidate",
      active: true,
      closed: false,
      resolved: false,
      endDate: "2027-12-31T04:00:00.000Z",
      bestBid: 0.52,
      bestAsk: 0.54,
      volume: 25_000,
      liquidity: 8_000,
      markoutInformativenessVerdict: "INFORMATIVE_CYCLES_PRESENT",
      informativeMarkoutCycleCount: 4,
      priceSource: "bid_ask_mid",
      nowIso: "2026-05-03T15:00:00.000Z",
    });
    assertEqual(r.suitabilityVerdict, "SUITABLE_FOR_PAPER_OBSERVATION", "suitable");
    assertTrue(r.canUseForMicrocapitalCandidate, "micro ok");
    assertEqual(r.reasons.includes("minimal_paper_candidate_checks_pass"), true, "explicit pass stamp");
  });

  test("dados ausentes — bloqueiam microcapital por defeito", () => {
    const r = computeMarketSuitability({});
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_MISSING_DATA", "missing");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro veto");
    assertEqual(r.reasons.includes("no_assessable_input"), true, "explain");
  });
});
