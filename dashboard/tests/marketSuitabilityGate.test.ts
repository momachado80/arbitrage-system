import { evaluateMarketSuitability } from "../lib/marketSuitabilityGate";
import { describe, test, assertEqual, assertTrue } from "./_assert";

describe("tests/marketSuitabilityGate.test.ts", () => {
  const nowPastApril = "2026-05-03T15:00:00.000Z";

  test('"Will Ethereum reach $4,000 in April?" com nowIso 2026-05-03 bloqueia por expirado', () => {
    const r = evaluateMarketSuitability({
      question: "Will Ethereum reach $4,000 in April?",
      marketId: "1823789",
      nowIso: nowPastApril,
      active: true,
      bestBid: 0.04,
      bestAsk: 0.06,
      endDate: "2027-01-01T00:00:00Z",
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_EXPIRED", "verdict expired april window");
    assertTrue(!r.canUseForPaperShadowCandidate, "paper shadow veto");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro always false");
    assertTrue(r.reasons.some(x => /april|temporal/i.test(x)), "reason temporal");
  });

  test("closed=true bloqueia", () => {
    const r = evaluateMarketSuitability({
      marketId: "x",
      closed: true,
      active: true,
      bestBid: 0.51,
      bestAsk: 0.53,
      volume: 9_999,
      liquidity: 9_999,
      endDate: "2027-06-01T00:00:00Z",
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_CLOSED_OR_RESOLVED", "closed verdict");
    assertTrue(!r.canUseForPaperShadowCandidate, "paper shadow veto");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro false");
  });

  test("resolved=true bloqueia", () => {
    const r = evaluateMarketSuitability({
      marketId: "x",
      resolved: true,
      active: true,
      bestBid: 0.41,
      bestAsk: 0.43,
      volume: 9_999,
      liquidity: 9_999,
      endDate: "2027-06-01T00:00:00Z",
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_CLOSED_OR_RESOLVED", "resolved verdict");
    assertTrue(!r.canUseForPaperShadowCandidate, "paper shadow veto");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro false");
  });

  test("preço preso perto de 0.001 bloqueia", () => {
    const r = evaluateMarketSuitability({
      marketId: "x",
      active: true,
      bestBid: null,
      bestAsk: 0.001,
      volume: 9_999,
      liquidity: 9_999,
      endDate: "2027-06-01T00:00:00Z",
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_PRICE_PINNED", "pinned low");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro false");
  });

  test("preço preso perto de 0.999 bloqueia", () => {
    const r = evaluateMarketSuitability({
      marketId: "x",
      active: true,
      bestBid: 0.01,
      bestAsk: 0.999,
      volume: 9_999,
      liquidity: 9_999,
      endDate: "2027-06-01T00:00:00Z",
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_PRICE_PINNED", "pinned high near 1");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro false");
  });

  test("falta bid/ask confiável bloqueia", () => {
    const r = evaluateMarketSuitability({
      marketId: "x",
      active: true,
      lastPrice: 0.41,
      volume: 9_999,
      liquidity: 9_999,
      endDate: "2027-06-01T00:00:00Z",
      nowIso: "2026-02-01T12:00:00.000Z",
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_NO_BOOK", "no book quotes");
    assertTrue(!r.canUseForPaperShadowCandidate, "paper shadow veto");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro false");
  });

  test("endDate ausente bloqueia paper shadow mesmo com dados de preço ", () => {
    const r = evaluateMarketSuitability({
      marketId: "n",
      question: "Neutral undated market?",
      active: true,
      bestBid: 0.33,
      bestAsk: 0.35,
      volume: 8_888,
      liquidity: 9_888,
      nowIso: "2026-06-01T12:00:00.000Z",
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_MISSING_DATA", "missing horizon");
    assertTrue(!r.canUseForPaperShadowCandidate, "cannot promote without endDate");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro false");
    assertTrue(r.reasons.some(x => x.includes("missing_endDate")), "reason");
  });

  test("mercado bem formado passa só para observação paper/shadow (sem microcapital)", () => {
    const r = evaluateMarketSuitability({
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
      nowIso: "2026-05-03T15:00:00.000Z",
      clobBookStructure: "two_sided",
    });
    assertEqual(r.suitabilityVerdict, "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION", "shadow suitable");
    assertTrue(r.canUseForPaperShadowCandidate, "paper shadow ok");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro remains false phase");
    assertEqual(
      r.reasons.some(x => x.includes("paper_shadow_readonly_checks_pass")),
      true,
      "pass stamp present",
    );
  });

  test("microcapital não é activado mesmo em mercados limpos", () => {
    const r = evaluateMarketSuitability({
      marketId: "z",
      question: "Ultra clean?",
      active: true,
      bestBid: 0.61,
      bestAsk: 0.62,
      volume: 77_777,
      liquidity: 77_777,
      endDate: "2028-01-01T00:00:00.000Z",
      nowIso: "2026-05-03T15:00:00.000Z",
      markoutInformativenessVerdict: "HEALTHY",
      informativeMarkoutCycleCount: 99,
      clobBookStructure: "two_sided",
    });
    assertTrue(!r.canUseForMicrocapitalCandidate, "explicit micro guard");
  });

  test("WEAK_FLAT_MARKOUTS bloqueia suitability económica/paper shadow quando conhecido", () => {
    const r = evaluateMarketSuitability({
      question: "Active but flat markouts?",
      marketId: "flat1",
      nowIso: "2026-02-01T12:00:00.000Z",
      active: true,
      bestBid: 0.2,
      bestAsk: 0.22,
      volume: 10_111,
      liquidity: 3_211,
      endDate: "2027-06-01T00:00:00.000Z",
      markoutInformativenessVerdict: "WEAK_FLAT_MARKOUTS",
      informativeMarkoutCycleCount: 0,
      clobBookStructure: "two_sided",
    });
    assertEqual(r.suitabilityVerdict, "UNSUITABLE_WEAK_FLAT_MARKOUTS", "economic ill");
    assertTrue(!r.canUseForPaperShadowCandidate, "blocked from promotion");
    assertTrue(!r.canUseForMicrocapitalCandidate, "micro false");
  });
});
