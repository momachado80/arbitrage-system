import fs from "fs";
import path from "path";

import {
  evaluateMarketUniverseQuality,
  isCleanRow,
} from "../lib/marketUniverseQuality";
import { describe, test, assertEqual, assertTrue } from "./_assert";

describe("tests/marketUniverseQuality.test.ts", () => {
  const baseSport = {
    suitabilityVerdict: "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION",
    active: true,
    closed: false,
    resolved: false,
    nowIso: "2026-05-05T17:00:00.000Z",
    endDate: "2026-07-01T00:00:00.000Z",
    bestBid: 0.59,
    bestAsk: 0.6,
    liquidity: 25_000,
    volume: 18_000,
    clobBookStructure: "two_sided",
    strongTwoSidedBook: true,
  };

  test("Jesus Christ before GTA VI rejeita por meme/absurdidade", () => {
    const r = evaluateMarketUniverseQuality({
      marketId: "540819",
      question: "Will Jesus Christ return before GTA VI?",
      slug: "will-jesus-christ-return-before-gta-vi",
      ...baseSport,
      bestBid: 0.48,
      bestAsk: 0.49,
    });
    assertEqual(r.universeQualityVerdict, "REJECT_MEME_OR_ABSURD", "verdict meme");
    assertTrue(r.disqualifiers.some(d => d.startsWith("meme_or_absurd:")), "tem disqualifier meme");
    assertEqual(r.isCleanForObservationScout, false, "não é clean");
    assertEqual(r.canUseForMicrocapitalCandidate, false, "micro sempre false");
  });

  test("Bitcoin to $1m before GTA VI rejeita por meme/absurdidade", () => {
    const r = evaluateMarketUniverseQuality({
      marketId: "540844",
      question: "Will bitcoin hit $1m before GTA VI?",
      slug: "will-bitcoin-hit-1m-before-gta-vi",
      ...baseSport,
      bestBid: 0.49,
      bestAsk: 0.493,
    });
    assertEqual(r.universeQualityVerdict, "REJECT_MEME_OR_ABSURD", "meme/absurd");
    assertTrue(
      r.disqualifiers.some(d => d.startsWith("meme_or_absurd:")),
      "GTA VI ou bitcoin $1m disqualifier",
    );
  });

  test("Trump out as President rejeita por politicalLegalRisk", () => {
    const r = evaluateMarketUniverseQuality({
      marketId: "540820",
      question: "Trump out as President before GTA VI?",
      slug: "trump-out-as-president-before-gta-vi",
      ...baseSport,
      bestBid: 0.2,
      bestAsk: 0.21,
    });
    assertEqual(r.universeQualityVerdict, "REJECT_POLITICAL_LEGAL", "politico/legal antes de meme");
    assertTrue(r.disqualifiers.some(d => d.startsWith("political_legal:")), "tem disqualifier politico");
    assertTrue(r.politicalLegalRisk >= 0.6, "risco politico alto");
  });

  test("mid abaixo de 0.06 rejeita como tail/tick fence", () => {
    const r = evaluateMarketUniverseQuality({
      marketId: "558934",
      question: "Will Spain win the 2026 FIFA World Cup?",
      slug: "will-spain-win-2026-fifa-world-cup",
      ...baseSport,
      endDate: "2026-07-20T00:00:00.000Z",
      bestBid: 0.04,
      bestAsk: 0.045,
    });
    assertEqual(r.universeQualityVerdict, "REJECT_TAIL_OR_TICK_FENCE", "cauda extrema");
    assertTrue(r.tailRisk >= 1, "tailRisk saturado em 1");
    assertEqual(r.isCleanForObservationScout, false, "não clean");
  });

  test("mercado closed rejeita como NOT_SUITABLE", () => {
    const r = evaluateMarketUniverseQuality({
      ...baseSport,
      question: "Some sports market",
      closed: true,
    });
    assertEqual(r.universeQualityVerdict, "REJECT_NOT_SUITABLE", "closed bloqueia");
    assertTrue(
      r.disqualifiers.includes("market_closed_resolved_expired_or_inactive"),
      "disqualifier closed",
    );
  });

  test("mercado resolved rejeita como NOT_SUITABLE", () => {
    const r = evaluateMarketUniverseQuality({
      ...baseSport,
      question: "NBA Finals 2025",
      resolved: true,
    });
    assertEqual(r.universeQualityVerdict, "REJECT_NOT_SUITABLE", "resolved bloqueia");
  });

  test("endDate passado rejeita como NOT_SUITABLE", () => {
    const r = evaluateMarketUniverseQuality({
      ...baseSport,
      question: "NBA Finals 2024",
      endDate: "2024-07-01T00:00:00.000Z",
    });
    assertEqual(r.universeQualityVerdict, "REJECT_NOT_SUITABLE", "expired");
  });

  test("suitabilityVerdict diferente de SUITABLE rejeita como NOT_SUITABLE", () => {
    const r = evaluateMarketUniverseQuality({
      ...baseSport,
      suitabilityVerdict: "UNSUITABLE_NO_BOOK",
      question: "Will the OKC Thunder win the 2026 NBA Finals?",
    });
    assertEqual(r.universeQualityVerdict, "REJECT_NOT_SUITABLE", "gate prévio bloqueia");
  });

  test("esporte objetivo, ativo, mid saudavel, spread baixo, two-sided ⇒ CLEAN", () => {
    const r = evaluateMarketUniverseQuality({
      ...baseSport,
      marketId: "553856",
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
      slug: "will-oklahoma-city-thunder-win-2026-nba-finals",
      bestBid: 0.59,
      bestAsk: 0.6,
    });
    assertEqual(r.universeQualityVerdict, "CLEAN_OBSERVATION_UNIVERSE", "clean");
    assertEqual(r.isCleanForObservationScout, true, "flag clean true");
    assertEqual(r.canUseForPaperShadowCandidate, true, "pode paper shadow");
    assertEqual(r.canUseForMicrocapitalCandidate, false, "micro sempre false");
    assertTrue(r.universeQualityScore >= 320, "score acima do limiar clean");
    assertTrue(isCleanRow({ universeQualityVerdict: r.universeQualityVerdict }), "predicate isCleanRow");
  });

  test("canUseForMicrocapitalCandidate sempre false mesmo em CLEAN", () => {
    const r = evaluateMarketUniverseQuality({
      ...baseSport,
      question: "Will the OKC Thunder win the 2026 NBA Finals?",
    });
    assertEqual(r.canUseForMicrocapitalCandidate, false, "micro false em qualquer verdict");
  });

  test("long horizon sem catalisador esportivo/financeiro vira REJECT_LONG_HORIZON_NO_CATALYST", () => {
    const r = evaluateMarketUniverseQuality({
      ...baseSport,
      question: "Will artist X release a third studio album in the next year?",
      slug: "will-artist-x-release-third-studio-album",
      endDate: "2027-04-30T00:00:00.000Z",
    });
    assertEqual(
      r.universeQualityVerdict,
      "REJECT_LONG_HORIZON_NO_CATALYST",
      "long sem sport/financial",
    );
    assertTrue(r.longHorizonNoCatalystRisk >= 0.5, "risco horizonte alto");
  });

  test("very_long sempre vira REJECT_LONG_HORIZON_NO_CATALYST mesmo com sport", () => {
    const r = evaluateMarketUniverseQuality({
      ...baseSport,
      question: "Will FC Some Team win the 2030 Champions League?",
      endDate: "2030-06-01T00:00:00.000Z",
    });
    assertEqual(r.universeQualityVerdict, "REJECT_LONG_HORIZON_NO_CATALYST", "very_long bloqueia");
  });

  test("output inclui reasons e disqualifiers como arrays", () => {
    const r = evaluateMarketUniverseQuality({
      ...baseSport,
      question: "Will the OKC Thunder win the 2026 NBA Finals?",
    });
    assertTrue(Array.isArray(r.reasons), "reasons array");
    assertTrue(Array.isArray(r.disqualifiers), "disqualifiers array");
    assertTrue(r.reasons.length > 0, "tem ao menos 1 reason");
  });

  test("eventProximityClass coerente com endDate", () => {
    const near = evaluateMarketUniverseQuality({
      ...baseSport,
      endDate: "2026-06-01T00:00:00.000Z",
    });
    assertEqual(near.eventProximityClass, "near", "near em ~25 dias");
    const veryNear = evaluateMarketUniverseQuality({
      ...baseSport,
      endDate: "2026-05-10T00:00:00.000Z",
    });
    assertEqual(veryNear.eventProximityClass, "very_near", "very_near em ~5 dias");
  });

  test("lib sem termos de execução reservados (regex palavra)", () => {
    const libPath = path.join(__dirname, "../lib/marketUniverseQuality.ts");
    const testPath = path.join(__dirname, "marketUniverseQuality.test.ts");
    const patterns: RegExp[] = [
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
      assertTrue(!re.test(hayLib), `lib evita ${re}`);
      assertTrue(!re.test(hayTest), `teste evita ${re}`);
    }
  });
});
