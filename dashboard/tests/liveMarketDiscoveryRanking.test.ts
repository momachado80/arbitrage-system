import {
  DISCOVERY_TOP_CANDIDATES_CAP,
  deriveHealthyMidRange,
  deriveQuotesNearTickFence,
  computeLiveDiscoveryEconomicRank,
  enrichDiscoverySuitableRow,
  enrichRowWithUniverseQuality,
  finalizeDiscoveryRankingSplit,
  finalizeDiscoveryRankingWithUniverseQuality,
} from "../lib/liveMarketDiscoveryRanking";
import { describe, test, assertEqual, assertTrue } from "./_assert";

describe("tests/liveMarketDiscoveryRanking.test.ts", () => {
  test("mid ~0.20 com spread estreito ranqueia melhor que cotas FIFA ~0.002/0.003", () => {
    const informative = computeLiveDiscoveryEconomicRank({
      bestBidUsed: 0.19,
      bestAskUsed: 0.21,
      liquidity: 30_000,
      volume: 12_000,
      clobBookStructure: "two_sided",
    });
    const tail = computeLiveDiscoveryEconomicRank({
      bestBidUsed: 0.002,
      bestAskUsed: 0.003,
      liquidity: 40_000,
      volume: 5_000_000,
      clobBookStructure: "two_sided",
    });
    assertTrue(informative.economicRankScore > tail.economicRankScore, "informative score acima da cauda");
    assertTrue(tail.microProbabilityTail, "cauda micro");
    assertTrue(tail.quotesNearTickFence, "fence na caixa baixa");
  });

  test("volume alto não vence sozinho na cauda near-zero relativamente ao mid saudável", () => {
    const heavyTail = computeLiveDiscoveryEconomicRank({
      bestBidUsed: 0.002,
      bestAskUsed: 0.003,
      liquidity: 2_000_000,
      volume: 9_999_999_999,
      clobBookStructure: "two_sided",
    });
    const lightInformative = computeLiveDiscoveryEconomicRank({
      bestBidUsed: 0.19,
      bestAskUsed: 0.205,
      liquidity: 5_000,
      volume: 800,
      clobBookStructure: "two_sided",
    });
    assertTrue(lightInformative.economicRankScore > heavyTail.economicRankScore, "mid informativo vence mesmo com volume menor");
  });

  test("quotesNearTickFence verdadeiro para 0.002 / 0.003", () => {
    const mid = (0.002 + 0.003) / 2;
    assertTrue(deriveQuotesNearTickFence(0.002, 0.003, mid), "pair FIFA-like");
    assertTrue(deriveQuotesNearTickFence(0.003, 0.004, (0.003 + 0.004) / 2), "ainda próximo do fence baixo");
  });

  test("healthyMidRange verdadeiro na faixa ~0.05–0.70", () => {
    assertTrue(deriveHealthyMidRange(0.12), "0.12 informativo");
    assertTrue(deriveHealthyMidRange(0.2), "0.20 informativo");
    assertEqual(deriveHealthyMidRange(0.02), false, "abaixo do mínimo");
    assertEqual(deriveHealthyMidRange(0.75), false, "acima do teto económico definido");
  });

  test("canUseForMicrocapitalCandidate permanece false no augment", () => {
    const r = computeLiveDiscoveryEconomicRank({
      bestBidUsed: 0.5,
      bestAskUsed: 0.52,
      liquidity: 1e6,
      volume: 1e6,
      clobBookStructure: "two_sided",
    });
    assertEqual(r.canUseForMicrocapitalCandidate, false, "compute micro flag");
    const row = enrichDiscoverySuitableRow({
      bestBidUsed: 0.5,
      bestAskUsed: 0.51,
      liquidity: 999,
      volume: 888,
      clobBookStructure: "two_sided",
    }) as Record<string, unknown>;
    assertEqual(row.canUseForMicrocapitalCandidate, false, "enrich micro flag");
  });

  test("split export: todos os suitable em candidates; topCandidates limitado ao cap mesmo com >cap linhas", () => {
    const enriched = [...Array.from({ length: 25 }).keys()].map(i =>
      enrichDiscoverySuitableRow({
        id: `id-${i}`,
        bestBidUsed: 0.1,
        bestAskUsed: 0.11 + i * 1e-9,
        liquidity: i + 50,
        volume: i + 10,
        clobBookStructure: "two_sided",
        suitabilityVerdict: "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION",
        reasons: [`ok-${i}`],
      }),
    );
    const { candidatesSorted, topCandidates } = finalizeDiscoveryRankingSplit(enriched);
    assertEqual(candidatesSorted.length, 25, "lista completa de suitable enrich");
    assertEqual(topCandidates.length, DISCOVERY_TOP_CANDIDATES_CAP, `topCandidates = ${DISCOVERY_TOP_CANDIDATES_CAP}`);
  });

  test("enrichRowWithUniverseQuality marca clean para esporte saudavel e propaga universeQuality*", () => {
    const row = enrichDiscoverySuitableRow({
      id: "553856",
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
      slug: "will-oklahoma-city-thunder-win-2026-nba-finals",
      endDate: "2026-07-01T00:00:00.000Z",
      bestBidUsed: 0.59,
      bestAskUsed: 0.6,
      liquidity: 25_000,
      volume: 18_000,
      clobBookStructure: "two_sided",
      active: true,
      closed: false,
      resolved: false,
      suitabilityVerdict: "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION",
    });
    const enrichedQ = enrichRowWithUniverseQuality(row, "2026-05-05T17:00:00.000Z");
    assertEqual(enrichedQ.universeQualityVerdict, "CLEAN_OBSERVATION_UNIVERSE", "verdict clean");
    assertEqual(enrichedQ.isCleanForObservationScout, true, "isClean true");
    assertTrue(typeof enrichedQ.universeQualityScore === "number", "score numero");
    assertTrue(Array.isArray(enrichedQ.disqualifiers), "disqualifiers array");
  });

  test("finalizeDiscoveryRankingWithUniverseQuality separa CLEAN, REJECT e mantem topCandidates", () => {
    const nowIso = "2026-05-05T17:00:00.000Z";
    const cleanRow = enrichDiscoverySuitableRow({
      id: "553856",
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
      slug: "will-oklahoma-city-thunder-win-2026-nba-finals",
      endDate: "2026-07-01T00:00:00.000Z",
      bestBidUsed: 0.59,
      bestAskUsed: 0.6,
      liquidity: 25_000,
      volume: 18_000,
      clobBookStructure: "two_sided",
      active: true,
      closed: false,
      resolved: false,
      suitabilityVerdict: "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION",
    });
    const memeRow = enrichDiscoverySuitableRow({
      id: "540819",
      question: "Will Jesus Christ return before GTA VI?",
      slug: "will-jesus-christ-return-before-gta-vi",
      endDate: "2026-07-31T12:00:00.000Z",
      bestBidUsed: 0.48,
      bestAskUsed: 0.49,
      liquidity: 5_000,
      volume: 5_000,
      clobBookStructure: "two_sided",
      active: true,
      closed: false,
      resolved: false,
      suitabilityVerdict: "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION",
    });
    const politicalRow = enrichDiscoverySuitableRow({
      id: "540820",
      question: "Trump out as President before GTA VI?",
      slug: "trump-out-as-president-before-gta-vi",
      endDate: "2026-07-31T12:00:00.000Z",
      bestBidUsed: 0.2,
      bestAskUsed: 0.21,
      liquidity: 8_000,
      volume: 4_000,
      clobBookStructure: "two_sided",
      active: true,
      closed: false,
      resolved: false,
      suitabilityVerdict: "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION",
    });
    const tailRow = enrichDiscoverySuitableRow({
      id: "558934",
      question: "Will Spain win the 2026 FIFA World Cup?",
      slug: "will-spain-win-2026-fifa-world-cup",
      endDate: "2026-07-20T00:00:00.000Z",
      bestBidUsed: 0.04,
      bestAskUsed: 0.045,
      liquidity: 30_000,
      volume: 12_000,
      clobBookStructure: "two_sided",
      active: true,
      closed: false,
      resolved: false,
      suitabilityVerdict: "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION",
    });

    const out = finalizeDiscoveryRankingWithUniverseQuality(
      [cleanRow, memeRow, politicalRow, tailRow],
      nowIso,
    );
    assertEqual(out.candidatesSorted.length, 4, "todos preservados em candidatesSorted");
    assertEqual(out.topCleanCandidates.length, 1, "apenas o esporte saudavel é clean");
    assertEqual(
      out.topCleanCandidates[0]!.id,
      "553856",
      "primeiro clean é OKC NBA Finals",
    );
    assertTrue(out.rejectedByUniverseQuality.length === 3, "tres rejeições por universe quality");
    const verdicts = new Set(
      out.rejectedByUniverseQuality.map(r => String(r.universeQualityVerdict)),
    );
    assertTrue(verdicts.has("REJECT_MEME_OR_ABSURD"), "tem meme");
    assertTrue(verdicts.has("REJECT_POLITICAL_LEGAL"), "tem politico/legal");
    assertTrue(verdicts.has("REJECT_TAIL_OR_TICK_FENCE"), "tem tail");
    const reasonsKeys = out.universeQualityRejectionReasons.map(x => x.verdict);
    assertTrue(reasonsKeys.includes("REJECT_MEME_OR_ABSURD"), "razão meme contagem");
  });

  test("topCleanCandidates tem cap igual a topCandidates", () => {
    const nowIso = "2026-05-05T17:00:00.000Z";
    const rows = [...Array.from({ length: DISCOVERY_TOP_CANDIDATES_CAP + 5 }).keys()].map(i =>
      enrichDiscoverySuitableRow({
        id: `id-${i}`,
        question: `Will team ${i} win the 2026 NBA Finals?`,
        slug: `team-${i}-2026-nba-finals`,
        endDate: "2026-07-01T00:00:00.000Z",
        bestBidUsed: 0.55,
        bestAskUsed: 0.56 + i * 1e-9,
        liquidity: 25_000 + i,
        volume: 18_000 + i,
        clobBookStructure: "two_sided",
        active: true,
        closed: false,
        resolved: false,
        suitabilityVerdict: "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION",
      }),
    );
    const out = finalizeDiscoveryRankingWithUniverseQuality(rows, nowIso);
    assertTrue(out.topCleanCandidates.length === DISCOVERY_TOP_CANDIDATES_CAP, "cap em topCleanCandidates");
    assertTrue(out.topCandidates.length === DISCOVERY_TOP_CANDIDATES_CAP, "cap em topCandidates");
  });
});
