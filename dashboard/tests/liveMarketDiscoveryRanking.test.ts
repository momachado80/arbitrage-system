import {
  DISCOVERY_TOP_CANDIDATES_CAP,
  deriveHealthyMidRange,
  deriveQuotesNearTickFence,
  computeLiveDiscoveryEconomicRank,
  enrichDiscoverySuitableRow,
  finalizeDiscoveryRankingSplit,
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
});
