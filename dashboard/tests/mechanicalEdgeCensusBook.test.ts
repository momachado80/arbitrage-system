import {
  computeVwap,
  bestPrice,
  depthTopN,
  binaryUnderroundFlag,
  binaryOverroundFlag,
  partitionUnderroundFlag,
  targetSharesPerLeg,
  classifyResolutionCategory,
  type BookLevel,
} from "../lib/mechanicalEdgeCensusBook";
import { describe, test, assertEqual, assertTrue } from "./_assert";

function near(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) < tol;
}

const asks: BookLevel[] = [
  { price: 0.40, size: 100 },
  { price: 0.42, size: 200 },
  { price: 0.45, size: 300 },
];

describe("tests/mechanicalEdgeCensusBook.test.ts", () => {
  test("computeVwap (buy): preenche dentro do primeiro nível → vwap = melhor preço", () => {
    const r = computeVwap(asks, 50, "buy");
    assertTrue(near(r.vwap ?? -1, 0.40), "vwap 0.40");
    assertEqual(r.filledShares, 50, "50 shares");
    assertTrue(r.fullyFilled, "preenchido");
  });

  test("computeVwap (buy): atravessa níveis → vwap ponderado", () => {
    /** 100@0.40 + 100@0.42 = 40 + 42 = 82 por 200 shares → 0.41 */
    const r = computeVwap(asks, 200, "buy");
    assertTrue(near(r.vwap ?? -1, 0.41), "vwap 0.41 ponderado");
    assertEqual(r.filledShares, 200, "200 shares");
    assertTrue(r.fullyFilled, "preenchido");
  });

  test("computeVwap (buy): profundidade insuficiente → fullyFilled false", () => {
    const r = computeVwap(asks, 1000, "buy");
    assertEqual(r.filledShares, 600, "só 600 disponíveis");
    assertTrue(!r.fullyFilled, "não totalmente preenchido");
    /** vwap dos 600: (40+84+135)/600 = 259/600 */
    assertTrue(near(r.vwap ?? -1, 259 / 600), "vwap parcial");
  });

  test("computeVwap (sell): varre bids do maior para o menor", () => {
    const bids: BookLevel[] = [
      { price: 0.55, size: 100 },
      { price: 0.50, size: 100 },
      { price: 0.45, size: 100 },
    ];
    /** 100@0.55 + 50@0.50 = 55 + 25 = 80 por 150 → 0.5333 */
    const r = computeVwap(bids, 150, "sell");
    assertTrue(near(r.vwap ?? -1, 80 / 150), "vwap sell ponderado");
    assertTrue(r.fullyFilled, "preenchido");
  });

  test("computeVwap: targetShares <= 0 → vazio", () => {
    const r = computeVwap(asks, 0, "buy");
    assertEqual(r.vwap, null, "vwap null");
    assertEqual(r.filledShares, 0, "0 shares");
  });

  test("computeVwap: livro vazio → vazio", () => {
    const r = computeVwap([], 100, "buy");
    assertEqual(r.vwap, null, "vwap null");
  });

  test("computeVwap: filtra níveis inválidos (preço/size fora de faixa)", () => {
    const dirty: BookLevel[] = [
      { price: 0, size: 100 },
      { price: 1.5, size: 100 },
      { price: 0.40, size: -5 },
      { price: 0.42, size: 100 },
    ];
    const r = computeVwap(dirty, 50, "buy");
    assertTrue(near(r.vwap ?? -1, 0.42), "só o nível válido conta");
  });

  test("bestPrice: buy=menor ask, sell=maior bid", () => {
    assertTrue(near(bestPrice(asks, "buy") ?? -1, 0.40), "menor ask");
    assertTrue(near(bestPrice(asks, "sell") ?? -1, 0.45), "maior bid");
    assertEqual(bestPrice([], "buy"), null, "vazio → null");
  });

  test("depthTopN: soma sizes dos N melhores níveis", () => {
    assertEqual(depthTopN(asks, 3, "buy"), 600, "top3 = 600");
    assertEqual(depthTopN(asks, 2, "buy"), 300, "top2 = 300 (0.40+0.42)");
    assertEqual(depthTopN(asks, 1, "buy"), 100, "top1 = 100");
  });

  test("binaryUnderroundFlag: ask_yes+ask_no < 1−minGross → flag", () => {
    assertTrue(binaryUnderroundFlag(0.47, 0.50, 0.003), "0.97 < 0.997 → flag");
    assertTrue(!binaryUnderroundFlag(0.499, 0.499, 0.003), "0.998 > 0.997 não cruza");
    assertTrue(!binaryUnderroundFlag(0.50, 0.51, 0.003), "1.01 coerente");
  });

  test("binaryOverroundFlag: bid_yes+bid_no > 1+minGross → flag", () => {
    assertTrue(binaryOverroundFlag(0.55, 0.50, 0.003), "1.05 > 1.003 → flag");
    assertTrue(!binaryOverroundFlag(0.50, 0.50, 0.003), "1.00 não cruza");
  });

  test("partitionUnderroundFlag: Σask < 1−minGross com ≥2 pernas", () => {
    assertTrue(partitionUnderroundFlag([0.19, 0.19, 0.19, 0.19, 0.19], 0.003), "0.95 → flag");
    assertTrue(!partitionUnderroundFlag([0.25, 0.25, 0.25, 0.25], 0.003), "1.00 coerente");
    assertTrue(!partitionUnderroundFlag([0.5], 0.003), "1 perna não conta");
    assertTrue(!partitionUnderroundFlag([0.5, 1.2], 0.003), "ask inválido filtrado → <2 válidas");
  });

  test("targetSharesPerLeg: tamanho/capital por unidade", () => {
    assertTrue(near(targetSharesPerLeg(100, 0.97), 100 / 0.97), "100/0.97");
    assertEqual(targetSharesPerLeg(100, 0), 0, "capital 0 → 0");
    assertEqual(targetSharesPerLeg(0, 0.97), 0, "size 0 → 0");
  });

  test("classifyResolutionCategory: crypto/sports/electoral/macro/subjective/unknown", () => {
    assertEqual(
      classifyResolutionCategory("Will Bitcoin reach $100,000 in 2026?"),
      "crypto_feed",
      "crypto",
    );
    assertEqual(
      classifyResolutionCategory("Will the Oklahoma City Thunder win the 2026 NBA Finals?"),
      "sports",
      "sports",
    );
    assertEqual(
      classifyResolutionCategory("Will the Republican win the 2028 presidential election?"),
      "electoral",
      "electoral",
    );
    assertEqual(
      classifyResolutionCategory("Will CPI inflation exceed 3% in Q3?"),
      "macro_data",
      "macro",
    );
    assertEqual(
      classifyResolutionCategory("Will Elon Musk tweet about Mars before 2027?"),
      "subjective",
      "subjective",
    );
    assertEqual(
      classifyResolutionCategory("Some ambiguous market with no keywords"),
      "unknown",
      "unknown fallback",
    );
  });
});
