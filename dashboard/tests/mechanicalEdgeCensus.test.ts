import fs from "fs";
import path from "path";

import {
  evaluateMecBasket,
  MEC_DEFAULT_COST_MODEL,
  MEC_VIABLE_NET,
  MEC_MARGINAL_NET,
  MEC_VERSION,
  type MecLeg,
  type MecBasketInput,
} from "../lib/mechanicalEdgeCensus";
import { describe, test, assertEqual, assertTrue } from "./_assert";

const COSTS = MEC_DEFAULT_COST_MODEL;

function leg(o: Partial<MecLeg> & { vwapPrice: number }): MecLeg {
  return {
    marketId: o.marketId ?? "m",
    side: o.side ?? "buy",
    vwapPrice: o.vwapPrice,
    bestPrice: o.bestPrice ?? o.vwapPrice,
    depthTop3: o.depthTop3 ?? 1000,
    spread: o.spread ?? 0.01,
  };
}

function near(a: number, b: number, tol = 1e-5): boolean {
  return Math.abs(a - b) < tol;
}

describe("tests/mechanicalEdgeCensus.test.ts", () => {
  test("constantes de banda coerentes (item 4: 0.5%–1%)", () => {
    assertEqual(MEC_MARGINAL_NET, 0.005, "marginal floor 0.5%");
    assertEqual(MEC_VIABLE_NET, 0.01, "viable floor 1.0%");
    assertTrue(MEC_VIABLE_NET > MEC_MARGINAL_NET, "viable acima de marginal");
  });

  test("CASO 1 — binary underround, resolução curta, profundo → viable_pending_persistence", () => {
    const input: MecBasketInput = {
      legs: [leg({ vwapPrice: 0.48 }), leg({ vwapPrice: 0.49 })],
      edgeType: "BINARY_UNDERROUND",
      daysToResolution: 7,
      category: "crypto_feed",
    };
    const r = evaluateMecBasket(input, COSTS);
    assertEqual(r.k, 2, "k=2");
    assertTrue(near(r.grossEdge, 0.03), "gross 0.03");
    assertTrue(r.netEdge >= MEC_VIABLE_NET, "net ≥ 1%");
    assertEqual(r.verdict, "viable_pending_persistence", "viable snapshot");
    assertEqual(r.canUseForExecution, false, "nunca autoriza execução");
    assertEqual(r.mecVersion, MEC_VERSION, "version tag");
  });

  test("CASO 2 — mesmo gross mas resolução longa (180d) → lockup mata → negative_after_costs", () => {
    const input: MecBasketInput = {
      legs: [leg({ vwapPrice: 0.48 }), leg({ vwapPrice: 0.49 })],
      edgeType: "BINARY_UNDERROUND",
      daysToResolution: 180,
      category: "crypto_feed",
    };
    const r = evaluateMecBasket(input, COSTS);
    assertTrue(near(r.grossEdge, 0.03), "gross ainda 0.03");
    assertTrue(r.costs.cLockup > 0.04, "lockup domina (>4%)");
    assertTrue(r.netEdge < 0, "net negativo");
    assertEqual(r.verdict, "negative_after_costs", "armadilha do lockup capturada");
  });

  test("CASO 3 — partition k=5 com net positivo mas profundidade insuficiente → capacity_insufficient", () => {
    const legs = Array.from({ length: 5 }, (_, i) =>
      leg({ marketId: `p${i}`, vwapPrice: 0.19, depthTop3: 50 }),
    );
    const r = evaluateMecBasket(
      { legs, edgeType: "PARTITION_UNDERROUND", daysToResolution: 10, category: "sports" },
      COSTS,
    );
    assertEqual(r.k, 5, "k=5");
    assertTrue(near(r.grossEdge, 0.05), "gross 0.05");
    assertTrue(r.netEdge >= MEC_VIABLE_NET, "net seria viável");
    assertTrue(!r.capacityOk, "capacidade insuficiente");
    assertEqual(r.verdict, "capacity_insufficient", "barrado por profundidade");
  });

  test("CASO 4 — net na banda marginal (0.5%–1%) → marginal_pending_persistence", () => {
    const r = evaluateMecBasket(
      {
        legs: [leg({ vwapPrice: 0.49, spread: 0.02 }), leg({ vwapPrice: 0.49, spread: 0.02 })],
        edgeType: "BINARY_UNDERROUND",
        daysToResolution: 7,
        category: "crypto_feed",
      },
      COSTS,
    );
    assertTrue(r.netEdge >= MEC_MARGINAL_NET && r.netEdge < MEC_VIABLE_NET, "net na faixa marginal");
    assertEqual(r.verdict, "marginal_pending_persistence", "marginal");
  });

  test("CASO 5 — net entre 0 e 0.5% → not_viable", () => {
    const r = evaluateMecBasket(
      {
        legs: [leg({ vwapPrice: 0.494 }), leg({ vwapPrice: 0.494 })],
        edgeType: "BINARY_UNDERROUND",
        daysToResolution: 7,
        category: "crypto_feed",
      },
      COSTS,
    );
    assertTrue(r.netEdge >= 0 && r.netEdge < MEC_MARGINAL_NET, "net pequeno positivo");
    assertEqual(r.verdict, "not_viable", "abaixo do piso marginal");
  });

  test("CASO 6 — UMA subjective (2%) derruba uma cesta que seria viável", () => {
    const base = {
      legs: [leg({ vwapPrice: 0.48 }), leg({ vwapPrice: 0.49 })],
      edgeType: "BINARY_UNDERROUND" as const,
      daysToResolution: 7,
    };
    const cryptoR = evaluateMecBasket({ ...base, category: "crypto_feed" }, COSTS);
    const subjR = evaluateMecBasket({ ...base, category: "subjective" }, COSTS);
    assertEqual(cryptoR.verdict, "viable_pending_persistence", "crypto viável");
    assertTrue(near(subjR.costs.cUma, 0.02), "uma subjective 2%");
    assertTrue(subjR.netEdge < cryptoR.netEdge, "uma maior reduz net");
    assertEqual(subjR.verdict, "not_viable", "subjective barra a cesta");
  });

  test("CASO 7 — slippage de profundidade: usa VWAP, não best level", () => {
    /** best somaria 0.94 (gross naive 0.06), mas VWAP soma 1.00 → gross real 0. */
    const r = evaluateMecBasket(
      {
        legs: [
          leg({ vwapPrice: 0.5, bestPrice: 0.47 }),
          leg({ vwapPrice: 0.5, bestPrice: 0.47 }),
        ],
        edgeType: "BINARY_UNDERROUND",
        daysToResolution: 7,
        category: "crypto_feed",
      },
      COSTS,
    );
    assertEqual(r.grossEdge, 0, "gross calculado no VWAP = 0 (não 0.06 do best)");
    assertTrue(near(r.costs.cSlippageDepth, 0.06), "slippage de profundidade registrado");
    assertEqual(r.verdict, "negative_after_costs", "sem gross, net negativo");
  });

  test("defensivo: menos de 2 pernas → invalid_input", () => {
    const r = evaluateMecBasket(
      { legs: [leg({ vwapPrice: 0.5 })], edgeType: "BINARY_UNDERROUND", daysToResolution: 5, category: "crypto_feed" },
      COSTS,
    );
    assertEqual(r.verdict, "invalid_input", "k<2 inválido");
  });

  test("defensivo: vwap fora de (0,1) → invalid_input", () => {
    const r = evaluateMecBasket(
      {
        legs: [leg({ vwapPrice: 1.2 }), leg({ vwapPrice: 0.4 })],
        edgeType: "BINARY_UNDERROUND",
        daysToResolution: 5,
        category: "crypto_feed",
      },
      COSTS,
    );
    assertEqual(r.verdict, "invalid_input", "preço inválido");
  });

  test("CASO 10 — overround (sell) básico: Σbid > 1 → gross positivo", () => {
    const r = evaluateMecBasket(
      {
        legs: [
          leg({ side: "sell", vwapPrice: 0.55 }),
          leg({ side: "sell", vwapPrice: 0.52 }),
        ],
        edgeType: "BINARY_OVERROUND",
        daysToResolution: 7,
        category: "crypto_feed",
      },
      COSTS,
    );
    assertTrue(near(r.grossEdge, 0.07), "gross 0.07 (1.07−1)");
    assertEqual(r.capitalPerUnit, 1, "collateral conservador = 1 no sell");
    assertEqual(r.verdict, "viable_pending_persistence", "overround viável");
  });

  test("CASO 11 — negrisk: fee de conversão alta empurra para negative_after_costs", () => {
    const legs = Array.from({ length: 4 }, (_, i) => leg({ marketId: `n${i}`, vwapPrice: 0.24 }));
    const noFee = evaluateMecBasket(
      { legs, edgeType: "NEGRISK_CONVERSION", daysToResolution: 30, category: "electoral", conversionFeeFrac: 0 },
      COSTS,
    );
    const bigFee = evaluateMecBasket(
      { legs, edgeType: "NEGRISK_CONVERSION", daysToResolution: 30, category: "electoral", conversionFeeFrac: 0.03 },
      COSTS,
    );
    assertEqual(noFee.costs.cConversion, 0, "sem fee");
    assertTrue(near(bigFee.costs.cConversion, 0.03), "fee 3% registrada");
    assertTrue(bigFee.netEdge < noFee.netEdge, "fee reduz net");
    assertEqual(bigFee.verdict, "negative_after_costs", "fee alta inviabiliza");
  });

  test("CASO 13 — dias negativos (resolvido) → lockup zero", () => {
    const r = evaluateMecBasket(
      {
        legs: [leg({ vwapPrice: 0.48 }), leg({ vwapPrice: 0.49 })],
        edgeType: "BINARY_UNDERROUND",
        daysToResolution: -5,
        category: "crypto_feed",
      },
      COSTS,
    );
    assertEqual(r.daysToResolution, 0, "dias negativos viram 0");
    assertEqual(r.costs.cLockup, 0, "lockup zero");
  });

  test("capacidade ok quando profundidade ≥ unidades-alvo", () => {
    const r = evaluateMecBasket(
      {
        legs: [leg({ vwapPrice: 0.48, depthTop3: 5000 }), leg({ vwapPrice: 0.49, depthTop3: 5000 })],
        edgeType: "BINARY_UNDERROUND",
        daysToResolution: 7,
        category: "crypto_feed",
      },
      COSTS,
    );
    assertTrue(r.capacityOk, "profundidade cobre alvo");
    assertTrue(r.unitsTargeted > 0, "unidades positivas");
  });

  test("default cost model expõe a tabela UMA calibrada", () => {
    const u = MEC_DEFAULT_COST_MODEL.umaHaircutByCategory;
    assertEqual(u.crypto_feed, 0.001, "crypto 0.1%");
    assertEqual(u.sports, 0.003, "sports 0.3%");
    assertEqual(u.macro_data, 0.004, "macro 0.4%");
    assertEqual(u.electoral, 0.006, "electoral 0.6%");
    assertEqual(u.subjective, 0.02, "subjective 2%");
    assertEqual(u.unknown, 0.01, "unknown 1%");
    assertEqual(MEC_DEFAULT_COST_MODEL.costOfCapitalAnnual, 0.1, "capital 10%");
    assertEqual(MEC_DEFAULT_COST_MODEL.targetSizeUsd, 100, "target $100");
  });

  test("governança: lib não importa execução / paper / wallet / dispatcher", () => {
    const libPath = path.join(__dirname, "../lib/mechanicalEdgeCensus.ts");
    const raw = fs.readFileSync(libPath, "utf8");
    /** Remove comentários (bloco e linha): o teste verifica CÓDIGO, não a
     *  declaração no cabeçalho que lista os termos proibidos para negá-los. */
    const hay = raw.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
    const forbidden: RegExp[] = [
      /executionDispatcher/,
      /paperPortfolioStore/,
      /shadowSimulationStore/,
      /opportunityEngine/,
      /probabilityScanner/,
      /\bcreateOrder\b/,
      /\bsubmitOrder\b/,
      /\bwallet\b/i,
      /\bsigner\b/i,
      /privateKey/i,
    ];
    for (const re of forbidden) {
      assertTrue(!re.test(hay), `lib evita ${re}`);
    }
  });
});
