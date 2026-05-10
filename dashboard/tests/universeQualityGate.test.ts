import fs from "fs";
import path from "path";

import {
  evaluateOpportunityUniverseQuality,
  extractLegMarketIds,
  type LegMarketLookup,
} from "../lib/universeQualityGate";
import type { NormalizedMarket } from "../lib/polymarketClient";
import {
  describe,
  test,
  assertEqual,
  assertTrue,
  assertIncludes,
} from "./_assert";

const NOW = "2026-05-09T20:00:00.000Z";

/** Helper minimalista — preenche default-clean em campos não relevantes para o teste. */
function mkMarket(
  overrides: Partial<NormalizedMarket> & { endDate?: string | null; id: string },
): NormalizedMarket & { endDate?: string | null } {
  return {
    id: overrides.id,
    question: overrides.question ?? "Will the Hurricanes win the 2026 Stanley Cup?",
    slug: overrides.slug ?? overrides.id,
    category: overrides.category ?? "sports",
    outcomes: overrides.outcomes ?? ["YES", "NO"],
    prices: overrides.prices ?? [0.40, 0.60],
    liquidity: overrides.liquidity ?? 5000,
    volume: overrides.volume ?? 20000,
    active: overrides.active ?? true,
    closed: overrides.closed ?? false,
    spread: overrides.spread ?? 0.05,
    probSum: overrides.probSum ?? 1.0,
    endDate: overrides.endDate ?? null,
  };
}

function lookupFrom(
  markets: Array<NormalizedMarket & { endDate?: string | null }>,
): LegMarketLookup {
  const byId = new Map(markets.map(m => [m.id, m]));
  return (legId: string) => byId.get(legId) ?? null;
}

describe("tests/universeQualityGate.test.ts", () => {
  test("1. bloqueia REJECT_POLITICAL_LEGAL — Trump impeachment (regex político real)", () => {
    /** O regex POLITICAL_LEGAL_PATTERNS de marketUniverseQuality casa em
     *  "trump", "impeach(ed|ment)", "election", "putin", "war", etc.
     *  Mid em banda saudável (0.30) — para garantir que NÃO disparou em tail. */
    const opp = {
      opportunityId: "std-impeach",
      sourceType: "standard",
      opportunityType: "cross_market",
      marketsInvolved: [
        { marketId: "TRUMP_M", question: "Will Donald Trump be impeached in 2027?" },
      ],
    };
    const lookup = lookupFrom([
      mkMarket({
        id: "TRUMP_M",
        question: "Will Donald Trump be impeached in 2027?",
        category: "politics",
        prices: [0.30, 0.70],
      }),
    ]);
    const result = evaluateOpportunityUniverseQuality(opp, lookup, NOW);
    assertEqual(result.rejected, true, "deveria rejeitar");
    if (result.rejected) {
      assertEqual(result.verdict, "REJECT_POLITICAL_LEGAL", "verdict político/legal");
      assertEqual(result.legMarketId, "TRUMP_M", "legMarketId reportado");
    }
  });

  test("2. bloqueia REJECT_TAIL_OR_TICK_FENCE — mid fora da banda 0.06-0.94", () => {
    const opp = {
      opportunityId: "std-cm-tailmarket",
      sourceType: "standard",
      opportunityType: "cross_market",
      marketsInvolved: [
        { marketId: "TAIL_M", question: "Will the underdog win the 2026 Final?" },
      ],
    };
    /** mid extremamente baixo via prices — long-shot side em 0.02. */
    const lookup = lookupFrom([
      mkMarket({
        id: "TAIL_M",
        question: "Will the underdog win the 2026 Final?",
        category: "sports",
        prices: [0.02, 0.98],
      }),
    ]);
    const result = evaluateOpportunityUniverseQuality(opp, lookup, NOW);
    assertEqual(result.rejected, true, "deveria rejeitar por tail");
    if (result.rejected) {
      assertEqual(result.verdict, "REJECT_TAIL_OR_TICK_FENCE", "verdict tail/tickFence");
    }
  });

  test("3. bloqueia REJECT_MEME_OR_ABSURD — GTA VI before X", () => {
    const opp = {
      opportunityId: "graph-cluster-meme",
      sourceType: "graph",
      opportunityType: "graph_subset",
      marketsInvolved: [
        { marketId: "MEME_GTA", question: "Will Bitcoin reach $1m before GTA VI?" },
      ],
    };
    const lookup = lookupFrom([
      mkMarket({
        id: "MEME_GTA",
        question: "Will Bitcoin reach $1m before GTA VI?",
        category: "tech",
        prices: [0.30, 0.70],
      }),
    ]);
    const result = evaluateOpportunityUniverseQuality(opp, lookup, NOW);
    assertEqual(result.rejected, true, "deveria rejeitar por meme");
    if (result.rejected) {
      assertEqual(result.verdict, "REJECT_MEME_OR_ABSURD", "verdict meme/absurdo");
    }
  });

  test("4. bloqueia REJECT_LONG_HORIZON_NO_CATALYST — endDate longe sem esporte/fin event", () => {
    const opp = {
      opportunityId: "std-longhorizon",
      sourceType: "standard",
      opportunityType: "cross_market",
      marketsInvolved: [
        { marketId: "LH_M", question: "Will humans land on Mars permanently?" },
      ],
    };
    /** endDate >12 meses adiante, sem keyword de sport/financial event => very_long. */
    const lookup = lookupFrom([
      mkMarket({
        id: "LH_M",
        question: "Will humans land on Mars permanently?",
        category: "general",
        prices: [0.30, 0.70],
        endDate: "2028-12-31T00:00:00.000Z",
      }),
    ]);
    const result = evaluateOpportunityUniverseQuality(opp, lookup, NOW);
    assertEqual(result.rejected, true, "deveria rejeitar por horizonte longo");
    if (result.rejected) {
      assertEqual(
        result.verdict,
        "REJECT_LONG_HORIZON_NO_CATALYST",
        "verdict longHorizon",
      );
    }
  });

  test("5. composite leg — A+B com uma perna ruim → bloqueia", () => {
    const opp = {
      opportunityId: "std-composite",
      sourceType: "standard",
      opportunityType: "cross_market",
      marketsInvolved: [
        { marketId: "GOOD_A+BAD_B", question: "composite" },
      ],
    };
    const lookup = lookupFrom([
      mkMarket({
        id: "GOOD_A",
        question: "Will the Hurricanes win the 2026 Stanley Cup?",
        category: "sports",
        prices: [0.40, 0.60],
      }),
      mkMarket({
        id: "BAD_B",
        question: "Will Trump win the 2028 election?",
        category: "politics",
        prices: [0.30, 0.70],
      }),
    ]);
    const result = evaluateOpportunityUniverseQuality(opp, lookup, NOW);
    assertEqual(result.rejected, true, "composite com 1 perna ruim = bloqueio");
    if (result.rejected) {
      assertEqual(result.legMarketId, "BAD_B", "perna culpada reportada");
      assertEqual(result.verdict, "REJECT_POLITICAL_LEGAL", "verdict bate na perna ruim");
    }
  });

  test("6. permite quando todas as pernas passam (sports clean)", () => {
    const opp = {
      opportunityId: "std-clean",
      sourceType: "standard",
      opportunityType: "cross_market",
      marketsInvolved: [
        { marketId: "CLEAN_X", question: "Will the Hurricanes win the 2026 Stanley Cup?" },
        { marketId: "CLEAN_Y", question: "Will the Spurs win the 2026 NBA Finals?" },
      ],
    };
    const lookup = lookupFrom([
      mkMarket({
        id: "CLEAN_X",
        question: "Will the Hurricanes win the 2026 Stanley Cup?",
        category: "sports",
        prices: [0.40, 0.60],
        liquidity: 5000,
        volume: 20000,
      }),
      mkMarket({
        id: "CLEAN_Y",
        question: "Will the Spurs win the 2026 NBA Finals?",
        category: "sports",
        prices: [0.30, 0.70],
        liquidity: 5000,
        volume: 20000,
      }),
    ]);
    const result = evaluateOpportunityUniverseQuality(opp, lookup, NOW);
    assertEqual(result.rejected, false, "todas as pernas limpas → permite");
    if (!result.rejected) {
      /** O verdict reportado pode ser CLEAN_OBSERVATION_UNIVERSE, ACCEPTABLE_BUT_SECONDARY
       *  ou REJECT_AMBIGUOUS (data-limited por endDate ausente — não é HARD reject).
       *  O contrato do gate é rejected=false; o verdict é puramente descritivo. */
      const HARD = new Set([
        "REJECT_POLITICAL_LEGAL",
        "REJECT_MEME_OR_ABSURD",
        "REJECT_TAIL_OR_TICK_FENCE",
        "REJECT_LONG_HORIZON_NO_CATALYST",
        "REJECT_NOT_SUITABLE",
      ]);
      assertTrue(
        !HARD.has(result.verdict),
        "verdict final não é HARD reject (rejected=false já garantiu)",
      );
      assertEqual(result.legsChecked, 2, "checou as 2 pernas");
    }
  });

  test("7. evaluateOpportunity NÃO é chamado quando bloqueado — verificado por estrutura do dispatcher", () => {
    /** Verifica que executionDispatcher.ts encadeia o gate ANTES do dispatch:
     *  1. import existe; 2. chamada do gate aparece antes da chamada de evaluateOpportunity;
     *  3. há um early-return entre o gate e o evaluate. */
    const dispatcherPath = path.resolve(__dirname, "../lib/executionDispatcher.ts");
    const src = fs.readFileSync(dispatcherPath, "utf8");
    assertIncludes(src, "evaluateOpportunityUniverseQuality", "gate é importado");
    assertIncludes(src, "BLOCKED_BY_UNIVERSE_QUALITY", "early-exit do UQ existe");
    const gateIdx = src.indexOf("evaluateOpportunityUniverseQuality(");
    const evalIdx = src.indexOf("evaluateOpportunity(opportunity)");
    assertTrue(
      gateIdx >= 0 && evalIdx > gateIdx,
      "gate precede evaluateOpportunity no fluxo",
    );
    /** Confirma que existe um `return` entre o gate (na branch rejected) e o evaluate. */
    const between = src.substring(gateIdx, evalIdx);
    assertIncludes(between, "uqGate.rejected", "checa rejected");
    assertIncludes(between, "return;", "retorna cedo no rejected");
  });

  test("8. cobre standard E graph — mesmo gate, opportunityTypes diferentes", () => {
    const stdOpp = {
      opportunityId: "std-cover-test",
      sourceType: "standard",
      opportunityType: "cross_market",
      marketsInvolved: [
        { marketId: "POL_M", question: "Will Putin remain in power in 2027?" },
      ],
    };
    const graphOpp = {
      opportunityId: "graph-cover-test",
      sourceType: "graph",
      opportunityType: "graph_cycle",
      marketsInvolved: [
        { marketId: "POL_M", question: "Will Putin remain in power in 2027?" },
      ],
    };
    const lookup = lookupFrom([
      mkMarket({
        id: "POL_M",
        question: "Will Putin remain in power in 2027?",
        category: "politics",
        prices: [0.30, 0.70],
      }),
    ]);
    const stdRes = evaluateOpportunityUniverseQuality(stdOpp, lookup, NOW);
    const graphRes = evaluateOpportunityUniverseQuality(graphOpp, lookup, NOW);
    assertEqual(stdRes.rejected, true, "standard rejeitado");
    assertEqual(graphRes.rejected, true, "graph rejeitado");
    if (stdRes.rejected && graphRes.rejected) {
      assertEqual(stdRes.verdict, graphRes.verdict, "verdict idêntico para mesmo dado");
    }
  });

  test("9. .paper não é criado no curso destes testes", () => {
    const repoRoot = path.resolve(__dirname, "../..");
    const dashboardRoot = path.resolve(__dirname, "..");
    const candidates = [
      path.join(repoRoot, ".paper"),
      path.join(dashboardRoot, ".paper"),
      path.join(process.cwd(), ".paper"),
    ];
    for (const p of candidates) {
      assertEqual(fs.existsSync(p), false, `não criou .paper em ${p}`);
    }
  });

  test("10. extractLegMarketIds: composto A+B → ['A', 'B'] (ordem preservada)", () => {
    const ids = extractLegMarketIds({
      marketsInvolved: [
        { marketId: "553856+565065", question: "x" },
        { marketId: "999999", question: "y" },
      ],
    });
    assertEqual(ids.length, 3, "3 pernas extraídas");
    assertTrue(ids.includes("553856"), "leg 553856");
    assertTrue(ids.includes("565065"), "leg 565065");
    assertTrue(ids.includes("999999"), "leg 999999");
  });

  /** ===== Gate A — opp.edge implica preço efetivo de cauda ===== */

  test("A1. graph_cycle com opp.edge=0.9905 + cache prices=[0.5,0.5] saudável → BLOCKED via REJECT_TAIL_VIA_OPP_EDGE", () => {
    /** Cenário Fed cycle-39 reproduzido. Cache reportaria mid saudável (0.5) e
     *  o gate por leg passaria. Mas o executor abriria em 1 - 0.9905 = 0.0095.
     *  O Gate A bloqueia ANTES, sem nem consultar o cache. */
    const opp = {
      opportunityId: "graph-cluster-general-0-cycle-39",
      sourceType: "graph",
      opportunityType: "graph_cycle",
      edge: 0.9905,
      marketsInvolved: [
        { marketId: "FED_10", question: "Will 10 Fed rate cuts happen in 2026?" },
      ],
    };
    /** Cache deliberadamente saudável — prova que Gate A bloqueia mesmo com cache "limpo". */
    const lookup = lookupFrom([
      mkMarket({
        id: "FED_10",
        question: "Will 10 Fed rate cuts happen in 2026?",
        category: "macro",
        prices: [0.50, 0.50],
        liquidity: 5000,
        volume: 20000,
      }),
    ]);
    const result = evaluateOpportunityUniverseQuality(opp, lookup, NOW);
    assertEqual(result.rejected, true, "deveria bloquear via opp.edge");
    if (result.rejected) {
      assertEqual(result.verdict, "REJECT_TAIL_VIA_OPP_EDGE", "verdict tail via opp.edge");
      assertEqual(result.legsChecked, 0, "fail-fast: pernas não foram iteradas");
      /** mid no payload de rejeição = effectiveEntryPriceProxy (1 - clampedEdge). */
      assertEqual(result.mid, 1 - 0.9905, "mid reporta proxy do preço efetivo");
      assertIncludes(
        result.reasons.join(","),
        "edge_implies_lower_tail_entry",
        "rationale lower tail",
      );
    }
  });

  test("A2. standard cross_market com opp.edge=0.95 → BLOCKED via REJECT_TAIL_VIA_OPP_EDGE", () => {
    const opp = {
      opportunityId: "std-cm-edge-tail",
      sourceType: "standard",
      opportunityType: "cross_market",
      edge: 0.95,
      marketsInvolved: [
        { marketId: "MA", question: "A" },
        { marketId: "MB", question: "B" },
      ],
    };
    const lookup = lookupFrom([
      mkMarket({ id: "MA", category: "sports", prices: [0.5, 0.5] }),
      mkMarket({ id: "MB", category: "sports", prices: [0.5, 0.5] }),
    ]);
    const result = evaluateOpportunityUniverseQuality(opp, lookup, NOW);
    assertEqual(result.rejected, true, "deveria bloquear");
    if (result.rejected) {
      assertEqual(result.verdict, "REJECT_TAIL_VIA_OPP_EDGE", "verdict tail via opp.edge");
    }
  });

  test("A3. graph com opp.edge=0.5 (mediano) + leg limpa → ALLOW (Gate A não dispara)", () => {
    const opp = {
      opportunityId: "graph-median",
      sourceType: "graph",
      opportunityType: "graph_subset",
      edge: 0.5,
      marketsInvolved: [
        { marketId: "CLEAN_X", question: "Will the Hurricanes win the 2026 Stanley Cup?" },
      ],
    };
    const lookup = lookupFrom([
      mkMarket({
        id: "CLEAN_X",
        question: "Will the Hurricanes win the 2026 Stanley Cup?",
        category: "sports",
        prices: [0.40, 0.60],
        liquidity: 5000,
        volume: 20000,
      }),
    ]);
    const result = evaluateOpportunityUniverseQuality(opp, lookup, NOW);
    assertEqual(result.rejected, false, "edge mediano + leg limpa = allow");
  });

  test("A4. boundary opp.edge=0.94 → ALLOW (priceProxy=0.06 está no safe band, NÃO em cauda)", () => {
    /** Boundary explícito: 1 - 0.94 = 0.06. UQ usa `mid < 0.06` (strict).
     *  Gate A espelha: priceProxy=0.06 está no safe band, NÃO bloqueia. */
    const opp = {
      opportunityId: "graph-boundary-094",
      sourceType: "graph",
      opportunityType: "graph_subset",
      edge: 0.94,
      marketsInvolved: [
        { marketId: "CLEAN_BD", question: "Will the Hurricanes win the 2026 Stanley Cup?" },
      ],
    };
    const lookup = lookupFrom([
      mkMarket({
        id: "CLEAN_BD",
        question: "Will the Hurricanes win the 2026 Stanley Cup?",
        category: "sports",
        prices: [0.40, 0.60],
        liquidity: 5000,
        volume: 20000,
      }),
    ]);
    const result = evaluateOpportunityUniverseQuality(opp, lookup, NOW);
    assertEqual(result.rejected, false, "boundary 0.94 não dispara Gate A");
    /** Sanity: edge=0.9401 dispararia. */
    const oppOver = { ...opp, edge: 0.9401 };
    const resultOver = evaluateOpportunityUniverseQuality(oppOver, lookup, NOW);
    assertEqual(resultOver.rejected, true, "0.9401 já é cauda — bloqueia");
  });

  test("A5. underround/edge baixo (0.01) → BLOCKED via tail superior (priceProxy=0.99)", () => {
    /** Quando o edge é tão pequeno que 1 - edge > 0.94, executor abriria em
     *  preço alto demais (cauda superior). Gate A bloqueia simetricamente. */
    const opp = {
      opportunityId: "std-underround-small-edge",
      sourceType: "standard",
      opportunityType: "underround",
      edge: 0.01,
      marketsInvolved: [
        { marketId: "U_M", question: "Some market" },
      ],
    };
    const lookup = lookupFrom([
      mkMarket({ id: "U_M", category: "sports", prices: [0.5, 0.5] }),
    ]);
    const result = evaluateOpportunityUniverseQuality(opp, lookup, NOW);
    assertEqual(result.rejected, true, "edge baixo → priceProxy alto → bloqueia");
    if (result.rejected) {
      assertEqual(result.verdict, "REJECT_TAIL_VIA_OPP_EDGE", "verdict tail via opp.edge");
      assertIncludes(
        result.reasons.join(","),
        "edge_implies_upper_tail_entry",
        "rationale upper tail",
      );
    }
    /** Bonus: edge negativo (clamp do executor) também cai aqui. */
    const oppNeg = { ...opp, edge: -0.5 };
    const resultNeg = evaluateOpportunityUniverseQuality(oppNeg, lookup, NOW);
    assertEqual(resultNeg.rejected, true, "edge negativo (clamp 0) → priceProxy=1 → bloqueia");
  });

  test("A6. opp sem campo edge ou edge NaN → fall through para leg check (não rejeita por Gate A)", () => {
    const oppNoEdge = {
      opportunityId: "no-edge",
      sourceType: "graph",
      opportunityType: "graph_subset",
      marketsInvolved: [
        { marketId: "CLEAN_NE", question: "Will the Hurricanes win the 2026 Stanley Cup?" },
      ],
    };
    const lookup = lookupFrom([
      mkMarket({
        id: "CLEAN_NE",
        question: "Will the Hurricanes win the 2026 Stanley Cup?",
        category: "sports",
        prices: [0.40, 0.60],
      }),
    ]);
    const result = evaluateOpportunityUniverseQuality(oppNoEdge, lookup, NOW);
    /** Gate A não rejeita por falta de edge; leg-check é quem decide. */
    if (result.rejected) {
      assertTrue(
        result.verdict !== "REJECT_TAIL_VIA_OPP_EDGE",
        "se rejeitar, NÃO foi por Gate A",
      );
    }
    /** edge NaN → mesmo comportamento. */
    const oppNaN = { ...oppNoEdge, edge: Number.NaN };
    const resultNaN = evaluateOpportunityUniverseQuality(oppNaN, lookup, NOW);
    if (resultNaN.rejected) {
      assertTrue(
        resultNaN.verdict !== "REJECT_TAIL_VIA_OPP_EDGE",
        "NaN não dispara Gate A",
      );
    }
  });

  test("A7. dispatcher bloqueia (early-return) quando Gate A rejeita — verificado por estrutura", () => {
    /** Mesma garantia estrutural do test 7, agora explicitando o novo verdict.
     *  Como o gate retorna { rejected: true } com verdict REJECT_TAIL_VIA_OPP_EDGE,
     *  o dispatcher executa o branch existente `if (uqGate.rejected)` → return,
     *  bloqueando antes de incrementDispatch / EEV / evaluateOpportunity. */
    const dispatcherPath = path.resolve(__dirname, "../lib/executionDispatcher.ts");
    const src = fs.readFileSync(dispatcherPath, "utf8");
    /** O branch genérico `if (uqGate.rejected)` cobre o novo verdict sem mudança
     *  no dispatcher: o early-return é a mesma instrução. */
    assertIncludes(src, "if (uqGate.rejected)", "branch genérico de rejeição existe");
    assertIncludes(src, "BLOCKED_BY_UNIVERSE_QUALITY:${uqGate.verdict}", "counter usa verdict dinâmico");
    /** Confere que a string do counter incluiria REJECT_TAIL_VIA_OPP_EDGE quando aplicável. */
    const opp = {
      opportunityId: "g-trip",
      sourceType: "graph",
      opportunityType: "graph_cycle",
      edge: 0.99,
      marketsInvolved: [{ marketId: "X", question: "x" }],
    };
    const r = evaluateOpportunityUniverseQuality(opp, () => null, NOW);
    assertEqual(r.rejected, true, "gate rejeita via Gate A");
    if (r.rejected) {
      assertEqual(r.verdict, "REJECT_TAIL_VIA_OPP_EDGE", "verdict alimentaria o counter dinâmico");
    }
  });
});
