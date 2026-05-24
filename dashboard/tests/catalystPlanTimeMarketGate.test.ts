import {
  evaluatePlanTimeMarketGate,
  PLAN_TIME_MAX_SPREAD,
  PLAN_TIME_MID_FLOOR,
  PLAN_TIME_MID_CEILING,
} from "../lib/catalystPlanTimeMarketGate";
import { describe, test, assertEqual, assertTrue } from "./_assert";

describe("tests/catalystPlanTimeMarketGate.test.ts", () => {
  test("constantes do gate são mais conservadoras que thresholds da hipótese", () => {
    /** Hipótese (referência, não modificada): MAX_POST_IMMEDIATE_SPREAD=0.05,
     *  TAIL_LOWER=0.06, TAIL_UPPER=0.94. */
    assertTrue(PLAN_TIME_MAX_SPREAD < 0.05, "spread mais apertado que hipótese");
    assertTrue(PLAN_TIME_MID_FLOOR > 0.06, "floor mais conservador que tail lower");
    assertTrue(PLAN_TIME_MID_CEILING < 0.94, "ceiling mais conservador que tail upper");
  });

  test("CASO 1 — mercado bom (spread 0.02, mid 0.41) passa", () => {
    const r = evaluatePlanTimeMarketGate({ bestBid: 0.40, bestAsk: 0.42 });
    assertEqual(r.accepted, true, "deve passar");
    assertEqual(r.reason, "passed", "reason=passed");
    assertTrue(Math.abs((r.probedSpread ?? 0) - 0.02) < 1e-9, "spread 0.02");
    assertTrue(Math.abs((r.probedMid ?? 0) - 0.41) < 1e-9, "mid 0.41");
  });

  test("CASO 2 — mercado com spread largo (0.10) é excluído", () => {
    const r = evaluatePlanTimeMarketGate({ bestBid: 0.40, bestAsk: 0.50 });
    assertEqual(r.accepted, false, "deve ser excluído");
    assertEqual(r.reason, "spread_above_plan_max", "reason correto");
    assertTrue(Math.abs((r.probedSpread ?? 0) - 0.10) < 1e-9, "spread 0.10 reportado");
    assertTrue(Math.abs((r.probedMid ?? 0) - 0.45) < 1e-9, "mid 0.45 reportado");
  });

  test("CASO 3a — mercado com mid baixo demais (0.05) é excluído por mid_below_plan_floor", () => {
    const r = evaluatePlanTimeMarketGate({ bestBid: 0.04, bestAsk: 0.06 });
    assertEqual(r.accepted, false, "deve ser excluído");
    assertEqual(r.reason, "mid_below_plan_floor", "reason correto");
    assertTrue(Math.abs((r.probedMid ?? 0) - 0.05) < 1e-9, "mid 0.05 reportado");
  });

  test("CASO 3b — mercado com mid alto demais (0.93) é excluído por mid_above_plan_ceiling", () => {
    const r = evaluatePlanTimeMarketGate({ bestBid: 0.92, bestAsk: 0.94 });
    assertEqual(r.accepted, false, "deve ser excluído");
    assertEqual(r.reason, "mid_above_plan_ceiling", "reason correto");
    assertTrue(Math.abs((r.probedMid ?? 0) - 0.93) < 1e-9, "mid 0.93 reportado");
  });

  test("boundary: spread exatamente no maxSpread (0.04) passa (inclusivo)", () => {
    const r = evaluatePlanTimeMarketGate({ bestBid: 0.40, bestAsk: 0.44 });
    assertEqual(r.accepted, true, "boundary inclusivo");
    assertTrue(Math.abs((r.probedSpread ?? 0) - 0.04) < 1e-9, "spread exato");
  });

  test("boundary: mid exatamente no midFloor (0.10) passa (inclusivo)", () => {
    const r = evaluatePlanTimeMarketGate({ bestBid: 0.09, bestAsk: 0.11 });
    assertEqual(r.accepted, true, "boundary inclusivo");
    assertTrue(Math.abs((r.probedMid ?? 0) - 0.10) < 1e-9, "mid exato");
  });

  test("boundary: mid exatamente no midCeiling (0.90) passa (inclusivo)", () => {
    const r = evaluatePlanTimeMarketGate({ bestBid: 0.89, bestAsk: 0.91 });
    assertEqual(r.accepted, true, "boundary inclusivo");
    assertTrue(Math.abs((r.probedMid ?? 0) - 0.90) < 1e-9, "mid exato");
  });

  test("defensive: bid ou ask null → missing_book_prices (rejeitado)", () => {
    assertEqual(
      evaluatePlanTimeMarketGate({ bestBid: null, bestAsk: 0.50 }).reason,
      "missing_book_prices",
      "bid null",
    );
    assertEqual(
      evaluatePlanTimeMarketGate({ bestBid: 0.40, bestAsk: null }).reason,
      "missing_book_prices",
      "ask null",
    );
    assertEqual(
      evaluatePlanTimeMarketGate({ bestBid: null, bestAsk: null }).reason,
      "missing_book_prices",
      "ambos null",
    );
  });

  test("defensive: NaN/Infinity em bid ou ask → missing_book_prices", () => {
    assertEqual(
      evaluatePlanTimeMarketGate({ bestBid: NaN, bestAsk: 0.50 }).reason,
      "missing_book_prices",
      "NaN bid",
    );
    assertEqual(
      evaluatePlanTimeMarketGate({ bestBid: 0.40, bestAsk: Infinity }).reason,
      "missing_book_prices",
      "Infinity ask",
    );
  });

  test("defensive: book invertido (ask < bid) → inverted_book", () => {
    const r = evaluatePlanTimeMarketGate({ bestBid: 0.50, bestAsk: 0.40 });
    assertEqual(r.accepted, false, "rejeita book invertido");
    assertEqual(r.reason, "inverted_book", "reason correto");
  });

  test("override de config aplica thresholds custom", () => {
    /** Gate mais frouxo passa um mercado de spread 0.08 que default rejeitaria. */
    const r = evaluatePlanTimeMarketGate(
      { bestBid: 0.40, bestAsk: 0.48 },
      { maxSpread: 0.10, midFloor: 0.05, midCeiling: 0.95 },
    );
    assertEqual(r.accepted, true, "config custom aceita");
    assertTrue(Math.abs((r.probedSpread ?? 0) - 0.08) < 1e-9, "spread 0.08");
  });

  test("cenário Fase 1.B: mercado típico que falhou (mid 0.04, spread 0.62) é rejeitado", () => {
    /** Reproduz o cenário da auditoria: mid de cauda + spread enorme.
     *  bid=0.0, ask=0.62 seria um book degenerado; uso valores plausíveis: */
    const r = evaluatePlanTimeMarketGate({ bestBid: -0.27, bestAsk: 0.35 });
    /** Book com bid negativo é tratado como missing/inválido em real-world; mas
     *  aqui simplesmente como inverted/missing — nosso ponto é que valores assim
     *  não passam. */
    assertEqual(r.accepted, false, "rejeitado");
  });

  test("cenário Fase 1.B realista: mid 0.10 com spread 0.06 → spread_above_plan_max", () => {
    /** Spread acima do plan_max (0.04), mesmo com mid no piso aceitável. */
    const r = evaluatePlanTimeMarketGate({ bestBid: 0.07, bestAsk: 0.13 });
    assertEqual(r.accepted, false, "rejeitado");
    assertEqual(r.reason, "spread_above_plan_max", "spread é o gargalo");
    assertTrue(Math.abs((r.probedSpread ?? 0) - 0.06) < 1e-9, "spread 0.06");
  });
});
