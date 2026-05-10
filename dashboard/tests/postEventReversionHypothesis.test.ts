import fs from "fs";
import path from "path";

import {
  isHypothesisEligibleMarket,
  computeReversionMetric,
  judgeHypothesis,
  SIGNAL_THRESHOLD,
  MIN_QUALIFIED_N,
  type EventSnapshot,
  type ReversionMetric,
  type SnapshotData,
} from "../lib/postEventReversionHypothesis";
import type { NormalizedMarket } from "../lib/polymarketClient";
import { describe, test, assertEqual, assertTrue } from "./_assert";

const NOW = "2026-05-09T20:00:00.000Z";

function mkMarket(
  o: Partial<NormalizedMarket> & { id: string; question: string },
): NormalizedMarket {
  return {
    id: o.id,
    question: o.question,
    slug: o.slug ?? o.id,
    category: o.category ?? "sports",
    outcomes: o.outcomes ?? ["YES", "NO"],
    prices: o.prices ?? [0.40, 0.60],
    liquidity: o.liquidity ?? 5000,
    volume: o.volume ?? 20000,
    active: o.active ?? true,
    closed: o.closed ?? false,
    spread: o.spread ?? 0.05,
    probSum: o.probSum ?? 1.0,
  };
}

function mkSnap(mid: number, spread = 0.02): SnapshotData {
  return {
    capturedAtUtc: NOW,
    mid,
    bid: Math.max(0, mid - spread / 2),
    ask: Math.min(1, mid + spread / 2),
    spread,
    liquidity: 5000,
    volume: 20000,
  };
}

function mkEvent(o: {
  marketId: string;
  question?: string;
  sport?: "NBA" | "NHL";
  midPre?: number;
  midPostImmediate?: number;
  midPostLate?: number;
  spreadPostImmediate?: number;
}): EventSnapshot {
  return {
    marketId: o.marketId,
    question: o.question ?? "Will the Hurricanes win the 2026 NHL Stanley Cup?",
    sport: o.sport ?? "NHL",
    catalystEventStartUtc: NOW,
    snapshots: {
      preEvent15m: o.midPre !== undefined ? mkSnap(o.midPre) : undefined,
      postEvent15m:
        o.midPostImmediate !== undefined
          ? mkSnap(o.midPostImmediate, o.spreadPostImmediate ?? 0.02)
          : undefined,
      postEvent60m: o.midPostLate !== undefined ? mkSnap(o.midPostLate) : undefined,
    },
  };
}

function mkQualifiedMetric(realized: number, idx: number): ReversionMetric {
  return {
    marketId: `m${idx}`,
    question: "x",
    sport: "NBA",
    signalFired: true,
    invalidationReason: null,
    midPre: 0.30,
    midPostImmediate: 0.40,
    midPostLate: 0.40 - realized,
    move: 0.10,
    signalDir: "short",
    realizedReversion: realized,
  };
}

describe("tests/postEventReversionHypothesis.test.ts", () => {
  test("1. NBA Finals championship binary com mid saudável → eligible NBA", () => {
    const m = mkMarket({
      id: "NBA_OKC",
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
      prices: [0.30, 0.70],
    });
    const r = isHypothesisEligibleMarket(m, NOW);
    assertEqual(r.eligible, true, "deve aceitar");
    if (r.eligible) assertEqual(r.sport, "NBA", "sport NBA");
  });

  test("2. NHL Stanley Cup championship binary com mid saudável → eligible NHL", () => {
    const m = mkMarket({
      id: "NHL_HURR",
      question: "Will the Carolina Hurricanes win the 2026 NHL Stanley Cup?",
      prices: [0.40, 0.60],
    });
    const r = isHypothesisEligibleMarket(m, NOW);
    assertEqual(r.eligible, true, "deve aceitar");
    if (r.eligible) assertEqual(r.sport, "NHL", "sport NHL");
  });

  test("3. NFL Super Bowl → NÃO eligible (fora do escopo)", () => {
    const m = mkMarket({
      id: "NFL_EAG",
      question: "Will the Eagles win the 2027 Super Bowl?",
      prices: [0.20, 0.80],
    });
    const r = isHypothesisEligibleMarket(m, NOW);
    assertEqual(r.eligible, false, "Super Bowl fora do escopo");
  });

  test("4. NBA Finals mas mid em cauda inferior (0.02) → NÃO eligible", () => {
    const m = mkMarket({
      id: "NBA_LONG",
      question: "Will the Lakers win the 2026 NBA Finals?",
      prices: [0.02, 0.98],
    });
    const r = isHypothesisEligibleMarket(m, NOW);
    assertEqual(r.eligible, false, "mid em cauda bloqueia");
    if (!r.eligible) assertEqual(r.reason, "mid_in_tail", "razão tail");
  });

  test("5. NBA Finals mas market.closed → NÃO eligible", () => {
    const m = mkMarket({
      id: "NBA_CLOSED",
      question: "Will the Thunder win the 2026 NBA Finals?",
      closed: true,
      prices: [0.30, 0.70],
    });
    const r = isHypothesisEligibleMarket(m, NOW);
    assertEqual(r.eligible, false, "fechado bloqueia");
  });

  test("6. NBA Finals mas multi-outcome (3) → NÃO eligible", () => {
    const m = mkMarket({
      id: "NBA_MULTI",
      question: "Will the Thunder win the 2026 NBA Finals?",
      outcomes: ["YES", "NO", "MAYBE"],
      prices: [0.30, 0.50, 0.20],
    });
    const r = isHypothesisEligibleMarket(m, NOW);
    assertEqual(r.eligible, false, "não binário");
    if (!r.eligible) assertEqual(r.reason, "not_binary", "razão not_binary");
  });

  test("7. computeReversionMetric: move +10%, reverte para 0.35 → realized 0.05", () => {
    const e = mkEvent({
      marketId: "M1",
      sport: "NBA",
      midPre: 0.30,
      midPostImmediate: 0.40,
      midPostLate: 0.35,
    });
    const r = computeReversionMetric(e);
    assertEqual(r.signalFired, true, "sinal disparou");
    assertEqual(r.signalDir, "short", "move positivo → short");
    assertTrue((r.realizedReversion ?? 0) > 0, "reverteu favoravelmente");
    assertTrue(
      Math.abs((r.realizedReversion ?? 0) - 0.05) < 1e-9,
      "realizada = 0.05",
    );
  });

  test("8. computeReversionMetric: move −10%, reverte para 0.36 → realized 0.06", () => {
    const e = mkEvent({
      marketId: "M2",
      midPre: 0.40,
      midPostImmediate: 0.30,
      midPostLate: 0.36,
    });
    const r = computeReversionMetric(e);
    assertEqual(r.signalFired, true, "sinal disparou");
    assertEqual(r.signalDir, "long", "move negativo → long");
    assertTrue(
      Math.abs((r.realizedReversion ?? 0) - 0.06) < 1e-9,
      "realizada = 0.06",
    );
  });

  test("9. |move| < 3% → signal NOT fired, invalidation signal_below_threshold", () => {
    const e = mkEvent({
      marketId: "M3",
      midPre: 0.30,
      midPostImmediate: 0.31,
      midPostLate: 0.32,
    });
    const r = computeReversionMetric(e);
    assertEqual(r.signalFired, false, "sinal não dispara");
    assertEqual(r.invalidationReason, "signal_below_threshold", "razão");
  });

  test("10. midPostImmediate em cauda → invalidação", () => {
    const e = mkEvent({
      marketId: "M4",
      midPre: 0.10,
      midPostImmediate: 0.03,
      midPostLate: 0.05,
    });
    const r = computeReversionMetric(e);
    assertEqual(r.signalFired, false, "invalidado");
    assertEqual(
      r.invalidationReason,
      "post_event_15m_mid_in_tail",
      "razão tail",
    );
  });

  test("11. spread POST_EVENT_15M > 5% → invalidação", () => {
    const e = mkEvent({
      marketId: "M5",
      midPre: 0.30,
      midPostImmediate: 0.40,
      midPostLate: 0.35,
      spreadPostImmediate: 0.08,
    });
    const r = computeReversionMetric(e);
    assertEqual(r.signalFired, false, "invalidado");
    assertEqual(
      r.invalidationReason,
      "post_event_15m_spread_too_wide",
      "razão spread",
    );
  });

  test("12. snapshot ausente → invalidação por missing", () => {
    const e = mkEvent({ marketId: "M6", midPre: 0.30, midPostImmediate: 0.40 });
    const r = computeReversionMetric(e);
    assertEqual(r.signalFired, false, "invalidado");
    assertEqual(r.invalidationReason, "missing_post_event_60m", "razão missing");
  });

  test("13. judgeHypothesis: n < 50 qualified → alive_collecting", () => {
    const ms: ReversionMetric[] = [];
    for (let i = 0; i < 10; i++) ms.push(mkQualifiedMetric(0.01, i));
    const v = judgeHypothesis(ms);
    assertEqual(v.status, "alive_collecting", "ainda coletando");
    assertEqual(v.n, 10, "n=10");
    assertEqual(v.meanRealizedReversion, null, "mean ainda null");
  });

  test("14. judgeHypothesis: n=50, mean≥0.008, hit≥55%, sharpe alto → alive_surviving", () => {
    const ms: ReversionMetric[] = [];
    /** 50 trades com mean ≈ 0.012 (constante para sharpe alto). */
    for (let i = 0; i < 50; i++) ms.push(mkQualifiedMetric(0.012, i));
    const v = judgeHypothesis(ms);
    assertEqual(v.status, "alive_surviving", "sobrevive");
    assertTrue(
      (v.meanRealizedReversion ?? 0) > 0.008,
      "mean acima do target",
    );
    assertEqual(v.hitRate, 1.0, "hit 100%");
  });

  test("15. judgeHypothesis: n=50, mean=0.001 → dead (mean_below_death_threshold)", () => {
    const ms: ReversionMetric[] = [];
    for (let i = 0; i < 50; i++) ms.push(mkQualifiedMetric(0.001, i));
    const v = judgeHypothesis(ms);
    assertEqual(v.status, "dead", "morto");
    assertEqual(
      v.deathReason,
      "mean_below_death_threshold",
      "razão mean",
    );
  });

  test("16. judgeHypothesis: n=50, mean entre 0.005 e 0.008 + hit alto → needs_refinement", () => {
    const ms: ReversionMetric[] = [];
    /** mean = 0.0065 (entre death 0.005 e target 0.008), hit 100%, std=0 → sharpe Infinity */
    for (let i = 0; i < 50; i++) ms.push(mkQualifiedMetric(0.0065, i));
    const v = judgeHypothesis(ms);
    assertEqual(v.status, "needs_refinement", "precisa refino");
    assertTrue(
      (v.meanRealizedReversion ?? 0) >= 0.005,
      "acima da death",
    );
    assertTrue(
      (v.meanRealizedReversion ?? 0) < 0.008,
      "abaixo do target",
    );
    assertTrue(
      (v.refinementReason ?? "").includes("mean_below_target"),
      "razão menciona mean abaixo do target",
    );
  });

  test("17. judgeHypothesis: hit_rate < 48% → dead (hit_rate_below_random)", () => {
    const ms: ReversionMetric[] = [];
    /** 20 wins de +0.05 + 30 losses de -0.001 → mean = 0.0194 (passa death mean),
     *  mas hit 20/50 = 40% < 48% → death by hit rate. */
    for (let i = 0; i < 20; i++) ms.push(mkQualifiedMetric(0.05, i));
    for (let i = 0; i < 30; i++) ms.push(mkQualifiedMetric(-0.001, 100 + i));
    const v = judgeHypothesis(ms);
    assertEqual(v.status, "dead", "morto");
    assertEqual(v.deathReason, "hit_rate_below_random", "razão hit rate");
  });

  test("18. judgeHypothesis: drawdown_ratio > 5 → dead (drawdown_dominates_mean)", () => {
    const ms: ReversionMetric[] = [];
    /** 49 wins de 0.008 + 1 loss massiva de -0.50 → mean ainda positivo (≈ -0.0021),
     *  hmm vamos calibrar: 49*0.008 - 0.50 = -0.108; mean = -0.0022 → dead mean.
     *  Para isolar drawdown criterio: 49*0.02 + 1*(-0.20) = 0.78; mean = 0.0156;
     *  hit 49/50 = 98% (pass); std grande; drawdownAbs = 0.20; ratio = 0.20/0.0156 = 12.8 > 5 */
    for (let i = 0; i < 49; i++) ms.push(mkQualifiedMetric(0.02, i));
    ms.push(mkQualifiedMetric(-0.20, 999));
    const v = judgeHypothesis(ms);
    assertEqual(v.status, "dead", "morto");
    assertEqual(
      v.deathReason,
      "drawdown_dominates_mean",
      "razão drawdown",
    );
  });

  test("19. .paper não é criado no curso destes testes", () => {
    const candidates = [
      path.resolve(__dirname, "../..", ".paper"),
      path.resolve(__dirname, "..", ".paper"),
    ];
    for (const p of candidates) {
      assertEqual(fs.existsSync(p), false, `não criou .paper em ${p}`);
    }
  });

  test("20. lib não importa execução real / paper engine / dispatcher", () => {
    const libPath = path.resolve(__dirname, "../lib/postEventReversionHypothesis.ts");
    const src = fs.readFileSync(libPath, "utf8");
    /** Identificadores de módulos que ESTA lib NUNCA deve importar. */
    const forbiddenImports = [
      "shadowSimulationService",
      "shadowSimulationStore",
      "paperPortfolioStore",
      "paperTradeEngine",
      "executionDispatcher",
      "executionSimulator",
      "realisticExecutionEngine",
      "graphScanService",
      "graphArbitrageEngine",
      "probabilityScanner",
      "opportunityEngine",
      "universeQualityGate",
    ];
    for (const f of forbiddenImports) {
      /** Procura padrão `from "./xxx"` ou `from "../xxx"` — proíbe import literal. */
      const importPattern = new RegExp(`from\\s+["'][^"']*\\b${f}\\b["']`);
      assertTrue(!importPattern.test(src), `lib não importa ${f}`);
    }
  });
});
