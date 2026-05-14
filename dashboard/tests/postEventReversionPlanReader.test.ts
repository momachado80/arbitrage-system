import {
  collectDueTargets,
  buildDedupeKey,
  isDueWithinTolerance,
  TARGET_WINDOW_TYPES,
  type PlanFile,
} from "../lib/postEventReversionPlanReader";
import type { NormalizedMarket } from "../lib/polymarketClient";
import { describe, test, assertEqual, assertTrue } from "./_assert";

function mkMarket(
  o: Partial<NormalizedMarket> & { id: string; question: string },
): NormalizedMarket {
  return {
    id: o.id,
    question: o.question,
    slug: o.slug ?? o.id,
    category: o.category ?? "sports",
    outcomes: o.outcomes ?? ["YES", "NO"],
    prices: o.prices ?? [0.30, 0.70],
    liquidity: o.liquidity ?? 5000,
    volume: o.volume ?? 20000,
    active: o.active ?? true,
    closed: o.closed ?? false,
    spread: o.spread ?? 0.05,
    probSum: o.probSum ?? 1.0,
  };
}

const TOLERANCE_MIN = 8;

describe("tests/postEventReversionPlanReader.test.ts", () => {
  test("collectDueTargets lê nextEvent.eventStartUtc aninhado (schema atual do plano) e produz dueTarget", () => {
    const eventStart = "2026-05-12T02:30:00.000Z";
    /** POST_EVENT_15M = eventStart + 120m. */
    const postEvent15m = "2026-05-12T04:30:00.000Z";
    /** now 2min depois da janela — dentro de tolerância de 8min. */
    const now = new Date("2026-05-12T04:32:00.000Z");
    const nowIso = now.toISOString();
    const market = mkMarket({
      id: "NBA_OKC",
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
    });
    const plan: PlanFile = {
      plan: [
        {
          marketId: "NBA_OKC",
          catalystReadinessVerdict: "HAS_NEAR_CATALYST",
          nextEvent: { eventStartUtc: eventStart },
          observationWindows: [
            {
              windowType: "POST_EVENT_15M",
              runAtUtc: postEvent15m,
              reason: "post_event_approx_plus_120m",
            },
          ],
        },
      ],
    };
    const marketsById = new Map([[market.id, market]]);
    const targets = collectDueTargets(plan, marketsById, nowIso, now, TOLERANCE_MIN);
    assertEqual(targets.length, 1, "1 dueTarget esperado a partir do schema aninhado");
    assertEqual(targets[0]!.marketId, "NBA_OKC", "marketId correto");
    assertEqual(targets[0]!.windowType, "POST_EVENT_15M", "windowType POST_EVENT_15M");
    assertEqual(targets[0]!.runAtUtc, postEvent15m, "runAtUtc preservado");
    assertEqual(
      targets[0]!.catalystEventStartUtc,
      eventStart,
      "catalystEventStartUtc resolvido via nextEvent.eventStartUtc aninhado",
    );
    assertEqual(targets[0]!.sport, "NBA", "sport NBA");
  });

  test("collectDueTargets aceita fallback top-level row.nextEventStartUtc (compat defensiva)", () => {
    const eventStart = "2026-05-12T02:30:00.000Z";
    const postEvent60m = "2026-05-12T06:30:00.000Z";
    /** Within tolerance (3 min after). */
    const now = new Date("2026-05-12T06:33:00.000Z");
    const nowIso = now.toISOString();
    const market = mkMarket({
      id: "NHL_AVA",
      question: "Will the Colorado Avalanche win the 2026 NHL Stanley Cup?",
    });
    const plan: PlanFile = {
      plan: [
        {
          marketId: "NHL_AVA",
          catalystReadinessVerdict: "HAS_NEAR_CATALYST",
          /** Schema legado: top-level, sem nested nextEvent. */
          nextEventStartUtc: eventStart,
          observationWindows: [
            {
              windowType: "POST_EVENT_60M",
              runAtUtc: postEvent60m,
              reason: "post_event_late_plus_240m",
            },
          ],
        },
      ],
    };
    const marketsById = new Map([[market.id, market]]);
    const targets = collectDueTargets(plan, marketsById, nowIso, now, TOLERANCE_MIN);
    assertEqual(targets.length, 1, "fallback top-level deve funcionar");
    assertEqual(
      targets[0]!.catalystEventStartUtc,
      eventStart,
      "catalystEventStartUtc resolvido via fallback",
    );
    assertEqual(targets[0]!.sport, "NHL", "sport NHL");
  });

  test("collectDueTargets prefere nextEvent.eventStartUtc aninhado quando ambos presentes", () => {
    const eventStartNested = "2026-05-12T02:30:00.000Z";
    const eventStartFlatStale = "2026-05-10T00:00:00.000Z";
    const postEvent15m = "2026-05-12T04:30:00.000Z";
    const now = new Date("2026-05-12T04:30:00.000Z");
    const nowIso = now.toISOString();
    const market = mkMarket({
      id: "NBA_OKC",
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
    });
    const plan: PlanFile = {
      plan: [
        {
          marketId: "NBA_OKC",
          catalystReadinessVerdict: "HAS_NEAR_CATALYST",
          nextEvent: { eventStartUtc: eventStartNested },
          nextEventStartUtc: eventStartFlatStale,
          observationWindows: [
            { windowType: "POST_EVENT_15M", runAtUtc: postEvent15m, reason: "x" },
          ],
        },
      ],
    };
    const marketsById = new Map([[market.id, market]]);
    const targets = collectDueTargets(plan, marketsById, nowIso, now, TOLERANCE_MIN);
    assertEqual(targets.length, 1, "1 alvo");
    assertEqual(
      targets[0]!.catalystEventStartUtc,
      eventStartNested,
      "aninhado tem precedência sobre fallback",
    );
  });

  test("collectDueTargets pula linha sem catalystStart (nem nested nem fallback)", () => {
    const postEvent15m = "2026-05-12T04:30:00.000Z";
    const now = new Date("2026-05-12T04:32:00.000Z");
    const nowIso = now.toISOString();
    const market = mkMarket({
      id: "NBA_OKC",
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
    });
    const plan: PlanFile = {
      plan: [
        {
          marketId: "NBA_OKC",
          catalystReadinessVerdict: "HAS_NEAR_CATALYST",
          /** Sem nextEvent.eventStartUtc e sem nextEventStartUtc. */
          observationWindows: [
            { windowType: "POST_EVENT_15M", runAtUtc: postEvent15m, reason: "x" },
          ],
        },
      ],
    };
    const marketsById = new Map([[market.id, market]]);
    const targets = collectDueTargets(plan, marketsById, nowIso, now, TOLERANCE_MIN);
    assertEqual(targets.length, 0, "sem catalystStart, linha é pulada");
  });

  test("collectDueTargets pula janela fora da tolerância", () => {
    const eventStart = "2026-05-12T02:30:00.000Z";
    const postEvent15m = "2026-05-12T04:30:00.000Z";
    /** 20 min depois da janela — fora de tolerância de 8min. */
    const now = new Date("2026-05-12T04:50:00.000Z");
    const nowIso = now.toISOString();
    const market = mkMarket({
      id: "NBA_OKC",
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
    });
    const plan: PlanFile = {
      plan: [
        {
          marketId: "NBA_OKC",
          catalystReadinessVerdict: "HAS_NEAR_CATALYST",
          nextEvent: { eventStartUtc: eventStart },
          observationWindows: [
            { windowType: "POST_EVENT_15M", runAtUtc: postEvent15m, reason: "x" },
          ],
        },
      ],
    };
    const marketsById = new Map([[market.id, market]]);
    const targets = collectDueTargets(plan, marketsById, nowIso, now, TOLERANCE_MIN);
    assertEqual(targets.length, 0, "janela fora de tolerância é pulada");
  });

  test("TARGET_WINDOW_TYPES inclui PRE_EVENT_15M, POST_EVENT_15M e POST_EVENT_60M", () => {
    assertTrue(TARGET_WINDOW_TYPES.has("PRE_EVENT_15M"), "PRE_EVENT_15M");
    assertTrue(TARGET_WINDOW_TYPES.has("POST_EVENT_15M"), "POST_EVENT_15M");
    assertTrue(TARGET_WINDOW_TYPES.has("POST_EVENT_60M"), "POST_EVENT_60M");
    /** Janelas explicitamente fora do alvo. */
    assertEqual(TARGET_WINDOW_TYPES.has("PRE_EVENT_60M"), false, "PRE_EVENT_60M fora");
    assertEqual(TARGET_WINDOW_TYPES.has("EVENT_START"), false, "EVENT_START fora");
    assertEqual(TARGET_WINDOW_TYPES.has("MID_EVENT_ESTIMATE"), false, "MID fora");
  });

  test("isDueWithinTolerance respeita janela simétrica antes/depois", () => {
    const ts = "2026-05-12T04:30:00.000Z";
    const before = new Date("2026-05-12T04:25:00.000Z");
    const after = new Date("2026-05-12T04:35:00.000Z");
    const farBefore = new Date("2026-05-12T04:00:00.000Z");
    const farAfter = new Date("2026-05-12T05:00:00.000Z");
    assertEqual(isDueWithinTolerance(ts, before, 8), true, "5min antes dentro");
    assertEqual(isDueWithinTolerance(ts, after, 8), true, "5min depois dentro");
    assertEqual(isDueWithinTolerance(ts, farBefore, 8), false, "30min antes fora");
    assertEqual(isDueWithinTolerance(ts, farAfter, 8), false, "30min depois fora");
  });

  test("buildDedupeKey é determinístico e inclui versão da hipótese", () => {
    const k = buildDedupeKey("NBA_OKC", "POST_EVENT_15M", "2026-05-12T04:30:00.000Z");
    assertEqual(
      k,
      "NBA_OKC|POST_EVENT_15M|2026-05-12T04:30:00.000Z|post_event_reversion_v1",
      "chave dedup determinística",
    );
  });
});
