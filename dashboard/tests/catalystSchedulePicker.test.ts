import {
  POST_EVENT_RETENTION_MS,
  pickEarliestEspnGameWithinRetention,
  type NextGamePick,
} from "../lib/catalystSchedulePicker";
import {
  buildObservationWindowsForCatalyst,
  evaluateCatalystObservationReadiness,
  inferCatalystProfileFromMarket,
} from "../lib/catalystObservationSchedule";
import { describe, test, assertEqual, assertTrue } from "./_assert";

const MIN = 60_000;
const HOUR = 60 * MIN;

function mkPick(eventStartUtc: string, name = "Thunder vs Lakers"): NextGamePick {
  return { eventName: name, eventStartUtc, opponentShort: "Lakers" };
}

describe("tests/catalystSchedulePicker.test.ts", () => {
  test("POST_EVENT_RETENTION_MS cobre POST_EVENT_60M (≥ 4h) com folga para tolerância do scout", () => {
    assertTrue(POST_EVENT_RETENTION_MS >= 4 * HOUR, "retenção ≥ 4h (cobre +240m POST_EVENT_60M)");
    assertEqual(POST_EVENT_RETENTION_MS, 5 * HOUR, "valor explícito = 5h");
  });

  test("picker retém jogo iniciado há 30min (dentro de retenção padrão)", () => {
    const now = new Date("2026-05-12T03:00:00.000Z");
    const horizonEnd = new Date(now.getTime() + 72 * HOUR);
    const start30mAgo = new Date(now.getTime() - 30 * MIN).toISOString();
    const candidates = [mkPick(start30mAgo, "OKC vs LAL")];
    const result = pickEarliestEspnGameWithinRetention(candidates, now, horizonEnd);
    assertTrue(result !== null, "jogo de 30min atrás deve ser retido");
    assertEqual(result!.eventStartUtc, start30mAgo, "preserva eventStartUtc");
  });

  test("picker descarta jogo iniciado há 6h (fora da retenção padrão de 5h)", () => {
    const now = new Date("2026-05-12T08:00:00.000Z");
    const horizonEnd = new Date(now.getTime() + 72 * HOUR);
    const start6hAgo = new Date(now.getTime() - 6 * HOUR).toISOString();
    const candidates = [mkPick(start6hAgo)];
    const result = pickEarliestEspnGameWithinRetention(candidates, now, horizonEnd);
    assertEqual(result, null, "jogo de 6h atrás deve ser descartado");
  });

  test("picker preserva o jogo mais cedo elegível entre múltiplos candidatos (passado + futuro)", () => {
    const now = new Date("2026-05-12T03:00:00.000Z");
    const horizonEnd = new Date(now.getTime() + 72 * HOUR);
    const pastInRetention = new Date(now.getTime() - 30 * MIN).toISOString();
    const futureSoon = new Date(now.getTime() + 2 * HOUR).toISOString();
    const futureLater = new Date(now.getTime() + 24 * HOUR).toISOString();
    const candidates = [mkPick(futureLater), mkPick(pastInRetention), mkPick(futureSoon)];
    const result = pickEarliestEspnGameWithinRetention(candidates, now, horizonEnd);
    assertTrue(result !== null, "deve escolher algum");
    assertEqual(
      result!.eventStartUtc,
      pastInRetention,
      "o mais cedo elegível (jogo recente em retenção) tem precedência",
    );
  });

  test("picker descarta jogo além de horizonEnd", () => {
    const now = new Date("2026-05-12T03:00:00.000Z");
    const horizonEnd = new Date(now.getTime() + 24 * HOUR);
    const beyondHorizon = new Date(horizonEnd.getTime() + HOUR).toISOString();
    const candidates = [mkPick(beyondHorizon)];
    const result = pickEarliestEspnGameWithinRetention(candidates, now, horizonEnd);
    assertEqual(result, null, "jogo além de horizonEnd é descartado");
  });

  test("picker retorna null para lista vazia ou só candidatos inválidos", () => {
    const now = new Date("2026-05-12T03:00:00.000Z");
    const horizonEnd = new Date(now.getTime() + 24 * HOUR);
    assertEqual(pickEarliestEspnGameWithinRetention([], now, horizonEnd), null, "lista vazia");
    const invalid = [mkPick("nao-iso-valido")];
    assertEqual(
      pickEarliestEspnGameWithinRetention(invalid, now, horizonEnd),
      null,
      "iso inválido descartado",
    );
  });

  test("picker permite override de retentionMs para zero (comportamento antigo, só futuros)", () => {
    const now = new Date("2026-05-12T03:00:00.000Z");
    const horizonEnd = new Date(now.getTime() + 24 * HOUR);
    const start30mAgo = new Date(now.getTime() - 30 * MIN).toISOString();
    const candidates = [mkPick(start30mAgo)];
    const result = pickEarliestEspnGameWithinRetention(candidates, now, horizonEnd, 0);
    assertEqual(result, null, "com retentionMs=0, jogo passado é descartado (comportamento legado)");
  });

  test("integração: jogo iniciado há 30min mantém janelas POST_EVENT_15M e POST_EVENT_60M futuras no plano", () => {
    /** Cenário do bug original: jogo Thunder vs Lakers começou há 30min.
     *  Com fix #2 retendo o jogo, a readiness fica HAS_NEAR_CATALYST e
     *  as janelas POST_EVENT_15M (+120m → +90m futuro) e POST_EVENT_60M
     *  (+240m → +210m futuro) ficam no plano para o scout coletar. */
    const now = new Date("2026-05-12T03:00:00.000Z");
    const horizonEnd = new Date(now.getTime() + 72 * HOUR);
    const start30mAgo = new Date(now.getTime() - 30 * MIN).toISOString();

    const picked = pickEarliestEspnGameWithinRetention([mkPick(start30mAgo)], now, horizonEnd);
    assertTrue(picked !== null, "picker retém o jogo");

    const profile = inferCatalystProfileFromMarket({
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
    });
    const readiness = evaluateCatalystObservationReadiness({
      nowIso: now.toISOString(),
      horizonHours: 72,
      profile,
      nextEventStartUtc: picked!.eventStartUtc,
      scheduleFetchStatus: "ok",
    });
    assertEqual(
      readiness.catalystReadinessVerdict,
      "HAS_NEAR_CATALYST",
      "readiness é HAS_NEAR_CATALYST mesmo com jogo já iniciado",
    );
    assertTrue(
      readiness.nextObservationWindowUtc !== null,
      "há próxima janela futura",
    );

    const windows = buildObservationWindowsForCatalyst({
      marketId: "NBA_OKC",
      label: null,
      catalystType: profile.catalystType,
      eventName: picked!.eventName,
      eventStartUtc: picked!.eventStartUtc,
      source: "espn_public_scoreboard",
      confidence: "medium",
    });
    const byType = new Map(windows.map(w => [w.windowType, w.runAtUtc]));

    const postE15 = byType.get("POST_EVENT_15M");
    const postE60 = byType.get("POST_EVENT_60M");
    assertTrue(postE15 !== undefined, "plano contém POST_EVENT_15M");
    assertTrue(postE60 !== undefined, "plano contém POST_EVENT_60M");
    assertTrue(
      new Date(postE15!).getTime() > now.getTime(),
      "POST_EVENT_15M ainda está no futuro (vs jogo iniciado há 30min)",
    );
    assertTrue(
      new Date(postE60!).getTime() > now.getTime(),
      "POST_EVENT_60M ainda está no futuro",
    );
  });
});
