import fs from "fs";
import path from "path";

import {
  buildObservationWindowsForCatalyst,
  evaluateCatalystObservationReadiness,
  inferCatalystProfileFromMarket,
} from "../lib/catalystObservationSchedule";
import { describe, test, assertEqual, assertTrue } from "./_assert";

describe("tests/catalystObservationSchedule.test.ts", () => {
  test("OKC Thunder Finals vira NBA_TEAM_FUTURE", () => {
    const p = inferCatalystProfileFromMarket({
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
      slug: "will-the-oklahoma-city-thunder-win-the-2026-nba-finals",
    });
    assertEqual(p.catalystType, "NBA_TEAM_FUTURE", "tipo");
    assertEqual(p.league, "NBA", "league");
    assertEqual(p.subject.toLowerCase().includes("oklahoma"), true, "subject OKC");
    assertEqual(p.catalystPriority, "HIGH", "prioridade");
  });

  test("Colorado Avalanche vira NHL_TEAM_FUTURE", () => {
    const p = inferCatalystProfileFromMarket({
      question: "Will the Colorado Avalanche win the 2026 NHL Stanley Cup?",
    });
    assertEqual(p.catalystType, "NHL_TEAM_FUTURE", "tipo");
    assertEqual(p.league, "NHL", "league");
    assertTrue(p.subject.includes("Colorado"), "subject");
    assertEqual(p.catalystPriority, "HIGH", "prioridade");
  });

  test("Spain World Cup vira FIFA_WORLD_CUP_FUTURE", () => {
    const p = inferCatalystProfileFromMarket({
      question: "Will Spain win the 2026 FIFA World Cup?",
    });
    assertEqual(p.catalystType, "FIFA_WORLD_CUP_FUTURE", "tipo");
    assertEqual(p.league, "FIFA", "league");
    assertEqual(p.subject, "Spain", "subject");
    assertEqual(p.catalystPriority, "MEDIUM", "prioridade média");
  });

  test("Bitcoin/GTA VI vira UNKNOWN_OR_LOW_SIGNAL", () => {
    const p = inferCatalystProfileFromMarket({
      question: "Will bitcoin hit $1m before GTA VI?",
    });
    assertEqual(p.catalystType, "UNKNOWN_OR_LOW_SIGNAL", "tipo baixo sinal");
    assertEqual(p.league, "UNKNOWN", "league");
    assertEqual(p.catalystPriority, "LOW", "prioridade");
  });

  test("buildObservationWindowsForCatalyst gera T-60, T-15, T+0, T+45, T+120, T+240", () => {
    const start = "2026-07-01T18:00:00.000Z";
    const wins = buildObservationWindowsForCatalyst({
      marketId: "553856",
      label: "OKC",
      catalystType: "NBA_TEAM_FUTURE",
      eventName: "Finals Game",
      eventStartUtc: start,
      source: "test",
      confidence: "high",
    });
    assertEqual(wins.length, 6, "seis janelas");
    assertEqual(wins[0]!.windowType, "PRE_EVENT_60M", "tipo");
    assertEqual(wins[0]!.runAtUtc, "2026-07-01T17:00:00.000Z", "t-60");
    assertEqual(wins[1]!.runAtUtc, "2026-07-01T17:45:00.000Z", "t-15");
    assertEqual(wins[2]!.runAtUtc, start, "t0");
    assertEqual(wins[3]!.runAtUtc, "2026-07-01T18:45:00.000Z", "t+45");
    assertEqual(wins[4]!.runAtUtc, "2026-07-01T20:00:00.000Z", "t+120");
    assertEqual(wins[5]!.runAtUtc, "2026-07-01T22:00:00.000Z", "t+240");
  });

  test("NO_NEAR_CATALYST quando FIFA sem fixture próximo", () => {
    const profile = inferCatalystProfileFromMarket({ question: "Will Spain win the 2026 FIFA World Cup?" });
    const r = evaluateCatalystObservationReadiness({
      nowIso: "2026-05-06T12:00:00.000Z",
      horizonHours: 72,
      profile,
      nextEventStartUtc: null,
      scheduleFetchStatus: "not_applicable",
    });
    assertEqual(r.catalystReadinessVerdict, "NO_NEAR_CATALYST", "sem catalisador FIFA automático");
    assertEqual(r.nextObservationWindowUtc, null, "sem próxima janela");
  });

  test("NO_NEAR_CATALYST quando NBA sem jogo resolvido dentro do horizonte", () => {
    const profile = inferCatalystProfileFromMarket({
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
    });
    const r = evaluateCatalystObservationReadiness({
      nowIso: "2026-05-06T12:00:00.000Z",
      horizonHours: 6,
      profile,
      nextEventStartUtc: null,
      scheduleFetchStatus: "ok",
    });
    assertEqual(r.catalystReadinessVerdict, "NO_NEAR_CATALYST", "sem jogo");
  });

  test("NEEDS_SCHEDULE_SOURCE quando schedule falhou para NBA", () => {
    const profile = inferCatalystProfileFromMarket({
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
    });
    const r = evaluateCatalystObservationReadiness({
      nowIso: "2026-05-06T12:00:00.000Z",
      horizonHours: 72,
      profile,
      nextEventStartUtc: null,
      scheduleFetchStatus: "failed",
    });
    assertEqual(r.catalystReadinessVerdict, "NEEDS_SCHEDULE_SOURCE", "precisa fonte");
  });

  test("HAS_NEAR_CATALYST quando há evento futuro dentro do horizonte", () => {
    const profile = inferCatalystProfileFromMarket({
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
    });
    const r = evaluateCatalystObservationReadiness({
      nowIso: "2026-05-06T12:00:00.000Z",
      horizonHours: 200,
      profile,
      nextEventStartUtc: "2026-05-07T00:30:00.000Z",
      scheduleFetchStatus: "ok",
    });
    assertEqual(r.catalystReadinessVerdict, "HAS_NEAR_CATALYST", "há catalisador");
    assertTrue(r.nextObservationWindowUtc !== null, "próxima janela definida");
  });

  test("canUseForMicrocapitalCandidate não existe na lib pura e perfil não sugere micro", () => {
    const p = inferCatalystProfileFromMarket({
      question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
    });
    assertEqual(
      Object.prototype.hasOwnProperty.call(p, "canUseForMicrocapitalCandidate"),
      false,
      "sem campo micro na lib pura",
    );
  });

  test("lib sem termos reservados de execução (regex)", () => {
    const libPath = path.join(__dirname, "../lib/catalystObservationSchedule.ts");
    const testPath = path.join(__dirname, "catalystObservationSchedule.test.ts");
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
