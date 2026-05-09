import fs from "fs";
import os from "os";
import path from "path";

import {
  assertSafeOrchestratorPath,
  buildDueObservationCliArgs,
  buildWebhookPayload,
  collectDedupeKeysFromLedger,
  computeDedupeKeysEmitted,
  notifyFreshDueWindows,
  parseOrchestratorEnv,
  partitionFreshDueWindows,
  buildWindowKey,
} from "../lib/catalystObservationOrchestratorHelpers";
import { describe, test, assertEqual, assertTrue, assertThrows } from "./_assert";

describe("tests/catalystObservationOrchestrator.test.ts", () => {
  test("ledger/plan paths bloqueiam .paper", () => {
    assertThrows(
      () => assertSafeOrchestratorPath("/tmp/foo/.paper/ledger.jsonl", "ledger"),
      "bloqueia .paper",
    );
    assertThrows(
      () => assertSafeOrchestratorPath("/tmp/.paper", "ledger"),
      "bloqueia .paper direto",
    );
  });

  test("buildWindowKey estável para dedupe", () => {
    const k = buildWindowKey({
      marketId: "553856",
      windowType: "PRE_EVENT_60M",
      runAtUtc: "2026-05-08T00:30:00.000Z",
    });
    assertEqual(k.includes("553856"), true, "contém marketId");
    assertEqual(k.endsWith("catalyst_scout_dry"), true, "scoutType suffix");
  });

  test("dedupe remove windowKey já vista no ledger", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "cat-orch-test-"));
    const ledger = path.join(dir, "ledger.jsonl");
    const w = {
      marketId: "553856",
      label: "OKC",
      windowType: "PRE_EVENT_60M",
      runAtUtc: "2026-05-08T00:30:00.000Z",
    };
    const key = buildWindowKey(w);
    fs.writeFileSync(
      ledger,
      `${JSON.stringify({
        dedupeKeysEmitted: [key],
      })}\n`,
      "utf8",
    );
    const seen = collectDedupeKeysFromLedger(ledger);
    const { freshKeys } = partitionFreshDueWindows([w], seen);
    assertEqual(freshKeys.length, 0, "não deve repetir mesma janela");
  });

  test("CLI due sempre inclui --dry-run true e --scout-readonly true", () => {
    const args = buildDueObservationCliArgs({
      planPath: "/tmp/plan.json",
      duePath: "/tmp/due.json",
      dueWithinMinutes: 15,
    });
    /** dry-run impede execução econômica; scout-readonly habilita leitura read-only do livro. */
    const dryRunIdx = args.indexOf("--dry-run");
    assertTrue(dryRunIdx >= 0 && args[dryRunIdx + 1] === "true", "--dry-run true forçado");
    const readonlyIdx = args.indexOf("--scout-readonly");
    assertTrue(
      readonlyIdx >= 0 && args[readonlyIdx + 1] === "true",
      "--scout-readonly true forçado",
    );
  });

  test("sem webhook: dedupe emite freshKeys mesmo sem POST", () => {
    const emitted = computeDedupeKeysEmitted({
      notifyWebhookUrl: null,
      notificationSent: false,
      freshKeys: ["a", "b"],
    });
    assertEqual(emitted.join(","), "a,b", "consome chaves sem webhook");
  });

  test("com webhook falho: não deduplica para permitir retry", () => {
    const emitted = computeDedupeKeysEmitted({
      notifyWebhookUrl: "https://example.invalid/hook",
      notificationSent: false,
      freshKeys: ["k1"],
    });
    assertEqual(emitted.length, 0, "retry posterior");
  });

  test("com webhook ok: dedupe emite chaves", () => {
    const emitted = computeDedupeKeysEmitted({
      notifyWebhookUrl: "https://example.invalid/hook",
      notificationSent: true,
      freshKeys: ["k1"],
    });
    assertEqual(emitted, ["k1"], "dedupe após sucesso");
  });

  test("parseOrchestratorEnv defaults coerentes", () => {
    const cfg = parseOrchestratorEnv({});
    assertEqual(cfg.horizonHours, 72, "horizon default");
    assertEqual(cfg.limit, 8, "limit default");
    assertEqual(cfg.dueWithinMinutes, 15, "due default");
    assertEqual(cfg.pollIntervalSeconds, 300, "poll default");
    assertEqual(cfg.dryRun, true, "dry env default true");
    assertEqual(cfg.automationEnabled, false, "automation off por omissão");
    assertEqual(cfg.planPath, "/tmp/catalyst-observation-plan.json", "plan default");
    assertEqual(cfg.duePath, "/tmp/due-observation-scouts.json", "due default");
  });

  test("payload webhook marca execution none e paper untouched", () => {
    const p = buildWebhookPayload({
      window: {
        marketId: "553856",
        label: "OKC",
        windowType: "PRE_EVENT_60M",
        runAtUtc: "2026-05-08T00:30:00.000Z",
      },
      utcNowIso: "2026-05-06T12:00:00.000Z",
      status: "test",
      recommendedNextState: "observe_only",
    });
    assertEqual(p.execution, "none", "sem execução real");
    assertEqual(p.paperState, "untouched", "paper intacto");
    assertTrue(typeof p.localTimeAmericaSaoPaulo === "string", "hora SP presente");
  });

  test("dueWindows vazio ⇒ partitionFreshDueWindows vazio", () => {
    const { freshKeys } = partitionFreshDueWindows([], new Set());
    assertEqual(freshKeys.length, 0, "sem fresh");
  });

  test("sem freshWindows ou sem URL ⇒ não chama fetch", async () => {
    let calls = 0;
    const fetchFn = async () => {
      calls++;
      return new Response(null, { status: 200 });
    };
    const r1 = await notifyFreshDueWindows({
      notifyWebhookUrl: null,
      freshWindows: [{ marketId: "1", windowType: "PRE_EVENT_60M", runAtUtc: "2026-01-01T00:00:00.000Z" }],
      utcNowIso: "2026-01-01T00:00:00.000Z",
      fetchFn,
    });
    assertEqual(r1.notificationSent, false, "sem URL");
    const r2 = await notifyFreshDueWindows({
      notifyWebhookUrl: "https://hooks.example.test/x",
      freshWindows: [],
      utcNowIso: "2026-01-01T00:00:00.000Z",
      fetchFn,
    });
    assertEqual(r2.notificationSent, false, "sem janelas");
    assertEqual(calls, 0, "fetch não invocado");
  });

  test("com URL e freshWindows ⇒ uma chamada fetch por janela e notificationSent true", async () => {
    let calls = 0;
    const fetchFn = async (_input: RequestInfo | URL, init?: RequestInit) => {
      calls++;
      assertTrue(init?.method === "POST", "POST");
      assertTrue(typeof init?.body === "string", "corpo JSON");
      const j = JSON.parse(init!.body as string) as Record<string, unknown>;
      assertEqual(j.execution, "none", "execution none");
      assertEqual(j.paperState, "untouched", "paper untouched");
      return new Response(null, { status: 200 });
    };
    const r = await notifyFreshDueWindows({
      notifyWebhookUrl: "https://hooks.example.test/x",
      freshWindows: [
        { marketId: "a", windowType: "PRE_EVENT_60M", runAtUtc: "2026-05-08T00:30:00.000Z" },
        { marketId: "b", windowType: "PRE_EVENT_15M", runAtUtc: "2026-05-08T01:00:00.000Z" },
      ],
      utcNowIso: "2026-05-06T12:00:00.000Z",
      fetchFn: fetchFn as typeof fetch,
    });
    assertEqual(calls, 2, "duas janelas");
    assertEqual(r.notificationSent, true, "sucesso acumulado");
  });
});
