import fs from "fs";
import os from "os";
import path from "path";

import {
  getPersistencePath,
  persistClosedTrades,
} from "../lib/shadowClosedTradePersistence";
import type { ShadowTrade } from "../lib/shadowSimulationStore";
import { describe, test, assertEqual, assertTrue } from "./_assert";

/**
 * Helpers de cleanup tolerantes — se o tempdir já não existe, ignorar.
 */
function safeRm(dir: string): void {
  try {
    fs.rmSync(dir, { recursive: true, force: true });
  } catch {
    /* noop */
  }
}

function withEnvSnapshot<T>(
  keys: ReadonlyArray<string>,
  fn: () => T,
): T {
  const snapshot: Record<string, string | undefined> = {};
  for (const k of keys) snapshot[k] = process.env[k];
  try {
    return fn();
  } finally {
    for (const k of keys) {
      const original = snapshot[k];
      if (original === undefined) delete process.env[k];
      else process.env[k] = original;
    }
  }
}

describe("tests/shadowClosedTradePersistence.test.ts", () => {
  test("getPersistencePath usa SHADOW_PERSISTENCE_PATH como base + nome fixo do arquivo", () => {
    withEnvSnapshot(["SHADOW_PERSISTENCE_PATH"], () => {
      process.env.SHADOW_PERSISTENCE_PATH = "/tmp/__shadow_persist_test_a";
      const p = getPersistencePath();
      assertTrue(
        p.startsWith("/tmp/__shadow_persist_test_a/"),
        "base = SHADOW_PERSISTENCE_PATH",
      );
      assertTrue(
        p.endsWith("/shadow-closed-trades.json"),
        "nome fixo do arquivo final",
      );
    });
  });

  test("getPersistencePath cai para DATA_PATH quando SHADOW_PERSISTENCE_PATH ausente", () => {
    withEnvSnapshot(["SHADOW_PERSISTENCE_PATH", "DATA_PATH"], () => {
      delete process.env.SHADOW_PERSISTENCE_PATH;
      process.env.DATA_PATH = "/tmp/__data_path_fallback_test";
      const p = getPersistencePath();
      assertTrue(
        p.startsWith("/tmp/__data_path_fallback_test/"),
        "fallback para DATA_PATH",
      );
    });
  });

  test("getPersistencePath ignora PAPER_STATE_DIR (não é env do bot)", () => {
    withEnvSnapshot(
      ["SHADOW_PERSISTENCE_PATH", "DATA_PATH", "PAPER_STATE_DIR"],
      () => {
        delete process.env.SHADOW_PERSISTENCE_PATH;
        delete process.env.DATA_PATH;
        process.env.PAPER_STATE_DIR = "/tmp/__paper_state_dir_must_be_ignored";
        const p = getPersistencePath();
        assertTrue(
          !p.includes("__paper_state_dir_must_be_ignored"),
          "PAPER_STATE_DIR não deve aparecer no caminho do bot",
        );
        assertTrue(
          p.endsWith("/shadow-closed-trades.json"),
          "nome do arquivo permanece estável",
        );
      },
    );
  });

  test(
    "persistClosedTrades(force=true) escreve apenas closedTrades em SHADOW_PERSISTENCE_PATH; activeTrades filtrados; PAPER_STATE_DIR intocado",
    () => {
      const tmpdirShadow = fs.mkdtempSync(
        path.join(os.tmpdir(), "shadow-persist-test-"),
      );
      const tmpdirIgnored = fs.mkdtempSync(
        path.join(os.tmpdir(), "paper-state-ignored-"),
      );
      try {
        withEnvSnapshot(
          ["SHADOW_PERSISTENCE_PATH", "DATA_PATH", "PAPER_STATE_DIR"],
          () => {
            process.env.SHADOW_PERSISTENCE_PATH = tmpdirShadow;
            delete process.env.DATA_PATH;
            process.env.PAPER_STATE_DIR = tmpdirIgnored;

            const closedTrade = {
              tradeId: "closed-A",
              status: "closed",
              closedAt: "2026-05-09T20:00:00.000Z",
            } as unknown as ShadowTrade;
            const activeTrade = {
              tradeId: "active-B",
              status: "active",
              closedAt: null,
            } as unknown as ShadowTrade;
            const closedNoTimestamp = {
              tradeId: "closed-but-missing-timestamp",
              status: "closed",
              closedAt: null,
            } as unknown as ShadowTrade;

            persistClosedTrades(
              {
                profile_under_test: [closedTrade, activeTrade, closedNoTimestamp],
              },
              { force: true },
            );

            const expectedFile = path.join(tmpdirShadow, "shadow-closed-trades.json");
            assertTrue(
              fs.existsSync(expectedFile),
              "shadow-closed-trades.json criado em SHADOW_PERSISTENCE_PATH",
            );

            const raw = fs.readFileSync(expectedFile, "utf-8");
            const parsed = JSON.parse(raw) as {
              schemaVersion: string;
              savedAt: string;
              byProfile: Record<string, Array<Record<string, unknown>>>;
            };
            assertEqual(parsed.schemaVersion, "1", "schemaVersion estável");
            assertTrue(
              typeof parsed.savedAt === "string" && parsed.savedAt.length > 0,
              "savedAt presente",
            );
            assertTrue(
              Array.isArray(parsed.byProfile.profile_under_test),
              "profile gravado",
            );
            assertEqual(
              parsed.byProfile.profile_under_test.length,
              1,
              "só 1 trade — active e closed-sem-timestamp são filtrados",
            );
            assertEqual(
              parsed.byProfile.profile_under_test[0]!.tradeId,
              "closed-A",
              "tradeId correto persistido",
            );
            assertEqual(
              parsed.byProfile.profile_under_test[0]!.status,
              "closed",
              "status closed preservado",
            );

            /** PAPER_STATE_DIR não deve receber nada — bot não usa essa env var. */
            const ignoredFile = path.join(tmpdirIgnored, "shadow-closed-trades.json");
            assertTrue(
              !fs.existsSync(ignoredFile),
              "shadow-closed-trades.json NÃO criado em PAPER_STATE_DIR",
            );
            const ignoredEntries = fs.readdirSync(tmpdirIgnored);
            assertEqual(
              ignoredEntries.length,
              0,
              "PAPER_STATE_DIR continua vazio",
            );
          },
        );
      } finally {
        safeRm(tmpdirShadow);
        safeRm(tmpdirIgnored);
      }
    },
  );
});
