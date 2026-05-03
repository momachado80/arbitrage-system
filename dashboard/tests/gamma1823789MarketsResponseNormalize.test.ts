import {
  GammaPayloadUnusableShapeError,
  normalizeGamma1823789MarketRow,
} from "../lib/gamma1823789MarketsResponseNormalize";
import { describe, test, assertEqual, assertIncludes, assertTrue, assertThrows } from "./_assert";

describe("tests/gamma1823789MarketsResponseNormalize.test.ts", () => {
  const mk = (): Record<string, unknown> => ({
    id: "1823789",
    question: "q",
    clobTokenIds: ["tid-yes"],
    outcomes: ["Yes", "No"],
  });

  test("aceita array não vazio", () => {
    const row = normalizeGamma1823789MarketRow([mk()]);
    assertEqual(row.id, "1823789", "id");
  });

  test("aceita objeto único", () => {
    const row = normalizeGamma1823789MarketRow(mk());
    assertEqual(row.clobTokenIds, ["tid-yes"], "token");
  });

  test("aceita data array", () => {
    const row = normalizeGamma1823789MarketRow({ data: [mk()] });
    assertEqual(row.id, "1823789", "id");
  });

  test("aceita data object", () => {
    const row = normalizeGamma1823789MarketRow({ data: mk() });
    assertEqual(row.question, "q", "question");
  });

  test("aceita markets array", () => {
    const row = normalizeGamma1823789MarketRow({ markets: [mk()] });
    assertEqual(row.id, "1823789", "id");
  });

  test("aceita results array", () => {
    const row = normalizeGamma1823789MarketRow({ results: [mk()] });
    assertEqual(row.id, "1823789", "id");
  });

  test("aceita market object", () => {
    const row = normalizeGamma1823789MarketRow({ market: mk() });
    assertEqual(row.id, "1823789", "id");
  });

  test("aceita markets objeto único", () => {
    const row = normalizeGamma1823789MarketRow({ markets: mk() });
    assertEqual(row.id, "1823789", "markets object");
  });

  test("aceita results objeto único", () => {
    const row = normalizeGamma1823789MarketRow({ results: mk() });
    assertEqual(row.question, "q", "results object");
  });

  test("rejeita array vazio com diagnóstico seguro", () => {
    try {
      normalizeGamma1823789MarketRow([]);
      throw new Error("expected throw");
    } catch (e) {
      assertTrue(e instanceof GammaPayloadUnusableShapeError, "type");
      const d = (e as GammaPayloadUnusableShapeError).diagnostic;
      assertEqual(d.rootKind, "array", "rootKind");
      assertEqual(d.arrayLength, 0, "length");
    }
  });

  test("rejeita null/string com diagnóstico seguro", () => {
    try {
      normalizeGamma1823789MarketRow(null);
      throw new Error("expected throw");
    } catch (e) {
      assertTrue(e instanceof GammaPayloadUnusableShapeError, "null type");
      assertEqual((e as GammaPayloadUnusableShapeError).diagnostic.rootKind, "null", "null root");
    }
    try {
      normalizeGamma1823789MarketRow("nope");
      throw new Error("expected throw string");
    } catch (e) {
      assertTrue(e instanceof GammaPayloadUnusableShapeError, "string type");
      assertEqual((e as GammaPayloadUnusableShapeError).diagnostic.rootKind, "string", "string root");
    }
  });

  test("não registra payload bruto completo", () => {
    const longNoise = "Z".repeat(5000);
    const invalid = { noise: longNoise, other: 1 };
    try {
      normalizeGamma1823789MarketRow(invalid);
      throw new Error("expected throw");
    } catch (e) {
      assertTrue(e instanceof GammaPayloadUnusableShapeError, "GammaPayloadUnusableShapeError");
      const msg = (e as Error).message;
      assertIncludes(msg, "gamma_payload_unusable_shape", "code prefix");
      assertTrue(msg.length < 800, `diagnostic message should stay compact (got ${msg.length})`);
      assertTrue(!msg.includes(longNoise), "must not echo large field values from payload");
    }
  });
});
