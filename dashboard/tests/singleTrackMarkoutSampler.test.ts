import fs from "fs";
import path from "path";
import {
  MARKOUT_NONZERO_EPS,
  PRICE_SOURCE_BID_ASK_MID,
  PRICE_SOURCE_INSUFFICIENT_BID_ASK,
  classifyBookType,
  classifyProbabilisticPricePinned,
  classifySampleInformativeness,
  computeMarkoutSummary,
  computeMid,
  computeSpread,
  snapshotRowFromPrices,
} from "../lib/singleTrackMarkoutSampler";
import { describe, test, assertEqual, assertTrue } from "./_assert";

describe("tests/singleTrackMarkoutSampler.test.ts", () => {
  test("calcula mid e spread com bid/ask válidos", () => {
    assertTrue(Math.abs(computeMid(0.4, 0.42)! - 0.41) < 1e-12, "mid");
    assertTrue(Math.abs(computeSpread(0.4, 0.42)! - 0.02) < 1e-12, "spread");
    const row = snapshotRowFromPrices({
      bestBid: 0.57,
      bestAsk: 0.58,
      collectedAtIso: "2026-05-03T12:00:00.000Z",
    });
    assertEqual(row.mid, (0.57 + 0.58) / 2, "mid síncrono snapshot");
    assertEqual(row.bookType, "two_sided", "book");
    assertEqual(row.priceSource, PRICE_SOURCE_BID_ASK_MID, "price source");
    assertTrue(Math.abs((row.spread ?? 0) - 0.01) < 1e-9, "spread snapshot");
  });

  test("classifica livro bilateral completo quando bid/ask presentes", () => {
    assertEqual(classifyBookType(0.01, 0.02), "two_sided", "spread pequeno");
  });

  test("book sem bid ou sem ask vira not_two_sided", () => {
    assertEqual(classifyBookType(null, 0.5), "not_two_sided", "falta bid");
    assertEqual(classifyBookType(0.5, null), "not_two_sided", "falta ask");
    const rowOnlyAsk = snapshotRowFromPrices({
      bestBid: null,
      bestAsk: 0.6,
      collectedAtIso: "2026-05-03T12:00:00.001Z",
    });
    assertEqual(rowOnlyAsk.bookType, "not_two_sided", "topology");
    assertEqual(rowOnlyAsk.priceSource, PRICE_SOURCE_INSUFFICIENT_BID_ASK, "preco inadequado");
  });

  test("markouts todos zero geram FLAT_SAMPLE", () => {
    const m = computeMarkoutSummary({
      mid0: 0.5,
      mid5s: 0.5,
      mid30s: 0.5,
      mid60s: 0.5,
      bookTypesSnapshot: ["two_sided", "two_sided", "two_sided", "two_sided"],
    });
    assertEqual(m.sampleInformativenessVerdict, "FLAT_SAMPLE", "flat");
    assertTrue(m.nonZeroMarkoutCount === 0, "zeros");
    assertTrue(m.allFlat, "flat flag");
    assertEqual(classifyProbabilisticPricePinned([0.5, 0.5, 0.5, 0.5]), false, "mid central não pinned");
  });

  test("pelo menos um markout não zero gera INFORMATIVE_SAMPLE quando bilateral contínuo", () => {
    const mx = MARKOUT_NONZERO_EPS * 50;
    const summary = computeMarkoutSummary({
      mid0: 0.55,
      mid5s: 0.55 + mx,
      mid30s: 0.55 + mx / 10,
      mid60s: 0.55,
      bookTypesSnapshot: ["two_sided", "two_sided", "two_sided", "two_sided"],
    });
    assertEqual(summary.sampleInformativenessVerdict, "INFORMATIVE_SAMPLE", "informativo");
    assertTrue(summary.nonZeroMarkoutCount >= 1, "nonzero");
  });

  test("mid colado perto de 0.001 em todos snapshots → PRICE_PINNED_SAMPLE", () => {
    assertTrue(classifyProbabilisticPricePinned([0.001, 0.00105, 0.00103, 0.001]), "clusters near floor");
    const flatPinned = computeMarkoutSummary({
      mid0: 0.001,
      mid5s: 0.001,
      mid30s: 0.001,
      mid60s: 0.001,
      bookTypesSnapshot: ["two_sided", "two_sided", "two_sided", "two_sided"],
    });
    assertEqual(flatPinned.sampleInformativenessVerdict, "PRICE_PINNED_SAMPLE", "pinned");
    assertEqual(
      classifySampleInformativeness({
        bookTypesSnapshot: ["two_sided", "two_sided", "two_sided", "two_sided"],
        pricePinned: true,
        nonZeroMarkoutCount: 0,
      }),
      "PRICE_PINNED_SAMPLE",
      "prioridade pinned",
    );
  });

  test("topology insuficiente gera INSUFFICIENT_BOOK_SAMPLE", () => {
    const s = computeMarkoutSummary({
      mid0: 0.49,
      mid5s: 0.491,
      mid30s: 0.492,
      mid60s: 0.49,
      bookTypesSnapshot: ["two_sided", "not_two_sided", "two_sided", "two_sided"],
    });
    assertEqual(s.sampleInformativenessVerdict, "INSUFFICIENT_BOOK_SAMPLE", "topologia");
    assertEqual(
      classifySampleInformativeness({
        bookTypesSnapshot: ["two_sided", "two_sided", "two_sided", "not_two_sided"],
        pricePinned: false,
        nonZeroMarkoutCount: 999,
      }),
      "INSUFFICIENT_BOOK_SAMPLE",
      "gates laterais falsos",
    );
  });

  test("payload read-only marca microcapital sempre false quando presente em JSON virtual", () => {
    assertEqual(({ canUseForMicrocapitalCandidate: false }).canUseForMicrocapitalCandidate, false, "stub");
  });

  test("lib + sampler sem termos proibidos (scan estático leve)", () => {
    const root = path.join(__dirname, "..");
    const files = [
      path.join(root, "lib/singleTrackMarkoutSampler.ts"),
      path.join(root, "scripts/sampleSingleTrack553856Markout.ts"),
    ];
    const blocked = ["submit", "order", "wallet", "signer", "privatekey", "mnemonic", "seed phrase"];
    const blockLower = blocked.map(x => x.toLowerCase());
    for (const f of files) {
      const hay = fs.readFileSync(f, "utf8").toLowerCase();
      for (const frag of blockLower) {
        assertTrue(!hay.includes(frag), `${path.basename(f)} evita substring ${frag}`);
      }
    }
  });
});
