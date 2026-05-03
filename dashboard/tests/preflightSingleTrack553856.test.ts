import fs from "fs";
import path from "path";
import {
  buildGamma553856MarketPreflightUrl,
  compute553856PreflightPayload,
  SINGLE_TRACK_553856_MARKET_ID,
  type GammaYesBookProbe,
} from "../scripts/preflightSingleTrack553856";
import { describe, test, assertEqual, assertTrue } from "./_assert";

describe("tests/preflightSingleTrack553856.test.ts", () => {
  const NOW = "2026-05-03T12:00:00.000Z";

  const baseThunderRow = (): Record<string, unknown> => ({
    id: SINGLE_TRACK_553856_MARKET_ID,
    question: "Will the Oklahoma City Thunder win the 2026 NBA Finals?",
    slug: "will-the-oklahoma-city-thunder-win-the-2026-nba-finals",
    active: true,
    closed: false,
    resolved: false,
    endDate: "2027-06-01T00:00:00Z",
    volume: "10142561.58",
    liquidity: "238625.08",
    conditionId: "0xdeadbeef",
    outcomes: '["Yes","No"]',
    clobTokenIds: '["YES_TOKEN_553856","NO_TOKEN_553856"]',
    bestBid: 0.57,
    bestAsk: 0.58,
  });

  const twoSidedProbe = (): GammaYesBookProbe => ({
    bookStructure: "two_sided",
    bestBidBook: 0.57,
    bestAskBook: 0.58,
    yesTokenPrefix: "YES_TOKEN_553856…",
  });

  test("preflight usa path endpoint /markets/553856", () => {
    const url = buildGamma553856MarketPreflightUrl();
    assertEqual(url, "https://gamma-api.polymarket.com/markets/553856", "gamma path url");
    assertTrue(url.includes(`/markets/${SINGLE_TRACK_553856_MARKET_ID}`), "relative path segment");
  });

  test("preflight nunca usa ?id=553856", () => {
    const url = buildGamma553856MarketPreflightUrl();
    assertTrue(!url.includes("?id="), "legacy query id must be absent");
    assertTrue(url.indexOf("?") < 0, "no query string on gamma market url");
  });

  test("resultado mantém canUseForMicrocapitalCandidate false (ready path)", () => {
    const payload = compute553856PreflightPayload({
      nowIso: NOW,
      gammaRow: baseThunderRow(),
      yesBookProbe: twoSidedProbe(),
    });
    assertEqual(payload.canUseForMicrocapitalCandidate, false, "micro flag");
    assertEqual(payload.suitability.canUseForMicrocapitalCandidate, false, "gate micro");
  });

  test("mercado fechado ou resolvido retorna NOT_READY", () => {
    const closedPay = compute553856PreflightPayload({
      nowIso: NOW,
      gammaRow: { ...baseThunderRow(), closed: true },
      yesBookProbe: twoSidedProbe(),
    });
    assertEqual(closedPay.preflightVerdict, "NOT_READY_FOR_PAPER_SHADOW_OBSERVATION", "closed");

    const resolvedPay = compute553856PreflightPayload({
      nowIso: NOW,
      gammaRow: { ...baseThunderRow(), resolved: true },
      yesBookProbe: twoSidedProbe(),
    });
    assertEqual(resolvedPay.preflightVerdict, "NOT_READY_FOR_PAPER_SHADOW_OBSERVATION", "resolved");
  });

  test("missing clobTokenIds retorna NOT_READY", () => {
    const row = { ...baseThunderRow(), clobTokenIds: null };
    const pay = compute553856PreflightPayload({
      nowIso: NOW,
      gammaRow: row,
      yesBookProbe: { bookStructure: "skipped", bestBidBook: null, bestAskBook: null, yesTokenPrefix: null },
    });
    assertEqual(pay.preflightVerdict, "NOT_READY_FOR_PAPER_SHADOW_OBSERVATION", "verdict");
    assertTrue(pay.reasons.some(r => r.includes("missing_gamma_clobTokenIds")), "reason token");
  });

  test("book sem bid/ask retorna NOT_READY", () => {
    const pay = compute553856PreflightPayload({
      nowIso: NOW,
      gammaRow: baseThunderRow(),
      yesBookProbe: {
        bookStructure: "two_sided",
        bestBidBook: null,
        bestAskBook: 0.601,
        yesTokenPrefix: "YES_TOKEN_553856…",
      },
    });
    assertEqual(pay.preflightVerdict, "NOT_READY_FOR_PAPER_SHADOW_OBSERVATION", "verdict topo vazio");
    assertTrue(pay.reasons.some(r => r.includes("clob_book_best_prices_missing")), "precisa denunciar topo vazio");
  });

  test("livro two-sided e suitability ok retorna READY_FOR_PAPER_SHADOW_OBSERVATION", () => {
    const pay = compute553856PreflightPayload({
      nowIso: NOW,
      gammaRow: baseThunderRow(),
      yesBookProbe: twoSidedProbe(),
    });
    assertEqual(pay.preflightVerdict, "READY_FOR_PAPER_SHADOW_OBSERVATION", "verdict ready");
    assertEqual(pay.suitability.canUseForPaperShadowCandidate, true, "shadow candidate gate");
    assertEqual(pay.suitability.suitabilityVerdict, "SUITABLE_FOR_PAPER_SHADOW_OBSERVATION", "suit verdict");
  });

  test("paths novos sem termos financeiros/execução proibidos (scan estático)", () => {
    const root = path.join(__dirname, "..");
    const files = [
      path.join(root, "scripts/preflightSingleTrack553856.ts"),
      path.join(root, "docs/single-track-observation-553856.md"),
    ];
    const banned = ["submit", "order", "wallet", "signer", "privatekey", "mnemonic", "seed phrase"];
    const lowerBanned = banned.map(x => x.toLowerCase());
    for (const f of files) {
      const txt = fs.readFileSync(f, "utf8").toLowerCase();
      for (const needle of lowerBanned) {
        assertTrue(!txt.includes(needle), `${path.basename(f)} deve evitar substring ${needle}`);
      }
    }
  });
});
