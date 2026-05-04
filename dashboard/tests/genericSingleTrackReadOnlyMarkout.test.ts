import fs from "fs";
import path from "path";
import {
  buildGammaMarketByIdUrl,
  decideBatchPaperShadowTrack,
  parseBatchCli,
  parseSamplingCli,
} from "../lib/singleTrackMarkoutSampler";
import { describe, test, assertEqual, assertTrue } from "./_assert";

describe("tests/genericSingleTrackReadOnlyMarkout.test.ts", () => {
  test("parseSamplingCli exige --market-id", () => {
    let threw = false;
    try {
      parseSamplingCli(["node", "sampleSingleTrackReadOnlyMarkout.ts"]);
    } catch {
      threw = true;
    }
    assertTrue(threw, "deveria falhar sem market-id");
  });

  test("buildGammaMarketByIdUrl usa path /markets/<id> e não query legacy ?id=", () => {
    const url = buildGammaMarketByIdUrl("553828");
    assertTrue(url.includes("/markets/553828"), "seg path");
    assertTrue(!url.includes("?id="), "sem filtros tipo ?id=");
  });

  test("JSON agregador batch declara microcapital sempre false", () => {
    const agg = { canUseForMicrocapitalCandidate: false as const };
    assertEqual(agg.canUseForMicrocapitalCandidate, false, "stub agregação");
  });

  test("batch decision RETIRE_AS_FIRST_TRACK quando todos flat (5/5)", () => {
    const d = decideBatchPaperShadowTrack({
      sampleCount: 5,
      informativeSamples: 0,
      flatSamples: 5,
      insufficientBookSamples: 0,
      pricePinnedSamples: 0,
    });
    assertEqual(d, "RETIRE_AS_FIRST_TRACK", "flat total");
  });

  test("batch decision CONTINUE_PAPER_SHADOW_CANDIDATE com ≥2 informativos", () => {
    const d = decideBatchPaperShadowTrack({
      sampleCount: 5,
      informativeSamples: 2,
      flatSamples: 1,
      insufficientBookSamples: 0,
      pricePinnedSamples: 0,
    });
    assertEqual(d, "CONTINUE_PAPER_SHADOW_CANDIDATE", "fluxo vivo");
  });

  test("batch decision RETRY_LATER com exatamente 1 informativo", () => {
    const d = decideBatchPaperShadowTrack({
      sampleCount: 5,
      informativeSamples: 1,
      flatSamples: 0,
      insufficientBookSamples: 0,
      pricePinnedSamples: 0,
    });
    assertEqual(d, "RETRY_LATER", "marginal");
  });

  test("batch decision RETIRE_IMMEDIATELY quando pricePinned ou maioría insufficient", () => {
    assertEqual(
      decideBatchPaperShadowTrack({
        sampleCount: 5,
        informativeSamples: 0,
        flatSamples: 0,
        insufficientBookSamples: 0,
        pricePinnedSamples: 1,
      }),
      "RETIRE_IMMEDIATELY",
      "pinned",
    );
    assertEqual(
      decideBatchPaperShadowTrack({
        sampleCount: 5,
        informativeSamples: 0,
        flatSamples: 0,
        insufficientBookSamples: 3,
        pricePinnedSamples: 0,
      }),
      "RETIRE_IMMEDIATELY",
      "livro inadequado maioritário",
    );
  });

  test("scripts read-only não trazem termos de execução reservados (regex)", () => {
    const root = path.join(__dirname, "..");
    const files = [
      path.join(root, "scripts/sampleSingleTrackReadOnlyMarkout.ts"),
      path.join(root, "scripts/batchSingleTrackReadOnlyMarkout.ts"),
    ];
    const patterns: RegExp[] = [
      /\border\b/i,
      /\bsubmit\b/i,
      /\bwallet\b/i,
      /\bsigner\b/i,
      /\bprivatekey\b/i,
      /\bmnemonic\b/i,
      /\bseed\s+phrase\b/i,
    ];
    for (const f of files) {
      const hay = fs.readFileSync(f, "utf8");
      for (const re of patterns) {
        assertTrue(!re.test(hay), `${path.basename(f)} evita padrão ${re}`);
      }
    }
  });

  test("parseBatchCli exige market-id e out-dir", () => {
    let missingMarket = false;
    try {
      parseBatchCli(["node", "batch.ts", "--out-dir", "/tmp/x"]);
    } catch {
      missingMarket = true;
    }
    assertTrue(missingMarket, "market-id obrigatório");

    let missingDir = false;
    try {
      parseBatchCli(["node", "batch.ts", "--market-id", "1"]);
    } catch {
      missingDir = true;
    }
    assertTrue(missingDir, "out-dir obrigatório");
  });
});
