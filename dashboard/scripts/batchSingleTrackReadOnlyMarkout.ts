/**
 * Batch read-only de markouts sobre `sampleSingleTrackReadOnlyMarkoutRun`.
 */

import fs from "fs";
import path from "path";

import {
  decideBatchPaperShadowTrack,
  parseBatchCli,
  type BatchTrackReadDecision,
} from "../lib/singleTrackMarkoutSampler";
import {
  assertSamplerOutputPath,
  sampleSingleTrackReadOnlyMarkoutRun,
} from "./sampleSingleTrackReadOnlyMarkout";

function isCompleteRecord(p: Record<string, unknown>): boolean {
  return p.sampleVerdict === "COMPLETE_READ_ONLY_MARKOUT_SAMPLING";
}

function delayMs(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function collectSpreadMidRange(body: Record<string, unknown>): { mids: number[]; spreads: number[] } {
  const hz = body.horizons;
  const mids: number[] = [];
  const spreads: number[] = [];
  if (hz && typeof hz === "object" && !Array.isArray(hz)) {
    for (const snap of Object.values(hz)) {
      if (!snap || typeof snap !== "object") continue;
      const r = snap as Record<string, unknown>;
      const m = typeof r.mid === "number" && Number.isFinite(r.mid) ? r.mid : null;
      const sp = typeof r.spread === "number" && Number.isFinite(r.spread) ? r.spread : null;
      if (m !== null) mids.push(m);
      if (sp !== null) spreads.push(sp);
    }
  }
  return { mids, spreads };
}

export async function runBatchSingleTrackReadOnlyMarkout(args: {
  marketId: string;
  label?: string;
  samples: number;
  gapSeconds: number;
  horizonsSec?: readonly number[];
  outDir: string;
}): Promise<{ aggregateJson: Record<string, unknown>; decision: BatchTrackReadDecision }> {
  assertSamplerOutputPath(path.join(args.outDir, "aggregate.json.placeholder"));

  const horizons = args.horizonsSec ?? [5, 30, 60];
  const normalizedDir = path.resolve(args.outDir);
  await fs.promises.mkdir(normalizedDir, { recursive: true });

  let informativeSamples = 0;
  let flatSamples = 0;
  let insufficientBookSamples = 0;
  let pricePinnedSamples = 0;
  let notReadyCount = 0;

  let midMin = Number.POSITIVE_INFINITY;
  let midMax = Number.NEGATIVE_INFINITY;
  let spreadMin = Number.POSITIVE_INFINITY;
  let spreadMax = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < args.samples; i++) {
    const payload = await sampleSingleTrackReadOnlyMarkoutRun({
      marketId: args.marketId,
      label: args.label,
      horizonsSec: horizons,
    });

    await fs.promises.writeFile(path.join(normalizedDir, `sample_${i}.json`), `${JSON.stringify(payload, null, 2)}\n`, {
      encoding: "utf8",
    });

    if (!isCompleteRecord(payload)) {
      notReadyCount++;
    } else {
      const verdictRaw = payload.sampleInformativenessVerdict;
      const verdict = typeof verdictRaw === "string" ? verdictRaw : "";
      if (verdict === "INFORMATIVE_SAMPLE") informativeSamples++;
      else if (verdict === "FLAT_SAMPLE") flatSamples++;
      else if (verdict === "INSUFFICIENT_BOOK_SAMPLE") insufficientBookSamples++;
      else if (verdict === "PRICE_PINNED_SAMPLE") pricePinnedSamples++;

      const { mids, spreads } = collectSpreadMidRange(payload);
      for (const m of mids) {
        midMin = Math.min(midMin, m);
        midMax = Math.max(midMax, m);
      }
      for (const s of spreads) {
        spreadMin = Math.min(spreadMin, s);
        spreadMax = Math.max(spreadMax, s);
      }
    }

    if (i + 1 < args.samples && args.gapSeconds > 0) {
      await delayMs(args.gapSeconds * 1000);
    }
  }

  const decision = decideBatchPaperShadowTrack({
    sampleCount: args.samples,
    informativeSamples,
    flatSamples,
    insufficientBookSamples,
    pricePinnedSamples,
  });

  const midRangeFinite = Number.isFinite(midMin) && Number.isFinite(midMax);
  const spreadRangeFinite = Number.isFinite(spreadMin) && Number.isFinite(spreadMax);

  const aggregateJson = {
    marketId: args.marketId.trim(),
    label: args.label,
    sampleCount: args.samples,
    notReadyCount,
    informativeSamples,
    flatSamples,
    insufficientBookSamples,
    pricePinnedSamples,
    midMin: midRangeFinite ? midMin : null,
    midMax: midRangeFinite ? midMax : null,
    midRange: midRangeFinite ? midMax - midMin : null,
    spreadMin: spreadRangeFinite ? spreadMin : null,
    spreadMax: spreadRangeFinite ? spreadMax : null,
    batchPaperShadowDecision: decision,
    canUseForMicrocapitalCandidate: false as const,
  };

  await fs.promises.writeFile(path.join(normalizedDir, `aggregate.json`), `${JSON.stringify(aggregateJson, null, 2)}\n`, {
    encoding: "utf8",
  });

  return { aggregateJson, decision };
}

async function main(): Promise<void> {
  const cli = parseBatchCli(process.argv);
  const { aggregateJson } = await runBatchSingleTrackReadOnlyMarkout({
    marketId: cli.marketId,
    label: cli.label,
    samples: cli.samples,
    gapSeconds: cli.gapSeconds,
    outDir: cli.outDir,
  });

  process.stdout.write(`${JSON.stringify(aggregateJson, null, 2)}\n`);
}

if (require.main === module) {
  main().catch(err => {
    process.stderr.write(
      `[batchSingleTrackReadOnlyMarkout] fatal: ${err instanceof Error ? err.message : String(err)}\n`,
    );
    process.exitCode = 1;
  });
}
