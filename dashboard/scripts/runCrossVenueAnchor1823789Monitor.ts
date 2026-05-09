/**
 * Worker persistente: validação refined cross-venue anchor apenas para mercado 1823789,
 * intervalo 10 min, histórico JSONL append-only em PAPER_STATE_DIR (nunca sobrescreve).
 *
 * Railway: `PAPER_STATE_DIR=/data` + volume montado nesse path; `PORT` para healthcheck HTTP.
 * Intervalo: `CROSS_VENUE_ANCHOR_1823789_MONITOR_INTERVAL_MS` (default 600_000).
 * Ciclo único local/diagnóstico: `--once`
 */

import http from "http";
import fs from "fs";
import path from "path";
import { buildCrossVenueAnchor1823789RefinedObservation } from "../lib/crossVenueAnchorPilotRefined";
import {
  registerLiveExperimentRunner,
  syncLiveExperimentOperationalMeta,
} from "../lib/liveExperimentMeta";
import { parseCrossVenueAnchor1823789 } from "../lib/liveExperimentStatsShared";
import { writeLiveExperimentWorkerSummary } from "../lib/liveExperimentWorkerSummary";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES, resolvePaperStateDir } from "../lib/paperStateDir";

function parseIntervalMs(): number {
  const raw = process.env.CROSS_VENUE_ANCHOR_1823789_MONITOR_INTERVAL_MS?.trim();
  if (raw) {
    const n = parseInt(raw, 10);
    if (Number.isFinite(n) && n >= 60_000) return n;
    console.warn(
      "[cross-venue-anchor-1823789] invalid CROSS_VENUE_ANCHOR_1823789_MONITOR_INTERVAL_MS; using default 600000 (10 min)",
    );
  }
  return 600_000;
}

function historyPath(): string {
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789MonitorHistory);
}

function sleep(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

function startDeployHealthServer(): void {
  const raw = process.env.PORT?.trim();
  const port = raw ? parseInt(raw, 10) : 8080;
  if (!Number.isFinite(port) || port <= 0) return;
  const server = http.createServer((req, res) => {
    const p = req.url?.split("?")[0] || "/";
    if (p === "/" || p === "/health") {
      res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("ok");
      return;
    }
    res.writeHead(404);
    res.end();
  });
  server.listen(port, "0.0.0.0", () => {
    console.log(`[cross-venue-anchor-1823789] deploy health on :${port}`);
  });
}

async function appendObservation(): Promise<void> {
  const obs = await buildCrossVenueAnchor1823789RefinedObservation();
  const m = obs.row;
  const isoTimestamp = obs.isoTimestamp;
  const row = {
    isoTimestamp,
    probeVersion: obs.probeVersion,
    marketId: m.marketId,
    marketTitle: m.marketTitle,
    refinedFairValue: m.refinedFairValue,
    anchorPriceObserved: m.anchorPriceObserved,
    polymarketPriceObserved: m.polymarketPriceObserved,
    rawAnchorGap: m.rawAnchorGap,
    estimatedEntryCost: m.estimatedEntryCost,
    estimatedExitCost: m.estimatedExitCost,
    estimatedHedgeCost: m.estimatedHedgeCost,
    estimatedLegRiskCost: m.estimatedLegRiskCost,
    estimatedNetAnchorCycle: m.estimatedNetAnchorCycle,
    refinedPilotVerdictPerMarket: m.refinedPilotVerdictPerMarket,
    supportingNote: m.supportingNote,
  };

  const fp = historyPath();
  const dir = path.dirname(fp);
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(fp, `${JSON.stringify(row)}\n`, { encoding: "utf8" });

  syncLiveExperimentOperationalMeta({
    experimentId: "cross_venue_anchor_1823789",
    serviceName: "cross-venue-anchor-1823789-monitor",
    metaFilename: PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789MonitorMeta,
    historyFilename: PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789MonitorHistory,
    metaEnvOverride: process.env.CROSS_VENUE_ANCHOR_1823789_MONITOR_META_PATH,
    intervalMs: parseIntervalMs(),
  });
  writeLiveExperimentWorkerSummary({
    experimentId: "cross_venue_anchor_1823789",
    experimentType: "cross_venue_anchor_refined_1823789_jsonl",
    primaryNetMetricKey: "estimatedNetAnchorCycle",
    summaryFilenameKey: "crossVenueAnchor1823789MonitorSummary",
    summaryEnvOverride: process.env.CROSS_VENUE_ANCHOR_1823789_MONITOR_SUMMARY_PATH,
    historyFilenameKey: "crossVenueAnchor1823789MonitorHistory",
    historyEnvOverride: process.env.CROSS_VENUE_ANCHOR_1823789_MONITOR_HISTORY_PATH,
    metaFilenameKey: "crossVenueAnchor1823789MonitorMeta",
    metaEnvOverride: process.env.CROSS_VENUE_ANCHOR_1823789_MONITOR_META_PATH,
    parse: parseCrossVenueAnchor1823789,
    intervalMs: parseIntervalMs(),
  });

  console.log(`[cross-venue-anchor-1823789] appended ${isoTimestamp} → ${fp}`);
}

async function main(): Promise<void> {
  const once = process.argv.includes("--once");
  if (!once) {
    startDeployHealthServer();
  }
  const dir = resolvePaperStateDir();
  const intervalMs = parseIntervalMs();
  registerLiveExperimentRunner({
    experimentId: "cross_venue_anchor_1823789",
    serviceName: "cross-venue-anchor-1823789-monitor",
    metaFilename: PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789MonitorMeta,
    historyFilename: PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789MonitorHistory,
    metaEnvOverride: process.env.CROSS_VENUE_ANCHOR_1823789_MONITOR_META_PATH,
    intervalMs,
  });
  console.log(
    `[cross-venue-anchor-1823789] PAPER_STATE_DIR=${dir} history=${historyPath()} intervalMs=${once ? "N/A (--once)" : String(intervalMs)}`,
  );

  if (once) {
    await appendObservation();
    return;
  }

  for (;;) {
    try {
      await appendObservation();
    } catch (e) {
      console.error("[cross-venue-anchor-1823789] cycle error:", e instanceof Error ? e.message : e);
    }
    await sleep(intervalMs);
  }
}

main().catch(e => {
  console.error("[cross-venue-anchor-1823789] fatal:", e);
  process.exit(1);
});
