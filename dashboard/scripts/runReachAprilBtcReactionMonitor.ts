/**
 * Worker persistente: observação BTC narrow (1823774, 1823775) com a mesma lógica calibrada
 * `buildBtcNarrowReactionMonitorObservation`, intervalo 10 min, histórico JSONL append-only em PAPER_STATE_DIR.
 *
 * Railway: `PAPER_STATE_DIR=/data` + volume; `PORT` para healthcheck HTTP.
 * Intervalo: REACH_APRIL_BTC_MONITOR_INTERVAL_MS (default 600_000).
 * Ciclo único: `--once`
 */

import http from "http";
import fs from "fs";
import path from "path";
import {
  buildBtcNarrowReactionMonitorObservation,
  REACH_APRIL_BTC_NARROW_MONITOR_MARKETS,
} from "../lib/reachAprilCalibratedReactionPilot";
import {
  registerLiveExperimentRunner,
  syncLiveExperimentOperationalMeta,
} from "../lib/liveExperimentMeta";
import { parseReachAprilBtc } from "../lib/liveExperimentStatsShared";
import { writeLiveExperimentWorkerSummary } from "../lib/liveExperimentWorkerSummary";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES, resolvePaperStateDir } from "../lib/paperStateDir";

function parseIntervalMs(): number {
  const raw = process.env.REACH_APRIL_BTC_MONITOR_INTERVAL_MS?.trim();
  if (raw) {
    const n = parseInt(raw, 10);
    if (Number.isFinite(n) && n >= 60_000) return n;
    console.warn(
      "[reach-april-btc-monitor] invalid REACH_APRIL_BTC_MONITOR_INTERVAL_MS; using default 600000 (10 min)",
    );
  }
  return 600_000;
}

function historyPath(): string {
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.reachAprilBtcReactionMonitorHistory);
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
    console.log(`[reach-april-btc-monitor] deploy health on :${port}`);
  });
}

async function appendObservation(): Promise<void> {
  const obs = await buildBtcNarrowReactionMonitorObservation();
  const fp = historyPath();
  const dir = path.dirname(fp);
  fs.mkdirSync(dir, { recursive: true });

  const allowedIds = new Set<string>(REACH_APRIL_BTC_NARROW_MONITOR_MARKETS.map(m => m.marketId));

  for (const m of obs.markets.filter(row => allowedIds.has(String(row.marketId)))) {
    const row = {
      isoTimestamp: obs.isoTimestamp,
      probeVersion: obs.probeVersion,
      observationWindowMs: obs.observationWindowMs,
      marketId: m.marketId,
      marketTitle: m.marketTitle,
      triggerCount: m.triggerCount,
      avgLagObserved: m.avgLagObserved,
      medianNetPerReactionCycle: m.medianNetPerReactionCycle,
      worstNetPerReactionCycle: m.worstNetPerReactionCycle,
      calibratedReactionVerdictPerMarket: m.calibratedReactionVerdictPerMarket,
      supportingNote: m.supportingNote,
    };
    fs.appendFileSync(fp, `${JSON.stringify(row)}\n`, { encoding: "utf8" });
  }

  syncLiveExperimentOperationalMeta({
    experimentId: "reach_april_btc_reaction_monitor",
    serviceName: "reach-april-btc-monitor-worker",
    metaFilename: PAPER_TRAIL_FILENAMES.reachAprilBtcReactionMonitorMeta,
    historyFilename: PAPER_TRAIL_FILENAMES.reachAprilBtcReactionMonitorHistory,
    metaEnvOverride: process.env.REACH_APRIL_BTC_MONITOR_META_PATH,
    intervalMs: parseIntervalMs(),
  });
  writeLiveExperimentWorkerSummary({
    experimentId: "reach_april_btc_reaction_monitor",
    experimentType: "reach_april_btc_reaction_monitor_jsonl",
    primaryNetMetricKey: "medianNetPerReactionCycle",
    summaryFilenameKey: "reachAprilBtcReactionMonitorSummary",
    summaryEnvOverride: process.env.REACH_APRIL_BTC_MONITOR_SUMMARY_PATH,
    historyFilenameKey: "reachAprilBtcReactionMonitorHistory",
    historyEnvOverride: process.env.REACH_APRIL_BTC_MONITOR_HISTORY_PATH,
    metaFilenameKey: "reachAprilBtcReactionMonitorMeta",
    metaEnvOverride: process.env.REACH_APRIL_BTC_MONITOR_META_PATH,
    parse: parseReachAprilBtc,
    intervalMs: parseIntervalMs(),
  });

  console.log(`[reach-april-btc-monitor] appended ${obs.isoTimestamp} → ${fp} (${obs.markets.length} markets)`);
}

async function main(): Promise<void> {
  const once = process.argv.includes("--once");
  if (!once) {
    startDeployHealthServer();
  }
  const dir = resolvePaperStateDir();
  const intervalMs = parseIntervalMs();
  registerLiveExperimentRunner({
    experimentId: "reach_april_btc_reaction_monitor",
    serviceName: "reach-april-btc-monitor-worker",
    metaFilename: PAPER_TRAIL_FILENAMES.reachAprilBtcReactionMonitorMeta,
    historyFilename: PAPER_TRAIL_FILENAMES.reachAprilBtcReactionMonitorHistory,
    metaEnvOverride: process.env.REACH_APRIL_BTC_MONITOR_META_PATH,
    intervalMs,
  });
  console.log(
    `[reach-april-btc-monitor] PAPER_STATE_DIR=${dir} history=${historyPath()} intervalMs=${once ? "N/A (--once)" : String(intervalMs)}`,
  );

  if (once) {
    await appendObservation();
    return;
  }

  for (;;) {
    try {
      await appendObservation();
    } catch (e) {
      console.error("[reach-april-btc-monitor] cycle error:", e instanceof Error ? e.message : e);
    }
    await sleep(intervalMs);
  }
}

main().catch(e => {
  console.error("[reach-april-btc-monitor] fatal:", e);
  process.exit(1);
});
