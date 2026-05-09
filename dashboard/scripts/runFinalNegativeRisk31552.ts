/**
 * Worker persistente: reexecuta a validação final venue-native do evento 31552
 * (buildFinalNegativeRiskValidation31552Digest) em intervalo fixo e regista JSONL em PAPER_STATE_DIR.
 *
 * Railway: definir PAPER_STATE_DIR=/data + volume montado nesse path; startCommand abaixo no package.json.
 * Intervalo: FINAL_NEG_RISK_31552_INTERVAL_MS (default 600_000 = 10 min).
 * Dry run único: --once
 */

import http from "http";
import fs from "fs";
import path from "path";
import { buildFinalNegativeRiskValidation31552Digest } from "../lib/finalNegativeRiskValidation31552";
import {
  registerLiveExperimentRunner,
  syncLiveExperimentOperationalMeta,
} from "../lib/liveExperimentMeta";
import { parseFinalNegRisk31552 } from "../lib/liveExperimentStatsShared";
import { writeLiveExperimentWorkerSummary } from "../lib/liveExperimentWorkerSummary";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES, resolvePaperStateDir } from "../lib/paperStateDir";

function parseIntervalMs(): number {
  const raw = process.env.FINAL_NEG_RISK_31552_INTERVAL_MS?.trim();
  if (raw) {
    const n = parseInt(raw, 10);
    if (Number.isFinite(n) && n >= 60_000) return n;
    console.warn(
      "[final-neg-risk-31552] invalid FINAL_NEG_RISK_31552_INTERVAL_MS; using default 600000 (10 min)",
    );
  }
  return 600_000;
}

function historyPath(): string {
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.finalNegativeRisk31552History);
}

function sleep(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

/** Responde em PORT para o healthcheck de deploy Railway (GET / ou /health). */
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
    console.log(`[final-neg-risk-31552] deploy health on :${port}`);
  });
}

async function appendObservation(): Promise<void> {
  const digest = await buildFinalNegativeRiskValidation31552Digest();
  const isoTimestamp = new Date().toISOString();
  const row = {
    isoTimestamp,
    eventId: digest.eventId,
    rawEstimatedConversionEdge: digest.rawEstimatedConversionEdge,
    observedEntryCost: digest.observedEntryCost,
    observedConversionCost: digest.observedConversionCost,
    observedExitCost: digest.observedExitCost,
    stressAdjustedNetConversionEdge: digest.stressAdjustedNetConversionEdge,
    capacityEstimate: digest.capacityEstimate,
    fragilityAssessment: digest.fragilityAssessment,
    finalNegativeRiskValidationVerdict: digest.finalNegativeRiskValidationVerdict,
    supportingNote: digest.finalNegativeRiskValidationSummaryLine,
  };
  const fp = historyPath();
  const dir = path.dirname(fp);
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(fp, `${JSON.stringify(row)}\n`, { encoding: "utf8" });
  syncLiveExperimentOperationalMeta({
    experimentId: "final_neg_risk_31552",
    serviceName: "final-neg-risk-31552-worker",
    metaFilename: PAPER_TRAIL_FILENAMES.finalNegativeRisk31552Meta,
    historyFilename: PAPER_TRAIL_FILENAMES.finalNegativeRisk31552History,
    metaEnvOverride: process.env.FINAL_NEG_RISK_31552_META_PATH,
    intervalMs: parseIntervalMs(),
  });
  writeLiveExperimentWorkerSummary({
    experimentId: "final_neg_risk_31552",
    experimentType: "final_negative_risk_validation_31552_jsonl",
    primaryNetMetricKey: "stressAdjustedNetConversionEdge",
    summaryFilenameKey: "finalNegativeRisk31552Summary",
    summaryEnvOverride: process.env.FINAL_NEG_RISK_31552_SUMMARY_PATH,
    historyFilenameKey: "finalNegativeRisk31552History",
    historyEnvOverride: process.env.FINAL_NEG_RISK_31552_HISTORY_PATH,
    metaFilenameKey: "finalNegativeRisk31552Meta",
    metaEnvOverride: process.env.FINAL_NEG_RISK_31552_META_PATH,
    parse: parseFinalNegRisk31552,
    intervalMs: parseIntervalMs(),
  });
  console.log(`[final-neg-risk-31552] appended ${isoTimestamp} → ${fp}`);
}

async function main(): Promise<void> {
  const once = process.argv.includes("--once");
  if (!once) {
    startDeployHealthServer();
  }
  const dir = resolvePaperStateDir();
  const intervalMs = parseIntervalMs();
  registerLiveExperimentRunner({
    experimentId: "final_neg_risk_31552",
    serviceName: "final-neg-risk-31552-worker",
    metaFilename: PAPER_TRAIL_FILENAMES.finalNegativeRisk31552Meta,
    historyFilename: PAPER_TRAIL_FILENAMES.finalNegativeRisk31552History,
    metaEnvOverride: process.env.FINAL_NEG_RISK_31552_META_PATH,
    intervalMs,
  });
  console.log(
    `[final-neg-risk-31552] PAPER_STATE_DIR=${dir} history=${historyPath()} intervalMs=${once ? "N/A (--once)" : String(intervalMs)}`,
  );

  if (once) {
    await appendObservation();
    return;
  }

  for (;;) {
    try {
      await appendObservation();
    } catch (e) {
      console.error("[final-neg-risk-31552] cycle error:", e instanceof Error ? e.message : e);
    }
    await sleep(intervalMs);
  }
}

main().catch(e => {
  console.error("[final-neg-risk-31552] fatal:", e);
  process.exit(1);
});
