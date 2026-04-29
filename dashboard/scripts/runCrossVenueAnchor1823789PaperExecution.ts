/**
 * Worker persistente: paper execution narrow só mercado 1823789.
 * Cada ciclo: avalia gates (narrow + execution); só grava ciclo "executado" em papel se ambos passarem.
 * Caso contrário grava ciclo bloqueado (histórico completo para auditoria).
 *
 * Railway: `PAPER_STATE_DIR=/data` + volume; `PORT` para healthcheck HTTP.
 * Intervalo: `CROSS_VENUE_ANCHOR_1823789_PAPER_EXECUTION_INTERVAL_MS` (default 600_000).
 * Ciclo único: `--once`
 *
 * Tier: fixo 1x (sem promoção automática a 2x/5x).
 *
 * Modos: `CROSS_VENUE_ANCHOR_1823789_EXECUTION_MODE=controlled_paper|minimal_micro_live`.
 * Em `minimal_micro_live` com halt desligado: shadow-only grava JSONL por ciclo (incl. blocked_by_gate), sem submissão real.
 */

import http from "http";
import fs from "fs";
import path from "path";
import { buildCrossVenueAnchor1823789ExecutionGateDigest } from "../lib/crossVenueAnchor1823789ExecutionGate";
import {
  parseCrossVenueAnchor1823789ExecutionModeFromEnv,
  parseCrossVenueAnchor1823789MaxNotionalUsd,
  parseCrossVenueAnchor1823789MicroLiveHaltFromEnv,
  resolveEffectiveCrossVenueAnchor1823789ExecutionMode,
} from "../lib/crossVenueAnchor1823789ExecutionMode";
import {
  buildCrossVenueAnchor1823789ShadowTargetLiveOrder,
  SHADOW_MICRO_LIVE_PROBE_VERSION,
  assertShadowOnlyNoRealSubmission,
  type CrossVenueAnchor1823789ShadowTargetLiveOrder,
} from "../lib/crossVenueAnchor1823789ShadowMicroLive";
import { runCrossVenueAnchor1823789ClobConnectivityProbe } from "../lib/crossVenueAnchor1823789ClobConnectivityProbe";
import {
  registerLiveExperimentRunner,
  syncLiveExperimentOperationalMeta,
} from "../lib/liveExperimentMeta";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES, resolvePaperStateDir } from "../lib/paperStateDir";

const PAPER_EXECUTION_PROBE_VERSION = "cross-venue-anchor-1823789-paper-execution-v1" as const;

const TIER_USED = "1x" as const;

function parseIntervalMs(): number {
  const raw = process.env.CROSS_VENUE_ANCHOR_1823789_PAPER_EXECUTION_INTERVAL_MS?.trim();
  if (raw) {
    const n = parseInt(raw, 10);
    if (Number.isFinite(n) && n >= 60_000) return n;
    console.warn(
      "[1823789-paper-exec] invalid CROSS_VENUE_ANCHOR_1823789_PAPER_EXECUTION_INTERVAL_MS; using default 600000",
    );
  }
  return 600_000;
}

function historyPath(): string {
  const raw = process.env.CROSS_VENUE_ANCHOR_1823789_PAPER_EXECUTION_HISTORY_PATH?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789PaperExecutionHistory);
}

function clobConnectivityHistoryPath(): string {
  const raw = process.env.CROSS_VENUE_ANCHOR_1823789_CLOB_CONNECTIVITY_HISTORY_PATH?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789ClobConnectivityHistory);
}

function shadowAuditHistoryPath(): string {
  const raw = process.env.CROSS_VENUE_ANCHOR_1823789_SHADOW_AUDIT_HISTORY_PATH?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789MicroLiveShadowAudit);
}

function parseConnectivityProbeAttempts(): number {
  const raw = process.env.CROSS_VENUE_ANCHOR_1823789_CLOB_CONNECTIVITY_ATTEMPTS?.trim();
  if (!raw) return 3;
  const n = parseInt(raw, 10);
  if (Number.isFinite(n) && n >= 1 && n <= 10) return n;
  return 3;
}

async function appendClobConnectivityProbe(): Promise<void> {
  const attempts = parseConnectivityProbeAttempts();
  const rows = await runCrossVenueAnchor1823789ClobConnectivityProbe(attempts);
  const fp = clobConnectivityHistoryPath();
  const dir = path.dirname(fp);
  fs.mkdirSync(dir, { recursive: true });
  for (const row of rows) {
    fs.appendFileSync(fp, `${JSON.stringify(row)}\n`, { encoding: "utf8" });
  }
  console.log(`[1823789-paper-exec] clob-connectivity appended attempts=${rows.length} → ${fp}`);
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
    console.log(`[1823789-paper-exec] deploy health on :${port}`);
  });
}

function gatesPass(gate: Awaited<ReturnType<typeof buildCrossVenueAnchor1823789ExecutionGateDigest>>): boolean {
  const n = gate.narrowValidationVerdict === "survives_stress_suite";
  const x =
    gate.executionGateVerdict === "survives_execution_gate" ||
    gate.executionGateVerdict === "survives_and_scalable";
  return n && x;
}

async function appendCycle(): Promise<void> {
  try {
    await appendClobConnectivityProbe();
  } catch (e) {
    console.error(
      "[1823789-paper-exec] clob-connectivity probe error:",
      e instanceof Error ? e.message : e,
    );
  }

  const isoTimestamp = new Date().toISOString();
  const gate = await buildCrossVenueAnchor1823789ExecutionGateDigest();
  const ok = gatesPass(gate);

  const entrySlip = gate.estimatedEntrySlippageByTier[TIER_USED];
  const exitSlip = gate.estimatedExitSlippageByTier[TIER_USED];
  const netAfter = ok ? gate.estimatedNetAnchorCycleByTier[TIER_USED] : null;

  let paperExecutionVerdict: "blocked_by_gate" | "paper_cycle_positive" | "paper_cycle_negative";
  if (!ok) {
    paperExecutionVerdict = "blocked_by_gate";
  } else if (netAfter! > 0) {
    paperExecutionVerdict = "paper_cycle_positive";
  } else {
    paperExecutionVerdict = "paper_cycle_negative";
  }

  const supportingNote = [
    `gates_pass=${ok}`,
    `execution_gate_summary=${gate.executionGateSummaryLine}`,
    `tier_fixed=${TIER_USED}`,
  ].join("|");

  const row = {
    probeVersion: PAPER_EXECUTION_PROBE_VERSION,
    isoTimestamp,
    marketId: gate.marketId,
    marketTitle: gate.marketTitle,
    baselineObservationIso: gate.baselineObservationIso,
    gateComputedAt: gate.computedAt,
    narrowValidationVerdict: gate.narrowValidationVerdict,
    executionGateVerdict: gate.executionGateVerdict,
    tierUsed: TIER_USED,
    rawAnchorGap: gate.rawAnchorGap,
    anchorPriceObserved: gate.anchorPriceObserved,
    polymarketPriceObserved: gate.polymarketPriceObserved,
    refinedFairValue: gate.refinedFairValue,
    clobBestBid: gate.clobBestBid,
    clobBestAsk: gate.clobBestAsk,
    clobMid: gate.clobMid,
    clobSpreadUsed: gate.clobSpreadUsed,
    clobBidLevels: gate.clobBidLevels,
    clobAskLevels: gate.clobAskLevels,
    entryDepthAvailable: gate.entryDepthAvailable,
    exitDepthAvailable: gate.exitDepthAvailable,
    estimatedNetBeforeExecution: gate.baselineNet,
    estimatedEntryCost: gate.baselineCosts.estimatedEntryCost,
    estimatedExitCost: gate.baselineCosts.estimatedExitCost,
    estimatedEntrySlippage: entrySlip,
    estimatedExitSlippage: exitSlip,
    estimatedHedgeCost: gate.baselineCosts.estimatedHedgeCost,
    estimatedLegRiskCost: gate.baselineCosts.estimatedLegRiskCost,
    estimatedNetAfterPaperExecution: netAfter,
    microstructureObservationState: gate.microstructureObservationState,
    microstructureDetail: gate.microstructureDetail,
    clobBookStructure: gate.clobBookStructure,
    paperExecutionVerdict,
    supportingNote,
  };

  const fp = historyPath();
  const dir = path.dirname(fp);
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(fp, `${JSON.stringify(row)}\n`, { encoding: "utf8" });

  const intervalMs = parseIntervalMs();

  const requestedMode = parseCrossVenueAnchor1823789ExecutionModeFromEnv();
  const halt = parseCrossVenueAnchor1823789MicroLiveHaltFromEnv();
  const effectiveMode = resolveEffectiveCrossVenueAnchor1823789ExecutionMode(requestedMode, halt);

  /**
   * Shadow antes do sync operacional: sync pode falhar em históricos grandes e não pode impedir o audit.
   * Com `minimal_micro_live` e halt desligado: uma linha por ciclo, inclusindo blocked_by_gate.
   */
  if (requestedMode === "minimal_micro_live" && !halt) {
    const capParse = parseCrossVenueAnchor1823789MaxNotionalUsd();
    let targetLiveOrder: CrossVenueAnchor1823789ShadowTargetLiveOrder | null = null;
    let shadowSizingNote = "";

    if (effectiveMode !== "minimal_micro_live") {
      shadowSizingNote = "effective_mode_controlled_paper_due_to_micro_live_halt";
    } else if (!capParse.ok) {
      shadowSizingNote = capParse.reason;
    } else {
      targetLiveOrder = buildCrossVenueAnchor1823789ShadowTargetLiveOrder({
        gate,
        gatesPass: ok,
        maxNotionalUsd: capParse.maxNotionalUsd,
      });
      if (ok && !targetLiveOrder) {
        shadowSizingNote = "no_target_after_gates_flat_or_zero_size";
      }
    }

    const shadowDecision = !ok
      ? ("blocked_by_gate" as const)
      : targetLiveOrder
        ? ("gate_pass_shadow_target" as const)
        : ("gate_pass_no_target" as const);

    const shadowRow = {
      probeVersion: SHADOW_MICRO_LIVE_PROBE_VERSION,
      isoTimestamp,
      experimentId: "cross_venue_anchor_1823789" as const,
      executionMode: "minimal_micro_live" as const,
      realSubmission: false,
      wouldSubmit: false,
      shadowDecision,
      blockReason: !ok
        ? {
            paperExecutionVerdict: "blocked_by_gate" as const,
            narrowValidationVerdict: gate.narrowValidationVerdict,
            executionGateVerdict: gate.executionGateVerdict,
            executionGateSummaryLine: gate.executionGateSummaryLine,
            gatesPass: ok,
          }
        : undefined,
      executionModeRequested: requestedMode,
      microLiveHaltEnv: halt,
      effectiveExecutionMode: effectiveMode,
      tierFixed: TIER_USED,
      maxNotionalUsd: capParse.ok ? capParse.maxNotionalUsd : null,
      maxNotionalUsdParse: capParse,
      gatesPass: ok,
      gateDigest: gate,
      shadowRealSubmission: false as const,
      shadowSubmissionPolicy: "shadow_only_no_api_submit_v1" as const,
      targetLiveOrder,
      shadowSizingNote,
    };

    const sfp = shadowAuditHistoryPath();
    const sdir = path.dirname(sfp);
    fs.mkdirSync(sdir, { recursive: true });
    try {
      assertShadowOnlyNoRealSubmission(shadowRow);
      fs.appendFileSync(sfp, `${JSON.stringify(shadowRow)}\n`, { encoding: "utf8" });
      if (!ok) {
        console.log(`[1823789-paper-exec] shadow audit appended blocked_by_gate → ${sfp}`);
      } else {
        console.log(`[1823789-paper-exec] shadow audit appended → ${sfp}`);
      }
    } catch (serErr) {
      const fallback = {
        probeVersion: SHADOW_MICRO_LIVE_PROBE_VERSION,
        isoTimestamp,
        experimentId: "cross_venue_anchor_1823789" as const,
        executionMode: "minimal_micro_live" as const,
        realSubmission: false,
        wouldSubmit: false,
        shadowDecision: !ok ? "blocked_by_gate" : "unknown",
        shadowAuditWriteError: serErr instanceof Error ? serErr.message : String(serErr),
        gatesPass: ok,
        narrowValidationVerdict: gate.narrowValidationVerdict,
        executionGateVerdict: gate.executionGateVerdict,
        executionGateSummaryLine: gate.executionGateSummaryLine,
        paperExecutionVerdict,
        shadowRealSubmission: false as const,
      };
      assertShadowOnlyNoRealSubmission(fallback);
      fs.appendFileSync(sfp, `${JSON.stringify(fallback)}\n`, { encoding: "utf8" });
      console.error("[1823789-paper-exec] shadow audit fallback line (stringify failed) →", sfp);
    }
  }

  try {
    syncLiveExperimentOperationalMeta({
      experimentId: "cross_venue_anchor_1823789",
      serviceName: "cross-venue-anchor-1823789-paper-execution",
      metaFilename: PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789PaperExecutionMeta,
      historyFilename: PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789PaperExecutionHistory,
      metaEnvOverride: process.env.CROSS_VENUE_ANCHOR_1823789_PAPER_EXECUTION_META_PATH,
      intervalMs,
    });
  } catch (e) {
    console.error(
      "[1823789-paper-exec] operational meta sync error (paper + shadow already written):",
      e instanceof Error ? e.message : e,
    );
  }

  console.log(`[1823789-paper-exec] appended ${isoTimestamp} verdict=${paperExecutionVerdict} → ${fp}`);
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
    serviceName: "cross-venue-anchor-1823789-paper-execution",
    metaFilename: PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789PaperExecutionMeta,
    historyFilename: PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789PaperExecutionHistory,
    metaEnvOverride: process.env.CROSS_VENUE_ANCHOR_1823789_PAPER_EXECUTION_META_PATH,
    intervalMs,
  });
  const reqMode = parseCrossVenueAnchor1823789ExecutionModeFromEnv();
  const haltOn = parseCrossVenueAnchor1823789MicroLiveHaltFromEnv();
  const effMode = resolveEffectiveCrossVenueAnchor1823789ExecutionMode(reqMode, haltOn);
  console.log(
    `[1823789-paper-exec] PAPER_STATE_DIR=${dir} history=${historyPath()} intervalMs=${once ? "N/A (--once)" : String(intervalMs)} executionMode=${reqMode} halt=${haltOn} effectiveMode=${effMode} shadowAudit=${reqMode === "minimal_micro_live" && !haltOn ? shadowAuditHistoryPath() : "n/a"}`,
  );

  if (once) {
    await appendCycle();
    return;
  }

  for (;;) {
    try {
      await appendCycle();
    } catch (e) {
      console.error("[1823789-paper-exec] cycle error:", e instanceof Error ? e.message : e);
    }
    await sleep(intervalMs);
  }
}

main().catch(e => {
  console.error("[1823789-paper-exec] fatal:", e);
  process.exit(1);
});
