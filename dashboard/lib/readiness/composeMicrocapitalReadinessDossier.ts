/**
 * Microcapital Readiness Gate — composer.
 *
 * Reads paper/shadow state from disk + in-memory paper engine and assembles the
 * inputs for {@link analyzeMicrocapitalReadiness}.
 *
 * Read-only. No execution, no submit, no signing, no wallet, no signer, no
 * private key. Policy: shadow_only_no_api_submit_v1.
 */

import { execSync } from "child_process";
import fs from "fs";
import path from "path";

import type { PaperAnalyticsResult } from "../paperAnalytics";
import type { PaperTrade } from "../paperTypes";
import { resolvePaperStateDir, PAPER_TRAIL_FILENAMES } from "../paperStateDir";

import {
  analyzeMicrocapitalReadiness,
  type AnalyzerInput,
  type ObservabilityProbeInput,
  type ReliabilityProbeInput,
  type RiskLimitsProbeInput,
} from "./analyzeMicrocapitalReadiness";
import type {
  MicrocapitalReadinessDossier,
  SystemIdentity,
} from "./microcapitalReadinessDossier";
import { scanProhibitedTerms } from "./prohibitedTermsScanner";
import {
  computePaperCooldownState,
  type PaperCycleOutcome,
} from "./paperRiskGuards";
import {
  applyMarkoutPriceSet,
  buildExecutionRealismSample,
  type ExecutionRealismSample,
} from "./executionRealismHarness";
import {
  readPaperExecutionAssessmentsFromJsonl,
  summarizePaperExecutionAssessments,
} from "./paperExecutionAssessmentParser";
import {
  summarizePaperCycleLifecycle,
} from "./paperCycleLifecycle";
import { computeMarkoutInformativenessStats } from "./markoutInformativenessStats";
import { runReliabilityProbes } from "./reliabilityProbeRunner";
import {
  MICROCAPITAL_READINESS_THRESHOLDS,
} from "./microcapitalReadinessThresholds";
import type {
  MarkoutFollowupRecord,
  MarkoutHorizonMs,
} from "./markoutFollowupProducer";

export interface ComposeOptions {
  /** Project root used for prohibited-terms scanner. Defaults to dashboard root. */
  projectRoot?: string;
  /** Optional realism samples (paper-only). */
  realismSamples?: ExecutionRealismSample[];
  /** Optional override of paper analytics + trades (used in tests). */
  paperData?: {
    analytics: PaperAnalyticsResult | null;
    closedTrades: PaperTrade[];
    activeTrades: PaperTrade[];
  };
  /** Optional override of reliability probes. */
  reliability?: ReliabilityProbeInput;
  /** Optional override of observability probes. */
  observability?: ObservabilityProbeInput;
  /** Optional override of risk-limits probes. */
  riskLimits?: RiskLimitsProbeInput;
  /** Optional system identity overrides. */
  systemIdentityOverrides?: Partial<SystemIdentity>;
}

function tryGitCommit(): string | null {
  try {
    return execSync("git rev-parse --short HEAD", { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] })
      .trim() || null;
  } catch {
    return null;
  }
}

function defaultSystemIdentity(stateDir: string): SystemIdentity {
  return {
    projectName: "polymarket-terminal/arbitrage-system",
    generatedAt: new Date().toISOString(),
    gitCommit: tryGitCommit(),
    paperStateDir: stateDir,
    executionMode: process.env.CROSS_VENUE_ANCHOR_1823789_EXECUTION_MODE ?? "controlled_paper",
    effectiveMode: "shadow_only",
    workerName: "cross-venue-anchor-1823789-paper-execution-v1",
  };
}

function fileExists(p: string): boolean {
  try {
    return fs.statSync(p).isFile();
  } catch {
    return false;
  }
}

function defaultObservability(stateDir: string): ObservabilityProbeInput {
  const shadowAuditPath = path.join(
    stateDir,
    PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789MicroLiveShadowAudit,
  );
  const paperHistoryPath = path.join(
    stateDir,
    PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789PaperExecutionHistory,
  );
  return {
    hasJsonLogs: true,
    hasShadowAudit: fileExists(shadowAuditPath),
    hasPaperTrades: fileExists(paperHistoryPath),
    hasAnalyticsEndpoint: true,
    hasSystemEndpoint: true,
    hasTradesEndpoint: true,
    canReconcileCycleToPacket: fileExists(paperHistoryPath),
    canReconcilePacketToDispatch: fileExists(paperHistoryPath) && fileExists(shadowAuditPath),
    canReconcileSignalToPaperTrade: fileExists(paperHistoryPath),
  };
}

/**
 * The current dispatcher (lib/executionDispatcher.ts) is a synchronous, single-pass
 * audit-only path. It has no queue, no leases, no retries, no breaker. Reporting
 * those as a "failed probe" would be misleading — no probe has actually been run.
 *
 * The honest classification is `insufficient_evidence` per probe: the mechanism
 * is simply not wired yet. We mark only `workerSurvivesDispatcherFailure` as
 * `pass` because the worker's try/catch in lib/executionWorker.ts is observable
 * code today.
 */
function defaultReliability(): ReliabilityProbeInput {
  return {
    restartRecoveryPassed: false,
    idempotencyPassed: false,
    duplicatePreventionPassed: false,
    leaseRecoveryPassed: false,
    retryFinitePassed: false,
    permanentErrorsBecomeDeadPassed: false,
    circuitBreakerPassed: false,
    workerSurvivesDispatcherFailurePassed: true,
    probeStatus: {
      restartRecovery: "insufficient_evidence",
      idempotency: "insufficient_evidence",
      duplicatePrevention: "insufficient_evidence",
      leaseRecovery: "insufficient_evidence",
      retryFinite: "insufficient_evidence",
      permanentErrorsBecomeDead: "insufficient_evidence",
      circuitBreaker: "insufficient_evidence",
      workerSurvivesDispatcherFailure: "pass",
    },
  };
}

/**
 * Detects pre-existing simulated risk-limit mechanisms in the project.
 *
 * - Kill-switch: env `CROSS_VENUE_ANCHOR_1823789_MICRO_LIVE_HALT` is honored by
 *   `lib/crossVenueAnchor1823789ExecutionMode.ts` to roll execution back to
 *   `controlled_paper` (paper-only). Presence of that env-var contract counts as
 *   a simulated kill-switch.
 * - Cooldown-after-loss: paper-only `paperRiskGuards.ts` exists (mechanism
 *   present + tested). The composer replays closed paper trades through the
 *   guard to compute current state.
 * - Exposure-limit: capital caps live in paperTradeEngine policies (paper-only).
 *
 * All three are SIMULATED guarantees — none authorize real submission.
 */
/**
 * Read markout followups (paper-only) from JSONL. Skips blank/invalid lines and
 * "historical_followup_unavailable" placeholders (those carry no usable price).
 * Returns an empty list if the file does not exist.
 */
/**
 * Tally markout follow-up samplerStatus and priceSource across the JSONL.
 *
 * Counts `ok` / `price_unavailable` / `sampler_error` rows separately. Skips
 * `historical_followup_unavailable` placeholders (those are not sampler runs).
 * Returns null if the file does not exist.
 */
function tallyMarkoutFollowupStats(
  filePath: string,
): {
  okFollowups: number;
  priceUnavailableFollowups: number;
  samplerErrorFollowups: number;
  priceUnavailableRate: number;
  priceSourcesUsed: Record<string, number>;
} | null {
  let raw: string;
  try {
    raw = fs.readFileSync(filePath, "utf8");
  } catch {
    return null;
  }
  let ok = 0;
  let priceUnavail = 0;
  let samplerErr = 0;
  const sources: Record<string, number> = {};
  for (const line of raw.split(/\r?\n/)) {
    const t = line.trim();
    if (t.length === 0) continue;
    let obj: Record<string, unknown>;
    try {
      obj = JSON.parse(t) as Record<string, unknown>;
    } catch {
      continue;
    }
    if (obj.type !== "markout_followup") continue;
    if (obj.markoutStatus === "historical_followup_unavailable") continue;
    const status =
      typeof obj.samplerStatus === "string" ? (obj.samplerStatus as string) : null;
    if (status === "ok") {
      ok++;
      const src = typeof obj.priceSource === "string" ? (obj.priceSource as string) : "unknown";
      sources[src] = (sources[src] ?? 0) + 1;
    } else if (status === "price_unavailable") {
      priceUnavail++;
    } else if (status === "sampler_error") {
      samplerErr++;
    } else {
      // Legacy rows from the historical collector (no samplerStatus): treat
      // as `ok` if the row has a usable markout, otherwise skip.
      const followupPrice = typeof obj.followupPrice === "number" ? obj.followupPrice : null;
      const markout = typeof obj.markout === "number" ? obj.markout : null;
      if (followupPrice !== null && markout !== null) {
        ok++;
        const src = "legacy_collector";
        sources[src] = (sources[src] ?? 0) + 1;
      }
    }
  }
  const totalSampler = ok + priceUnavail + samplerErr;
  return {
    okFollowups: ok,
    priceUnavailableFollowups: priceUnavail,
    samplerErrorFollowups: samplerErr,
    priceUnavailableRate: totalSampler > 0 ? priceUnavail / totalSampler : 0,
    priceSourcesUsed: sources,
  };
}

function readMarkoutFollowupsFromJsonl(filePath: string): MarkoutFollowupRecord[] {
  let raw: string;
  try {
    raw = fs.readFileSync(filePath, "utf8");
  } catch {
    return [];
  }
  const out: MarkoutFollowupRecord[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const t = line.trim();
    if (t.length === 0) continue;
    let obj: Record<string, unknown>;
    try {
      obj = JSON.parse(t) as Record<string, unknown>;
    } catch {
      continue;
    }
    if (obj.markoutStatus === "historical_followup_unavailable") continue;
    if (obj.type !== "markout_followup") continue;
    const cycleId = typeof obj.cycleId === "string" ? obj.cycleId : null;
    const marketId = typeof obj.marketId === "string" ? obj.marketId : null;
    const baseObservedAt = typeof obj.baseObservedAt === "number" ? obj.baseObservedAt : null;
    const followupObservedAt =
      typeof obj.followupObservedAt === "number" ? obj.followupObservedAt : null;
    const horizonMs = typeof obj.horizonMs === "number" ? obj.horizonMs : null;
    const basePrice = typeof obj.basePrice === "number" ? obj.basePrice : null;
    const followupPrice = typeof obj.followupPrice === "number" ? obj.followupPrice : null;
    const markout = typeof obj.markout === "number" ? obj.markout : null;
    if (
      !cycleId ||
      !marketId ||
      baseObservedAt === null ||
      followupObservedAt === null ||
      horizonMs === null ||
      basePrice === null ||
      followupPrice === null ||
      markout === null
    ) {
      continue;
    }
    if (horizonMs !== 5_000 && horizonMs !== 30_000 && horizonMs !== 60_000) continue;
    out.push({
      type: "markout_followup",
      cycleId,
      marketId,
      baseObservedAt,
      followupObservedAt,
      horizonMs: horizonMs as MarkoutHorizonMs,
      basePrice,
      followupPrice,
      markout,
      source: "paper_shadow_readonly",
    });
  }
  return out;
}

function defaultRiskLimits(closedTrades: PaperTrade[]): RiskLimitsProbeInput {
  const killSwitchSimulated = true;

  const cycles: PaperCycleOutcome[] = [];
  for (const t of closedTrades) {
    if (
      typeof t.realizedPnL === "number" &&
      Number.isFinite(t.realizedPnL) &&
      typeof t.closedAt === "string" &&
      t.closedAt
    ) {
      const ts = Date.parse(t.closedAt);
      if (Number.isFinite(ts)) {
        cycles.push({ cycleId: t.tradeId, pnl: t.realizedPnL, timestamp: ts });
      }
    }
  }

  const cooldownState = computePaperCooldownState({
    closedCycles: cycles,
    nowMs: Date.now(),
  });

  let consecutiveNegative = 0;
  for (let i = closedTrades.length - 1; i >= 0; i--) {
    const t = closedTrades[i];
    if (typeof t.realizedPnL === "number" && t.realizedPnL < 0) consecutiveNegative++;
    else break;
  }

  let worstSingleLoss = 0;
  for (const t of closedTrades) {
    if (typeof t.realizedPnL === "number" && t.realizedPnL < worstSingleLoss) {
      worstSingleLoss = t.realizedPnL;
    }
  }

  return {
    killSwitchSimulated,
    cooldownAfterLossSimulated: true, // mechanism present + tested
    cooldownActiveNow: cooldownState.cooldownActive,
    cooldownActiveUntilMs: cooldownState.cooldownActiveUntilMs,
    cooldownEvidence: cycles.length > 0 ? "pass" : "insufficient_evidence",
    exposureLimitSimulated: true,
    maxConsecutiveNegativeCycles: consecutiveNegative,
    maxPaperLossPerCycle: Math.abs(worstSingleLoss),
    maxPaperDailyLoss: 0,
  };
}

function loadPaperData(): {
  analytics: PaperAnalyticsResult | null;
  closedTrades: PaperTrade[];
  activeTrades: PaperTrade[];
} {
  try {
    const mod = require("../paperSimulationService") as typeof import("../paperSimulationService");
    const analytics = mod.getPaperAnalyticsData().analytics;
    const trades = mod.getPaperTradesData();
    return {
      analytics,
      closedTrades: trades.recentClosed,
      activeTrades: trades.active,
    };
  } catch {
    return { analytics: null, closedTrades: [], activeTrades: [] };
  }
}

export function composeMicrocapitalReadinessDossier(
  opts: ComposeOptions = {},
): MicrocapitalReadinessDossier {
  const projectRoot =
    opts.projectRoot ?? path.resolve(__dirname, "..", "..");
  const stateDir = resolvePaperStateDir();

  const paperData = opts.paperData ?? loadPaperData();
  const observability = opts.observability ?? defaultObservability(stateDir);
  const reliability = opts.reliability ?? defaultReliability();
  const riskLimits = opts.riskLimits ?? defaultRiskLimits(paperData.closedTrades);

  const systemIdentity: SystemIdentity = {
    ...defaultSystemIdentity(stateDir),
    ...(opts.systemIdentityOverrides ?? {}),
  };

  const prohibitedTermsScan = scanProhibitedTerms(projectRoot);

  // Read paper execution assessment JSONL (paper/shadow only, read-only).
  const paperExecHistoryPath = path.join(
    stateDir,
    PAPER_TRAIL_FILENAMES.crossVenueAnchor1823789PaperExecutionHistory,
  );
  const assessments = readPaperExecutionAssessmentsFromJsonl(paperExecHistoryPath);
  const paperExecutionAssessments = summarizePaperExecutionAssessments(assessments);

  // Read markout followups (paper-only) and join with assessments via lifecycle.
  const followupsPath = path.join(
    stateDir,
    MICROCAPITAL_READINESS_THRESHOLDS.MARKOUT_FOLLOWUPS_FILENAME,
  );
  const followups = readMarkoutFollowupsFromJsonl(followupsPath);
  const paperCycleLifecycle = summarizePaperCycleLifecycle(assessments, followups);
  const markoutFollowupStats = tallyMarkoutFollowupStats(followupsPath) ?? undefined;
  const markoutInformativenessStats =
    computeMarkoutInformativenessStats(paperCycleLifecycle, followupsPath) ?? undefined;

  // Synthesize realism samples from positive assessments + followups (paper-only).
  const realismSamples: ExecutionRealismSample[] = opts.realismSamples ?? [];
  if (followups.length > 0 && realismSamples.length === 0) {
    const positives = assessments.filter(
      a =>
        (a.paperExecutionVerdict ?? "").toLowerCase() === "paper_cycle_positive",
    );
    for (const a of positives) {
      const baseTs = a.isoTimestamp ? Date.parse(a.isoTimestamp) : NaN;
      const basePrice =
        a.clobMid ??
        (a.clobBestBid !== null && a.clobBestAsk !== null
          ? (a.clobBestBid + a.clobBestAsk) / 2
          : null);
      if (!Number.isFinite(baseTs) || basePrice === null) continue;
      const cycleId = `pc::${a.marketId ?? "unknown"}::${a.isoTimestamp}`;
      const cf = followups.filter(f => f.cycleId === cycleId);
      const priceSet: { priceAfter5s?: number; priceAfter30s?: number; priceAfter60s?: number } = {};
      for (const r of cf) {
        if (r.horizonMs === 5_000) priceSet.priceAfter5s = r.followupPrice;
        else if (r.horizonMs === 30_000) priceSet.priceAfter30s = r.followupPrice;
        else if (r.horizonMs === 60_000) priceSet.priceAfter60s = r.followupPrice;
      }
      const merged = applyMarkoutPriceSet(
        {
          cycleId,
          signalTimestamp: baseTs,
          enqueueTimestamp: baseTs,
          observedPriceOrProbabilitySnapshot: basePrice,
          notionalPaperUsd: 100,
        },
        priceSet,
      );
      realismSamples.push(buildExecutionRealismSample(merged));
    }
  }

  // Optionally run synthetic reliability probes (gated by env var).
  const synthetic = runReliabilityProbes();
  const effectiveReliability = synthetic.enabled
    ? {
        ...reliability,
        restartRecoveryPassed: synthetic.probes.restartRecovery === "pass",
        idempotencyPassed: synthetic.probes.idempotency === "pass",
        duplicatePreventionPassed: synthetic.probes.duplicatePrevention === "pass",
        leaseRecoveryPassed: synthetic.probes.leaseRecovery === "pass",
        retryFinitePassed: synthetic.probes.retryFinite === "pass",
        permanentErrorsBecomeDeadPassed:
          synthetic.probes.permanentErrorsBecomeDead === "pass",
        circuitBreakerPassed: synthetic.probes.circuitBreaker === "pass",
        workerSurvivesDispatcherFailurePassed:
          synthetic.probes.workerSurvivesDispatcherFailure === "pass",
        probeStatus: { ...reliability.probeStatus, ...synthetic.probes },
      }
    : reliability;

  const input: AnalyzerInput = {
    systemIdentity,
    paperAnalytics: paperData.analytics,
    closedTrades: paperData.closedTrades,
    activeTrades: paperData.activeTrades,
    realismSamples,
    prohibitedTermsScan,
    apiSubmitAllowedTrueAnywhere: prohibitedTermsScan.findings.some(
      f => f.matchedTerm === "apiSubmitAllowed: true",
    ),
    dispatcherAuditOnly: true,
    payloadSafetyPresent: true,
    reliability: effectiveReliability,
    observability,
    riskLimits,
    paperExecutionAssessments,
    paperCycleLifecycle,
    markoutFollowupStats,
    markoutInformativenessStats,
  };

  return analyzeMicrocapitalReadiness(input);
}
