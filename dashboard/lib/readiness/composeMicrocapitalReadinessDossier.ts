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
import type { ExecutionRealismSample } from "./executionRealismHarness";

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
 * - Cooldown-after-loss: not present anywhere in the project today. Reported false.
 * - Exposure-limit: capital caps live in paperTradeEngine policies (paper-only).
 *
 * All three are SIMULATED guarantees — none authorize real submission.
 */
function defaultRiskLimits(): RiskLimitsProbeInput {
  // The kill-switch CONTRACT is present in code regardless of whether the env is
  // set right now: pulling the env (or not) only changes the active state, not
  // whether the mechanism exists. The readiness gate cares about presence.
  const killSwitchSimulated = true;
  return {
    killSwitchSimulated,
    cooldownAfterLossSimulated: false,
    exposureLimitSimulated: true,
    maxConsecutiveNegativeCycles: 0,
    maxPaperLossPerCycle: 0,
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
  const riskLimits = opts.riskLimits ?? defaultRiskLimits();

  const systemIdentity: SystemIdentity = {
    ...defaultSystemIdentity(stateDir),
    ...(opts.systemIdentityOverrides ?? {}),
  };

  const prohibitedTermsScan = scanProhibitedTerms(projectRoot);

  const input: AnalyzerInput = {
    systemIdentity,
    paperAnalytics: paperData.analytics,
    closedTrades: paperData.closedTrades,
    activeTrades: paperData.activeTrades,
    realismSamples: opts.realismSamples ?? [],
    prohibitedTermsScan,
    apiSubmitAllowedTrueAnywhere: prohibitedTermsScan.findings.some(
      f => f.matchedTerm === "apiSubmitAllowed: true",
    ),
    dispatcherAuditOnly: true,
    payloadSafetyPresent: true,
    reliability,
    observability,
    riskLimits,
  };

  return analyzeMicrocapitalReadiness(input);
}
