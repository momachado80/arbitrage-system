/**
 * Pocket execution probe — observação de executabilidade mínima em micro-pockets
 * já estáveis + promotionAssessment ≠ not_ready. Sem fills agressivos, sem PnL, sem
 * integração com execution engine ou paper portfolio.
 */

import fs from "fs";
import path from "path";
import {
  buildPocketEconomicsDigest,
  snapshotPocketFamilyEligibleBucketsForExecutionProbe,
  getMicroBucketEligibleStreakFromEconomics,
} from "./pocketEconomicsProbe";
import type { PocketEconomicsMarketRow, PocketEconomicsPromotionVerdict } from "./pocketEconomicsProbe";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES } from "./paperStateDir";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}

const SCAN_INTERVAL_MS = () => envNum("POCKET_EXEC_SCAN_INTERVAL_MS", 6 * 3600_000);
const SNAPSHOT_GAP_MS = () => envNum("POCKET_EXEC_SNAPSHOT_GAP_MS", 3000);
const BOOT_DELAY_MS = () => envNum("POCKET_EXEC_BOOT_DELAY_MS", 8000);
const MAX_PRICE_JUMP = () => envNum("POCKET_EXEC_MAX_PRICE_JUMP", 0.07);
const MAX_SPREAD_DRIFT = () => envNum("POCKET_EXEC_MAX_SPREAD_DRIFT", 0.06);
const SPREAD_CAP = 0.15;
const MIN_LIQ = 500;

const EXEC_HISTORY_MAX = () => envNum("POCKET_EXEC_HISTORY_MAX", 30);
const PERSIST_THROTTLE_MS = () => envNum("POCKET_EXEC_PERSIST_THROTTLE_MS", 30_000);
const PERSIST_VERSION = 1 as const;

/** Gates explícitos para “merece paper execution probe mínimo?” — só leitura temporal/trajectory; não cria paper. */
const EXEC_PROMO_MIN_TEMPORAL_CYCLES = () => envNum("POCKET_EXEC_EXEC_PROMO_MIN_TEMPORAL_CYCLES", 6);
const EXEC_PROMO_MIN_CONSEC_OBS_OR_BETTER = () =>
  envNum("POCKET_EXEC_EXEC_PROMO_MIN_CONSEC_OBSERVATION_OR_BETTER", 4);
const EXEC_PROMO_MIN_CONSEC_MINIMAL = () =>
  envNum("POCKET_EXEC_EXEC_PROMO_MIN_CONSEC_MINIMAL_SIGNAL", 2);
const EXEC_PROMO_MIN_STABLE_EXEC_KEYS = () =>
  envNum("POCKET_EXEC_EXEC_PROMO_MIN_STABLE_EXEC_CANDIDATES", 1);
const EXEC_PROMO_MAX_DEGRADED = () => envNum("POCKET_EXEC_EXEC_PROMO_MAX_DEGRADED_CANDIDATES", 1);

const TARGET_FAMILY_KEY = "other:price_above:>3m" as const;
const GLOBAL_KEY = "__pocketExecutionProbe_v1";
const PERSIST_META_GLOBAL_KEY = "__pocketExecutionPersistMeta_v1";

export type PocketExecutionPersistenceLoadStatus =
  | "ok"
  | "missing"
  | "invalid"
  | "error"
  | "disabled"
  | "pending";

export interface PocketExecutionPersistenceDigest {
  persistenceEnabled: boolean;
  persistencePath: string | null;
  lastPersistenceWriteAt: string | null;
  lastPersistenceLoadAt: string | null;
  persistenceLoadStatus: PocketExecutionPersistenceLoadStatus;
}

export type ExecutionObservationVerdict = "blocked" | "observation_only" | "minimal_executable_signal";

export interface PocketExecutionVerdictHistoryEntry {
  scannedAt: string;
  executionObservationVerdict: ExecutionObservationVerdict;
  executionReadinessScore: number | null;
  aggregateSpreadFeasibility: number | null;
  aggregateEntrySnapshotQuality: number | null;
  candidateCount: number;
  promotionVerdictAtScan: PocketEconomicsPromotionVerdict;
  blockingReasonsShort: string[];
  supportingReasonsShort: string[];
}

export type PocketExecutionReadinessTrend = "up" | "down" | "flat" | "unknown";
export type PocketExecutionSpreadTrend = "up" | "down" | "flat" | "unknown";

export type ExecutionTrajectoryAssessment =
  | "stagnant"
  | "improving_but_not_ready"
  | "unstable"
  | "consistently_blocked"
  | "consistently_minimal_executable_signal";

export interface PocketExecutionTemporalDigest {
  verdictHistory: PocketExecutionVerdictHistoryEntry[];
  consecutiveObservationOnlyCount: number;
  consecutiveMinimalExecutableSignalCount: number;
  consecutiveBlockedCount: number;
  readinessTrend: PocketExecutionReadinessTrend;
  spreadTrend: PocketExecutionSpreadTrend;
  stableExecutionCandidates: string[];
  degradedExecutionCandidates: string[];
  improvingExecutionCandidates: string[];
}

export interface PocketExecutionTrajectoryDigest {
  executionTrajectoryAssessment: ExecutionTrajectoryAssessment;
  trajectoryNotes: string[];
}

export type OverallExecutionPromotionVerdict =
  | "not_ready"
  | "borderline"
  | "ready_for_minimal_paper_execution_probe";

export interface ExecutionPromotionThresholdsDigest {
  minTemporalCycles: number;
  minConsecutiveObservationOrBetter: number;
  minConsecutiveMinimalExecutableSignal: number;
  minStableExecutionCandidates: number;
  maxDegradedExecutionCandidates: number;
}

export interface ExecutionPromotionAssessmentDigest {
  minimumTemporalCyclesSatisfied: boolean;
  minimumConsecutiveObservationOrBetterSatisfied: boolean;
  minimumConsecutiveMinimalExecutableSignalSatisfied: boolean;
  readinessTrendSatisfied: boolean;
  spreadTrendSatisfied: boolean;
  stableExecutionCandidatesSatisfied: boolean;
  degradedExecutionCandidatesAcceptable: boolean;
  executionTrajectoryAssessmentSupportive: boolean;
  overallExecutionPromotionVerdict: OverallExecutionPromotionVerdict;
  promotionReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: ExecutionPromotionThresholdsDigest;
}

export interface ExecutionProbeMarketObservation {
  id: string;
  question: string;
  liquidity: number;
  spread: number;
  prices: number[];
  outcomes: string[];
  priceJumpMaxVsPriorSnapshot: number | null;
  spreadDeltaVsPriorSnapshot: number | null;
}

export interface CandidateExecutionPocket {
  microBucketKey: string;
  eligibleMarketCount: number;
  components: ExecutionProbeMarketObservation[];
  pocketPersistenceAtExecution: number;
  entrySnapshotQuality: number | null;
  spreadFeasibility: number | null;
  betweenSnapshotsStable: boolean;
  priceJumpMaxInPocket: number | null;
  spreadDriftMaxInPocket: number | null;
  simulatedEntryWindows: string[];
  simulatedExitWindows: string[];
  pocketSupportingReasons: string[];
  pocketBlockingReasons: string[];
}

export interface PocketExecutionLastRun {
  candidateExecutionPockets: CandidateExecutionPocket[];
  executionReadinessScore: number | null;
  aggregateEntrySnapshotQuality: number | null;
  aggregateSpreadFeasibility: number | null;
  pocketPersistenceAtExecutionSummary: string | null;
  simulatedEntryWindows: string[];
  simulatedExitWindows: string[];
  executionObservationVerdict: ExecutionObservationVerdict;
  blockingReasons: string[];
  supportingReasons: string[];
  promotionVerdictAtScan: PocketEconomicsPromotionVerdict;
  stableMicroKeysAtScan: string[];
  snapshotGapMs: number;
  firstSnapshotAt: number | null;
  secondSnapshotAt: number | null;
  totalMarketsScannedFirst: number;
  totalMarketsScannedSecond: number;
}

export interface PocketExecutionDigest {
  computedAt: string;
  probeVersion: "pocket-execution-v1";
  scanStatus: "completed" | "scanning" | "idle" | "error";
  targetFamilyKey: typeof TARGET_FAMILY_KEY;
  note: string;
  promotionVerdictAtScan: PocketEconomicsPromotionVerdict | null;
  stableMicroKeysAtScan: string[];
  candidateExecutionPockets: CandidateExecutionPocket[];
  executionReadinessScore: number | null;
  aggregateEntrySnapshotQuality: number | null;
  aggregateSpreadFeasibility: number | null;
  pocketPersistenceAtExecutionSummary: string | null;
  simulatedEntryWindows: string[];
  simulatedExitWindows: string[];
  executionObservationVerdict: ExecutionObservationVerdict;
  blockingReasons: string[];
  supportingReasons: string[];
  snapshotGapMs: number;
  firstSnapshotAt: string | null;
  secondSnapshotAt: string | null;
  totalMarketsScannedFirst: number;
  totalMarketsScannedSecond: number;
  lastScanStartAt: string | null;
  lastScanEndAt: string | null;
  lastSuccessfulScanAt: string | null;
  lastScanErrorAt: string | null;
  lastScanErrorMessage: string | null;
  isScanRunning: boolean;
  currentRunId: number;
  nextScheduledScanAt: string | null;
  schedulerStartedAt: string | null;
  totalScanAttempts: number;
  totalScanSuccess: number;
  totalScanErrors: number;
  totalScanSkippedBusy: number;
  persistence: PocketExecutionPersistenceDigest;
  temporal: PocketExecutionTemporalDigest;
  trajectory: PocketExecutionTrajectoryDigest;
  executionPromotionAssessment: ExecutionPromotionAssessmentDigest;
}

interface PocketExecutionPocketSummaryPersisted {
  microBucketKey: string;
  pocketReadinessScore: number | null;
  spreadFeasibility: number | null;
  betweenSnapshotsStable: boolean;
  eligibleMarketCount: number;
}

interface TemporalCyclePersisted {
  cycleAt: number;
  verdict: ExecutionObservationVerdict;
  executionReadinessScore: number | null;
  aggregateSpreadFeasibility: number | null;
  aggregateEntrySnapshotQuality: number | null;
  candidateCount: number;
  promotionVerdictAtScan: PocketEconomicsPromotionVerdict;
  blockingReasonsShort: string[];
  supportingReasonsShort: string[];
  pocketReadinessSummaries: PocketExecutionPocketSummaryPersisted[];
  stableMicroKeysCount: number;
}

interface PocketExecutionPersistedFileV1 {
  version: typeof PERSIST_VERSION;
  savedAt: string;
  temporalCycles: TemporalCyclePersisted[];
  currentRunId: number;
  totalScanAttempts: number;
  totalScanSuccess: number;
  totalScanErrors: number;
  totalScanSkippedBusy: number;
  lastScanTimestamp: number | null;
  lastScanDurationMs: number | null;
  lastScanStartAt: number | null;
  lastScanEndAt: number | null;
  lastSuccessfulScanAt: number | null;
  lastScanErrorAt: number | null;
  lastScanErrorMessage: string | null;
}

interface PocketExecutionPersistMeta {
  lastWriteAtMs: number | null;
  lastLoadAtMs: number | null;
  loadStatus: PocketExecutionPersistenceLoadStatus;
  lastThrottleWallMs: number;
}

function getPersistMetaExec(): PocketExecutionPersistMeta {
  const g = globalThis as unknown as Record<string, PocketExecutionPersistMeta | undefined>;
  if (!g[PERSIST_META_GLOBAL_KEY]) {
    g[PERSIST_META_GLOBAL_KEY] = {
      lastWriteAtMs: null,
      lastLoadAtMs: null,
      loadStatus: "pending",
      lastThrottleWallMs: 0,
    };
  }
  return g[PERSIST_META_GLOBAL_KEY]!;
}

function isExecPersistenceDiskDisabled(): boolean {
  return process.env.POCKET_EXEC_DISABLE_DISK === "1";
}

function defaultExecPersistencePath(): string {
  const raw = process.env.POCKET_EXEC_STATE_PATH?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.pocketExecution);
}

function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function summarizeReasonsShort(arr: string[], maxN: number, maxLen: number): string[] {
  return arr.slice(0, maxN).map(s => (s.length > maxLen ? `${s.slice(0, maxLen)}…` : s));
}

function pocketReadinessApprox(c: CandidateExecutionPocket): number | null {
  const base =
    ((c.entrySnapshotQuality ?? 0) + (c.spreadFeasibility ?? 0)) / 2 * (c.betweenSnapshotsStable ? 1 : 0.35);
  return Number.isFinite(base) ? r4(base) : null;
}

function recordTemporalCycle(st: ProbeState, lr: PocketExecutionLastRun): void {
  const pocketReadinessSummaries: PocketExecutionPocketSummaryPersisted[] =
    lr.candidateExecutionPockets.map(c => ({
      microBucketKey:
        c.microBucketKey.length > 400 ? `${c.microBucketKey.slice(0, 400)}…` : c.microBucketKey,
      pocketReadinessScore: pocketReadinessApprox(c),
      spreadFeasibility: c.spreadFeasibility,
      betweenSnapshotsStable: c.betweenSnapshotsStable,
      eligibleMarketCount: c.eligibleMarketCount,
    }));
  const rec: TemporalCyclePersisted = {
    cycleAt: Date.now(),
    verdict: lr.executionObservationVerdict,
    executionReadinessScore: lr.executionReadinessScore,
    aggregateSpreadFeasibility: lr.aggregateSpreadFeasibility,
    aggregateEntrySnapshotQuality: lr.aggregateEntrySnapshotQuality,
    candidateCount: lr.candidateExecutionPockets.length,
    promotionVerdictAtScan: lr.promotionVerdictAtScan,
    blockingReasonsShort: summarizeReasonsShort(lr.blockingReasons, 6, 220),
    supportingReasonsShort: summarizeReasonsShort(lr.supportingReasons, 6, 220),
    pocketReadinessSummaries,
    stableMicroKeysCount: lr.stableMicroKeysAtScan.length,
  };
  st.temporalCycles.push(rec);
  const maxH = EXEC_HISTORY_MAX();
  while (st.temporalCycles.length > maxH) st.temporalCycles.shift();
}

function buildPersistPayloadExecution(st: ProbeState): PocketExecutionPersistedFileV1 {
  return {
    version: PERSIST_VERSION,
    savedAt: new Date().toISOString(),
    temporalCycles: st.temporalCycles.map(c => ({
      ...c,
      pocketReadinessSummaries: c.pocketReadinessSummaries.map(p => ({ ...p })),
      blockingReasonsShort: [...c.blockingReasonsShort],
      supportingReasonsShort: [...c.supportingReasonsShort],
    })),
    currentRunId: st.currentRunId,
    totalScanAttempts: st.totalScanAttempts,
    totalScanSuccess: st.totalScanSuccess,
    totalScanErrors: st.totalScanErrors,
    totalScanSkippedBusy: st.totalScanSkippedBusy,
    lastScanTimestamp: st.lastScanTimestamp,
    lastScanDurationMs: st.lastScanDurationMs,
    lastScanStartAt: st.lastScanStartAt,
    lastScanEndAt: st.lastScanEndAt,
    lastSuccessfulScanAt: st.lastSuccessfulScanAt,
    lastScanErrorAt: st.lastScanErrorAt,
    lastScanErrorMessage: st.lastScanErrorMessage,
  };
}

function writeExecPersistenceAtomic(filePath: string, payload: PocketExecutionPersistedFileV1): void {
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(payload), "utf8");
  fs.renameSync(tmp, filePath);
}

function maybePersistExecutionState(st: ProbeState, forceBecauseScanSuccess: boolean): void {
  if (isExecPersistenceDiskDisabled()) return;
  const meta = getPersistMetaExec();
  const now = Date.now();
  if (!forceBecauseScanSuccess && now - meta.lastThrottleWallMs < PERSIST_THROTTLE_MS()) {
    return;
  }
  try {
    writeExecPersistenceAtomic(defaultExecPersistencePath(), buildPersistPayloadExecution(st));
    meta.lastThrottleWallMs = now;
    meta.lastWriteAtMs = now;
    if (meta.loadStatus === "missing" || meta.loadStatus === "pending") {
      meta.loadStatus = "ok";
    }
  } catch (e) {
    console.warn("[PocketExecution] Persistence write failed (non-fatal):", e instanceof Error ? e.message : e);
  }
}

function sanitizeTemporalCycles(raw: unknown): TemporalCyclePersisted[] {
  if (!Array.isArray(raw)) return [];
  const out: TemporalCyclePersisted[] = [];
  const verdicts: ExecutionObservationVerdict[] = ["blocked", "observation_only", "minimal_executable_signal"];
  const promos: PocketEconomicsPromotionVerdict[] = [
    "not_ready",
    "borderline",
    "ready_for_minimal_execution_probe",
  ];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const o = item as Record<string, unknown>;
    const v = o.verdict;
    if (typeof v !== "string" || !verdicts.includes(v as ExecutionObservationVerdict)) continue;
    const pv = o.promotionVerdictAtScan;
    if (typeof pv !== "string" || !promos.includes(pv as PocketEconomicsPromotionVerdict)) continue;
    const cycleAt = typeof o.cycleAt === "number" && Number.isFinite(o.cycleAt) ? o.cycleAt : Date.now();
    const summariesRaw = o.pocketReadinessSummaries;
    const pocketReadinessSummaries: PocketExecutionPocketSummaryPersisted[] = [];
    if (Array.isArray(summariesRaw)) {
      for (const s of summariesRaw) {
        if (!s || typeof s !== "object") continue;
        const ps = s as Record<string, unknown>;
        const mk = ps.microBucketKey;
        if (typeof mk !== "string" || mk.length < 1 || mk.length > 420) continue;
        const pr = ps.pocketReadinessScore;
        const sf = ps.spreadFeasibility;
        pocketReadinessSummaries.push({
          microBucketKey: mk,
          pocketReadinessScore:
            pr === null || pr === undefined
              ? null
              : Number.isFinite(Number(pr))
                ? r4(Number(pr))
                : null,
          spreadFeasibility:
            sf === null || sf === undefined
              ? null
              : Number.isFinite(Number(sf))
                ? r4(Number(sf))
                : null,
          betweenSnapshotsStable: ps.betweenSnapshotsStable === true,
          eligibleMarketCount:
            typeof ps.eligibleMarketCount === "number" && Number.isFinite(ps.eligibleMarketCount)
              ? Math.max(0, Math.floor(ps.eligibleMarketCount))
              : 0,
        });
      }
    }
    const br = Array.isArray(o.blockingReasonsShort)
      ? o.blockingReasonsShort.filter((x): x is string => typeof x === "string" && x.length < 300).slice(0, 8)
      : [];
    const sr = Array.isArray(o.supportingReasonsShort)
      ? o.supportingReasonsShort.filter((x): x is string => typeof x === "string" && x.length < 300).slice(0, 8)
      : [];
    out.push({
      cycleAt,
      verdict: v as ExecutionObservationVerdict,
      executionReadinessScore:
        o.executionReadinessScore === null || o.executionReadinessScore === undefined
          ? null
          : Number.isFinite(Number(o.executionReadinessScore))
            ? r4(Number(o.executionReadinessScore))
            : null,
      aggregateSpreadFeasibility:
        o.aggregateSpreadFeasibility === null || o.aggregateSpreadFeasibility === undefined
          ? null
          : Number.isFinite(Number(o.aggregateSpreadFeasibility))
            ? r4(Number(o.aggregateSpreadFeasibility))
            : null,
      aggregateEntrySnapshotQuality:
        o.aggregateEntrySnapshotQuality === null || o.aggregateEntrySnapshotQuality === undefined
          ? null
          : Number.isFinite(Number(o.aggregateEntrySnapshotQuality))
            ? r4(Number(o.aggregateEntrySnapshotQuality))
            : null,
      candidateCount:
        typeof o.candidateCount === "number" && Number.isFinite(o.candidateCount)
          ? Math.max(0, Math.floor(o.candidateCount))
          : 0,
      promotionVerdictAtScan: pv as PocketEconomicsPromotionVerdict,
      blockingReasonsShort: br,
      supportingReasonsShort: sr,
      pocketReadinessSummaries,
      stableMicroKeysCount:
        typeof o.stableMicroKeysCount === "number" && Number.isFinite(o.stableMicroKeysCount)
          ? Math.max(0, Math.floor(o.stableMicroKeysCount))
          : 0,
    });
  }
  return out.slice(-EXEC_HISTORY_MAX());
}

function hydrateExecutionFromDisk(st: ProbeState): void {
  const meta = getPersistMetaExec();
  if (isExecPersistenceDiskDisabled()) {
    meta.loadStatus = "disabled";
    meta.lastLoadAtMs = Date.now();
    return;
  }
  const p = defaultExecPersistencePath();
  try {
    if (!fs.existsSync(p)) {
      meta.loadStatus = "missing";
      meta.lastLoadAtMs = Date.now();
      return;
    }
    const j = JSON.parse(fs.readFileSync(p, "utf8")) as Partial<PocketExecutionPersistedFileV1>;
    if (j.version !== PERSIST_VERSION) {
      meta.loadStatus = "invalid";
      meta.lastLoadAtMs = Date.now();
      return;
    }
    st.temporalCycles = sanitizeTemporalCycles(j.temporalCycles);
    if (typeof j.currentRunId === "number" && Number.isFinite(j.currentRunId)) {
      st.currentRunId = Math.max(st.currentRunId, Math.floor(j.currentRunId));
    }
    if (typeof j.totalScanAttempts === "number" && Number.isFinite(j.totalScanAttempts)) {
      st.totalScanAttempts = Math.max(st.totalScanAttempts, Math.floor(j.totalScanAttempts));
    }
    if (typeof j.totalScanSuccess === "number" && Number.isFinite(j.totalScanSuccess)) {
      st.totalScanSuccess = Math.max(st.totalScanSuccess, Math.floor(j.totalScanSuccess));
    }
    if (typeof j.totalScanErrors === "number" && Number.isFinite(j.totalScanErrors)) {
      st.totalScanErrors = Math.max(st.totalScanErrors, Math.floor(j.totalScanErrors));
    }
    if (typeof j.totalScanSkippedBusy === "number" && Number.isFinite(j.totalScanSkippedBusy)) {
      st.totalScanSkippedBusy = Math.max(st.totalScanSkippedBusy, Math.floor(j.totalScanSkippedBusy));
    }
    const num = (x: unknown) =>
      typeof x === "number" && Number.isFinite(x) ? x : null;
    st.lastScanTimestamp = num(j.lastScanTimestamp) ?? st.lastScanTimestamp;
    st.lastScanDurationMs = num(j.lastScanDurationMs) ?? st.lastScanDurationMs;
    st.lastScanStartAt = num(j.lastScanStartAt) ?? st.lastScanStartAt;
    st.lastScanEndAt = num(j.lastScanEndAt) ?? st.lastScanEndAt;
    st.lastSuccessfulScanAt = num(j.lastSuccessfulScanAt) ?? st.lastSuccessfulScanAt;
    st.lastScanErrorAt = num(j.lastScanErrorAt) ?? st.lastScanErrorAt;
    st.lastScanErrorMessage =
      typeof j.lastScanErrorMessage === "string" && j.lastScanErrorMessage.length < 4000
        ? j.lastScanErrorMessage
        : st.lastScanErrorMessage;

    meta.loadStatus = "ok";
    meta.lastLoadAtMs = Date.now();
    if (typeof j.savedAt === "string") {
      const t = Date.parse(j.savedAt);
      if (Number.isFinite(t)) meta.lastWriteAtMs = t;
    }
    console.log("[PocketExecution] Rehydrated temporal history from disk:", p, "cycles=", st.temporalCycles.length);
  } catch (e) {
    meta.loadStatus = "error";
    meta.lastLoadAtMs = Date.now();
    console.warn("[PocketExecution] Persistence load failed (non-fatal):", e instanceof Error ? e.message : e);
  }
}

function countConsecutiveVerdictFromEnd(
  cycles: TemporalCyclePersisted[],
  verdict: ExecutionObservationVerdict,
): number {
  let n = 0;
  for (let i = cycles.length - 1; i >= 0; i--) {
    if (cycles[i].verdict === verdict) n++;
    else break;
  }
  return n;
}

function avgNullable(nums: (number | null)[]): number | null {
  const v = nums.filter((x): x is number => x !== null && Number.isFinite(x));
  if (!v.length) return null;
  return r4(v.reduce((a, b) => a + b, 0) / v.length);
}

function buildTemporalDigest(cycles: TemporalCyclePersisted[]): PocketExecutionTemporalDigest {
  const verdictHistory: PocketExecutionVerdictHistoryEntry[] = cycles.map(c => ({
    scannedAt: new Date(c.cycleAt).toISOString(),
    executionObservationVerdict: c.verdict,
    executionReadinessScore: c.executionReadinessScore,
    aggregateSpreadFeasibility: c.aggregateSpreadFeasibility,
    aggregateEntrySnapshotQuality: c.aggregateEntrySnapshotQuality,
    candidateCount: c.candidateCount,
    promotionVerdictAtScan: c.promotionVerdictAtScan,
    blockingReasonsShort: [...c.blockingReasonsShort],
    supportingReasonsShort: [...c.supportingReasonsShort],
  }));

  const consecutiveObservationOnlyCount = countConsecutiveVerdictFromEnd(cycles, "observation_only");
  const consecutiveMinimalExecutableSignalCount = countConsecutiveVerdictFromEnd(
    cycles,
    "minimal_executable_signal",
  );
  const consecutiveBlockedCount = countConsecutiveVerdictFromEnd(cycles, "blocked");

  let readinessTrend: PocketExecutionReadinessTrend = "unknown";
  let spreadTrend: PocketExecutionSpreadTrend = "unknown";
  if (cycles.length >= 6) {
    const rRecent = avgNullable(cycles.slice(-3).map(c => c.executionReadinessScore));
    const rOlder = avgNullable(cycles.slice(-6, -3).map(c => c.executionReadinessScore));
    if (rRecent !== null && rOlder !== null) {
      if (rRecent > rOlder + 0.025) readinessTrend = "up";
      else if (rRecent < rOlder - 0.025) readinessTrend = "down";
      else readinessTrend = "flat";
    }
    const sRecent = avgNullable(cycles.slice(-3).map(c => c.aggregateSpreadFeasibility));
    const sOlder = avgNullable(cycles.slice(-6, -3).map(c => c.aggregateSpreadFeasibility));
    if (sRecent !== null && sOlder !== null) {
      if (sRecent > sOlder + 0.03) spreadTrend = "up";
      else if (sRecent < sOlder - 0.03) spreadTrend = "down";
      else spreadTrend = "flat";
    }
  }

  const stableExecutionCandidates: string[] = [];
  const degradedExecutionCandidates: string[] = [];
  const improvingExecutionCandidates: string[] = [];
  if (cycles.length >= 2) {
    const last = cycles[cycles.length - 1];
    const prev = cycles[cycles.length - 2];
    const pm = new Map(prev.pocketReadinessSummaries.map(x => [x.microBucketKey, x]));
    for (const pl of last.pocketReadinessSummaries) {
      const pp = pm.get(pl.microBucketKey);
      if (pl.betweenSnapshotsStable && pp?.betweenSnapshotsStable) {
        stableExecutionCandidates.push(pl.microBucketKey);
      }
      const rL = pl.pocketReadinessScore;
      const rP = pp?.pocketReadinessScore ?? null;
      if (rL !== null && rP !== null) {
        if (rL < rP - 0.06) degradedExecutionCandidates.push(pl.microBucketKey);
        if (rL > rP + 0.06) improvingExecutionCandidates.push(pl.microBucketKey);
      }
    }
    stableExecutionCandidates.sort();
    degradedExecutionCandidates.sort();
    improvingExecutionCandidates.sort();
  }

  return {
    verdictHistory,
    consecutiveObservationOnlyCount,
    consecutiveMinimalExecutableSignalCount,
    consecutiveBlockedCount,
    readinessTrend,
    spreadTrend,
    stableExecutionCandidates,
    degradedExecutionCandidates,
    improvingExecutionCandidates,
  };
}

function buildTrajectoryDigest(
  cycles: TemporalCyclePersisted[],
  temporal: PocketExecutionTemporalDigest,
): PocketExecutionTrajectoryDigest {
  const notes: string[] = [];
  if (cycles.length === 0) {
    return {
      executionTrajectoryAssessment: "stagnant",
      trajectoryNotes: ["Nenhum ciclo persistido ainda — aguardar scans."],
    };
  }

  if (temporal.consecutiveMinimalExecutableSignalCount >= 3) {
    notes.push(
      `${temporal.consecutiveMinimalExecutableSignalCount} ciclos consecutivos minimal_executable_signal`,
    );
    return { executionTrajectoryAssessment: "consistently_minimal_executable_signal", trajectoryNotes: notes };
  }

  const last5 = cycles.slice(-5);
  if (
    temporal.consecutiveBlockedCount >= 4 ||
    (last5.length >= 5 && last5.every(c => c.verdict === "blocked"))
  ) {
    notes.push("Janela recente dominada por blocked");
    return { executionTrajectoryAssessment: "consistently_blocked", trajectoryNotes: notes };
  }

  if (last5.length >= 4) {
    const distinct = new Set(last5.map(c => c.verdict));
    if (distinct.size >= 3) {
      notes.push("Alternância de veredictos na janela recente");
      return { executionTrajectoryAssessment: "unstable", trajectoryNotes: notes };
    }
  }

  const latest = cycles[cycles.length - 1];
  if (
    temporal.improvingExecutionCandidates.length > 0 &&
    latest.verdict !== "minimal_executable_signal"
  ) {
    notes.push("Readiness por pocket melhora vs ciclo anterior sem minimal_executable_signal");
    return { executionTrajectoryAssessment: "improving_but_not_ready", trajectoryNotes: notes };
  }

  if (
    temporal.readinessTrend === "up" &&
    latest.verdict === "observation_only"
  ) {
    notes.push("Tendência de readiness a subir com veredicto ainda observation_only");
    return { executionTrajectoryAssessment: "improving_but_not_ready", trajectoryNotes: notes };
  }

  if (
    temporal.consecutiveObservationOnlyCount >= 3 &&
    temporal.readinessTrend === "flat" &&
    temporal.spreadTrend === "flat"
  ) {
    notes.push("Plateau observation_only com tendências flat");
    return { executionTrajectoryAssessment: "stagnant", trajectoryNotes: notes };
  }

  notes.push("Leitura temporal por defeito — dados insuficientes para outra etiqueta");
  return { executionTrajectoryAssessment: "stagnant", trajectoryNotes: notes };
}

function countConsecutiveObservationOrBetterFromEnd(cycles: TemporalCyclePersisted[]): number {
  let n = 0;
  for (let i = cycles.length - 1; i >= 0; i--) {
    const v = cycles[i].verdict;
    if (v === "observation_only" || v === "minimal_executable_signal") n++;
    else break;
  }
  return n;
}

function buildExecutionPromotionAssessment(
  cycles: TemporalCyclePersisted[],
  temporal: PocketExecutionTemporalDigest,
  trajectory: PocketExecutionTrajectoryDigest,
): ExecutionPromotionAssessmentDigest {
  const thresholdsUsed: ExecutionPromotionThresholdsDigest = {
    minTemporalCycles: EXEC_PROMO_MIN_TEMPORAL_CYCLES(),
    minConsecutiveObservationOrBetter: EXEC_PROMO_MIN_CONSEC_OBS_OR_BETTER(),
    minConsecutiveMinimalExecutableSignal: EXEC_PROMO_MIN_CONSEC_MINIMAL(),
    minStableExecutionCandidates: EXEC_PROMO_MIN_STABLE_EXEC_KEYS(),
    maxDegradedExecutionCandidates: EXEC_PROMO_MAX_DEGRADED(),
  };

  const nCycles = cycles.length;
  const consecObsOrBetter = countConsecutiveObservationOrBetterFromEnd(cycles);
  const consecMinimal = temporal.consecutiveMinimalExecutableSignalCount;

  const minimumTemporalCyclesSatisfied = nCycles >= thresholdsUsed.minTemporalCycles;
  const minimumConsecutiveObservationOrBetterSatisfied =
    consecObsOrBetter >= thresholdsUsed.minConsecutiveObservationOrBetter;
  const minimumConsecutiveMinimalExecutableSignalSatisfied =
    consecMinimal >= thresholdsUsed.minConsecutiveMinimalExecutableSignal;

  const readinessTrendSatisfied =
    temporal.readinessTrend === "up" || temporal.readinessTrend === "flat";
  const spreadTrendSatisfied =
    temporal.spreadTrend === "up" || temporal.spreadTrend === "flat";

  const stableExecutionCandidatesSatisfied =
    temporal.stableExecutionCandidates.length >= thresholdsUsed.minStableExecutionCandidates;
  const degradedExecutionCandidatesAcceptable =
    temporal.degradedExecutionCandidates.length <= thresholdsUsed.maxDegradedExecutionCandidates;

  const executionTrajectoryAssessmentSupportive =
    trajectory.executionTrajectoryAssessment !== "unstable" &&
    trajectory.executionTrajectoryAssessment !== "consistently_blocked";

  const flags = [
    minimumTemporalCyclesSatisfied,
    minimumConsecutiveObservationOrBetterSatisfied,
    minimumConsecutiveMinimalExecutableSignalSatisfied,
    readinessTrendSatisfied,
    spreadTrendSatisfied,
    stableExecutionCandidatesSatisfied,
    degradedExecutionCandidatesAcceptable,
    executionTrajectoryAssessmentSupportive,
  ];
  const satisfiedCount = flags.filter(Boolean).length;

  const promotionReasons: string[] = [];
  const blockingReasons: string[] = [];

  if (minimumTemporalCyclesSatisfied) {
    promotionReasons.push(`temporal cycles ${nCycles} >= ${thresholdsUsed.minTemporalCycles}`);
  } else {
    blockingReasons.push(`temporal cycles ${nCycles} < ${thresholdsUsed.minTemporalCycles}`);
  }

  if (minimumConsecutiveObservationOrBetterSatisfied) {
    promotionReasons.push(
      `consecutive observation_only|minimal_executable from end: ${consecObsOrBetter} >= ${thresholdsUsed.minConsecutiveObservationOrBetter}`,
    );
  } else {
    blockingReasons.push(
      `consecutive observation_or_better ${consecObsOrBetter} < ${thresholdsUsed.minConsecutiveObservationOrBetter}`,
    );
  }

  if (minimumConsecutiveMinimalExecutableSignalSatisfied) {
    promotionReasons.push(
      `consecutive minimal_executable_signal: ${consecMinimal} >= ${thresholdsUsed.minConsecutiveMinimalExecutableSignal}`,
    );
  } else {
    blockingReasons.push(
      `consecutive minimal_executable_signal ${consecMinimal} < ${thresholdsUsed.minConsecutiveMinimalExecutableSignal}`,
    );
  }

  if (readinessTrendSatisfied) {
    promotionReasons.push(`readinessTrend=${temporal.readinessTrend} (up or flat)`);
  } else {
    blockingReasons.push(
      `readinessTrend=${temporal.readinessTrend} (require up or flat; unknown/down block)`,
    );
  }

  if (spreadTrendSatisfied) {
    promotionReasons.push(`spreadTrend=${temporal.spreadTrend} (up or flat)`);
  } else {
    blockingReasons.push(
      `spreadTrend=${temporal.spreadTrend} (require up or flat; unknown/down block)`,
    );
  }

  if (stableExecutionCandidatesSatisfied) {
    promotionReasons.push(
      `stableExecutionCandidates count ${temporal.stableExecutionCandidates.length} >= ${thresholdsUsed.minStableExecutionCandidates}`,
    );
  } else {
    blockingReasons.push(
      `stableExecutionCandidates count ${temporal.stableExecutionCandidates.length} < ${thresholdsUsed.minStableExecutionCandidates}`,
    );
  }

  if (degradedExecutionCandidatesAcceptable) {
    promotionReasons.push(
      `degradedExecutionCandidates count ${temporal.degradedExecutionCandidates.length} <= ${thresholdsUsed.maxDegradedExecutionCandidates}`,
    );
  } else {
    blockingReasons.push(
      `degradedExecutionCandidates count ${temporal.degradedExecutionCandidates.length} > ${thresholdsUsed.maxDegradedExecutionCandidates}`,
    );
  }

  if (executionTrajectoryAssessmentSupportive) {
    promotionReasons.push(
      `executionTrajectoryAssessment=${trajectory.executionTrajectoryAssessment} (not unstable/consistently_blocked)`,
    );
  } else {
    blockingReasons.push(
      `executionTrajectoryAssessment=${trajectory.executionTrajectoryAssessment} is not supportive for paper probe gate`,
    );
  }

  let overallExecutionPromotionVerdict: OverallExecutionPromotionVerdict = "not_ready";
  if (satisfiedCount === 8) {
    overallExecutionPromotionVerdict = "ready_for_minimal_paper_execution_probe";
  } else if (
    minimumTemporalCyclesSatisfied &&
    executionTrajectoryAssessmentSupportive &&
    satisfiedCount >= 5
  ) {
    overallExecutionPromotionVerdict = "borderline";
  }

  return {
    minimumTemporalCyclesSatisfied,
    minimumConsecutiveObservationOrBetterSatisfied,
    minimumConsecutiveMinimalExecutableSignalSatisfied,
    readinessTrendSatisfied,
    spreadTrendSatisfied,
    stableExecutionCandidatesSatisfied,
    degradedExecutionCandidatesAcceptable,
    executionTrajectoryAssessmentSupportive,
    overallExecutionPromotionVerdict,
    promotionReasons,
    blockingReasons,
    thresholdsUsed,
  };
}

function buildPersistenceDigestExec(): PocketExecutionPersistenceDigest {
  const disabled = isExecPersistenceDiskDisabled();
  const meta = getPersistMetaExec();
  return {
    persistenceEnabled: !disabled,
    persistencePath: disabled ? null : defaultExecPersistencePath(),
    lastPersistenceWriteAt: meta.lastWriteAtMs ? new Date(meta.lastWriteAtMs).toISOString() : null,
    lastPersistenceLoadAt: meta.lastLoadAtMs ? new Date(meta.lastLoadAtMs).toISOString() : null,
    persistenceLoadStatus: meta.loadStatus,
  };
}

interface ProbeState {
  loopStarted: boolean;
  scanning: boolean;
  scanError: string | null;
  lastScanTimestamp: number | null;
  lastScanDurationMs: number | null;
  lastScanStartAt: number | null;
  lastScanEndAt: number | null;
  lastSuccessfulScanAt: number | null;
  lastScanErrorAt: number | null;
  lastScanErrorMessage: string | null;
  currentRunId: number;
  nextScheduledScanAt: number | null;
  schedulerStartedAt: number | null;
  scheduledTimeoutId: ReturnType<typeof setTimeout> | null;
  totalScanAttempts: number;
  totalScanSuccess: number;
  totalScanErrors: number;
  totalScanSkippedBusy: number;
  lastRun: PocketExecutionLastRun | null;
  temporalCycles: TemporalCyclePersisted[];
}

function getState(): ProbeState {
  const g = globalThis as unknown as Record<string, ProbeState | undefined>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = {
      loopStarted: false,
      scanning: false,
      scanError: null,
      lastScanTimestamp: null,
      lastScanDurationMs: null,
      lastScanStartAt: null,
      lastScanEndAt: null,
      lastSuccessfulScanAt: null,
      lastScanErrorAt: null,
      lastScanErrorMessage: null,
      currentRunId: 0,
      nextScheduledScanAt: null,
      schedulerStartedAt: null,
      scheduledTimeoutId: null,
      totalScanAttempts: 0,
      totalScanSuccess: 0,
      totalScanErrors: 0,
      totalScanSkippedBusy: 0,
      lastRun: null,
      temporalCycles: [],
    };
    hydrateExecutionFromDisk(g[GLOBAL_KEY]!);
  }
  const st = g[GLOBAL_KEY]!;
  if (!st.temporalCycles) st.temporalCycles = [];
  return st;
}

function median(nums: number[]): number | null {
  if (!nums.length) return null;
  const s = [...nums].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
}

function maxPriceJump(a: number[], b: number[]): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    m = Math.max(m, Math.abs(a[i] - b[i]));
  }
  return m;
}

function compareMarket(
  m1: PocketEconomicsMarketRow,
  m2: PocketEconomicsMarketRow | undefined,
): { priceJump: number | null; spreadDelta: number | null } {
  if (!m2 || m1.outcomes.length !== m2.outcomes.length) {
    return { priceJump: null, spreadDelta: null };
  }
  return {
    priceJump: maxPriceJump(m1.prices, m2.prices),
    spreadDelta: Math.abs(m1.spread - m2.spread),
  };
}

function buildCandidateFromSnapshots(
  microBucketKey: string,
  rowsA: PocketEconomicsMarketRow[],
  mapB: Map<string, PocketEconomicsMarketRow[]>,
): CandidateExecutionPocket {
  const rowsB = mapB.get(microBucketKey) ?? [];
  const byIdB = new Map(rowsB.map(r => [r.id, r]));
  const streak = getMicroBucketEligibleStreakFromEconomics(microBucketKey);

  const spreads = rowsA.map(r => r.spread);
  const medSpread = median(spreads);
  const spreadFeasibility =
    medSpread !== null ? r4(Math.max(0, Math.min(1, 1 - medSpread / SPREAD_CAP))) : null;

  let okPrices = 0;
  for (const r of rowsA) {
    if (r.prices.every(p => Number.isFinite(p)) && r.liquidity >= MIN_LIQ) okPrices++;
  }
  const entrySnapshotQuality =
    rowsA.length > 0 ? r4(okPrices / rowsA.length) : null;

  const components: ExecutionProbeMarketObservation[] = [];
  let priceJumpMaxInPocket: number | null = 0;
  let spreadDriftMaxInPocket: number | null = 0;
  let allStable = rowsA.length > 0;

  const maxJump = MAX_PRICE_JUMP();
  const maxDrift = MAX_SPREAD_DRIFT();

  for (const r of rowsA) {
    const { priceJump, spreadDelta } = compareMarket(r, byIdB.get(r.id));
    if (priceJump === null || !Number.isFinite(priceJump)) {
      allStable = false;
      priceJumpMaxInPocket = null;
    } else if (priceJumpMaxInPocket !== null) {
      priceJumpMaxInPocket = Math.max(priceJumpMaxInPocket, priceJump);
      if (priceJump > maxJump) allStable = false;
    }
    if (spreadDelta === null || !Number.isFinite(spreadDelta)) {
      allStable = false;
      spreadDriftMaxInPocket = null;
    } else if (spreadDriftMaxInPocket !== null) {
      spreadDriftMaxInPocket = Math.max(spreadDriftMaxInPocket, spreadDelta);
      if (spreadDelta > maxDrift) allStable = false;
    }
    components.push({
      id: r.id,
      question: r.question,
      liquidity: r.liquidity,
      spread: r.spread,
      prices: r.prices,
      outcomes: r.outcomes,
      priceJumpMaxVsPriorSnapshot: priceJump !== null && Number.isFinite(priceJump) ? r4(priceJump) : null,
      spreadDeltaVsPriorSnapshot: spreadDelta !== null && Number.isFinite(spreadDelta) ? r4(spreadDelta) : null,
    });
  }

  if (rowsA.length === 0) {
    allStable = false;
    priceJumpMaxInPocket = null;
    spreadDriftMaxInPocket = null;
  }

  const pocketSupportingReasons: string[] = [];
  const pocketBlockingReasons: string[] = [];
  if (streak >= 2) {
    pocketSupportingReasons.push(`eligibleHistory streak ${streak} (pocket-economics)`);
  } else {
    pocketBlockingReasons.push(`eligibleHistory streak ${streak} < 2`);
  }
  if (spreadFeasibility !== null && spreadFeasibility >= 0.45) {
    pocketSupportingReasons.push(`median spread proxy compatible with cap ${SPREAD_CAP}`);
  } else {
    pocketBlockingReasons.push("median spread proxy weak vs cap");
  }
  if (entrySnapshotQuality !== null && entrySnapshotQuality >= 0.8) {
    pocketSupportingReasons.push("entry snapshot: most components have finite prices + min liq");
  }
  if (allStable) {
    pocketSupportingReasons.push(`between-snapshots stable (max price jump <= ${maxJump}, spread drift <= ${maxDrift})`);
  } else {
    pocketBlockingReasons.push("between-snapshots not stable or missing ids in 2nd pull");
  }

  const simulatedEntryWindows = [
    `Observation window: first Gamma pull at t0; second pull after ${SNAPSHOT_GAP_MS()}ms — no simulated fill.`,
  ];
  const simulatedExitWindows = [
    "Exit not simulated: second snapshot only checks book still present; no MTM or profit.",
  ];

  return {
    microBucketKey,
    eligibleMarketCount: rowsA.length,
    components,
    pocketPersistenceAtExecution: streak,
    entrySnapshotQuality,
    spreadFeasibility,
    betweenSnapshotsStable: allStable,
    priceJumpMaxInPocket: priceJumpMaxInPocket !== null ? r4(priceJumpMaxInPocket) : null,
    spreadDriftMaxInPocket: spreadDriftMaxInPocket !== null ? r4(spreadDriftMaxInPocket) : null,
    simulatedEntryWindows,
    simulatedExitWindows,
    pocketSupportingReasons,
    pocketBlockingReasons,
  };
}

function scheduleNextScan(delayMs: number): void {
  const st = getState();
  if (st.scheduledTimeoutId !== null) {
    clearTimeout(st.scheduledTimeoutId);
    st.scheduledTimeoutId = null;
  }
  st.nextScheduledScanAt = Date.now() + delayMs;
  st.scheduledTimeoutId = setTimeout(() => {
    st.scheduledTimeoutId = null;
    void runScan().finally(() => {
      scheduleNextScan(SCAN_INTERVAL_MS());
    });
  }, delayMs);
}

async function runScan(): Promise<void> {
  const st = getState();
  if (st.scanning) {
    st.totalScanSkippedBusy++;
    console.warn("[PocketExecution] runScan skipped: already scanning");
    return;
  }
  st.totalScanAttempts++;
  st.currentRunId++;
  const runId = st.currentRunId;
  st.lastScanStartAt = Date.now();
  const t0 = Date.now();
  const gap = SNAPSHOT_GAP_MS();

  const emptyLastRun = (partial: Partial<PocketExecutionLastRun>): PocketExecutionLastRun => ({
    candidateExecutionPockets: [],
    executionReadinessScore: null,
    aggregateEntrySnapshotQuality: null,
    aggregateSpreadFeasibility: null,
    pocketPersistenceAtExecutionSummary: null,
    simulatedEntryWindows: [],
    simulatedExitWindows: [],
    executionObservationVerdict: "blocked",
    blockingReasons: [],
    supportingReasons: [],
    promotionVerdictAtScan: "not_ready",
    stableMicroKeysAtScan: [],
    snapshotGapMs: gap,
    firstSnapshotAt: null,
    secondSnapshotAt: null,
    totalMarketsScannedFirst: 0,
    totalMarketsScannedSecond: 0,
    ...partial,
  });

  try {
    st.scanning = true;
    st.scanError = null;

    const econDigest = buildPocketEconomicsDigest();
    const promo = econDigest.promotionAssessment;
    const stableKeys = [...econDigest.stableMicroBuckets].sort();

    const blockingReasons: string[] = [];
    const supportingReasons: string[] = [];

    if (promo.overallPromotionVerdict === "not_ready") {
      blockingReasons.push(
        `promotionAssessment.overallPromotionVerdict is not_ready — execution probe stays idle for promoted pockets.`,
      );
    }
    if (stableKeys.length === 0) {
      blockingReasons.push("No stableMicroBuckets from pocket-economics for this cycle.");
    }

    if (promo.overallPromotionVerdict === "not_ready" || stableKeys.length === 0) {
      st.lastRun = emptyLastRun({
        promotionVerdictAtScan: promo.overallPromotionVerdict,
        stableMicroKeysAtScan: stableKeys,
        blockingReasons,
        supportingReasons: [],
        executionObservationVerdict: "blocked",
      });
      st.lastScanTimestamp = Date.now();
      st.lastScanDurationMs = Date.now() - t0;
      st.lastSuccessfulScanAt = st.lastScanTimestamp;
      st.lastScanErrorMessage = null;
      st.lastScanErrorAt = null;
      st.totalScanSuccess++;
      recordTemporalCycle(st, st.lastRun);
      maybePersistExecutionState(st, true);
      console.log(`[PocketExecution] Scan #${runId} ok (blocked gating): stableKeys=${stableKeys.length}`);
      return;
    }

    supportingReasons.push(
      `promotionAssessment=${promo.overallPromotionVerdict}; stable micro-buckets=${stableKeys.length}.`,
    );

    const snap1 = await snapshotPocketFamilyEligibleBucketsForExecutionProbe();
    await new Promise<void>(r => {
      setTimeout(r, gap);
    });
    const snap2 = await snapshotPocketFamilyEligibleBucketsForExecutionProbe();

    const candidates: CandidateExecutionPocket[] = [];
    for (const key of stableKeys) {
      const rowsA = snap1.bucketMap.get(key) ?? [];
      if (rowsA.length === 0) {
        blockingReasons.push(
          `Stable key "${key.length > 72 ? `${key.slice(0, 72)}…` : key}" absent from live eligible snapshot t0.`,
        );
        continue;
      }
      candidates.push(buildCandidateFromSnapshots(key, rowsA, snap2.bucketMap));
    }

    if (candidates.length === 0) {
      blockingReasons.push("No eligible live buckets for any stable micro-key.");
    }

    const scores: number[] = [];
    for (const c of candidates) {
      const base =
        ((c.entrySnapshotQuality ?? 0) + (c.spreadFeasibility ?? 0)) / 2 * (c.betweenSnapshotsStable ? 1 : 0.35);
      scores.push(r4(base));
    }
    const executionReadinessScore =
      scores.length > 0 ? r4(Math.min(...scores)) : null;
    const aggregateEntrySnapshotQuality =
      candidates.length > 0
        ? r4(
            candidates.reduce((s, c) => s + (c.entrySnapshotQuality ?? 0), 0) / candidates.length,
          )
        : null;
    const aggregateSpreadFeasibility =
      candidates.length > 0
        ? r4(
            candidates.reduce((s, c) => s + (c.spreadFeasibility ?? 0), 0) / candidates.length,
          )
        : null;

    const streaks = candidates.map(c => c.pocketPersistenceAtExecution);
    const pocketPersistenceAtExecutionSummary =
      streaks.length > 0
        ? `streaks min=${Math.min(...streaks)} max=${Math.max(...streaks)} (eligible scans w/ pocket)`
        : null;

    const globalEntryWindows = [
      `Dual Gamma snapshot gap=${gap}ms; family=${TARGET_FAMILY_KEY}; observation-only.`,
    ];
    const globalExitWindows = [
      "Second snapshot checks continuity of quoted components; no exit fill model.",
    ];

    let verdict: ExecutionObservationVerdict = "blocked";
    if (candidates.length === 0) {
      verdict = "blocked";
    } else {
      const allStable = candidates.every(c => c.betweenSnapshotsStable);
      const scoresOk =
        executionReadinessScore !== null &&
        executionReadinessScore >= 0.4 &&
        aggregateSpreadFeasibility !== null &&
        aggregateSpreadFeasibility >= 0.45;
      if (allStable && scoresOk && promo.overallPromotionVerdict === "ready_for_minimal_execution_probe") {
        verdict = "minimal_executable_signal";
        supportingReasons.push(
          "All stable pockets present live; dual-snapshot stability + readiness score pass conservative gates.",
        );
      } else if (candidates.length > 0) {
        verdict = "observation_only";
        if (!allStable) {
          blockingReasons.push("At least one pocket failed dual-snapshot stability — observation only.");
        }
        if (promo.overallPromotionVerdict === "borderline") {
          supportingReasons.push("Promotion borderline: execution probe records quotes but does not upgrade to minimal_executable_signal.");
        }
        if (!scoresOk) {
          blockingReasons.push("executionReadinessScore / spread feasibility below conservative bar for minimal_executable_signal.");
        }
      }
    }

    st.lastRun = {
      candidateExecutionPockets: candidates,
      executionReadinessScore,
      aggregateEntrySnapshotQuality,
      aggregateSpreadFeasibility,
      pocketPersistenceAtExecutionSummary,
      simulatedEntryWindows: globalEntryWindows,
      simulatedExitWindows: globalExitWindows,
      executionObservationVerdict: verdict,
      blockingReasons,
      supportingReasons,
      promotionVerdictAtScan: promo.overallPromotionVerdict,
      stableMicroKeysAtScan: stableKeys,
      snapshotGapMs: gap,
      firstSnapshotAt: snap1.fetchedAt,
      secondSnapshotAt: snap2.fetchedAt,
      totalMarketsScannedFirst: snap1.totalMarketsScanned,
      totalMarketsScannedSecond: snap2.totalMarketsScanned,
    };

    st.lastScanTimestamp = Date.now();
    st.lastScanDurationMs = Date.now() - t0;
    st.lastSuccessfulScanAt = st.lastScanTimestamp;
    st.lastScanErrorMessage = null;
    st.lastScanErrorAt = null;
    st.totalScanSuccess++;
    recordTemporalCycle(st, st.lastRun);
    maybePersistExecutionState(st, true);
    console.log(
      `[PocketExecution] Scan #${runId} ok: candidates=${candidates.length} verdict=${verdict}`,
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    st.scanError = msg;
    st.lastScanErrorMessage = msg;
    st.lastScanErrorAt = Date.now();
    st.totalScanErrors++;
    console.error(`[PocketExecution] Scan #${runId} error:`, msg);
  } finally {
    st.lastScanEndAt = Date.now();
    st.scanning = false;
  }
}

export function ensurePocketExecutionProbe(): void {
  const st = getState();
  if (st.loopStarted) return;
  st.loopStarted = true;
  st.schedulerStartedAt = Date.now();
  console.log("[PocketExecution] Scheduler started (after pocket-economics)");
  scheduleNextScan(BOOT_DELAY_MS());
}

export function getPocketExecutionHealth(): {
  pocketExecutionSchedulerRunning: boolean;
  pocketExecutionIsScanRunning: boolean;
  lastSuccessfulPocketExecutionScanAt: string | null;
  lastPocketExecutionErrorAt: string | null;
  pocketExecutionSchedulerStartedAt: string | null;
  pocketExecutionLastScanStartAt: string | null;
  pocketExecutionLastScanEndAt: string | null;
  pocketExecutionLastScanErrorMessage: string | null;
  pocketExecutionNextScheduledScanAt: string | null;
  pocketExecutionCurrentRunId: number;
  pocketExecutionTotalScanAttempts: number;
  pocketExecutionTotalScanSuccess: number;
  pocketExecutionTotalScanErrors: number;
  pocketExecutionTotalScanSkippedBusy: number;
  pocketExecutionPersistenceEnabled: boolean;
  pocketExecutionPersistencePath: string | null;
  pocketExecutionLastPersistenceWriteAt: string | null;
  pocketExecutionLastPersistenceLoadAt: string | null;
  pocketExecutionPersistenceLoadStatus: PocketExecutionPersistenceLoadStatus;
  pocketExecutionTemporalCyclesPersisted: number;
} {
  const st = getState();
  const p = buildPersistenceDigestExec();
  return {
    pocketExecutionSchedulerRunning: st.loopStarted,
    pocketExecutionIsScanRunning: st.scanning,
    lastSuccessfulPocketExecutionScanAt: st.lastSuccessfulScanAt
      ? new Date(st.lastSuccessfulScanAt).toISOString()
      : null,
    lastPocketExecutionErrorAt: st.lastScanErrorAt ? new Date(st.lastScanErrorAt).toISOString() : null,
    pocketExecutionSchedulerStartedAt: st.schedulerStartedAt
      ? new Date(st.schedulerStartedAt).toISOString()
      : null,
    pocketExecutionLastScanStartAt: st.lastScanStartAt ? new Date(st.lastScanStartAt).toISOString() : null,
    pocketExecutionLastScanEndAt: st.lastScanEndAt ? new Date(st.lastScanEndAt).toISOString() : null,
    pocketExecutionLastScanErrorMessage: st.lastScanErrorMessage,
    pocketExecutionNextScheduledScanAt: st.nextScheduledScanAt
      ? new Date(st.nextScheduledScanAt).toISOString()
      : null,
    pocketExecutionCurrentRunId: st.currentRunId,
    pocketExecutionTotalScanAttempts: st.totalScanAttempts,
    pocketExecutionTotalScanSuccess: st.totalScanSuccess,
    pocketExecutionTotalScanErrors: st.totalScanErrors,
    pocketExecutionTotalScanSkippedBusy: st.totalScanSkippedBusy,
    pocketExecutionPersistenceEnabled: p.persistenceEnabled,
    pocketExecutionPersistencePath: p.persistencePath,
    pocketExecutionLastPersistenceWriteAt: p.lastPersistenceWriteAt,
    pocketExecutionLastPersistenceLoadAt: p.lastPersistenceLoadAt,
    pocketExecutionPersistenceLoadStatus: p.persistenceLoadStatus,
    pocketExecutionTemporalCyclesPersisted: st.temporalCycles.length,
  };
}

export function buildPocketExecutionDigest(): PocketExecutionDigest {
  const st = getState();
  const scanStatus: PocketExecutionDigest["scanStatus"] =
    st.scanError ? "error" : st.scanning ? "scanning" : st.lastScanTimestamp ? "completed" : "idle";

  const lr = st.lastRun;
  const temporal = buildTemporalDigest(st.temporalCycles);
  const trajectory = buildTrajectoryDigest(st.temporalCycles, temporal);
  const executionPromotionAssessment = buildExecutionPromotionAssessment(
    st.temporalCycles,
    temporal,
    trajectory,
  );

  return {
    computedAt: new Date().toISOString(),
    probeVersion: "pocket-execution-v1",
    scanStatus,
    targetFamilyKey: TARGET_FAMILY_KEY,
    note:
      "Observação apenas: micro-pockets estáveis + promotion ≠ not_ready; dual Gamma (POCKET_EXEC_SNAPSHOT_GAP_MS). Persistência pocket-execution-state.json sob PAPER_STATE_DIR ou cwd/.paper. executionPromotionAssessment: gate explícito para paper execution probe mínimo (POCKET_EXEC_EXEC_PROMO_*); não cria paper. Sem PnL, fills simulados agressivos ou execution engine.",
    promotionVerdictAtScan: lr?.promotionVerdictAtScan ?? null,
    stableMicroKeysAtScan: lr?.stableMicroKeysAtScan ?? [],
    candidateExecutionPockets: lr?.candidateExecutionPockets ?? [],
    executionReadinessScore: lr?.executionReadinessScore ?? null,
    aggregateEntrySnapshotQuality: lr?.aggregateEntrySnapshotQuality ?? null,
    aggregateSpreadFeasibility: lr?.aggregateSpreadFeasibility ?? null,
    pocketPersistenceAtExecutionSummary: lr?.pocketPersistenceAtExecutionSummary ?? null,
    simulatedEntryWindows: lr?.simulatedEntryWindows ?? [],
    simulatedExitWindows: lr?.simulatedExitWindows ?? [],
    executionObservationVerdict: lr?.executionObservationVerdict ?? "blocked",
    blockingReasons: lr?.blockingReasons ?? [],
    supportingReasons: lr?.supportingReasons ?? [],
    snapshotGapMs: lr?.snapshotGapMs ?? SNAPSHOT_GAP_MS(),
    firstSnapshotAt: lr?.firstSnapshotAt ? new Date(lr.firstSnapshotAt).toISOString() : null,
    secondSnapshotAt: lr?.secondSnapshotAt ? new Date(lr.secondSnapshotAt).toISOString() : null,
    totalMarketsScannedFirst: lr?.totalMarketsScannedFirst ?? 0,
    totalMarketsScannedSecond: lr?.totalMarketsScannedSecond ?? 0,
    lastScanStartAt: st.lastScanStartAt ? new Date(st.lastScanStartAt).toISOString() : null,
    lastScanEndAt: st.lastScanEndAt ? new Date(st.lastScanEndAt).toISOString() : null,
    lastSuccessfulScanAt: st.lastSuccessfulScanAt ? new Date(st.lastSuccessfulScanAt).toISOString() : null,
    lastScanErrorAt: st.lastScanErrorAt ? new Date(st.lastScanErrorAt).toISOString() : null,
    lastScanErrorMessage: st.lastScanErrorMessage,
    isScanRunning: st.scanning,
    currentRunId: st.currentRunId,
    nextScheduledScanAt: st.nextScheduledScanAt ? new Date(st.nextScheduledScanAt).toISOString() : null,
    schedulerStartedAt: st.schedulerStartedAt ? new Date(st.schedulerStartedAt).toISOString() : null,
    totalScanAttempts: st.totalScanAttempts,
    totalScanSuccess: st.totalScanSuccess,
    totalScanErrors: st.totalScanErrors,
    totalScanSkippedBusy: st.totalScanSkippedBusy,
    persistence: buildPersistenceDigestExec(),
    temporal,
    trajectory,
    executionPromotionAssessment,
  };
}
