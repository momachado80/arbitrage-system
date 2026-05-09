/**
 * Minimal paper execution probe — isolado; só age quando pocket-execution
 * executionPromotionAssessment ≠ not_ready. Sem execution engine global, sem capital real.
 * Outcomes são rotulados como observacionais/paper; sem PnL monetizado.
 */

import fs from "fs";
import path from "path";
import { buildPocketEconomicsDigest } from "./pocketEconomicsProbe";
import { snapshotPocketFamilyEligibleBucketsForExecutionProbe } from "./pocketEconomicsProbe";
import { buildPocketExecutionDigest } from "./pocketExecutionProbe";
import type { CandidateExecutionPocket, PocketExecutionDigest } from "./pocketExecutionProbe";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES } from "./paperStateDir";
import {
  buildMicroEdgeAssessmentFromEntries,
  type MicroEdgeAssessmentDigest,
} from "./minimalPaperMicroEdgeAssessment";
import {
  buildCapturabilityAssessment,
  type CapturabilityAssessmentDigest,
} from "./minimalPaperCapturabilityAssessment";
import {
  buildMultiWindowAssessment,
  type MultiWindowAssessmentDigest,
  type MinimalPaperAdditionalObservedWindow,
} from "./minimalPaperMultiWindowAssessment";
import {
  buildRefinedOutcomeClassification,
  type RefinedOutcomeClassificationDigest,
} from "./minimalPaperRefinedClassification";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}

const SCAN_INTERVAL_MS = () => envNum("MINIMAL_PAPER_SCAN_INTERVAL_MS", 6 * 3600_000);
const BOOT_DELAY_MS = () => envNum("MINIMAL_PAPER_BOOT_DELAY_MS", 20_000);
const FOLLOWUP_GAP_MS = () => envNum("MINIMAL_PAPER_FOLLOWUP_GAP_MS", 2500);
const NEW_ENTRIES_PER_CYCLE = () => envNum("MINIMAL_PAPER_NEW_ENTRIES_PER_CYCLE", 1);
const MAX_OPEN_ENTRIES = () => envNum("MINIMAL_PAPER_MAX_OPEN_ENTRIES", 2);
const MAX_PERSISTED_ENTRIES = () => envNum("MINIMAL_PAPER_MAX_PERSISTED_ENTRIES", 40);
const PERSIST_THROTTLE_MS = () => envNum("MINIMAL_PAPER_PERSIST_THROTTLE_MS", 30_000);
const PERSIST_VERSION = 1 as const;

const TARGET_FAMILY_KEY = "other:price_above:>3m" as const;

const GLOBAL_KEY = "__minimalPaperExecutionProbe_v1";
const PERSIST_META_KEY = "__minimalPaperExecutionPersistMeta_v1";

export type MinimalPaperObservationalOutcomeLabel =
  | "insufficient_data"
  | "book_quotes_unchanged_within_eps"
  | "book_quotes_drift_observed"
  | "component_missing_in_followup";

export type MinimalPaperAssessmentVerdict =
  | "paper_active"
  | "blocked_not_ready_gate"
  | "blocked_no_stable_candidates"
  | "idle_capped"
  | "observation_cycle_ok";

const ASSESSMENT_VERDICTS: MinimalPaperAssessmentVerdict[] = [
  "paper_active",
  "blocked_not_ready_gate",
  "blocked_no_stable_candidates",
  "idle_capped",
  "observation_cycle_ok",
];

export interface MinimalPaperMarketLite {
  id: string;
  spread: number;
  prices: number[];
}

export interface MinimalPaperStructuralConditions {
  overallExecutionPromotionVerdict: string;
  pocketEconomicsPromotionVerdict: string;
  stableMicroKeysAtScanCount: number;
  executionObservationVerdictAtEntry: string;
}

export interface MinimalPaperObservedWindow {
  observedAt: string;
  marketsLiteAfter: MinimalPaperMarketLite[];
  maxAbsPriceDeltaAcrossComponents: number | null;
  observationalOutcomeLabel: MinimalPaperObservationalOutcomeLabel;
  outcomeNotes: string[];
}

export interface MinimalPaperEntry {
  id: string;
  microBucketKey: string;
  paperEntryAt: string;
  entrySnapshot: { marketsLite: MinimalPaperMarketLite[] };
  rationale: string;
  structuralConditionsAtEntry: MinimalPaperStructuralConditions;
  observedWindow?: MinimalPaperObservedWindow;
  /** Janelas adicionais (ex. imediata); backward-compatible — undefined para entries históricas. */
  additionalObservedWindows?: MinimalPaperAdditionalObservedWindow[];
}

export interface MinimalPaperExecutionAssessmentDigest {
  assessmentVerdict: MinimalPaperAssessmentVerdict;
  blockingReasons: string[];
  supportingReasons: string[];
  gateOverallExecutionPromotionVerdict: string | null;
  openPaperEntriesCount: number;
  observedEntriesCount: number;
}

export type MinimalPaperPersistenceLoadStatus =
  | "ok"
  | "missing"
  | "invalid"
  | "error"
  | "disabled"
  | "pending";

export interface MinimalPaperPersistenceDigest {
  persistenceEnabled: boolean;
  persistencePath: string | null;
  lastPersistenceWriteAt: string | null;
  lastPersistenceLoadAt: string | null;
  persistenceLoadStatus: MinimalPaperPersistenceLoadStatus;
}

export interface MinimalPaperExecutionDigest {
  computedAt: string;
  probeVersion: "minimal-paper-execution-v1";
  scanStatus: "completed" | "scanning" | "idle" | "error";
  targetFamilyKey: typeof TARGET_FAMILY_KEY;
  note: string;
  entries: MinimalPaperEntry[];
  minimalPaperExecutionAssessment: MinimalPaperExecutionAssessmentDigest;
  /** Leitura agregada observacional sobre episódios fechados; não altera gates nem PnL monetizado. */
  microEdgeAssessment: MicroEdgeAssessmentDigest;
  capturabilityAssessment: CapturabilityAssessmentDigest;
  multiWindowAssessment: MultiWindowAssessmentDigest;
  refinedOutcomeClassification: RefinedOutcomeClassificationDigest;
  persistence: MinimalPaperPersistenceDigest;
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
}

interface PersistMeta {
  lastWriteAtMs: number | null;
  lastLoadAtMs: number | null;
  loadStatus: MinimalPaperPersistenceLoadStatus;
  lastThrottleWallMs: number;
}

function getPersistMeta(): PersistMeta {
  const g = globalThis as unknown as Record<string, PersistMeta | undefined>;
  if (!g[PERSIST_META_KEY]) {
    g[PERSIST_META_KEY] = {
      lastWriteAtMs: null,
      lastLoadAtMs: null,
      loadStatus: "pending",
      lastThrottleWallMs: 0,
    };
  }
  return g[PERSIST_META_KEY]!;
}

function isDiskDisabled(): boolean {
  return process.env.MINIMAL_PAPER_DISABLE_DISK === "1";
}

function defaultPath(): string {
  const raw = process.env.MINIMAL_PAPER_STATE_PATH?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.minimalPaperExecution);
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
  entries: MinimalPaperEntry[];
  lastAssessmentVerdict: MinimalPaperAssessmentVerdict | null;
}

interface PersistedFileV1 {
  version: typeof PERSIST_VERSION;
  savedAt: string;
  entries: MinimalPaperEntry[];
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
  lastAssessmentVerdict?: string | null;
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
      entries: [],
      lastAssessmentVerdict: null,
    };
    hydrate(g[GLOBAL_KEY]!);
  }
  if (!g[GLOBAL_KEY]!.entries) g[GLOBAL_KEY]!.entries = [];
  return g[GLOBAL_KEY]!;
}

function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function marketsToLite(
  rows: { id: string; spread: number; prices: number[] }[],
): MinimalPaperMarketLite[] {
  return rows.map(r => ({
    id: r.id,
    spread: r4(r.spread),
    prices: r.prices.map(p => r4(p)),
  }));
}

function maxAbsPriceDelta(
  before: MinimalPaperMarketLite[],
  after: Map<string, MinimalPaperMarketLite>,
): { max: number | null; notes: string[] } {
  const notes: string[] = [];
  let max = 0;
  let any = false;
  for (const b of before) {
    const a = after.get(b.id);
    if (!a) {
      notes.push(`missing id ${b.id} in follow-up snapshot`);
      continue;
    }
    if (b.prices.length !== a.prices.length) {
      notes.push(`price length mismatch ${b.id}`);
      continue;
    }
    for (let i = 0; i < b.prices.length; i++) {
      any = true;
      max = Math.max(max, Math.abs(b.prices[i] - a.prices[i]));
    }
  }
  return { max: any ? r4(max) : null, notes };
}

function outcomeLabel(maxDelta: number | null, eps = 0.02): MinimalPaperObservationalOutcomeLabel {
  if (maxDelta === null) return "insufficient_data";
  if (maxDelta <= eps) return "book_quotes_unchanged_within_eps";
  return "book_quotes_drift_observed";
}

function buildPersistenceDigest(): MinimalPaperPersistenceDigest {
  const meta = getPersistMeta();
  const dis = isDiskDisabled();
  return {
    persistenceEnabled: !dis,
    persistencePath: dis ? null : defaultPath(),
    lastPersistenceWriteAt: meta.lastWriteAtMs ? new Date(meta.lastWriteAtMs).toISOString() : null,
    lastPersistenceLoadAt: meta.lastLoadAtMs ? new Date(meta.lastLoadAtMs).toISOString() : null,
    persistenceLoadStatus: meta.loadStatus,
  };
}

function sanitizeEntries(raw: unknown): MinimalPaperEntry[] {
  if (!Array.isArray(raw)) return [];
  const out: MinimalPaperEntry[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const o = item as Record<string, unknown>;
    const id = o.id;
    const mk = o.microBucketKey;
    if (typeof id !== "string" || id.length < 2 || id.length > 80) continue;
    if (typeof mk !== "string" || mk.length < 1 || mk.length > 420) continue;
    const pe = o.paperEntryAt;
    if (typeof pe !== "string") continue;
    const es = o.entrySnapshot;
    if (!es || typeof es !== "object") continue;
    const ml = (es as { marketsLite?: unknown }).marketsLite;
    const marketsLite: MinimalPaperMarketLite[] = [];
    if (Array.isArray(ml)) {
      for (const m of ml) {
        if (!m || typeof m !== "object") continue;
        const x = m as Record<string, unknown>;
        if (typeof x.id !== "string" || x.id.length > 64) continue;
        const sp = Number(x.spread);
        const pr = Array.isArray(x.prices) ? x.prices.map(Number).filter(Number.isFinite) : [];
        marketsLite.push({ id: x.id, spread: r4(sp), prices: pr.map(p => r4(p)) });
      }
    }
    const sc = o.structuralConditionsAtEntry;
    let structuralConditionsAtEntry: MinimalPaperStructuralConditions = {
      overallExecutionPromotionVerdict: "unknown",
      pocketEconomicsPromotionVerdict: "unknown",
      stableMicroKeysAtScanCount: 0,
      executionObservationVerdictAtEntry: "unknown",
    };
    if (sc && typeof sc === "object") {
      const s = sc as Record<string, unknown>;
      structuralConditionsAtEntry = {
        overallExecutionPromotionVerdict:
          typeof s.overallExecutionPromotionVerdict === "string"
            ? s.overallExecutionPromotionVerdict.slice(0, 64)
            : "unknown",
        pocketEconomicsPromotionVerdict:
          typeof s.pocketEconomicsPromotionVerdict === "string"
            ? s.pocketEconomicsPromotionVerdict.slice(0, 64)
            : "unknown",
        stableMicroKeysAtScanCount:
          typeof s.stableMicroKeysAtScanCount === "number" && Number.isFinite(s.stableMicroKeysAtScanCount)
            ? Math.max(0, Math.floor(s.stableMicroKeysAtScanCount))
            : 0,
        executionObservationVerdictAtEntry:
          typeof s.executionObservationVerdictAtEntry === "string"
            ? s.executionObservationVerdictAtEntry.slice(0, 64)
            : "unknown",
      };
    }
    const rationale = typeof o.rationale === "string" ? o.rationale.slice(0, 2000) : "";
    let observedWindow: MinimalPaperObservedWindow | undefined;
    const ow = o.observedWindow;
    if (ow && typeof ow === "object") {
      const w = ow as Record<string, unknown>;
      const oa = w.observedAt;
      if (typeof oa === "string") {
        const afterRaw = w.marketsLiteAfter;
        const marketsLiteAfter: MinimalPaperMarketLite[] = [];
        if (Array.isArray(afterRaw)) {
          for (const m of afterRaw) {
            if (!m || typeof m !== "object") continue;
            const x = m as Record<string, unknown>;
            if (typeof x.id !== "string") continue;
            const sp = Number(x.spread);
            const pr = Array.isArray(x.prices) ? x.prices.map(Number).filter(Number.isFinite) : [];
            marketsLiteAfter.push({ id: x.id, spread: r4(sp), prices: pr.map(p => r4(p)) });
          }
        }
        const lbl = w.observationalOutcomeLabel;
        const labels: MinimalPaperObservationalOutcomeLabel[] = [
          "insufficient_data",
          "book_quotes_unchanged_within_eps",
          "book_quotes_drift_observed",
          "component_missing_in_followup",
        ];
        const on = Array.isArray(w.outcomeNotes)
          ? w.outcomeNotes.filter((x): x is string => typeof x === "string").slice(0, 20).map(s => s.slice(0, 400))
          : [];
        const md = w.maxAbsPriceDeltaAcrossComponents;
        observedWindow = {
          observedAt: oa,
          marketsLiteAfter,
          maxAbsPriceDeltaAcrossComponents:
            md === null || md === undefined
              ? null
              : Number.isFinite(Number(md))
                ? r4(Number(md))
                : null,
          observationalOutcomeLabel: labels.includes(lbl as MinimalPaperObservationalOutcomeLabel)
            ? (lbl as MinimalPaperObservationalOutcomeLabel)
            : "insufficient_data",
          outcomeNotes: on,
        };
      }
    }
    let additionalObservedWindows: MinimalPaperAdditionalObservedWindow[] | undefined;
    const rawAW = o.additionalObservedWindows;
    if (Array.isArray(rawAW) && rawAW.length > 0) {
      additionalObservedWindows = [];
      for (const aw of rawAW) {
        if (!aw || typeof aw !== "object") continue;
        const w = aw as Record<string, unknown>;
        if (typeof w.windowLabel !== "string" || typeof w.observedAt !== "string") continue;
        const awML: MinimalPaperMarketLite[] = [];
        if (Array.isArray(w.marketsLiteAfter)) {
          for (const m of w.marketsLiteAfter) {
            if (!m || typeof m !== "object") continue;
            const x = m as Record<string, unknown>;
            if (typeof x.id !== "string") continue;
            const sp = Number(x.spread);
            const pr = Array.isArray(x.prices) ? x.prices.map(Number).filter(Number.isFinite) : [];
            awML.push({ id: x.id, spread: r4(sp), prices: pr.map(p => r4(p)) });
          }
        }
        const awMd = w.maxAbsPriceDeltaAcrossComponents;
        const validLabels: MinimalPaperObservationalOutcomeLabel[] = [
          "insufficient_data", "book_quotes_unchanged_within_eps", "book_quotes_drift_observed", "component_missing_in_followup",
        ];
        additionalObservedWindows.push({
          windowLabel: String(w.windowLabel).slice(0, 32),
          observedAt: w.observedAt,
          gapFromEntryMs: typeof w.gapFromEntryMs === "number" ? w.gapFromEntryMs : 0,
          marketsLiteAfter: awML,
          maxAbsPriceDeltaAcrossComponents: awMd == null ? null : Number.isFinite(Number(awMd)) ? r4(Number(awMd)) : null,
          observationalOutcomeLabel: validLabels.includes(w.observationalOutcomeLabel as MinimalPaperObservationalOutcomeLabel)
            ? (w.observationalOutcomeLabel as MinimalPaperObservationalOutcomeLabel)
            : "insufficient_data",
          outcomeNotes: Array.isArray(w.outcomeNotes) ? w.outcomeNotes.filter((x): x is string => typeof x === "string").slice(0, 10) : [],
        });
      }
      if (additionalObservedWindows.length === 0) additionalObservedWindows = undefined;
    }

    out.push({
      id,
      microBucketKey: mk,
      paperEntryAt: pe,
      entrySnapshot: { marketsLite },
      rationale,
      structuralConditionsAtEntry,
      observedWindow,
      ...(additionalObservedWindows ? { additionalObservedWindows } : {}),
    });
  }
  return out.slice(-MAX_PERSISTED_ENTRIES());
}

function hydrate(st: ProbeState): void {
  const meta = getPersistMeta();
  if (isDiskDisabled()) {
    meta.loadStatus = "disabled";
    meta.lastLoadAtMs = Date.now();
    return;
  }
  const p = defaultPath();
  try {
    if (!fs.existsSync(p)) {
      meta.loadStatus = "missing";
      meta.lastLoadAtMs = Date.now();
      return;
    }
    const j = JSON.parse(fs.readFileSync(p, "utf8")) as Partial<PersistedFileV1>;
    if (j.version !== PERSIST_VERSION) {
      meta.loadStatus = "invalid";
      meta.lastLoadAtMs = Date.now();
      return;
    }
    st.entries = sanitizeEntries(j.entries);
    const n = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? x : null);
    if (typeof j.currentRunId === "number") st.currentRunId = Math.max(st.currentRunId, Math.floor(j.currentRunId));
    if (typeof j.totalScanAttempts === "number") {
      st.totalScanAttempts = Math.max(st.totalScanAttempts, Math.floor(j.totalScanAttempts));
    }
    if (typeof j.totalScanSuccess === "number") {
      st.totalScanSuccess = Math.max(st.totalScanSuccess, Math.floor(j.totalScanSuccess));
    }
    if (typeof j.totalScanErrors === "number") {
      st.totalScanErrors = Math.max(st.totalScanErrors, Math.floor(j.totalScanErrors));
    }
    if (typeof j.totalScanSkippedBusy === "number") {
      st.totalScanSkippedBusy = Math.max(st.totalScanSkippedBusy, Math.floor(j.totalScanSkippedBusy));
    }
    st.lastScanTimestamp = n(j.lastScanTimestamp) ?? st.lastScanTimestamp;
    st.lastScanDurationMs = n(j.lastScanDurationMs) ?? st.lastScanDurationMs;
    st.lastScanStartAt = n(j.lastScanStartAt) ?? st.lastScanStartAt;
    st.lastScanEndAt = n(j.lastScanEndAt) ?? st.lastScanEndAt;
    st.lastSuccessfulScanAt = n(j.lastSuccessfulScanAt) ?? st.lastSuccessfulScanAt;
    st.lastScanErrorAt = n(j.lastScanErrorAt) ?? st.lastScanErrorAt;
    st.lastScanErrorMessage =
      typeof j.lastScanErrorMessage === "string" && j.lastScanErrorMessage.length < 4000
        ? j.lastScanErrorMessage
        : st.lastScanErrorMessage;
    if (typeof j.lastAssessmentVerdict === "string") {
      const v = j.lastAssessmentVerdict as MinimalPaperAssessmentVerdict;
      if (ASSESSMENT_VERDICTS.includes(v)) st.lastAssessmentVerdict = v;
    }
    meta.loadStatus = "ok";
    meta.lastLoadAtMs = Date.now();
    if (typeof j.savedAt === "string") {
      const t = Date.parse(j.savedAt);
      if (Number.isFinite(t)) meta.lastWriteAtMs = t;
    }
    console.log("[MinimalPaper] Rehydrated from disk:", p, "entries=", st.entries.length);
  } catch (e) {
    meta.loadStatus = "error";
    meta.lastLoadAtMs = Date.now();
    console.warn("[MinimalPaper] Load failed:", e instanceof Error ? e.message : e);
  }
}

function buildPersistPayload(st: ProbeState): PersistedFileV1 {
  return {
    version: PERSIST_VERSION,
    savedAt: new Date().toISOString(),
    entries: st.entries.map(e => JSON.parse(JSON.stringify(e)) as MinimalPaperEntry),
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
    lastAssessmentVerdict: st.lastAssessmentVerdict,
  };
}

function writeAtomic(fp: string, data: PersistedFileV1): void {
  const dir = path.dirname(fp);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = `${fp}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data), "utf8");
  fs.renameSync(tmp, fp);
}

function maybePersist(st: ProbeState, force: boolean): void {
  if (isDiskDisabled()) return;
  const meta = getPersistMeta();
  const now = Date.now();
  if (!force && now - meta.lastThrottleWallMs < PERSIST_THROTTLE_MS()) return;
  try {
    writeAtomic(defaultPath(), buildPersistPayload(st));
    meta.lastThrottleWallMs = now;
    meta.lastWriteAtMs = now;
    if (meta.loadStatus === "missing" || meta.loadStatus === "pending") meta.loadStatus = "ok";
  } catch (e) {
    console.warn("[MinimalPaper] Persist failed:", e instanceof Error ? e.message : e);
  }
}

function pickStableCandidates(execDigest: ReturnType<typeof buildPocketExecutionDigest>): CandidateExecutionPocket[] {
  const stable = new Set(execDigest.stableMicroKeysAtScan);
  return execDigest.candidateExecutionPockets.filter(
    c => stable.has(c.microBucketKey) && c.betweenSnapshotsStable && c.eligibleMarketCount > 0,
  );
}

function openEntryKeys(entries: MinimalPaperEntry[]): Set<string> {
  const s = new Set<string>();
  for (const e of entries) {
    if (!e.observedWindow) s.add(e.microBucketKey);
  }
  return s;
}

function trimEntries(st: ProbeState): void {
  const max = MAX_PERSISTED_ENTRIES();
  if (st.entries.length <= max) return;
  st.entries = st.entries.slice(-max);
}

async function runScan(): Promise<void> {
  const st = getState();
  if (st.scanning) {
    st.totalScanSkippedBusy++;
    console.warn("[MinimalPaper] runScan skipped: busy");
    return;
  }
  st.totalScanAttempts++;
  st.currentRunId++;
  const runId = st.currentRunId;
  st.lastScanStartAt = Date.now();
  const t0 = Date.now();

  try {
    st.scanning = true;
    st.scanError = null;

    const execDigest = buildPocketExecutionDigest();
    const econDigest = buildPocketEconomicsDigest();
    const gate = execDigest.executionPromotionAssessment.overallExecutionPromotionVerdict;

    const blockingReasons: string[] = [];
    const supportingReasons: string[] = [];
    let assessmentVerdict: MinimalPaperAssessmentVerdict = "observation_cycle_ok";

    const openWithoutObs = st.entries.filter(e => !e.observedWindow);

    if (gate === "not_ready") {
      blockingReasons.push(
        "executionPromotionAssessment.overallExecutionPromotionVerdict is not_ready — no new paper entries.",
      );
      if (openWithoutObs.length === 0) {
        assessmentVerdict = "blocked_not_ready_gate";
        st.lastAssessmentVerdict = assessmentVerdict;
        st.lastScanTimestamp = Date.now();
        st.lastScanDurationMs = Date.now() - t0;
        st.lastSuccessfulScanAt = st.lastScanTimestamp;
        st.lastScanErrorMessage = null;
        st.lastScanErrorAt = null;
        st.totalScanSuccess++;
        maybePersist(st, true);
        console.log(`[MinimalPaper] Scan #${runId} gate blocked, no open entries`);
        return;
      }
      supportingReasons.push(
        `Gate not_ready but completing observedWindow for ${openWithoutObs.length} open paper log(s).`,
      );
    } else {
      supportingReasons.push(`execution gate allows new paper observation: overallExecutionPromotionVerdict=${gate}`);
    }

    const snap1 = await snapshotPocketFamilyEligibleBucketsForExecutionProbe();
    await new Promise<void>(r => setTimeout(r, FOLLOWUP_GAP_MS()));
    const snap2 = await snapshotPocketFamilyEligibleBucketsForExecutionProbe();

    const map2 = snap2.bucketMap;

    for (const e of st.entries) {
      if (e.observedWindow) continue;
      const rowsNow = map2.get(e.microBucketKey) ?? [];
      const afterLite = marketsToLite(rowsNow.map(r => ({ id: r.id, spread: r.spread, prices: r.prices })));
      const afterMap = new Map(afterLite.map(m => [m.id, m]));
      const { max, notes } = maxAbsPriceDelta(e.entrySnapshot.marketsLite, afterMap);
      const extraNotes = [...notes];
      if (rowsNow.length === 0) {
        extraNotes.push("pocket absent in follow-up Gamma pull");
      }
      let label = outcomeLabel(max);
      if (extraNotes.some(x => x.includes("missing") || x.includes("absent"))) {
        label = "component_missing_in_followup";
      }
      e.observedWindow = {
        observedAt: new Date().toISOString(),
        marketsLiteAfter: afterLite,
        maxAbsPriceDeltaAcrossComponents: max,
        observationalOutcomeLabel: label,
        outcomeNotes: [
          "observational/paper only — not monetized PnL; quote drift vs paper entry snapshot",
          ...extraNotes,
        ],
      };
    }

    if (gate === "not_ready") {
      assessmentVerdict = "blocked_not_ready_gate";
      trimEntries(st);
      st.lastAssessmentVerdict = assessmentVerdict;
      st.lastScanTimestamp = Date.now();
      st.lastScanDurationMs = Date.now() - t0;
      st.lastSuccessfulScanAt = st.lastScanTimestamp;
      st.lastScanErrorMessage = null;
      st.lastScanErrorAt = null;
      st.totalScanSuccess++;
      maybePersist(st, true);
      console.log(`[MinimalPaper] Scan #${runId} gate not_ready; closed open observations only`);
      return;
    }

    const candidates = pickStableCandidates(execDigest);
    if (candidates.length === 0) {
      blockingReasons.push("No stable betweenSnapshotsStable candidate pockets in pocket-execution digest.");
      assessmentVerdict = "blocked_no_stable_candidates";
      trimEntries(st);
      st.lastAssessmentVerdict = assessmentVerdict;
      st.lastScanTimestamp = Date.now();
      st.lastScanDurationMs = Date.now() - t0;
      st.lastSuccessfulScanAt = st.lastScanTimestamp;
      st.totalScanSuccess++;
      maybePersist(st, true);
      console.log(`[MinimalPaper] Scan #${runId} no candidates`);
      return;
    }

    const openKeys = openEntryKeys(st.entries);
    const openCount = openKeys.size;
    let created = 0;
    const maxNew = NEW_ENTRIES_PER_CYCLE();
    const maxOpen = MAX_OPEN_ENTRIES();

    if (openCount >= maxOpen) {
      blockingReasons.push(`open paper entries ${openCount} >= max ${maxOpen}`);
      assessmentVerdict = "idle_capped";
    } else {
      for (const c of candidates) {
        if (created >= maxNew) break;
        if (openKeys.has(c.microBucketKey)) continue;
        if (openCount + created >= maxOpen) break;
        const rows = snap1.bucketMap.get(c.microBucketKey) ?? [];
        if (rows.length === 0) continue;
        const entrySnapshot = { marketsLite: marketsToLite(rows.map(r => ({ id: r.id, spread: r.spread, prices: r.prices }))) };
        const id = `mp-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
        const paperEntryAt = new Date().toISOString();
        const newEntry: MinimalPaperEntry = {
          id,
          microBucketKey: c.microBucketKey,
          paperEntryAt,
          entrySnapshot,
          rationale:
            `Conservative paper log only. Gate=${gate}. Family ${TARGET_FAMILY_KEY}. Stable pocket with dual-snapshot stability from pocket-execution digest. No fill, no sizing, no global execution engine.`,
          structuralConditionsAtEntry: {
            overallExecutionPromotionVerdict: gate,
            pocketEconomicsPromotionVerdict: econDigest.promotionAssessment.overallPromotionVerdict,
            stableMicroKeysAtScanCount: execDigest.stableMicroKeysAtScan.length,
            executionObservationVerdictAtEntry: execDigest.executionObservationVerdict,
          },
        };
        const immRows = map2.get(c.microBucketKey) ?? [];
        if (immRows.length > 0) {
          const immAfter = marketsToLite(immRows.map(r => ({ id: r.id, spread: r.spread, prices: r.prices })));
          const immMap = new Map(immAfter.map(m => [m.id, m]));
          const immDelta = maxAbsPriceDelta(entrySnapshot.marketsLite, immMap);
          let immLabel = outcomeLabel(immDelta.max);
          if (immDelta.notes.some(x => x.includes("missing"))) immLabel = "component_missing_in_followup";
          newEntry.additionalObservedWindows = [{
            windowLabel: "immediate",
            observedAt: new Date().toISOString(),
            gapFromEntryMs: Date.now() - Date.parse(paperEntryAt),
            marketsLiteAfter: immAfter,
            maxAbsPriceDeltaAcrossComponents: immDelta.max,
            observationalOutcomeLabel: immLabel,
            outcomeNotes: ["immediate window (seconds); same scan as entry creation"],
          }];
        }
        st.entries.push(newEntry);
        openKeys.add(c.microBucketKey);
        created++;
        supportingReasons.push(`new paper entry ${id} microBucketKey=${c.microBucketKey.slice(0, 48)}…`);
      }
      if (created > 0) assessmentVerdict = "paper_active";
    }

    trimEntries(st);
    st.lastAssessmentVerdict = assessmentVerdict;
    st.lastScanTimestamp = Date.now();
    st.lastScanDurationMs = Date.now() - t0;
    st.lastSuccessfulScanAt = st.lastScanTimestamp;
    st.lastScanErrorMessage = null;
    st.lastScanErrorAt = null;
    st.totalScanSuccess++;
    maybePersist(st, true);
    console.log(`[MinimalPaper] Scan #${runId} ok assessment=${assessmentVerdict} created=${created}`);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    st.scanError = msg;
    st.lastScanErrorMessage = msg;
    st.lastScanErrorAt = Date.now();
    st.totalScanErrors++;
    console.error(`[MinimalPaper] Scan #${runId} error`, msg);
  } finally {
    st.lastScanEndAt = Date.now();
    st.scanning = false;
  }
}

function scheduleNext(delayMs: number): void {
  const st = getState();
  if (st.scheduledTimeoutId !== null) {
    clearTimeout(st.scheduledTimeoutId);
    st.scheduledTimeoutId = null;
  }
  st.nextScheduledScanAt = Date.now() + delayMs;
  st.scheduledTimeoutId = setTimeout(() => {
    st.scheduledTimeoutId = null;
    void runScan().finally(() => scheduleNext(SCAN_INTERVAL_MS()));
  }, delayMs);
}

export function ensureMinimalPaperExecutionProbe(): void {
  const st = getState();
  if (st.loopStarted) return;
  st.loopStarted = true;
  st.schedulerStartedAt = Date.now();
  console.log("[MinimalPaper] Scheduler started (subordinate to pocket-execution promotion gate)");
  scheduleNext(BOOT_DELAY_MS());
}

export function getMinimalPaperExecutionHealth(): {
  minimalPaperSchedulerRunning: boolean;
  minimalPaperIsScanRunning: boolean;
  minimalPaperLastSuccessfulScanAt: string | null;
  minimalPaperLastErrorAt: string | null;
  minimalPaperSchedulerStartedAt: string | null;
  minimalPaperLastScanStartAt: string | null;
  minimalPaperLastScanEndAt: string | null;
  minimalPaperLastScanErrorMessage: string | null;
  minimalPaperNextScheduledScanAt: string | null;
  minimalPaperCurrentRunId: number;
  minimalPaperTotalScanAttempts: number;
  minimalPaperTotalScanSuccess: number;
  minimalPaperTotalScanErrors: number;
  minimalPaperTotalScanSkippedBusy: number;
  minimalPaperPersistenceEnabled: boolean;
  minimalPaperPersistencePath: string | null;
  minimalPaperLastPersistenceWriteAt: string | null;
  minimalPaperLastPersistenceLoadAt: string | null;
  minimalPaperPersistenceLoadStatus: MinimalPaperPersistenceLoadStatus;
  minimalPaperEntriesCount: number;
  minimalPaperOpenEntriesCount: number;
} {
  const st = getState();
  const p = buildPersistenceDigest();
  let open = 0;
  for (const e of st.entries) {
    if (!e.observedWindow) open++;
  }
  return {
    minimalPaperSchedulerRunning: st.loopStarted,
    minimalPaperIsScanRunning: st.scanning,
    minimalPaperLastSuccessfulScanAt: st.lastSuccessfulScanAt
      ? new Date(st.lastSuccessfulScanAt).toISOString()
      : null,
    minimalPaperLastErrorAt: st.lastScanErrorAt ? new Date(st.lastScanErrorAt).toISOString() : null,
    minimalPaperSchedulerStartedAt: st.schedulerStartedAt
      ? new Date(st.schedulerStartedAt).toISOString()
      : null,
    minimalPaperLastScanStartAt: st.lastScanStartAt ? new Date(st.lastScanStartAt).toISOString() : null,
    minimalPaperLastScanEndAt: st.lastScanEndAt ? new Date(st.lastScanEndAt).toISOString() : null,
    minimalPaperLastScanErrorMessage: st.lastScanErrorMessage,
    minimalPaperNextScheduledScanAt: st.nextScheduledScanAt
      ? new Date(st.nextScheduledScanAt).toISOString()
      : null,
    minimalPaperCurrentRunId: st.currentRunId,
    minimalPaperTotalScanAttempts: st.totalScanAttempts,
    minimalPaperTotalScanSuccess: st.totalScanSuccess,
    minimalPaperTotalScanErrors: st.totalScanErrors,
    minimalPaperTotalScanSkippedBusy: st.totalScanSkippedBusy,
    minimalPaperPersistenceEnabled: p.persistenceEnabled,
    minimalPaperPersistencePath: p.persistencePath,
    minimalPaperLastPersistenceWriteAt: p.lastPersistenceWriteAt,
    minimalPaperLastPersistenceLoadAt: p.lastPersistenceLoadAt,
    minimalPaperPersistenceLoadStatus: p.persistenceLoadStatus,
    minimalPaperEntriesCount: st.entries.length,
    minimalPaperOpenEntriesCount: open,
  };
}

function buildAssessmentDigest(
  st: ProbeState,
  reuseExecDigest?: PocketExecutionDigest,
): MinimalPaperExecutionAssessmentDigest {
  const execDigest = reuseExecDigest ?? buildPocketExecutionDigest();
  const gate = execDigest.executionPromotionAssessment.overallExecutionPromotionVerdict;
  let open = 0;
  let obs = 0;
  for (const e of st.entries) {
    if (!e.observedWindow) open++;
    else obs++;
  }
  const blockingReasons: string[] = [];
  const supportingReasons: string[] = [];
  let assessmentVerdict: MinimalPaperAssessmentVerdict =
    st.lastAssessmentVerdict ?? "observation_cycle_ok";

  if (gate === "not_ready") {
    blockingReasons.push("Gate not_ready — probe não abre novas entradas paper.");
    assessmentVerdict = "blocked_not_ready_gate";
  } else {
    supportingReasons.push(`Gate permite observação paper: ${gate}`);
  }

  const candidates = pickStableCandidates(execDigest);
  if (gate !== "not_ready" && candidates.length === 0) {
    blockingReasons.push("Sem candidatos estáveis no digest pocket-execution.");
  }

  if (open >= MAX_OPEN_ENTRIES()) {
    blockingReasons.push(`Entradas abertas ${open} (aguardam observedWindow no próximo ciclo).`);
  }

  return {
    assessmentVerdict,
    blockingReasons,
    supportingReasons,
    gateOverallExecutionPromotionVerdict: gate,
    openPaperEntriesCount: open,
    observedEntriesCount: obs,
  };
}

/** Snapshot só leitura para agregações externas (ex. universo irmão). Não mutar o array. */
export function getMinimalPaperExecutionEntriesReadonly(): readonly MinimalPaperEntry[] {
  return getState().entries;
}

export function buildMinimalPaperExecutionDigest(options?: {
  /** Evita segundo buildPocketExecutionDigest no mesmo request (ex.: system-ladder). */
  reusePocketExecutionDigest?: PocketExecutionDigest;
}): MinimalPaperExecutionDigest {
  const st = getState();
  const scanStatus: MinimalPaperExecutionDigest["scanStatus"] =
    st.scanError ? "error" : st.scanning ? "scanning" : st.lastScanTimestamp ? "completed" : "idle";

  return {
    computedAt: new Date().toISOString(),
    probeVersion: "minimal-paper-execution-v1",
    scanStatus,
    targetFamilyKey: TARGET_FAMILY_KEY,
    note:
      "Paper isolado: só novas entradas se pocket-execution executionPromotionAssessment ≠ not_ready; família other:price_above:>3m; candidatos estáveis do digest pocket-execution. Sem execution engine global. Outcomes são observacionais (drift de quotes vs snapshot de entrada). microEdgeAssessment: proxy conservador não monetizado sobre episódios fechados (spreads/preços entry vs observedWindow). Persistência: minimal-paper-execution-state.json sob PAPER_STATE_DIR ou cwd/.paper (MINIMAL_PAPER_STATE_PATH, MINIMAL_PAPER_DISABLE_DISK=1).",
    entries: st.entries.map(e => JSON.parse(JSON.stringify(e)) as MinimalPaperEntry),
    minimalPaperExecutionAssessment: buildAssessmentDigest(st, options?.reusePocketExecutionDigest),
    microEdgeAssessment: buildMicroEdgeAssessmentFromEntries(st.entries),
    capturabilityAssessment: buildCapturabilityAssessment(st.entries),
    multiWindowAssessment: buildMultiWindowAssessment(st.entries),
    refinedOutcomeClassification: buildRefinedOutcomeClassification(st.entries),
    persistence: buildPersistenceDigest(),
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
  };
}
