/**
 * Metadados operacionais por experimento Railway (ficheiro JSON à parte do JSONL append-only).
 * Não altera linhas JSONL nem lógica de estratégia — só governa tempo, contagens esperadas e saúde da recolha.
 */

import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";
import { defaultPaperTrailStatePath, resolvePaperStateDir } from "./paperStateDir";

export const OPERATIONAL_META_PROBE_VERSION = "live-experiment-operational-meta-v1" as const;

export interface LiveExperimentOperationalMetaStored {
  probeVersion: typeof OPERATIONAL_META_PROBE_VERSION;
  experimentId: string;
  serviceName: string;
  runId: string;
  runnerStartedAt: string;
  firstObservationAt: string | null;
  lastObservationAt: string | null;
  intervalMs: number;
  totalObservations: number;
  runtimeHours: number;
  restartCount: number;
  historyPath: string;
  railwayProject: string | null;
  railwayEnvironment: string | null;
  isHealthy: boolean;
  expectedObservationsByNow: number;
  observationsMissingEstimate: number;
  metaWrittenAt: string;
}

export type CollectionHealthVerdict =
  | "healthy"
  | "behind_schedule"
  | "stalled"
  | "no_history_file"
  | "metadata_missing"
  | "unknown";

/** Tolerância de ticks em falta antes de marcar behind_schedule (operacional). */
export const MISSING_TICK_TOLERANCE = 3;

/** Factor × interval para considerar stalled sem novas observações. */
export const STALL_INTERVAL_FACTOR = 2.5;

export function resolveOperationalMetaPath(
  envOverride: string | undefined,
  filename: string,
  cwd: string = process.cwd(),
): string {
  const raw = envOverride?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(filename, cwd);
}

function readMeta(fp: string): LiveExperimentOperationalMetaStored | null {
  try {
    if (!fs.existsSync(fp)) return null;
    const t = fs.readFileSync(fp, "utf8");
    const j = JSON.parse(t) as LiveExperimentOperationalMetaStored;
    if (j && j.probeVersion === OPERATIONAL_META_PROBE_VERSION) return j;
    return null;
  } catch {
    return null;
  }
}

function writeMetaAtomic(fp: string, obj: LiveExperimentOperationalMetaStored): void {
  const dir = path.dirname(fp);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = `${fp}.tmp.${process.pid}`;
  fs.writeFileSync(tmp, `${JSON.stringify(obj, null, 2)}\n`, "utf8");
  fs.renameSync(tmp, fp);
}

/**
 * Percorre JSONL e devolve contagens e limites temporais (campo isoTimestamp).
 */
export function scanJsonlIsoSummary(historyPath: string): {
  totalLines: number;
  firstObservationAt: string | null;
  lastObservationAt: string | null;
} {
  if (!fs.existsSync(historyPath)) {
    return { totalLines: 0, firstObservationAt: null, lastObservationAt: null };
  }
  const raw = fs.readFileSync(historyPath, "utf8");
  const lines = raw.split("\n").map(l => l.trim()).filter(Boolean);
  let first: string | null = null;
  let last: string | null = null;
  for (const line of lines) {
    try {
      const j = JSON.parse(line) as { isoTimestamp?: string };
      const iso = typeof j.isoTimestamp === "string" ? j.isoTimestamp : "";
      if (!iso) continue;
      if (!first || iso < first) first = iso;
      if (!last || iso > last) last = iso;
    } catch {
      continue;
    }
  }
  return {
    totalLines: lines.length,
    firstObservationAt: first,
    lastObservationAt: last,
  };
}

function computeExpectedAndMissing(
  runnerStartedAt: string,
  intervalMs: number,
  totalObservations: number,
): { expectedObservationsByNow: number; observationsMissingEstimate: number; runtimeHours: number } {
  const t0 = Date.parse(runnerStartedAt);
  const now = Date.now();
  const elapsed = Math.max(0, now - t0);
  const runtimeHours = Math.round((elapsed / 3_600_000) * 10_000) / 10_000;
  const expectedObservationsByNow =
    intervalMs > 0 ? 1 + Math.floor(elapsed / intervalMs) : totalObservations;
  const observationsMissingEstimate = Math.max(0, expectedObservationsByNow - totalObservations);
  return { expectedObservationsByNow, observationsMissingEstimate, runtimeHours };
}

function computeIsHealthy(
  observationsMissingEstimate: number,
  lastObservationAt: string | null,
  intervalMs: number,
): boolean {
  if (observationsMissingEstimate > MISSING_TICK_TOLERANCE) return false;
  if (lastObservationAt && intervalMs > 0) {
    const age = Date.now() - Date.parse(lastObservationAt);
    if (age > intervalMs * STALL_INTERVAL_FACTOR) return false;
  }
  return true;
}

export function deriveCollectionHealthVerdict(input: {
  historyFileExists: boolean;
  metaFileExists: boolean;
  totalLines: number;
  lastObservationAt: string | null;
  intervalMs: number | null;
  observationsMissingEstimate: number | null;
}): CollectionHealthVerdict {
  if (!input.historyFileExists) return "no_history_file";
  if (!input.metaFileExists) return "metadata_missing";
  if (input.totalLines === 0) return "unknown";
  const miss = input.observationsMissingEstimate;
  if (miss != null && miss > MISSING_TICK_TOLERANCE) return "behind_schedule";
  if (input.lastObservationAt && input.intervalMs && input.intervalMs > 0) {
    const age = Date.now() - Date.parse(input.lastObservationAt);
    if (age > input.intervalMs * STALL_INTERVAL_FACTOR) return "stalled";
  }
  return "healthy";
}

export interface RegisterRunnerParams {
  cwd?: string;
  experimentId: string;
  serviceName: string;
  metaFilename: string;
  historyFilename: string;
  metaEnvOverride?: string;
  intervalMs: number;
}

/** Chamada uma vez no arranque do processo worker (antes do loop). */
export function registerLiveExperimentRunner(p: RegisterRunnerParams): void {
  const cwd = p.cwd ?? process.cwd();
  const metaPath = resolveOperationalMetaPath(p.metaEnvOverride, p.metaFilename, cwd);
  const historyPath = defaultPaperTrailStatePath(p.historyFilename, cwd);
  const prev = readMeta(metaPath);
  const now = new Date().toISOString();
  const restartCount = prev ? (prev.restartCount ?? 0) + 1 : 0;

  const base: LiveExperimentOperationalMetaStored = {
    probeVersion: OPERATIONAL_META_PROBE_VERSION,
    experimentId: p.experimentId,
    serviceName: p.serviceName,
    runId: randomUUID(),
    runnerStartedAt: now,
    firstObservationAt: null,
    lastObservationAt: null,
    intervalMs: p.intervalMs,
    totalObservations: 0,
    runtimeHours: 0,
    restartCount,
    historyPath,
    railwayProject: process.env.RAILWAY_PROJECT_NAME?.trim() || null,
    railwayEnvironment: process.env.RAILWAY_ENVIRONMENT_NAME?.trim() || null,
    isHealthy: false,
    expectedObservationsByNow: 0,
    observationsMissingEstimate: 0,
    metaWrittenAt: now,
  };

  writeMetaAtomic(metaPath, base);
}

export interface SyncMetaParams extends RegisterRunnerParams {}

/** Chamada após cada append ao JSONL (mantém contagens alinhadas ao disco). */
export function syncLiveExperimentOperationalMeta(p: SyncMetaParams): void {
  const cwd = p.cwd ?? process.cwd();
  const metaPath = resolveOperationalMetaPath(p.metaEnvOverride, p.metaFilename, cwd);
  const historyPath = defaultPaperTrailStatePath(p.historyFilename, cwd);
  const meta = readMeta(metaPath);
  if (!meta) {
    registerLiveExperimentRunner(p);
    return syncLiveExperimentOperationalMeta(p);
  }

  const scan = scanJsonlIsoSummary(historyPath);
  const { expectedObservationsByNow, observationsMissingEstimate, runtimeHours } =
    computeExpectedAndMissing(meta.runnerStartedAt, p.intervalMs, scan.totalLines);

  const isHealthy = computeIsHealthy(
    observationsMissingEstimate,
    scan.lastObservationAt,
    p.intervalMs,
  );

  const next: LiveExperimentOperationalMetaStored = {
    ...meta,
    intervalMs: p.intervalMs,
    historyPath,
    totalObservations: scan.totalLines,
    firstObservationAt: scan.firstObservationAt,
    lastObservationAt: scan.lastObservationAt,
    runtimeHours,
    expectedObservationsByNow,
    observationsMissingEstimate,
    isHealthy,
    railwayProject: process.env.RAILWAY_PROJECT_NAME?.trim() || meta.railwayProject,
    railwayEnvironment: process.env.RAILWAY_ENVIRONMENT_NAME?.trim() || meta.railwayEnvironment,
    metaWrittenAt: new Date().toISOString(),
  };

  writeMetaAtomic(metaPath, next);
}

export function readOperationalMetaFile(
  metaPath: string,
): LiveExperimentOperationalMetaStored | null {
  return readMeta(metaPath);
}
