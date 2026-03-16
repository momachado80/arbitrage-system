/**
 * Experiment Ledger Persistence — durable storage for experiment history.
 * Reuses SHADOW_PERSISTENCE_PATH convention. Does not touch shadow closed trades.
 */

import { writeFileSync, readFileSync, mkdirSync, existsSync, renameSync } from "fs";
import { join, dirname } from "path";

const SCHEMA_VERSION = "1";
const FILENAME = "experiment-ledger.json";

function getBasePath(): string {
  return (
    process.env.SHADOW_PERSISTENCE_PATH ||
    process.env.DATA_PATH ||
    process.cwd()
  );
}

function getLedgerPath(): string {
  return join(getBasePath(), FILENAME);
}

export interface PersistedExperimentLedger {
  schemaVersion: string;
  savedAt: string;
  entries: ExperimentLedgerEntry[];
}

export interface ExperimentLedgerEntry {
  experimentId: string;
  challengerProfileId: string;
  familyId: string;
  baseProfileId: string;
  hypothesis: string;
  isolatedVariable: string;
  activationMode: "manual" | "automated";
  status: string;
  startTimestamp: string;
  endTimestamp?: string;
  minimumReadoutThreshold?: number;
  observedSummaryMetrics?: {
    closedTrades?: number;
    avgRealizedPnL?: number;
    winRate?: number;
    avgHoldingTimeMs?: number;
    avgFilledCapital?: number;
    avgFillRatio?: number;
  };
  decision?: "successful" | "failed" | "inconclusive" | "superseded";
  decisionReason?: string;
  supersedesExperimentId?: string;
  invalidatedByExperimentId?: string;
  notes?: string;
}

let lastWriteMs = 0;
const THROTTLE_MS = 2000;

/**
 * Persist ledger entries. Throttled to avoid excessive writes.
 */
export function persistExperimentLedger(entries: ExperimentLedgerEntry[]): void {
  const now = Date.now();
  if (now - lastWriteMs < THROTTLE_MS) return;

  try {
    const filePath = getLedgerPath();
    const dir = dirname(filePath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    const snapshot: PersistedExperimentLedger = {
      schemaVersion: SCHEMA_VERSION,
      savedAt: new Date(now).toISOString(),
      entries,
    };

    const json = JSON.stringify(snapshot, null, 2);
    const tmpPath = filePath + ".tmp";
    writeFileSync(tmpPath, json, "utf-8");
    renameSync(tmpPath, filePath);
    lastWriteMs = now;
  } catch (err) {
    console.warn("[ExperimentLedger] Write failed (non-fatal):", (err as Error).message);
  }
}

/**
 * Restore ledger from disk. Returns empty array if no file or invalid.
 */
export function restoreExperimentLedger(): ExperimentLedgerEntry[] {
  try {
    const filePath = getLedgerPath();
    if (!existsSync(filePath)) return [];

    const raw = readFileSync(filePath, "utf-8");
    const parsed: PersistedExperimentLedger = JSON.parse(raw);

    if (parsed.schemaVersion !== SCHEMA_VERSION) {
      console.warn(`[ExperimentLedger] Schema mismatch: got ${parsed.schemaVersion}. Skipping restore.`);
      return [];
    }

    if (!Array.isArray(parsed.entries)) {
      console.warn("[ExperimentLedger] Invalid structure. Skipping restore.");
      return [];
    }

    console.log(`[ExperimentLedger] Restored ${parsed.entries.length} entries (saved ${parsed.savedAt})`);
    return parsed.entries;
  } catch (err) {
    console.warn("[ExperimentLedger] Restore failed (non-fatal):", (err as Error).message);
    return [];
  }
}

export function getExperimentLedgerPath(): string {
  return getLedgerPath();
}

export function isExperimentLedgerAvailable(): boolean {
  try {
    return existsSync(getLedgerPath());
  } catch {
    return false;
  }
}
