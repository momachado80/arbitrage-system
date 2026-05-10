/**
 * Shadow Closed Trade Persistence — durable storage for closed shadow trades.
 * Survives Railway redeploys when SHADOW_PERSISTENCE_PATH points to a volume (e.g. /data).
 * Set SHADOW_PERSISTENCE_PATH=/data and add a Railway Volume at /data.
 * Does NOT change trading logic; only reads/writes closed trade history.
 */

import { writeFileSync, readFileSync, renameSync, mkdirSync, existsSync } from "fs";
import { join, dirname } from "path";
import type { ShadowTrade } from "./shadowSimulationStore";

const SCHEMA_VERSION = "1";
const MAX_CLOSED_PER_PROFILE = 500;
const THROTTLE_MS = 5_000;

export interface PersistedClosedTradesSnapshot {
  schemaVersion: string;
  savedAt: string;
  byProfile: Record<string, ShadowTrade[]>;
}

function getFilePath(): string {
  const base =
    process.env.SHADOW_PERSISTENCE_PATH ||
    process.env.DATA_PATH ||
    process.cwd();
  return join(base, "shadow-closed-trades.json");
}

let lastWriteMs = 0;

/**
 * Persist closed trades for all profiles.
 * Default: throttled to 1 write per THROTTLE_MS (5s).
 * `opts.force=true` ignora o throttle — usado pelo flush de shutdown no botRunner.
 */
export function persistClosedTrades(
  byProfile: Record<string, ShadowTrade[]>,
  opts?: { force?: boolean },
): void {
  const now = Date.now();
  const force = opts?.force === true;
  if (!force && now - lastWriteMs < THROTTLE_MS) return;

  try {
    const filePath = getFilePath();
    const dir = dirname(filePath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    const trimmed: Record<string, ShadowTrade[]> = {};
    for (const [profileId, trades] of Object.entries(byProfile)) {
      const closed = trades.filter((t) => t.status === "closed" && t.closedAt);
      if (closed.length > 0) {
        trimmed[profileId] = closed.slice(-MAX_CLOSED_PER_PROFILE);
      }
    }
    if (Object.keys(trimmed).length === 0) return;

    const snapshot: PersistedClosedTradesSnapshot = {
      schemaVersion: SCHEMA_VERSION,
      savedAt: new Date(now).toISOString(),
      byProfile: trimmed,
    };

    const json = JSON.stringify(snapshot);
    const tmpPath = filePath + ".tmp";
    writeFileSync(tmpPath, json, "utf-8");
    renameSync(tmpPath, filePath);
    lastWriteMs = now;
  } catch (err) {
    console.warn(
      "[ShadowPersistence] Write failed (non-fatal):",
      (err as Error).message
    );
  }
}

/**
 * Restore persisted closed trades. Returns null if no file or invalid.
 */
export function restoreClosedTrades(): PersistedClosedTradesSnapshot | null {
  try {
    const filePath = getFilePath();
    if (!existsSync(filePath)) return null;

    const raw = readFileSync(filePath, "utf-8");
    const parsed: PersistedClosedTradesSnapshot = JSON.parse(raw);

    if (parsed.schemaVersion !== SCHEMA_VERSION) {
      console.warn(
        `[ShadowPersistence] Schema mismatch: got ${parsed.schemaVersion}, expected ${SCHEMA_VERSION}. Skipping restore.`
      );
      return null;
    }

    if (!parsed.byProfile || typeof parsed.byProfile !== "object") {
      console.warn("[ShadowPersistence] Invalid structure. Skipping restore.");
      return null;
    }

    let total = 0;
    for (const trades of Object.values(parsed.byProfile)) {
      if (Array.isArray(trades)) {
        total += trades.filter(
          (t) => t && t.tradeId && t.status === "closed" && t.closedAt
        ).length;
      }
    }

    console.log(
      `[ShadowPersistence] Restored ${total} closed trades from ${Object.keys(parsed.byProfile).length} profiles (saved ${parsed.savedAt})`
    );

    return parsed;
  } catch (err) {
    console.warn(
      "[ShadowPersistence] Restore failed (non-fatal):",
      (err as Error).message
    );
    return null;
  }
}

/** Check if persistence file exists (for observability). */
export function isPersistenceAvailable(): boolean {
  try {
    const filePath = getFilePath();
    return existsSync(filePath);
  } catch {
    return false;
  }
}

/** Path used for persistence (for observability). */
export function getPersistencePath(): string {
  return getFilePath();
}
