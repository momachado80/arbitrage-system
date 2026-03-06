import { writeFileSync, readFileSync, renameSync, mkdirSync, existsSync } from "fs";
import { join, dirname } from "path";
import type { GraphEpisode } from "./graphEpisodeTracker";

const SCHEMA_VERSION = "1";
const MAX_PERSISTED_ACTIVE = 200;
const MAX_PERSISTED_CLOSED = 500;
const THROTTLE_MS = 15_000;

interface PersistedSnapshot {
  schemaVersion: string;
  savedAt: string;
  active: GraphEpisode[];
  recentClosed: GraphEpisode[];
}

function getFilePath(): string {
  const tmpDir = process.env.TMPDIR || process.env.TEMP || "/tmp";
  return join(tmpDir, "polymarket-graph-episodes.json");
}

let lastWriteMs = 0;

export function persistEpisodes(
  active: GraphEpisode[],
  recentClosed: GraphEpisode[]
): void {
  const now = Date.now();
  if (now - lastWriteMs < THROTTLE_MS) return;

  try {
    const filePath = getFilePath();
    const dir = dirname(filePath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    const snapshot: PersistedSnapshot = {
      schemaVersion: SCHEMA_VERSION,
      savedAt: new Date(now).toISOString(),
      active: active.slice(0, MAX_PERSISTED_ACTIVE).map(stripForPersistence),
      recentClosed: recentClosed.slice(-MAX_PERSISTED_CLOSED).map(stripForPersistence),
    };

    const json = JSON.stringify(snapshot);
    const tmpPath = filePath + ".tmp";
    writeFileSync(tmpPath, json, "utf-8");
    renameSync(tmpPath, filePath);
    lastWriteMs = now;
  } catch (err) {
    console.warn("[GraphPersistence] Write failed (non-fatal):", (err as Error).message);
  }
}

function stripForPersistence(ep: GraphEpisode): GraphEpisode {
  const copy = { ...ep };
  if (copy.recentSamples && copy.recentSamples.length > 5) {
    copy.recentSamples = copy.recentSamples.slice(-5);
  }
  return copy;
}

export function restoreEpisodes(): {
  active: GraphEpisode[];
  recentClosed: GraphEpisode[];
} | null {
  try {
    const filePath = getFilePath();
    if (!existsSync(filePath)) return null;

    const raw = readFileSync(filePath, "utf-8");
    const parsed: PersistedSnapshot = JSON.parse(raw);

    if (parsed.schemaVersion !== SCHEMA_VERSION) {
      console.warn(
        `[GraphPersistence] Schema mismatch: got ${parsed.schemaVersion}, expected ${SCHEMA_VERSION}. Starting clean.`
      );
      return null;
    }

    if (!Array.isArray(parsed.active) || !Array.isArray(parsed.recentClosed)) {
      console.warn("[GraphPersistence] Corrupted structure. Starting clean.");
      return null;
    }

    const activeValid = parsed.active.filter(
      (ep) => ep && ep.episodeId && ep.key && ep.status === "active"
    );
    const closedValid = parsed.recentClosed.filter(
      (ep) => ep && ep.episodeId && ep.key && ep.status === "closed"
    );

    console.log(
      `[GraphPersistence] Restored ${activeValid.length} active + ${closedValid.length} closed episodes from disk (saved ${parsed.savedAt})`
    );

    return { active: activeValid, recentClosed: closedValid };
  } catch (err) {
    console.warn("[GraphPersistence] Restore failed (non-fatal):", (err as Error).message);
    return null;
  }
}
