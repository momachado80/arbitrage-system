/**
 * Caminhos e higiene de ficheiros .paper dos probes pocket-economics, pocket-execution,
 * minimal-paper-execution. Mantido alinhado com defaultPersistencePath / defaultExecPersistencePath /
 * minimalPaper defaultPath (env: POCKET_ECON_STATE_PATH, POCKET_EXEC_STATE_PATH, MINIMAL_PAPER_STATE_PATH, PAPER_STATE_DIR).
 * Não altera probes nem regras de negócio.
 */

import fs from "fs";
import path from "path";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES } from "./paperStateDir";

export const PAPER_PERSISTED_PROBE_IDS = [
  "pocket-economics",
  "pocket-execution",
  "minimal-paper-execution",
] as const;

export type PaperPersistedProbeId = (typeof PAPER_PERSISTED_PROBE_IDS)[number];

export function isPaperPersistedProbeId(s: string): s is PaperPersistedProbeId {
  return (PAPER_PERSISTED_PROBE_IDS as readonly string[]).includes(s);
}

export function resolvePaperPersistedStatePath(
  probe: PaperPersistedProbeId,
  cwd: string = process.cwd(),
): string {
  switch (probe) {
    case "pocket-economics": {
      const raw = process.env.POCKET_ECON_STATE_PATH?.trim();
      return raw ? path.resolve(raw) : defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.pocketEconomics, cwd);
    }
    case "pocket-execution": {
      const raw = process.env.POCKET_EXEC_STATE_PATH?.trim();
      return raw ? path.resolve(raw) : defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.pocketExecution, cwd);
    }
    case "minimal-paper-execution": {
      const raw = process.env.MINIMAL_PAPER_STATE_PATH?.trim();
      return raw
        ? path.resolve(raw)
        : defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.minimalPaperExecution, cwd);
    }
    default: {
      const _x: never = probe;
      return _x;
    }
  }
}

const TEST_STRING_RE =
  /rehydrat|rehydrate|mp-rehydrate|synthetic|demonstra(c|ç)(a|ã)o|prova de reidrata|demo:|test-only|fake-bucket/i;

function pushHint(hints: string[], msg: string, cap = 6): void {
  if (hints.length >= cap) return;
  hints.push(msg);
}

function scanStringField(hints: string[], label: string, v: unknown): void {
  if (typeof v !== "string" || v.length === 0) return;
  if (TEST_STRING_RE.test(v)) pushHint(hints, `${label}: test/demo-like substring`);
}

function analyzeMinimalPaper(j: Record<string, unknown>, hints: string[]): void {
  const entries = j.entries;
  if (!Array.isArray(entries)) return;
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    if (!e || typeof e !== "object") continue;
    const o = e as Record<string, unknown>;
    scanStringField(hints, `entries[${i}].id`, o.id);
    scanStringField(hints, `entries[${i}].microBucketKey`, o.microBucketKey);
    scanStringField(hints, `entries[${i}].rationale`, o.rationale);
    const ow = o.observedWindow;
    if (ow && typeof ow === "object") {
      const on = (ow as { outcomeNotes?: unknown }).outcomeNotes;
      if (Array.isArray(on)) {
        for (let k = 0; k < on.length; k++) {
          scanStringField(hints, `entries[${i}].observedWindow.outcomeNotes[${k}]`, on[k]);
        }
      }
    }
  }
}

function analyzePocketEconomics(j: Record<string, unknown>, hints: string[]): void {
  const lite = j.lastDigestBucketsLite;
  if (!Array.isArray(lite)) return;
  for (let i = 0; i < lite.length; i++) {
    const b = lite[i];
    if (!b || typeof b !== "object") continue;
    scanStringField(hints, `lastDigestBucketsLite[${i}].microBucketKey`, (b as { microBucketKey?: unknown }).microBucketKey);
  }
}

function analyzePocketExecution(j: Record<string, unknown>, hints: string[]): void {
  const cycles = j.temporalCycles;
  if (!Array.isArray(cycles)) return;
  for (let c = 0; c < cycles.length; c++) {
    const cy = cycles[c];
    if (!cy || typeof cy !== "object") continue;
    const sums = (cy as { pocketReadinessSummaries?: unknown }).pocketReadinessSummaries;
    if (!Array.isArray(sums)) continue;
    for (let s = 0; s < sums.length; s++) {
      const row = sums[s];
      if (!row || typeof row !== "object") continue;
      scanStringField(
        hints,
        `temporalCycles[${c}].pocketReadinessSummaries[${s}].microBucketKey`,
        (row as { microBucketKey?: unknown }).microBucketKey,
      );
    }
  }
}

/** Heurística conservadora: só marcadores explícitos em strings conhecidas (sem alterar thresholds). */
export function analyzeParsedStateForTestLikeMarkers(
  parsed: unknown,
  probe: PaperPersistedProbeId,
): { appearsTestLike: boolean; hints: string[] } {
  const hints: string[] = [];
  if (!parsed || typeof parsed !== "object") {
    return { appearsTestLike: false, hints: [] };
  }
  const j = parsed as Record<string, unknown>;

  const savedAt = j.savedAt;
  const scanMsFields = ["lastScanStartAt", "lastScanEndAt", "lastSuccessfulScanAt", "lastScanTimestamp"] as const;
  if (typeof savedAt === "string") {
    const s = Date.parse(savedAt);
    if (Number.isFinite(s)) {
      for (const k of scanMsFields) {
        const v = j[k];
        if (typeof v !== "number" || !Number.isFinite(v)) continue;
        const delta = Math.abs(v - s);
        // Conservador: só avisar desvios grandes (evita falso positivo com ficheiros antigos legítimos).
        if (delta > 400 * 86400000) {
          pushHint(hints, `timestamp_skew: |${k} - savedAt| > 400d (possible stale/injected data)`);
        }
      }
    }
  }

  switch (probe) {
    case "minimal-paper-execution":
      analyzeMinimalPaper(j, hints);
      break;
    case "pocket-economics":
      analyzePocketEconomics(j, hints);
      break;
    case "pocket-execution":
      analyzePocketExecution(j, hints);
      break;
    default: {
      const _p: never = probe;
      return _p;
    }
  }

  return { appearsTestLike: hints.length > 0, hints };
}

export interface PaperPersistedProbeHygieneRow {
  probe: PaperPersistedProbeId;
  statePath: string;
  fileExists: boolean;
  appearsTestLike: boolean;
  hints: string[];
}

export function buildPaperPersistedStateHygieneRows(cwd?: string): PaperPersistedProbeHygieneRow[] {
  const base = cwd ?? process.cwd();
  return PAPER_PERSISTED_PROBE_IDS.map(probe => {
    const statePath = resolvePaperPersistedStatePath(probe, base);
    if (!fs.existsSync(statePath)) {
      return { probe, statePath, fileExists: false, appearsTestLike: false, hints: [] };
    }
    try {
      const raw = fs.readFileSync(statePath, "utf8");
      const parsed = JSON.parse(raw) as unknown;
      const { appearsTestLike, hints } = analyzeParsedStateForTestLikeMarkers(parsed, probe);
      return { probe, statePath, fileExists: true, appearsTestLike, hints: hints.slice(0, 6) };
    } catch {
      return {
        probe,
        statePath,
        fileExists: true,
        appearsTestLike: false,
        hints: ["unreadable_or_invalid_json"],
      };
    }
  });
}

/** Resumo compacto para healthz. */
export function buildPaperPersistedStateHygieneHealth(cwd?: string): {
  paperPersistedStateAnyAppearsTestLike: boolean;
  paperPersistedStateHygieneProbes: PaperPersistedProbeHygieneRow[];
} {
  const rows = buildPaperPersistedStateHygieneRows(cwd);
  return {
    paperPersistedStateAnyAppearsTestLike: rows.some(r => r.appearsTestLike),
    paperPersistedStateHygieneProbes: rows,
  };
}
