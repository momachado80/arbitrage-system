/**
 * Histórico leve e persistido só para o agregado system-ladder (não toca nos probes).
 * Ficheiro: system-ladder-history.json sob PAPER_STATE_DIR ou cwd/.paper (SYSTEM_LADDER_HISTORY_PATH, SYSTEM_LADDER_HISTORY_DISABLE_DISK=1).
 */

import fs from "fs";
import path from "path";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES } from "./paperStateDir";

const PERSIST_VERSION = 1 as const;

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}

const MAX_SNAPSHOTS = () =>
  Math.min(50, Math.max(5, envNum("SYSTEM_LADDER_HISTORY_MAX_SNAPSHOTS", 15)));

export type LadderTemporalVerdictStored = "consistent" | "partially_stale" | "not_comparable_yet";

export type LadderTrajectoryAssessmentVerdict =
  | "improving"
  | "stagnant"
  | "oscillating"
  | "blocked_by_temporal_inconsistency";

export interface LadderSnapshotLite {
  at: string;
  recurrentPocketExists: boolean;
  economicsPromotionVerdict: string;
  executionObservationVerdict: string;
  executionPromotionVerdict: string;
  minimalPaperExecutionAssessmentVerdict: string;
  temporalConsistencyVerdict: LadderTemporalVerdictStored;
  currentStage: string;
}

interface PersistedFileV1 {
  version: typeof PERSIST_VERSION;
  savedAt: string;
  snapshots: LadderSnapshotLite[];
}

function isDiskDisabled(): boolean {
  return process.env.SYSTEM_LADDER_HISTORY_DISABLE_DISK === "1";
}

export function defaultLadderHistoryPath(cwd: string = process.cwd()): string {
  const raw = process.env.SYSTEM_LADDER_HISTORY_PATH?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.systemLadderHistory, cwd);
}

function sanitizeSnapshots(raw: unknown): LadderSnapshotLite[] {
  if (!Array.isArray(raw)) return [];
  const out: LadderSnapshotLite[] = [];
  const tvOk = new Set<string>(["consistent", "partially_stale", "not_comparable_yet"]);
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const o = item as Record<string, unknown>;
    const at = o.at;
    if (typeof at !== "string" || at.length < 8) continue;
    const t = o.temporalConsistencyVerdict;
    if (typeof t !== "string" || !tvOk.has(t)) continue;
    out.push({
      at,
      recurrentPocketExists: Boolean(o.recurrentPocketExists),
      economicsPromotionVerdict:
        typeof o.economicsPromotionVerdict === "string" ? o.economicsPromotionVerdict.slice(0, 64) : "unknown",
      executionObservationVerdict:
        typeof o.executionObservationVerdict === "string" ? o.executionObservationVerdict.slice(0, 64) : "unknown",
      executionPromotionVerdict:
        typeof o.executionPromotionVerdict === "string" ? o.executionPromotionVerdict.slice(0, 64) : "unknown",
      minimalPaperExecutionAssessmentVerdict:
        typeof o.minimalPaperExecutionAssessmentVerdict === "string"
          ? o.minimalPaperExecutionAssessmentVerdict.slice(0, 64)
          : "unknown",
      temporalConsistencyVerdict: t as LadderTemporalVerdictStored,
      currentStage: typeof o.currentStage === "string" ? o.currentStage.slice(0, 500) : "",
    });
  }
  return out.slice(-MAX_SNAPSHOTS());
}

export function loadLadderHistorySnapshots(cwd?: string): LadderSnapshotLite[] {
  if (isDiskDisabled()) return [];
  const fp = defaultLadderHistoryPath(cwd);
  try {
    if (!fs.existsSync(fp)) return [];
    const j = JSON.parse(fs.readFileSync(fp, "utf8")) as Partial<PersistedFileV1>;
    if (j.version !== PERSIST_VERSION) return [];
    return sanitizeSnapshots(j.snapshots);
  } catch {
    return [];
  }
}

function snapshotPayloadEqual(a: LadderSnapshotLite, b: LadderSnapshotLite): boolean {
  return (
    a.recurrentPocketExists === b.recurrentPocketExists &&
    a.economicsPromotionVerdict === b.economicsPromotionVerdict &&
    a.executionObservationVerdict === b.executionObservationVerdict &&
    a.executionPromotionVerdict === b.executionPromotionVerdict &&
    a.minimalPaperExecutionAssessmentVerdict === b.minimalPaperExecutionAssessmentVerdict &&
    a.temporalConsistencyVerdict === b.temporalConsistencyVerdict &&
    a.currentStage === b.currentStage
  );
}

export function appendLadderSnapshotIfChanged(
  existing: LadderSnapshotLite[],
  snap: LadderSnapshotLite,
): LadderSnapshotLite[] {
  const last = existing[existing.length - 1];
  if (last && snapshotPayloadEqual(last, snap)) return existing;
  return [...existing, snap].slice(-MAX_SNAPSHOTS());
}

function writeAtomic(fp: string, data: PersistedFileV1): void {
  const dir = path.dirname(fp);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = `${fp}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data), "utf8");
  fs.renameSync(tmp, fp);
}

export function persistLadderHistorySnapshots(snapshots: LadderSnapshotLite[], cwd?: string): void {
  if (isDiskDisabled()) return;
  try {
    writeAtomic(defaultLadderHistoryPath(cwd), {
      version: PERSIST_VERSION,
      savedAt: new Date().toISOString(),
      snapshots,
    });
  } catch (e) {
    console.warn("[SystemLadderHistory] persist failed:", e instanceof Error ? e.message : e);
  }
}

export function parseStageDegreeFromSummary(stage: string): number | null {
  const m = stage.match(/Degrau\s+(\d+)/i);
  if (m) {
    const n = parseInt(m[1]!, 10);
    return Number.isFinite(n) ? n : null;
  }
  if (/Degrau\s*3\+/i.test(stage)) return 3;
  return null;
}

function buildTrajectoryAssessment(
  snapshots: LadderSnapshotLite[],
  currentTemporal: LadderTemporalVerdictStored,
): { verdict: LadderTrajectoryAssessmentVerdict; notes: string[] } {
  const notes: string[] = [];
  if (currentTemporal === "not_comparable_yet") {
    notes.push("Temporal atual: not_comparable_yet.");
    return { verdict: "blocked_by_temporal_inconsistency", notes };
  }

  const window = snapshots.slice(-6);
  const last4Temporal = snapshots.slice(-4).map(s => s.temporalConsistencyVerdict);
  const notComparableInLast4 = last4Temporal.filter(t => t === "not_comparable_yet").length;

  if (last4Temporal.length >= 3 && notComparableInLast4 >= 3) {
    notes.push("Maioria recente de snapshots em not_comparable_yet.");
    return { verdict: "blocked_by_temporal_inconsistency", notes };
  }

  const degrees = window.map(s => parseStageDegreeFromSummary(s.currentStage)).filter((x): x is number => x !== null);
  if (degrees.length < 3) {
    notes.push("Histórico insuficiente para tendência de degrau (menos de 3 snapshots com degrau legível).");
    return { verdict: "stagnant", notes };
  }

  const last4 = degrees.slice(-4);
  let hasInc = false;
  let hasDec = false;
  for (let i = 1; i < last4.length; i++) {
    if (last4[i]! > last4[i - 1]!) hasInc = true;
    if (last4[i]! < last4[i - 1]!) hasDec = true;
  }

  if (hasInc && hasDec) {
    notes.push("Degraus recentes sobem e descem — possível oscilação na trilha agregada.");
    return { verdict: "oscillating", notes };
  }
  if (hasInc && !hasDec) {
    notes.push("Degraus recentes não decrescem e há pelo menos um aumento.");
    return { verdict: "improving", notes };
  }

  const allEq = last4.every(d => d === last4[0]);
  if (allEq && last4.length >= 3) {
    notes.push("Mesmo degrau reportado em sequência recente.");
    return { verdict: "stagnant", notes };
  }

  notes.push("Sem padrão claro de subida ou oscilação forte; tratar como estável.");
  return { verdict: "stagnant", notes };
}

export interface LadderHistoryDigest {
  persistenceEnabled: boolean;
  persistencePath: string | null;
  maxSnapshots: number;
  snapshotCount: number;
  recentSnapshots: LadderSnapshotLite[];
  stageTrend: string;
  promotionTrend: string[];
  temporalConsistencyTrend: string[];
  repeatedBlockedPattern: boolean;
  repeatedNotComparablePattern: boolean;
}

export interface LadderTrajectoryAssessmentDigest {
  verdict: LadderTrajectoryAssessmentVerdict;
  notes: string[];
}

export function buildLadderSnapshotLite(args: {
  promotionLadder: {
    recurrentPocketExists: boolean;
    economicsPromotionVerdict: string;
    executionObservationVerdict: string;
    executionPromotionVerdict: string;
    minimalPaperExecutionAssessmentVerdict: string;
  };
  temporalConsistencyVerdict: LadderTemporalVerdictStored;
  currentStage: string;
}): LadderSnapshotLite {
  return {
    at: new Date().toISOString(),
    recurrentPocketExists: args.promotionLadder.recurrentPocketExists,
    economicsPromotionVerdict: args.promotionLadder.economicsPromotionVerdict,
    executionObservationVerdict: args.promotionLadder.executionObservationVerdict,
    executionPromotionVerdict: args.promotionLadder.executionPromotionVerdict,
    minimalPaperExecutionAssessmentVerdict: args.promotionLadder.minimalPaperExecutionAssessmentVerdict,
    temporalConsistencyVerdict: args.temporalConsistencyVerdict,
    currentStage: args.currentStage,
  };
}

export function processLadderHistoryForDigest(args: {
  promotionLadder: {
    recurrentPocketExists: boolean;
    economicsPromotionVerdict: string;
    executionObservationVerdict: string;
    executionPromotionVerdict: string;
    minimalPaperExecutionAssessmentVerdict: string;
  };
  temporalConsistencyVerdict: LadderTemporalVerdictStored;
  currentStage: string;
  cwd?: string;
}): { ladderHistory: LadderHistoryDigest; ladderTrajectoryAssessment: LadderTrajectoryAssessmentDigest } {
  const prior = loadLadderHistorySnapshots(args.cwd);
  const snap = buildLadderSnapshotLite({
    promotionLadder: args.promotionLadder,
    temporalConsistencyVerdict: args.temporalConsistencyVerdict,
    currentStage: args.currentStage,
  });
  const merged = appendLadderSnapshotIfChanged(prior, snap);
  if (merged !== prior) {
    persistLadderHistorySnapshots(merged, args.cwd);
  }

  const dis = isDiskDisabled();
  const persistencePath = dis ? null : defaultLadderHistoryPath(args.cwd);
  const recent = merged.slice(-20);

  const stageDegrees = recent.map(s => parseStageDegreeFromSummary(s.currentStage));
  const stageTrend =
    stageDegrees.every(d => d !== null) && stageDegrees.length > 0
      ? (stageDegrees as number[]).join("→")
      : recent
          .slice(-5)
          .map(s => {
            const d = parseStageDegreeFromSummary(s.currentStage);
            return d !== null ? String(d) : "?";
          })
          .join("→");

  const promotionTrend = recent.slice(-10).map(
    s =>
      `econ=${s.economicsPromotionVerdict};execObs=${s.executionObservationVerdict};execPromo=${s.executionPromotionVerdict};min=${s.minimalPaperExecutionAssessmentVerdict}`,
  );

  const temporalConsistencyTrend = recent.slice(-10).map(s => s.temporalConsistencyVerdict);

  const tail5 = merged.slice(-5);
  const repeatedBlockedPattern =
    tail5.length >= 4 && tail5.filter(s => s.executionObservationVerdict === "blocked").length >= 4;
  const repeatedNotComparablePattern =
    tail5.length >= 4 && tail5.filter(s => s.temporalConsistencyVerdict === "not_comparable_yet").length >= 4;

  const ladderHistory: LadderHistoryDigest = {
    persistenceEnabled: !dis,
    persistencePath,
    maxSnapshots: MAX_SNAPSHOTS(),
    snapshotCount: merged.length,
    recentSnapshots: recent,
    stageTrend,
    promotionTrend,
    temporalConsistencyTrend,
    repeatedBlockedPattern,
    repeatedNotComparablePattern,
  };

  const ladderTrajectoryAssessment = buildTrajectoryAssessment(merged, args.temporalConsistencyVerdict);

  return { ladderHistory, ladderTrajectoryAssessment };
}
