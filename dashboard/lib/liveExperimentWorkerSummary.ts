/**
 * Ficheiro JSON compacto por worker (substitui leitura de JSONL cruzado no dashboard).
 */

import fs from "fs";
import path from "path";
import {
  deriveCollectionHealthVerdict,
  readOperationalMetaFile,
  resolveOperationalMetaPath,
  scanJsonlIsoSummary,
  type CollectionHealthVerdict,
} from "./liveExperimentMeta";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES } from "./paperStateDir";
import {
  derivePromotabilityVerdict,
  ENOUGH_HISTORY_FOR_RANKING,
  NEAR_ZERO_ABS_BAND,
  readJsonlLines,
  summarizeObservationFrames,
  type ExperimentObservationFrame,
  type LiveExperimentId,
  type PromotabilityVerdict,
} from "./liveExperimentStatsShared";
import {
  getStrategyTrackClassification,
  STRATEGY_TRACK_MODE_SINGLE,
  type StrategyTrackClassification,
} from "./strategyTrackPolicy";

export const WORKER_SUMMARY_PROBE_VERSION = "live-experiment-worker-summary-v1" as const;

export interface LiveExperimentWorkerSummaryFile {
  probeVersion: typeof WORKER_SUMMARY_PROBE_VERSION;
  experimentId: LiveExperimentId;
  experimentType: string;
  primaryNetMetricKey: string;
  summaryWrittenAt: string;
  historyPath: string;
  metaPath: string;
  summaryPath: string;
  historyFileExists: boolean;
  operationalMetaFileExists: boolean;
  runnerStartedAt: string | null;
  firstObservationAt: string | null;
  lastObservationAt: string | null;
  runtimeHours: number | null;
  totalObservations: number;
  historyJsonlLineCount: number;
  meanNet: number;
  medianNet: number;
  minNet: number;
  maxNet: number;
  pctPositive: number;
  pctNearZero: number;
  strongestWindow: string | null;
  weakestWindow: string | null;
  currentVerdict: string | null;
  promotabilityVerdict: PromotabilityVerdict;
  collectionHealthVerdict: CollectionHealthVerdict;
  serviceName: string | null;
  runId: string | null;
  restartCount: number | null;
  intervalMsOperational: number | null;
  expectedObservationsByNow: number | null;
  observationsMissingEstimate: number | null;
  operationalIsHealthy: boolean | null;
  metadataTotalObservations: number | null;
  enoughObservationsForRanking: boolean;
  strategyTrackMode: typeof STRATEGY_TRACK_MODE_SINGLE;
  strategyTrackClassification: StrategyTrackClassification;
}

function resolveSummaryPath(
  envOverride: string | undefined,
  filename: string,
  cwd: string,
): string {
  const raw = envOverride?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(filename, cwd);
}

function writeSummaryAtomic(fp: string, obj: LiveExperimentWorkerSummaryFile): void {
  const dir = path.dirname(fp);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = `${fp}.tmp.${process.pid}`;
  fs.writeFileSync(tmp, `${JSON.stringify(obj, null, 2)}\n`, "utf8");
  fs.renameSync(tmp, fp);
}

export type SummaryParseFn = (lines: string[]) => {
  frames: ExperimentObservationFrame[];
  lastVerdict: string | null;
};

export function writeLiveExperimentWorkerSummary(params: {
  cwd?: string;
  experimentId: LiveExperimentId;
  experimentType: string;
  primaryNetMetricKey: string;
  summaryFilenameKey: keyof typeof PAPER_TRAIL_FILENAMES;
  summaryEnvOverride?: string;
  historyFilenameKey: keyof typeof PAPER_TRAIL_FILENAMES;
  historyEnvOverride?: string;
  metaFilenameKey: keyof typeof PAPER_TRAIL_FILENAMES;
  metaEnvOverride?: string;
  parse: SummaryParseFn;
  intervalMs: number;
}): void {
  const cwd = params.cwd ?? process.cwd();
  const historyPath = params.historyEnvOverride?.trim()
    ? path.resolve(params.historyEnvOverride.trim())
    : defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES[params.historyFilenameKey], cwd);
  const metaPath = resolveOperationalMetaPath(
    params.metaEnvOverride,
    PAPER_TRAIL_FILENAMES[params.metaFilenameKey],
    cwd,
  );
  const summaryPath = resolveSummaryPath(
    params.summaryEnvOverride,
    PAPER_TRAIL_FILENAMES[params.summaryFilenameKey],
    cwd,
  );

  const historyFileExists = fs.existsSync(historyPath);
  const lines = readJsonlLines(historyPath);
  const { frames, lastVerdict } = params.parse(lines);
  const n = frames.length;
  const stats = summarizeObservationFrames(frames, NEAR_ZERO_ABS_BAND);
  const promotability = derivePromotabilityVerdict(n, stats.meanNet, stats.medianNet, stats.pctPositive);
  const enough = n >= ENOUGH_HISTORY_FOR_RANKING;

  const metaSnap = readOperationalMetaFile(metaPath);
  const metaExists = metaSnap !== null;
  const scanIso = scanJsonlIsoSummary(historyPath);

  const intervalForHealth = metaSnap?.intervalMs ?? params.intervalMs;
  const missEst =
    metaSnap?.observationsMissingEstimate ??
    (metaSnap?.expectedObservationsByNow != null
      ? Math.max(0, metaSnap.expectedObservationsByNow - scanIso.totalLines)
      : null);

  const collectionHealthVerdict = deriveCollectionHealthVerdict({
    historyFileExists,
    metaFileExists: metaExists,
    totalLines: scanIso.totalLines,
    lastObservationAt: metaSnap?.lastObservationAt ?? scanIso.lastObservationAt,
    intervalMs: metaSnap?.intervalMs ?? intervalForHealth,
    observationsMissingEstimate: missEst,
  });

  const out: LiveExperimentWorkerSummaryFile = {
    probeVersion: WORKER_SUMMARY_PROBE_VERSION,
    experimentId: params.experimentId,
    experimentType: params.experimentType,
    primaryNetMetricKey: params.primaryNetMetricKey,
    summaryWrittenAt: new Date().toISOString(),
    historyPath,
    metaPath,
    summaryPath,
    historyFileExists,
    operationalMetaFileExists: metaExists,
    runnerStartedAt: metaSnap?.runnerStartedAt ?? null,
    firstObservationAt: metaSnap?.firstObservationAt ?? scanIso.firstObservationAt,
    lastObservationAt: metaSnap?.lastObservationAt ?? scanIso.lastObservationAt,
    runtimeHours: metaSnap?.runtimeHours ?? null,
    totalObservations: n,
    historyJsonlLineCount: scanIso.totalLines,
    meanNet: stats.meanNet,
    medianNet: stats.medianNet,
    minNet: stats.minNet,
    maxNet: stats.maxNet,
    pctPositive: stats.pctPositive,
    pctNearZero: stats.pctNearZero,
    strongestWindow: stats.strongestWindow,
    weakestWindow: stats.weakestWindow,
    currentVerdict: lastVerdict,
    promotabilityVerdict: promotability,
    collectionHealthVerdict,
    serviceName: metaSnap?.serviceName ?? null,
    runId: metaSnap?.runId ?? null,
    restartCount: metaSnap?.restartCount ?? null,
    intervalMsOperational: metaSnap?.intervalMs ?? null,
    expectedObservationsByNow: metaSnap?.expectedObservationsByNow ?? null,
    observationsMissingEstimate: metaSnap?.observationsMissingEstimate ?? null,
    operationalIsHealthy: metaSnap?.isHealthy ?? null,
    metadataTotalObservations: metaSnap?.totalObservations ?? null,
    enoughObservationsForRanking: enough,
    strategyTrackMode: STRATEGY_TRACK_MODE_SINGLE,
    strategyTrackClassification: getStrategyTrackClassification(params.experimentId),
  };

  writeSummaryAtomic(summaryPath, out);
}

export function readLiveExperimentWorkerSummaryFile(
  fp: string,
): LiveExperimentWorkerSummaryFile | null {
  try {
    if (!fs.existsSync(fp)) return null;
    const t = fs.readFileSync(fp, "utf8");
    const j = JSON.parse(t) as LiveExperimentWorkerSummaryFile;
    if (j && j.probeVersion === WORKER_SUMMARY_PROBE_VERSION) {
      if (j.experimentId) {
        if (!j.strategyTrackClassification) {
          j.strategyTrackClassification = getStrategyTrackClassification(j.experimentId);
        }
        if (!j.strategyTrackMode) {
          j.strategyTrackMode = STRATEGY_TRACK_MODE_SINGLE;
        }
      }
      return j;
    }
    return null;
  } catch {
    return null;
  }
}
