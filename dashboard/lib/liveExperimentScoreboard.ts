/**
 * Scoreboard só de leitura v3: agrega artefactos compactos *-summary.json por worker
 * (sem ler JSONL cru cruzado — adequado quando cada worker tem volume próprio).
 */

import path from "path";
import type { CollectionHealthVerdict } from "./liveExperimentMeta";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES, resolvePaperStateDir } from "./paperStateDir";
import {
  ENOUGH_HISTORY_FOR_RANKING,
  type LiveExperimentId,
  type PromotabilityVerdict,
} from "./liveExperimentStatsShared";
import {
  readLiveExperimentWorkerSummaryFile,
  type LiveExperimentWorkerSummaryFile,
} from "./liveExperimentWorkerSummary";
import {
  ACTIVE_SINGLE_TRACK_EXPERIMENT_ID,
  getStrategyTrackClassification,
  isPromotionActiveTrackClassification,
  STRATEGY_TRACK_MODE_SINGLE,
  type StrategyTrackClassification,
} from "./strategyTrackPolicy";

export type { LiveExperimentId, PromotabilityVerdict };
export type { StrategyTrackClassification };
export { ENOUGH_HISTORY_FOR_RANKING };
export type { CollectionHealthVerdict };

export type LiveExperimentScoreboardVerdict =
  | "no_live_experiment_ready"
  | "one_borderline_candidate"
  | "multiple_borderline_candidates"
  | "one_promotable_leader"
  | "multiple_promotable_leaders";

export interface LiveExperimentScoreboardRow {
  experimentId: LiveExperimentId;
  experimentType: string;
  summaryPathResolved: string;
  summaryFileExists: boolean;
  historyPathResolved: string;
  historyFileExists: boolean;
  operationalMetaPathResolved: string;
  operationalMetaFileExists: boolean;
  primaryNetMetricKey: string;
  historyJsonlLineCount: number;
  totalObservations: number;
  metadataTotalObservations: number | null;
  serviceName: string | null;
  runId: string | null;
  restartCount: number | null;
  intervalMsOperational: number | null;
  runnerStartedAt: string | null;
  firstObservationAt: string | null;
  lastObservationAt: string | null;
  runtimeHours: number | null;
  expectedObservationsByNow: number | null;
  observationsMissingEstimate: number | null;
  collectionHealthVerdict: CollectionHealthVerdict;
  operationalIsHealthy: boolean | null;
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
  enoughObservationsForRanking: boolean;
  strategyTrackMode: typeof STRATEGY_TRACK_MODE_SINGLE;
  strategyTrackClassification: StrategyTrackClassification;
  supportingNote: string;
}

export interface ScoreboardRankingEntry {
  rank: number;
  experimentId: LiveExperimentId;
  medianNet: number;
  pctPositive: number;
  includedInRanking: boolean;
}

export interface LiveExperimentScoreboardDigest {
  probeVersion: "live-experiment-scoreboard-v4";
  readDisclaimer: string;
  strategyTrackMode: typeof STRATEGY_TRACK_MODE_SINGLE;
  activeStrategyTrackExperimentId: typeof ACTIVE_SINGLE_TRACK_EXPERIMENT_ID;
  paperStateDirResolved: string;
  experimentsObserved: number;
  experimentsWithEnoughHistory: number;
  bestExperimentByMedian: LiveExperimentId | null;
  bestExperimentByPctPositive: LiveExperimentId | null;
  scoreboardRanking: ScoreboardRankingEntry[];
  liveExperimentScoreboardVerdict: LiveExperimentScoreboardVerdict;
  liveExperimentScoreboardSummaryLine: string;
  stabilityNote: string | null;
  anyPromotableExperiment: boolean;
  experiments: LiveExperimentScoreboardRow[];
  computedAt: string;
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

function mapSummaryToRow(s: LiveExperimentWorkerSummaryFile): LiveExperimentScoreboardRow {
  return {
    experimentId: s.experimentId,
    experimentType: s.experimentType,
    summaryPathResolved: s.summaryPath,
    summaryFileExists: true,
    historyPathResolved: s.historyPath,
    historyFileExists: s.historyFileExists,
    operationalMetaPathResolved: s.metaPath,
    operationalMetaFileExists: s.operationalMetaFileExists,
    primaryNetMetricKey: s.primaryNetMetricKey,
    historyJsonlLineCount: s.historyJsonlLineCount,
    totalObservations: s.totalObservations,
    metadataTotalObservations: s.metadataTotalObservations,
    serviceName: s.serviceName,
    runId: s.runId,
    restartCount: s.restartCount,
    intervalMsOperational: s.intervalMsOperational,
    runnerStartedAt: s.runnerStartedAt,
    firstObservationAt: s.firstObservationAt,
    lastObservationAt: s.lastObservationAt,
    runtimeHours: s.runtimeHours,
    expectedObservationsByNow: s.expectedObservationsByNow,
    observationsMissingEstimate: s.observationsMissingEstimate,
    collectionHealthVerdict: s.collectionHealthVerdict,
    operationalIsHealthy: s.operationalIsHealthy,
    meanNet: s.meanNet,
    medianNet: s.medianNet,
    minNet: s.minNet,
    maxNet: s.maxNet,
    pctPositive: s.pctPositive,
    pctNearZero: s.pctNearZero,
    strongestWindow: s.strongestWindow,
    weakestWindow: s.weakestWindow,
    currentVerdict: s.currentVerdict,
    promotabilityVerdict: s.promotabilityVerdict,
    enoughObservationsForRanking: s.enoughObservationsForRanking,
    strategyTrackMode: s.strategyTrackMode ?? STRATEGY_TRACK_MODE_SINGLE,
    strategyTrackClassification:
      s.strategyTrackClassification ?? getStrategyTrackClassification(s.experimentId),
    supportingNote: [
      `source=worker_summary_v1`,
      `summary=${s.summaryPath}`,
      `ranking_threshold_n>=${ENOUGH_HISTORY_FOR_RANKING}`,
      `track=${s.strategyTrackClassification ?? getStrategyTrackClassification(s.experimentId)}`,
    ].join("|"),
  };
}

function emptyRow(
  id: LiveExperimentId,
  type: string,
  summaryFp: string,
  metricKey: string,
): LiveExperimentScoreboardRow {
  return {
    experimentId: id,
    experimentType: type,
    summaryPathResolved: summaryFp,
    summaryFileExists: false,
    historyPathResolved: "",
    historyFileExists: false,
    operationalMetaPathResolved: "",
    operationalMetaFileExists: false,
    primaryNetMetricKey: metricKey,
    historyJsonlLineCount: 0,
    totalObservations: 0,
    metadataTotalObservations: null,
    serviceName: null,
    runId: null,
    restartCount: null,
    intervalMsOperational: null,
    runnerStartedAt: null,
    firstObservationAt: null,
    lastObservationAt: null,
    runtimeHours: null,
    expectedObservationsByNow: null,
    observationsMissingEstimate: null,
    collectionHealthVerdict: "unknown",
    operationalIsHealthy: null,
    meanNet: NaN,
    medianNet: NaN,
    minNet: NaN,
    maxNet: NaN,
    pctPositive: 0,
    pctNearZero: 0,
    strongestWindow: null,
    weakestWindow: null,
    currentVerdict: null,
    promotabilityVerdict: "insufficient_history",
    enoughObservationsForRanking: false,
    strategyTrackMode: STRATEGY_TRACK_MODE_SINGLE,
    strategyTrackClassification: getStrategyTrackClassification(id),
    supportingNote: `summary_missing path=${summaryFp}`,
  };
}

function globalScoreboardVerdict(rows: LiveExperimentScoreboardRow[]): LiveExperimentScoreboardVerdict {
  const ranked = rows.filter(
    r =>
      r.enoughObservationsForRanking &&
      isPromotionActiveTrackClassification(r.strategyTrackClassification),
  );
  if (ranked.length === 0) {
    return "no_live_experiment_ready";
  }
  const promotable = ranked.filter(r => r.promotabilityVerdict === "promotable_candidate");
  const borderline = ranked.filter(r => r.promotabilityVerdict === "borderline_keep_running");

  if (promotable.length >= 2) return "multiple_promotable_leaders";
  if (promotable.length === 1) return "one_promotable_leader";
  if (borderline.length >= 2) return "multiple_borderline_candidates";
  if (borderline.length === 1) return "one_borderline_candidate";
  return "no_live_experiment_ready";
}

export function buildLiveExperimentScoreboardDigest(
  cwd: string = process.cwd(),
): LiveExperimentScoreboardDigest {
  const paperDir = resolvePaperStateDir(cwd);

  const specs: Array<{
    id: LiveExperimentId;
    type: string;
    summaryFilename: keyof typeof PAPER_TRAIL_FILENAMES;
    summaryEnv: string | undefined;
    metricKey: string;
  }> = [
    {
      id: "final_neg_risk_31552",
      type: "final_negative_risk_validation_31552_jsonl",
      summaryFilename: "finalNegativeRisk31552Summary",
      summaryEnv: process.env.FINAL_NEG_RISK_31552_SUMMARY_PATH,
      metricKey: "stressAdjustedNetConversionEdge",
    },
    {
      id: "reach_april_btc_reaction_monitor",
      type: "reach_april_btc_reaction_monitor_jsonl",
      summaryFilename: "reachAprilBtcReactionMonitorSummary",
      summaryEnv: process.env.REACH_APRIL_BTC_MONITOR_SUMMARY_PATH,
      metricKey: "medianNetPerReactionCycle",
    },
    {
      id: "cross_venue_anchor_1823789",
      type: "cross_venue_anchor_refined_1823789_jsonl",
      summaryFilename: "crossVenueAnchor1823789MonitorSummary",
      summaryEnv: process.env.CROSS_VENUE_ANCHOR_1823789_MONITOR_SUMMARY_PATH,
      metricKey: "estimatedNetAnchorCycle",
    },
  ];

  const experiments: LiveExperimentScoreboardRow[] = [];

  for (const s of specs) {
    const summaryFp = resolveSummaryPath(s.summaryEnv, PAPER_TRAIL_FILENAMES[s.summaryFilename], cwd);
    const snap = readLiveExperimentWorkerSummaryFile(summaryFp);
    if (snap) {
      experiments.push(mapSummaryToRow(snap));
    } else {
      experiments.push(emptyRow(s.id, s.type, summaryFp, s.metricKey));
    }
  }

  const experimentsObserved = experiments.filter(e => e.summaryFileExists).length;
  const experimentsWithEnoughHistory = experiments.filter(e => e.enoughObservationsForRanking).length;

  const eligibleActive = experiments.filter(
    e =>
      e.enoughObservationsForRanking &&
      isPromotionActiveTrackClassification(e.strategyTrackClassification),
  );
  const sortedByMedian = [...eligibleActive].sort((a, b) => b.medianNet - a.medianNet);
  const sortedByPct = [...eligibleActive].sort((a, b) => b.pctPositive - a.pctPositive);

  const bestExperimentByMedian = sortedByMedian.length ? sortedByMedian[0].experimentId : null;
  const bestExperimentByPctPositive = sortedByPct.length ? sortedByPct[0].experimentId : null;

  const activePool = experiments.filter(e =>
    isPromotionActiveTrackClassification(e.strategyTrackClassification),
  );
  const archivedPool = experiments.filter(
    e => e.strategyTrackClassification === "archived_consistently_negative",
  );
  const sortedActive = [...activePool].sort((a, b) => {
    if (a.enoughObservationsForRanking !== b.enoughObservationsForRanking) {
      return a.enoughObservationsForRanking ? -1 : 1;
    }
    if (b.medianNet !== a.medianNet) return b.medianNet - a.medianNet;
    return b.pctPositive - a.pctPositive;
  });
  const sortedForRank = [...sortedActive, ...archivedPool];
  let nextRank = 1;
  const scoreboardRanking: ScoreboardRankingEntry[] = sortedForRank.map(e => {
    const included =
      e.enoughObservationsForRanking &&
      isPromotionActiveTrackClassification(e.strategyTrackClassification);
    return {
      rank: included ? nextRank++ : 0,
      experimentId: e.experimentId,
      medianNet: e.medianNet,
      pctPositive: e.pctPositive,
      includedInRanking: included,
    };
  });

  const thin = experiments.filter(e => e.totalObservations > 0 && !e.enoughObservationsForRanking);
  const stabilityNote =
    thin.length > 0
      ? `Some experiments have < ${ENOUGH_HISTORY_FOR_RANKING} observations (${thin.map(t => t.experimentId).join(", ")}); median/pct comparisons are shaded until history accumulates.`
      : experiments.some(e => !e.summaryFileExists)
        ? "One or more worker summary files are missing; mount or copy *-summary.json into PAPER_STATE_DIR or set *_SUMMARY_PATH overrides."
        : null;

  const verdict = globalScoreboardVerdict(experiments);
  const anyPromotableExperiment = experiments.some(
    e =>
      isPromotionActiveTrackClassification(e.strategyTrackClassification) &&
      e.promotabilityVerdict === "promotable_candidate" &&
      e.enoughObservationsForRanking,
  );

  const liveExperimentScoreboardSummaryLine = `live_experiment_scoreboard_v4 single_track=${ACTIVE_SINGLE_TRACK_EXPERIMENT_ID}: verdict=${verdict} | summaries=${experimentsObserved}/3 | enough_hist=${experimentsWithEnoughHistory} | best_med=${bestExperimentByMedian ?? "none"} | best_pct_pos=${bestExperimentByPctPositive ?? "none"} | promotable_any=${anyPromotableExperiment}`;

  return {
    probeVersion: "live-experiment-scoreboard-v4",
    readDisclaimer:
      "Read-only judge v4 (single-track): active promotion track is cross_venue_anchor_1823789 only; final_neg_risk_31552 and reach_april_btc_reaction_monitor are classified archived_consistently_negative in outputs — do not reopen for strategy promotion. Consumes *-summary.json; path overrides unchanged. Ranking and promotion verdicts consider the active track only.",
    strategyTrackMode: STRATEGY_TRACK_MODE_SINGLE,
    activeStrategyTrackExperimentId: ACTIVE_SINGLE_TRACK_EXPERIMENT_ID,
    paperStateDirResolved: paperDir,
    experimentsObserved,
    experimentsWithEnoughHistory,
    bestExperimentByMedian,
    bestExperimentByPctPositive,
    scoreboardRanking,
    liveExperimentScoreboardVerdict: verdict,
    liveExperimentScoreboardSummaryLine,
    stabilityNote,
    anyPromotableExperiment,
    experiments,
    computedAt: new Date().toISOString(),
  };
}
