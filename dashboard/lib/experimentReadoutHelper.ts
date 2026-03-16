/**
 * Experiment Readout Helper — structured comparison of baseline, winner, candidate.
 * Computes standardized metrics for experiment evaluation.
 */

import type { ProfileComparisonEntry } from "./shadowClosedTradeAudit";

export interface ExperimentReadoutComparison {
  profileId: string;
  closedTrades: number;
  avgRealizedPnL: number;
  medianRealizedPnL: number;
  avgHoldingTimeMs: number;
  avgFilledCapital: number;
  avgFillRatio: number;
  winRate: number;
}

export interface ReadoutTriplet {
  baseline: ExperimentReadoutComparison | null;
  currentWinner: ExperimentReadoutComparison | null;
  candidate: ExperimentReadoutComparison | null;
}

export interface StandardizedReadout {
  baseline: ExperimentReadoutComparison | null;
  currentWinner: ExperimentReadoutComparison | null;
  candidate: ExperimentReadoutComparison | null;
  directionOfImprovement: "candidate_better" | "winner_better" | "inconclusive" | "insufficient_data";
  flowImpactAssessment: string;
  readinessAssessment: string;
}

function toReadout(entry: ProfileComparisonEntry | undefined): ExperimentReadoutComparison | null {
  if (!entry) return null;
  return {
    profileId: entry.profileId,
    closedTrades: entry.totalClosed,
    avgRealizedPnL: entry.avgRealizedPnL,
    medianRealizedPnL: entry.medianRealizedPnL,
    avgHoldingTimeMs: entry.avgHoldingTimeMs,
    avgFilledCapital: entry.avgFilledCapital,
    avgFillRatio: entry.avgFillRatio,
    winRate: entry.winRate,
  };
}

/**
 * Build standardized readout from profile comparison.
 */
export function computeExperimentReadout(
  profileComparison: ProfileComparisonEntry[],
  baselineProfileId: string,
  winnerProfileId: string | null,
  candidateProfileId: string | null
): StandardizedReadout {
  const baseline = toReadout(profileComparison.find((p) => p.profileId === baselineProfileId));
  const currentWinner = winnerProfileId
    ? toReadout(profileComparison.find((p) => p.profileId === winnerProfileId))
    : null;
  const candidate = candidateProfileId
    ? toReadout(profileComparison.find((p) => p.profileId === candidateProfileId))
    : null;

  let directionOfImprovement: StandardizedReadout["directionOfImprovement"] = "insufficient_data";
  let flowImpactAssessment = "No sufficient data for assessment.";
  let readinessAssessment = "Insufficient data.";

  const ref = currentWinner ?? baseline;
  if (candidate && ref && candidate.closedTrades >= 10 && ref.closedTrades >= 10) {
    if (candidate.avgRealizedPnL > ref.avgRealizedPnL + 0.001) {
      directionOfImprovement = "candidate_better";
    } else if (ref.avgRealizedPnL > candidate.avgRealizedPnL + 0.001) {
      directionOfImprovement = "winner_better";
    } else {
      directionOfImprovement = "inconclusive";
    }
    flowImpactAssessment = `Candidate avgPnL ${candidate.avgRealizedPnL.toFixed(4)} vs reference ${ref.avgRealizedPnL.toFixed(4)}.`;
    readinessAssessment =
      candidate.closedTrades >= 30
        ? "Sufficient sample for preliminary assessment."
        : "Sample growing; more trades needed for confidence.";
  }

  return {
    baseline,
    currentWinner,
    candidate,
    directionOfImprovement,
    flowImpactAssessment,
    readinessAssessment,
  };
}
