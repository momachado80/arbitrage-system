/**
 * Experiment Decision Engine — recommends next experiments without repeating failed ideas.
 * Does NOT auto-activate. Distinguishes failed ideas, failed refinements, viable lines, active lines.
 */

import type { ExperimentLedgerEntry } from "./experimentLedgerPersistence";
import { getExperimentLedgerEntries } from "./experimentLedger";
import { EXPERIMENT_FAMILIES } from "./experimentRegistry";

export interface ExperimentRecommendation {
  challengerProfileId: string;
  familyId: string;
  baseProfileId: string;
  recommendationReason: string;
  blocked: boolean;
  blockedReason?: string;
}

/** Failed/failed refinement statuses that should not be retried as exact repeats */
const BLOCKED_STATUSES = new Set(["failed", "superseded", "archived"]);

/** Challengers that are known failed at top-level; refinements on winning branch may still be viable */
const TOP_LEVEL_FAILED_CHALLENGERS = new Set([
  "shadow_1000_adapt_edgegate_v1",
  "shadow_1000_adapt_pairpenalty_v1",
]);

/** Winning branch candidate; refinements on top are distinct experiments */
const WINNING_BRANCH_CANDIDATE = "shadow_1000_adapt_captrade_entryfloor_v1";

/**
 * Get recommended next experiments. Does not recommend exact repeats of failed experiments.
 */
export function getExperimentRecommendations(
  ledgerEntries: ExperimentLedgerEntry[],
  enabledChallengerIds: string[],
  adaptiveChallengerIds: string[]
): ExperimentRecommendation[] {
  const recommendations: ExperimentRecommendation[] = [];
  const byChallenger = new Map<string, ExperimentLedgerEntry>();
  for (const e of ledgerEntries) {
    if (!byChallenger.has(e.challengerProfileId) || (e.endTimestamp && !byChallenger.get(e.challengerProfileId)!.endTimestamp)) {
      byChallenger.set(e.challengerProfileId, e);
    }
  }

  for (const family of EXPERIMENT_FAMILIES) {
    if (!family.eligibleForRecommendation || !family.challengerPattern) continue;

    const challengerId = family.challengerPattern;
    const entry = byChallenger.get(challengerId);
    const isActive = enabledChallengerIds.includes(challengerId);
    const isAvailable = adaptiveChallengerIds.includes(challengerId);

    let blocked = false;
    let blockedReason: string | undefined;
    let recommendationReason = "Available for controlled activation.";

    if (isActive) {
      blocked = true;
      blockedReason = "Already active; avoid duplicate activation.";
    } else if (entry) {
      if (BLOCKED_STATUSES.has(entry.status) || entry.decision === "failed" || entry.decision === "superseded") {
        if (challengerId === "shadow_1000_adapt_captrade_entryfloor_pairpenalty_v1") {
          blocked = false;
          blockedReason = undefined;
          recommendationReason = "Refinement on winning branch; distinct from top-level pairpenalty failure.";
        } else if (TOP_LEVEL_FAILED_CHALLENGERS.has(challengerId)) {
          blocked = true;
          blockedReason = "Top-level experiment previously failed; do not repeat as exact copy.";
        } else {
          blocked = true;
          blockedReason = `Experiment marked ${entry.status}.`;
        }
      }
    }

    if (family.familyId === "entryfloor_pairpenalty") {
      const winnerEntry = byChallenger.get(WINNING_BRANCH_CANDIDATE);
      if (winnerEntry && (winnerEntry.decision === "failed" || winnerEntry.status === "failed")) {
        blocked = true;
        blockedReason = "Base (entryfloor) weakened; refinement not recommended until base improves.";
      } else {
        recommendationReason = "Layered on winning branch; single variable (pair penalty).";
      }
    }

    recommendations.push({
      challengerProfileId: challengerId,
      familyId: family.familyId,
      baseProfileId: family.allowedBaseProfiles[0] ?? "shadow_1000",
      recommendationReason,
      blocked,
      blockedReason,
    });
  }

  return recommendations;
}

/**
 * Get experiments that are blocked or redundant based on ledger.
 */
export function getBlockedExperiments(ledgerEntries: ExperimentLedgerEntry[]): string[] {
  const blocked: string[] = [];
  for (const e of ledgerEntries) {
    if (BLOCKED_STATUSES.has(e.status) && e.decision === "failed") {
      blocked.push(e.challengerProfileId);
    }
  }
  return Array.from(new Set(blocked));
}
