/**
 * Experiment Decision Engine — evidence-aware recommendations with branch dominance logic.
 * Does NOT auto-activate. Distinguishes winner, weakened refinement, failed top-level, available-but-not-recommended.
 */

import type { ExperimentLedgerEntry } from "./experimentLedgerPersistence";
import { getExperimentLedgerEntries } from "./experimentLedger";
import { EXPERIMENT_FAMILIES } from "./experimentRegistry";

export type RecommendationClassification =
  | "recommended"
  | "available_but_not_recommended"
  | "blocked"
  | "blocked_by_recent_evidence"
  | "superseded"
  | "already_active"
  | "refinement_base_weakened";

export interface ExperimentRecommendation {
  challengerProfileId: string;
  familyId: string;
  baseProfileId: string;
  recommendationReason: string;
  blocked: boolean;
  blockedReason?: string;
  /** Derived classification for operational use */
  recommendationClassification: RecommendationClassification;
  /** When not recommended, why (e.g. weakened refinement) */
  whyNotRecommended?: string;
  /** 0-1 confidence in this recommendation */
  recommendationConfidence?: number;
}

export interface DecisionEngineResult {
  recommendations: ExperimentRecommendation[];
  recommendedNextExperiments: ExperimentRecommendation[];
  availableButNotRecommendedExperiments: ExperimentRecommendation[];
  blockedByRecentEvidence: string[];
  supersededExperiments: string[];
  currentWinnerBranch: string;
  winnerBranchReason: string;
  recommendationRationale: string;
  residualBottleneckHypothesis: string;
  recommendationConfidence: number;
  branchComparisons: Array<{
    base: string;
    refinement?: string;
    outcome: string;
    evidenceNote: string;
  }>;
}

const BLOCKED_STATUSES = new Set(["failed", "superseded", "archived"]);
const WEAKENED_STATUSES = new Set(["weakened", "failed", "inconclusive"]);

/** Current strongest known line from production evidence */
const CURRENT_WINNER_BRANCH = "shadow_1000_adapt_captrade_v1";

/** Refinement that weakened its base (entryfloor on captrade) */
const WEAKENED_REFINEMENT = "shadow_1000_adapt_captrade_entryfloor_v1";

/** Top-level experiments that failed; layered versions may be distinct */
const TOP_LEVEL_FAILED_CHALLENGERS = new Set([
  "shadow_1000_adapt_edgegate_v1",
  "shadow_1000_adapt_pairpenalty_v1",
]);

/** entryfloor_pairpenalty refines entryfloor; if entryfloor weakened, this is not a top recommendation */
const ENTRYFLOOR_PAIRPENALTY = "shadow_1000_adapt_captrade_entryfloor_pairpenalty_v1";

function buildByChallengerMap(ledgerEntries: ExperimentLedgerEntry[]): Map<string, ExperimentLedgerEntry> {
  const byChallenger = new Map<string, ExperimentLedgerEntry>();
  for (const e of ledgerEntries) {
    const existing = byChallenger.get(e.challengerProfileId);
    if (!existing || (e.endTimestamp && !existing.endTimestamp)) {
      byChallenger.set(e.challengerProfileId, e);
    }
  }
  return byChallenger;
}

/**
 * Get full decision engine result with classification and precedence.
 */
export function getDecisionEngineResult(
  ledgerEntries: ExperimentLedgerEntry[],
  enabledChallengerIds: string[],
  adaptiveChallengerIds: string[]
): DecisionEngineResult {
  const byChallenger = buildByChallengerMap(ledgerEntries);
  const recommendations: ExperimentRecommendation[] = [];
  const blockedByRecentEvidence: string[] = [];
  const supersededExperiments: string[] = [];

  const captradeEntry = byChallenger.get(CURRENT_WINNER_BRANCH);
  const entryfloorEntry = byChallenger.get(WEAKENED_REFINEMENT);

  const branchComparisons: DecisionEngineResult["branchComparisons"] = [];

  if (captradeEntry && entryfloorEntry) {
    branchComparisons.push({
      base: CURRENT_WINNER_BRANCH,
      refinement: WEAKENED_REFINEMENT,
      outcome: "refinement_weakened_base",
      evidenceNote: "entryfloor_v1 weakened captrade branch in recent evidence; captrade_v1 remains strongest.",
    });
  }

  for (const family of EXPERIMENT_FAMILIES) {
    if (!family.eligibleForRecommendation || !family.challengerPattern) continue;

    const challengerId = family.challengerPattern;
    const entry = byChallenger.get(challengerId);
    const isActive = enabledChallengerIds.includes(challengerId);
    const isAvailable = adaptiveChallengerIds.includes(challengerId);

    let classification: RecommendationClassification = "available_but_not_recommended";
    let blocked = false;
    let blockedReason: string | undefined;
    let recommendationReason = "Available for controlled activation.";
    let whyNotRecommended: string | undefined;
    let confidence = 0.5;

    if (isActive) {
      classification = "already_active";
      blocked = true;
      blockedReason = "Already active; avoid duplicate activation.";
      confidence = 0;
    } else if (TOP_LEVEL_FAILED_CHALLENGERS.has(challengerId)) {
      classification = "blocked";
      blocked = true;
      blockedReason = "Top-level experiment previously failed; do not repeat as exact copy.";
      whyNotRecommended = "Top-level pairpenalty/edgegate failed in prior runs.";
      blockedByRecentEvidence.push(challengerId);
      confidence = 0;
    } else if (challengerId === WEAKENED_REFINEMENT) {
      classification = "refinement_base_weakened";
      blocked = true;
      blockedReason = "Refinement weakened the captrade branch; superseded by stronger base.";
      whyNotRecommended = "entryfloor_v1 weakened captrade in recent evidence; do not re-recommend as top step.";
      recommendationReason = "Weakened refinement; not recommended until new evidence supports.";
      supersededExperiments.push(challengerId);
      blockedByRecentEvidence.push(challengerId);
      confidence = 0;
    } else if (challengerId === ENTRYFLOOR_PAIRPENALTY) {
      if (entryfloorEntry && WEAKENED_STATUSES.has(entryfloorEntry.status)) {
        classification = "available_but_not_recommended";
        blocked = false;
        blockedReason = undefined;
        whyNotRecommended = "Base (entryfloor) weakened; refinement not justified until base improves.";
        recommendationReason = "Layered on weakened base; distinct from top-level pairpenalty but not top recommendation.";
        confidence = 0.2;
      } else {
        classification = "available_but_not_recommended";
        blocked = false;
        recommendationReason = "Layered on winning branch; distinct from top-level pairpenalty. Base status uncertain.";
        confidence = 0.4;
      }
    } else if (challengerId === CURRENT_WINNER_BRANCH) {
      if (isActive) {
        classification = "already_active";
        blocked = true;
        blockedReason = "Already active; preserve as winner.";
      } else {
        classification = "recommended";
        blocked = false;
        recommendationReason = "Current strongest line; activate to preserve winner.";
        confidence = 0.9;
      }
    } else if (entry && (BLOCKED_STATUSES.has(entry.status) || entry.decision === "failed")) {
      classification = "blocked";
      blocked = true;
      blockedReason = `Experiment marked ${entry.status}.`;
      confidence = 0;
    } else {
      classification = "available_but_not_recommended";
      recommendationReason = "Available for controlled activation; not top recommendation based on current evidence.";
      confidence = 0.3;
    }

    recommendations.push({
      challengerProfileId: challengerId,
      familyId: family.familyId,
      baseProfileId: family.allowedBaseProfiles[0] ?? "shadow_1000",
      recommendationReason,
      blocked,
      blockedReason,
      recommendationClassification: classification,
      whyNotRecommended,
      recommendationConfidence: confidence,
    });
  }

  const recommendedNext = recommendations.filter(
    (r) => !r.blocked && r.recommendationClassification === "recommended"
  );

  const availableButNotRecommended = recommendations.filter(
    (r) =>
      !r.blocked &&
      r.recommendationClassification === "available_but_not_recommended"
  );

  const winnerReason =
    captradeEntry?.decision === "successful"
      ? "captrade_v1 outperformed baseline in multiple reads; strongest known line."
      : "captrade_v1 is current winner branch from project evidence.";

  const residualBottleneck =
    "Pair concentration and fill quality may remain residual bottlenecks; pair penalty layered on winner could target these if base is favored.";

  const recommendationRationale =
    recommendedNext.length > 0
      ? `Prioritize preserving ${CURRENT_WINNER_BRANCH}. Weakened refinements (entryfloor) not recommended.`
      : "No top recommendation; current winner preserved. Layered pair penalty available but not recommended while base (entryfloor) weakened.";

  return {
    recommendations,
    recommendedNextExperiments: recommendedNext,
    availableButNotRecommendedExperiments: availableButNotRecommended,
    blockedByRecentEvidence: Array.from(new Set(blockedByRecentEvidence)),
    supersededExperiments: Array.from(new Set(supersededExperiments)),
    currentWinnerBranch: CURRENT_WINNER_BRANCH,
    winnerBranchReason: winnerReason,
    recommendationRationale,
    residualBottleneckHypothesis: residualBottleneck,
    recommendationConfidence: recommendedNext.length > 0 ? 0.7 : 0.4,
    branchComparisons,
  };
}

/**
 * Get recommended next experiments (backward compatible).
 */
export function getExperimentRecommendations(
  ledgerEntries: ExperimentLedgerEntry[],
  enabledChallengerIds: string[],
  adaptiveChallengerIds: string[]
): ExperimentRecommendation[] {
  return getDecisionEngineResult(
    ledgerEntries,
    enabledChallengerIds,
    adaptiveChallengerIds
  ).recommendations;
}

/**
 * Get experiments that are blocked based on ledger.
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
