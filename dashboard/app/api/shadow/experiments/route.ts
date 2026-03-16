/**
 * Shadow Experiments API — experiment governance framework.
 * Exposes registry, ledger, recommendations. Does NOT auto-activate challengers.
 */

import { NextResponse } from "next/server";
import { ensureShadowSimulation, getProfilesForExecution, getProfileConfig } from "@/lib/shadowSimulationService";
import { getAllShadowProfiles } from "@/lib/shadowSimulationStore";
import { computeClosedTradeAudit } from "@/lib/shadowClosedTradeAudit";
import { computeAdaptiveCalibration } from "@/lib/adaptiveCalibrationEngine";
import {
  rehydrateExperimentLedger,
  seedExperimentLedgerIfEmpty,
  getExperimentLedgerEntries,
} from "@/lib/experimentLedger";
import { getSeededHistoricalEntries } from "@/lib/experimentLedgerSeed";
import {
  getDecisionEngineResult,
  getBlockedExperiments,
} from "@/lib/experimentDecisionEngine";
import { computeExperimentReadout } from "@/lib/experimentReadoutHelper";
import { EXPERIMENT_FAMILIES } from "@/lib/experimentRegistry";
import {
  getExperimentLedgerPath,
  isExperimentLedgerAvailable,
} from "@/lib/experimentLedgerPersistence";

export const dynamic = "force-dynamic";

function getEnabledChallengerIds(): string[] {
  const raw = process.env.ENABLED_ADAPTIVE_CHALLENGERS ?? "";
  return raw.split(",").map((s) => s.trim()).filter(Boolean);
}

export async function GET() {
  try {
    ensureShadowSimulation();
    getProfilesForExecution();
    rehydrateExperimentLedger();
    seedExperimentLedgerIfEmpty(getSeededHistoricalEntries());

    const profiles = getAllShadowProfiles();
    const audit = computeClosedTradeAudit(profiles, getProfileConfig);
    const adaptiveResult = computeAdaptiveCalibration(audit);

    const enabledIds = getEnabledChallengerIds();
    const adaptiveChallengerIds = adaptiveResult.adaptiveChallengers.map((c) => c.profileId);
    const ledgerEntries = getExperimentLedgerEntries();

    const decisionResult = getDecisionEngineResult(
      ledgerEntries,
      enabledIds,
      adaptiveChallengerIds
    );
    const blocked = getBlockedExperiments(ledgerEntries);

    const currentWinnerCandidate = "shadow_1000_adapt_captrade_entryfloor_v1";
    const candidateRefinement = "shadow_1000_adapt_captrade_entryfloor_pairpenalty_v1";
    const readout = computeExperimentReadout(
      audit.profileComparison ?? [],
      "shadow_1000",
      enabledIds.includes(currentWinnerCandidate) ? currentWinnerCandidate : null,
      adaptiveChallengerIds.includes(candidateRefinement) ? candidateRefinement : null
    );

    return NextResponse.json({
      experimentRegistry: {
        families: EXPERIMENT_FAMILIES.map((f) => ({
          familyId: f.familyId,
          label: f.label,
          allowedBaseProfiles: f.allowedBaseProfiles,
          isolatedVariable: f.isolatedVariable,
          eligibleForRecommendation: f.eligibleForRecommendation,
          allowedForAutomatedActivation: f.allowedForAutomatedActivation,
        })),
      },
      experimentLedger: {
        entryCount: ledgerEntries.length,
        entries: ledgerEntries,
        persistencePath: getExperimentLedgerPath(),
        ledgerAvailable: isExperimentLedgerAvailable(),
      },
      currentStatus: {
        enabledChallengerIds: enabledIds,
        activeChallengerIds: enabledIds,
        adaptiveChallengerIds,
        currentWinnerBranch: enabledIds.length > 0 ? enabledIds[0] : null,
      },
      blockedExperiments: blocked,
      recommendedNextExperiments: decisionResult.recommendedNextExperiments,
      blockedOrRedundantExperiments: decisionResult.recommendations.filter((r) => r.blocked),
      availableButNotRecommendedExperiments: decisionResult.availableButNotRecommendedExperiments,
      blockedByRecentEvidence: decisionResult.blockedByRecentEvidence,
      supersededExperiments: decisionResult.supersededExperiments,
      currentWinnerBranch: decisionResult.currentWinnerBranch,
      winnerBranchReason: decisionResult.winnerBranchReason,
      recommendationRationale: decisionResult.recommendationRationale,
      residualBottleneckHypothesis: decisionResult.residualBottleneckHypothesis,
      recommendationConfidence: decisionResult.recommendationConfidence,
      branchComparisons: decisionResult.branchComparisons,
      recommendationWithWhyNotRecommended: decisionResult.recommendations
        .filter((r) => r.whyNotRecommended)
        .map((r) => ({ challengerProfileId: r.challengerProfileId, whyNotRecommended: r.whyNotRecommended })),
      readout: {
        directionOfImprovement: readout.directionOfImprovement,
        flowImpactAssessment: readout.flowImpactAssessment,
        readinessAssessment: readout.readinessAssessment,
        baseline: readout.baseline,
        currentWinner: readout.currentWinner,
        candidate: readout.candidate,
      },
      guardrails: {
        noAutomaticActivation: true,
        noAutomaticVariableChanges: true,
        noAutomaticPromotion: true,
      },
      generatedAt: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /shadow/experiments]", err);
    return NextResponse.json(
      {
        error: err instanceof Error ? err.message : "Unknown error",
        generatedAt: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
