/**
 * Shadow Adaptive Calibration API — v1.
 * Returns calibration recommendations and challenger specs.
 * Does NOT change baseline profiles, execution, or production behavior.
 */

import { NextResponse } from "next/server";
import { ensureShadowSimulation, getProfileConfig, getProfilesForExecution } from "@/lib/shadowSimulationService";
import { getAllShadowProfiles, getPersistenceStatus } from "@/lib/shadowSimulationStore";
import { computeClosedTradeAudit } from "@/lib/shadowClosedTradeAudit";
import {
  computeAdaptiveCalibration,
  buildSyntheticChallengerSpecForActivation,
} from "@/lib/adaptiveCalibrationEngine";

export const dynamic = "force-dynamic";

function getEnabledChallengerIds(): Set<string> {
  const raw = process.env.ENABLED_ADAPTIVE_CHALLENGERS ?? "";
  return new Set(raw.split(",").map((s) => s.trim()).filter(Boolean));
}

export async function GET() {
  try {
    ensureShadowSimulation();
    getProfilesForExecution(); // Materialize enabled challengers before audit (ensureProfileState)
    const profiles = getAllShadowProfiles();
    const audit = computeClosedTradeAudit(profiles, getProfileConfig);
    const result = computeAdaptiveCalibration(audit);
    const enabledIds = getEnabledChallengerIds();

    // Include synthetic specs for challengers explicitly enabled but not in adaptiveChallengers
    // (e.g. when recommendation conditions are not met; ensures activation path works)
    const challengerIdsInResult = new Set(result.adaptiveChallengers.map((c) => c.profileId));
    const syntheticSpecs: typeof result.adaptiveChallengers = [];
    for (const id of Array.from(enabledIds)) {
      if (!challengerIdsInResult.has(id)) {
        const spec = buildSyntheticChallengerSpecForActivation(id);
        if (spec) syntheticSpecs.push(spec);
      }
    }

    const adaptiveChallengers = [
      ...result.adaptiveChallengers.map((c) => ({
        ...c,
        enabledForExecution: enabledIds.has(c.profileId),
      })),
      ...syntheticSpecs.map((c) => ({
        ...c,
        enabledForExecution: true,
      })),
    ];

    const persistenceStatus = getPersistenceStatus();
    return NextResponse.json({
      status: result.status,
      generatedAt: result.generatedAt,
      enoughData: result.enoughData,
      recommendations: result.recommendations,
      adaptiveChallengers,
      promotionReadiness: result.promotionReadiness,
      enabledChallengerIds: Array.from(enabledIds),
      experimentationThresholdsMet: result.experimentationThresholdsMet,
      promotionThresholdsMet: result.promotionThresholdsMet,
      whyChallengersGenerated: result.whyChallengersGenerated,
      persistence: {
        ...persistenceStatus,
      },
    });
  } catch (err) {
    console.error("[API /shadow/adaptive]", err);
    return NextResponse.json(
      {
        status: "error",
        error: err instanceof Error ? err.message : "Unknown error",
        generatedAt: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
