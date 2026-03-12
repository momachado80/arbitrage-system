/**
 * Shadow Adaptive Calibration API — v1.
 * Returns calibration recommendations and challenger specs.
 * Does NOT change baseline profiles, execution, or production behavior.
 */

import { NextResponse } from "next/server";
import { ensureShadowSimulation } from "@/lib/shadowSimulationService";
import { getAllShadowProfiles } from "@/lib/shadowSimulationStore";
import { computeClosedTradeAudit } from "@/lib/shadowClosedTradeAudit";
import { computeAdaptiveCalibration } from "@/lib/adaptiveCalibrationEngine";

export const dynamic = "force-dynamic";

function getEnabledChallengerIds(): Set<string> {
  const raw = process.env.ENABLED_ADAPTIVE_CHALLENGERS ?? "";
  return new Set(raw.split(",").map((s) => s.trim()).filter(Boolean));
}

export async function GET() {
  try {
    ensureShadowSimulation();
    const profiles = getAllShadowProfiles();
    const audit = computeClosedTradeAudit(profiles);
    const result = computeAdaptiveCalibration(audit);
    const enabledIds = getEnabledChallengerIds();

    const adaptiveChallengers = result.adaptiveChallengers.map((c) => ({
      ...c,
      enabledForExecution: enabledIds.has(c.profileId),
    }));

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
