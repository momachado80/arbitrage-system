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

export async function GET() {
  try {
    ensureShadowSimulation();
    const profiles = getAllShadowProfiles();
    const audit = computeClosedTradeAudit(profiles);
    const result = computeAdaptiveCalibration(audit);

    return NextResponse.json({
      status: result.status,
      generatedAt: result.generatedAt,
      enoughData: result.enoughData,
      recommendations: result.recommendations,
      adaptiveChallengers: result.adaptiveChallengers,
      promotionReadiness: result.promotionReadiness,
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
