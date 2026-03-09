/**
 * Shadow Closed Trade Audit API — diagnostic endpoint only.
 * Returns full audit of closed shadow trades. No business logic changes.
 */

import { NextResponse } from "next/server";
import { ensureShadowSimulation } from "@/lib/shadowSimulationService";
import { getAllShadowProfiles } from "@/lib/shadowSimulationStore";
import { computeClosedTradeAudit } from "@/lib/shadowClosedTradeAudit";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureShadowSimulation();
    const profiles = getAllShadowProfiles();
    const audit = computeClosedTradeAudit(profiles);
    return NextResponse.json(audit);
  } catch (err) {
    console.error("[API /shadow/audit]", err);
    return NextResponse.json(
      { error: "Audit failed", timestamp: new Date().toISOString() },
      { status: 500 }
    );
  }
}
