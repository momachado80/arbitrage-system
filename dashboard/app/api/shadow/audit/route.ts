/**
 * Shadow Closed Trade Audit API — diagnostic endpoint only.
 * Returns full audit of closed shadow trades. No business logic changes.
 */

import { NextResponse } from "next/server";
import { ensureShadowSimulation, getShadowSystemStatus } from "@/lib/shadowSimulationService";
import { getAllShadowProfiles, getRejectionCountsByProfile } from "@/lib/shadowSimulationStore";
import { computeClosedTradeAudit } from "@/lib/shadowClosedTradeAudit";
import { getServiceStats } from "@/lib/marketDataService";
import { getGraphScanStats } from "@/lib/graphScanService";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureShadowSimulation();
    const profiles = getAllShadowProfiles();
    const audit = computeClosedTradeAudit(profiles);
    const status = getShadowSystemStatus();
    const rejectionCountsByProfile = getRejectionCountsByProfile();
    const marketStats = getServiceStats();
    const graphStats = getGraphScanStats();
    return NextResponse.json({
      ...audit,
      opportunitiesSeenLastCycle: status.opportunitiesSeenLastCycle,
      rejectionCountsByProfile,
      upstreamDiagnostics: {
        marketsTracked: marketStats.marketsTracked,
        marketLastError: marketStats.lastError ?? null,
        graphOpportunitiesCount: graphStats.opportunitiesCount,
      },
    });
  } catch (err) {
    console.error("[API /shadow/audit]", err);
    return NextResponse.json(
      { error: "Audit failed", timestamp: new Date().toISOString() },
      { status: 500 }
    );
  }
}
