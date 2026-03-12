/**
 * Shadow Closed Trade Audit API — diagnostic endpoint only.
 * Returns full audit of closed shadow trades. No business logic changes.
 */

import { NextResponse } from "next/server";
import { ensureShadowSimulation, getShadowSystemStatus } from "@/lib/shadowSimulationService";
import { getAllShadowProfiles, getRejectionCountsByProfile, getPersistenceStatus } from "@/lib/shadowSimulationStore";
import { computeClosedTradeAudit } from "@/lib/shadowClosedTradeAudit";
import { getProfileById } from "@/lib/shadowSimulationProfiles";
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
    const maxHoldingTimeMsByProfile: Record<string, number> = {};
    for (const p of profiles) {
      const cfg = getProfileById(p.profileId);
      if (cfg) maxHoldingTimeMsByProfile[p.profileId] = cfg.maxHoldingTimeMs;
    }
    const persistenceStatus = getPersistenceStatus();
    return NextResponse.json({
      ...audit,
      maxHoldingTimeMsByProfile,
      opportunitiesSeenLastCycle: status.opportunitiesSeenLastCycle,
      rejectionCountsByProfile,
      persistence: {
        ...persistenceStatus,
      },
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
