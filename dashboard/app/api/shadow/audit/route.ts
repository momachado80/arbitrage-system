/**
 * Shadow Closed Trade Audit API — diagnostic endpoint only.
 * Returns full audit of closed shadow trades. No business logic changes.
 */

import { NextResponse } from "next/server";
import { ensureShadowSimulation, getShadowSystemStatus, getProfileConfig, getProfilesForExecution } from "@/lib/shadowSimulationService";
import { getAllShadowProfiles, getRejectionCountsByProfile, getPersistenceStatus } from "@/lib/shadowSimulationStore";
import { computeClosedTradeAudit } from "@/lib/shadowClosedTradeAudit";
import { getSelectionDiagnostics } from "@/lib/shadowSelectionDiagnostics";
import { getFillGuardDiagnostics } from "@/lib/fillGuardDiagnostics";
import { getEntryThresholdCausalDiagnostics } from "@/lib/entryThresholdCausalDiagnostics";
import { getProfileById } from "@/lib/shadowSimulationProfiles";
import { getServiceStats } from "@/lib/marketDataService";
import { getGraphScanStats } from "@/lib/graphScanService";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureShadowSimulation();
    getProfilesForExecution(); // Materialize enabled challengers before audit (ensureProfileState for challengers)
    const profiles = getAllShadowProfiles();
    const audit = computeClosedTradeAudit(profiles, getProfileConfig);
    const status = getShadowSystemStatus();
    const rejectionCountsByProfile = getRejectionCountsByProfile();
    const marketStats = getServiceStats();
    const graphStats = getGraphScanStats();
    const maxHoldingTimeMsByProfile: Record<string, number> = {};
    for (const p of profiles) {
      const cfg = getProfileById(p.profileId) ?? getProfileConfig(p.profileId);
      if (cfg) maxHoldingTimeMsByProfile[p.profileId] = cfg.maxHoldingTimeMs;
    }
    const persistenceStatus = getPersistenceStatus();
    const selectionDiagnostics = getSelectionDiagnostics(
      profiles.map((p) => ({
        profileId: p.profileId,
        closedTrades: p.closedTrades,
        activeTrades: p.activeTrades,
      })),
      rejectionCountsByProfile
    );
    const fillGuardDiagnostics = getFillGuardDiagnostics(profiles);
    const entryThresholdCausalDiagnostics = getEntryThresholdCausalDiagnostics((pid) => {
      const cfg = getProfileById(pid) ?? getProfileConfig(pid);
      return cfg as { minNetCapturableEdgeToTrade?: number; minCapturableEdgeToTrade?: number } | undefined;
    });
    const effectiveEntryThresholdByProfile: Record<string, number> = {};
    for (const p of profiles) {
      const cfg = getProfileById(p.profileId) ?? getProfileConfig(p.profileId);
      const v = cfg?.minCapturableEdgeToTrade ?? cfg?.minNetCapturableEdgeToTrade;
      if (typeof v === "number") effectiveEntryThresholdByProfile[p.profileId] = v;
    }
    return NextResponse.json({
      ...audit,
      maxHoldingTimeMsByProfile,
      opportunitiesSeenLastCycle: status.opportunitiesSeenLastCycle,
      rejectionCountsByProfile,
      selectionDiagnostics,
      fillGuardDiagnostics,
      entryThresholdCausalDiagnostics,
      effectiveEntryThresholdByProfile,
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
