import { NextResponse } from "next/server";
import {
  ensureShadowSimulation,
  getProfilesForExecution,
  getProfileConfig,
} from "@/lib/shadowSimulationService";
import { getAllShadowProfiles, getRejectionCountsByProfile } from "@/lib/shadowSimulationStore";
import { getProfileFunnelSummary } from "@/lib/profileEligibilityFunnel";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureShadowSimulation();
    getProfilesForExecution();
    const rejectionCountsByProfile = getRejectionCountsByProfile();
    const profiles = getAllShadowProfiles().map((p) => {
      const config = getProfileConfig(p.profileId);
      const closedCount = p.closedTrades.filter((t) => t.status === "closed" && t.closedAt).length;
      const funnel = getProfileFunnelSummary(
        p.profileId,
        closedCount,
        p.realizedPnL,
        rejectionCountsByProfile[p.profileId]
      );
      return {
        profileId: p.profileId,
        label: p.label,
        startingCapital: p.startingCapital,
        maxHoldingTimeMs: config?.maxHoldingTimeMs ?? null,
        currentEquity: p.currentEquity,
        availableCapital: p.availableCapital,
        reservedCapital: p.reservedCapital,
        realizedPnL: p.realizedPnL,
        unrealizedPnL: p.unrealizedPnL,
        maxDrawdown: p.maxDrawdown,
        activeTrades: p.activeTrades.length,
        closedTrades: closedCount,
        lastUpdate: p.lastUpdate,
        lastCycleProcessedAt: (p as { lastCycleProcessedAt?: string | null }).lastCycleProcessedAt ?? null,
        isAdaptive: config?.isAdaptive ?? false,
        baseProfileId: config?.baseProfileId ?? null,
        eligibilityFunnel: funnel,
      };
    });
    return NextResponse.json({
      profiles,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /shadow/profiles]", err);
    return NextResponse.json(
      { profiles: [], timestamp: new Date().toISOString() },
      { status: 200 }
    );
  }
}
