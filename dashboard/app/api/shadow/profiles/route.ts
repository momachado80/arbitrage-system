import { NextResponse } from "next/server";
import {
  ensureShadowSimulation,
  getProfilesForExecution,
  getProfileConfig,
} from "@/lib/shadowSimulationService";
import { getAllShadowProfiles } from "@/lib/shadowSimulationStore";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureShadowSimulation();
    getProfilesForExecution();
    const profiles = getAllShadowProfiles().map((p) => {
      const config = getProfileConfig(p.profileId);
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
        closedTrades: p.closedTrades.length,
        lastUpdate: p.lastUpdate,
        isAdaptive: config?.isAdaptive ?? false,
        baseProfileId: config?.baseProfileId ?? null,
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
