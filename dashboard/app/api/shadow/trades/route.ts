import { NextResponse } from "next/server";
import { ensureShadowSimulation } from "@/lib/shadowSimulationService";
import { getShadowTrades, getAllShadowProfiles } from "@/lib/shadowSimulationStore";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  try {
    ensureShadowSimulation();
    const { searchParams } = new URL(req.url);
    const profileId = searchParams.get("profileId");

    if (profileId) {
      const data = getShadowTrades(profileId);
      return NextResponse.json({
        profileId,
        active: data.active,
        recentClosed: data.recentClosed,
        timestamp: new Date().toISOString(),
      });
    }

    const profiles = getAllShadowProfiles();
    const result: Record<string, { active: unknown[]; recentClosed: unknown[] }> = {};
    for (const p of profiles) {
      const data = getShadowTrades(p.profileId);
      result[p.profileId] = { active: data.active, recentClosed: data.recentClosed };
    }
    return NextResponse.json({
      profiles: result,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /shadow/trades]", err);
    return NextResponse.json(
      { active: [], recentClosed: [], timestamp: new Date().toISOString() },
      { status: 200 }
    );
  }
}
