import { NextResponse } from "next/server";
import { ensureShadowSimulation } from "@/lib/shadowSimulationService";
import { getShadowAnalytics, getAllShadowProfiles } from "@/lib/shadowSimulationStore";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  try {
    ensureShadowSimulation();
    const { searchParams } = new URL(req.url);
    const profileId = searchParams.get("profileId");

    if (profileId) {
      const analytics = getShadowAnalytics(profileId);
      const state = getAllShadowProfiles().find((p) => p.profileId === profileId);
      return NextResponse.json({
        profileId,
        analytics: analytics ?? {},
        equityCurve: state?.equityCurve ?? [],
        timestamp: new Date().toISOString(),
      });
    }

    const profiles = getAllShadowProfiles();
    const result: Record<string, { analytics: unknown; equityCurve: unknown[] }> = {};
    for (const p of profiles) {
      const analytics = getShadowAnalytics(p.profileId);
      result[p.profileId] = {
        analytics: analytics ?? {},
        equityCurve: p.equityCurve ?? [],
      };
    }
    return NextResponse.json({
      profiles: result,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /shadow/analytics]", err);
    return NextResponse.json(
      { analytics: {}, equityCurve: [], timestamp: new Date().toISOString() },
      { status: 200 }
    );
  }
}
