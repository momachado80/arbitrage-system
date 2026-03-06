import { NextResponse } from "next/server";
import { getShadowSystemStatus } from "@/lib/shadowSimulationService";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const status = getShadowSystemStatus();
    return NextResponse.json({
      status: status.status,
      lastUpdate: status.lastUpdate,
      profilesRunning: status.profilesRunning,
      opportunitiesSeenLastCycle: status.opportunitiesSeenLastCycle,
      notes: status.notes,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /shadow/system]", err);
    return NextResponse.json(
      {
        status: "error",
        lastUpdate: null,
        profilesRunning: [],
        opportunitiesSeenLastCycle: 0,
        notes: "Shadow simulation unavailable",
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  }
}
