import { NextResponse } from "next/server";
import { buildReachAprilCalibratedReactionDigest } from "@/lib/reachAprilCalibratedReactionPilot";

export const dynamic = "force-dynamic";
export const maxDuration = 300;
export const runtime = "nodejs";

export async function GET() {
  try {
    const digest = await buildReachAprilCalibratedReactionDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/reach-april-calibrated-reaction-pilot]", err);
    return NextResponse.json(
      {
        error: "reach_april_calibrated_reaction_pilot_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
