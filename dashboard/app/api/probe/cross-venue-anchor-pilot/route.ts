import { NextResponse } from "next/server";
import { buildCrossVenueAnchorPilotDigest } from "@/lib/crossVenueAnchorPilot";

export const dynamic = "force-dynamic";
export const maxDuration = 60;
export const runtime = "nodejs";

export async function GET() {
  try {
    const digest = await buildCrossVenueAnchorPilotDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/cross-venue-anchor-pilot]", err);
    return NextResponse.json(
      {
        error: "cross_venue_anchor_pilot_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
