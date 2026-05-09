import { NextResponse } from "next/server";
import { buildCrossVenueAnchorPilotRefinedDigest } from "@/lib/crossVenueAnchorPilotRefined";

export const dynamic = "force-dynamic";
export const maxDuration = 60;
export const runtime = "nodejs";

export async function GET() {
  try {
    const digest = await buildCrossVenueAnchorPilotRefinedDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/cross-venue-anchor-pilot-refined]", err);
    return NextResponse.json(
      {
        error: "cross_venue_anchor_pilot_refined_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
