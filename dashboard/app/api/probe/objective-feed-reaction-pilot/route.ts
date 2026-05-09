import { NextResponse } from "next/server";
import { buildObjectiveFeedReactionPilotDigest } from "@/lib/objectiveFeedReactionPilot";

export const dynamic = "force-dynamic";
export const maxDuration = 300;
export const runtime = "nodejs";

export async function GET() {
  try {
    const digest = await buildObjectiveFeedReactionPilotDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/objective-feed-reaction-pilot]", err);
    return NextResponse.json(
      {
        error: "objective_feed_reaction_pilot_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
