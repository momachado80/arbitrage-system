import { NextResponse } from "next/server";
import { buildRewardAwareMakerPilotDigest } from "@/lib/rewardAwareMakerPilot";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function GET() {
  try {
    const digest = await buildRewardAwareMakerPilotDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/reward-aware-maker-pilot]", err);
    return NextResponse.json(
      {
        error: "reward_aware_maker_pilot_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
