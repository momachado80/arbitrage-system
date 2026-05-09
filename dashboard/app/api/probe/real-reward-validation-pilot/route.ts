import { NextResponse } from "next/server";
import { buildRealRewardValidationPilotDigest } from "@/lib/realRewardValidationPilot";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function GET() {
  try {
    const digest = await buildRealRewardValidationPilotDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/real-reward-validation-pilot]", err);
    return NextResponse.json(
      {
        error: "real_reward_validation_pilot_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
