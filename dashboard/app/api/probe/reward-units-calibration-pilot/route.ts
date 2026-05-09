import { NextResponse } from "next/server";
import { buildRewardUnitsCalibrationPilotDigest } from "@/lib/rewardUnitsCalibrationPilot";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function GET() {
  try {
    const digest = await buildRewardUnitsCalibrationPilotDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/reward-units-calibration-pilot]", err);
    return NextResponse.json(
      {
        error: "reward_units_calibration_pilot_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
