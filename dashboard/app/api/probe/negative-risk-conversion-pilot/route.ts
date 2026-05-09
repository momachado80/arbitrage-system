import { NextResponse } from "next/server";
import { buildNegativeRiskConversionPilotDigest } from "@/lib/negativeRiskConversionPilot";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function GET() {
  try {
    const digest = await buildNegativeRiskConversionPilotDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/negative-risk-conversion-pilot]", err);
    return NextResponse.json(
      {
        error: "negative_risk_conversion_pilot_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
