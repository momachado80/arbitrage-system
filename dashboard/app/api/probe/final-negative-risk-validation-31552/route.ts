import { NextResponse } from "next/server";
import { buildFinalNegativeRiskValidation31552Digest } from "@/lib/finalNegativeRiskValidation31552";

export const dynamic = "force-dynamic";
export const maxDuration = 180;

export async function GET() {
  try {
    const digest = await buildFinalNegativeRiskValidation31552Digest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/final-negative-risk-validation-31552]", err);
    return NextResponse.json(
      {
        error: "final_negative_risk_validation_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
