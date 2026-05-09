import { NextResponse } from "next/server";
import { buildExecutionCostRecalibrationDigest } from "@/lib/executionCostRecalibration";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function GET() {
  try {
    const digest = await buildExecutionCostRecalibrationDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/execution-cost-recalibration]", err);
    return NextResponse.json(
      {
        error: "execution_cost_recalibration_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
