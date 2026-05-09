import { NextResponse } from "next/server";
import { buildExecutionTruthCaptureDigest } from "@/lib/executionTruthCapture";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function GET() {
  try {
    const digest = await buildExecutionTruthCaptureDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/execution-truth-capture]", err);
    return NextResponse.json(
      {
        error: "execution_truth_capture_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
