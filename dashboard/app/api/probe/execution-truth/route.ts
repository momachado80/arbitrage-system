import { NextResponse } from "next/server";
import { buildExecutionTruthDigest } from "@/lib/executionTruthEngine";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const digest = buildExecutionTruthDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/execution-truth]", err);
    return NextResponse.json(
      { error: "execution_truth_failed", detail: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}
