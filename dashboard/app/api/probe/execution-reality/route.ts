import { NextResponse } from "next/server";
import { buildExecutionRealityProbeDigest } from "@/lib/executionRealityProbe";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const digest = buildExecutionRealityProbeDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/execution-reality]", err);
    return NextResponse.json(
      { error: "probe_digest_failed", detail: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}
