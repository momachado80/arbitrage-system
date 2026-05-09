import { NextResponse } from "next/server";
import { buildConstraintFirstEdgeDiscoveryDigest } from "@/lib/constraintFirstEdgeDiscovery";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const digest = buildConstraintFirstEdgeDiscoveryDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/constraint-first-edge]", err);
    return NextResponse.json(
      { error: "probe_digest_failed", detail: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}
