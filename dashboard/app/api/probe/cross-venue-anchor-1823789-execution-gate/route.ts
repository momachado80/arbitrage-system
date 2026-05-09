import { NextResponse } from "next/server";
import { buildCrossVenueAnchor1823789ExecutionGateDigest } from "@/lib/crossVenueAnchor1823789ExecutionGate";

export const dynamic = "force-dynamic";
export const maxDuration = 60;
export const runtime = "nodejs";

export async function GET() {
  try {
    const digest = await buildCrossVenueAnchor1823789ExecutionGateDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/cross-venue-anchor-1823789-execution-gate]", err);
    return NextResponse.json(
      {
        error: "cross_venue_anchor_1823789_execution_gate_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
