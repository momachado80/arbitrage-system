import { NextResponse } from "next/server";
import {
  ensureLowLiquidityProbe,
  buildLowLiquidityProbeDigest,
} from "@/lib/lowLiquidityEdgeProbe";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureLowLiquidityProbe();
    const digest = buildLowLiquidityProbeDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/low-liquidity]", err);
    return NextResponse.json(
      { error: "probe_digest_failed", detail: err instanceof Error ? err.message : String(err) },
      { status: 500 }
    );
  }
}
