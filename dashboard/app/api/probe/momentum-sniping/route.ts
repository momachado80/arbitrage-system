import { NextResponse } from "next/server";
import {
  ensureMomentumSnipingProbe,
  buildMomentumSnipingDigest,
} from "@/lib/momentumSnipingProbe";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureMomentumSnipingProbe();
    const digest = buildMomentumSnipingDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/momentum-sniping]", err);
    return NextResponse.json(
      {
        error: "momentum_sniping_digest_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
