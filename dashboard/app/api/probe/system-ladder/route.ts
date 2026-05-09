import { NextResponse } from "next/server";
import { buildProbeSystemLadderDigest } from "@/lib/probeSystemLadderDigest";

export const dynamic = "force-dynamic";

export async function GET() {
  const t0 = Date.now();
  try {
    const digest = buildProbeSystemLadderDigest();
    console.log(`[system-ladder] route totalMs=${Date.now() - t0}`);
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/system-ladder]", err);
    return NextResponse.json(
      {
        error: "probe_system_ladder_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
