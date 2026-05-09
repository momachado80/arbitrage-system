import { NextResponse } from "next/server";
import { buildMakerInventoryProbeDigest } from "@/lib/makerInventoryProbe";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const digest = buildMakerInventoryProbeDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/maker-inventory]", err);
    return NextResponse.json(
      { error: "probe_digest_failed", detail: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}
