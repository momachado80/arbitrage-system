import { NextResponse } from "next/server";
import { ensureCatalogPocketProbe } from "@/lib/catalogPocketProbe";
import { ensurePocketEconomicsProbe } from "@/lib/pocketEconomicsProbe";
import { ensurePocketExecutionProbe } from "@/lib/pocketExecutionProbe";
import { ensureMinimalPaperExecutionProbe } from "@/lib/minimalPaperExecutionProbe";
import { buildSiblingMicroUniverseDigest } from "@/lib/siblingMicroUniverseProbe";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureCatalogPocketProbe();
    ensurePocketEconomicsProbe();
    ensurePocketExecutionProbe();
    ensureMinimalPaperExecutionProbe();
    const digest = buildSiblingMicroUniverseDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/sibling-micro-universe]", err);
    return NextResponse.json(
      {
        error: "sibling_micro_universe_digest_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
