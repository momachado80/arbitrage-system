import { NextResponse } from "next/server";
import {
  ensurePocketEconomicsProbe,
  buildPocketEconomicsDigest,
} from "@/lib/pocketEconomicsProbe";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensurePocketEconomicsProbe();
    const digest = buildPocketEconomicsDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/pocket-economics]", err);
    return NextResponse.json(
      {
        error: "pocket_economics_digest_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
