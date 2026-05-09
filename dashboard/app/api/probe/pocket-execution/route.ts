import { NextResponse } from "next/server";
import {
  ensurePocketExecutionProbe,
  buildPocketExecutionDigest,
} from "@/lib/pocketExecutionProbe";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensurePocketExecutionProbe();
    const digest = buildPocketExecutionDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/pocket-execution]", err);
    return NextResponse.json(
      {
        error: "pocket_execution_digest_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
