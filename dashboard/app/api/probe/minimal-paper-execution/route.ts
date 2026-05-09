import { NextResponse } from "next/server";
import {
  ensureMinimalPaperExecutionProbe,
  buildMinimalPaperExecutionDigest,
} from "@/lib/minimalPaperExecutionProbe";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureMinimalPaperExecutionProbe();
    const digest = buildMinimalPaperExecutionDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/minimal-paper-execution]", err);
    return NextResponse.json(
      {
        error: "minimal_paper_execution_digest_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
