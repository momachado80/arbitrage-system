import { NextResponse } from "next/server";
import { buildFinalPaperFillValidation629558Digest } from "@/lib/finalPaperFillValidation629558";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function GET() {
  try {
    const digest = await buildFinalPaperFillValidation629558Digest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/final-paper-fill-validation-629558]", err);
    return NextResponse.json(
      {
        error: "final_paper_fill_validation_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
