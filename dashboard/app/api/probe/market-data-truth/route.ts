import { NextResponse } from "next/server";
import { buildMarketDataTruthDigest } from "@/lib/marketDataTruthCapture";

export const dynamic = "force-dynamic";
export const maxDuration = 180;

export async function GET() {
  try {
    const digest = await buildMarketDataTruthDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/market-data-truth]", err);
    return NextResponse.json(
      {
        error: "market_data_truth_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
