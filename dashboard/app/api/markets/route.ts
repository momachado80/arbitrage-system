import { NextResponse } from "next/server";
import { getAllMarkets } from "@/lib/marketDataService";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const markets = getAllMarkets();
    return NextResponse.json({
      count: markets.length,
      markets: markets.slice(0, 200),
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /markets]", err);
    return NextResponse.json(
      { count: 0, markets: [], timestamp: new Date().toISOString() },
      { status: 200 }
    );
  }
}
