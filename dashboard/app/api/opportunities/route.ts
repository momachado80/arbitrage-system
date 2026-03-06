import { NextResponse } from "next/server";
import { getAllMarkets } from "@/lib/marketDataService";
import { scanMarkets } from "@/lib/probabilityScanner";
import { rankOpportunities } from "@/lib/opportunityEngine";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const markets = getAllMarkets();
    const edges = scanMarkets(markets);
    const ranked = rankOpportunities(edges);

    return NextResponse.json({
      count: ranked.length,
      opportunities: ranked,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /opportunities]", err);
    return NextResponse.json(
      { count: 0, opportunities: [], timestamp: new Date().toISOString() },
      { status: 200 }
    );
  }
}
