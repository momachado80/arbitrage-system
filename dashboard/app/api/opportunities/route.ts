import { NextResponse } from "next/server";
import { getAllMarkets } from "@/lib/marketDataService";
import { scanMarkets } from "@/lib/probabilityScanner";
import { rankOpportunities } from "@/lib/opportunityEngine";
import { dispatchOpportunity } from "@/lib/executionDispatcher";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const markets = getAllMarkets();
    const edges = scanMarkets(markets);
    const ranked = rankOpportunities(edges);

    for (const opp of ranked) {
      dispatchOpportunity(opp as unknown as Record<string, unknown>);
    }

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
