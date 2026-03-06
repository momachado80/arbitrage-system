import { NextResponse } from "next/server";
import { getPaperAnalyticsData } from "@/lib/paperSimulationService";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = getPaperAnalyticsData();
    return NextResponse.json({
      analytics: data.analytics,
      equityCurve: data.equityCurve,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /paper/analytics]", err);
    return NextResponse.json(
      {
        analytics: {
          totalTrades: 0,
          closedTrades: 0,
          winRate: 0,
          avgReturn: 0,
          avgPnL: 0,
          totalPnL: 0,
          profitFactor: 0,
          maxDrawdown: 0,
          avgHoldingTimeMs: 0,
          avgNetEdgeAtEntry: 0,
          avgFilledCapital: 0,
          utilizationRate: 0,
          pnlByOpportunityType: {},
          pnlBySourceType: {},
          averageCapacityConfidence: 0,
        },
        equityCurve: [],
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  }
}
