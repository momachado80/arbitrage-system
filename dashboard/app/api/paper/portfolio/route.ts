import { NextResponse } from "next/server";
import { getPaperPortfolioData } from "@/lib/paperSimulationService";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = getPaperPortfolioData();
    return NextResponse.json({
      summary: data.summary,
      exposureByType: data.exposureByType,
      exposureByCluster: data.exposureByCluster,
      exposureByMarket: data.exposureByMarket,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /paper/portfolio]", err);
    return NextResponse.json(
      {
        summary: {
          startingCapital: 10000,
          currentEquity: 10000,
          availableCapital: 10000,
          reservedCapital: 0,
          realizedPnL: 0,
          unrealizedPnL: 0,
          maxDrawdown: 0,
          activeTrades: 0,
          closedTrades: 0,
          exposureByType: {},
          exposureByCluster: {},
          exposureByMarket: {},
        },
        exposureByType: {},
        exposureByCluster: {},
        exposureByMarket: {},
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  }
}
