import { NextResponse } from "next/server";
import { getPaperSystemStatus } from "@/lib/paperSimulationService";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const status = getPaperSystemStatus();
    return NextResponse.json({
      status: status.status,
      lastUpdate: status.lastUpdate,
      startingCapital: status.startingCapital,
      currentEquity: status.currentEquity,
      availableCapital: status.availableCapital,
      reservedCapital: status.reservedCapital,
      activeTrades: status.activeTrades,
      closedTrades: status.closedTrades,
      realizedPnL: status.realizedPnL,
      unrealizedPnL: status.unrealizedPnL,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /paper/system]", err);
    return NextResponse.json(
      {
        status: "error",
        lastUpdate: null,
        startingCapital: 10000,
        currentEquity: 10000,
        availableCapital: 10000,
        reservedCapital: 0,
        activeTrades: 0,
        closedTrades: 0,
        realizedPnL: 0,
        unrealizedPnL: 0,
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  }
}
