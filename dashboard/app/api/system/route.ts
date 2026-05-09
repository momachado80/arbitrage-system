import { NextResponse } from "next/server";
import { getAllMarkets, getServiceStats } from "@/lib/marketDataService";
import { scanMarkets } from "@/lib/probabilityScanner";
import { rankOpportunities, computeSystemMetrics } from "@/lib/opportunityEngine";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const markets = getAllMarkets();
    const edges = scanMarkets(markets);
    const ranked = rankOpportunities(edges);
    const stats = getServiceStats();
    const metrics = computeSystemMetrics(ranked, markets.length);

    return NextResponse.json({
      ...metrics,
      systemStatus: stats.lastError ? "degraded" : "operational",
      lastUpdate: stats.lastRefreshMs
        ? new Date(stats.lastRefreshMs).toISOString()
        : null,
      lastError: stats.lastError,
      fetchCount: stats.fetchCount,
      marketRefreshMetrics: stats.refreshMetrics,
      inFlightRefreshCount: stats.inFlightRefreshCount,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /system]", err);
    return NextResponse.json(
      {
        marketQuality: 0,
        marketsTracked: 0,
        opportunitiesDetected: 0,
        systemStatus: "error",
        lastUpdate: null,
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  }
}
