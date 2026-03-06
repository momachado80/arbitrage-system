import { NextResponse } from "next/server";
import { getGraphOpportunities, getGraphSummary, getGraphScanStats } from "@/lib/graphScanService";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const ranked = getGraphOpportunities();
    const summary = getGraphSummary();
    const stats = getGraphScanStats();

    return NextResponse.json({
      summary: {
        ...summary,
        lastUpdate: stats.lastScanMs
          ? new Date(stats.lastScanMs).toISOString()
          : null,
        isScanning: stats.isScanning,
      },
      opportunities: ranked,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /graph-opportunities]", err);
    return NextResponse.json(
      {
        summary: {
          graphOpportunitiesDetected: 0,
          averageConfidence: 0,
          averageEdge: 0,
          numberOfClustersScanned: 0,
          lastUpdate: null,
        },
        opportunities: [],
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  }
}
