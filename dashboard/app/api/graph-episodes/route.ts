import { NextResponse } from "next/server";
import {
  getActiveGraphEpisodes,
  getRecentClosedGraphEpisodes,
  getGraphEpisodeSummary,
} from "@/lib/graphEpisodeStore";
import { ensureGraphScanning } from "@/lib/graphScanService";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureGraphScanning();

    const summary = getGraphEpisodeSummary();
    const active = getActiveGraphEpisodes();
    const recentClosed = getRecentClosedGraphEpisodes(50);

    return NextResponse.json({
      summary,
      active,
      recentClosed,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /graph-episodes]", err);
    return NextResponse.json(
      {
        summary: {
          activeEpisodes: 0,
          closedEpisodes: 0,
          avgActiveDurationMs: 0,
          longestRecentEpisodeMs: 0,
          lastUpdate: null,
          totalEpisodesTracked: 0,
        },
        active: [],
        recentClosed: [],
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  }
}
