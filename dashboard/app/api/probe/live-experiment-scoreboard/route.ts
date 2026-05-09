import { NextResponse } from "next/server";
import { buildLiveExperimentScoreboardDigest } from "@/lib/liveExperimentScoreboard";

export const dynamic = "force-dynamic";
export const maxDuration = 60;
export const runtime = "nodejs";

export async function GET() {
  try {
    const digest = buildLiveExperimentScoreboardDigest(process.cwd());
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/live-experiment-scoreboard]", err);
    return NextResponse.json(
      {
        error: "live_experiment_scoreboard_failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  }
}
