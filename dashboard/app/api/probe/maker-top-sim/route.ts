import { NextResponse } from "next/server";
import { buildMakerTopCandidateSimulationDigest } from "@/lib/makerTopCandidateSimulation";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const digest = buildMakerTopCandidateSimulationDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/maker-top-sim]", err);
    return NextResponse.json(
      { error: "maker_top_sim_failed", detail: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}
