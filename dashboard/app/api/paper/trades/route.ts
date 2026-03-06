import { NextResponse } from "next/server";
import { getPaperTradesData } from "@/lib/paperSimulationService";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = getPaperTradesData();
    return NextResponse.json({
      active: data.active,
      recentClosed: data.recentClosed,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /paper/trades]", err);
    return NextResponse.json(
      { active: [], recentClosed: [], timestamp: new Date().toISOString() },
      { status: 200 }
    );
  }
}
