import { NextResponse } from "next/server";
import { getRejectionStats } from "@/lib/tradeRejectionLogger";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const stats = getRejectionStats();
    return NextResponse.json({
      countsByReason: stats.countsByReason,
      totalRejected: stats.totalRejected,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /system/rejection-stats]", err);
    return NextResponse.json(
      {
        countsByReason: {},
        totalRejected: 0,
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  }
}
