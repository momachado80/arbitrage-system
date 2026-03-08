import { NextResponse } from "next/server";
import { getRejectionStats } from "@/lib/tradeRejectionLogger";
import { getPipelineDiagnostics } from "@/lib/shadowPipelineDiagnostics";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const stats = getRejectionStats();
    const pipeline = getPipelineDiagnostics();
    return NextResponse.json({
      countsByReason: stats.countsByReason,
      totalRejected: stats.totalRejected,
      timestamp: new Date().toISOString(),
      pipeline: {
        totalDispatches: pipeline.totalDispatches,
        totalEvaluateCalls: pipeline.totalEvaluateCalls,
        totalExecutionCalls: pipeline.totalExecutionCalls,
        totalShadowTradesOpened: pipeline.totalShadowTradesOpened,
        earlyExitCounts: pipeline.earlyExitCounts,
        pipelineTimestamp: pipeline.timestamp,
      },
    });
  } catch (err) {
    console.error("[API /system/rejection-stats]", err);
    return NextResponse.json(
      {
        countsByReason: {},
        totalRejected: 0,
        timestamp: new Date().toISOString(),
        pipeline: {
          totalDispatches: 0,
          totalEvaluateCalls: 0,
          totalExecutionCalls: 0,
          totalShadowTradesOpened: 0,
          earlyExitCounts: {},
          pipelineTimestamp: new Date().toISOString(),
        },
      },
      { status: 200 }
    );
  }
}
