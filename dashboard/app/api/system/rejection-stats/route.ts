import { NextResponse } from "next/server";
import { getRejectionStats } from "@/lib/tradeRejectionLogger";
import { getPipelineDiagnostics } from "@/lib/shadowPipelineDiagnostics";
import { getEEVFilterQualitySummary } from "@/lib/eevFilterQualityTracker";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const stats = getRejectionStats();
    const pipeline = getPipelineDiagnostics();
    const eevQuality = getEEVFilterQualitySummary();
    return NextResponse.json({
      countsByReason: stats.countsByReason,
      totalRejected: stats.totalRejected,
      timestamp: new Date().toISOString(),
      pipeline: {
        totalDispatches: pipeline.totalDispatches,
        totalEvaluateCalls: pipeline.totalEvaluateCalls,
        totalExecutionCalls: pipeline.totalExecutionCalls,
        totalShadowTradesOpened: pipeline.totalShadowTradesOpened,
        totalFilteredByEEV: pipeline.totalFilteredByEEV,
        totalPassedEEVFilter: pipeline.totalPassedEEVFilter,
        earlyExitCounts: pipeline.earlyExitCounts,
        pipelineTimestamp: pipeline.timestamp,
        eevFilterQualitySummary: eevQuality,
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
          totalFilteredByEEV: 0,
          totalPassedEEVFilter: 0,
          earlyExitCounts: {},
          pipelineTimestamp: new Date().toISOString(),
          eevFilterQualitySummary: null,
        },
      },
      { status: 200 }
    );
  }
}
