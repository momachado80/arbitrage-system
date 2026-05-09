import { NextResponse } from "next/server";
import { getPaperSystemStatus } from "@/lib/paperSimulationService";
import { getProcessRuntimeSummary } from "@/lib/nodeProcessRuntimeState";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const status = getPaperSystemStatus();
    return NextResponse.json({
      status: status.status,
      lastUpdate: status.lastUpdate,
      lastPaperCycleAsyncError: status.lastPaperCycleAsyncError,
      lastPaperCycleAsyncErrorAt: status.lastPaperCycleAsyncErrorAt,
      startingCapital: status.startingCapital,
      currentEquity: status.currentEquity,
      availableCapital: status.availableCapital,
      reservedCapital: status.reservedCapital,
      activeTrades: status.activeTrades,
      closedTrades: status.closedTrades,
      realizedPnL: status.realizedPnL,
      unrealizedPnL: status.unrealizedPnL,
      openEntryDiagnostics: status.openEntryDiagnostics,
      openUpstreamDiagnostics: status.openUpstreamDiagnostics,
      gammaFetchByIdDiagnostics: status.gammaFetchByIdDiagnostics,
      paperWhitelistHealth: status.paperWhitelistHealth,
      paperAdaptiveWhitelist: status.paperAdaptiveWhitelist,
      paperTradeLifecycleDiagnostics: status.paperTradeLifecycleDiagnostics,
      simulateEntryDiagnostics: status.simulateEntryDiagnostics,
      paperEntryPolicy: status.paperEntryPolicy,
      paperStateIntegrity: status.paperStateIntegrity,
      processRuntime: getProcessRuntimeSummary(),
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[API /paper/system]", err);
    return NextResponse.json(
      {
        status: "error",
        lastUpdate: null,
        lastPaperCycleAsyncError: null,
        lastPaperCycleAsyncErrorAt: null,
        startingCapital: 10000,
        currentEquity: 10000,
        availableCapital: 10000,
        reservedCapital: 0,
        activeTrades: 0,
        closedTrades: 0,
        realizedPnL: 0,
        unrealizedPnL: 0,
        openEntryDiagnostics: null,
        openUpstreamDiagnostics: null,
        gammaFetchByIdDiagnostics: null,
        paperWhitelistHealth: null,
        paperAdaptiveWhitelist: null,
        paperTradeLifecycleDiagnostics: null,
        simulateEntryDiagnostics: null,
        paperEntryPolicy: null,
        paperStateIntegrity: null,
        processRuntime: null,
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  }
}
