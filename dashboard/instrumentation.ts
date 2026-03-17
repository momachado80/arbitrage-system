/**
 * Next.js instrumentation — runs on server boot.
 * Starts execution worker, market data, and shadow. Does not block HTTP server startup.
 * CAUSA RAIZ: Bloqueio de 15s impedia o HTTP server de aceitar conexões; runCycle fazia
 * fetch para localhost que ficava pendente ou falhava.
 */

export async function register(): Promise<void> {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    const { recordInstrumentationRan } = await import("./lib/shadowRuntimeDiagnostics");
    recordInstrumentationRan();

    const { startExecutionWorker } = await import("./lib/executionWorker");
    const { ensureShadowSimulation } = await import("./lib/shadowSimulationService");
    const { ensureRunning: ensureMarketDataRunning } = await import("./lib/marketDataService");

    startExecutionWorker();
    ensureMarketDataRunning();
    ensureShadowSimulation();
    console.log("[BOOT] Shadow simulation bootstrap complete (non-blocking)");
  }
}
