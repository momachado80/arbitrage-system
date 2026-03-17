/**
 * Next.js instrumentation — runs on server boot.
 * Starts the execution worker and shadow simulation eagerly so upstream/markets bootstrap before first request.
 */

const BOOTSTRAP_MARKET_WAIT_MS = 15_000;
const BOOTSTRAP_POLL_MS = 500;

export async function register(): Promise<void> {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    const { startExecutionWorker } = await import("./lib/executionWorker");
    const { ensureShadowSimulation } = await import("./lib/shadowSimulationService");
    const { ensureRunning: ensureMarketDataRunning, getServiceStats } = await import("./lib/marketDataService");

    startExecutionWorker();
    ensureMarketDataRunning();

    await new Promise((r) => setTimeout(r, 1000));

    // Wait up to BOOTSTRAP_MARKET_WAIT_MS for first successful market load (or visible failure)
    const deadline = Date.now() + BOOTSTRAP_MARKET_WAIT_MS;
    while (Date.now() < deadline) {
      const stats = getServiceStats();
      if (stats.marketsTracked > 0) {
        console.log("[BOOT] Market bootstrap complete:", stats.marketsTracked, "markets");
        break;
      }
      if (stats.lastError && stats.refreshAttemptedCount >= 1) {
        console.warn("[BOOT] Market bootstrap failed:", stats.lastError);
        break;
      }
      await new Promise((r) => setTimeout(r, BOOTSTRAP_POLL_MS));
    }

    ensureShadowSimulation();
    console.log("[BOOT] Shadow simulation bootstrap complete");
  }
}
