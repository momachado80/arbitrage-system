/**
 * Next.js instrumentation — runs on server boot.
 * Starts the execution worker for continuous arbitrage simulation.
 */

export async function register(): Promise<void> {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    const { startExecutionWorker } = await import("./lib/executionWorker");
    startExecutionWorker();
  }
}
