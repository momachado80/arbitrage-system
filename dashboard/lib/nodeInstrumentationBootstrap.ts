/**
 * Server-only bootstrap (fs, workers, probes). Loaded from instrumentation.ts
 * only when NEXT_RUNTIME is "nodejs" so the edge-instrumentation bundle does not
 * pull Node built-ins.
 */

const SHADOW_LOOP_DEFER_MS = 5_000;

export async function runDeferredNodeInstrumentation(): Promise<void> {
  const g = globalThis as unknown as { __arbInstrumentationInvocations?: number };
  g.__arbInstrumentationInvocations = (g.__arbInstrumentationInvocations ?? 0) + 1;
  if (g.__arbInstrumentationInvocations === 1) {
    console.log("[BOOT] instrumentation deferred bootstrap (first invocation this process)");
  } else {
    console.log(
      `[BOOT] instrumentation deferred bootstrap again (count=${g.__arbInstrumentationInvocations}; HMR or duplicate boot — background loops are globalThis-guarded)`
    );
  }

  const { recordInstrumentationRan } = await import("./shadowRuntimeDiagnostics");
  recordInstrumentationRan();

  const { startExecutionWorker } = await import("./executionWorker");
  const { ensureShadowSimulation } = await import("./shadowSimulationService");
  const { ensureRunning: ensureMarketDataRunning } = await import("./marketDataService");

  ensureMarketDataRunning();
  startExecutionWorker();

  const { ensureCatalogPocketProbe } = await import("./catalogPocketProbe");
  ensureCatalogPocketProbe();
  console.log("[BOOT] catalog_pocket_probe scheduler ensured (no first-GET dependency)");

  const { ensurePocketEconomicsProbe } = await import("./pocketEconomicsProbe");
  ensurePocketEconomicsProbe();
  console.log("[BOOT] pocket_economics_probe scheduler ensured (family other:price_above:>3m only)");

  const { ensurePocketExecutionProbe } = await import("./pocketExecutionProbe");
  ensurePocketExecutionProbe();
  console.log("[BOOT] pocket_execution_probe scheduler ensured (stable promoted pockets, observation-only)");

  const { ensureMinimalPaperExecutionProbe } = await import("./minimalPaperExecutionProbe");
  ensureMinimalPaperExecutionProbe();
  console.log("[BOOT] minimal_paper_execution_probe ensured (paper-only, subordinate to execution promotion gate)");

  const { ensureMomentumSnipingProbe } = await import("./momentumSnipingProbe");
  ensureMomentumSnipingProbe();
  console.log("[BOOT] momentum_sniping_probe ensured (observational microstructure, no execution)");

  const { ensureMarketDataTruthCollector } = await import("./marketDataTruthCapture");
  ensureMarketDataTruthCollector();
  console.log("[BOOT] market_data_truth collector ensured (Gamma + CLOB REST microstructure to disk)");

  setTimeout(() => {
    ensureShadowSimulation();
    console.log("[BOOT] Shadow simulation bootstrap complete");
  }, SHADOW_LOOP_DEFER_MS);
}
