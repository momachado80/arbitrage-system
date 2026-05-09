/**
 * Next.js instrumentation — compiled for both Node and Edge. Heavy bootstrap lives in
 * lib/nodeInstrumentationBootstrap.ts and runs only when NEXT_RUNTIME is "nodejs"
 * (define-replaced at build time), so edge-instrumentation.js stays free of fs/path.
 *
 * Deferred with setImmediate so register() returns immediately and the HTTP server can
 * bind before background work runs (Railway healthcheck).
 */

export async function register(): Promise<void> {
  if (process.env.NEXT_RUNTIME !== "nodejs") {
    return;
  }
  console.log("[BOOT] Next.js instrumentation register (scheduling deferred bootstrap)");
  setImmediate(() => {
    void import("./lib/nodeInstrumentationBootstrap")
      .then((m) => m.runDeferredNodeInstrumentation())
      .catch((err) => {
        console.error("[BOOT] deferred instrumentation failed", err);
      });
  });
}
