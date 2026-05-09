/**
 * Substituído via next.config webpack alias apenas no compiler edge-server,
 * para o ficheiro real em lib/nodeInstrumentationBootstrap.ts não entrar no edge-instrumentation.js.
 */
export async function runDeferredNodeInstrumentation(): Promise<void> {}
