/**
 * Diagnóstico upstream do ciclo paper (antes do prefilter recommendedCapital).
 * Estado em globalThis — mesmo motivo que paperOpenDiagnostics (chunks Next).
 */

const GLOBAL_KEY = "__paperUpstreamDiagnostics_v1";

export type UpstreamStandardMetrics =
  | {
      mode: "whitelist_scan";
      whitelistMarketIdsCount: number;
      /** Subset operacional após auto-exclusão / reposição (paper adaptive whitelist). */
      operationalWhitelistMarketIdsCount?: number;
      allMarketsCacheCount: number;
      whitelistHitsFromAllMarkets: number;
      fetchNormalizedAttempts: number;
      fetchNormalizedSucceeded: number;
      subsetMarketsPassedToScan: number;
      scanEdgesRaw: number;
      scanEdgesAfterWhitelistFilter: number;
    }
  | {
      mode: "http_opportunities";
      httpOpportunitiesReturned: number;
    };

export type PaperUpstreamLastCycle = UpstreamStandardMetrics & {
  atIso: string;
  graphOpportunitiesRaw: number;
  graphAfterWhitelist: number;
  mergedRawCount: number;
  mergedAfterWhitelistCount: number;
  enteringRecommendedCapitalPositiveCount: number;
};

type Cumulative = {
  cyclesCompleted: number;
  sumMergedAfterWhitelist: number;
  sumGraphRaw: number;
  sumStandardOppsReturned: number;
  /** último ciclo: stdOpps.length (pode diferir de scanEdgesAfterWhitelistFilter se a lógica mudar no futuro) */
  lastStandardOppsReturned: number;
};

type SyncHint = {
  /** Incrementado no início síncrono de cada runCycle (antes do Promise.all). */
  runCycleStarts: number;
  /** Tamanho de PAPER_MARKET_IDS quando em modo whitelist; null = modo HTTP. */
  lastWhitelistMarketIdsCount: number | null;
};

type Store = {
  lastCycle: PaperUpstreamLastCycle | null;
  cumulative: Cumulative;
  syncHint: SyncHint;
};

function emptyCumulative(): Cumulative {
  return {
    cyclesCompleted: 0,
    sumMergedAfterWhitelist: 0,
    sumGraphRaw: 0,
    sumStandardOppsReturned: 0,
    lastStandardOppsReturned: 0,
  };
}

function getStore(): Store {
  const g = globalThis as unknown as Record<string, Store>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = {
      lastCycle: null,
      cumulative: emptyCumulative(),
      syncHint: { runCycleStarts: 0, lastWhitelistMarketIdsCount: null },
    };
  }
  return g[GLOBAL_KEY];
}

/** Chamado no topo síncrono de runCycle (não depende do async completar). */
export function recordPaperUpstreamRunCycleStart(whitelist: Set<string> | null): void {
  const st = getStore();
  st.syncHint.runCycleStarts += 1;
  st.syncHint.lastWhitelistMarketIdsCount = whitelist ? whitelist.size : null;
}

export function recordPaperUpstreamCycleComplete(args: {
  standard: UpstreamStandardMetrics;
  standardOppsReturned: number;
  graphOpportunitiesRaw: number;
  graphAfterWhitelist: number;
  mergedRawCount: number;
  mergedAfterWhitelistCount: number;
  enteringRecommendedCapitalPositiveCount: number;
}): void {
  const st = getStore();
  const atIso = new Date().toISOString();
  const last: PaperUpstreamLastCycle = {
    ...args.standard,
    atIso,
    graphOpportunitiesRaw: args.graphOpportunitiesRaw,
    graphAfterWhitelist: args.graphAfterWhitelist,
    mergedRawCount: args.mergedRawCount,
    mergedAfterWhitelistCount: args.mergedAfterWhitelistCount,
    enteringRecommendedCapitalPositiveCount: args.enteringRecommendedCapitalPositiveCount,
  };
  st.lastCycle = last;
  st.cumulative.cyclesCompleted += 1;
  st.cumulative.sumMergedAfterWhitelist += args.mergedAfterWhitelistCount;
  st.cumulative.sumGraphRaw += args.graphOpportunitiesRaw;
  st.cumulative.sumStandardOppsReturned += args.standardOppsReturned;
  st.cumulative.lastStandardOppsReturned = args.standardOppsReturned;
}

export function getPaperUpstreamDiagnostics(): {
  lastCycle: PaperUpstreamLastCycle | null;
  cumulative: Cumulative;
  syncHint: SyncHint;
} {
  const st = getStore();
  return {
    lastCycle: st.lastCycle,
    cumulative: { ...st.cumulative },
    syncHint: { ...st.syncHint },
  };
}
