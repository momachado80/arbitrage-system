/**
 * Market Source Diagnostics — observability for opportunity sources.
 * Tracks standard vs graph fetch and merge. In-memory only.
 */

let lastStandardFetchAt: string | null = null;
let lastStandardFetchError: string | null = null;
let lastStandardCount = 0;

let lastGraphFetchAt: string | null = null;
let lastGraphFetchError: string | null = null;
let lastGraphCount = 0;

let lastMergedAt: string | null = null;
let lastMergedCount = 0;

export function recordStandardFetch(count: number, error?: string | null): void {
  lastStandardCount = count;
  lastStandardFetchAt = new Date().toISOString();
  lastStandardFetchError = error ?? null;
}

export function recordGraphFetch(count: number, error?: string | null): void {
  lastGraphCount = count;
  lastGraphFetchAt = new Date().toISOString();
  lastGraphFetchError = error ?? null;
}

export function recordMerged(count: number): void {
  lastMergedCount = count;
  lastMergedAt = new Date().toISOString();
}

export interface MarketSourceDiagnostics {
  standardMarketsCount: number;
  graphMarketsCount: number;
  mergedMarketsCount: number;
  lastStandardFetchAt: string | null;
  lastGraphFetchAt: string | null;
  lastMergedAt: string | null;
  lastStandardFetchError: string | null;
  lastGraphFetchError: string | null;
  generatedAt: string;
}

export function getMarketSourceDiagnostics(): MarketSourceDiagnostics {
  return {
    standardMarketsCount: lastStandardCount,
    graphMarketsCount: lastGraphCount,
    mergedMarketsCount: lastMergedCount,
    lastStandardFetchAt: lastStandardFetchAt,
    lastGraphFetchAt: lastGraphFetchAt,
    lastMergedAt: lastMergedAt,
    lastStandardFetchError: lastStandardFetchError,
    lastGraphFetchError: lastGraphFetchError,
    generatedAt: new Date().toISOString(),
  };
}
