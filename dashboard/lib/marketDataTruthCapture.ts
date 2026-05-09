/**
 * Market Data Truth — infraestrutura de captura temporal (snapshots com ISO timestamps)
 * persistidos sob PAPER_STATE_DIR. Sem estratégia nova; sem alterar veredictos de viabilidade.
 * Tick: Gamma raw (clobTokenIds, bestBid/Ask) + livro CLOB REST quando disponível; fallback outcome prices.
 */

import fs from "fs/promises";
import path from "path";
import { fetchNormalizedMarketById, type NormalizedMarket } from "./polymarketClient";
import {
  extractGammaBestBidAsk,
  fetchGammaMarketRawJson,
  fetchParsedClobBook,
  parseClobTokenIds,
} from "./clobMicrostructure";
import { getAllMarkets } from "./marketDataService";
import {
  buildExecutionTruthDigest,
  estimatedNetPerCycle,
  isMachineObserved,
  isRobotQuoteableGate,
  observedDepth,
  observedSpread,
} from "./executionTruthEngine";
import { resolvePaperStateDir } from "./paperStateDir";

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function stdev(nums: number[]): number {
  if (nums.length < 2) return 0;
  const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
  const v = nums.reduce((s, x) => s + (x - mean) ** 2, 0) / (nums.length - 1);
  return Math.sqrt(v);
}

function seriesSummary(nums: number[]): {
  count: number;
  min: number;
  max: number;
  mean: number;
  stdev: number;
  last: number;
} {
  if (nums.length === 0) {
    return { count: 0, min: 0, max: 0, mean: 0, stdev: 0, last: 0 };
  }
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
  return {
    count: nums.length,
    min: r6(min),
    max: r6(max),
    mean: r6(mean),
    stdev: r6(stdev(nums)),
    last: r6(nums[nums.length - 1]),
  };
}

function midFromMarket(m: NormalizedMarket): number {
  if (m.prices.length < 2) return r6(m.prices[0] ?? 0);
  return r6((m.prices[0] + m.prices[1]) / 2);
}

function bidAskFromMarket(m: NormalizedMarket): { bid: number; ask: number } {
  const p = [...m.prices].filter(x => Number.isFinite(x));
  if (p.length === 0) return { bid: 0, ask: 0 };
  const bid = Math.min(...p);
  const ask = Math.max(...p);
  return { bid: r6(bid), ask: r6(ask) };
}

export type SnapshotDataSource = "clob_rest" | "gamma_best_hint" | "gamma_outcome_only";

export interface MarketMicroSnapshot {
  isoTimestamp: string;
  /** Origem dos preços neste ponto da série. */
  dataSource: SnapshotDataSource;
  observedBid: number;
  observedAsk: number;
  spread: number;
  depth: number;
  mid: number;
  depthBidTop3?: number;
  depthAskTop3?: number;
  clobTokenIdUsed?: string | null;
  clobBidLevels?: number;
  clobAskLevels?: number;
  /** |mid - mid_prev| no mesmo mercado. */
  midStepDelta?: number;
}

export interface MarketTruthSeriesDisk {
  marketId: string;
  marketTitle: string;
  snapshots: MarketMicroSnapshot[];
}

export interface MarketDataTruthDiskStore {
  storeVersion: 1;
  updatedAt: string;
  maxSnapshotsPerMarket: number;
  markets: Record<string, MarketTruthSeriesDisk>;
}

export type MarketDataTruthVerdict =
  | "insufficient_temporal_market_data"
  | "temporal_sequences_building"
  | "temporal_truth_partially_usable"
  | "temporal_truth_ready_for_probe_consumption";

export interface SeriesSummaryBlock {
  count: number;
  min: number;
  max: number;
  mean: number;
  stdev: number;
  last: number;
}

export interface StrongestTemporalMarket {
  marketId: string;
  marketTitle: string;
  snapshotCount: number;
  midDriftMax: number;
  spreadStdev: number;
}

export interface MarketDataTruthMarketDigest {
  marketId: string;
  marketTitle: string;
  snapshotCount: number;
  firstTimestamp: string;
  lastTimestamp: string;
  spreadSeriesSummary: SeriesSummaryBlock;
  midSeriesSummary: SeriesSummaryBlock;
  driftEvidence: string;
  decayEvidence: string;
  dataQualityNotes: string;
}

export interface MarketDataTruthDigest {
  probeVersion: "market-data-truth-v1";
  readDisclaimer: string;
  collectorRunning: boolean;
  collectionIntervalMs: number;
  lastCollectionTimestamp: string | null;
  totalCollectionTicks: number;
  lastCollectorError: string | null;
  marketDataTruthVerdict: MarketDataTruthVerdict;
  marketsTracked: number;
  marketsWithTemporalSequence: number;
  marketsWithSpreadSeries: number;
  marketsWithMidDriftSeries: number;
  marketsWithDecayEvidence: number;
  marketsWithMicrostructureSource: number;
  marketsWithBidAskSeries: number;
  marketsWithDepthSeries: number;
  marketsWithAdverseMoveEvidence: number;
  sourceMixSummary: string;
  strongestTemporalMarkets: StrongestTemporalMarket[];
  marketDataTruthSummaryLine: string;
  markets: MarketDataTruthMarketDigest[];
  computedAt: string;
}

const STORE_FILENAME = "market-data-truth-store.json";
const MAX_SNAPSHOTS_PER_MARKET = 180;
const MAX_TRACKED_MARKETS = 10;
const QUOTEABLE_PICK = 6;
const REJECTED_PICK = 4;
const INGEST_GAP_MS = 120;

const COLLECTOR_GLOBAL_KEY = "__arbMarketDataTruthCollector_v1";

type CollectorRuntime = {
  intervalMs: number;
  timer: ReturnType<typeof setInterval> | null;
  running: boolean;
  lastCollectionTimestamp: string | null;
  totalCollectionTicks: number;
  lastError: string | null;
  bootLogged: boolean;
};

function getCollectorRuntime(): CollectorRuntime {
  const g = globalThis as unknown as Record<string, CollectorRuntime>;
  if (!g[COLLECTOR_GLOBAL_KEY]) {
    const raw = parseInt(process.env.MARKET_DATA_TRUTH_COLLECTOR_MS || "45000", 10);
    const intervalMs = Number.isFinite(raw) ? Math.min(120_000, Math.max(30_000, raw)) : 45_000;
    g[COLLECTOR_GLOBAL_KEY] = {
      intervalMs,
      timer: null,
      running: false,
      lastCollectionTimestamp: null,
      totalCollectionTicks: 0,
      lastError: null,
      bootLogged: false,
    };
  }
  return g[COLLECTOR_GLOBAL_KEY]!;
}

/** Estado do colector (para digest / health). */
export function getMarketDataTruthCollectorStatus(): {
  collectorRunning: boolean;
  collectionIntervalMs: number;
  lastCollectionTimestamp: string | null;
  totalCollectionTicks: number;
  lastCollectorError: string | null;
} {
  const r = getCollectorRuntime();
  return {
    collectorRunning: r.running && r.timer != null,
    collectionIntervalMs: r.intervalMs,
    lastCollectionTimestamp: r.lastCollectionTimestamp,
    totalCollectionTicks: r.totalCollectionTicks,
    lastCollectorError: r.lastError,
  };
}

/**
 * Inicia o loop de ingestão periódica (instrumentation). Guard globalThis — idempotente.
 * MARKET_DATA_TRUTH_COLLECTOR_DISABLE=1 desliga. MARKET_DATA_TRUTH_COLLECTOR_MS (30000–120000).
 */
export function ensureMarketDataTruthCollector(): void {
  if (process.env.MARKET_DATA_TRUTH_COLLECTOR_DISABLE === "1") return;
  const r = getCollectorRuntime();
  if (r.timer != null) return;

  r.running = true;
  const runTick = async () => {
    try {
      await ingestMarketDataTruthTick();
      r.lastCollectionTimestamp = new Date().toISOString();
      r.totalCollectionTicks++;
      r.lastError = null;
    } catch (e) {
      r.lastError = e instanceof Error ? e.message : String(e);
      console.error("[MarketDataTruth] collector tick failed", e);
    }
  };

  const firstDelayMs = Math.min(15_000, Math.max(4_000, Math.floor(r.intervalMs / 4)));
  setTimeout(() => {
    void runTick();
  }, firstDelayMs);

  r.timer = setInterval(() => {
    void runTick();
  }, r.intervalMs);

  if (!r.bootLogged) {
    console.log(
      `[BOOT] market_data_truth collector ensured first_tick_ms=${firstDelayMs} interval_ms=${r.intervalMs}`,
    );
    r.bootLogged = true;
  }
}

function storePath(): string {
  return path.join(resolvePaperStateDir(process.cwd()), STORE_FILENAME);
}

function shortenTitle(q: string, n = 128): string {
  return q.length > n ? `${q.slice(0, n - 1)}…` : q;
}

function pickTrackList(): { id: string; title: string }[] {
  const digest = buildExecutionTruthDigest();
  const all = getAllMarkets();
  const out: { id: string; title: string }[] = [];
  const seen = new Set<string>();

  for (const s of digest.strongestQuoteableMarkets.slice(0, QUOTEABLE_PICK)) {
    if (seen.has(s.marketId)) continue;
    seen.add(s.marketId);
    out.push({ id: s.marketId, title: s.marketTitle });
  }

  const rejected = all
    .filter(m => isMachineObserved(m) && !isRobotQuoteableGate(m))
    .sort((a, b) => estimatedNetPerCycle(b) - estimatedNetPerCycle(a));

  let rejectedAdded = 0;
  for (const m of rejected) {
    if (out.length >= MAX_TRACKED_MARKETS) break;
    if (seen.has(m.id)) continue;
    seen.add(m.id);
    out.push({ id: m.id, title: shortenTitle(m.question) });
    rejectedAdded++;
    if (rejectedAdded >= REJECTED_PICK) break;
  }

  return out.slice(0, MAX_TRACKED_MARKETS);
}

async function buildMicroSnapshot(
  marketId: string,
  m: NormalizedMarket,
  prevMid: number | null,
): Promise<MarketMicroSnapshot> {
  const raw = await fetchGammaMarketRawJson(marketId);
  const tokens = raw ? parseClobTokenIds(raw) : [];

  let dataSource: SnapshotDataSource = "gamma_outcome_only";
  let bid: number;
  let ask: number;
  let depth: number;
  let depthBidTop3: number | undefined;
  let depthAskTop3: number | undefined;
  let clobTokenIdUsed: string | null | undefined;
  let clobBidLevels: number | undefined;
  let clobAskLevels: number | undefined;

  const book = tokens[0] ? await fetchParsedClobBook(tokens[0]) : null;
  if (book) {
    dataSource = "clob_rest";
    bid = book.bestBid;
    ask = book.bestAsk;
    depth = book.depthBidTop3 + book.depthAskTop3;
    depthBidTop3 = book.depthBidTop3;
    depthAskTop3 = book.depthAskTop3;
    clobTokenIdUsed = tokens[0];
    clobBidLevels = book.bidLevels;
    clobAskLevels = book.askLevels;
  } else if (raw) {
    const gbb = extractGammaBestBidAsk(raw);
    if (gbb) {
      dataSource = "gamma_best_hint";
      bid = gbb.bid;
      ask = gbb.ask;
      depth = observedDepth(m);
    } else {
      const ba = bidAskFromMarket(m);
      bid = ba.bid;
      ask = ba.ask;
      depth = observedDepth(m);
    }
  } else {
    const ba = bidAskFromMarket(m);
    bid = ba.bid;
    ask = ba.ask;
    depth = observedDepth(m);
  }

  const spread = r6(Math.max(0, ask - bid));
  const mid = r6((bid + ask) / 2);
  const snap: MarketMicroSnapshot = {
    isoTimestamp: new Date().toISOString(),
    dataSource,
    observedBid: r6(bid),
    observedAsk: r6(ask),
    spread,
    depth: r6(depth),
    mid,
    depthBidTop3,
    depthAskTop3,
    clobTokenIdUsed: clobTokenIdUsed ?? null,
    clobBidLevels,
    clobAskLevels,
  };
  if (prevMid != null && Number.isFinite(prevMid)) {
    snap.midStepDelta = r6(Math.abs(mid - prevMid));
  }
  return snap;
}

export async function loadMarketDataTruthStore(): Promise<MarketDataTruthDiskStore> {
  const p = storePath();
  try {
    const raw = await fs.readFile(p, "utf8");
    const j = JSON.parse(raw) as MarketDataTruthDiskStore;
    if (j && j.storeVersion === 1 && j.markets && typeof j.markets === "object") {
      for (const k of Object.keys(j.markets)) {
        const row = j.markets[k];
        row.snapshots = row.snapshots.map(s => {
          const legacy = s as MarketMicroSnapshot & { dataSource?: SnapshotDataSource };
          return {
            ...legacy,
            dataSource: legacy.dataSource ?? "gamma_outcome_only",
          };
        });
      }
      return {
        ...j,
        maxSnapshotsPerMarket: Math.min(
          MAX_SNAPSHOTS_PER_MARKET,
          j.maxSnapshotsPerMarket || MAX_SNAPSHOTS_PER_MARKET,
        ),
        markets: j.markets,
      };
    }
  } catch {
    /* missing */
  }
  return {
    storeVersion: 1,
    updatedAt: new Date().toISOString(),
    maxSnapshotsPerMarket: MAX_SNAPSHOTS_PER_MARKET,
    markets: {},
  };
}

async function saveMarketDataTruthStore(store: MarketDataTruthDiskStore): Promise<void> {
  if (process.env.MARKET_DATA_TRUTH_DISABLE_DISK === "1") return;
  const dir = path.dirname(storePath());
  await fs.mkdir(dir, { recursive: true });
  store.updatedAt = new Date().toISOString();
  const tmp = `${storePath()}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(store), "utf8");
  await fs.rename(tmp, storePath());
}

function trimRing(series: MarketTruthSeriesDisk, cap: number): void {
  if (series.snapshots.length > cap) {
    series.snapshots = series.snapshots.slice(-cap);
  }
}

function maxConsecutiveDelta(vals: number[]): number {
  if (vals.length < 2) return 0;
  let mx = 0;
  for (let i = 1; i < vals.length; i++) {
    mx = Math.max(mx, Math.abs(vals[i] - vals[i - 1]));
  }
  return r6(mx);
}

function decayFromSpreadSeries(spreads: number[]): { hasDecay: boolean; note: string } {
  if (spreads.length < 8) {
    return { hasDecay: false, note: "insufficient_points_for_decay_window" };
  }
  const half = Math.floor(spreads.length / 2);
  const a = spreads.slice(0, half);
  const b = spreads.slice(half);
  const ma = a.reduce((s, x) => s + x, 0) / a.length;
  const mb = b.reduce((s, x) => s + x, 0) / b.length;
  const slope = (mb - ma) / Math.max(1, half);
  if (slope < -1e-5) {
    return { hasDecay: true, note: `spread_second_half_mean_lower slope_per_step=${r6(slope)}` };
  }
  if (stdev(spreads) > 0.00015) {
    return { hasDecay: true, note: `spread_volatility_high stdev=${r6(stdev(spreads))}` };
  }
  return { hasDecay: false, note: "no_clear_decay_pattern" };
}

function buildMarketDigest(id: string, series: MarketTruthSeriesDisk): MarketDataTruthMarketDigest {
  const snaps = series.snapshots;
  const spreads = snaps.map(s => s.spread);
  const mids = snaps.map(s => s.mid);
  const driftChain = maxConsecutiveDelta(mids);
  const spreadJump = maxConsecutiveDelta(spreads);
  const stepDeltas = snaps.map(s => s.midStepDelta).filter((x): x is number => x != null && Number.isFinite(x));
  const maxStepDelta = stepDeltas.length > 0 ? r6(Math.max(...stepDeltas)) : 0;
  const drift = r6(Math.max(driftChain, maxStepDelta));
  const decay = decayFromSpreadSeries(spreads);
  const decayEvidence = decay.hasDecay ? decay.note : `no_decay_detected|${decay.note}`;
  const srcs = Array.from(new Set(snaps.map(s => s.dataSource))).join(",");
  const clobN = snaps.filter(s => s.dataSource === "clob_rest").length;

  return {
    marketId: id,
    marketTitle: series.marketTitle,
    snapshotCount: snaps.length,
    firstTimestamp: snaps[0]?.isoTimestamp ?? "",
    lastTimestamp: snaps[snaps.length - 1]?.isoTimestamp ?? "",
    spreadSeriesSummary: seriesSummary(spreads),
    midSeriesSummary: seriesSummary(mids),
    driftEvidence: `max_abs_mid_step=${drift} max_abs_spread_step=${spreadJump} mid_step_delta_max=${maxStepDelta}`,
    decayEvidence,
    dataQualityNotes: `sources=${srcs} clob_points=${clobN} | gamma+CLOB public REST`,
  };
}

/** Expõe série bruta para probes futuros (leitura só). */
export function getMarketTruthSeriesFromStore(
  store: MarketDataTruthDiskStore,
  marketId: string,
): MarketTruthSeriesDisk | undefined {
  return store.markets[marketId];
}

export async function ingestMarketDataTruthTick(): Promise<MarketDataTruthDiskStore> {
  const store = await loadMarketDataTruthStore();
  const track = pickTrackList();
  const cap = store.maxSnapshotsPerMarket;

  for (let i = 0; i < track.length; i++) {
    const { id, title } = track[i];
    const m = await fetchNormalizedMarketById(id);
    if (!m) continue;
    if (!store.markets[id]) {
      store.markets[id] = { marketId: id, marketTitle: shortenTitle(m.question || title), snapshots: [] };
    }
    const row = store.markets[id];
    row.marketTitle = shortenTitle(m.question || row.marketTitle);
    const prev = row.snapshots[row.snapshots.length - 1]?.mid ?? null;
    const snap = await buildMicroSnapshot(id, m, prev);
    row.snapshots.push(snap);
    trimRing(row, cap);
    if (i < track.length - 1) await new Promise(res => setTimeout(res, INGEST_GAP_MS));
  }

  await saveMarketDataTruthStore(store);
  return store;
}

function computeDigestFromStore(store: MarketDataTruthDiskStore): Omit<
  MarketDataTruthDigest,
  "probeVersion" | "readDisclaimer" | "computedAt" | "collectorRunning" | "collectionIntervalMs" | "lastCollectionTimestamp" | "totalCollectionTicks" | "lastCollectorError"
> {
  const ids = Object.keys(store.markets);
  const marketsTracked = ids.length;

  let mixClob = 0;
  let mixGammaBest = 0;
  let mixGammaOut = 0;
  for (const id of ids) {
    for (const s of store.markets[id].snapshots) {
      if (s.dataSource === "clob_rest") mixClob++;
      else if (s.dataSource === "gamma_best_hint") mixGammaBest++;
      else mixGammaOut++;
    }
  }
  const sourceMixSummary = `clob_rest=${mixClob} gamma_best_hint=${mixGammaBest} gamma_outcome_only=${mixGammaOut}`;

  const digests = ids.map(id => buildMarketDigest(id, store.markets[id])).sort((a, b) => b.snapshotCount - a.snapshotCount);

  const marketsWithTemporalSequence = digests.filter(m => m.snapshotCount >= 3).length;
  const marketsWithSpreadSeries = digests.filter(m => m.spreadSeriesSummary.stdev > 1e-8).length;
  const marketsWithMidDriftSeries = digests.filter(m => {
    const v = parseFloat(m.driftEvidence.replace(/.*max_abs_mid_step=([0-9.eE+-]+).*/, "$1")) || 0;
    return v > 5e-6;
  }).length;
  const marketsWithDecayEvidence = ids.filter(id => {
    const spreads = store.markets[id].snapshots.map(s => s.spread);
    return decayFromSpreadSeries(spreads).hasDecay;
  }).length;

  const marketsWithMicrostructureSource = ids.filter(id =>
    store.markets[id].snapshots.some(s => s.dataSource === "clob_rest"),
  ).length;

  const marketsWithBidAskSeries = digests.filter(d => {
    const s = store.markets[d.marketId].snapshots;
    if (s.length < 4) return false;
    const bids = s.map(x => x.observedBid);
    const asks = s.map(x => x.observedAsk);
    return stdev(bids) > 1e-9 || stdev(asks) > 1e-9;
  }).length;

  const marketsWithDepthSeries = digests.filter(d => {
    const s = store.markets[d.marketId].snapshots;
    if (s.length < 4) return false;
    return stdev(s.map(x => x.depth)) > 1e-5;
  }).length;

  const marketsWithAdverseMoveEvidence = ids.filter(id => {
    const deltas = store.markets[id].snapshots
      .map(s => s.midStepDelta)
      .filter((x): x is number => x != null && x > 0);
    return deltas.length > 0 && Math.max(...deltas) > 5e-6;
  }).length;

  const strongestTemporalMarkets: StrongestTemporalMarket[] = digests
    .map(d => {
      const driftMatch = /max_abs_mid_step=([0-9.+-eE]+)/.exec(d.driftEvidence);
      const drift = driftMatch ? parseFloat(driftMatch[1]) : 0;
      return {
        marketId: d.marketId,
        marketTitle: d.marketTitle,
        snapshotCount: d.snapshotCount,
        midDriftMax: r6(drift),
        spreadStdev: d.spreadSeriesSummary.stdev,
      };
    })
    .sort((a, b) => b.snapshotCount - a.snapshotCount || b.midDriftMax - a.midDriftMax)
    .slice(0, 8);

  const totalSnaps = digests.reduce((s, m) => s + m.snapshotCount, 0);

  let marketDataTruthVerdict: MarketDataTruthVerdict;
  if (marketsTracked < 2 || totalSnaps < 12) {
    marketDataTruthVerdict = "insufficient_temporal_market_data";
  } else if (marketsWithTemporalSequence < 2 || totalSnaps < 40) {
    marketDataTruthVerdict = "temporal_sequences_building";
  } else if (
    marketsWithDecayEvidence >= 2 &&
    marketsWithTemporalSequence >= 3 &&
    digests.filter(d => d.snapshotCount >= 18).length >= 2
  ) {
    marketDataTruthVerdict = "temporal_truth_ready_for_probe_consumption";
  } else if (marketsWithSpreadSeries >= 1 || marketsWithMidDriftSeries >= 1) {
    marketDataTruthVerdict = "temporal_truth_partially_usable";
  } else {
    marketDataTruthVerdict = "temporal_sequences_building";
  }

  const marketDataTruthSummaryLine = `market_data_truth: verdict=${marketDataTruthVerdict} | tracked=${marketsTracked} temporal>=3snaps=${marketsWithTemporalSequence} spread_series=${marketsWithSpreadSeries} mid_drift=${marketsWithMidDriftSeries} decay=${marketsWithDecayEvidence} micro_src=${marketsWithMicrostructureSource} bidask_series=${marketsWithBidAskSeries} depth_series=${marketsWithDepthSeries} adverse=${marketsWithAdverseMoveEvidence} total_snapshots=${totalSnaps}`;

  return {
    marketDataTruthVerdict,
    marketsTracked,
    marketsWithTemporalSequence,
    marketsWithSpreadSeries,
    marketsWithMidDriftSeries,
    marketsWithDecayEvidence,
    marketsWithMicrostructureSource,
    marketsWithBidAskSeries,
    marketsWithDepthSeries,
    marketsWithAdverseMoveEvidence,
    sourceMixSummary,
    strongestTemporalMarkets,
    marketDataTruthSummaryLine,
    markets: digests,
  };
}

/** Digest só leitura (disco + estado do colector). Não faz ingest — o colector em background acumula. */
export async function buildMarketDataTruthDigest(): Promise<MarketDataTruthDigest> {
  const store = await loadMarketDataTruthStore();
  const core = computeDigestFromStore(store);
  const c = getMarketDataTruthCollectorStatus();
  return {
    probeVersion: "market-data-truth-v1",
    readDisclaimer:
      "Market data truth v1: sequências em disco; colector em background. Cada tick: Gamma raw (clobTokenIds, bestBid/Ask) + livro CLOB REST quando disponível; fallback outcome prices. Endpoint só lê o store. Infra apenas — sem alterar veredictos de viabilidade.",
    collectorRunning: c.collectorRunning,
    collectionIntervalMs: c.collectionIntervalMs,
    lastCollectionTimestamp: c.lastCollectionTimestamp,
    totalCollectionTicks: c.totalCollectionTicks,
    lastCollectorError: c.lastCollectorError,
    ...core,
    computedAt: new Date().toISOString(),
  };
}
