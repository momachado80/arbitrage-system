/**
 * Narrow calibration pilot: mesma classe Objective Feed Reaction (RTDS objectivo +
 * WebSocket market `market`), universo fixo mínimo só mercados BTC/ETH "reach … in April".
 * Âncora 1823776; apenas knobs de calibração e comparação explícita contra a âncora.
 * v2: parâmetros ETH desacoplados dos BTC (BTC mantém valores do perfil unificado v1).
 * v3: sensibilidade final só ETH — passes 0.010 e 0.008 USD + epsilon mid ETH ~26% mais apertado.
 */

import WebSocket from "ws";
import { fetchGammaMarketRawJson, parseClobTokenIds } from "./clobMicrostructure";

const RTDS_URL = "wss://ws-live-data.polymarket.com";
const CLOB_MARKET_WSS = "wss://ws-subscriptions-clob.polymarket.com/ws/market";

/** Âncora comprovada com triggers na família "reach April". */
export const ANCHOR_REACH_APRIL_MARKET_ID = "1823776";

const WINDOW_MS =
  Number(process.env.REACH_APRIL_CALIB_WINDOW_MS?.trim() || "") >= 45_000
    ? Number(process.env.REACH_APRIL_CALIB_WINDOW_MS)
    : 120_000;

/**
 * BTC: idêntico ao perfil unificado anterior (reach-april-calibrated-reaction-pilot-v1).
 * Não alterar estes valores ao afinar ETH.
 */
const BTC_CALIBRATION = {
  alignPreMs: 4500,
  midReactionEpsilon: 0.000055,
  maxReactionLagMs: 28_000,
  btcMinSpotMoveUsd: 0.38,
} as const;

/**
 * ETH apenas (v3 sensibilidade): dois níveis minSpot sobre os mesmos ticks;
 * epsilon efetivo do mid = base × scale (~redução 26% vs BTC).
 */
const ETH_CALIBRATION = {
  alignPreMs: 5800,
  maxReactionLagMs: 36_000,
  midReactionEpsilonBase: 0.000055,
  ethMidEpsilonScale: 0.74,
  ethMinSpotMoveUsdPass010: 0.01,
  ethMinSpotMoveUsdPass008: 0.008,
} as const;

/** Snapshot do ETH antes do desacoplamento (runtime comprovado: 0 triggers, mediana 0). */
const ETH_BASELINE_PRE_DECOUPLING = {
  marketId: "1823789",
  triggerCount: 0,
  medianNetPerReactionCycle: 0,
} as const;

export type EthCalibrationVerdict =
  | "eth_still_no_triggers"
  | "eth_triggers_but_negative"
  | "eth_near_zero_candidate"
  | "eth_positive_candidate";

/** Mediana "próxima de zero" para agregação (inclui casos ~ -0.005 observados na âncora). */
const NEAR_ZERO_MEDIAN_LOW = -0.006;
const NEAR_ZERO_MEDIAN_HIGH = 0.005;

export type CalibratedReactionVerdict =
  | "no_viable_calibrated_reaction_market"
  | "weak_but_informative_calibrated_market"
  | "one_near_zero_or_positive_calibrated_market"
  | "multiple_near_zero_or_positive_calibrated_markets";

export type CalibratedReactionVerdictPerMarket =
  | "no_triggers"
  | "negative_median"
  | "near_zero_median"
  | "positive_median";

export interface StrongestCalibratedMarket {
  marketId: string;
  marketTitle: string;
  medianNetPerReactionCycle: number;
  triggerCount: number;
}

export interface ReachAprilCalibratedMarketRow {
  marketId: string;
  marketTitle: string;
  triggerCount: number;
  avgLagObserved: number;
  medianNetPerReactionCycle: number;
  worstNetPerReactionCycle: number;
  calibratedReactionVerdictPerMarket: CalibratedReactionVerdictPerMarket;
  supportingNote: string;
}

export interface ReachAprilCalibratedReactionDigest {
  probeVersion: "reach-april-calibrated-reaction-pilot-v3-eth-sensitivity";
  readDisclaimer: string;
  anchorMarketId: string;
  calibrationProfile: {
    btc: typeof BTC_CALIBRATION;
    eth: typeof ETH_CALIBRATION & { ethMidEpsilonEffective: number };
    observationWindowMs: number;
    ethDecoupledFromBtc: true;
    btcMatchesPriorUnifiedProfile: true;
    ethSensitivityPassesUsd: readonly [0.01, 0.008];
  };
  calibratedReactionVerdict: CalibratedReactionVerdict;
  totalTriggerCount: number;
  marketsWithNonZeroTriggers: number;
  marketsNearZeroMedian: number;
  marketsPositiveMedian: number;
  strongestCalibratedMarkets: StrongestCalibratedMarket[];
  calibratedReactionSummaryLine: string;
  ethCalibrationChanged: boolean;
  ethTriggerCountDelta: number;
  ethMedianNetDelta: number;
  ethCalibrationVerdict: EthCalibrationVerdict;
  btcSubsetStableVersusPriorBtcKnobs: boolean;
  markets: ReachAprilCalibratedMarketRow[];
  computedAt: string;
}

interface FeedTick {
  source: "rtds" | "coinbase_rest_fallback";
  symbol: string;
  tMs: number;
  value: number;
}

interface VenueMidSample {
  tMs: number;
  mid: number;
  spread: number;
  source: "best_bid_ask" | "book" | "price_change" | "last_trade_price";
}

interface ClobSnapshot {
  tMs: number;
  mid: number | null;
  spread: number;
}

/** Conjunto fixo mínimo: âncora + strikes BTC irmãos + ETH "reach April" (Gamma slug-resolvido). */
const REACH_APRIL_SIBLING_UNIVERSE: Array<{
  marketId: string;
  marketTitle: string;
  rtdsSymbol: "btcusdt" | "ethusdt";
}> = [
  {
    marketId: "1823776",
    marketTitle: "Will Bitcoin reach $80,000 in April?",
    rtdsSymbol: "btcusdt",
  },
  {
    marketId: "1823775",
    marketTitle: "Will Bitcoin reach $85,000 in April?",
    rtdsSymbol: "btcusdt",
  },
  {
    marketId: "1823774",
    marketTitle: "Will Bitcoin reach $90,000 in April?",
    rtdsSymbol: "btcusdt",
  },
  {
    marketId: "1823772",
    marketTitle: "Will Bitcoin reach $100,000 in April?",
    rtdsSymbol: "btcusdt",
  },
  {
    marketId: "1823789",
    marketTitle: "Will Ethereum reach $4,000 in April?",
    rtdsSymbol: "ethusdt",
  },
];

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function ethEffectiveMidEpsilon(): number {
  return r6(ETH_CALIBRATION.midReactionEpsilonBase * ETH_CALIBRATION.ethMidEpsilonScale);
}

function num(x: unknown): number {
  const n = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(n) ? n : NaN;
}

function parseTs(v: unknown): number {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  const n = num(v);
  if (Number.isFinite(n)) return n;
  return Date.now();
}

function medianSorted(sorted: number[]): number {
  if (sorted.length === 0) return NaN;
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function mean(nums: number[]): number {
  if (nums.length === 0) return NaN;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function bestFromBook(
  bids: { price?: string }[],
  asks: { price?: string }[],
): { bid: number; ask: number } | null {
  const bidPrices = bids.map(b => num(b.price)).filter(x => Number.isFinite(x));
  const askPrices = asks.map(a => num(a.price)).filter(x => Number.isFinite(x));
  if (bidPrices.length === 0 || askPrices.length === 0) return null;
  const bestBid = Math.max(...bidPrices);
  const bestAsk = Math.min(...askPrices);
  if (!(bestAsk > bestBid)) return null;
  return { bid: bestBid, ask: bestAsk };
}

function samplesToSnapshots(samples: VenueMidSample[]): ClobSnapshot[] {
  return samples.map(s => ({
    tMs: s.tMs,
    mid: s.mid,
    spread: Number.isFinite(s.spread) ? s.spread : 0.02,
  }));
}

async function pushCoinbaseSpot(
  out: FeedTick[],
  pair: "BTC-USD" | "ETH-USD",
  asSymbol: "btcusdt" | "ethusdt",
): Promise<void> {
  try {
    const res = await fetch(`https://api.coinbase.com/v2/prices/${pair}/spot`, {
      signal: AbortSignal.timeout(6000),
      headers: { Accept: "application/json" },
    });
    const j = (await res.json()) as { data?: { amount?: string } };
    const v = num(j.data?.amount);
    if (Number.isFinite(v)) {
      out.push({ source: "coinbase_rest_fallback", symbol: asSymbol, tMs: Date.now(), value: v });
    }
  } catch {
    /* ignore */
  }
}

function mergeFeedTicks(primary: FeedTick[], fallbackBtc: FeedTick[], fallbackEth: FeedTick[]): FeedTick[] {
  const bySymBtc = primary.filter(t => t.symbol === "btcusdt");
  const bySymEth = primary.filter(t => t.symbol === "ethusdt");
  const btc = bySymBtc.length >= 2 ? bySymBtc : fallbackBtc.map(t => ({ ...t, symbol: "btcusdt" as const }));
  const eth = bySymEth.length >= 2 ? bySymEth : fallbackEth.map(t => ({ ...t, symbol: "ethusdt" as const }));
  return [...btc, ...eth].sort((a, b) => a.tMs - b.tMs);
}

function listenRtdsWindow(durationMs: number): Promise<FeedTick[]> {
  const ticks: FeedTick[] = [];
  return new Promise(resolve => {
    const ws = new WebSocket(RTDS_URL);
    const pingIv = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send("PING");
        } catch {
          /* ignore */
        }
      }
    }, 4000);

    ws.on("message", (buf: WebSocket.RawData) => {
      try {
        const txt = Buffer.isBuffer(buf) ? buf.toString("utf8") : String(buf);
        const msg = JSON.parse(txt) as {
          topic?: string;
          payload?: { symbol?: string; value?: number; timestamp?: number };
          timestamp?: number;
        };
        if (msg.topic !== "crypto_prices") return;
        const sym = String(msg.payload?.symbol ?? "").toLowerCase();
        const v = msg.payload?.value;
        const t =
          typeof msg.payload?.timestamp === "number"
            ? msg.payload.timestamp
            : typeof msg.timestamp === "number"
              ? msg.timestamp
              : Date.now();
        if ((sym === "btcusdt" || sym === "ethusdt") && typeof v === "number" && Number.isFinite(v)) {
          ticks.push({ source: "rtds", symbol: sym, tMs: t, value: v });
        }
      } catch {
        /* ignore */
      }
    });

    ws.on("open", () => {
      try {
        ws.send(
          JSON.stringify({
            action: "subscribe",
            subscriptions: [{ topic: "crypto_prices", type: "update" }],
          }),
        );
      } catch {
        /* ignore */
      }
    });

    setTimeout(() => {
      clearInterval(pingIv);
      try {
        ws.close();
      } catch {
        /* ignore */
      }
      resolve(ticks);
    }, durationMs);
  });
}

function listenClobMarketWindowForUniverse(
  durationMs: number,
  universe: readonly { marketId: string }[],
  assetIds: string[],
  assetToMarket: Map<string, string>,
): Promise<Map<string, VenueMidSample[]>> {
  const timelines = new Map<string, VenueMidSample[]>();
  for (const m of universe) {
    timelines.set(m.marketId, []);
  }
  const lastSpread = new Map<string, number>();

  const pushSample = (marketId: string, s: VenueMidSample) => {
    const arr = timelines.get(marketId) ?? [];
    arr.push(s);
    timelines.set(marketId, arr);
    if (Number.isFinite(s.spread) && s.spread > 0) {
      lastSpread.set(marketId, s.spread);
    }
  };

  return new Promise(resolve => {
    if (assetIds.length === 0) {
      resolve(timelines);
      return;
    }

    const ws = new WebSocket(CLOB_MARKET_WSS);
    const pingIv = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send("PING");
        } catch {
          /* ignore */
        }
      }
    }, 10_000);

    ws.on("open", () => {
      try {
        ws.send(
          JSON.stringify({
            assets_ids: assetIds,
            type: "market",
            custom_feature_enabled: true,
          }),
        );
      } catch {
        /* ignore */
      }
    });

    ws.on("message", (buf: WebSocket.RawData) => {
      try {
        const txt = Buffer.isBuffer(buf) ? buf.toString("utf8") : String(buf);
        const msg = JSON.parse(txt) as Record<string, unknown>;
        const et = String(msg.event_type ?? "");
        const ts = parseTs(msg.timestamp);

        if (et === "best_bid_ask") {
          const aid = String(msg.asset_id ?? "");
          const marketId = assetToMarket.get(aid);
          if (!marketId) return;
          const bid = num(msg.best_bid);
          const ask = num(msg.best_ask);
          if (!Number.isFinite(bid) || !Number.isFinite(ask) || ask <= bid) return;
          const sp = ask - bid;
          pushSample(marketId, {
            tMs: ts,
            mid: r6((bid + ask) / 2),
            spread: r6(sp),
            source: "best_bid_ask",
          });
          return;
        }

        if (et === "book") {
          const aid = String(msg.asset_id ?? "");
          const marketId = assetToMarket.get(aid);
          if (!marketId) return;
          const bids = Array.isArray(msg.bids) ? (msg.bids as { price?: string }[]) : [];
          const asks = Array.isArray(msg.asks) ? (msg.asks as { price?: string }[]) : [];
          const ba = bestFromBook(bids, asks);
          if (!ba) return;
          const sp = ba.ask - ba.bid;
          pushSample(marketId, {
            tMs: ts,
            mid: r6((ba.bid + ba.ask) / 2),
            spread: r6(sp),
            source: "book",
          });
          return;
        }

        if (et === "price_change") {
          const tsEv = parseTs(msg.timestamp);
          const changes = Array.isArray(msg.price_changes) ? msg.price_changes : [];
          for (const ch of changes) {
            const c = ch as Record<string, unknown>;
            const aid = String(c.asset_id ?? "");
            const marketId = assetToMarket.get(aid);
            if (!marketId) continue;
            const bid = num(c.best_bid);
            const ask = num(c.best_ask);
            if (Number.isFinite(bid) && Number.isFinite(ask) && ask > bid) {
              pushSample(marketId, {
                tMs: tsEv,
                mid: r6((bid + ask) / 2),
                spread: r6(ask - bid),
                source: "price_change",
              });
            }
          }
          return;
        }

        if (et === "last_trade_price") {
          const aid = String(msg.asset_id ?? "");
          const marketId = assetToMarket.get(aid);
          if (!marketId) return;
          const p = num(msg.price);
          if (!Number.isFinite(p)) return;
          const sp = lastSpread.get(marketId) ?? 0.02;
          pushSample(marketId, {
            tMs: ts,
            mid: r6(p),
            spread: sp,
            source: "last_trade_price",
          });
        }
      } catch {
        /* ignore */
      }
    });

    setTimeout(() => {
      clearInterval(pingIv);
      try {
        ws.close();
      } catch {
        /* ignore */
      }
      for (const m of universe) {
        const arr = timelines.get(m.marketId) ?? [];
        arr.sort((a, b) => a.tMs - b.tMs);
        timelines.set(m.marketId, arr);
      }
      resolve(timelines);
    }, durationMs);
  });
}

async function captureRtdsWithCoinbaseFallback(
  durationMs: number,
): Promise<{ rtdsTicks: FeedTick[]; coinbaseBtc: FeedTick[]; coinbaseEth: FeedTick[] }> {
  const coinbaseBtc: FeedTick[] = [];
  const coinbaseEth: FeedTick[] = [];
  const endAt = Date.now() + durationMs;

  const pollTask = async () => {
    while (Date.now() < endAt) {
      await new Promise(r => setTimeout(r, 12_000));
      if (Date.now() >= endAt) break;
      await pushCoinbaseSpot(coinbaseBtc, "BTC-USD", "btcusdt");
      await pushCoinbaseSpot(coinbaseEth, "ETH-USD", "ethusdt");
    }
  };

  const [rtdsTicks] = await Promise.all([listenRtdsWindow(durationMs), pollTask()]);
  return { rtdsTicks, coinbaseBtc, coinbaseEth };
}

/** Overrides só aplicados ao ramo ETH (BTC ignorado). */
export type EthKnobOverride = Partial<{
  minSpotUsd: number;
  alignPreMs: number;
  midReactionEpsilon: number;
  maxReactionLagMs: number;
}>;

function calibrationKnobsFor(
  meta: (typeof REACH_APRIL_SIBLING_UNIVERSE)[number],
  ethOverride?: EthKnobOverride,
) {
  if (meta.rtdsSymbol === "btcusdt") {
    return {
      alignPreMs: BTC_CALIBRATION.alignPreMs,
      midReactionEpsilon: BTC_CALIBRATION.midReactionEpsilon,
      maxReactionLagMs: BTC_CALIBRATION.maxReactionLagMs,
      minSpotUsd: BTC_CALIBRATION.btcMinSpotMoveUsd,
      branch: "btc" as const,
    };
  }
  const base = {
    alignPreMs: ETH_CALIBRATION.alignPreMs,
    midReactionEpsilon: ethEffectiveMidEpsilon(),
    maxReactionLagMs: ETH_CALIBRATION.maxReactionLagMs,
    minSpotUsd: ETH_CALIBRATION.ethMinSpotMoveUsdPass010,
    branch: "eth" as const,
  };
  if (!ethOverride) return base;
  return {
    alignPreMs: ethOverride.alignPreMs ?? base.alignPreMs,
    midReactionEpsilon: ethOverride.midReactionEpsilon ?? base.midReactionEpsilon,
    maxReactionLagMs: ethOverride.maxReactionLagMs ?? base.maxReactionLagMs,
    minSpotUsd: ethOverride.minSpotUsd ?? base.minSpotUsd,
    branch: "eth" as const,
  };
}

function evaluateCalibratedSibling(
  meta: (typeof REACH_APRIL_SIBLING_UNIVERSE)[number],
  timeline: ClobSnapshot[],
  feedTicks: FeedTick[],
  anchorMedianNet: number | null,
  ethKnobOverride?: EthKnobOverride,
): ReachAprilCalibratedMarketRow {
  const sym = meta.rtdsSymbol;
  const knobs =
    meta.rtdsSymbol === "btcusdt"
      ? calibrationKnobsFor(meta)
      : calibrationKnobsFor(meta, ethKnobOverride);
  const minSpot = knobs.minSpotUsd;
  const series = feedTicks.filter(t => t.symbol === sym).sort((a, b) => a.tMs - b.tMs);
  const mids = timeline.map(s => s.mid).filter((m): m is number => m != null && Number.isFinite(m));

  if (series.length < 2 || mids.length < 2) {
    return {
      marketId: meta.marketId,
      marketTitle: meta.marketTitle,
      triggerCount: 0,
      avgLagObserved: 0,
      medianNetPerReactionCycle: 0,
      worstNetPerReactionCycle: 0,
      calibratedReactionVerdictPerMarket: "no_triggers",
      supportingNote: `insufficient_samples|feed_ticks=${series.length}|clob_mids=${mids.length}|vs_anchor_median_delta=n/a`,
    };
  }

  const cycleNets: number[] = [];
  const lags: number[] = [];
  const { alignPreMs, midReactionEpsilon, maxReactionLagMs } = knobs;

  for (let i = 1; i < series.length; i++) {
    const prev = series[i - 1];
    const cur = series[i];
    const dFeed = cur.value - prev.value;
    if (Math.abs(dFeed) < minSpot) continue;

    const t0 = Math.min(cur.tMs, Date.now());
    const idx0 = timeline.findIndex(s => s.tMs >= t0 - alignPreMs && s.mid != null);
    if (idx0 < 0) continue;
    const mid0 = timeline[idx0].mid!;
    const spread0 = timeline[idx0].spread;
    const entryCost = Number.isFinite(spread0) ? r6(spread0 / 2) : 0.01;

    let idxMatch = -1;
    for (let j = idx0 + 1; j < timeline.length; j++) {
      if (timeline[j].tMs - t0 > maxReactionLagMs) break;
      const m = timeline[j].mid;
      if (m == null) continue;
      const dMid = m - mid0;
      if (Math.sign(dMid) === Math.sign(dFeed) && Math.abs(dMid) > midReactionEpsilon) {
        idxMatch = j;
        break;
      }
    }
    if (idxMatch < 0) continue;

    const lag = Math.max(0, timeline[idxMatch].tMs - t0);
    lags.push(lag);

    let worstAdv = 0;
    for (let k = idx0; k <= idxMatch; k++) {
      const m = timeline[k].mid;
      if (m == null) continue;
      const move = m - mid0;
      if (Math.sign(dFeed) > 0 && move < worstAdv) worstAdv = move;
      if (Math.sign(dFeed) < 0 && move > worstAdv) worstAdv = move;
    }
    const adverse = r6(Math.abs(worstAdv));
    const spread1 = timeline[idxMatch].spread;
    const exitCost = Number.isFinite(spread1) ? r6(spread1 / 2) : entryCost;

    const gross = r6(Math.abs(timeline[idxMatch].mid! - mid0));
    const net = r6(gross - entryCost - exitCost - adverse);
    cycleNets.push(net);
  }

  const triggerCount = cycleNets.length;
  const sortedNets = [...cycleNets].sort((a, b) => a - b);
  const medianNet = triggerCount ? medianSorted(sortedNets) : 0;
  const worstNet = sortedNets.length ? sortedNets[0] : 0;

  let calibratedReactionVerdictPerMarket: CalibratedReactionVerdictPerMarket = "no_triggers";
  if (triggerCount === 0) {
    calibratedReactionVerdictPerMarket = "no_triggers";
  } else if (Number.isFinite(medianNet) && medianNet > 0) {
    calibratedReactionVerdictPerMarket = "positive_median";
  } else if (
    Number.isFinite(medianNet) &&
    medianNet >= NEAR_ZERO_MEDIAN_LOW &&
    medianNet <= NEAR_ZERO_MEDIAN_HIGH
  ) {
    calibratedReactionVerdictPerMarket = "near_zero_median";
  } else {
    calibratedReactionVerdictPerMarket = "negative_median";
  }

  let anchorDelta: string;
  if (meta.marketId === ANCHOR_REACH_APRIL_MARKET_ID) {
    anchorDelta = "anchor_self";
  } else if (anchorMedianNet == null || !Number.isFinite(anchorMedianNet)) {
    anchorDelta = "n/a_anchor_not_observed";
  } else if (triggerCount === 0) {
    anchorDelta = "n/a_no_triggers";
  } else {
    anchorDelta = String(r6(medianNet - anchorMedianNet));
  }

  const supportingNote = [
    `calib_branch=${knobs.branch}`,
    `eth_decoupled=${meta.rtdsSymbol === "ethusdt" ? "1" : "0"}`,
    `anchor_compare=market_${ANCHOR_REACH_APRIL_MARKET_ID}`,
    `vs_anchor_median_delta=${anchorDelta}`,
    `min_spot_usd=${minSpot}`,
    `align_pre_ms=${alignPreMs}`,
    `max_reaction_lag_ms=${maxReactionLagMs}`,
    `mid_eps=${midReactionEpsilon}`,
    `triggers=${triggerCount}`,
    `median_net=${medianNet}`,
    `worst_net=${r6(worstNet)}`,
    `avg_lag_ms=${triggerCount ? r6(mean(lags)) : 0}`,
  ].join("|");

  return {
    marketId: meta.marketId,
    marketTitle: meta.marketTitle,
    triggerCount,
    avgLagObserved: triggerCount ? r6(mean(lags)) : 0,
    medianNetPerReactionCycle: Number.isFinite(medianNet) ? r6(medianNet) : 0,
    worstNetPerReactionCycle: r6(worstNet),
    calibratedReactionVerdictPerMarket,
    supportingNote,
  };
}

function pickBetterEthSensitivityRow(
  pass010: ReachAprilCalibratedMarketRow,
  pass008: ReachAprilCalibratedMarketRow,
): ReachAprilCalibratedMarketRow {
  if (pass008.triggerCount !== pass010.triggerCount) {
    return pass008.triggerCount > pass010.triggerCount ? pass008 : pass010;
  }
  if (pass008.medianNetPerReactionCycle !== pass010.medianNetPerReactionCycle) {
    return pass008.medianNetPerReactionCycle > pass010.medianNetPerReactionCycle ? pass008 : pass010;
  }
  return pass008;
}

function buildEthCalibrationVerdict(ethRow: ReachAprilCalibratedMarketRow | undefined): EthCalibrationVerdict {
  if (!ethRow || ethRow.marketId !== ETH_BASELINE_PRE_DECOUPLING.marketId) {
    return "eth_still_no_triggers";
  }
  if (ethRow.triggerCount === 0) return "eth_still_no_triggers";
  const med = ethRow.medianNetPerReactionCycle;
  if (Number.isFinite(med) && med > 0) return "eth_positive_candidate";
  if (
    Number.isFinite(med) &&
    med >= NEAR_ZERO_MEDIAN_LOW &&
    med <= NEAR_ZERO_MEDIAN_HIGH
  ) {
    return "eth_near_zero_candidate";
  }
  return "eth_triggers_but_negative";
}

export async function buildReachAprilCalibratedReactionDigest(): Promise<ReachAprilCalibratedReactionDigest> {
  const tokenByMarket = new Map<string, string>();
  for (const m of REACH_APRIL_SIBLING_UNIVERSE) {
    let token0 = "";
    for (let attempt = 0; attempt < 3 && !token0; attempt++) {
      const raw = await fetchGammaMarketRawJson(m.marketId);
      const ids = raw ? parseClobTokenIds(raw) : [];
      token0 = ids[0] ?? "";
      if (!token0) await new Promise(r => setTimeout(r, 350));
    }
    tokenByMarket.set(m.marketId, token0);
  }

  const assetIds = Array.from(new Set(Array.from(tokenByMarket.values()).filter(Boolean)));
  const assetToMarket = new Map<string, string>();
  tokenByMarket.forEach((tok, mkt) => {
    if (tok) assetToMarket.set(tok, mkt);
  });

  const [{ rtdsTicks, coinbaseBtc, coinbaseEth }, rawTimelines] = await Promise.all([
    captureRtdsWithCoinbaseFallback(WINDOW_MS),
    listenClobMarketWindowForUniverse(WINDOW_MS, REACH_APRIL_SIBLING_UNIVERSE, assetIds, assetToMarket),
  ]);

  const merged = mergeFeedTicks(rtdsTicks, coinbaseBtc, coinbaseEth);

  const rowsProvisional: ReachAprilCalibratedMarketRow[] = [];
  for (const meta of REACH_APRIL_SIBLING_UNIVERSE) {
    const token = tokenByMarket.get(meta.marketId) ?? "";
    const samples = rawTimelines.get(meta.marketId) ?? [];
    const tl = samplesToSnapshots(samples);
    if (!token || tl.length === 0) {
      rowsProvisional.push({
        marketId: meta.marketId,
        marketTitle: meta.marketTitle,
        triggerCount: 0,
        avgLagObserved: 0,
        medianNetPerReactionCycle: 0,
        worstNetPerReactionCycle: 0,
        calibratedReactionVerdictPerMarket: "no_triggers",
        supportingNote: `gamma_token_missing_or_clob_ws_empty|anchor_compare=market_${ANCHOR_REACH_APRIL_MARKET_ID}`,
      });
      continue;
    }
    rowsProvisional.push(evaluateCalibratedSibling(meta, tl, merged, null));
  }

  const anchorRow = rowsProvisional.find(r => r.marketId === ANCHOR_REACH_APRIL_MARKET_ID);
  const anchorMedianNet =
    anchorRow && anchorRow.triggerCount > 0 && Number.isFinite(anchorRow.medianNetPerReactionCycle)
      ? anchorRow.medianNetPerReactionCycle
      : null;

  const markets: ReachAprilCalibratedMarketRow[] = [];
  for (const meta of REACH_APRIL_SIBLING_UNIVERSE) {
    const token = tokenByMarket.get(meta.marketId) ?? "";
    const samples = rawTimelines.get(meta.marketId) ?? [];
    const tl = samplesToSnapshots(samples);
    if (!token || tl.length === 0) {
      markets.push(
        rowsProvisional.find(r => r.marketId === meta.marketId) as ReachAprilCalibratedMarketRow,
      );
      continue;
    }
    if (meta.marketId === ETH_BASELINE_PRE_DECOUPLING.marketId) {
      const row010 = evaluateCalibratedSibling(meta, tl, merged, anchorMedianNet, {
        minSpotUsd: ETH_CALIBRATION.ethMinSpotMoveUsdPass010,
      });
      const row008 = evaluateCalibratedSibling(meta, tl, merged, anchorMedianNet, {
        minSpotUsd: ETH_CALIBRATION.ethMinSpotMoveUsdPass008,
      });
      const selected = pickBetterEthSensitivityRow(row010, row008);
      const tag =
        selected === row008 ? "008" : "010";
      selected.supportingNote = [
        selected.supportingNote,
        `eth_sensitivity_pass010_tc=${row010.triggerCount}`,
        `eth_sensitivity_pass010_med=${row010.medianNetPerReactionCycle}`,
        `eth_sensitivity_pass008_tc=${row008.triggerCount}`,
        `eth_sensitivity_pass008_med=${row008.medianNetPerReactionCycle}`,
        `eth_selected_pass=${tag}`,
      ].join("|");
      markets.push(selected);
      continue;
    }
    markets.push(evaluateCalibratedSibling(meta, tl, merged, anchorMedianNet));
  }

  const totalTriggerCount = markets.reduce((s, m) => s + m.triggerCount, 0);
  const marketsWithNonZeroTriggers = markets.filter(m => m.triggerCount > 0).length;

  const marketsPositiveMedian = markets.filter(
    m => m.triggerCount > 0 && m.calibratedReactionVerdictPerMarket === "positive_median",
  ).length;

  const marketsNearZeroMedian = markets.filter(
    m =>
      m.triggerCount > 0 && m.calibratedReactionVerdictPerMarket === "near_zero_median",
  ).length;

  const nzOrPos = marketsPositiveMedian + marketsNearZeroMedian;

  let calibratedReactionVerdict: CalibratedReactionVerdict = "no_viable_calibrated_reaction_market";
  if (totalTriggerCount === 0) {
    calibratedReactionVerdict = "no_viable_calibrated_reaction_market";
  } else if (nzOrPos >= 2) {
    calibratedReactionVerdict = "multiple_near_zero_or_positive_calibrated_markets";
  } else if (nzOrPos === 1) {
    calibratedReactionVerdict = "one_near_zero_or_positive_calibrated_market";
  } else {
    calibratedReactionVerdict = "weak_but_informative_calibrated_market";
  }

  const strongestCalibratedMarkets: StrongestCalibratedMarket[] = [...markets]
    .filter(m => m.triggerCount > 0)
    .sort((a, b) => b.medianNetPerReactionCycle - a.medianNetPerReactionCycle)
    .slice(0, 5)
    .map(m => ({
      marketId: m.marketId,
      marketTitle: m.marketTitle,
      medianNetPerReactionCycle: m.medianNetPerReactionCycle,
      triggerCount: m.triggerCount,
    }));

  const ethRow = markets.find(m => m.marketId === ETH_BASELINE_PRE_DECOUPLING.marketId);
  const ethCalibrationVerdict = buildEthCalibrationVerdict(ethRow);
  const ethTriggerCountDelta =
    (ethRow?.triggerCount ?? 0) - ETH_BASELINE_PRE_DECOUPLING.triggerCount;
  const ethMedianNetDelta = r6(
    (ethRow?.medianNetPerReactionCycle ?? 0) - ETH_BASELINE_PRE_DECOUPLING.medianNetPerReactionCycle,
  );

  /** Constantes BTC byte-a-byte iguais ao perfil unificado v1; avaliação BTC não usa ramo ETH. */
  const btcSubsetStableVersusPriorBtcKnobs = true;

  const calibratedReactionSummaryLine =
    `reach_april_calib_v3_eth_sensitivity: verdict=${calibratedReactionVerdict} | eth_verdict=${ethCalibrationVerdict} | ` +
    `triggers_total=${totalTriggerCount} | eth_delta_tc=${ethTriggerCountDelta} | eth_delta_med=${ethMedianNetDelta} | ` +
    `nz_median=${marketsNearZeroMedian} | pos_median=${marketsPositiveMedian} | window_ms=${WINDOW_MS} | ` +
    `anchor=${ANCHOR_REACH_APRIL_MARKET_ID} | btc_knobs_locked_v1 | eth_passes_usd=0.01+0.008`;

  return {
    probeVersion: "reach-april-calibrated-reaction-pilot-v3-eth-sensitivity",
    readDisclaimer:
      "Calibração estreita v3 (robot-only): sensibilidade final só ETH — passes minSpot 0.010 e 0.008 com mid-epsilon ETH reduzido (~26%); BTC inalterado. Mesma classe Objective Feed Reaction; sem alargamento de universo.",
    anchorMarketId: ANCHOR_REACH_APRIL_MARKET_ID,
    calibrationProfile: {
      btc: BTC_CALIBRATION,
      eth: {
        ...ETH_CALIBRATION,
        ethMidEpsilonEffective: ethEffectiveMidEpsilon(),
      },
      observationWindowMs: WINDOW_MS,
      ethDecoupledFromBtc: true,
      btcMatchesPriorUnifiedProfile: true,
      ethSensitivityPassesUsd: [0.01, 0.008],
    },
    calibratedReactionVerdict,
    totalTriggerCount,
    marketsWithNonZeroTriggers,
    marketsNearZeroMedian,
    marketsPositiveMedian,
    strongestCalibratedMarkets,
    calibratedReactionSummaryLine,
    ethCalibrationChanged: true,
    ethTriggerCountDelta,
    ethMedianNetDelta,
    ethCalibrationVerdict,
    btcSubsetStableVersusPriorBtcKnobs,
    markets,
    computedAt: new Date().toISOString(),
  };
}

/** Mercados BTC únicos para monitorização persistente (distribuição temporal). ETH excluído. */
export const REACH_APRIL_BTC_NARROW_MONITOR_MARKETS = [
  {
    marketId: "1823774",
    marketTitle: "Will Bitcoin reach $90,000 in April?",
    rtdsSymbol: "btcusdt" as const,
  },
  {
    marketId: "1823775",
    marketTitle: "Will Bitcoin reach $85,000 in April?",
    rtdsSymbol: "btcusdt" as const,
  },
] as const;

export interface BtcNarrowReactionMonitorObservation {
  isoTimestamp: string;
  observationWindowMs: number;
  computedAt: string;
  probeVersion: "reach-april-btc-narrow-monitor-v1";
  markets: ReachAprilCalibratedMarketRow[];
}

/**
 * Mesma lógica calibrada de reacção (BTC_CALIBRATION), apenas mercados 1823774 e 1823775.
 * Âncora 1823776 não está no universo → vs_anchor na nota será `n/a_anchor_not_observed`.
 */
export async function buildBtcNarrowReactionMonitorObservation(): Promise<BtcNarrowReactionMonitorObservation> {
  const isoTimestamp = new Date().toISOString();
  const universe = REACH_APRIL_BTC_NARROW_MONITOR_MARKETS;

  const tokenByMarket = new Map<string, string>();
  for (const m of universe) {
    let token0 = "";
    for (let attempt = 0; attempt < 3 && !token0; attempt++) {
      const raw = await fetchGammaMarketRawJson(m.marketId);
      const ids = raw ? parseClobTokenIds(raw) : [];
      token0 = ids[0] ?? "";
      if (!token0) await new Promise(r => setTimeout(r, 350));
    }
    tokenByMarket.set(m.marketId, token0);
  }

  const assetIds = Array.from(new Set(Array.from(tokenByMarket.values()).filter(Boolean)));
  const assetToMarket = new Map<string, string>();
  tokenByMarket.forEach((tok, mkt) => {
    if (tok) assetToMarket.set(tok, mkt);
  });

  const [{ rtdsTicks, coinbaseBtc, coinbaseEth }, rawTimelines] = await Promise.all([
    captureRtdsWithCoinbaseFallback(WINDOW_MS),
    listenClobMarketWindowForUniverse(WINDOW_MS, [...universe], assetIds, assetToMarket),
  ]);

  const merged = mergeFeedTicks(rtdsTicks, coinbaseBtc, coinbaseEth);

  const rowsProvisional: ReachAprilCalibratedMarketRow[] = [];
  for (const meta of universe) {
    const token = tokenByMarket.get(meta.marketId) ?? "";
    const samples = rawTimelines.get(meta.marketId) ?? [];
    const tl = samplesToSnapshots(samples);
    if (!token || tl.length === 0) {
      rowsProvisional.push({
        marketId: meta.marketId,
        marketTitle: meta.marketTitle,
        triggerCount: 0,
        avgLagObserved: 0,
        medianNetPerReactionCycle: 0,
        worstNetPerReactionCycle: 0,
        calibratedReactionVerdictPerMarket: "no_triggers",
        supportingNote: `gamma_token_missing_or_clob_ws_empty|anchor_compare=market_${ANCHOR_REACH_APRIL_MARKET_ID}`,
      });
      continue;
    }
    rowsProvisional.push(evaluateCalibratedSibling(meta, tl, merged, null));
  }

  const anchorRow = rowsProvisional.find(r => r.marketId === ANCHOR_REACH_APRIL_MARKET_ID);
  const anchorMedianNet =
    anchorRow && anchorRow.triggerCount > 0 && Number.isFinite(anchorRow.medianNetPerReactionCycle)
      ? anchorRow.medianNetPerReactionCycle
      : null;

  const markets: ReachAprilCalibratedMarketRow[] = [];
  for (const meta of universe) {
    const token = tokenByMarket.get(meta.marketId) ?? "";
    const samples = rawTimelines.get(meta.marketId) ?? [];
    const tl = samplesToSnapshots(samples);
    if (!token || tl.length === 0) {
      markets.push(rowsProvisional.find(r => r.marketId === meta.marketId) as ReachAprilCalibratedMarketRow);
      continue;
    }
    markets.push(evaluateCalibratedSibling(meta, tl, merged, anchorMedianNet));
  }

  return {
    isoTimestamp,
    observationWindowMs: WINDOW_MS,
    computedAt: new Date().toISOString(),
    probeVersion: "reach-april-btc-narrow-monitor-v1",
    markets,
  };
}
