/**
 * Objective Feed Reaction Bot Pilot — mesma classe estratégica: lag entre RTDS (feed objectivo)
 * e microestrutura CLOB em tempo real via WebSocket `market` (best_bid_ask, price_change,
 * last_trade_price, book). Sem trading live; universo fixo 5 mercados crypto-linked.
 */

import WebSocket from "ws";
import { fetchGammaMarketRawJson, parseClobTokenIds } from "./clobMicrostructure";

const RTDS_URL = "wss://ws-live-data.polymarket.com";
const CLOB_MARKET_WSS = "wss://ws-subscriptions-clob.polymarket.com/ws/market";

/** Janela de observação materialmente longa para capturar movimento spot + atualizações CLOB. */
const OBSERVATION_WINDOW_MS = Number(
  process.env.OBJECTIVE_FEED_REACTION_WINDOW_MS?.trim() || "",
) >= 45_000
  ? Number(process.env.OBJECTIVE_FEED_REACTION_WINDOW_MS)
  : 120_000;

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
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

export type ExternalFeedTypeKey =
  | "rtds_crypto_binance_btcusdt"
  | "rtds_crypto_binance_ethusdt"
  | "rest_coinbase_spot_fallback";

export type ReactionPilotVerdictPerMarket = "positive_expected" | "marginal" | "not_viable";

export type ObjectiveFeedReactionVerdict =
  | "no_viable_reaction_market"
  | "weak_reaction_candidate_only"
  | "one_viable_reaction_candidate"
  | "multiple_viable_reaction_candidates";

export interface ObjectiveFeedReactionMarketRow {
  marketId: string;
  marketTitle: string;
  externalFeedType: ExternalFeedTypeKey | string;
  triggerCount: number;
  avgLagObserved: number;
  avgEntryCost: number;
  avgExitCost: number;
  avgAdverseMoveAfterTrigger: number;
  estimatedNetPerReactionCycle: number;
  medianNetPerReactionCycle: number;
  worstNetPerReactionCycle: number;
  reactionPilotVerdictPerMarket: ReactionPilotVerdictPerMarket;
  supportingNote: string;
}

export interface StrongestReactionMarket {
  marketId: string;
  marketTitle: string;
  medianNetPerReactionCycle: number;
}

export type ObjectiveFeedReactionProbeVersion = "objective-feed-reaction-pilot-v2";

export interface ObjectiveFeedReactionPilotDigest {
  probeVersion: ObjectiveFeedReactionProbeVersion;
  readDisclaimer: string;
  venueDataSources: string[];
  observationWindowMs: number;
  objectiveFeedReactionVerdict: ObjectiveFeedReactionVerdict;
  marketsEvaluated: number;
  marketsWithValidFeedMapping: number;
  marketsWithPositiveMedianReactionNet: number;
  strongestReactionMarkets: StrongestReactionMarket[];
  objectiveFeedReactionSummaryLine: string;
  markets: ObjectiveFeedReactionMarketRow[];
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
  bestBid: number | null;
  bestAsk: number | null;
  spread: number;
  lastTradePrice: number | null;
  lastTradeSide: string;
}

const PILOT_MARKETS: Array<{
  marketId: string;
  marketTitle: string;
  feed: ExternalFeedTypeKey;
  rtdsSymbol: "btcusdt" | "ethusdt";
  minSpotMove: number;
}> = [
  {
    marketId: "1938161",
    marketTitle: "Will the price of Bitcoin be above $78,000 on April 17?",
    feed: "rtds_crypto_binance_btcusdt",
    rtdsSymbol: "btcusdt",
    minSpotMove: 0.5,
  },
  {
    marketId: "1938160",
    marketTitle: "Will the price of Bitcoin be above $76,000 on April 17?",
    feed: "rtds_crypto_binance_btcusdt",
    rtdsSymbol: "btcusdt",
    minSpotMove: 0.5,
  },
  {
    marketId: "1938162",
    marketTitle: "Will the price of Bitcoin be above $80,000 on April 17?",
    feed: "rtds_crypto_binance_btcusdt",
    rtdsSymbol: "btcusdt",
    minSpotMove: 0.5,
  },
  {
    marketId: "1823776",
    marketTitle: "Will Bitcoin reach $80,000 in April?",
    feed: "rtds_crypto_binance_btcusdt",
    rtdsSymbol: "btcusdt",
    minSpotMove: 0.5,
  },
  {
    marketId: "1823789",
    marketTitle: "Will Ethereum reach $4,000 in April?",
    feed: "rtds_crypto_binance_ethusdt",
    rtdsSymbol: "ethusdt",
    minSpotMove: 0.06,
  },
];

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
    bestBid: null,
    bestAsk: null,
    spread: Number.isFinite(s.spread) ? s.spread : 0.02,
    lastTradePrice: s.source === "last_trade_price" ? s.mid : null,
    lastTradeSide: "",
  }));
}

async function pushCoinbaseSpot(out: FeedTick[], pair: "BTC-USD" | "ETH-USD", asSymbol: "btcusdt" | "ethusdt"): Promise<void> {
  try {
    const res = await fetch(`https://api.coinbase.com/v2/prices/${pair}/spot`, {
      signal: AbortSignal.timeout(6000),
      headers: { Accept: "application/json" },
    });
    const j = (await res.json()) as { data?: { amount?: string } };
    const v = num(j.data?.amount);
    if (Number.isFinite(v)) {
      out.push({
        source: "coinbase_rest_fallback",
        symbol: asSymbol,
        tMs: Date.now(),
        value: v,
      });
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

function listenClobMarketWindow(
  durationMs: number,
  assetIds: string[],
  assetToMarket: Map<string, string>,
): Promise<Map<string, VenueMidSample[]>> {
  const timelines = new Map<string, VenueMidSample[]>();
  for (const m of PILOT_MARKETS) {
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
      for (const m of PILOT_MARKETS) {
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

function buildExternalType(
  usedRtdsBtc: boolean,
  usedRtdsEth: boolean,
  meta: (typeof PILOT_MARKETS)[number],
): ExternalFeedTypeKey | string {
  if (meta.rtdsSymbol === "btcusdt") {
    return usedRtdsBtc ? "rtds_crypto_binance_btcusdt" : "rest_coinbase_spot_fallback";
  }
  return usedRtdsEth ? "rtds_crypto_binance_ethusdt" : "rest_coinbase_spot_fallback";
}

function evaluateMarketAgainstFeed(
  meta: (typeof PILOT_MARKETS)[number],
  timeline: ClobSnapshot[],
  feedTicks: FeedTick[],
  externalType: ExternalFeedTypeKey | string,
): ObjectiveFeedReactionMarketRow {
  const sym = meta.rtdsSymbol;
  const series = feedTicks.filter(t => t.symbol === sym).sort((a, b) => a.tMs - b.tMs);
  const mids = timeline.map(s => s.mid).filter((m): m is number => m != null && Number.isFinite(m));

  if (series.length < 2 || mids.length < 2) {
    return {
      marketId: meta.marketId,
      marketTitle: meta.marketTitle,
      externalFeedType: externalType,
      triggerCount: 0,
      avgLagObserved: 0,
      avgEntryCost: 0,
      avgExitCost: 0,
      avgAdverseMoveAfterTrigger: 0,
      estimatedNetPerReactionCycle: 0,
      medianNetPerReactionCycle: 0,
      worstNetPerReactionCycle: 0,
      reactionPilotVerdictPerMarket: "not_viable",
      supportingNote: `insufficient_samples|feed_ticks=${series.length}|clob_ws_mids=${mids.length}`,
    };
  }

  const cycleNets: number[] = [];
  const lags: number[] = [];
  const entryCosts: number[] = [];
  const exitCosts: number[] = [];
  const adverses: number[] = [];

  for (let i = 1; i < series.length; i++) {
    const prev = series[i - 1];
    const cur = series[i];
    const dFeed = cur.value - prev.value;
    if (Math.abs(dFeed) < meta.minSpotMove) continue;

    const t0 = Math.min(cur.tMs, Date.now());
    const idx0 = timeline.findIndex(s => s.tMs >= t0 - 2_000 && s.mid != null);
    if (idx0 < 0) continue;
    const mid0 = timeline[idx0].mid!;
    const spread0 = timeline[idx0].spread;
    const entryCost = Number.isFinite(spread0) ? r6(spread0 / 2) : 0.01;

    let idxMatch = -1;
    for (let j = idx0 + 1; j < timeline.length; j++) {
      const m = timeline[j].mid;
      if (m == null) continue;
      const dMid = m - mid0;
      if (Math.sign(dMid) === Math.sign(dFeed) && Math.abs(dMid) > 0.00008) {
        idxMatch = j;
        break;
      }
    }
    if (idxMatch < 0) continue;

    const lag = Math.max(0, timeline[idxMatch].tMs - t0);
    lags.push(lag);
    entryCosts.push(entryCost);
    const spread1 = timeline[idxMatch].spread;
    const exitCost = Number.isFinite(spread1) ? r6(spread1 / 2) : entryCost;
    exitCosts.push(exitCost);

    let worstAdv = 0;
    for (let k = idx0; k <= idxMatch; k++) {
      const m = timeline[k].mid;
      if (m == null) continue;
      const move = m - mid0;
      if (Math.sign(dFeed) > 0 && move < worstAdv) worstAdv = move;
      if (Math.sign(dFeed) < 0 && move > worstAdv) worstAdv = move;
    }
    const adverse = r6(Math.abs(worstAdv));
    adverses.push(adverse);

    const gross = r6(Math.abs(timeline[idxMatch].mid! - mid0));
    const net = r6(gross - entryCost - exitCost - adverse);
    cycleNets.push(net);
  }

  const triggerCount = cycleNets.length;
  const sortedNets = [...cycleNets].sort((a, b) => a - b);
  const medianNet = triggerCount ? medianSorted(sortedNets) : 0;
  const meanNet = triggerCount ? mean(cycleNets) : 0;
  const worstNet = sortedNets.length ? sortedNets[0] : 0;

  let reactionPilotVerdictPerMarket: ReactionPilotVerdictPerMarket = "not_viable";
  if (triggerCount && Number.isFinite(medianNet) && medianNet > 0.0005) reactionPilotVerdictPerMarket = "positive_expected";
  else if (triggerCount && Number.isFinite(medianNet) && medianNet > 0) reactionPilotVerdictPerMarket = "marginal";

  const supportingParts = [
    `feed=${externalType}`,
    `clob_ws=market_channel`,
    `triggers=${triggerCount}`,
    `median_net_cycle=${medianNet}`,
    `avg_lag_ms=${triggerCount ? r6(mean(lags)) : 0}`,
    `venue=ws_best_bid_ask+price_change+last_trade+book`,
  ];

  return {
    marketId: meta.marketId,
    marketTitle: meta.marketTitle,
    externalFeedType: externalType,
    triggerCount,
    avgLagObserved: triggerCount ? r6(mean(lags)) : 0,
    avgEntryCost: triggerCount ? r6(mean(entryCosts)) : 0,
    avgExitCost: triggerCount ? r6(mean(exitCosts)) : 0,
    avgAdverseMoveAfterTrigger: triggerCount ? r6(mean(adverses)) : 0,
    estimatedNetPerReactionCycle: Number.isFinite(meanNet) ? r6(meanNet) : 0,
    medianNetPerReactionCycle: Number.isFinite(medianNet) ? r6(medianNet) : 0,
    worstNetPerReactionCycle: r6(worstNet),
    reactionPilotVerdictPerMarket,
    supportingNote: supportingParts.join("|"),
  };
}

export async function buildObjectiveFeedReactionPilotDigest(): Promise<ObjectiveFeedReactionPilotDigest> {
  const venueDataSources = [
    "rtds_wss_ws-live-data.polymarket.com_crypto_prices",
    "clob_wss_ws-subscriptions-clob.polymarket.com_ws_market",
    "clob_ws_event_best_bid_ask",
    "clob_ws_event_price_change",
    "clob_ws_event_last_trade_price",
    "clob_ws_event_book",
  ];

  const tokenByMarket = new Map<string, string>();
  for (const m of PILOT_MARKETS) {
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
    captureRtdsWithCoinbaseFallback(OBSERVATION_WINDOW_MS),
    listenClobMarketWindow(OBSERVATION_WINDOW_MS, assetIds, assetToMarket),
  ]);

  const usedRtdsBtc = rtdsTicks.some(t => t.symbol === "btcusdt" && t.source === "rtds");
  const usedRtdsEth = rtdsTicks.some(t => t.symbol === "ethusdt" && t.source === "rtds");
  const merged = mergeFeedTicks(rtdsTicks, coinbaseBtc, coinbaseEth);

  const markets: ObjectiveFeedReactionMarketRow[] = [];
  for (const meta of PILOT_MARKETS) {
    const token = tokenByMarket.get(meta.marketId) ?? "";
    const samples = rawTimelines.get(meta.marketId) ?? [];
    const tl = samplesToSnapshots(samples);
    if (!token || tl.length === 0) {
      markets.push({
        marketId: meta.marketId,
        marketTitle: meta.marketTitle,
        externalFeedType: buildExternalType(usedRtdsBtc, usedRtdsEth, meta),
        triggerCount: 0,
        avgLagObserved: 0,
        avgEntryCost: 0,
        avgExitCost: 0,
        avgAdverseMoveAfterTrigger: 0,
        estimatedNetPerReactionCycle: 0,
        medianNetPerReactionCycle: 0,
        worstNetPerReactionCycle: 0,
        reactionPilotVerdictPerMarket: "not_viable",
        supportingNote: "gamma_token_missing_or_clob_ws_empty",
      });
      continue;
    }
    const ext = buildExternalType(usedRtdsBtc, usedRtdsEth, meta);
    markets.push(evaluateMarketAgainstFeed(meta, tl, merged, ext));
  }

  const marketsEvaluated = markets.length;
  const marketsWithValidFeedMapping = markets.filter(
    m => m.supportingNote !== "gamma_token_missing_or_clob_ws_empty",
  ).length;
  const marketsWithPositiveMedianReactionNet = markets.filter(
    m => Number.isFinite(m.medianNetPerReactionCycle) && m.medianNetPerReactionCycle > 0,
  ).length;

  const strongestReactionMarkets: StrongestReactionMarket[] = [...markets]
    .sort((a, b) => b.medianNetPerReactionCycle - a.medianNetPerReactionCycle)
    .slice(0, 3)
    .map(m => ({
      marketId: m.marketId,
      marketTitle: m.marketTitle,
      medianNetPerReactionCycle: m.medianNetPerReactionCycle,
    }));

  const viableStrong = markets.filter(m => m.reactionPilotVerdictPerMarket === "positive_expected").length;
  const viableMarginal = markets.filter(m => m.reactionPilotVerdictPerMarket === "marginal").length;

  let objectiveFeedReactionVerdict: ObjectiveFeedReactionVerdict = "no_viable_reaction_market";
  if (viableStrong >= 2) objectiveFeedReactionVerdict = "multiple_viable_reaction_candidates";
  else if (viableStrong === 1) objectiveFeedReactionVerdict = "one_viable_reaction_candidate";
  else if (viableStrong === 0 && viableMarginal >= 1 && marketsWithPositiveMedianReactionNet >= 1) {
    objectiveFeedReactionVerdict = "weak_reaction_candidate_only";
  } else if (marketsWithPositiveMedianReactionNet >= 2 && viableStrong === 0) {
    objectiveFeedReactionVerdict = "weak_reaction_candidate_only";
  }

  const objectiveFeedReactionSummaryLine = `obj_feed_reaction_pilot_v2: verdict=${objectiveFeedReactionVerdict} | window_ms=${OBSERVATION_WINDOW_MS} | mkts=${marketsEvaluated} | feed_ok=${marketsWithValidFeedMapping} | pos_median_net=${marketsWithPositiveMedianReactionNet} | rtds_btc=${usedRtdsBtc} rtds_eth=${usedRtdsEth}`;

  return {
    probeVersion: "objective-feed-reaction-pilot-v2",
    readDisclaimer:
      "Pilot robot-only (v2): RTDS = feed objectivo (Binance via Polymarket); CLOB = WebSocket público `market` com best_bid_ask (custom_feature), price_change, last_trade_price, book. Janela longa configurável (OBJECTIVE_FEED_REACTION_WINDOW_MS). Coinbase REST apenas como fallback se RTDS BTC/ETH for escasso. Sem ordens live; sem scan alargado.",
    venueDataSources,
    observationWindowMs: OBSERVATION_WINDOW_MS,
    objectiveFeedReactionVerdict,
    marketsEvaluated,
    marketsWithValidFeedMapping,
    marketsWithPositiveMedianReactionNet,
    strongestReactionMarkets,
    objectiveFeedReactionSummaryLine,
    markets,
    computedAt: new Date().toISOString(),
  };
}
