import type { AnalyticsData } from "./api";

interface SystemResponse {
  marketQuality: number;
  marketsTracked: number;
  opportunitiesDetected: number;
  avgConfidence: number;
  avgEdge: number;
  overroundCount: number;
  underroundCount: number;
  crossMarketCount: number;
  systemStatus: string;
  lastUpdate: string | null;
  lastError: string | null;
  fetchCount: number;
  timestamp: string;
}

interface RawOpportunity {
  rank: number;
  compositeScore: number;
  marketId: string;
  question: string;
  slug: string;
  edge: number;
  type: "overround" | "underround" | "cross_market";
  probSum: number;
  outcomes: string[];
  prices: number[];
  liquidity: number;
  volume: number;
  confidence: number;
}

interface OpportunitiesResponse {
  count: number;
  opportunities: RawOpportunity[];
  timestamp: string;
}

interface GraphOpportunityRaw {
  id: string;
  type: string;
  title: string;
  description: string;
  edge: number;
  confidence: number;
  liquidity: number;
  compositeScore: number;
  marketsInvolved: Array<{ marketId: string; question: string }>;
}

interface GraphOpportunitiesResponse {
  summary: {
    graphOpportunitiesDetected: number;
    averageConfidence: number;
    averageEdge: number;
    numberOfClustersScanned: number;
  };
  opportunities: GraphOpportunityRaw[];
}

function qualityToState(q: number): "high_edge" | "neutral" | "hostile" {
  if (q >= 0.65) return "high_edge";
  if (q >= 0.35) return "neutral";
  return "hostile";
}

function mapSystemToRegime(sys: SystemResponse): AnalyticsData["market_regime"] {
  const density = Math.min(1, sys.opportunitiesDetected / 20);
  return {
    regime_score: sys.marketQuality,
    state: qualityToState(sys.marketQuality),
    components: {
      edge_persistence: Math.min(1, sys.avgEdge * 10),
      opportunity_density: density,
      liquidity_score: sys.avgConfidence,
      competition: Math.max(0, 1 - sys.avgConfidence),
    },
  };
}

function mapToOpportunityRanking(
  opps: RawOpportunity[]
): NonNullable<AnalyticsData["opportunity_ranking"]> {
  return opps
    .filter((o) => o.type !== "cross_market")
    .slice(0, 10)
    .map((o) => ({
      score: o.compositeScore * 100,
      market_id: o.question.length > 40 ? o.question.slice(0, 40) + "…" : o.question,
      expected_pnl_per_hour: o.edge * 10000,
      fill_rate: o.confidence,
      liquidity_score: Math.min(1, Math.log10(Math.max(1, o.liquidity)) / 6),
    }));
}

function mapToCrossMarketCandidates(
  opps: RawOpportunity[]
): NonNullable<AnalyticsData["cross_market_trade_candidates"]> {
  return opps
    .filter((o) => o.type === "cross_market")
    .slice(0, 10)
    .map((o, i) => ({
      basket_id: o.marketId.slice(0, 12) || `basket-${i}`,
      type: "exclusivity",
      edge_bps: o.edge * 10000,
      basket_expected_pnl: o.compositeScore * 1000,
      basket_fill_rate: o.confidence,
      basket_alpha_capture_ratio: o.confidence * 0.8,
      n_legs: o.outcomes.length,
    }));
}

function mapGraphToCrossMarketCandidates(
  graphOpps: GraphOpportunityRaw[]
): NonNullable<AnalyticsData["cross_market_trade_candidates"]> {
  return graphOpps.slice(0, 10).map((o) => ({
    basket_id: o.id.slice(0, 12),
    type: o.type.replace("graph_", ""),
    edge_bps: o.edge * 10000,
    basket_expected_pnl: o.compositeScore * 1000,
    basket_fill_rate: o.confidence,
    basket_alpha_capture_ratio: o.confidence * 0.7,
    n_legs: o.marketsInvolved.length,
  }));
}

async function fetchJson<T>(path: string): Promise<T> {
  const base = typeof window !== "undefined" ? "" : "http://localhost:3000";
  const res = await fetch(`${base}${path}`, {
    cache: "no-store",
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

interface GraphEpisodeSummaryResponse {
  activeEpisodes: number;
  closedEpisodes: number;
  avgActiveDurationMs: number;
  longestRecentEpisodeMs: number;
  totalEpisodesTracked: number;
}

interface GraphEpisodesResponse {
  summary: GraphEpisodeSummaryResponse;
}

const EMPTY_EPISODES: GraphEpisodesResponse = {
  summary: {
    activeEpisodes: 0, closedEpisodes: 0,
    avgActiveDurationMs: 0, longestRecentEpisodeMs: 0, totalEpisodesTracked: 0,
  },
};

interface PaperSystemResponse {
  status: string;
  lastUpdate: string | null;
  startingCapital: number;
  currentEquity: number;
  availableCapital: number;
  reservedCapital: number;
  activeTrades: number;
  closedTrades: number;
  realizedPnL: number;
  unrealizedPnL: number;
}

interface PaperAnalyticsResponse {
  analytics: { winRate: number };
  equityCurve: unknown[];
}

export async function fetchLiveAnalytics(): Promise<AnalyticsData> {
  const [systemData, oppData, graphData, episodeData, paperSystem, paperAnalytics, shadowProfilesRes] = await Promise.all([
    fetchJson<SystemResponse>("/api/system"),
    fetchJson<OpportunitiesResponse>("/api/opportunities"),
    fetchJson<GraphOpportunitiesResponse>("/api/graph-opportunities").catch(
      (): GraphOpportunitiesResponse => ({
        summary: { graphOpportunitiesDetected: 0, averageConfidence: 0, averageEdge: 0, numberOfClustersScanned: 0 },
        opportunities: [],
      })
    ),
    fetchJson<GraphEpisodesResponse>("/api/graph-episodes").catch(
      (): GraphEpisodesResponse => EMPTY_EPISODES
    ),
    fetchJson<PaperSystemResponse>("/api/paper/system").catch(() => null),
    fetchJson<PaperAnalyticsResponse>("/api/paper/analytics").catch(() => null),
    fetchJson<{ profiles: Array<{ profileId: string; label: string; startingCapital: number; currentEquity: number; realizedPnL: number; activeTrades: number; closedTrades: number }> }>("/api/shadow/profiles").catch(() => ({ profiles: [] })),
  ]);

  const scannerCross = mapToCrossMarketCandidates(oppData.opportunities);
  const graphCross = mapGraphToCrossMarketCandidates(graphData.opportunities);
  const mergedCross = [...graphCross, ...scannerCross].slice(0, 15);

  const allOppCount = oppData.count + graphData.summary.graphOpportunitiesDetected;
  const totalPnl = oppData.opportunities.reduce((s, o) => s + o.compositeScore * 100, 0);

  const epSummary = episodeData.summary;
  const persistenceFactor = epSummary.avgActiveDurationMs > 0
    ? Math.min(1, epSummary.avgActiveDurationMs / 60000)
    : 0;

  const regime = mapSystemToRegime(systemData);
  if (regime?.components && persistenceFactor > 0) {
    regime.components.edge_persistence = Math.min(
      1,
      regime.components.edge_persistence * 0.5 + persistenceFactor * 0.5
    );
  }

  const shadowSummary: AnalyticsData["shadow_trading_summary"] = {
    shadow_trades: allOppCount + epSummary.totalEpisodesTracked,
    total_expected_pnl: totalPnl,
    mean_expected_pnl: allOppCount > 0 ? totalPnl / allOppCount : 0,
    mean_fill_rate:
      oppData.count > 0
        ? oppData.opportunities.reduce((s, o) => s + o.confidence, 0) / oppData.count
        : 0,
    profitable_trades: oppData.opportunities.filter((o) => o.edge > 0.02).length +
      graphData.opportunities.filter((o) => o.edge > 0.02).length,
    profitable_pct:
      allOppCount > 0
        ? (oppData.opportunities.filter((o) => o.edge > 0.02).length +
            graphData.opportunities.filter((o) => o.edge > 0.02).length) /
          allOppCount
        : 0,
  };

  if (paperSystem && (paperSystem.closedTrades > 0 || paperSystem.activeTrades > 0)) {
    shadowSummary.paper_trading = {
      realizedPnL: paperSystem.realizedPnL,
      currentEquity: paperSystem.currentEquity,
      activeTrades: paperSystem.activeTrades,
      closedTrades: paperSystem.closedTrades,
      winRate: paperAnalytics?.analytics?.winRate ?? 0,
    };
  }

  const paperTrading =
    paperSystem && (paperSystem.activeTrades > 0 || paperSystem.closedTrades > 0)
      ? {
          realizedPnL: paperSystem.realizedPnL,
          currentEquity: paperSystem.currentEquity,
          activeTrades: paperSystem.activeTrades,
          closedTrades: paperSystem.closedTrades,
          winRate: paperAnalytics?.analytics?.winRate ?? 0,
        }
      : undefined;

  const shadowProfiles = (shadowProfilesRes?.profiles ?? []).map((p) => ({
    profileId: p.profileId,
    label: p.label,
    startingCapital: p.startingCapital ?? 0,
    currentEquity: p.currentEquity ?? 0,
    realizedPnL: p.realizedPnL ?? 0,
    activeTrades: p.activeTrades ?? 0,
    closedTrades: p.closedTrades ?? 0,
  }));

  return {
    market_regime: regime,
    opportunity_ranking: mapToOpportunityRanking(oppData.opportunities),
    cross_market_trade_candidates: mergedCross,
    shadow_trading_summary: shadowSummary,
    paper_trading: paperTrading,
    shadow_profiles: shadowProfiles,
    risk_engine_state: {
      inventory: {},
      pnl: totalPnl,
      total_inventory: systemData.marketsTracked,
      markets_with_position: epSummary.activeEpisodes,
      drawdown_remaining: 1000,
    },
  };
}
