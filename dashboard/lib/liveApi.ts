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

async function fetchJson<T>(path: string): Promise<T> {
  const base = typeof window !== "undefined" ? "" : "http://localhost:3000";
  const res = await fetch(`${base}${path}`, {
    cache: "no-store",
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

export async function fetchLiveAnalytics(): Promise<AnalyticsData> {
  const [systemData, oppData] = await Promise.all([
    fetchJson<SystemResponse>("/api/system"),
    fetchJson<OpportunitiesResponse>("/api/opportunities"),
  ]);

  return {
    market_regime: mapSystemToRegime(systemData),
    opportunity_ranking: mapToOpportunityRanking(oppData.opportunities),
    cross_market_trade_candidates: mapToCrossMarketCandidates(oppData.opportunities),
    shadow_trading_summary: {
      shadow_trades: oppData.count,
      total_expected_pnl: oppData.opportunities.reduce((s, o) => s + o.compositeScore * 100, 0),
      mean_expected_pnl:
        oppData.count > 0
          ? oppData.opportunities.reduce((s, o) => s + o.compositeScore * 100, 0) / oppData.count
          : 0,
      mean_fill_rate:
        oppData.count > 0
          ? oppData.opportunities.reduce((s, o) => s + o.confidence, 0) / oppData.count
          : 0,
      profitable_trades: oppData.opportunities.filter((o) => o.edge > 0.02).length,
      profitable_pct:
        oppData.count > 0
          ? oppData.opportunities.filter((o) => o.edge > 0.02).length / oppData.count
          : 0,
    },
    risk_engine_state: {
      inventory: {},
      pnl: oppData.opportunities.reduce((s, o) => s + o.compositeScore * 100, 0),
      total_inventory: systemData.marketsTracked,
      markets_with_position: systemData.opportunitiesDetected,
      drawdown_remaining: 1000,
    },
  };
}
