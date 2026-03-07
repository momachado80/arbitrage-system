export interface AnalyticsData {
  market_regime?: {
    regime_score: number;
    state: "high_edge" | "neutral" | "hostile";
    components?: {
      edge_persistence: number;
      opportunity_density: number;
      liquidity_score: number;
      competition: number;
    };
  };
  opportunity_ranking?: Array<{
    score: number;
    market_id: string;
    expected_pnl_per_hour: number;
    fill_rate: number;
    liquidity_score: number;
  }>;
  cross_market_trade_candidates?: Array<{
    basket_id: string;
    type: string;
    edge_bps: number;
    basket_expected_pnl: number;
    basket_fill_rate: number;
    basket_alpha_capture_ratio: number;
    n_legs: number;
  }>;
  shadow_trading_summary?: {
    shadow_trades: number;
    total_expected_pnl: number;
    mean_expected_pnl: number;
    mean_fill_rate: number;
    profitable_trades: number;
    profitable_pct: number;
    paper_trading?: {
      realizedPnL: number;
      currentEquity: number;
      activeTrades: number;
      closedTrades: number;
      winRate: number;
    };
  };
  paper_trading?: {
    realizedPnL: number;
    currentEquity: number;
    activeTrades: number;
    closedTrades: number;
    winRate: number;
  };
  shadow_profiles?: Array<{
    profileId: string;
    label: string;
    startingCapital: number;
    currentEquity: number;
    realizedPnL: number;
    activeTrades: number;
    closedTrades: number;
  }>;
  risk_engine_state?: {
    inventory: Record<string, number>;
    pnl: number;
    total_inventory: number;
    markets_with_position: number;
    drawdown_remaining: number;
  };
  edge_survival_curve?: Record<string, number>;
  edge_decay_curve?: Record<string, number>;
  edge_hazard_curve?: Record<string, number>;
  aggregates?: Record<string, unknown>;
  rejection_stats?: {
    countsByReason: Record<string, number>;
    totalRejected: number;
    timestamp: string;
  };
  error?: string;
}

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://web-production-800bb.up.railway.app";

export async function fetchAnalytics(): Promise<AnalyticsData> {
  const res = await fetch(`${API_URL}/analytics`, {
    cache: "no-store",
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
