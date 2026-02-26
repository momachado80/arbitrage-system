"""Tests for the new edge-based StrategyEngine."""

import time
import pytest

from src.strategy.strategy_engine import (
    StrategyConfig,
    StrategyEngine,
    StrategySignal,
)
from src.oracle.probability_oracle import (
    MockOracleProvider,
    ProbabilityOracle,
)
from src.watcher.bridge import MarketSnapshot


def _make_snapshot(token_id="t1", bid=0.45, ask=0.55, ts=None):
    return MarketSnapshot(
        timestamp=ts or time.time(),
        token_id=token_id,
        best_bid_price=bid,
        best_bid_size=100,
        best_ask_price=ask,
        best_ask_size=100,
    )


class TestStrategyEngine:
    def test_no_oracle_returns_none(self):
        engine = StrategyEngine(oracle=None)
        result = engine.process_snapshot(_make_snapshot())
        assert result is None

    def test_no_signal_when_edge_small(self):
        """When oracle agrees with market, no signal."""
        mock = MockOracleProvider(default_prob=0.50, default_conf=0.8)
        oracle = ProbabilityOracle(provider=mock)
        engine = StrategyEngine(oracle=oracle)
        result = engine.process_snapshot(_make_snapshot(bid=0.49, ask=0.51))
        assert result is None

    def test_buy_signal_on_positive_edge(self):
        """Oracle thinks higher probability → BUY signal."""
        mock = MockOracleProvider(default_prob=0.80, default_conf=0.9)
        oracle = ProbabilityOracle(provider=mock)
        config = StrategyConfig(
            edge_buy_threshold=0.03,
            fee_estimate=0.01,
            slippage_estimate=0.001,
            spread_impact_factor=0.1,
        )
        engine = StrategyEngine(config=config, oracle=oracle)
        # mid = 0.50, oracle = 0.80, raw_edge = 0.30 >> threshold
        result = engine.process_snapshot(_make_snapshot(bid=0.49, ask=0.51))
        assert result is not None
        assert result.side == "BUY"
        assert result.edge_estimate > 0

    def test_sell_signal_on_negative_edge(self):
        """Oracle thinks lower probability → SELL signal."""
        mock = MockOracleProvider(default_prob=0.20, default_conf=0.9)
        oracle = ProbabilityOracle(provider=mock)
        config = StrategyConfig(
            edge_sell_threshold=0.03,
            fee_estimate=0.01,
            slippage_estimate=0.001,
            spread_impact_factor=0.1,
        )
        engine = StrategyEngine(config=config, oracle=oracle)
        # mid = 0.50, oracle = 0.20, raw_edge = -0.30 << -threshold
        result = engine.process_snapshot(_make_snapshot(bid=0.49, ask=0.51))
        assert result is not None
        assert result.side == "SELL"
        assert result.edge_estimate < 0

    def test_net_edge_includes_costs(self):
        """Net edge must be less than raw edge by costs."""
        mock = MockOracleProvider(default_prob=0.70, default_conf=0.9)
        oracle = ProbabilityOracle(provider=mock)
        config = StrategyConfig(
            fee_estimate=0.02,
            slippage_estimate=0.005,
            spread_impact_factor=0.5,
            edge_buy_threshold=0.001,  # Very low to trigger
        )
        engine = StrategyEngine(config=config, oracle=oracle)
        result = engine.process_snapshot(_make_snapshot(bid=0.49, ask=0.51))
        assert result is not None
        # net_edge = 0.20 - 0.02 - 0.005 - 0.01 = 0.165
        assert result.edge_estimate < 0.20  # Less than raw edge

    def test_safety_blocks_buy(self):
        """Buy should be blocked when safety_state.buy_blocked."""
        from src.safety.safety_state import SafetyState, SAFE_MODE
        ss = SafetyState()
        ss.block_buy(SAFE_MODE)

        mock = MockOracleProvider(default_prob=0.80, default_conf=0.9)
        oracle = ProbabilityOracle(provider=mock)
        engine = StrategyEngine(oracle=oracle, safety_state=ss)
        result = engine.process_snapshot(_make_snapshot(bid=0.49, ask=0.51))
        assert result is None  # Blocked

    def test_low_oracle_confidence_no_signal(self):
        """Low oracle confidence should not generate signal."""
        mock = MockOracleProvider(default_prob=0.80, default_conf=0.1)
        oracle = ProbabilityOracle(provider=mock)
        config = StrategyConfig(min_oracle_confidence=0.2)
        engine = StrategyEngine(config=config, oracle=oracle)
        result = engine.process_snapshot(_make_snapshot(bid=0.49, ask=0.51))
        assert result is None

    def test_oracle_blocked_no_signal(self):
        """Blocked oracle (bad Brier) should not generate signal."""
        mock = MockOracleProvider(default_prob=0.80, default_conf=0.9)
        oracle = ProbabilityOracle(provider=mock)
        # Simulate bad oracle
        for i in range(25):
            oracle._performance.record_prediction(f"t{i}", 0.9, 0.5, 0.4, 0.9, "mock")
            oracle._performance.record_outcome(f"t{i}", 0.0)
        assert oracle.should_block_oracle() is True
        engine = StrategyEngine(oracle=oracle)
        result = engine.process_snapshot(_make_snapshot(bid=0.49, ask=0.51))
        assert result is None

    def test_invalid_snapshot(self):
        engine = StrategyEngine()
        # No bid
        result = engine.process_snapshot(_make_snapshot(bid=0, ask=0.5))
        assert result is None
        # Ask <= bid
        result = engine.process_snapshot(_make_snapshot(bid=0.6, ask=0.5))
        assert result is None

    def test_get_market_data(self):
        mock = MockOracleProvider(default_prob=0.60, default_conf=0.7)
        oracle = ProbabilityOracle(provider=mock)
        engine = StrategyEngine(oracle=oracle)
        engine.process_snapshot(_make_snapshot())
        data = engine.get_market_data()
        assert len(data) == 1
        assert data[0]["token_id"] == "t1"
        assert "midpoint" in data[0]
        assert "z_score" in data[0]  # compat field
        assert "p_oracle" in data[0]
        assert "net_edge" in data[0]
        assert "ewma_mean" in data[0]  # compat

    def test_get_signal_history(self):
        mock = MockOracleProvider(default_prob=0.80, default_conf=0.9)
        oracle = ProbabilityOracle(provider=mock)
        config = StrategyConfig(
            edge_buy_threshold=0.01,
            fee_estimate=0.001,
            slippage_estimate=0.001,
            spread_impact_factor=0.01,
        )
        engine = StrategyEngine(config=config, oracle=oracle)
        engine.process_snapshot(_make_snapshot(bid=0.49, ask=0.51))
        signals = engine.get_signal_history()
        assert len(signals) >= 1
        assert "p_oracle" in signals[0]
        assert "p_market" in signals[0]
        assert "raw_edge" in signals[0]

    def test_get_strategy_summary(self):
        mock = MockOracleProvider(default_prob=0.50, default_conf=0.7)
        oracle = ProbabilityOracle(provider=mock)
        engine = StrategyEngine(oracle=oracle)
        engine.process_snapshot(_make_snapshot())
        summary = engine.get_strategy_summary()
        assert summary["strategy"] == "Probability Edge (Oracle-based)"
        assert "config" in summary
        assert "edge_metrics" in summary
        assert "oracle_metrics" in summary

    def test_get_z_history_compat(self):
        mock = MockOracleProvider(default_prob=0.60, default_conf=0.7)
        oracle = ProbabilityOracle(provider=mock)
        engine = StrategyEngine(oracle=oracle)
        engine.process_snapshot(_make_snapshot())
        hist = engine.get_z_history("t1")
        assert len(hist) == 1
        assert "z" in hist[0]  # compat field name

    def test_get_mid_history(self):
        engine = StrategyEngine()
        engine.process_snapshot(_make_snapshot())
        hist = engine.get_mid_history("t1")
        assert len(hist) == 1
        assert "mid" in hist[0]

    def test_metrics_collector_compat(self):
        """metrics_collector property returns proxy with get_metrics()."""
        engine = StrategyEngine()
        collector = engine.metrics_collector
        assert hasattr(collector, "get_metrics")
        m = collector.get_metrics()
        assert "current_z" in m
        assert "signal_count_last_24h" in m

    def test_set_oracle(self):
        engine = StrategyEngine()
        assert engine._oracle is None
        mock = MockOracleProvider()
        oracle = ProbabilityOracle(provider=mock)
        engine.set_oracle(oracle)
        assert engine._oracle is not None

    def test_spread_limits(self):
        """Spread outside limits should not generate signal."""
        mock = MockOracleProvider(default_prob=0.80, default_conf=0.9)
        oracle = ProbabilityOracle(provider=mock)
        config = StrategyConfig(min_spread=0.01, max_spread=0.05)
        engine = StrategyEngine(config=config, oracle=oracle)
        # Spread too wide
        result = engine.process_snapshot(_make_snapshot(bid=0.30, ask=0.70))
        assert result is None

    def test_adaptive_sizing_in_confidence(self):
        """Adaptive sizing from oracle affects combined confidence."""
        mock = MockOracleProvider(default_prob=0.80, default_conf=0.9)
        oracle = ProbabilityOracle(provider=mock)
        # No Brier data → adaptive = 0.3
        config = StrategyConfig(
            edge_buy_threshold=0.01,
            fee_estimate=0.001,
            slippage_estimate=0.001,
            spread_impact_factor=0.01,
        )
        engine = StrategyEngine(config=config, oracle=oracle)
        result = engine.process_snapshot(_make_snapshot(bid=0.49, ask=0.51))
        assert result is not None
        # confidence = oracle_conf * adaptive = 0.9 * 0.3 = 0.27
        assert result.confidence < 0.9  # Reduced by adaptive factor

    def test_signal_fields_compat_with_decision_pipeline(self):
        """Signal must have all fields DecisionPipeline expects."""
        mock = MockOracleProvider(default_prob=0.80, default_conf=0.9)
        oracle = ProbabilityOracle(provider=mock)
        config = StrategyConfig(
            edge_buy_threshold=0.01,
            fee_estimate=0.001,
            slippage_estimate=0.001,
            spread_impact_factor=0.01,
        )
        engine = StrategyEngine(config=config, oracle=oracle)
        signal = engine.process_snapshot(_make_snapshot(bid=0.49, ask=0.51))
        assert signal is not None
        # All fields DecisionPipeline accesses:
        assert hasattr(signal, "timestamp")
        assert hasattr(signal, "token_id")
        assert hasattr(signal, "side")
        assert hasattr(signal, "z_score")
        assert hasattr(signal, "midpoint")
        assert hasattr(signal, "spread")
        assert hasattr(signal, "edge_estimate")
        assert hasattr(signal, "confidence")
        assert hasattr(signal, "p_est")
        assert hasattr(signal, "best_bid")
        assert hasattr(signal, "best_ask")
        assert hasattr(signal, "rationale")
