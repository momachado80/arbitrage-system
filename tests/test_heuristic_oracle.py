"""Tests for HeuristicMicroOracle."""

import time
import pytest

from src.oracle.heuristic_oracle import HeuristicMicroOracle, MicroConfig


class TestHeuristicMicroOracle:
    def test_supports_all_by_default(self):
        oracle = HeuristicMicroOracle()
        assert oracle.supports("any_token") is True

    def test_supports_restricted_tokens(self):
        oracle = HeuristicMicroOracle()
        oracle.set_supported_tokens(["t1", "t2"])
        assert oracle.supports("t1") is True
        assert oracle.supports("t3") is False

    def test_name(self):
        oracle = HeuristicMicroOracle()
        assert oracle.name == "heuristic_micro"

    def test_insufficient_data_low_confidence(self):
        oracle = HeuristicMicroOracle(MicroConfig(min_samples=5))
        est = oracle.estimate("t1", {"mid": 0.50, "spread": 0.02, "bid_size": 100, "ask_size": 100})
        assert est is not None
        assert est.confidence == 0.1
        assert est.metadata.get("reason") == "insufficient_data"

    def test_no_market_data_returns_none(self):
        oracle = HeuristicMicroOracle()
        assert oracle.estimate("t1", None) is None

    def test_zero_mid_returns_none(self):
        oracle = HeuristicMicroOracle()
        assert oracle.estimate("t1", {"mid": 0.0, "spread": 0.02, "bid_size": 100, "ask_size": 100}) is None

    def test_estimate_with_enough_data(self):
        oracle = HeuristicMicroOracle(MicroConfig(min_samples=3, lookback=10))
        md = {"mid": 0.50, "spread": 0.02, "bid_size": 100, "ask_size": 100, "volume": 500}
        # Feed enough data
        for _ in range(5):
            oracle.estimate("t1", md)
        est = oracle.estimate("t1", md)
        assert est is not None
        assert est.confidence > 0.1
        assert 0.001 <= est.probability <= 0.999

    def test_buy_pressure_raises_probability(self):
        """More bids than asks should push p_oracle above mid."""
        oracle = HeuristicMicroOracle(MicroConfig(
            alpha=0.10, beta=0.0, gamma=0.0, delta=0.0,
            min_samples=2, lookback=10,
        ))
        # Balanced
        balanced_md = {"mid": 0.50, "spread": 0.02, "bid_size": 100, "ask_size": 100, "volume": 500}
        for _ in range(3):
            oracle.estimate("t1", balanced_md)

        # Now heavy buy pressure
        buy_md = {"mid": 0.50, "spread": 0.02, "bid_size": 500, "ask_size": 50, "volume": 500}
        est = oracle.estimate("t1", buy_md)
        assert est is not None
        assert est.probability > 0.50  # Pushed up by imbalance

    def test_sell_pressure_lowers_probability(self):
        """More asks than bids should push p_oracle below mid."""
        oracle = HeuristicMicroOracle(MicroConfig(
            alpha=0.10, beta=0.0, gamma=0.0, delta=0.0,
            min_samples=2, lookback=10,
        ))
        balanced_md = {"mid": 0.50, "spread": 0.02, "bid_size": 100, "ask_size": 100, "volume": 500}
        for _ in range(3):
            oracle.estimate("t1", balanced_md)

        sell_md = {"mid": 0.50, "spread": 0.02, "bid_size": 50, "ask_size": 500, "volume": 500}
        est = oracle.estimate("t1", sell_md)
        assert est is not None
        assert est.probability < 0.50

    def test_max_adjustment_capped(self):
        oracle = HeuristicMicroOracle(MicroConfig(
            alpha=1.0, beta=0.0, gamma=0.0, delta=0.0,
            min_samples=2, lookback=10, max_adjustment=0.05,
        ))
        md = {"mid": 0.50, "spread": 0.02, "bid_size": 1000, "ask_size": 1, "volume": 500}
        for _ in range(3):
            oracle.estimate("t1", md)
        est = oracle.estimate("t1", md)
        assert est is not None
        # Should be capped at mid + max_adjustment = 0.55
        assert est.probability <= 0.55 + 1e-6

    def test_diagnostics(self):
        oracle = HeuristicMicroOracle()
        md = {"mid": 0.50, "spread": 0.02, "bid_size": 100, "ask_size": 100, "volume": 500}
        oracle.estimate("t1", md)
        diag = oracle.get_diagnostics("t1")
        assert diag["samples"] == 1
        assert diag["latest_mid"] == 0.50

    def test_diagnostics_empty(self):
        oracle = HeuristicMicroOracle()
        diag = oracle.get_diagnostics("t1")
        assert diag["samples"] == 0
