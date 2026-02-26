"""Tests for ProbabilityOracle, providers, performance tracker, and edge decay."""

import time
import pytest

from src.oracle.probability_oracle import (
    CompositeOracleProvider,
    EdgeDecayTracker,
    MockOracleProvider,
    OracleEstimate,
    OraclePerformanceTracker,
    ProbabilityOracle,
)


# =========================================================================
# MockOracleProvider
# =========================================================================

class TestMockOracleProvider:
    def test_default_estimate(self):
        p = MockOracleProvider()
        est = p.estimate("any_token")
        assert est is not None
        assert est.probability == 0.5
        assert est.confidence == 0.5
        assert est.source == "mock"

    def test_custom_estimates(self):
        p = MockOracleProvider(estimates={"t1": (0.7, 0.9)})
        est = p.estimate("t1")
        assert est.probability == 0.7
        assert est.confidence == 0.9

    def test_fallback_to_default(self):
        p = MockOracleProvider(
            estimates={"t1": (0.7, 0.9)},
            default_prob=0.3,
            default_conf=0.4,
        )
        est = p.estimate("unknown_token")
        assert est.probability == 0.3
        assert est.confidence == 0.4

    def test_set_and_remove_estimate(self):
        p = MockOracleProvider()
        p.set_estimate("t1", 0.8, 0.9)
        assert p.estimate("t1").probability == 0.8
        p.remove_estimate("t1")
        assert p.estimate("t1").probability == 0.5

    def test_supports_all(self):
        p = MockOracleProvider()
        assert p.supports("anything") is True

    def test_name(self):
        assert MockOracleProvider().name == "mock"


# =========================================================================
# CompositeOracleProvider
# =========================================================================

class TestCompositeOracleProvider:
    def test_single_provider(self):
        mock = MockOracleProvider(estimates={"t1": (0.6, 0.8)})
        comp = CompositeOracleProvider([mock])
        est = comp.estimate("t1")
        assert est is not None
        assert est.probability == 0.6

    def test_weighted_average(self):
        p1 = MockOracleProvider(default_prob=0.4, default_conf=0.5)
        p2 = MockOracleProvider(default_prob=0.8, default_conf=0.5)
        comp = CompositeOracleProvider([p1, p2])
        est = comp.estimate("t1")
        assert est is not None
        assert 0.55 <= est.probability <= 0.65  # ~0.6 weighted avg

    def test_no_providers(self):
        comp = CompositeOracleProvider([])
        assert comp.estimate("t1") is None

    def test_supports(self):
        comp = CompositeOracleProvider([MockOracleProvider()])
        assert comp.supports("t1") is True
        comp2 = CompositeOracleProvider([])
        assert comp2.supports("t1") is False

    def test_name(self):
        assert CompositeOracleProvider().name == "composite"


# =========================================================================
# OraclePerformanceTracker
# =========================================================================

class TestOraclePerformanceTracker:
    def test_record_prediction(self):
        tracker = OraclePerformanceTracker()
        tracker.record_prediction("t1", 0.7, 0.5, 0.2, 0.8, "mock")
        summary = tracker.get_performance_summary()
        assert summary["pending"] == 1
        assert summary["resolved"] == 0

    def test_record_outcome_brier(self):
        tracker = OraclePerformanceTracker()
        tracker.record_prediction("t1", 0.8, 0.5, 0.3, 0.9, "mock")
        tracker.record_outcome("t1", 1.0)
        brier = tracker.get_brier_score()
        assert brier is not None
        assert brier == pytest.approx((0.8 - 1.0) ** 2, rel=1e-4)

    def test_perfect_brier(self):
        tracker = OraclePerformanceTracker()
        tracker.record_prediction("t1", 1.0, 0.5, 0.5, 1.0, "mock")
        tracker.record_outcome("t1", 1.0)
        assert tracker.get_brier_score() == pytest.approx(0.0)

    def test_worst_brier(self):
        tracker = OraclePerformanceTracker()
        tracker.record_prediction("t1", 0.0, 0.5, -0.5, 1.0, "mock")
        tracker.record_outcome("t1", 1.0)
        assert tracker.get_brier_score() == pytest.approx(1.0)

    def test_calibration_error(self):
        tracker = OraclePerformanceTracker()
        # All predicted at 0.7, half resolve YES
        for i in range(10):
            tid = f"t{i}"
            tracker.record_prediction(tid, 0.7, 0.5, 0.2, 0.8, "mock")
            tracker.record_outcome(tid, 1.0 if i < 7 else 0.0)

        cal = tracker.get_calibration_error()
        assert "buckets" in cal
        assert cal["mean_absolute_calibration_error"] is not None

    def test_edge_efficiency(self):
        tracker = OraclePerformanceTracker()
        tracker.record_prediction("t1", 0.7, 0.5, 0.2, 0.8, "mock")
        tracker.record_outcome("t1", 1.0)
        eff = tracker.get_edge_efficiency()
        assert eff["sample_count"] == 1
        assert eff["avg_predicted_edge"] is not None

    def test_no_data(self):
        tracker = OraclePerformanceTracker()
        assert tracker.get_brier_score() is None
        assert tracker.get_calibration_error()["buckets"] == []
        summary = tracker.get_performance_summary()
        assert summary["brier_interpretation"] == "insufficient_data"

    def test_token_specific_brier(self):
        tracker = OraclePerformanceTracker()
        tracker.record_prediction("t1", 0.9, 0.5, 0.4, 0.9, "mock")
        tracker.record_outcome("t1", 1.0)
        assert tracker.get_brier_score_for_token("t1") is not None
        assert tracker.get_brier_score_for_token("unknown") is None


# =========================================================================
# EdgeDecayTracker
# =========================================================================

class TestEdgeDecayTracker:
    def test_register_and_decay(self):
        tracker = EdgeDecayTracker()
        tracker.register_signal("t1", 0.10, 0.50, 0.60)

        # Edge not decayed yet
        tracker.update_market_price("t1", 0.52)
        metrics = tracker.get_metrics()
        assert metrics["active_tracking"] == 1
        assert metrics["sample_count"] == 0

        # Edge decayed to half
        tracker.update_market_price("t1", 0.55)  # edge = 0.60 - 0.55 = 0.05, half of 0.10
        metrics = tracker.get_metrics()
        assert metrics["sample_count"] == 1
        assert metrics["avg_half_life_seconds"] is not None

    def test_no_signal(self):
        tracker = EdgeDecayTracker()
        tracker.update_market_price("t1", 0.50)  # No signal registered
        metrics = tracker.get_metrics()
        assert metrics["sample_count"] == 0

    def test_metrics_empty(self):
        tracker = EdgeDecayTracker()
        m = tracker.get_metrics()
        assert m["avg_half_life_seconds"] is None
        assert m["capturable"] is None


# =========================================================================
# ProbabilityOracle (main orchestrator)
# =========================================================================

class TestProbabilityOracle:
    def test_basic_estimate(self):
        oracle = ProbabilityOracle(
            provider=MockOracleProvider(estimates={"t1": (0.7, 0.9)}),
        )
        est = oracle.get_estimate("t1", p_market=0.5)
        assert est is not None
        assert est.probability == 0.7
        assert est.confidence == 0.9

    def test_cache(self):
        oracle = ProbabilityOracle(cache_ttl=60.0)
        oracle.get_estimate("t1")
        oracle.get_estimate("t1")
        metrics = oracle.get_full_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["total_queries"] == 2

    def test_performance_tracking(self):
        oracle = ProbabilityOracle()
        oracle.get_estimate("t1", p_market=0.5)
        oracle.record_outcome("t1", 1.0)
        assert oracle.get_brier_score() is not None

    def test_adaptive_sizing_no_data(self):
        oracle = ProbabilityOracle()
        assert oracle.get_adaptive_sizing_multiplier() == 0.3

    def test_adaptive_sizing_good_brier(self):
        oracle = ProbabilityOracle()
        # Simulate many good predictions
        for i in range(30):
            tid = f"t{i}"
            oracle._performance.record_prediction(tid, 0.9, 0.5, 0.4, 0.9, "mock")
            oracle._performance.record_outcome(tid, 1.0)
        # Brier ~0.01, should give full sizing
        assert oracle.get_adaptive_sizing_multiplier() == 1.0

    def test_should_block_bad_oracle(self):
        oracle = ProbabilityOracle()
        # Simulate terrible predictions
        for i in range(25):
            tid = f"t{i}"
            oracle._performance.record_prediction(tid, 0.9, 0.5, 0.4, 0.9, "mock")
            oracle._performance.record_outcome(tid, 0.0)  # Always wrong
        assert oracle.should_block_oracle() is True

    def test_should_not_block_good_oracle(self):
        oracle = ProbabilityOracle()
        for i in range(25):
            tid = f"t{i}"
            oracle._performance.record_prediction(tid, 0.8, 0.5, 0.3, 0.9, "mock")
            oracle._performance.record_outcome(tid, 1.0)
        assert oracle.should_block_oracle() is False

    def test_full_metrics(self):
        oracle = ProbabilityOracle()
        oracle.get_estimate("t1", p_market=0.5)
        metrics = oracle.get_full_metrics()
        assert "total_queries" in metrics
        assert "performance" in metrics
        assert "edge_decay" in metrics
        assert "provider" in metrics

    def test_edge_decay_integration(self):
        oracle = ProbabilityOracle()
        oracle.register_edge_signal("t1", 0.10, 0.50, 0.60)
        oracle.update_decay_tracking("t1", 0.55)
        metrics = oracle.get_full_metrics()
        assert "edge_decay" in metrics

    def test_unsupported_token(self):
        """Provider que não suporta o token retorna None."""
        class SelectiveProvider(MockOracleProvider):
            def supports(self, token_id: str) -> bool:
                return token_id == "supported"
        oracle = ProbabilityOracle(provider=SelectiveProvider())
        assert oracle.get_estimate("unsupported") is None
        assert oracle.get_estimate("supported") is not None
