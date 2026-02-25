"""Tests for OracleMetricsStore."""

import os
import tempfile
import pytest

from src.oracle.metrics import OracleMetricsStore


class TestOracleMetricsStore:
    def _make_store(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test_metrics.jsonl")
        return OracleMetricsStore(metrics_path=path)

    def test_empty_metrics(self):
        store = self._make_store()
        assert store.rolling_brier_score() is None
        assert store.edge_capture_ratio() is None
        assert store.profit_per_unit_edge() is None

    def test_brier_score_perfect(self):
        store = self._make_store()
        store.record_prediction("t1", p_oracle=0.9, p_market=0.5, confidence=0.8, source="test")
        store.record_outcome("t1", outcome=1.0)
        brier = store.rolling_brier_score()
        assert brier is not None
        # Brier = (0.9 - 1.0)^2 = 0.01
        assert abs(brier - 0.01) < 1e-6

    def test_brier_score_worst(self):
        store = self._make_store()
        store.record_prediction("t1", p_oracle=0.9, p_market=0.5, confidence=0.8, source="test")
        store.record_outcome("t1", outcome=0.0)
        brier = store.rolling_brier_score()
        assert brier is not None
        # Brier = (0.9 - 0.0)^2 = 0.81
        assert abs(brier - 0.81) < 1e-6

    def test_per_token_brier(self):
        store = self._make_store()
        store.record_prediction("t1", p_oracle=0.8, p_market=0.5, confidence=0.8, source="test")
        store.record_outcome("t1", outcome=1.0)
        brier = store.brier_score_for_token("t1")
        assert brier is not None
        assert brier < 0.1

    def test_calibration_bins(self):
        store = self._make_store()
        for i in range(10):
            p = 0.7
            store.record_prediction(f"t{i}", p_oracle=p, p_market=0.5, confidence=0.8, source="test")
            store.record_outcome(f"t{i}", outcome=1.0 if i < 7 else 0.0)
        cal = store.calibration_bins()
        assert "bins" in cal
        assert len(cal["bins"]) > 0

    def test_edge_capture_ratio(self):
        store = self._make_store()
        # Oracle predicts p=0.7 vs market=0.5, edge=0.2
        # Outcome = 1.0, captured = 1.0 - 0.5 = 0.5
        store.record_prediction("t1", p_oracle=0.7, p_market=0.5, confidence=0.8, source="test")
        store.record_outcome("t1", outcome=1.0)
        ratio = store.edge_capture_ratio()
        assert ratio is not None
        assert ratio > 0

    def test_profit_per_unit_edge(self):
        store = self._make_store()
        store.record_prediction("t1", p_oracle=0.7, p_market=0.5, confidence=0.8, source="test")
        store.record_outcome("t1", outcome=1.0, realized_pnl=0.10)
        ppu = store.profit_per_unit_edge()
        assert ppu is not None
        # edge = 0.2, pnl = 0.10, ratio = 0.5
        assert abs(ppu - 0.5) < 1e-6

    def test_edge_decay_distribution(self):
        store = self._make_store()
        store.record_decay("t1", half_life_seconds=30.0, initial_edge=0.05)
        store.record_decay("t2", half_life_seconds=60.0, initial_edge=0.03)
        dist = store.edge_decay_distribution()
        assert dist["sample_count"] == 2
        assert dist["avg_seconds"] == 45.0

    def test_edge_signal_recording(self):
        store = self._make_store()
        store.record_edge_signal("t1", net_edge=0.05, side="BUY", p_oracle=0.60, p_market=0.50, execution_price=0.52)
        signals = store.get_signals_last_24h()
        assert len(signals) == 1
        assert signals[0]["side"] == "BUY"

    def test_full_report(self):
        store = self._make_store()
        store.record_prediction("t1", p_oracle=0.8, p_market=0.5, confidence=0.9, source="test")
        store.record_outcome("t1", outcome=1.0)
        report = store.get_full_report()
        assert report["rolling_brier_score"] is not None
        assert report["total_outcomes"] == 1
        assert report["brier_interpretation"] in ("excellent", "good", "moderate", "poor")

    def test_persistence_to_jsonl(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test_metrics.jsonl")
        store = OracleMetricsStore(metrics_path=path)
        store.record_prediction("t1", p_oracle=0.7, p_market=0.5, confidence=0.8, source="test")
        store.record_outcome("t1", outcome=1.0)
        # File should exist and have 2 lines
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # 1 prediction + 1 outcome

    def test_outcome_without_prediction_ignored(self):
        store = self._make_store()
        store.record_outcome("t_unknown", outcome=1.0)
        assert store.rolling_brier_score() is None
