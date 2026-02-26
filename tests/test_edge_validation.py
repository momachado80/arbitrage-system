"""Tests for EdgeValidator research module."""

import os
import tempfile
import pytest

from src.research.edge_validation import EdgeValidator


class TestEdgeValidator:
    def _make_validator(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test_validation.jsonl")
        return EdgeValidator(log_path=path)

    def test_empty_report(self):
        v = self._make_validator()
        report = v.generate_report()
        assert report["total_opportunities"] == 0
        assert report["edge_quality"] == "no_data"

    def test_record_opportunity(self):
        v = self._make_validator()
        oid = v.record_opportunity(
            token_id="t1", side="BUY",
            p_oracle=0.70, p_market=0.50,
            best_bid=0.48, best_ask=0.52,
            execution_price=0.52, net_edge=0.155,
            raw_edge=0.20, oracle_confidence=0.9,
        )
        assert oid == 1
        report = v.generate_report()
        assert report["total_opportunities"] == 1
        assert report["pending"] == 1

    def test_record_outcome_buy(self):
        v = self._make_validator()
        v.record_opportunity(
            token_id="t1", side="BUY",
            p_oracle=0.70, p_market=0.50,
            best_bid=0.48, best_ask=0.52,
            execution_price=0.52, net_edge=0.155,
            raw_edge=0.20, oracle_confidence=0.9,
        )
        v.record_outcome("t1", outcome=1.0)
        report = v.generate_report()
        assert report["resolved"] == 1
        assert report["hit_rate"] == 1.0
        # PnL = 1.0 - 0.52 = 0.48
        assert abs(report["total_simulated_pnl"] - 0.48) < 1e-4

    def test_record_outcome_sell(self):
        v = self._make_validator()
        v.record_opportunity(
            token_id="t1", side="SELL",
            p_oracle=0.30, p_market=0.50,
            best_bid=0.48, best_ask=0.52,
            execution_price=0.48, net_edge=0.155,
            raw_edge=-0.20, oracle_confidence=0.9,
        )
        v.record_outcome("t1", outcome=0.0)
        report = v.generate_report()
        assert report["hit_rate"] == 1.0
        # PnL = 0.48 - 0.0 = 0.48
        assert abs(report["total_simulated_pnl"] - 0.48) < 1e-4

    def test_losing_trade(self):
        v = self._make_validator()
        v.record_opportunity(
            token_id="t1", side="BUY",
            p_oracle=0.70, p_market=0.50,
            best_bid=0.48, best_ask=0.52,
            execution_price=0.52, net_edge=0.155,
            raw_edge=0.20, oracle_confidence=0.9,
        )
        v.record_outcome("t1", outcome=0.0)
        report = v.generate_report()
        assert report["hit_rate"] == 0.0
        # PnL = 0.0 - 0.52 = -0.52
        assert report["total_simulated_pnl"] < 0

    def test_brier_in_report(self):
        v = self._make_validator()
        v.record_opportunity(
            token_id="t1", side="BUY",
            p_oracle=0.90, p_market=0.50,
            best_bid=0.48, best_ask=0.52,
            execution_price=0.52, net_edge=0.35,
            raw_edge=0.40, oracle_confidence=0.9,
        )
        v.record_outcome("t1", outcome=1.0)
        report = v.generate_report()
        # Brier = (0.90 - 1.0)^2 = 0.01
        assert abs(report["brier_score"] - 0.01) < 1e-4

    def test_edge_quality_strong(self):
        v = self._make_validator()
        for i in range(10):
            v.record_opportunity(
                token_id=f"t{i}", side="BUY",
                p_oracle=0.90, p_market=0.50,
                best_bid=0.48, best_ask=0.52,
                execution_price=0.52, net_edge=0.35,
                raw_edge=0.40, oracle_confidence=0.9,
            )
            v.record_outcome(f"t{i}", outcome=1.0 if i < 8 else 0.0)
        report = v.generate_report()
        assert report["edge_quality"] in ("strong_edge", "moderate_edge")

    def test_by_side_breakdown(self):
        v = self._make_validator()
        v.record_opportunity(
            token_id="t1", side="BUY",
            p_oracle=0.70, p_market=0.50,
            best_bid=0.48, best_ask=0.52,
            execution_price=0.52, net_edge=0.155,
            raw_edge=0.20, oracle_confidence=0.9,
        )
        v.record_opportunity(
            token_id="t2", side="SELL",
            p_oracle=0.30, p_market=0.50,
            best_bid=0.48, best_ask=0.52,
            execution_price=0.48, net_edge=0.155,
            raw_edge=-0.20, oracle_confidence=0.9,
        )
        report = v.generate_report()
        assert "BUY" in report["by_side"]
        assert "SELL" in report["by_side"]
        assert report["by_side"]["BUY"]["count"] == 1
        assert report["by_side"]["SELL"]["count"] == 1

    def test_recent_opportunities(self):
        v = self._make_validator()
        for i in range(5):
            v.record_opportunity(
                token_id=f"t{i}", side="BUY",
                p_oracle=0.70, p_market=0.50,
                best_bid=0.48, best_ask=0.52,
                execution_price=0.52, net_edge=0.155,
                raw_edge=0.20, oracle_confidence=0.9,
            )
        recent = v.get_recent_opportunities(limit=3)
        assert len(recent) == 3
        assert recent[0]["id"] == 3  # Most recent 3

    def test_persistence(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test_validation.jsonl")
        v = EdgeValidator(log_path=path)
        v.record_opportunity(
            token_id="t1", side="BUY",
            p_oracle=0.70, p_market=0.50,
            best_bid=0.48, best_ask=0.52,
            execution_price=0.52, net_edge=0.155,
            raw_edge=0.20, oracle_confidence=0.9,
        )
        v.record_outcome("t1", outcome=1.0)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # opportunity + outcome

    def test_decay_recording(self):
        v = self._make_validator()
        v.record_opportunity(
            token_id="t1", side="BUY",
            p_oracle=0.70, p_market=0.50,
            best_bid=0.48, best_ask=0.52,
            execution_price=0.52, net_edge=0.155,
            raw_edge=0.20, oracle_confidence=0.9,
        )
        v.record_decay("t1", decay_seconds=45.0)
        report = v.generate_report()
        assert report["average_decay_time"] == 45.0
