"""Testes para Slippage / Impact Estimator."""

import pytest
from src.execution.slippage import estimate_fill_cost


class TestBasic:
    def test_buy_single_level(self):
        book = {"bids": [(0.64, 500)], "asks": [(0.66, 500)]}
        r = estimate_fill_cost(book, "BUY", 100)
        assert r["required_liquidity_ok"] is True
        assert r["expected_avg_price"] == 0.66
        assert r["worst_price"] == 0.66

    def test_sell_single_level(self):
        book = {"bids": [(0.64, 500)], "asks": [(0.66, 500)]}
        r = estimate_fill_cost(book, "SELL", 100)
        assert r["required_liquidity_ok"] is True
        assert r["expected_avg_price"] == 0.64
        assert r["worst_price"] == 0.64


class TestMultipleLevels:
    def test_buy_walk_through_asks(self):
        book = {
            "bids": [(0.64, 100)],
            "asks": [(0.65, 50), (0.66, 50), (0.68, 100)],
        }
        r = estimate_fill_cost(book, "BUY", 100)
        assert r["required_liquidity_ok"] is True
        expected = (50 * 0.65 + 50 * 0.66) / 100
        assert abs(r["expected_avg_price"] - expected) < 1e-6
        assert r["worst_price"] == 0.66

    def test_sell_walk_through_bids(self):
        book = {
            "bids": [(0.65, 30), (0.64, 30), (0.62, 100)],
            "asks": [(0.66, 100)],
        }
        r = estimate_fill_cost(book, "SELL", 60)
        assert r["required_liquidity_ok"] is True
        expected = (30 * 0.65 + 30 * 0.64) / 60
        assert abs(r["expected_avg_price"] - expected) < 1e-6
        assert r["worst_price"] == 0.64


class TestInsufficientLiquidity:
    def test_shallow_book(self):
        book = {"bids": [(0.64, 10)], "asks": [(0.66, 10)]}
        r = estimate_fill_cost(book, "BUY", 100)
        assert r["required_liquidity_ok"] is False

    def test_empty_book(self):
        r = estimate_fill_cost({"bids": [], "asks": []}, "BUY", 100)
        assert r["required_liquidity_ok"] is False

    def test_missing_side(self):
        r = estimate_fill_cost({"bids": [(0.64, 100)]}, "BUY", 100)
        assert r["required_liquidity_ok"] is False


class TestSlippageGrowsWithSize:
    def test_slippage_increases(self):
        book = {
            "bids": [(0.64, 100)],
            "asks": [(0.65, 50), (0.67, 50), (0.70, 100)],
        }
        r_small = estimate_fill_cost(book, "BUY", 30)
        r_large = estimate_fill_cost(book, "BUY", 100)
        assert r_small["required_liquidity_ok"] is True
        assert r_large["required_liquidity_ok"] is True
        assert r_large["slippage_vs_mid"] >= r_small["slippage_vs_mid"]
        assert r_large["worst_price"] >= r_small["worst_price"]


class TestEdgeCases:
    def test_zero_size(self):
        book = {"bids": [(0.64, 100)], "asks": [(0.66, 100)]}
        r = estimate_fill_cost(book, "BUY", 0)
        assert r["required_liquidity_ok"] is False

    def test_exact_fit(self):
        book = {"bids": [(0.64, 100)], "asks": [(0.66, 100)]}
        r = estimate_fill_cost(book, "BUY", 100)
        assert r["required_liquidity_ok"] is True
        assert r["expected_avg_price"] == 0.66


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
