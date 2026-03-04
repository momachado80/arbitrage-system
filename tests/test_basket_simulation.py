"""Tests for basket simulation — determinism, PnL aggregation, scaling."""

import time
from src.strategy.arbitrage_basket_builder import ArbitrageBasketBuilder
from src.sim.basket_simulator import simulate_basket, simulate_baskets, rank_baskets


def _make_basket(probs=None, edge=None, capital=100.0):
    probs = probs or [0.40, 0.35, 0.30]
    if edge is None:
        edge = round(sum(probs) - 1.0, 6)
    opp = {
        "type": "exclusivity",
        "edge": edge,
        "markets": [
            {"market_id": f"m{i}", "prob": p, "best_bid": p - 0.01, "best_ask": p + 0.01}
            for i, p in enumerate(probs)
        ],
    }
    builder = ArbitrageBasketBuilder()
    return builder.build(opp, capital=capital)


_CFG = {
    "fee_taker_bps": 2.0,
    "slippage_from_spread_mult": 0.5,
    "slippage_bps_floor": 1.0,
    "lat_signal_to_order_ms": 50,
    "lat_order_to_exchange_ms": 150,
    "lat_exchange_to_ack_ms": 50,
}


# ---- single basket simulation ----

def test_simulate_returns_expected_fields():
    basket = _make_basket()
    result = simulate_basket(basket, _CFG, seed=42)
    assert "basket_id" in result
    assert "basket_expected_pnl" in result
    assert "basket_fill_rate" in result
    assert "basket_alpha_capture_ratio" in result
    assert "legs" in result
    assert "total_slippage_bps" in result
    assert "total_fees_bps" in result


def test_simulate_all_legs_filled():
    basket = _make_basket()
    result = simulate_basket(basket, _CFG)
    assert result["basket_fill_rate"] == 1.0
    assert all(leg["filled"] for leg in result["legs"])


def test_simulate_pnl_aggregation():
    basket = _make_basket()
    result = simulate_basket(basket, _CFG)
    leg_pnls = sum(leg["pnl_bps"] for leg in result["legs"])
    assert abs(result["basket_expected_pnl"] - leg_pnls) < 0.1


def test_simulate_slippage_non_negative():
    basket = _make_basket()
    result = simulate_basket(basket, _CFG)
    for leg in result["legs"]:
        assert leg["slippage_bps"] >= 0


def test_simulate_fees_match_config():
    basket = _make_basket()
    result = simulate_basket(basket, _CFG)
    for leg in result["legs"]:
        assert leg["fees_bps"] == _CFG["fee_taker_bps"]


# ---- determinism ----

def test_deterministic_same_seed():
    basket = _make_basket()
    r1 = simulate_basket(basket, _CFG, seed=123)
    r2 = simulate_basket(basket, _CFG, seed=123)
    assert r1["basket_expected_pnl"] == r2["basket_expected_pnl"]
    assert r1["legs"] == r2["legs"]


def test_different_seeds_same_structure():
    basket = _make_basket()
    r1 = simulate_basket(basket, _CFG, seed=1)
    r2 = simulate_basket(basket, _CFG, seed=2)
    assert r1["n_legs"] == r2["n_legs"]


# ---- empty / edge cases ----

def test_empty_basket():
    basket = {"basket_id": "", "type": "exclusivity", "edge": 0, "trades": []}
    result = simulate_basket(basket, _CFG)
    assert result["basket_expected_pnl"] == 0.0
    assert result["basket_fill_rate"] == 0.0
    assert result["legs"] == []


def test_zero_fee_config():
    cfg = {**_CFG, "fee_taker_bps": 0.0}
    basket = _make_basket()
    result = simulate_basket(basket, cfg)
    assert result["total_fees_bps"] == 0.0


# ---- multi-basket simulation ----

def test_simulate_baskets_aggregation():
    baskets = [_make_basket([0.40, 0.35, 0.30]), _make_basket([0.60, 0.50])]
    result = simulate_baskets(baskets, _CFG, seed=42)
    assert "basket_simulations" in result
    assert "basket_summary" in result
    assert result["basket_summary"]["total_baskets"] == 2


def test_simulate_baskets_summary_fields():
    baskets = [_make_basket()]
    result = simulate_baskets(baskets, _CFG)
    summary = result["basket_summary"]
    assert "mean_pnl_bps" in summary
    assert "mean_fill_rate" in summary
    assert "mean_alpha_capture" in summary
    assert "total_pnl_bps" in summary
    assert "profitable_baskets" in summary
    assert "profitable_pct" in summary


def test_simulate_baskets_empty():
    result = simulate_baskets([], _CFG)
    assert result["basket_summary"]["total_baskets"] == 0


# ---- ranking ----

def test_rank_baskets_sorted():
    baskets = [_make_basket([0.40, 0.35, 0.30]), _make_basket([0.80, 0.80])]
    result = simulate_baskets(baskets, _CFG, seed=42)
    ranked = rank_baskets(result["basket_simulations"], top_n=10)
    pnls = [r["basket_expected_pnl"] for r in ranked]
    assert pnls == sorted(pnls, reverse=True)


def test_rank_baskets_top_n():
    baskets = [_make_basket([0.60, 0.50]) for _ in range(20)]
    result = simulate_baskets(baskets, _CFG)
    ranked = rank_baskets(result["basket_simulations"], top_n=5)
    assert len(ranked) == 5


def test_rank_baskets_fields():
    baskets = [_make_basket()]
    result = simulate_baskets(baskets, _CFG)
    ranked = rank_baskets(result["basket_simulations"])
    r = ranked[0]
    assert "basket_id" in r
    assert "basket_expected_pnl" in r
    assert "basket_fill_rate" in r
    assert "basket_alpha_capture_ratio" in r


# ---- scaling ----

def test_1000_baskets_under_1_second():
    baskets = [_make_basket([0.40, 0.35, 0.30]) for _ in range(1000)]
    start = time.monotonic()
    result = simulate_baskets(baskets, _CFG, seed=42)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"Took {elapsed:.2f}s, expected < 1s"
    assert result["basket_summary"]["total_baskets"] == 1000
