"""Tests for ShadowTrader."""

import time
from src.trading.shadow_trader import ShadowTrader


def _sim_positive(opp):
    return {"pnl_bps": 10.0, "fill_rate": 1.0, "slippage_bps": 2.0}


def _sim_negative(opp):
    return {"pnl_bps": -5.0, "fill_rate": 0.8, "slippage_bps": 3.0}


# ---- basic recording ----

def test_process_records_trade():
    trader = ShadowTrader(simulator=_sim_positive)
    trade = trader.process_opportunity({"market_id": "m1", "side": "BUY"})
    assert trade["expected_pnl"] == 10.0
    assert trade["fill_rate"] == 1.0
    assert trader.trade_count == 1


def test_trade_has_required_fields():
    trader = ShadowTrader(simulator=_sim_positive)
    trade = trader.process_opportunity({"market_id": "m1"})
    assert "shadow_trade_id" in trade
    assert "timestamp" in trade
    assert "market_id" in trade
    assert "expected_pnl" in trade
    assert "fill_rate" in trade
    assert "opportunity" in trade


def test_unique_trade_ids():
    trader = ShadowTrader(simulator=_sim_positive)
    t1 = trader.process_opportunity({"market_id": "m1"})
    t2 = trader.process_opportunity({"market_id": "m2"})
    assert t1["shadow_trade_id"] != t2["shadow_trade_id"]


# ---- batch processing ----

def test_process_many():
    trader = ShadowTrader(simulator=_sim_positive)
    opps = [{"market_id": f"m{i}"} for i in range(5)]
    results = trader.process_many(opps)
    assert len(results) == 5
    assert trader.trade_count == 5


# ---- summary ----

def test_summary_with_trades():
    trader = ShadowTrader(simulator=_sim_positive)
    for i in range(3):
        trader.process_opportunity({"market_id": f"m{i}"})
    s = trader.summary()
    assert s["shadow_trades"] == 3
    assert s["total_expected_pnl"] == 30.0
    assert s["mean_expected_pnl"] == 10.0
    assert s["profitable_trades"] == 3
    assert s["profitable_pct"] == 1.0


def test_summary_empty():
    trader = ShadowTrader()
    s = trader.summary()
    assert s["shadow_trades"] == 0
    assert s["total_expected_pnl"] == 0.0


def test_summary_mixed_pnl():
    def alternating(opp):
        if opp.get("win"):
            return {"pnl_bps": 10.0, "fill_rate": 1.0, "slippage_bps": 0}
        return {"pnl_bps": -5.0, "fill_rate": 0.5, "slippage_bps": 0}

    trader = ShadowTrader(simulator=alternating)
    trader.process_opportunity({"market_id": "a", "win": True})
    trader.process_opportunity({"market_id": "b", "win": False})
    s = trader.summary()
    assert s["shadow_trades"] == 2
    assert s["profitable_trades"] == 1
    assert s["profitable_pct"] == 0.5


# ---- reset ----

def test_reset():
    trader = ShadowTrader(simulator=_sim_positive)
    trader.process_opportunity({"market_id": "m1"})
    trader.reset()
    assert trader.trade_count == 0
    assert trader.summary()["shadow_trades"] == 0


# ---- noop simulator ----

def test_noop_simulator():
    trader = ShadowTrader()
    trade = trader.process_opportunity({"market_id": "x"})
    assert trade["expected_pnl"] == 0.0
    assert trade["fill_rate"] == 0.0


# ---- scaling ----

def test_1000_opportunities():
    trader = ShadowTrader(simulator=_sim_positive)
    opps = [{"market_id": f"m{i}"} for i in range(1000)]
    trader.process_many(opps)
    assert trader.trade_count == 1000
    assert trader.summary()["shadow_trades"] == 1000
