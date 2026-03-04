"""Tests for RiskEngine."""

from src.trading.risk_engine import RiskEngine


# ---- basic allow/deny ----

def test_allow_within_limits():
    eng = RiskEngine(max_position_per_market=100, max_total_inventory=1000)
    assert eng.allow_trade({"market_id": "m1", "size": 50}) is True


def test_deny_exceeds_per_market():
    eng = RiskEngine(max_position_per_market=100)
    assert eng.allow_trade({"market_id": "m1", "size": 150}) is False


def test_deny_exceeds_total_inventory():
    eng = RiskEngine(max_total_inventory=100)
    eng.update({"market_id": "m1", "size": 80})
    assert eng.allow_trade({"market_id": "m2", "size": 30}) is False


def test_deny_after_drawdown():
    eng = RiskEngine(max_drawdown=-100)
    eng.pnl = -150
    assert eng.allow_trade({"market_id": "m1", "size": 10}) is False


def test_allow_at_exact_drawdown_boundary():
    eng = RiskEngine(max_drawdown=-100)
    eng.pnl = -100
    assert eng.allow_trade({"market_id": "m1", "size": 10}) is False


def test_allow_just_above_drawdown():
    eng = RiskEngine(max_drawdown=-100)
    eng.pnl = -99
    assert eng.allow_trade({"market_id": "m1", "size": 10}) is True


# ---- position accumulation ----

def test_position_accumulates():
    eng = RiskEngine(max_position_per_market=100)
    eng.update({"market_id": "m1", "size": 60})
    assert eng.allow_trade({"market_id": "m1", "size": 50}) is False
    assert eng.allow_trade({"market_id": "m1", "size": 30}) is True


def test_multiple_markets_independent():
    eng = RiskEngine(max_position_per_market=100, max_total_inventory=10000)
    eng.update({"market_id": "m1", "size": 90})
    assert eng.allow_trade({"market_id": "m2", "size": 90}) is True


# ---- update and state ----

def test_update_tracks_pnl():
    eng = RiskEngine()
    eng.update({"market_id": "m1", "size": 10}, pnl=5.0)
    eng.update({"market_id": "m1", "size": 10}, pnl=-2.0)
    assert eng.pnl == 3.0


def test_state_output():
    eng = RiskEngine()
    eng.update({"market_id": "m1", "size": 50}, pnl=10.0)
    s = eng.state()
    assert s["pnl"] == 10.0
    assert s["total_inventory"] == 50.0
    assert s["markets_with_position"] == 1
    assert "drawdown_remaining" in s
    assert "inventory" in s


# ---- reset ----

def test_reset():
    eng = RiskEngine()
    eng.update({"market_id": "m1", "size": 50}, pnl=10.0)
    eng.reset()
    assert eng.pnl == 0.0
    assert eng.inventory == {}
    assert eng.state()["total_inventory"] == 0.0


# ---- empty trade ----

def test_empty_trade_allowed():
    eng = RiskEngine()
    assert eng.allow_trade({"market_id": "x", "size": 0}) is True


def test_missing_size_defaults_zero():
    eng = RiskEngine()
    assert eng.allow_trade({"market_id": "x"}) is True
