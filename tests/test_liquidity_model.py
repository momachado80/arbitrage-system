"""Tests for LiquidityModel — slippage computation."""

from src.sim.liquidity_model import LiquidityModel


def test_slippage_always_non_negative():
    lm = LiquidityModel(impact_coeff=0.1)
    for liq in [0, 1, 10, 100, 1000]:
        for spread in [0, 50, 100, 500]:
            slip = lm.compute_slippage(10, liq, spread)
            assert slip >= 0, f"Negative slippage: liq={liq}, spread={spread}"


def test_zero_liquidity_returns_spread():
    lm = LiquidityModel(impact_coeff=0.5)
    assert lm.compute_slippage(10, 0, 200) == 200.0


def test_high_liquidity_low_impact():
    lm = LiquidityModel(impact_coeff=0.1)
    slip_low_liq = lm.compute_slippage(10, 10, 100)
    slip_high_liq = lm.compute_slippage(10, 10000, 100)
    assert slip_high_liq <= slip_low_liq


def test_simple_slippage_respects_floor():
    lm = LiquidityModel()
    slip = lm.compute_slippage_simple(spread_bps=50, spread_mult=0.5, floor_bps=30)
    assert slip == 30.0  # max(50*0.5=25, 30) = 30


def test_simple_slippage_uses_spread_when_higher():
    lm = LiquidityModel()
    slip = lm.compute_slippage_simple(spread_bps=100, spread_mult=0.5, floor_bps=10)
    assert slip == 50.0  # max(100*0.5=50, 10) = 50


def test_empty_spread_uses_floor():
    lm = LiquidityModel()
    slip = lm.compute_slippage_simple(spread_bps=0, spread_mult=0.5, floor_bps=5.0)
    assert slip == 5.0
