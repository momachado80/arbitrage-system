"""Tests for OpportunityPrioritizer."""

import time
from src.trading.opportunity_prioritizer import OpportunityPrioritizer


def _opp(market_id, pnl=10, liq=0.8, fill=0.9):
    return {
        "market_id": market_id,
        "expected_pnl_per_hour": pnl,
        "liquidity_score": liq,
        "fill_rate": fill,
    }


# ---- scoring ----

def test_score_positive():
    p = OpportunityPrioritizer()
    s = p.score(_opp("m1", pnl=10, liq=0.8, fill=0.9), regime_score=0.7)
    assert s > 0


def test_score_zero_pnl():
    p = OpportunityPrioritizer()
    assert p.score(_opp("m1", pnl=0), regime_score=0.5) == 0.0


def test_score_scales_with_regime():
    p = OpportunityPrioritizer()
    low = p.score(_opp("m1"), regime_score=0.1)
    high = p.score(_opp("m1"), regime_score=0.9)
    assert high > low


def test_score_scales_with_pnl():
    p = OpportunityPrioritizer()
    low = p.score(_opp("m1", pnl=1))
    high = p.score(_opp("m1", pnl=100))
    assert high > low


# ---- ranking ----

def test_rank_sorted_descending():
    p = OpportunityPrioritizer()
    opps = [_opp("a", pnl=5), _opp("b", pnl=20), _opp("c", pnl=10)]
    ranked = p.rank(opps, regime_score=0.7)
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_top_n():
    p = OpportunityPrioritizer()
    opps = [_opp(f"m{i}", pnl=i) for i in range(20)]
    ranked = p.rank(opps, regime_score=0.5, top_n=5)
    assert len(ranked) == 5


def test_rank_output_fields():
    p = OpportunityPrioritizer()
    ranked = p.rank([_opp("m1")], regime_score=0.5)
    r = ranked[0]
    assert "score" in r
    assert "market_id" in r
    assert "expected_pnl_per_hour" in r
    assert "opportunity" in r


def test_rank_preserves_opportunity():
    p = OpportunityPrioritizer()
    opp = _opp("m1", pnl=10)
    ranked = p.rank([opp], regime_score=0.5)
    assert ranked[0]["opportunity"] is opp


# ---- determinism ----

def test_deterministic():
    p = OpportunityPrioritizer()
    opps = [_opp(f"m{i}", pnl=i * 3) for i in range(10)]
    r1 = p.rank(opps, regime_score=0.6)
    r2 = p.rank(opps, regime_score=0.6)
    assert [r["score"] for r in r1] == [r["score"] for r in r2]


# ---- empty input ----

def test_empty_opportunities():
    p = OpportunityPrioritizer()
    assert p.rank([], regime_score=0.5) == []


def test_missing_fields_safe():
    p = OpportunityPrioritizer()
    s = p.score({}, regime_score=0.5)
    assert s == 0.0


# ---- scaling ----

def test_1000_opportunities():
    p = OpportunityPrioritizer()
    opps = [_opp(f"m{i}", pnl=i * 0.1) for i in range(1000)]
    start = time.monotonic()
    ranked = p.rank(opps, regime_score=0.7)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0
    assert len(ranked) == 1000
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
