"""Tests for MarketRegimeDetector."""

from src.strategy.market_regime_detector import MarketRegimeDetector


def _analytics(
    survival_500=0.0, survival_2000=0.0,
    episodes_per_hour=0.0, spread=0.01,
    bid_size=0.0, ask_size=0.0,
    hazard_250_500=0.0, hazard_500_1000=0.0,
):
    return {
        "edge_survival_curve": {
            "50ms": 1.0, "100ms": 0.95, "250ms": 0.90,
            "500ms": survival_500, "1000ms": 0.5,
            "2000ms": survival_2000, "5000ms": 0.1,
        },
        "edge_hazard_curve": {
            "0-100ms": 0.05, "100-250ms": 0.10,
            "250-500ms": hazard_250_500,
            "500-1000ms": hazard_500_1000,
            "1000-2000ms": 0.15, "2000-5000ms": 0.10,
        },
        "aggregates": {
            "episodes_per_hour": episodes_per_hour,
            "spread_bps_mean": spread,
            "bid_size_mean": bid_size,
            "ask_size_mean": ask_size,
        },
    }


# ---- score bounds ----

def test_score_between_0_and_1():
    det = MarketRegimeDetector()
    for s500 in [0, 0.5, 1.0]:
        for eph in [0, 50, 200]:
            result = det.compute(_analytics(survival_500=s500, episodes_per_hour=eph))
            assert 0.0 <= result["regime_score"] <= 1.0


def test_score_clamped_at_zero():
    det = MarketRegimeDetector()
    result = det.compute(_analytics(hazard_500_1000=1.0))
    assert result["regime_score"] >= 0.0


def test_score_clamped_at_one():
    det = MarketRegimeDetector()
    result = det.compute(_analytics(
        survival_500=1.0, survival_2000=1.0,
        episodes_per_hour=200, bid_size=500, ask_size=500,
    ))
    assert result["regime_score"] <= 1.0


# ---- classification thresholds ----

def test_high_edge_classification():
    det = MarketRegimeDetector()
    result = det.compute(_analytics(
        survival_500=0.9, survival_2000=0.7,
        episodes_per_hour=80, bid_size=100, ask_size=100,
    ))
    assert result["state"] == "high_edge"
    assert result["regime_score"] > 0.65


def test_hostile_classification():
    det = MarketRegimeDetector()
    result = det.compute(_analytics(
        survival_500=0.0, survival_2000=0.0,
        episodes_per_hour=0, hazard_500_1000=0.8,
    ))
    assert result["state"] == "hostile"
    assert result["regime_score"] < 0.35


def test_neutral_classification():
    det = MarketRegimeDetector()
    result = det.compute(_analytics(
        survival_500=0.5, survival_2000=0.3,
        episodes_per_hour=30,
    ))
    assert result["state"] == "neutral"
    assert 0.35 <= result["regime_score"] <= 0.65


# ---- component breakdown ----

def test_components_present():
    det = MarketRegimeDetector()
    result = det.compute(_analytics(survival_500=0.5))
    comps = result["components"]
    assert "edge_persistence" in comps
    assert "opportunity_density" in comps
    assert "liquidity_score" in comps
    assert "competition" in comps


def test_edge_persistence_from_survival():
    det = MarketRegimeDetector()
    result = det.compute(_analytics(survival_500=0.80, survival_2000=0.40))
    assert abs(result["components"]["edge_persistence"] - 0.60) < 0.01


def test_opportunity_density_normalized():
    det = MarketRegimeDetector()
    r1 = det.compute(_analytics(episodes_per_hour=50))
    r2 = det.compute(_analytics(episodes_per_hour=200))
    assert r1["components"]["opportunity_density"] == 0.5
    assert r2["components"]["opportunity_density"] == 1.0


def test_competition_from_hazard():
    det = MarketRegimeDetector()
    result = det.compute(_analytics(hazard_500_1000=0.35))
    assert abs(result["components"]["competition"] - 0.35) < 0.01


# ---- empty / missing input ----

def test_empty_analytics():
    det = MarketRegimeDetector()
    result = det.compute({})
    assert result["regime_score"] == 0.0
    assert result["state"] == "hostile"


def test_none_analytics():
    det = MarketRegimeDetector()
    result = det.compute(None)
    assert result["state"] == "hostile"


def test_missing_survival_curve():
    det = MarketRegimeDetector()
    result = det.compute({"aggregates": {"episodes_per_hour": 10}})
    assert result["components"]["edge_persistence"] == 0.0
    assert 0.0 <= result["regime_score"] <= 1.0


def test_missing_hazard():
    det = MarketRegimeDetector()
    result = det.compute({"edge_survival_curve": {"500ms": 0.8, "2000ms": 0.5}})
    assert result["components"]["competition"] == 0.0


# ---- custom weights ----

def test_custom_weights():
    det = MarketRegimeDetector(w_persistence=1.0, w_opportunity=0, w_liquidity=0, w_competition=0)
    result = det.compute(_analytics(survival_500=0.80, survival_2000=0.60))
    assert abs(result["regime_score"] - 0.70) < 0.01


def test_custom_thresholds():
    det = MarketRegimeDetector(high_threshold=0.90, hostile_threshold=0.10)
    result = det.compute(_analytics(survival_500=0.5, episodes_per_hour=30))
    assert result["state"] == "neutral"


# ---- higher survival → higher score ----

def test_more_persistence_higher_score():
    det = MarketRegimeDetector()
    low = det.compute(_analytics(survival_500=0.1, survival_2000=0.05))
    high = det.compute(_analytics(survival_500=0.9, survival_2000=0.7))
    assert high["regime_score"] > low["regime_score"]


# ---- more competition → lower score ----

def test_more_competition_lower_score():
    det = MarketRegimeDetector()
    low_comp = det.compute(_analytics(survival_500=0.5, hazard_500_1000=0.0))
    high_comp = det.compute(_analytics(survival_500=0.5, hazard_500_1000=0.8))
    assert low_comp["regime_score"] > high_comp["regime_score"]
