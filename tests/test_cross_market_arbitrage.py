"""Tests for CrossMarketArbitrageEngine."""

from src.strategy.cross_market_arbitrage import CrossMarketArbitrageEngine


def _mk(market_id, prob, category="election", bid=None, ask=None):
    return {
        "market_id": market_id,
        "prob": prob,
        "category": category,
        "best_bid": bid or prob - 0.01,
        "best_ask": ask or prob + 0.01,
    }


# ---- exclusivity ----

def test_exclusivity_detected_when_sum_exceeds_one():
    engine = CrossMarketArbitrageEngine()
    markets = [_mk("A", 0.40), _mk("B", 0.35), _mk("C", 0.30)]
    result = engine.detect_exclusivity(markets)
    assert result is not None
    assert result["type"] == "exclusivity"
    assert result["edge"] > 0
    assert abs(result["edge"] - 0.05) < 1e-4


def test_exclusivity_not_detected_when_sum_under_one():
    engine = CrossMarketArbitrageEngine()
    markets = [_mk("A", 0.30), _mk("B", 0.30), _mk("C", 0.30)]
    assert engine.detect_exclusivity(markets) is None


def test_exclusivity_exactly_one_returns_none():
    engine = CrossMarketArbitrageEngine()
    markets = [_mk("A", 0.50), _mk("B", 0.50)]
    assert engine.detect_exclusivity(markets) is None


def test_exclusivity_empty_input():
    engine = CrossMarketArbitrageEngine()
    assert engine.detect_exclusivity([]) is None


def test_exclusivity_single_market():
    engine = CrossMarketArbitrageEngine()
    assert engine.detect_exclusivity([_mk("A", 0.8)]) is None


def test_exclusivity_output_fields():
    engine = CrossMarketArbitrageEngine()
    markets = [_mk("A", 0.60), _mk("B", 0.50)]
    result = engine.detect_exclusivity(markets)
    assert "type" in result
    assert "edge" in result
    assert "markets" in result
    assert "timestamp" in result
    assert "total_prob" in result


# ---- complement ----

def test_complement_detected():
    engine = CrossMarketArbitrageEngine()
    a = _mk("yes", 0.55)
    b = _mk("no", 0.50)
    result = engine.detect_complement(a, b)
    assert result is not None
    assert result["type"] == "complement"
    assert result["edge"] > 0.02


def test_complement_within_threshold_returns_none():
    engine = CrossMarketArbitrageEngine()
    a = _mk("yes", 0.51)
    b = _mk("no", 0.49)
    assert engine.detect_complement(a, b) is None


def test_complement_underpriced():
    engine = CrossMarketArbitrageEngine()
    a = _mk("yes", 0.40)
    b = _mk("no", 0.50)
    result = engine.detect_complement(a, b)
    assert result is not None
    assert result["direction"] == "underpriced"


def test_complement_overpriced():
    engine = CrossMarketArbitrageEngine()
    a = _mk("yes", 0.60)
    b = _mk("no", 0.50)
    result = engine.detect_complement(a, b)
    assert result is not None
    assert result["direction"] == "overpriced"


def test_complement_custom_threshold():
    engine = CrossMarketArbitrageEngine(complement_threshold=0.10)
    a = _mk("yes", 0.55)
    b = _mk("no", 0.50)
    assert engine.detect_complement(a, b) is None


# ---- detect_all ----

def test_detect_all_combined():
    engine = CrossMarketArbitrageEngine()
    groups = {
        "election": [_mk("A", 0.40, "election"), _mk("B", 0.35, "election"), _mk("C", 0.30, "election")],
        "sports": [_mk("X", 0.30, "sports"), _mk("Y", 0.30, "sports")],
    }
    pairs = [(_mk("yes", 0.60), _mk("no", 0.50))]
    results = engine.detect_all(groups, complement_pairs=pairs)
    assert len(results) >= 1
    types = {r["type"] for r in results}
    assert "exclusivity" in types or "complement" in types


def test_detect_all_sorted_by_edge():
    engine = CrossMarketArbitrageEngine()
    groups = {
        "g1": [_mk("A", 0.60), _mk("B", 0.60)],
        "g2": [_mk("C", 0.80), _mk("D", 0.80)],
    }
    results = engine.detect_all(groups)
    edges = [r["edge"] for r in results]
    assert edges == sorted(edges, reverse=True)


def test_detect_all_no_opportunities():
    engine = CrossMarketArbitrageEngine()
    groups = {"safe": [_mk("A", 0.30), _mk("B", 0.30)]}
    assert engine.detect_all(groups) == []
