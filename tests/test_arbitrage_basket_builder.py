"""Tests for ArbitrageBasketBuilder."""

from src.strategy.arbitrage_basket_builder import ArbitrageBasketBuilder


def _excl_opp(probs, edge=None):
    markets = [
        {"market_id": f"m{i}", "prob": p, "best_bid": p - 0.01, "best_ask": p + 0.01, "category": "test"}
        for i, p in enumerate(probs)
    ]
    if edge is None:
        edge = round(sum(probs) - 1.0, 6)
    return {"type": "exclusivity", "edge": edge, "markets": markets, "total_prob": sum(probs)}


def _compl_opp(pa, pb, direction="overpriced"):
    return {
        "type": "complement",
        "edge": round(abs(pa + pb - 1.0), 6),
        "direction": direction,
        "markets": [
            {"market_id": "yes", "prob": pa, "best_bid": pa - 0.01, "best_ask": pa + 0.01},
            {"market_id": "no", "prob": pb, "best_bid": pb - 0.01, "best_ask": pb + 0.01},
        ],
    }


# ---- exclusivity baskets ----

def test_exclusivity_basket_weights_sum_to_one():
    builder = ArbitrageBasketBuilder()
    opp = _excl_opp([0.40, 0.35, 0.30])
    basket = builder.build(opp)
    weights = [t["weight"] for t in basket["trades"]]
    assert abs(sum(weights) - 1.0) < 1e-4


def test_exclusivity_basket_sizes_sum_to_capital():
    builder = ArbitrageBasketBuilder()
    opp = _excl_opp([0.40, 0.35, 0.30])
    basket = builder.build(opp, capital=200.0)
    sizes = [t["size"] for t in basket["trades"]]
    assert abs(sum(sizes) - 200.0) < 0.1


def test_exclusivity_all_sides_sell():
    builder = ArbitrageBasketBuilder()
    opp = _excl_opp([0.40, 0.35, 0.30])
    basket = builder.build(opp)
    assert all(t["side"] == "SELL" for t in basket["trades"])


def test_exclusivity_basket_has_correct_fields():
    builder = ArbitrageBasketBuilder()
    opp = _excl_opp([0.60, 0.50])
    basket = builder.build(opp)
    assert "basket_id" in basket
    assert "edge" in basket
    assert "trades" in basket
    assert "n_legs" in basket
    assert "timestamp" in basket
    assert basket["n_legs"] == 2
    assert basket["type"] == "exclusivity"


def test_exclusivity_trade_fields():
    builder = ArbitrageBasketBuilder()
    opp = _excl_opp([0.60, 0.50])
    basket = builder.build(opp)
    trade = basket["trades"][0]
    assert "market_id" in trade
    assert "side" in trade
    assert "size" in trade
    assert "weight" in trade
    assert "prob" in trade


# ---- complement baskets ----

def test_complement_basket_two_legs():
    builder = ArbitrageBasketBuilder()
    opp = _compl_opp(0.60, 0.50, "overpriced")
    basket = builder.build(opp)
    assert basket["n_legs"] == 2
    assert basket["type"] == "complement"


def test_complement_equal_weights():
    builder = ArbitrageBasketBuilder()
    opp = _compl_opp(0.60, 0.50)
    basket = builder.build(opp)
    assert all(t["weight"] == 0.5 for t in basket["trades"])


# ---- edge cases ----

def test_empty_markets_returns_empty():
    builder = ArbitrageBasketBuilder()
    opp = {"type": "exclusivity", "edge": 0.05, "markets": []}
    basket = builder.build(opp)
    assert basket["trades"] == []


def test_zero_edge_returns_empty():
    builder = ArbitrageBasketBuilder()
    opp = {"type": "exclusivity", "edge": 0, "markets": [{"market_id": "x", "prob": 0.5}]}
    basket = builder.build(opp)
    assert basket["trades"] == []


# ---- determinism ----

def test_deterministic_basket_id():
    builder = ArbitrageBasketBuilder()
    opp = _excl_opp([0.40, 0.35, 0.30])
    b1 = builder.build(opp)
    b2 = builder.build(opp)
    assert b1["basket_id"] == b2["basket_id"]
    assert len(b1["basket_id"]) == 12


def test_different_opportunities_different_ids():
    builder = ArbitrageBasketBuilder()
    opp1 = {"type": "exclusivity", "edge": 0.10, "markets": [
        {"market_id": "alpha", "prob": 0.60}, {"market_id": "beta", "prob": 0.50}
    ]}
    opp2 = {"type": "exclusivity", "edge": 0.10, "markets": [
        {"market_id": "gamma", "prob": 0.70}, {"market_id": "delta", "prob": 0.40}
    ]}
    b1 = builder.build(opp1)
    b2 = builder.build(opp2)
    assert b1["basket_id"] != b2["basket_id"]


# ---- build_many ----

def test_build_many():
    builder = ArbitrageBasketBuilder()
    opps = [_excl_opp([0.40, 0.35, 0.30]), _excl_opp([0.60, 0.50])]
    baskets = builder.build_many(opps, capital=100)
    assert len(baskets) == 2
    edges = [b["edge"] for b in baskets]
    assert edges == sorted(edges, reverse=True)


def test_build_many_max_limit():
    builder = ArbitrageBasketBuilder()
    opps = [_excl_opp([0.60, 0.50])] * 20
    baskets = builder.build_many(opps, max_baskets=5)
    assert len(baskets) == 5


def test_build_many_empty():
    builder = ArbitrageBasketBuilder()
    assert builder.build_many([]) == []
