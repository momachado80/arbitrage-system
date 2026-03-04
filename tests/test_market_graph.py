"""Tests for MarketGraph."""

from src.strategy.market_graph import MarketGraph


def test_add_market_and_lookup():
    g = MarketGraph()
    g.add_market("m1", {"name": "Market 1"})
    assert "m1" in g.nodes
    assert g.market_count == 1


def test_add_edge_bidirectional():
    g = MarketGraph()
    g.add_market("a")
    g.add_market("b")
    g.add_edge("a", "b", "complement")
    assert ("b", "complement") in g.related_markets("a")
    assert ("a", "complement") in g.related_markets("b")


def test_add_edge_auto_creates_nodes():
    g = MarketGraph()
    g.add_edge("x", "y", "mutual_exclusive")
    assert "x" in g.nodes
    assert "y" in g.nodes


def test_invalid_relation_ignored():
    g = MarketGraph()
    g.add_edge("a", "b", "invalid_relation")
    assert g.related_markets("a") == []
    assert g.edge_count == 0


def test_related_by_type():
    g = MarketGraph()
    g.add_edge("a", "b", "complement")
    g.add_edge("a", "c", "mutual_exclusive")
    assert g.related_by_type("a", "complement") == ["b"]
    assert g.related_by_type("a", "mutual_exclusive") == ["c"]


def test_no_duplicate_edges():
    g = MarketGraph()
    g.add_edge("a", "b", "complement")
    g.add_edge("a", "b", "complement")
    assert len(g.related_markets("a")) == 1


def test_remove_market():
    g = MarketGraph()
    g.add_edge("a", "b", "complement")
    g.add_edge("a", "c", "mutual_exclusive")
    g.remove_market("a")
    assert "a" not in g.nodes
    assert g.related_markets("b") == []
    assert g.related_markets("c") == []


def test_get_exclusive_groups():
    g = MarketGraph()
    g.add_edge("a", "b", "mutual_exclusive")
    g.add_edge("b", "c", "mutual_exclusive")
    g.add_market("d")
    groups = g.get_exclusive_groups()
    assert len(groups) == 1
    assert sorted(groups[0]) == ["a", "b", "c"]


def test_get_complement_pairs():
    g = MarketGraph()
    g.add_edge("yes1", "no1", "complement")
    g.add_edge("yes2", "no2", "complement")
    pairs = g.get_complement_pairs()
    assert len(pairs) == 2


def test_complement_pairs_deduplicated():
    g = MarketGraph()
    g.add_edge("a", "b", "complement")
    pairs = g.get_complement_pairs()
    assert len(pairs) == 1


def test_empty_graph():
    g = MarketGraph()
    assert g.market_count == 0
    assert g.edge_count == 0
    assert g.get_exclusive_groups() == []
    assert g.get_complement_pairs() == []


def test_summary():
    g = MarketGraph()
    g.add_edge("a", "b", "mutual_exclusive")
    g.add_edge("c", "d", "complement")
    s = g.summary()
    assert s["markets"] == 4
    assert s["edges"] == 2
    assert s["exclusive_groups"] == 1
    assert s["complement_pairs"] == 1


def test_related_markets_empty():
    g = MarketGraph()
    assert g.related_markets("nonexistent") == []
