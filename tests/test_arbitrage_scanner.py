"""Tests for ArbitrageScanner — scaling and correctness."""

import time
from src.strategy.arbitrage_scanner import ArbitrageScanner
from src.strategy.cross_market_arbitrage import CrossMarketArbitrageEngine
from src.strategy.market_graph import MarketGraph


def _mk(market_id, prob, category="default"):
    return {
        "market_id": market_id,
        "prob": prob,
        "category": category,
        "best_bid": prob - 0.01,
        "best_ask": prob + 0.01,
    }


# ---- basic scanning ----

def test_scan_finds_exclusivity():
    scanner = ArbitrageScanner()
    markets = [_mk("A", 0.40, "election"), _mk("B", 0.35, "election"), _mk("C", 0.30, "election")]
    result = scanner.scan(markets)
    opps = result["cross_market_arbitrage_opportunities"]
    assert len(opps) >= 1
    assert opps[0]["type"] == "exclusivity"


def test_scan_no_opportunity():
    scanner = ArbitrageScanner()
    markets = [_mk("A", 0.30, "cat1"), _mk("B", 0.30, "cat1")]
    result = scanner.scan(markets)
    assert result["cross_market_arbitrage_opportunities"] == []


def test_scan_empty_markets():
    scanner = ArbitrageScanner()
    result = scanner.scan([])
    assert result["cross_market_arbitrage_opportunities"] == []
    assert result["scan_metadata"]["markets_scanned"] == 0


def test_scan_metadata_present():
    scanner = ArbitrageScanner()
    markets = [_mk(f"m{i}", 0.10, "cat") for i in range(10)]
    result = scanner.scan(markets)
    meta = result["scan_metadata"]
    assert "markets_scanned" in meta
    assert "categories" in meta
    assert "opportunities_found" in meta
    assert "scan_time_ms" in meta
    assert meta["markets_scanned"] == 10


def test_scan_groups_by_category():
    scanner = ArbitrageScanner()
    markets = [
        _mk("A", 0.60, "election"),
        _mk("B", 0.50, "election"),
        _mk("C", 0.60, "sports"),
        _mk("D", 0.50, "sports"),
    ]
    result = scanner.scan(markets)
    opps = result["cross_market_arbitrage_opportunities"]
    assert len(opps) == 2


# ---- graph integration ----

def test_scan_with_graph_complements():
    graph = MarketGraph()
    graph.add_edge("yes1", "no1", "complement")
    engine = CrossMarketArbitrageEngine(complement_threshold=0.02)
    scanner = ArbitrageScanner(engine=engine, graph=graph)
    markets = [
        _mk("yes1", 0.60, "cat"),
        _mk("no1", 0.50, "cat"),
    ]
    result = scanner.scan(markets)
    opps = result["cross_market_arbitrage_opportunities"]
    types = {o["type"] for o in opps}
    assert "complement" in types


def test_scan_with_graph_exclusive_groups():
    graph = MarketGraph()
    graph.add_edge("x", "y", "mutual_exclusive")
    graph.add_edge("y", "z", "mutual_exclusive")
    scanner = ArbitrageScanner(graph=graph)
    markets = [_mk("x", 0.40, "a"), _mk("y", 0.35, "b"), _mk("z", 0.30, "c")]
    result = scanner.scan(markets)
    opps = result["cross_market_arbitrage_opportunities"]
    exclusivities = [o for o in opps if o["type"] == "exclusivity"]
    assert len(exclusivities) >= 1


# ---- determinism ----

def test_deterministic_results():
    scanner = ArbitrageScanner()
    markets = [_mk(f"m{i}", 0.20 + i * 0.05, "cat") for i in range(8)]
    r1 = scanner.scan(markets)
    r2 = scanner.scan(markets)
    o1 = r1["cross_market_arbitrage_opportunities"]
    o2 = r2["cross_market_arbitrage_opportunities"]
    assert len(o1) == len(o2)
    for a, b in zip(o1, o2):
        assert a["edge"] == b["edge"]
        assert a["type"] == b["type"]


# ---- scaling ----

def test_scan_1000_markets_under_500ms():
    scanner = ArbitrageScanner()
    markets = [_mk(f"m{i}", 0.10 + (i % 10) * 0.01, f"cat_{i % 50}") for i in range(1000)]
    start = time.monotonic()
    result = scanner.scan(markets)
    elapsed_ms = (time.monotonic() - start) * 1000
    assert elapsed_ms < 500, f"Scan took {elapsed_ms:.1f}ms, expected < 500ms"
    assert result["scan_metadata"]["markets_scanned"] == 1000


def test_scan_5000_markets_no_crash():
    scanner = ArbitrageScanner()
    markets = [_mk(f"m{i}", 0.10 + (i % 10) * 0.01, f"cat_{i % 100}") for i in range(5000)]
    result = scanner.scan(markets)
    assert result["scan_metadata"]["markets_scanned"] == 5000


def test_opportunities_sorted_by_edge():
    scanner = ArbitrageScanner()
    markets = [
        _mk("A", 0.60, "g1"), _mk("B", 0.50, "g1"),
        _mk("C", 0.80, "g2"), _mk("D", 0.80, "g2"),
    ]
    result = scanner.scan(markets)
    opps = result["cross_market_arbitrage_opportunities"]
    edges = [o["edge"] for o in opps]
    assert edges == sorted(edges, reverse=True)
