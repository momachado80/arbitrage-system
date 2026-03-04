"""
Arbitrage Scanner — Efficient cross-market scan
================================================

Groups markets by category for O(n·k) scanning instead of O(n²).
Designed for 1000+ market universes.

No external dependencies.
"""

import time
from typing import Any, Dict, List, Optional

from src.strategy.cross_market_arbitrage import CrossMarketArbitrageEngine
from src.strategy.market_graph import MarketGraph


class ArbitrageScanner:
    """Scan a universe of markets for arbitrage using category grouping + graph."""

    def __init__(
        self,
        engine: Optional[CrossMarketArbitrageEngine] = None,
        graph: Optional[MarketGraph] = None,
    ) -> None:
        self.engine = engine or CrossMarketArbitrageEngine()
        self.graph = graph or MarketGraph()

    def scan(
        self,
        markets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Scan all markets for arbitrage opportunities.

        Each market dict must contain:
          market_id, prob, category, best_bid, best_ask

        Returns dict with opportunities list and scan metadata.
        """
        ts_start = time.monotonic()
        opportunities: List[Dict[str, Any]] = []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for m in markets:
            cat = m.get("category", "unknown")
            grouped.setdefault(cat, []).append(m)

        for _cat, group in grouped.items():
            opp = self.engine.detect_exclusivity(group)
            if opp is not None:
                opportunities.append(opp)

        if self.graph.market_count > 0:
            self._scan_graph_complements(markets, opportunities)
            self._scan_graph_exclusives(markets, opportunities)

        opportunities.sort(key=lambda x: -x.get("edge", 0))

        elapsed_ms = round((time.monotonic() - ts_start) * 1000, 2)
        return {
            "cross_market_arbitrage_opportunities": opportunities,
            "scan_metadata": {
                "markets_scanned": len(markets),
                "categories": len(grouped),
                "opportunities_found": len(opportunities),
                "scan_time_ms": elapsed_ms,
            },
        }

    def _scan_graph_complements(
        self,
        markets: List[Dict[str, Any]],
        opportunities: List[Dict[str, Any]],
    ) -> None:
        market_map = {m["market_id"]: m for m in markets if "market_id" in m}
        for a_id, b_id in self.graph.get_complement_pairs():
            a = market_map.get(a_id)
            b = market_map.get(b_id)
            if a and b:
                opp = self.engine.detect_complement(a, b)
                if opp is not None:
                    opportunities.append(opp)

    def _scan_graph_exclusives(
        self,
        markets: List[Dict[str, Any]],
        opportunities: List[Dict[str, Any]],
    ) -> None:
        market_map = {m["market_id"]: m for m in markets if "market_id" in m}
        for group_ids in self.graph.get_exclusive_groups():
            group = [market_map[mid] for mid in group_ids if mid in market_map]
            if len(group) >= 2:
                opp = self.engine.detect_exclusivity(group)
                if opp is not None:
                    opportunities.append(opp)
