"""
Cross-Market Arbitrage Engine
=============================

Detects arbitrage opportunities across related Polymarket markets:
  - Mutually exclusive outcomes where P(A)+P(B)+...+P(N) > 1
  - Complement relationships where P(A)+P(not-A) != 1

Pure functions, no side-effects, no external dependencies.
"""

import time
from typing import Any, Dict, List, Optional


class CrossMarketArbitrageEngine:
    """Detect cross-market arbitrage from probability mispricings."""

    def __init__(self, complement_threshold: float = 0.02) -> None:
        self.complement_threshold = complement_threshold

    def detect_exclusivity(
        self,
        markets: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        If a group of mutually exclusive outcomes sums to > 1.0,
        there is a guaranteed arbitrage by selling all outcomes.
        """
        if not markets:
            return None

        total_prob = sum(m.get("prob", 0) for m in markets)
        if total_prob > 1.0:
            edge = round(total_prob - 1.0, 6)
            return {
                "type": "exclusivity",
                "edge": edge,
                "markets": [
                    {
                        "market_id": m.get("market_id", ""),
                        "prob": m.get("prob", 0),
                        "best_bid": m.get("best_bid"),
                        "best_ask": m.get("best_ask"),
                        "category": m.get("category"),
                    }
                    for m in markets
                ],
                "total_prob": round(total_prob, 6),
                "timestamp": time.time(),
            }
        return None

    def detect_complement(
        self,
        market_a: Dict[str, Any],
        market_b: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Two markets that should sum to 1.0 (e.g. A and not-A).
        If |P(A)+P(B) - 1| > threshold, there's a mispricing.
        """
        p = market_a.get("prob", 0) + market_b.get("prob", 0)
        deviation = abs(p - 1.0)
        if deviation > self.complement_threshold:
            return {
                "type": "complement",
                "edge": round(deviation, 6),
                "direction": "overpriced" if p > 1.0 else "underpriced",
                "markets": [
                    {
                        "market_id": m.get("market_id", ""),
                        "prob": m.get("prob", 0),
                        "best_bid": m.get("best_bid"),
                        "best_ask": m.get("best_ask"),
                        "category": m.get("category"),
                    }
                    for m in [market_a, market_b]
                ],
                "combined_prob": round(p, 6),
                "timestamp": time.time(),
            }
        return None

    def detect_all(
        self,
        market_groups: Dict[str, List[Dict[str, Any]]],
        complement_pairs: Optional[List[tuple]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run all detectors over grouped markets and optional complement pairs.
        Returns list of detected opportunities.
        """
        opportunities: List[Dict[str, Any]] = []

        for _group_key, group in market_groups.items():
            opp = self.detect_exclusivity(group)
            if opp is not None:
                opportunities.append(opp)

        if complement_pairs:
            for a, b in complement_pairs:
                opp = self.detect_complement(a, b)
                if opp is not None:
                    opportunities.append(opp)

        opportunities.sort(key=lambda x: -x.get("edge", 0))
        return opportunities
