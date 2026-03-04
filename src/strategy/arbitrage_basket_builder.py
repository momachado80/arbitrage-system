"""
Arbitrage Basket Builder
========================

Converts cross-market arbitrage opportunities into tradeable baskets
with capital-weighted component trades.

Pure functions, no side effects, no external dependencies.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional


class ArbitrageBasketBuilder:
    """Build trade baskets from detected arbitrage opportunities."""

    def build(
        self,
        opportunity: Dict[str, Any],
        capital: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Convert an arbitrage opportunity into a weighted trade basket.

        For exclusivity arbs: sell all outcomes proportional to probability.
        For complement arbs: buy underpriced / sell overpriced.
        """
        opp_type = opportunity.get("type", "")
        markets = opportunity.get("markets", [])
        edge = opportunity.get("edge", 0)

        if not markets or edge <= 0:
            return self._empty_basket(opportunity)

        if opp_type == "exclusivity":
            trades = self._build_exclusivity_trades(markets, capital)
        elif opp_type == "complement":
            trades = self._build_complement_trades(opportunity, capital)
        else:
            trades = self._build_exclusivity_trades(markets, capital)

        basket_id = self._basket_id(opportunity)

        return {
            "basket_id": basket_id,
            "type": opp_type,
            "edge": round(edge, 6),
            "capital": round(capital, 2),
            "trades": trades,
            "n_legs": len(trades),
            "timestamp": time.time(),
        }

    def build_many(
        self,
        opportunities: List[Dict[str, Any]],
        capital: float = 100.0,
        max_baskets: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Build baskets for multiple opportunities, sorted by edge descending."""
        baskets = []
        for opp in opportunities[:max_baskets]:
            b = self.build(opp, capital=capital)
            if b.get("trades"):
                baskets.append(b)
        baskets.sort(key=lambda x: -x.get("edge", 0))
        return baskets

    def _build_exclusivity_trades(
        self,
        markets: List[Dict[str, Any]],
        capital: float,
    ) -> List[Dict[str, Any]]:
        total_prob = sum(m.get("prob", 0) for m in markets)
        if total_prob <= 0:
            return []
        trades = []
        for m in markets:
            weight = m.get("prob", 0) / total_prob
            trades.append({
                "market_id": m.get("market_id", ""),
                "side": "SELL",
                "size": round(capital * weight, 4),
                "weight": round(weight, 6),
                "prob": m.get("prob", 0),
                "best_bid": m.get("best_bid"),
                "best_ask": m.get("best_ask"),
            })
        return trades

    def _build_complement_trades(
        self,
        opportunity: Dict[str, Any],
        capital: float,
    ) -> List[Dict[str, Any]]:
        markets = opportunity.get("markets", [])
        direction = opportunity.get("direction", "overpriced")
        if len(markets) < 2:
            return []
        trades = []
        half = capital / 2.0
        for m in markets:
            side = "SELL" if direction == "overpriced" else "BUY"
            trades.append({
                "market_id": m.get("market_id", ""),
                "side": side,
                "size": round(half, 4),
                "weight": 0.5,
                "prob": m.get("prob", 0),
                "best_bid": m.get("best_bid"),
                "best_ask": m.get("best_ask"),
            })
        return trades

    @staticmethod
    def _basket_id(opportunity: Dict[str, Any]) -> str:
        markets = opportunity.get("markets", [])
        ids = sorted(m.get("market_id", "") for m in markets)
        raw = f"{opportunity.get('type', '')}:{'|'.join(ids)}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _empty_basket(opportunity: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "basket_id": "",
            "type": opportunity.get("type", ""),
            "edge": 0,
            "capital": 0,
            "trades": [],
            "n_legs": 0,
            "timestamp": time.time(),
        }
