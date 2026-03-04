"""
Market Regime Detection Engine
==============================

Evaluates whether current market conditions are favorable for trading
using metrics already exposed in /analytics.

O(1) computation — uses pre-aggregated analytics, not raw episodes.
Pure function, no side effects, no external dependencies.
"""

from typing import Any, Dict


class MarketRegimeDetector:
    """Classify market regime from analytics snapshot."""

    def __init__(
        self,
        w_persistence: float = 0.4,
        w_opportunity: float = 0.3,
        w_liquidity: float = 0.2,
        w_competition: float = 0.3,
        high_threshold: float = 0.65,
        hostile_threshold: float = 0.35,
        opp_density_cap: float = 100.0,
    ) -> None:
        self.w_persistence = w_persistence
        self.w_opportunity = w_opportunity
        self.w_liquidity = w_liquidity
        self.w_competition = w_competition
        self.high_threshold = high_threshold
        self.hostile_threshold = hostile_threshold
        self.opp_density_cap = opp_density_cap

    def compute(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute regime from an analytics dict (same shape as /analytics response).

        Returns dict with regime_score, state, and component breakdowns.
        """
        if not analytics:
            return self._default()

        edge_persistence = self._edge_persistence(analytics)
        opportunity_density = self._opportunity_density(analytics)
        liquidity_score = self._liquidity_score(analytics)
        competition = self._competition_score(analytics)

        raw = (
            self.w_persistence * edge_persistence
            + self.w_opportunity * opportunity_density
            + self.w_liquidity * liquidity_score
            - self.w_competition * competition
        )
        regime_score = max(0.0, min(1.0, raw))

        if regime_score > self.high_threshold:
            state = "high_edge"
        elif regime_score < self.hostile_threshold:
            state = "hostile"
        else:
            state = "neutral"

        return {
            "regime_score": round(regime_score, 4),
            "state": state,
            "components": {
                "edge_persistence": round(edge_persistence, 4),
                "opportunity_density": round(opportunity_density, 4),
                "liquidity_score": round(liquidity_score, 4),
                "competition": round(competition, 4),
            },
        }

    def _edge_persistence(self, analytics: Dict[str, Any]) -> float:
        survival = analytics.get("edge_survival_curve", {})
        if not survival:
            return 0.0
        s500 = survival.get("500ms", 0.0) or 0.0
        s2000 = survival.get("2000ms", 0.0) or 0.0
        return (s500 + s2000) / 2.0

    def _opportunity_density(self, analytics: Dict[str, Any]) -> float:
        agg = analytics.get("aggregates", {})
        eph = agg.get("episodes_per_hour", 0.0) if isinstance(agg, dict) else 0.0
        if not eph or eph <= 0:
            eph = analytics.get("episodes_per_hour", 0.0) or 0.0
        if self.opp_density_cap <= 0:
            return 0.0
        return min(1.0, max(0.0, eph / self.opp_density_cap))

    def _liquidity_score(self, analytics: Dict[str, Any]) -> float:
        agg = analytics.get("aggregates", {})
        spread = 0.0
        bid_sz = 0.0
        ask_sz = 0.0
        if isinstance(agg, dict):
            spread = agg.get("spread_bps_mean", 0.0) or agg.get("spread", 0.0) or 0.0
            bid_sz = agg.get("bid_size_mean", 0.0) or 0.0
            ask_sz = agg.get("ask_size_mean", 0.0) or 0.0
        if not spread:
            spread = analytics.get("spread", 0.01) or 0.01

        base = 1.0 / (1.0 + abs(spread))
        depth = (bid_sz + ask_sz) / 2.0
        if depth > 0:
            depth_factor = min(1.0, depth / 100.0)
            return base * (0.5 + 0.5 * depth_factor)
        return base

    def _competition_score(self, analytics: Dict[str, Any]) -> float:
        hazard = analytics.get("edge_hazard_curve", {})
        if not hazard:
            return 0.0
        h_500 = 0.0
        for key in ("250-500ms", "500-1000ms"):
            v = hazard.get(key)
            if v is not None:
                h_500 = max(h_500, v)
        return min(1.0, max(0.0, h_500))

    @staticmethod
    def _default() -> Dict[str, Any]:
        return {
            "regime_score": 0.0,
            "state": "hostile",
            "components": {
                "edge_persistence": 0.0,
                "opportunity_density": 0.0,
                "liquidity_score": 0.0,
                "competition": 0.0,
            },
        }
