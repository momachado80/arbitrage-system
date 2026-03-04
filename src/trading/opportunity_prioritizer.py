"""
Opportunity Prioritizer
=======================

Ranks trading opportunities by composite profitability score,
incorporating regime conditions.

Pure functions. No I/O. No external dependencies.
"""

from typing import Any, Dict, List


class OpportunityPrioritizer:
    """Score and rank opportunities by expected profitability."""

    def score(
        self,
        opportunity: Dict[str, Any],
        regime_score: float = 0.5,
    ) -> float:
        """
        Composite score = pnl × regime × liquidity × fill_rate.
        All factors default to safe values if missing.
        """
        pnl = opportunity.get("expected_pnl_per_hour", 0.0) or 0.0
        liquidity = opportunity.get("liquidity_score", 0.5) or 0.5
        fill_rate = opportunity.get("fill_rate", 0.5) or 0.5
        regime = max(0.0, min(1.0, regime_score))

        return pnl * regime * liquidity * fill_rate

    def rank(
        self,
        opportunities: List[Dict[str, Any]],
        regime_score: float = 0.5,
        top_n: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Score all opportunities, sort descending, return enriched list.
        If top_n > 0, limit output.
        """
        scored = []
        for opp in opportunities:
            s = self.score(opp, regime_score)
            scored.append({
                "score": round(s, 4),
                "market_id": opp.get("market_id", ""),
                "expected_pnl_per_hour": opp.get("expected_pnl_per_hour", 0),
                "fill_rate": opp.get("fill_rate", 0),
                "liquidity_score": opp.get("liquidity_score", 0),
                "opportunity": opp,
            })
        scored.sort(key=lambda x: -x["score"])
        if top_n > 0:
            scored = scored[:top_n]
        return scored
