"""
Shadow Trader
=============

Simulates real trading decisions in a live environment without sending orders.
Consumes opportunities, simulates execution via a pluggable simulator callable,
and records results for analytics.

No side effects beyond in-memory state. No external dependencies.
"""

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional


class ShadowTrader:
    """Paper-trade against live opportunities using a simulator function."""

    def __init__(
        self,
        simulator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self._simulator = simulator or self._noop_simulator
        self._trades: List[Dict[str, Any]] = []

    def process_opportunity(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a trade for the given opportunity and record result."""
        sim_result = self._simulator(opportunity)

        trade: Dict[str, Any] = {
            "shadow_trade_id": self._make_id(opportunity),
            "timestamp": time.time(),
            "market_id": opportunity.get("market_id", ""),
            "side": opportunity.get("side", ""),
            "expected_pnl": sim_result.get("pnl_bps", 0.0),
            "fill_rate": sim_result.get("fill_rate", 0.0),
            "slippage_bps": sim_result.get("slippage_bps", 0.0),
            "opportunity": opportunity,
        }
        self._trades.append(trade)
        return trade

    def process_many(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of opportunities."""
        return [self.process_opportunity(o) for o in opportunities]

    @property
    def trades(self) -> List[Dict[str, Any]]:
        return list(self._trades)

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    def summary(self) -> Dict[str, Any]:
        """Aggregate shadow trading statistics."""
        if not self._trades:
            return {
                "shadow_trades": 0,
                "total_expected_pnl": 0.0,
                "mean_expected_pnl": 0.0,
                "mean_fill_rate": 0.0,
                "profitable_trades": 0,
                "profitable_pct": 0.0,
            }
        pnls = [t["expected_pnl"] for t in self._trades]
        fills = [t["fill_rate"] for t in self._trades]
        profitable = sum(1 for p in pnls if p > 0)
        n = len(self._trades)
        return {
            "shadow_trades": n,
            "total_expected_pnl": round(sum(pnls), 2),
            "mean_expected_pnl": round(sum(pnls) / n, 2),
            "mean_fill_rate": round(sum(fills) / n, 4),
            "profitable_trades": profitable,
            "profitable_pct": round(profitable / n, 4),
        }

    def reset(self) -> None:
        self._trades.clear()

    @staticmethod
    def _make_id(opportunity: Dict[str, Any]) -> str:
        raw = f"{time.monotonic_ns()}:{opportunity.get('market_id', '')}:{id(opportunity)}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _noop_simulator(opportunity: Dict[str, Any]) -> Dict[str, Any]:
        return {"pnl_bps": 0.0, "fill_rate": 0.0, "slippage_bps": 0.0}
