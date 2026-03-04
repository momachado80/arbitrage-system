"""
Risk Engine
===========

Prevents excessive risk exposure by enforcing position limits,
total inventory caps, and drawdown thresholds.

Pure in-memory state. No I/O. No external dependencies.
"""

from typing import Any, Dict


class RiskEngine:
    """Enforce trading risk limits."""

    def __init__(
        self,
        max_position_per_market: float = 1000.0,
        max_total_inventory: float = 10000.0,
        max_drawdown: float = -500.0,
    ) -> None:
        self.max_position_per_market = max_position_per_market
        self.max_total_inventory = max_total_inventory
        self.max_drawdown = max_drawdown
        self.inventory: Dict[str, float] = {}
        self.pnl: float = 0.0

    def allow_trade(self, trade: Dict[str, Any]) -> bool:
        """Check whether a proposed trade passes all risk limits."""
        market = trade.get("market_id", "")
        size = trade.get("size", 0.0)

        current_pos = self.inventory.get(market, 0.0)
        if abs(current_pos + size) > self.max_position_per_market:
            return False

        total_inv = sum(abs(v) for v in self.inventory.values())
        if total_inv + abs(size) > self.max_total_inventory:
            return False

        if self.pnl <= self.max_drawdown:
            return False

        return True

    def update(self, trade: Dict[str, Any], pnl: float = 0.0) -> None:
        """Record a filled trade and its PnL."""
        market = trade.get("market_id", "")
        size = trade.get("size", 0.0)
        self.inventory[market] = self.inventory.get(market, 0.0) + size
        self.pnl += pnl

    def state(self) -> Dict[str, Any]:
        """Current risk engine state for analytics."""
        return {
            "inventory": dict(self.inventory),
            "pnl": round(self.pnl, 2),
            "total_inventory": round(sum(abs(v) for v in self.inventory.values()), 2),
            "markets_with_position": len([v for v in self.inventory.values() if v != 0]),
            "drawdown_remaining": round(self.pnl - self.max_drawdown, 2),
        }

    def reset(self) -> None:
        self.inventory.clear()
        self.pnl = 0.0
