"""
Fill Engine — Handles taker and maker fill logic.
"""

from typing import Any, Dict, Optional


class FillEngine:
    """Determine fill price and quantity for taker / maker orders."""

    def process_taker(
        self,
        side: str,
        qty: float,
        book: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Immediate fill at best opposite price."""
        if side == "BUY":
            price = book.get("best_ask", book.get("mid", 0))
        else:
            price = book.get("best_bid", book.get("mid", 0))
        return {"qty": qty, "price": price, "filled": True}

    def process_maker(
        self,
        queue_position: float,
        traded_volume: float,
    ) -> bool:
        """Return True if traded volume exceeds our queue position."""
        return traded_volume >= queue_position

    def maker_fill_price(
        self,
        side: str,
        book: Dict[str, Any],
    ) -> float:
        """Limit order price = our side's best."""
        if side == "BUY":
            return book.get("best_bid", book.get("mid", 0))
        return book.get("best_ask", book.get("mid", 0))
