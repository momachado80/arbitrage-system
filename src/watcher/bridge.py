"""
Watcher Bridge — Async to Sync
=================================

Ponte entre o WebSocket async e o pipeline sync via queue.Queue.

DATACLASSES IMUTÁVEIS:
- MarketSnapshot: topo do order book (best bid/ask)
- MarketTrade:    trade em tempo real

ESTRATÉGIA DE FILA:
- Snapshots: só enviados quando best bid/ask muda (sem redundância)
- Trades:    enviados imediatamente (prioritários)
- Drop-tail quando fila cheia:
    - Snapshot: remove item mais antigo, insere novo
    - Trade: tenta inserir; se cheia, descarta + log [WATCHER:TRADE_DROP]
- Nunca bloquear (apenas put_nowait)
"""

import logging
import queue
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.utils.precision import round_price, round_shares

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketSnapshot:
    """Snapshot imutável do topo do order book."""
    timestamp: float         # epoch UTC
    token_id: str
    best_bid_price: Optional[float]
    best_bid_size: Optional[float]
    best_ask_price: Optional[float]
    best_ask_size: Optional[float]


@dataclass(frozen=True)
class MarketTrade:
    """Trade imutável recebido em tempo real."""
    timestamp: float         # epoch UTC
    token_id: str
    price: float
    size: float
    side: str


# Tipo interno para cache: (bid_price, bid_size, ask_price, ask_size)
_BBA = Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]


class WatcherBridge:
    """Ponte async → sync via queue.Queue."""

    def __init__(self, out_queue: queue.Queue) -> None:  # type: ignore[type-arg]
        self._queue = out_queue
        self._last_bba: Dict[str, _BBA] = {}  # token_id → last best bid/ask

    # ------------------------------------------------------------------
    # Book update → MarketSnapshot (only if changed)
    # ------------------------------------------------------------------

    def on_book_updated(self, token_id: str, book: Any) -> None:
        """
        Chamado após cada snapshot/update do book.

        Só envia MarketSnapshot se best bid/ask mudou.
        """
        best_bid = book.best_bid()
        best_ask = book.best_ask()

        bid_price = best_bid[0] if best_bid else None
        bid_size = best_bid[1] if best_bid else None
        ask_price = best_ask[0] if best_ask else None
        ask_size = best_ask[1] if best_ask else None

        current: _BBA = (bid_price, bid_size, ask_price, ask_size)
        prev = self._last_bba.get(token_id)

        if current == prev:
            return  # Sem mudança — não enviar snapshot redundante

        self._last_bba[token_id] = current

        snap = MarketSnapshot(
            timestamp=time.time(),
            token_id=token_id,
            best_bid_price=bid_price,
            best_bid_size=bid_size,
            best_ask_price=ask_price,
            best_ask_size=ask_size,
        )
        self._enqueue_snapshot(snap)

    # ------------------------------------------------------------------
    # Trade → MarketTrade (immediately)
    # ------------------------------------------------------------------

    def on_trade(self, token_id: str, data: dict) -> None:
        """
        Chamado ao receber trade. Envia MarketTrade imediatamente.
        """
        trade = MarketTrade(
            timestamp=time.time(),
            token_id=token_id,
            price=round_price(float(data.get("price", "0"))),
            size=round_shares(float(data.get("size", "0"))),
            side=str(data.get("side", "")),
        )
        self._enqueue_trade(trade)

    # ------------------------------------------------------------------
    # Queue strategy
    # ------------------------------------------------------------------

    def _enqueue_snapshot(self, snap: MarketSnapshot) -> None:
        """Drop-tail: se cheia, remove item mais antigo e insere novo."""
        try:
            self._queue.put_nowait(snap)
        except queue.Full:
            try:
                self._queue.get_nowait()  # Drop oldest
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(snap)
            except queue.Full:
                pass  # Shouldn't happen after get

    def _enqueue_trade(self, trade: MarketTrade) -> None:
        """Trades prioritários: tenta inserir, descarta se cheia."""
        try:
            self._queue.put_nowait(trade)
        except queue.Full:
            logger.warning(
                f"[WATCHER:TRADE_DROP] token={trade.token_id} "
                f"price={trade.price} size={trade.size}"
            )
