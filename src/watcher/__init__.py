"""
Watcher module — Polymarket CLOB WebSocket client.

Componentes:
- PolymarketClient: WebSocket resiliente com reconexão automática
- LocalOrderBook:   Espelho local do order book por token_id
- WatcherBridge:    Ponte async → sync via Queue
- MarketSnapshot:   Dataclass imutável (topo do book)
- MarketTrade:      Dataclass imutável (trade em tempo real)
"""

from src.watcher.order_book import LocalOrderBook, BookCorruptionError
from src.watcher.bridge import WatcherBridge, MarketSnapshot, MarketTrade
from src.watcher.client import PolymarketClient

__all__ = [
    "PolymarketClient",
    "LocalOrderBook",
    "BookCorruptionError",
    "WatcherBridge",
    "MarketSnapshot",
    "MarketTrade",
]
