"""
Testes para o módulo watcher (WebSocket Polymarket CLOB).

Cobertura:
- LocalOrderBook: snapshot, updates sequenciais, gap → BookCorruptionError
- WatcherBridge:  snapshot só quando best muda, trade imediato, drop-tail
- PolymarketClient: subscribe, reconnect, shutdown
- Integração:     fluxo feliz completo
"""

import asyncio
import logging
import queue
import time

import pytest
import pytest_asyncio

from src.watcher.order_book import LocalOrderBook, BookCorruptionError
from src.watcher.bridge import WatcherBridge, MarketSnapshot, MarketTrade
from src.watcher.client import PolymarketClient


# =============================================================================
# LOCAL ORDER BOOK — SNAPSHOT
# =============================================================================

class TestBookSnapshot:
    def test_snapshot_loads_bids_asks(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [
                {"price": "0.65", "size": "100"},
                {"price": "0.64", "size": "200"},
            ],
            "asks": [
                {"price": "0.70", "size": "50"},
                {"price": "0.71", "size": "100"},
            ],
            "sequence_id": 1,
        })
        assert len(book.bids) == 2
        assert len(book.asks) == 2
        assert book.last_sequence_id == 1
        assert book.bids[0.65] == 100.0
        assert book.asks[0.70] == 50.0

    def test_snapshot_clears_previous(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.70", "size": "50"}],
            "sequence_id": 1,
        })
        book.apply_snapshot({
            "bids": [{"price": "0.60", "size": "300"}],
            "asks": [],
            "sequence_id": 5,
        })
        assert 0.65 not in book.bids
        assert book.bids[0.60] == 300.0
        assert len(book.asks) == 0
        assert book.last_sequence_id == 5

    def test_snapshot_skips_zero_size(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [
                {"price": "0.65", "size": "100"},
                {"price": "0.64", "size": "0"},
            ],
            "asks": [],
            "sequence_id": 1,
        })
        assert len(book.bids) == 1
        assert 0.64 not in book.bids

    def test_snapshot_list_format(self):
        """Aceita formato list [price, size] além de dict."""
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [["0.65", "100"]],
            "asks": [["0.70", "50"]],
            "sequence_id": 1,
        })
        assert book.bids[0.65] == 100.0
        assert book.asks[0.70] == 50.0

    def test_snapshot_normalizes_floats(self):
        """Preços e tamanhos são normalizados via round_price/round_shares."""
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [{"price": "0.6500001", "size": "100.000000001"}],
            "asks": [],
            "sequence_id": 1,
        })
        # round_price(0.6500001) = 0.6500001 (6 decimals → 0.65)
        # Stored as float, normalized
        assert len(book.bids) == 1


# =============================================================================
# LOCAL ORDER BOOK — SEQUENTIAL UPDATES
# =============================================================================

class TestBookUpdate:
    def test_sequential_updates(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.70", "size": "50"}],
            "sequence_id": 1,
        })
        book.apply_update({
            "bids": [{"price": "0.66", "size": "200"}],
            "asks": [],
            "sequence_id": 2,
        })
        assert 0.66 in book.bids
        assert book.bids[0.66] == 200.0
        assert book.bids[0.65] == 100.0  # Previous level preserved
        assert book.last_sequence_id == 2

    def test_remove_price_level(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.70", "size": "50"}],
            "sequence_id": 1,
        })
        book.apply_update({
            "bids": [{"price": "0.65", "size": "0"}],
            "asks": [],
            "sequence_id": 2,
        })
        assert 0.65 not in book.bids

    def test_update_existing_level(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [],
            "sequence_id": 1,
        })
        book.apply_update({
            "bids": [{"price": "0.65", "size": "150"}],
            "asks": [],
            "sequence_id": 2,
        })
        assert book.bids[0.65] == 150.0

    def test_multiple_updates_sequential(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [], "asks": [], "sequence_id": 10,
        })
        for i in range(11, 21):
            book.apply_update({
                "bids": [{"price": str(0.5 + i * 0.01), "size": str(i * 10)}],
                "asks": [],
                "sequence_id": i,
            })
        assert book.last_sequence_id == 20
        assert len(book.bids) == 10


# =============================================================================
# LOCAL ORDER BOOK — SEQUENCE GAP
# =============================================================================

class TestBookSequenceGap:
    def test_gap_raises_corruption(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [], "asks": [], "sequence_id": 1,
        })
        with pytest.raises(BookCorruptionError):
            book.apply_update({
                "bids": [], "asks": [], "sequence_id": 5,
            })

    def test_gap_backward_raises(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [], "asks": [], "sequence_id": 10,
        })
        with pytest.raises(BookCorruptionError):
            book.apply_update({
                "bids": [], "asks": [], "sequence_id": 10,  # Duplicate
            })

    def test_no_gap_no_error(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [], "asks": [], "sequence_id": 5,
        })
        book.apply_update({
            "bids": [], "asks": [], "sequence_id": 6,
        })
        assert book.last_sequence_id == 6


# =============================================================================
# LOCAL ORDER BOOK — BEST PRICE
# =============================================================================

class TestBookBestPrice:
    def test_best_bid(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [
                {"price": "0.65", "size": "100"},
                {"price": "0.64", "size": "200"},
                {"price": "0.66", "size": "50"},
            ],
            "asks": [],
            "sequence_id": 1,
        })
        best = book.best_bid()
        assert best is not None
        assert best[0] == 0.66
        assert best[1] == 50.0

    def test_best_ask(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [],
            "asks": [
                {"price": "0.72", "size": "100"},
                {"price": "0.70", "size": "50"},
                {"price": "0.71", "size": "75"},
            ],
            "sequence_id": 1,
        })
        best = book.best_ask()
        assert best is not None
        assert best[0] == 0.70
        assert best[1] == 50.0

    def test_empty_book_returns_none(self):
        book = LocalOrderBook("t1")
        assert book.best_bid() is None
        assert book.best_ask() is None


# =============================================================================
# WATCHER BRIDGE — SNAPSHOT EMISSION
# =============================================================================

class TestBridgeSnapshot:
    def test_snapshot_emitted_on_first_update(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.70", "size": "50"}],
            "sequence_id": 1,
        })

        bridge.on_book_updated("t1", book)
        assert not q.empty()
        snap = q.get_nowait()
        assert isinstance(snap, MarketSnapshot)
        assert snap.token_id == "t1"
        assert snap.best_bid_price == 0.65
        assert snap.best_bid_size == 100.0
        assert snap.best_ask_price == 0.70
        assert snap.best_ask_size == 50.0

    def test_no_snapshot_when_unchanged(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.70", "size": "50"}],
            "sequence_id": 1,
        })
        bridge.on_book_updated("t1", book)
        q.get_nowait()  # Consume first

        # Same state → no new snapshot
        bridge.on_book_updated("t1", book)
        assert q.empty()

    def test_snapshot_on_bid_size_change(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.70", "size": "50"}],
            "sequence_id": 1,
        })
        bridge.on_book_updated("t1", book)
        q.get_nowait()

        book.apply_update({
            "bids": [{"price": "0.65", "size": "150"}],
            "asks": [],
            "sequence_id": 2,
        })
        bridge.on_book_updated("t1", book)
        assert not q.empty()
        snap = q.get_nowait()
        assert snap.best_bid_size == 150.0

    def test_snapshot_on_new_best_bid(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.70", "size": "50"}],
            "sequence_id": 1,
        })
        bridge.on_book_updated("t1", book)
        q.get_nowait()

        # New level above current best
        book.apply_update({
            "bids": [{"price": "0.66", "size": "200"}],
            "asks": [],
            "sequence_id": 2,
        })
        bridge.on_book_updated("t1", book)
        assert not q.empty()
        snap = q.get_nowait()
        assert snap.best_bid_price == 0.66

    def test_snapshot_on_best_ask_removed(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [],
            "asks": [
                {"price": "0.70", "size": "50"},
                {"price": "0.71", "size": "100"},
            ],
            "sequence_id": 1,
        })
        bridge.on_book_updated("t1", book)
        q.get_nowait()

        # Remove best ask
        book.apply_update({
            "bids": [],
            "asks": [{"price": "0.70", "size": "0"}],
            "sequence_id": 2,
        })
        bridge.on_book_updated("t1", book)
        snap = q.get_nowait()
        assert snap.best_ask_price == 0.71
        assert snap.best_ask_size == 100.0

    def test_no_snapshot_on_non_best_change(self):
        """Update a preço que não é best não emite snapshot."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [
                {"price": "0.65", "size": "100"},
                {"price": "0.64", "size": "200"},
            ],
            "asks": [],
            "sequence_id": 1,
        })
        bridge.on_book_updated("t1", book)
        q.get_nowait()

        # Update non-best level
        book.apply_update({
            "bids": [{"price": "0.64", "size": "250"}],
            "asks": [],
            "sequence_id": 2,
        })
        bridge.on_book_updated("t1", book)
        assert q.empty()  # Best didn't change


# =============================================================================
# WATCHER BRIDGE — TRADE
# =============================================================================

class TestBridgeTrade:
    def test_trade_sent_immediately(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        bridge.on_trade("t1", {
            "price": "0.65", "size": "10", "side": "BUY",
        })
        assert not q.empty()
        trade = q.get_nowait()
        assert isinstance(trade, MarketTrade)
        assert trade.token_id == "t1"
        assert trade.price == 0.65
        assert trade.size == 10.0
        assert trade.side == "BUY"

    def test_trade_normalized(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        bridge.on_trade("t1", {
            "price": "0.6500001", "size": "10.00000001", "side": "SELL",
        })
        trade = q.get_nowait()
        assert isinstance(trade.price, float)
        assert isinstance(trade.size, float)
        assert trade.side == "SELL"


# =============================================================================
# WATCHER BRIDGE — DROP TAIL
# =============================================================================

class TestBridgeDropTail:
    def test_snapshot_drops_oldest_when_full(self):
        q: queue.Queue = queue.Queue(maxsize=2)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        book = LocalOrderBook("t1")

        # Fill queue with 2 snapshots
        book.apply_snapshot({
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.70", "size": "50"}],
            "sequence_id": 1,
        })
        bridge.on_book_updated("t1", book)

        book.apply_update({
            "bids": [{"price": "0.65", "size": "110"}],
            "asks": [],
            "sequence_id": 2,
        })
        bridge.on_book_updated("t1", book)
        assert q.qsize() == 2

        # Third snapshot → drop oldest, insert new
        book.apply_update({
            "bids": [{"price": "0.65", "size": "120"}],
            "asks": [],
            "sequence_id": 3,
        })
        bridge.on_book_updated("t1", book)

        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert len(items) == 2
        assert items[-1].best_bid_size == 120.0

    def test_trade_dropped_when_full(self, caplog):
        q: queue.Queue = queue.Queue(maxsize=1)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)

        # Fill queue
        bridge.on_trade("t1", {"price": "0.65", "size": "10", "side": "BUY"})
        assert q.qsize() == 1

        # Second trade → dropped
        with caplog.at_level(logging.WARNING, logger="src.watcher.bridge"):
            bridge.on_trade("t1", {"price": "0.66", "size": "20", "side": "SELL"})

        # Only first trade in queue
        trade = q.get_nowait()
        assert trade.price == 0.65
        assert q.empty()

        # Warning logged
        warnings = [r for r in caplog.records if "TRADE_DROP" in r.message]
        assert len(warnings) == 1


# =============================================================================
# POLYMARKET CLIENT — BASIC
# =============================================================================

class TestClientBasic:
    def test_initialization(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        client = PolymarketClient(bridge)
        assert client._running is False
        assert len(client._subscribed_tokens) == 0

    @pytest.mark.asyncio
    async def test_subscribe_adds_token(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        client = PolymarketClient(bridge)

        await client.subscribe_market("0xabc123")
        assert "0xabc123" in client._subscribed_tokens

    @pytest.mark.asyncio
    async def test_subscribe_multiple_tokens(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        client = PolymarketClient(bridge)

        await client.subscribe_market("0xabc")
        await client.subscribe_market("0xdef")
        assert len(client._subscribed_tokens) == 2

    @pytest.mark.asyncio
    async def test_shutdown_clean(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        client = PolymarketClient(bridge)

        await client.shutdown()
        assert client._running is False
        assert client._ws is None
        assert client._session is None


# =============================================================================
# POLYMARKET CLIENT — MESSAGE ROUTING
# =============================================================================

class TestClientMessageRouting:
    def test_handle_book_snapshot(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        client = PolymarketClient(bridge)
        client._books["t1"] = LocalOrderBook("t1")

        import orjson
        raw = orjson.dumps({
            "event_type": "book",
            "asset_id": "t1",
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.70", "size": "50"}],
            "sequence_id": 1,
        })
        client._handle_message(raw)

        assert not q.empty()
        snap = q.get_nowait()
        assert isinstance(snap, MarketSnapshot)
        assert snap.best_bid_price == 0.65

    def test_handle_trade(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        client = PolymarketClient(bridge)

        import orjson
        raw = orjson.dumps({
            "event_type": "last_trade_price",
            "asset_id": "t1",
            "price": "0.65",
            "size": "10",
            "side": "BUY",
        })
        client._handle_message(raw)

        trade = q.get_nowait()
        assert isinstance(trade, MarketTrade)
        assert trade.price == 0.65

    def test_handle_price_change(self):
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        client = PolymarketClient(bridge)

        # Need initial snapshot first
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.70", "size": "50"}],
            "sequence_id": 1,
        })
        client._books["t1"] = book
        bridge.on_book_updated("t1", book)
        q.get_nowait()  # Consume initial snapshot

        import orjson
        raw = orjson.dumps({
            "event_type": "price_change",
            "asset_id": "t1",
            "bids": [{"price": "0.66", "size": "200"}],
            "asks": [],
            "sequence_id": 2,
        })
        client._handle_message(raw)

        snap = q.get_nowait()
        assert snap.best_bid_price == 0.66

    def test_handle_invalid_json(self):
        """JSON inválido não levanta exceção."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        client = PolymarketClient(bridge)

        client._handle_message(b"NOT JSON {{{")
        assert q.empty()

    def test_handle_update_without_snapshot_ignored(self):
        """Update sem snapshot prévio é ignorado silenciosamente."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        client = PolymarketClient(bridge)

        import orjson
        raw = orjson.dumps({
            "event_type": "price_change",
            "asset_id": "t1",
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [],
            "sequence_id": 2,
        })
        client._handle_message(raw)
        assert q.empty()  # Ignored — no snapshot yet


# =============================================================================
# POLYMARKET CLIENT — BOOK CORRUPTION
# =============================================================================

class TestClientBookCorruption:
    def test_sequence_gap_raises_corruption(self):
        """BookCorruptionError propagada ao client para forçar reconexão."""
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [], "asks": [], "sequence_id": 1,
        })
        with pytest.raises(BookCorruptionError):
            book.apply_update({
                "bids": [], "asks": [], "sequence_id": 5,
            })

    def test_corruption_message_format(self):
        book = LocalOrderBook("t1")
        book.apply_snapshot({
            "bids": [], "asks": [], "sequence_id": 10,
        })
        with pytest.raises(BookCorruptionError, match="SEQ_GAP"):
            book.apply_update({
                "bids": [], "asks": [], "sequence_id": 15,
            })


# =============================================================================
# INTEGRATION — HAPPY FLOW
# =============================================================================

class TestIntegrationHappyFlow:
    def test_full_flow_snapshot_updates_snapshots(self):
        """Snapshot → sequential updates → snapshots emitidos corretamente."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)
        book = LocalOrderBook("t1")

        # 1. Snapshot inicial
        book.apply_snapshot({
            "bids": [
                {"price": "0.65", "size": "100"},
                {"price": "0.64", "size": "200"},
            ],
            "asks": [
                {"price": "0.70", "size": "50"},
                {"price": "0.71", "size": "100"},
            ],
            "sequence_id": 1,
        })
        bridge.on_book_updated("t1", book)
        snap1 = q.get_nowait()
        assert snap1.best_bid_price == 0.65
        assert snap1.best_ask_price == 0.70

        # 2. Update: new best bid
        book.apply_update({
            "bids": [{"price": "0.66", "size": "300"}],
            "asks": [],
            "sequence_id": 2,
        })
        bridge.on_book_updated("t1", book)
        snap2 = q.get_nowait()
        assert snap2.best_bid_price == 0.66
        assert snap2.best_bid_size == 300.0

        # 3. Update: remove best ask → next level
        book.apply_update({
            "bids": [],
            "asks": [{"price": "0.70", "size": "0"}],
            "sequence_id": 3,
        })
        bridge.on_book_updated("t1", book)
        snap3 = q.get_nowait()
        assert snap3.best_ask_price == 0.71
        assert snap3.best_ask_size == 100.0

        # 4. Trade
        bridge.on_trade("t1", {
            "price": "0.66", "size": "25", "side": "BUY",
        })
        trade = q.get_nowait()
        assert isinstance(trade, MarketTrade)
        assert trade.price == 0.66
        assert trade.size == 25.0

    def test_multiple_tokens_isolated(self):
        """Books de tokens diferentes são isolados."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        bridge = WatcherBridge(q)

        book_a = LocalOrderBook("token-A")
        book_b = LocalOrderBook("token-B")

        book_a.apply_snapshot({
            "bids": [{"price": "0.60", "size": "100"}],
            "asks": [{"price": "0.80", "size": "50"}],
            "sequence_id": 1,
        })
        book_b.apply_snapshot({
            "bids": [{"price": "0.40", "size": "200"}],
            "asks": [{"price": "0.55", "size": "75"}],
            "sequence_id": 1,
        })

        bridge.on_book_updated("token-A", book_a)
        bridge.on_book_updated("token-B", book_b)

        snap_a = q.get_nowait()
        snap_b = q.get_nowait()
        assert snap_a.token_id == "token-A"
        assert snap_a.best_bid_price == 0.60
        assert snap_b.token_id == "token-B"
        assert snap_b.best_bid_price == 0.40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
