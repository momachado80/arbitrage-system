"""
Tests for AUTO_UNIVERSE feature — auto-discovery and subscription.
"""

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.market.universe_manager import (
    MarketInfo,
    MarketUniverseManager,
    UniverseSnapshot,
)


# ======================================================================
# MarketUniverseManager — core logic
# ======================================================================


class TestMarketUniverseManagerInit:
    """Test initialization and env var reading."""

    def test_defaults(self):
        mgr = MarketUniverseManager()
        assert mgr._min_volume > 0
        assert mgr._min_liquidity > 0
        assert mgr._max_spread > 0
        assert mgr._top_n > 0
        assert mgr._interval > 0

    def test_env_max_markets(self):
        with patch.dict(os.environ, {"MAX_MARKETS": "25"}, clear=False):
            mgr = MarketUniverseManager()
            assert mgr._top_n == 25

    def test_env_refresh_seconds(self):
        with patch.dict(os.environ, {"UNIVERSE_REFRESH_SECONDS": "600"}, clear=False):
            mgr = MarketUniverseManager()
            assert mgr._interval == 600

    def test_env_http_timeout(self):
        with patch.dict(os.environ, {"UNIVERSE_HTTP_TIMEOUT": "3"}, clear=False):
            mgr = MarketUniverseManager()
            assert mgr._timeout == 3.0


class TestDiscoverFromRaw:
    """Test market filtering and ranking."""

    def _make_market(
        self,
        question="Will X happen?",
        volume=5000,
        yes_id="tok_yes",
        no_id="tok_no",
        condition_id="cond_1",
        liquidity=None,
    ):
        return {
            "question": question,
            "volume24hr": volume,
            "conditionId": condition_id,
            "liquidity": liquidity,
            "tokens": [
                {"outcome": "Yes", "token_id": yes_id},
                {"outcome": "No", "token_id": no_id},
            ],
        }

    def test_filters_low_volume(self):
        mgr = MarketUniverseManager(min_volume_usd=1000)
        raw = [self._make_market(volume=500)]
        result = mgr._discover_from_raw(raw)
        assert len(result) == 0

    def test_passes_high_volume(self):
        mgr = MarketUniverseManager(min_volume_usd=1000)
        raw = [self._make_market(volume=5000)]
        result = mgr._discover_from_raw(raw)
        assert len(result) > 0

    def test_respects_top_n(self):
        mgr = MarketUniverseManager(top_n=2, min_volume_usd=100, min_liquidity_usd=10)
        raw = [
            self._make_market(volume=5000, yes_id=f"y{i}", no_id=f"n{i}")
            for i in range(10)
        ]
        result = mgr._discover_from_raw(raw)
        assert len(result) <= 2

    def test_extracts_question(self):
        mgr = MarketUniverseManager(min_volume_usd=100, min_liquidity_usd=10)
        raw = [self._make_market(question="Will BTC hit 100k?", volume=5000)]
        result = mgr._discover_from_raw(raw)
        assert any(m.question == "Will BTC hit 100k?" for m in result)

    def test_uses_liquidity_field(self):
        mgr = MarketUniverseManager(min_volume_usd=100, min_liquidity_usd=1000)
        raw = [self._make_market(volume=500, liquidity=2000)]
        result = mgr._discover_from_raw(raw)
        assert len(result) > 0

    def test_filters_low_liquidity(self):
        mgr = MarketUniverseManager(min_volume_usd=100, min_liquidity_usd=1000)
        raw = [self._make_market(volume=500, liquidity=100)]
        result = mgr._discover_from_raw(raw)
        assert len(result) == 0

    def test_no_tokens_skipped(self):
        mgr = MarketUniverseManager(min_volume_usd=100, min_liquidity_usd=10)
        raw = [{"question": "Test", "volume24hr": 5000, "tokens": []}]
        result = mgr._discover_from_raw(raw)
        assert len(result) == 0

    def test_clob_fallback(self):
        mgr = MarketUniverseManager(min_volume_usd=100, min_liquidity_usd=10)
        raw = [{
            "question": "Test",
            "volume24hr": 5000,
            "tokens": [],
            "clobTokenIds": '["tok_a", "tok_b"]',
        }]
        result = mgr._discover_from_raw(raw)
        assert any(m.token_id == "tok_a" for m in result)


class TestUniverseSnapshot:
    """Test snapshot and market info methods."""

    def test_get_snapshot_empty(self):
        mgr = MarketUniverseManager()
        snap = mgr.get_snapshot()
        assert snap.total_markets_found == 0
        assert snap.token_ids == []
        assert snap.error is None

    def test_get_market_infos_empty(self):
        mgr = MarketUniverseManager()
        infos = mgr.get_market_infos()
        assert infos == []

    def test_get_market_infos_populated(self):
        mgr = MarketUniverseManager()
        mgr._market_infos = [
            MarketInfo("tok1", "c1", 5000, 500, 2.0, "yes", "Will X?"),
            MarketInfo("tok2", "c2", 3000, 300, 1.5, "no", "Will Y?"),
        ]
        infos = mgr.get_market_infos()
        assert len(infos) == 2
        assert infos[0]["question"] == "Will X?"
        assert "volume_24h" in infos[0]


class TestRunUpdate:
    """Test the _run_update method with mocked HTTP."""

    def test_run_update_success(self):
        mgr = MarketUniverseManager(
            top_n=5,
            min_volume_usd=100,
            min_liquidity_usd=10,
        )
        subscribed = []
        mgr._subscribe_cb = lambda tid: subscribed.append(tid)

        raw_markets = [
            {
                "question": f"Market {i}",
                "volume24hr": 5000 - i * 100,
                "conditionId": f"cond_{i}",
                "tokens": [
                    {"outcome": "Yes", "token_id": f"yes_{i}"},
                    {"outcome": "No", "token_id": f"no_{i}"},
                ],
            }
            for i in range(10)
        ]

        with patch.object(mgr, "_fetch_markets_sync", return_value=raw_markets):
            mgr._run_update()

        snap = mgr.get_snapshot()
        assert snap.total_markets_found == 10
        assert snap.total_markets_subscribed > 0
        assert snap.error is None
        assert len(subscribed) > 0

    def test_run_update_empty_response(self):
        mgr = MarketUniverseManager()
        with patch.object(mgr, "_fetch_markets_sync", return_value=[]):
            mgr._run_update()
        snap = mgr.get_snapshot()
        assert snap.error is not None

    def test_run_update_tracks_removals(self):
        mgr = MarketUniverseManager(
            top_n=5,
            min_volume_usd=100,
            min_liquidity_usd=10,
        )
        unsubscribed = []
        mgr._unsubscribe_cb = lambda tid: unsubscribed.append(tid)
        mgr._subscribe_cb = lambda tid: None

        # First update with markets
        raw1 = [{
            "question": "M1",
            "volume24hr": 5000,
            "conditionId": "c1",
            "tokens": [
                {"outcome": "Yes", "token_id": "yes_1"},
                {"outcome": "No", "token_id": "no_1"},
            ],
        }]
        with patch.object(mgr, "_fetch_markets_sync", return_value=raw1):
            mgr._run_update()

        # Second update with different markets
        raw2 = [{
            "question": "M2",
            "volume24hr": 5000,
            "conditionId": "c2",
            "tokens": [
                {"outcome": "Yes", "token_id": "yes_2"},
                {"outcome": "No", "token_id": "no_2"},
            ],
        }]
        with patch.object(mgr, "_fetch_markets_sync", return_value=raw2):
            mgr._run_update()

        assert len(unsubscribed) > 0


class TestStartStop:
    """Test background thread lifecycle."""

    def test_start_stop(self):
        mgr = MarketUniverseManager(update_interval_sec=1)
        with patch.object(mgr, "_run_update"):
            mgr.start()
            assert mgr._running
            assert mgr._thread is not None
            time.sleep(0.1)
            mgr.stop()
            assert not mgr._running

    def test_run_once(self):
        mgr = MarketUniverseManager(
            top_n=5,
            min_volume_usd=100,
            min_liquidity_usd=10,
        )
        raw = [{
            "question": "Test",
            "volume24hr": 5000,
            "conditionId": "c1",
            "tokens": [
                {"outcome": "Yes", "token_id": "yes_1"},
                {"outcome": "No", "token_id": "no_1"},
            ],
        }]
        with patch.object(mgr, "_fetch_markets_sync", return_value=raw):
            token_ids = mgr.run_once()
        assert len(token_ids) > 0
        snap = mgr.get_snapshot()
        assert snap.total_markets_subscribed > 0


# ======================================================================
# Metrics integration
# ======================================================================


class TestMetricsIntegration:
    """Test that universe manager updates global metrics."""

    def test_set_universe_source(self):
        from src.metrics import get_universe_metrics, set_universe_source
        set_universe_source("auto")
        m = get_universe_metrics()
        assert m["universe_source"] == "auto"
        set_universe_source("none")

    def test_inc_ws_disconnects(self):
        from src.metrics import get_universe_metrics, inc_ws_disconnects
        before = get_universe_metrics()["ws_disconnects_total"]
        inc_ws_disconnects()
        after = get_universe_metrics()["ws_disconnects_total"]
        assert after == before + 1

    def test_inc_gaps(self):
        from src.metrics import get_universe_metrics, inc_gaps
        before = get_universe_metrics()["gaps_total"]
        inc_gaps()
        after = get_universe_metrics()["gaps_total"]
        assert after == before + 1

    def test_set_universe_last_refresh(self):
        from src.metrics import get_universe_metrics, set_universe_last_refresh
        now = time.time()
        set_universe_last_refresh(now, 42)
        m = get_universe_metrics()
        assert m["universe_last_refresh_timestamp"] == now
        assert m["universe_market_count"] == 42

    def test_set_universe_error(self):
        from src.metrics import get_universe_metrics, set_universe_error
        set_universe_error("test error")
        m = get_universe_metrics()
        assert m["universe_error"] == "test error"
        set_universe_error(None)
        m = get_universe_metrics()
        assert m["universe_error"] is None


# ======================================================================
# Watcher client disconnect tracking
# ======================================================================


class TestWatcherDisconnectTracking:
    """Test that ws client tracks disconnects."""

    def test_initial_counts(self):
        from src.watcher.bridge import WatcherBridge
        from src.watcher.client import PolymarketClient
        import queue

        q = queue.Queue(maxsize=10)
        bridge = WatcherBridge(q)
        client = PolymarketClient(bridge)
        assert client.disconnect_count == 0
        assert client.gap_count == 0


# ======================================================================
# Server AUTO_UNIVERSE detection
# ======================================================================


class TestAutoUniverseDetection:
    """Test _is_auto_universe env var parsing."""

    def test_default_true(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTO_UNIVERSE", None)
            from src.server import _is_auto_universe
            assert _is_auto_universe() is True

    def test_explicit_true(self):
        with patch.dict(os.environ, {"AUTO_UNIVERSE": "true"}, clear=False):
            from src.server import _is_auto_universe
            assert _is_auto_universe() is True

    def test_explicit_false(self):
        with patch.dict(os.environ, {"AUTO_UNIVERSE": "false"}, clear=False):
            from src.server import _is_auto_universe
            assert _is_auto_universe() is False

    def test_numeric_one(self):
        with patch.dict(os.environ, {"AUTO_UNIVERSE": "1"}, clear=False):
            from src.server import _is_auto_universe
            assert _is_auto_universe() is True


# ======================================================================
# Dashboard /api/performance universe info
# ======================================================================


class TestPerformanceUniverseInfo:
    """Test _build_universe_info helper."""

    def test_build_without_manager(self):
        from src.api.dashboard_server import _build_universe_info
        info = _build_universe_info(None)
        assert "source" in info
        assert "markets_subscribed" in info

    def test_build_with_manager(self):
        from src.api.dashboard_server import _build_universe_info
        mgr = MagicMock()
        mgr.get_market_infos.return_value = [
            {"token_id": "abc...", "question": "Test?", "volume_24h": 5000, "liquidity": 500, "side": "yes"}
        ]
        info = _build_universe_info(mgr)
        assert "top_markets" in info
        assert len(info["top_markets"]) == 1
