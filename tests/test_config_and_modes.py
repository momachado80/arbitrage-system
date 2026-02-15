"""
Tests — AppConfig, RUN_MODE, ConfigSnapshot, HealthChecks
===========================================================

Cobrindo:
- Default run_mode OBSERVE
- LIVE denied quando reasons ativas
- LIVE denied quando config changed
- Config snapshot block BUY on change
- ACK config changed após validação
- Health check exchange unreachable blocks buy
- Thread dead blocks buy
- Thresholds from ENV override
- OBSERVE não coloca ordens
- PAPER usa stub execution
"""

import os
import tempfile
import threading
import uuid
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from src.config.config import (
    AppConfig,
    load_app_config_from_env,
    VALID_RUN_MODES,
)
from src.config.config_snapshot import ConfigSnapshotManager
from src.safety.safety_state import SafetyState, CONFIG_CHANGED
from src.safety.health_checks import HealthChecker
from src.risk.safety_gate import can_execute
from src.orchestrator import create_system, ack_reason
from src.execution.mode_stubs import ObserveOrderManager, PaperExchangeAPI
from src.execution.order_manager import OrderManager, OrderSide
from src.risk.capital_manager import CapitalManager
from src.engine.trade_ledger import TradeLedger


# =============================================================================
# 1. Default run_mode OBSERVE
# =============================================================================

class TestDefaultRunModeObserve:
    def test_default_run_mode_is_observe(self):
        with patch.dict(os.environ, {}, clear=False):
            if "RUN_MODE" in os.environ:
                del os.environ["RUN_MODE"]
            cfg = load_app_config_from_env()
        assert cfg.run_mode == "OBSERVE"

    def test_run_mode_from_env(self):
        with patch.dict(os.environ, {"RUN_MODE": "PAPER"}):
            cfg = load_app_config_from_env()
        assert cfg.run_mode == "PAPER"

    def test_invalid_run_mode_raises(self):
        with patch.dict(os.environ, {"RUN_MODE": "INVALID"}):
            with pytest.raises(ValueError, match="inválido"):
                load_app_config_from_env()


# =============================================================================
# 2. LIVE denied quando reasons ativas
# =============================================================================

class TestLiveDeniedWhenReasonsActive:
    def test_live_denied_when_reasons_active(self, tmp_path):
        cfg = AppConfig(
            run_mode="LIVE",
            reconcile_loop_interval_seconds=30,
            reconcile_stale_seconds=60,
            queue_maxsize=100,
            trade_drop_strikes_to_clear=3,
            invariant_strikes_to_clear=3,
            capital_mismatch_percent_tolerance=0.001,
            capital_mismatch_abs_tolerance=0.01,
            health_check_interval_seconds=30,
        )
        system = create_system(config=cfg, state_dir=str(tmp_path))
        om = system["order_manager"]
        assert om.__class__.__name__ == "ObserveOrderManager"

    def test_live_denied_when_streaks_insufficient(self, tmp_path):
        cfg = AppConfig(
            run_mode="LIVE",
            reconcile_loop_interval_seconds=30,
            reconcile_stale_seconds=60,
            queue_maxsize=100,
            trade_drop_strikes_to_clear=3,
            invariant_strikes_to_clear=3,
            capital_mismatch_percent_tolerance=0.001,
            capital_mismatch_abs_tolerance=0.01,
            health_check_interval_seconds=30,
        )
        system = create_system(config=cfg, state_dir=str(tmp_path))
        om = system["order_manager"]
        assert om.__class__.__name__ == "ObserveOrderManager"


# =============================================================================
# 3. LIVE denied quando config changed
# =============================================================================

class TestLiveDeniedWhenConfigChanged:
    def test_live_denied_when_config_changed(self, tmp_path):
        cfg = AppConfig(
            run_mode="LIVE",
            reconcile_loop_interval_seconds=30,
            reconcile_stale_seconds=60,
            queue_maxsize=100,
            trade_drop_strikes_to_clear=3,
            invariant_strikes_to_clear=3,
            capital_mismatch_percent_tolerance=0.001,
            capital_mismatch_abs_tolerance=0.01,
            health_check_interval_seconds=30,
        )
        create_system(config=cfg, state_dir=str(tmp_path))

        cfg2 = AppConfig(
            run_mode="LIVE",
            reconcile_loop_interval_seconds=60,
            reconcile_stale_seconds=60,
            queue_maxsize=100,
            trade_drop_strikes_to_clear=3,
            invariant_strikes_to_clear=3,
            capital_mismatch_percent_tolerance=0.001,
            capital_mismatch_abs_tolerance=0.01,
            health_check_interval_seconds=30,
        )
        system = create_system(config=cfg2, state_dir=str(tmp_path))
        assert CONFIG_CHANGED in system["safety_state"].reasons

        om = system["order_manager"]
        assert om.__class__.__name__ == "ObserveOrderManager"


# =============================================================================
# 4. Config snapshot blocks BUY on change
# =============================================================================

class TestConfigSnapshotBlocksBuy:
    def test_config_snapshot_blocks_buy_on_change(self, tmp_path):
        cfg1 = AppConfig(
            run_mode="OBSERVE",
            reconcile_loop_interval_seconds=30,
            reconcile_stale_seconds=60,
            queue_maxsize=100,
            trade_drop_strikes_to_clear=3,
            invariant_strikes_to_clear=3,
            capital_mismatch_percent_tolerance=0.001,
            capital_mismatch_abs_tolerance=0.01,
            health_check_interval_seconds=30,
        )
        ss = SafetyState()
        snap_path = str(tmp_path / "snap.json")
        snap = ConfigSnapshotManager(state_path=snap_path)
        snap.initialize_or_validate(cfg1, ss)
        assert not ss.buy_blocked

        cfg2 = AppConfig(
            run_mode="OBSERVE",
            reconcile_loop_interval_seconds=90,
            reconcile_stale_seconds=60,
            queue_maxsize=100,
            trade_drop_strikes_to_clear=3,
            invariant_strikes_to_clear=3,
            capital_mismatch_percent_tolerance=0.001,
            capital_mismatch_abs_tolerance=0.01,
            health_check_interval_seconds=30,
        )
        snap2 = ConfigSnapshotManager(state_path=snap_path)
        snap2.initialize_or_validate(cfg2, ss)
        assert ss.buy_blocked
        assert CONFIG_CHANGED in ss.reasons

        ok, _ = can_execute("BUY", safety_state=ss)
        assert not ok


# =============================================================================
# 5. ACK config changed após validação
# =============================================================================

class TestAckConfigChanged:
    def test_ack_config_changed_allows_after_validation(self, tmp_path):
        cfg = AppConfig(
            run_mode="OBSERVE",
            reconcile_loop_interval_seconds=90,
            reconcile_stale_seconds=60,
            queue_maxsize=100,
            trade_drop_strikes_to_clear=3,
            invariant_strikes_to_clear=3,
            capital_mismatch_percent_tolerance=0.001,
            capital_mismatch_abs_tolerance=0.01,
            health_check_interval_seconds=30,
        )
        system = create_system(config=cfg, state_dir=str(tmp_path))
        ss = system["safety_state"]

        cfg2 = AppConfig(
            run_mode="OBSERVE",
            reconcile_loop_interval_seconds=120,
            reconcile_stale_seconds=60,
            queue_maxsize=100,
            trade_drop_strikes_to_clear=3,
            invariant_strikes_to_clear=3,
            capital_mismatch_percent_tolerance=0.001,
            capital_mismatch_abs_tolerance=0.01,
            health_check_interval_seconds=30,
        )
        snap = ConfigSnapshotManager(state_path=str(tmp_path / "config_snapshot.json"))
        snap.initialize_or_validate(cfg2, ss)
        assert CONFIG_CHANGED in ss.reasons

        system["config"] = cfg2
        result = ack_reason(CONFIG_CHANGED, "validated", system)
        assert result
        assert CONFIG_CHANGED not in ss.reasons


# =============================================================================
# 6. Health check exchange unreachable blocks buy
# =============================================================================

class TestHealthCheckExchangeUnreachable:
    def test_health_check_exchange_unreachable_blocks_buy(self):
        ss = SafetyState()
        api = Mock()
        api.get_account_summary = Mock(side_effect=Exception("network error"))
        cfg = AppConfig(
            run_mode="OBSERVE",
            reconcile_loop_interval_seconds=30,
            reconcile_stale_seconds=60,
            queue_maxsize=100,
            trade_drop_strikes_to_clear=3,
            invariant_strikes_to_clear=3,
            capital_mismatch_percent_tolerance=0.001,
            capital_mismatch_abs_tolerance=0.01,
            health_check_interval_seconds=30,
        )

        checker = HealthChecker(api, ss, cfg)
        checker.run_health_check()

        assert "EXCHANGE_UNREACHABLE" in ss.reasons
        ok, _ = can_execute("BUY", safety_state=ss)
        assert not ok


# =============================================================================
# 7. Thread dead blocks buy
# =============================================================================

class TestThreadDeadBlocksBuy:
    def test_thread_dead_blocks_buy(self):
        ss = SafetyState()
        dead_thread = threading.Thread(target=lambda: None)
        dead_thread.start()
        dead_thread.join(timeout=0.1)

        cfg = AppConfig(
            run_mode="OBSERVE",
            reconcile_loop_interval_seconds=30,
            reconcile_stale_seconds=60,
            queue_maxsize=100,
            trade_drop_strikes_to_clear=3,
            invariant_strikes_to_clear=3,
            capital_mismatch_percent_tolerance=0.001,
            capital_mismatch_abs_tolerance=0.01,
            health_check_interval_seconds=30,
        )

        checker = HealthChecker(
            Mock(), ss, cfg,
            market_engine=Mock(is_alive=Mock(return_value=True)),
            reconciler_loop=Mock(is_alive=Mock(return_value=True)),
            watcher_thread=dead_thread,
        )
        checker.run_health_check()

        assert "THREAD_DEAD" in ss.reasons
        ok, _ = can_execute("BUY", safety_state=ss)
        assert not ok


# =============================================================================
# 8. Thresholds from ENV
# =============================================================================

class TestThresholdsFromEnv:
    def test_thresholds_from_env_override_defaults(self):
        with patch.dict(
            os.environ,
            {
                "RUN_MODE": "OBSERVE",
                "RECONCILE_LOOP_INTERVAL_SECONDS": "45",
                "RECONCILE_STALE_SECONDS": "90",
                "QUEUE_MAXSIZE": "200",
            },
        ):
            cfg = load_app_config_from_env()
        assert cfg.reconcile_loop_interval_seconds == 45
        assert cfg.reconcile_stale_seconds == 90
        assert cfg.queue_maxsize == 200


# =============================================================================
# 9. OBSERVE não coloca ordens
# =============================================================================

class TestObserveModeNoOrders:
    def test_observe_mode_does_not_place_orders(self):
        om = ObserveOrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.5, 100)
        assert ids == []
        assert om.cancel_all_buy_open_orders_best_effort() == []


# =============================================================================
# 10. PAPER usa stub execution
# =============================================================================

class TestPaperModeStub:
    def test_paper_mode_uses_stub_execution(self, tmp_path):
        with patch.dict(os.environ, {"RUN_MODE": "PAPER"}):
            cfg = load_app_config_from_env()
        system = create_system(config=cfg, state_dir=str(tmp_path))
        om = system["order_manager"]
        api = system["exchange_api"]

        assert om.__class__.__name__ == "OrderManager"
        assert api.__class__.__name__ == "PaperExchangeAPI"

        ids = om.place_limit_order("m1", OrderSide.BUY, 0.5, 50)
        assert len(ids) > 0
        fills = api.get_fills(ids[0])
        assert len(fills) == 1
        assert fills[0]["fill_id"].startswith("paper_fill_")
        assert fills[0]["price"] == 0.5
