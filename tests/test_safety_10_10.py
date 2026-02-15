"""
Tests — Financial Safety 10/10
================================

Testes para:
- Invariant fail → SAFE_MODE escalada
- 3 strikes to clear (INVARIANT_FAIL, CAPITAL_MISMATCH)
- ACK com precondições (THREAD_DEAD, EXCHANGE_UNREACHABLE, SAFE_MODE)
- Reserved notional invariant vs locked_in_open_orders
- Mitigação ativa: cancel BUY orders
- Audit report conteúdo
- Session id monotônico
"""

import json
import time
import uuid
import threading
from typing import Any, Dict, List, Optional

import pytest

from src.safety.safety_state import (
    SafetyState,
    RECONCILE_STALE,
    TRADE_DROP,
    CAPITAL_MISMATCH,
    INVARIANT_FAIL,
    THREAD_DEAD,
    EXCHANGE_UNREACHABLE,
    SAFE_MODE,
    STRIKES_TO_CLEAR,
)
from src.safety.invariants import RuntimeInvariantsEngine
from src.safety.safety_ack import SafetyAckStore
from src.safety.audit_report import build_audit_report
from src.risk.safety_gate import can_execute
from src.risk.capital_manager import CapitalManager
from src.risk.position_manager import PositionManager
from src.engine.trade_ledger import TradeLedger, TradeStatus
from src.execution.order_manager import OrderManager, OrderSide
from src.execution.reconciler import OrderReconciler, ExchangeAPI
from src.orchestrator import (
    ack_reason,
    ReconcilerLoop,
    ExchangeAPIStub,
    PipelineHandler,
)
from src.engine.market_data import MarketDataEngine


# =============================================================================
# Helpers / Fixtures
# =============================================================================

@pytest.fixture
def ss():
    return SafetyState()


@pytest.fixture
def cm():
    return CapitalManager(10000.0)


@pytest.fixture
def pm():
    return PositionManager()


@pytest.fixture
def ledger(tmp_path):
    return TradeLedger(state_path=str(tmp_path / "ledger.json"))


@pytest.fixture
def engine(cm, pm, ledger, ss):
    return RuntimeInvariantsEngine(cm, pm, ledger, ss)


@pytest.fixture
def ack_store(tmp_path):
    return SafetyAckStore(state_path=str(tmp_path / "ack.json"))


@pytest.fixture
def om():
    return OrderManager(account_mode="SPOT_ONLY")


class StubExchangeAPI(ExchangeAPI):
    """Stub para testes."""

    def __init__(self, account_summary=None):
        self._account_summary = account_summary

    def get_order_status(self, order_id):
        return {"order_id": order_id, "status": "PENDING", "filled_size": 0.0}

    def get_fills(self, order_id):
        return []

    def get_position(self, market_id):
        return {"market_id": market_id, "size": 0.0, "avg_entry": 0.0}

    def get_account_summary(self):
        return self._account_summary


def _make_system(tmp_path, cm=None, ss=None, om=None, exchange_api=None):
    """Cria um dict de sistema minimal para testes de ack_reason."""
    import queue
    from src.watcher.bridge import WatcherBridge

    _cm = cm or CapitalManager(10000.0)
    _ss = ss or SafetyState()
    _pm = PositionManager()
    _ledger = TradeLedger(state_path=str(tmp_path / "ledger.json"))
    _om = om or OrderManager(account_mode="SPOT_ONLY")
    _ack_store = SafetyAckStore(state_path=str(tmp_path / "ack.json"))
    _api = exchange_api or ExchangeAPIStub()
    _reconciler = OrderReconciler()
    _inv = RuntimeInvariantsEngine(_cm, _pm, _ledger, _ss)
    _q = queue.Queue(maxsize=10)
    _bridge = WatcherBridge(_q)

    pipeline_handler = PipelineHandler()
    reconciler_loop = ReconcilerLoop(
        reconciler=_reconciler,
        order_manager=_om,
        trade_ledger=_ledger,
        exchange_api=_api,
    )
    market_engine = MarketDataEngine(
        update_queue=_q,
        pipeline=pipeline_handler,
        reconciler=reconciler_loop,
    )

    return {
        "capital_manager": _cm,
        "position_manager": _pm,
        "trade_ledger": _ledger,
        "order_manager": _om,
        "safety_state": _ss,
        "ack_store": _ack_store,
        "invariants_engine": _inv,
        "reconciler": _reconciler,
        "reconciler_loop": reconciler_loop,
        "market_engine": market_engine,
        "pipeline_handler": pipeline_handler,
        "exchange_api": _api,
    }


# =============================================================================
# 1. Invariant fail → SAFE_MODE
# =============================================================================

class TestInvariantFailEnablesSafeMode:
    def test_invariant_fail_enables_safe_mode_and_blocks_buy(self, ss, cm, pm, ledger):
        # Criar reserva fantasma → invariant fail
        cm._order_reserves["phantom"] = 9999999.0
        eng = RuntimeInvariantsEngine(cm, pm, ledger, ss)
        eng.check_all("test")

        assert ss.buy_blocked
        assert INVARIANT_FAIL in ss.reasons
        # Devido ao reserve check, SAFE_MODE deve ser ativado
        assert ss.global_safe_mode
        assert SAFE_MODE in ss.reasons


# =============================================================================
# 2. 3 strikes to clear: INVARIANT_FAIL
# =============================================================================

class TestInvariantsOkStreakClearsAfter3:
    def test_invariants_ok_streak_clears_invariant_fail_after_3(
        self, ss, cm, pm, ledger,
    ):
        eng = RuntimeInvariantsEngine(cm, pm, ledger, ss)

        # Forçar invariant fail
        cm._realized_pnl = -20000.0
        eng.check_all("fail1")
        assert INVARIANT_FAIL in ss.reasons
        assert ss.invariants_ok_streak == 0

        # Corrigir capital
        cm._realized_pnl = 0.0

        # 1 check OK
        eng.check_all("ok1")
        assert ss.invariants_ok_streak == 1
        assert INVARIANT_FAIL in ss.reasons  # Ainda ativo (streak < 3)

        # 2 checks OK
        eng.check_all("ok2")
        assert ss.invariants_ok_streak == 2
        assert INVARIANT_FAIL in ss.reasons

        # 3 checks OK → auto-clear (se não em SAFE_MODE)
        # Note: SAFE_MODE pode estar ativo se reserve check falhou anteriormente
        # Vamos limpar safe_mode para isolar o teste de strikes
        ss.disable_safe_mode()
        eng.check_all("ok3")
        assert ss.invariants_ok_streak >= 3
        assert INVARIANT_FAIL not in ss.reasons


# =============================================================================
# 3. 3 strikes to clear: CAPITAL_MISMATCH
# =============================================================================

class TestCapitalMismatchOkStreakClearsAfter3:
    def test_capital_mismatch_ok_streak_clears_after_3_reconciles(self, ss):
        # Simular mismatch
        ss.block_buy(CAPITAL_MISMATCH)
        ss.mark_capital_mismatch_fail()

        # 3 reconciles sem mismatch
        ss.mark_capital_mismatch_ok()
        assert ss.capital_mismatch_ok_streak == 1
        assert not ss.maybe_clear_capital_mismatch()

        ss.mark_capital_mismatch_ok()
        assert not ss.maybe_clear_capital_mismatch()

        ss.mark_capital_mismatch_ok()
        assert ss.capital_mismatch_ok_streak == 3
        assert ss.maybe_clear_capital_mismatch()
        assert CAPITAL_MISMATCH not in ss.reasons

    def test_fail_resets_streak(self, ss):
        ss.mark_capital_mismatch_ok()
        ss.mark_capital_mismatch_ok()
        assert ss.capital_mismatch_ok_streak == 2
        ss.mark_capital_mismatch_fail()
        assert ss.capital_mismatch_ok_streak == 0

    def test_safe_mode_blocks_auto_clear(self, ss):
        ss.block_buy(CAPITAL_MISMATCH)
        ss.enable_safe_mode(CAPITAL_MISMATCH)
        for _ in range(5):
            ss.mark_capital_mismatch_ok()
        assert not ss.maybe_clear_capital_mismatch()


# =============================================================================
# 4. ACK: THREAD_DEAD requer threads alive
# =============================================================================

class TestAckThreadDeadRequiresThreadsAlive:
    def test_ack_thread_dead_requires_threads_alive(self, tmp_path):
        system = _make_system(tmp_path)
        ss = system["safety_state"]
        ss.block_buy(THREAD_DEAD)

        # Threads não foram started → not alive
        result = ack_reason(THREAD_DEAD, "test", system)
        assert not result
        assert THREAD_DEAD in ss.reasons

    def test_ack_thread_dead_succeeds_with_live_threads(self, tmp_path):
        system = _make_system(tmp_path)
        ss = system["safety_state"]
        ss.block_buy(THREAD_DEAD)

        # Start threads
        rl = system["reconciler_loop"]
        me = system["market_engine"]
        rl.start()
        me.start()
        try:
            result = ack_reason(THREAD_DEAD, "threads ok", system)
            assert result
            assert THREAD_DEAD not in ss.reasons
        finally:
            rl.stop()
            me.stop()
            rl.join(timeout=5)
            me.join(timeout=5)


# =============================================================================
# 5. ACK: EXCHANGE_UNREACHABLE requer health check
# =============================================================================

class TestAckExchangeUnreachable:
    def test_ack_exchange_unreachable_requires_health_check(self, tmp_path):
        # Stub que retorna None → exchange "não disponível"
        api = StubExchangeAPI(account_summary=None)
        system = _make_system(tmp_path, exchange_api=api)
        ss = system["safety_state"]
        ss.block_buy(EXCHANGE_UNREACHABLE)

        result = ack_reason(EXCHANGE_UNREACHABLE, "test", system)
        assert not result
        assert EXCHANGE_UNREACHABLE in ss.reasons

    def test_ack_exchange_unreachable_succeeds_when_reachable(self, tmp_path):
        api = StubExchangeAPI(account_summary={"total_balance": 10000.0})
        system = _make_system(tmp_path, exchange_api=api)
        ss = system["safety_state"]
        ss.block_buy(EXCHANGE_UNREACHABLE)

        result = ack_reason(EXCHANGE_UNREACHABLE, "exchange ok", system)
        assert result
        assert EXCHANGE_UNREACHABLE not in ss.reasons


# =============================================================================
# 6. ACK: SAFE_MODE requer nenhuma outra reason
# =============================================================================

class TestSafeModeAck:
    def test_safe_mode_ack_requires_no_other_reasons(self, tmp_path):
        system = _make_system(tmp_path)
        ss = system["safety_state"]
        ss.enable_safe_mode(INVARIANT_FAIL)

        # INVARIANT_FAIL + SAFE_MODE ambos ativos
        result = ack_reason(SAFE_MODE, "test", system)
        assert not result
        assert SAFE_MODE in ss.reasons

    def test_safe_mode_ack_succeeds_when_clean(self, tmp_path):
        system = _make_system(tmp_path)
        ss = system["safety_state"]
        ss.enable_safe_mode(INVARIANT_FAIL)

        # Limpar INVARIANT_FAIL primeiro
        ss.clear_reason(INVARIANT_FAIL)
        result = ack_reason(SAFE_MODE, "all clear", system)
        assert result
        assert not ss.global_safe_mode
        assert SAFE_MODE not in ss.reasons


# =============================================================================
# 7. Reserved notional invariant vs locked_in_open_orders
# =============================================================================

class TestReservedNotionalInvariant:
    def test_matching_reserves_pass(self, ss, cm, pm, ledger):
        tid = str(uuid.uuid4())
        cm.reserve_for_order(tid, 500.0)
        ledger.record_entry(
            tid, "m1", "C", "T", 0.5, 500, 1, 500, 0.5,
            action="BUY",
        )
        eng = RuntimeInvariantsEngine(cm, pm, ledger, ss)
        eng.check_all("test")
        assert not ss.buy_blocked

    def test_phantom_reserve_fails(self, ss, cm, pm, ledger):
        """Capital reservado sem trade correspondente → invariant fail."""
        cm._order_reserves["phantom_trade"] = 1000.0
        eng = RuntimeInvariantsEngine(cm, pm, ledger, ss)
        eng.check_all("test")
        assert ss.buy_blocked
        assert INVARIANT_FAIL in ss.reasons

    def test_sell_trades_ignored_in_reserve_calc(self, ss, cm, pm, ledger):
        """SELL trades não contam nas reservas."""
        tid = str(uuid.uuid4())
        ledger.record_entry(
            tid, "m1", "C", "T", 0.5, 100, 1, 100, 0.5,
            action="SELL",
        )
        eng = RuntimeInvariantsEngine(cm, pm, ledger, ss)
        eng.check_all("test")
        assert not ss.buy_blocked


# =============================================================================
# 8. Mitigação: cancel BUY orders
# =============================================================================

class TestMitigationCancelsOnlyBuyOrders:
    def test_mitigation_cancels_only_buy_orders_best_effort(self, om):
        # Criar BUY e SELL orders
        buy_ids = om.place_limit_order("m1", OrderSide.BUY, 0.5, 100)
        sell_ids = om.place_limit_order("m1", OrderSide.SELL, 0.6, 50)

        cancelled = om.cancel_all_buy_open_orders_best_effort()

        # BUY canceladas
        assert len(cancelled) > 0
        for oid in buy_ids:
            status = om.check_order_status(oid)
            assert status["status"] == "CANCELLED"

        # SELL intactas
        for oid in sell_ids:
            status = om.check_order_status(oid)
            assert status["status"] != "CANCELLED"

    def test_mitigation_idempotent(self, om):
        om.place_limit_order("m1", OrderSide.BUY, 0.5, 100)
        c1 = om.cancel_all_buy_open_orders_best_effort()
        c2 = om.cancel_all_buy_open_orders_best_effort()
        assert len(c1) > 0
        assert len(c2) == 0  # Já foram canceladas


# =============================================================================
# 9. Audit report contém seções obrigatórias
# =============================================================================

class TestAuditReport:
    def test_audit_report_contains_required_sections(self, cm, pm, ledger, ss):
        # Criar algum estado
        tid = str(uuid.uuid4())
        cm.reserve_for_order(tid, 200.0)
        ledger.record_entry(
            tid, "m1", "C", "T", 0.5, 200, 1, 200, 0.5, action="BUY",
        )
        ss.block_buy(INVARIANT_FAIL)

        report = build_audit_report(cm, pm, ledger, ss)

        assert "AUDIT REPORT" in report
        assert "SAFETY STATE" in report
        assert "buy_blocked" in report
        assert "INVARIANT_FAIL" in report
        assert "CAPITAL" in report
        assert "total_capital" in report
        assert "available_capital" in report
        assert "TRADES BY STATUS" in report
        assert "POSITIONS" in report
        assert "Generated at" in report

    def test_audit_report_none_safe(self):
        """Report não falha com None components."""
        report = build_audit_report(None, None, None, None)
        assert "AUDIT REPORT" in report
        assert "not available" in report


# =============================================================================
# 10. Session id monotônico no reconciler
# =============================================================================

class TestReconcileSessionId:
    def test_session_id_increments(self, ss, tmp_path):
        om = OrderManager(account_mode="SPOT_ONLY")
        ledger = TradeLedger(state_path=str(tmp_path / "l.json"))
        api = StubExchangeAPI()
        rec = OrderReconciler()

        rec.reconcile_open_trades(om, ledger, api, safety_state=ss)
        assert ss.last_reconcile_session_id == 1

        rec.reconcile_open_trades(om, ledger, api, safety_state=ss)
        assert ss.last_reconcile_session_id == 2

        rec.reconcile_open_trades(om, ledger, api, safety_state=ss)
        assert ss.last_reconcile_session_id == 3

    def test_session_id_monotonic(self, ss, tmp_path):
        om = OrderManager(account_mode="SPOT_ONLY")
        ledger = TradeLedger(state_path=str(tmp_path / "l.json"))
        api = StubExchangeAPI()
        rec = OrderReconciler()

        ids = []
        for _ in range(5):
            rec.reconcile_open_trades(om, ledger, api, safety_state=ss)
            ids.append(ss.last_reconcile_session_id)

        assert ids == sorted(ids)
        assert len(set(ids)) == 5  # Todos únicos


# =============================================================================
# 11. SafetyAckStore persistência
# =============================================================================

class TestSafetyAckStore:
    def test_record_and_read(self, ack_store):
        ack_store.record_ack("THREAD_DEAD", "test note", {"x": 1})
        last = ack_store.get_last_ack("THREAD_DEAD")
        assert last is not None
        assert last["reason"] == "THREAD_DEAD"
        assert last["operator_note"] == "test note"
        assert last["preconditions"] == {"x": 1}

    def test_get_last_returns_most_recent(self, ack_store):
        ack_store.record_ack("THREAD_DEAD", "first", {})
        ack_store.record_ack("THREAD_DEAD", "second", {})
        last = ack_store.get_last_ack("THREAD_DEAD")
        assert last["operator_note"] == "second"

    def test_get_last_nonexistent_returns_none(self, ack_store):
        assert ack_store.get_last_ack("NONEXISTENT") is None

    def test_persistence_roundtrip(self, tmp_path):
        path = str(tmp_path / "ack.json")
        store1 = SafetyAckStore(state_path=path)
        store1.record_ack("TEST", "note1", {"a": True})

        store2 = SafetyAckStore(state_path=path)
        last = store2.get_last_ack("TEST")
        assert last is not None
        assert last["operator_note"] == "note1"


# =============================================================================
# 12. Safe Mode: clear automático bloqueado quando ativo
# =============================================================================

class TestSafeModePreventsAutoClear:
    def test_safe_mode_blocks_invariant_auto_clear(self, ss):
        ss.enable_safe_mode(INVARIANT_FAIL)
        for _ in range(10):
            ss.mark_invariants_ok()
        assert not ss.maybe_clear_invariant_fail()
        assert INVARIANT_FAIL in ss.reasons

    def test_safe_mode_blocks_capital_mismatch_auto_clear(self, ss):
        ss.enable_safe_mode(CAPITAL_MISMATCH)
        for _ in range(10):
            ss.mark_capital_mismatch_ok()
        assert not ss.maybe_clear_capital_mismatch()
        assert CAPITAL_MISMATCH in ss.reasons
