"""
Tests — Financial Safety Layer
================================

Testes para:
- SafetyState
- RuntimeInvariantsEngine
- Gates adicionais no SafetyGate (staleness, trade drop, capital mismatch)
- Two-phase commit no TradeLedger
- Migração de statuses antigos
"""

import json
import time
import uuid
import threading
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.safety.safety_state import (
    SafetyState,
    RECONCILE_STALE,
    TRADE_DROP,
    CAPITAL_MISMATCH,
    INVARIANT_FAIL,
    THREAD_DEAD,
    EXCHANGE_UNREACHABLE,
)
from src.safety.invariants import RuntimeInvariantsEngine
from src.risk.safety_gate import can_execute, RECONCILE_STALE_SECONDS
from src.risk.capital_manager import CapitalManager
from src.risk.position_manager import PositionManager
from src.engine.trade_ledger import TradeLedger, TradeStatus


# =============================================================================
# Fixtures
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


# =============================================================================
# 1. SafetyState unit tests
# =============================================================================

class TestSafetyStateBasic:
    def test_initial_state_not_blocked(self, ss):
        assert not ss.buy_blocked
        assert ss.reasons == set()
        assert ss.invariant_fail_count == 0

    def test_block_buy(self, ss):
        ss.block_buy(INVARIANT_FAIL)
        assert ss.buy_blocked
        assert INVARIANT_FAIL in ss.reasons

    def test_block_buy_idempotent(self, ss):
        ss.block_buy(INVARIANT_FAIL)
        ss.block_buy(INVARIANT_FAIL)
        assert len(ss.reasons) == 1

    def test_clear_reason(self, ss):
        ss.block_buy(INVARIANT_FAIL)
        ss.clear_reason(INVARIANT_FAIL)
        assert not ss.buy_blocked

    def test_clear_nonexistent_reason(self, ss):
        ss.clear_reason("NONEXISTENT")  # No error

    def test_multiple_reasons(self, ss):
        ss.block_buy(INVARIANT_FAIL)
        ss.block_buy(TRADE_DROP)
        assert ss.buy_blocked
        ss.clear_reason(INVARIANT_FAIL)
        assert ss.buy_blocked  # Still blocked by TRADE_DROP
        ss.clear_reason(TRADE_DROP)
        assert not ss.buy_blocked

    def test_set_last_reconcile_ok(self, ss):
        ts = time.time()
        ss.set_last_reconcile_ok(ts)
        assert ss.last_reconcile_ok_ts == ts

    def test_mark_trade_drop(self, ss):
        ss.mark_trade_drop()
        assert ss.trade_drop_since_last_reconcile
        assert ss.buy_blocked
        assert TRADE_DROP in ss.reasons

    def test_clear_trade_drop_flag(self, ss):
        ss.mark_trade_drop()
        ss.clear_trade_drop_flag()
        assert not ss.trade_drop_since_last_reconcile
        assert TRADE_DROP not in ss.reasons

    def test_mark_capital_mismatch(self, ss):
        ts = time.time()
        ss.mark_capital_mismatch(ts)
        assert ss.last_capital_mismatch_ts == ts
        assert CAPITAL_MISMATCH in ss.reasons

    def test_increment_invariant_fail(self, ss):
        ss.increment_invariant_fail()
        ss.increment_invariant_fail()
        assert ss.invariant_fail_count == 2

    def test_snapshot(self, ss):
        ss.block_buy(INVARIANT_FAIL)
        snap = ss.snapshot()
        assert snap["buy_blocked"] is True
        assert "INVARIANT_FAIL" in snap["buy_block_reasons"]
        assert snap["invariant_fail_count"] == 0  # increment not called


class TestSafetyStateThreadSafety:
    def test_concurrent_block_clear(self, ss):
        errors = []

        def blocker():
            try:
                for _ in range(500):
                    ss.block_buy(INVARIANT_FAIL)
                    ss.clear_reason(INVARIANT_FAIL)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=blocker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors


# =============================================================================
# 2. RuntimeInvariantsEngine tests
# =============================================================================

class TestInvariantsCapital:
    def test_healthy_capital_passes(self, engine, ss):
        engine.check_all("test")
        assert not ss.buy_blocked

    def test_negative_total_capital_fails(self, engine, ss, cm):
        # Force negative total
        cm._realized_pnl = -20000.0
        engine.check_all("test_negative_total")
        assert ss.buy_blocked
        assert INVARIANT_FAIL in ss.reasons

    def test_capital_sum_mismatch_fails(self, engine, ss, cm):
        # Force inconsistency: add unreported reserve
        cm._order_reserves["phantom"] = 9999999.0
        engine.check_all("test_mismatch")
        # available goes negative → should fail
        assert ss.buy_blocked


class TestInvariantsPositions:
    def test_healthy_positions_pass(self, engine, ss, pm):
        pm.apply_fill("m1", "YES", 10.0, 0.5, "BUY")
        engine.check_all("test")
        assert not ss.buy_blocked

    def test_nan_shares_fails(self, engine, ss, pm):
        pm.apply_fill("m1", "YES", 10.0, 0.5, "BUY")
        # Force NaN
        key = ("m1", "YES")
        pm._positions[key].shares = float("nan")
        engine.check_all("test_nan")
        assert ss.buy_blocked


class TestInvariantsLedger:
    def test_healthy_ledger_passes(self, engine, ss, ledger, cm):
        tid = str(uuid.uuid4())
        cm.reserve_for_order(tid, 100.0)  # Manter reservas consistentes
        ledger.record_entry(tid, "m1", "C", "T", 0.5, 100, 1, 100, 0.5)
        engine.check_all("test")
        assert not ss.buy_blocked

    def test_negative_cumulative_fails(self, engine, ss, ledger, cm):
        tid = str(uuid.uuid4())
        cm.reserve_for_order(tid, 100.0)
        ledger.record_entry(tid, "m1", "C", "T", 0.5, 100, 1, 100, 0.5)
        # Force negative cumulative
        ledger._trades[tid].cumulative_filled_size = -1.0
        engine.check_all("test_neg_cum")
        assert ss.buy_blocked

    def test_cumulative_exceeds_final_fails(self, engine, ss, ledger, cm):
        tid = str(uuid.uuid4())
        cm.reserve_for_order(tid, 100.0)
        ledger.record_entry(tid, "m1", "C", "T", 0.5, 100, 1, 100, 0.5)
        ledger._trades[tid].cumulative_filled_size = 200.0
        engine.check_all("test_exceed")
        assert ss.buy_blocked

    def test_filled_but_cumulative_mismatch_fails(self, engine, ss, ledger, cm):
        tid = str(uuid.uuid4())
        cm.reserve_for_order(tid, 100.0)
        ledger.record_entry(tid, "m1", "C", "T", 0.5, 100, 1, 100, 0.5)
        ledger._trades[tid].status = TradeStatus.FILLED
        ledger._trades[tid].cumulative_filled_size = 50.0  # Should be ~100
        engine.check_all("test_filled_mismatch")
        assert ss.buy_blocked

    def test_nan_avg_price_fails(self, engine, ss, ledger, cm):
        tid = str(uuid.uuid4())
        cm.reserve_for_order(tid, 100.0)
        ledger.record_entry(tid, "m1", "C", "T", 0.5, 100, 1, 100, 0.5)
        ledger._trades[tid].weighted_avg_price = float("inf")
        engine.check_all("test_inf_price")
        assert ss.buy_blocked


# =============================================================================
# 3. SafetyGate — Reconcile Staleness (Gate A)
# =============================================================================

class TestReconcileStalenessBlocksBuy:
    def test_stale_reconcile_blocks_buy(self):
        ss = SafetyState()
        # Last reconcile was 120s ago (> 60s default)
        ss.set_last_reconcile_ok(time.time() - 120)

        ok, reason = can_execute("BUY", safety_state=ss)
        assert not ok
        assert "stale" in reason.lower()

    def test_recent_reconcile_allows_buy(self):
        ss = SafetyState()
        ss.set_last_reconcile_ok(time.time())

        ok, reason = can_execute("BUY", safety_state=ss)
        assert ok

    def test_no_reconcile_yet_allows_buy(self):
        """Sem reconcile anterior (sistema recém-iniciado) → permite."""
        ss = SafetyState()
        ok, reason = can_execute("BUY", safety_state=ss)
        assert ok

    def test_sell_always_allowed_even_stale(self):
        ss = SafetyState()
        ss.set_last_reconcile_ok(time.time() - 300)

        ok, _ = can_execute("SELL", safety_state=ss)
        assert ok


# =============================================================================
# 4. SafetyGate — Trade Drop (Gate B)
# =============================================================================

class TestTradeDropBlocksBuyUntilReconcileOk:
    def test_trade_drop_blocks_buy(self):
        ss = SafetyState()
        ss.mark_trade_drop()

        ok, reason = can_execute("BUY", safety_state=ss)
        assert not ok
        assert "trade drop" in reason.lower()

    def test_reconcile_ok_clears_trade_drop(self):
        ss = SafetyState()
        ss.mark_trade_drop()

        # Simular reconcile OK
        ss.set_last_reconcile_ok(time.time())
        ss.clear_trade_drop_flag()

        ok, _ = can_execute("BUY", safety_state=ss)
        assert ok


# =============================================================================
# 5. SafetyGate — Invariant fail (Gate via reasons)
# =============================================================================

class TestInvariantFailBlocksBuy:
    def test_invariant_fail_blocks_buy(self):
        ss = SafetyState()
        cm = CapitalManager(1000.0)
        pm = PositionManager()
        ledger = TradeLedger.__new__(TradeLedger)
        ledger._lock = threading.RLock()
        ledger._trades = {}

        # Force negative capital
        cm._realized_pnl = -5000.0

        engine = RuntimeInvariantsEngine(cm, pm, ledger, ss)
        engine.check_all("test")

        ok, reason = can_execute("BUY", safety_state=ss)
        assert not ok
        assert "invariante" in reason.lower() or "INVARIANT" in reason


# =============================================================================
# 6. Capital Mismatch (Gate C)
# =============================================================================

class TestCapitalMismatchBlocksBuy:
    def test_mismatch_blocks_buy(self):
        ss = SafetyState()
        ss.mark_capital_mismatch(time.time())

        ok, reason = can_execute("BUY", safety_state=ss)
        assert not ok
        assert "mismatch" in reason.lower()


# =============================================================================
# 7. Two-Phase Commit — TradeLedger
# =============================================================================

class TestTwoPhaseCommit:
    def test_full_lifecycle(self, tmp_path):
        cm = CapitalManager(10000.0)
        led = TradeLedger(state_path=str(tmp_path / "l.json"), capital_manager=cm)
        tid = str(uuid.uuid4())

        # Phase 1: Intent
        assert led.create_intent(tid, "m1", "C", "T", 0.5, 100, 100, 0.5)
        assert led.get_trade(tid)["status"] == TradeStatus.INTENT_CREATED

        # Phase 2: Capital reserved
        cm.reserve_for_order(tid, 100.0)
        led.advance_to_capital_reserved(tid)
        assert led.get_trade(tid)["status"] == TradeStatus.CAPITAL_RESERVED

        # Phase 3: Order submitted
        led.advance_to_order_submitted(tid, ["ord1"])
        assert led.get_trade(tid)["status"] == TradeStatus.ORDER_SUBMITTED
        assert led.get_trade(tid)["order_ids"] == ["ord1"]

        # Phase 4: Open
        led.advance_to_open(tid)
        assert led.get_trade(tid)["status"] == TradeStatus.OPEN

        # Phase 5: Partial fill → PARTIALLY_FILLED
        led.update_fill_status(tid, 50.0, 0.51)
        assert led.get_trade(tid)["status"] == TradeStatus.PARTIALLY_FILLED

        # Phase 6: Full fill → FILLED
        led.update_fill_status(tid, 100.0, 0.50)
        assert led.get_trade(tid)["status"] == TradeStatus.FILLED

        # Phase 7: Resolution → RESOLVED
        led.record_resolution(tid, 1)
        assert led.get_trade(tid)["status"] == TradeStatus.RESOLVED

    def test_idempotent_transitions(self, tmp_path):
        led = TradeLedger(state_path=str(tmp_path / "l.json"))
        tid = str(uuid.uuid4())

        led.create_intent(tid, "m1", "C", "T", 0.5, 100, 100, 0.5)

        # Repeat create_intent → idempotent (returns False)
        assert not led.create_intent(tid, "m1", "C", "T", 0.5, 100, 100, 0.5)

    def test_idempotent_advance_capital_reserved(self, tmp_path):
        led = TradeLedger(state_path=str(tmp_path / "l.json"))
        tid = str(uuid.uuid4())
        led.create_intent(tid, "m1", "C", "T", 0.5, 100, 100, 0.5)
        led.advance_to_capital_reserved(tid)
        # Second call should be idempotent
        led.advance_to_capital_reserved(tid)
        assert led.get_trade(tid)["status"] == TradeStatus.CAPITAL_RESERVED

    def test_idempotent_advance_order_submitted(self, tmp_path):
        led = TradeLedger(state_path=str(tmp_path / "l.json"))
        tid = str(uuid.uuid4())
        led.create_intent(tid, "m1", "C", "T", 0.5, 100, 100, 0.5)
        led.advance_to_capital_reserved(tid)
        led.advance_to_order_submitted(tid, ["o1"])
        led.advance_to_order_submitted(tid, ["o2"])  # idempotent
        assert led.get_trade(tid)["status"] == TradeStatus.ORDER_SUBMITTED

    def test_idempotent_advance_open(self, tmp_path):
        led = TradeLedger(state_path=str(tmp_path / "l.json"))
        tid = str(uuid.uuid4())
        led.create_intent(tid, "m1", "C", "T", 0.5, 100, 100, 0.5)
        led.advance_to_capital_reserved(tid)
        led.advance_to_order_submitted(tid, ["o1"])
        led.advance_to_open(tid)
        led.advance_to_open(tid)  # idempotent
        assert led.get_trade(tid)["status"] == TradeStatus.OPEN

    def test_wrong_transition_raises(self, tmp_path):
        led = TradeLedger(state_path=str(tmp_path / "l.json"))
        tid = str(uuid.uuid4())
        led.create_intent(tid, "m1", "C", "T", 0.5, 100, 100, 0.5)
        with pytest.raises(RuntimeError):
            led.advance_to_open(tid)  # Can't skip CAPITAL_RESERVED

    def test_cancel_pre_open(self, tmp_path):
        cm = CapitalManager(10000.0)
        led = TradeLedger(state_path=str(tmp_path / "l.json"), capital_manager=cm)
        tid = str(uuid.uuid4())
        led.create_intent(tid, "m1", "C", "T", 0.5, 100, 100, 0.5)
        cm.reserve_for_order(tid, 100.0)
        led.advance_to_capital_reserved(tid)
        led.cancel_trade(tid)
        assert led.get_trade(tid)["status"] == TradeStatus.CANCELLED
        # Capital liberado
        assert cm.get_available_capital() == 10000.0

    def test_cancel_idempotent(self, tmp_path):
        led = TradeLedger(state_path=str(tmp_path / "l.json"))
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "C", "T", 0.5, 100, 1, 100, 0.5)
        led.cancel_trade(tid)
        led.cancel_trade(tid)  # idempotent
        assert led.get_trade(tid)["status"] == TradeStatus.CANCELLED

    def test_rejected_and_expired(self, tmp_path):
        cm = CapitalManager(10000.0)
        led = TradeLedger(state_path=str(tmp_path / "l.json"), capital_manager=cm)

        t1 = str(uuid.uuid4())
        led.create_intent(t1, "m1", "C", "T", 0.5, 100, 100, 0.5)
        cm.reserve_for_order(t1, 100.0)
        led.mark_rejected(t1)
        assert led.get_trade(t1)["status"] == TradeStatus.REJECTED
        assert cm.get_available_capital() == 10000.0

        t2 = str(uuid.uuid4())
        led.create_intent(t2, "m2", "C", "T", 0.5, 200, 200, 0.5)
        cm.reserve_for_order(t2, 200.0)
        led.mark_expired(t2)
        assert led.get_trade(t2)["status"] == TradeStatus.EXPIRED
        assert cm.get_available_capital() == 10000.0


# =============================================================================
# 8. Migração de statuses antigos
# =============================================================================

class TestLedgerMigrationOldStatuses:
    def test_closed_migrates_to_resolved(self, tmp_path):
        """Ledger antigo com CLOSED → deve migrar para RESOLVED."""
        ledger_path = tmp_path / "ledger.json"
        tid = str(uuid.uuid4())
        old_data = {
            "schema_version": 1,
            "updated_at": "2025-01-01T00:00:00+00:00",
            "trade_count": 1,
            "trades": {
                tid: {
                    "trade_id": tid,
                    "market_id": "m1",
                    "category": "C",
                    "trade_type": "T",
                    "predicted_probability": 0.5,
                    "base_size": 100.0,
                    "multiplier_k": 1.0,
                    "final_size": 100.0,
                    "entry_price": 0.5,
                    "timestamp_entry": "2025-01-01T00:00:00+00:00",
                    "status": "CLOSED",  # Old status
                    "resolution_outcome": 1,
                    "timestamp_resolution": "2025-02-01T00:00:00+00:00",
                    "order_ids": [],
                    "cumulative_filled_size": 100.0,
                    "weighted_avg_price": 0.5,
                }
            },
        }
        with open(ledger_path, "w") as f:
            json.dump(old_data, f)

        led = TradeLedger(state_path=str(ledger_path))
        t = led.get_trade(tid)
        assert t is not None
        assert t["status"] == TradeStatus.RESOLVED

    def test_open_remains_open(self, tmp_path):
        """Ledger antigo com OPEN → permanece OPEN."""
        ledger_path = tmp_path / "ledger.json"
        tid = str(uuid.uuid4())
        old_data = {
            "schema_version": 1,
            "updated_at": "2025-01-01T00:00:00+00:00",
            "trade_count": 1,
            "trades": {
                tid: {
                    "trade_id": tid,
                    "market_id": "m1",
                    "category": "C",
                    "trade_type": "T",
                    "predicted_probability": 0.5,
                    "base_size": 100.0,
                    "multiplier_k": 1.0,
                    "final_size": 100.0,
                    "entry_price": 0.5,
                    "timestamp_entry": "2025-01-01T00:00:00+00:00",
                    "status": "OPEN",
                    "order_ids": ["o1"],
                    "cumulative_filled_size": 0.0,
                    "weighted_avg_price": 0.0,
                }
            },
        }
        with open(ledger_path, "w") as f:
            json.dump(old_data, f)

        led = TradeLedger(state_path=str(ledger_path))
        t = led.get_trade(tid)
        assert t["status"] == TradeStatus.OPEN


# =============================================================================
# 9. SafetyGate preserva comportamento nominal
# =============================================================================

class TestSafetyGateNominalBehavior:
    """Quando tudo saudável, BUY e SELL devem ser permitidos."""

    def test_no_safety_state_buy_allowed(self):
        ok, _ = can_execute("BUY")
        assert ok

    def test_no_safety_state_sell_allowed(self):
        ok, _ = can_execute("SELL")
        assert ok

    def test_healthy_safety_state_buy_allowed(self):
        ss = SafetyState()
        ok, _ = can_execute("BUY", safety_state=ss)
        assert ok

    def test_sell_always_allowed_despite_blocks(self):
        ss = SafetyState()
        ss.block_buy(INVARIANT_FAIL)
        ss.block_buy(TRADE_DROP)
        ss.block_buy(RECONCILE_STALE)
        ok, _ = can_execute("SELL", safety_state=ss)
        assert ok
