"""Testes para TradeLedger e pipeline de execução."""

import json
import uuid
import shutil
import pytest
from pathlib import Path

from src.engine.probability_calibrator import (
    ProbabilityCalibrator, MarketCategory, TradeType, N_MIN,
)
from src.execution.order_manager import OrderManager, OrderSide
from src.engine.trade_ledger import TradeLedger, TradeStatus, LEDGER_SCHEMA_VERSION
from src.execution.pipeline import execute_trade_decision, resolve_trade
from src.risk.risk_manager import RiskManager


@pytest.fixture
def ws_tmp(request):
    tmp_dir = Path(__file__).parent.parent / ".test_tmp" / request.node.name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def calibrator(ws_tmp):
    return ProbabilityCalibrator(state_path=str(ws_tmp / "cal.json"))


@pytest.fixture
def ledger(ws_tmp, calibrator):
    return TradeLedger(state_path=str(ws_tmp / "ledger.json"), calibrator=calibrator)


@pytest.fixture
def order_manager():
    return OrderManager()


# =============================================================================
# ENTRY
# =============================================================================

class TestLedgerEntry:
    def test_record_entry(self, ledger):
        tid = str(uuid.uuid4())
        assert ledger.record_entry(
            tid, "m1", "POLITICS", "DIRECTIONAL", 0.65, 100.0, 0.85, 85.0, 0.65
        ) is True
        trade = ledger.get_trade(tid)
        assert trade["status"] == TradeStatus.OPEN
        assert trade["final_size"] == 85.0

    def test_idempotent(self, ledger):
        tid = str(uuid.uuid4())
        assert ledger.record_entry(tid, "m1", "P", "D", 0.5, 50, 1, 50, 0.5) is True
        assert ledger.record_entry(tid, "m1", "P", "D", 0.5, 50, 1, 50, 0.5) is False
        assert ledger.trade_count == 1

    def test_with_order_ids(self, ledger):
        tid = str(uuid.uuid4())
        oids = ["o1", "o2", "o3"]
        ledger.record_entry(tid, "m1", "C", "A", 0.5, 50, 1, 50, 0.5, order_ids=oids)
        assert ledger.get_trade(tid)["order_ids"] == oids

    def test_multiple_entries(self, ledger):
        for _ in range(10):
            ledger.record_entry(str(uuid.uuid4()), "m1", "C", "A", 0.5, 50, 1, 50, 0.5)
        assert ledger.trade_count == 10


# =============================================================================
# RESOLUTION
# =============================================================================

class TestLedgerResolution:
    def test_resolve(self, ledger):
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.65, 100, 1, 100, 0.65)
        ledger.record_resolution(tid, 1)
        t = ledger.get_trade(tid)
        assert t["status"] == TradeStatus.RESOLVED
        assert t["resolution_outcome"] == 1

    def test_resolve_filled_trade(self, ledger):
        """Trades FILLED também podem ser resolvidos."""
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.65, 100, 1, 100, 0.65)
        ledger.update_fill_status(tid, 100.0, 0.65)
        assert ledger.get_trade(tid)["status"] == TradeStatus.FILLED
        ledger.record_resolution(tid, 0)
        assert ledger.get_trade(tid)["status"] == TradeStatus.RESOLVED

    def test_resolve_nonexistent(self, ledger):
        with pytest.raises(KeyError):
            ledger.record_resolution("fake", 1)

    def test_resolve_closed_raises(self, ledger):
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.65, 100, 1, 100, 0.65)
        ledger.record_resolution(tid, 1)
        with pytest.raises(RuntimeError, match="não pode ser resolvido"):
            ledger.record_resolution(tid, 0)

    def test_resolve_invalid_outcome(self, ledger):
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.65, 100, 1, 100, 0.65)
        with pytest.raises(ValueError):
            ledger.record_resolution(tid, 2)

    def test_cancel(self, ledger):
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.65, 100, 1, 100, 0.65)
        ledger.cancel_trade(tid)
        assert ledger.get_trade(tid)["status"] == TradeStatus.CANCELLED


# =============================================================================
# FILL STATUS
# =============================================================================

class TestFillStatus:
    def test_update_fill(self, ledger):
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "C", "A", 0.5, 100, 1, 100, 0.5)
        ledger.update_fill_status(tid, 50.0, 0.51)
        t = ledger.get_trade(tid)
        assert t["cumulative_filled_size"] == 50.0
        assert t["weighted_avg_price"] == 0.51
        assert t["status"] == TradeStatus.PARTIALLY_FILLED  # Parcial

    def test_full_fill_marks_filled(self, ledger):
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "C", "A", 0.5, 100, 1, 100, 0.5)
        ledger.update_fill_status(tid, 100.0, 0.50)
        assert ledger.get_trade(tid)["status"] == TradeStatus.FILLED

    def test_active_trades(self, ledger):
        t1 = str(uuid.uuid4())
        t2 = str(uuid.uuid4())
        t3 = str(uuid.uuid4())
        ledger.record_entry(t1, "m1", "C", "A", 0.5, 50, 1, 50, 0.5)
        ledger.record_entry(t2, "m1", "C", "A", 0.5, 50, 1, 50, 0.5)
        ledger.record_entry(t3, "m1", "C", "A", 0.5, 50, 1, 50, 0.5)
        ledger.update_fill_status(t2, 50.0, 0.50)  # → FILLED
        ledger.record_resolution(t3, 1)  # → CLOSED
        active = ledger.get_active_trades()
        assert t1 in active  # OPEN
        assert t2 in active  # FILLED
        assert t3 not in active  # CLOSED


# =============================================================================
# PERSISTÊNCIA
# =============================================================================

class TestLedgerPersistence:
    def test_save_and_reload(self, ws_tmp, calibrator):
        sp = ws_tmp / "ledger.json"
        l1 = TradeLedger(state_path=str(sp), calibrator=calibrator)
        tid = str(uuid.uuid4())
        l1.record_entry(tid, "m1", "C", "A", 0.5, 50, 1, 50, 0.5, order_ids=["o1"])
        l2 = TradeLedger(state_path=str(sp), calibrator=calibrator)
        t = l2.get_trade(tid)
        assert t is not None
        assert t["order_ids"] == ["o1"]

    def test_atomic_write(self, ws_tmp, calibrator):
        sp = ws_tmp / "ledger.json"
        ledger = TradeLedger(state_path=str(sp), calibrator=calibrator)
        for _ in range(10):
            ledger.record_entry(str(uuid.uuid4()), "m1", "C", "A", 0.5, 50, 1, 50, 0.5)
            with open(sp) as f:
                assert json.load(f)["schema_version"] == LEDGER_SCHEMA_VERSION

    def test_persisted_trades_complete(self, ws_tmp, calibrator):
        sp = ws_tmp / "ledger.json"
        ledger = TradeLedger(state_path=str(sp), calibrator=calibrator)
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.7, 100, 0.8, 80, 0.7)
        ledger.record_resolution(tid, 1)
        with open(sp) as f:
            d = json.load(f)["trades"][tid]
        assert d["status"] == TradeStatus.RESOLVED
        assert d["resolution_outcome"] == 1


# =============================================================================
# CALIBRATOR INTEGRAÇÃO
# =============================================================================

class TestLedgerCalibratorIntegration:
    def test_resolution_calls_calibrator(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "cal.json"))
        ledger = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "POLITICS", "DIRECTIONAL", 0.65, 100, 1, 100, 0.65)
        ledger.record_resolution(tid, 1)
        assert cal.get_metrics(MarketCategory.POLITICS, TradeType.DIRECTIONAL)["trade_count"] == 1

    def test_multiple_resolutions(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "cal.json"))
        ledger = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        for i in range(10):
            tid = str(uuid.uuid4())
            ledger.record_entry(tid, "m1", "FINANCE", "HEDGING", 0.6, 50, 1, 50, 0.6)
            ledger.record_resolution(tid, i % 2)
        assert cal.get_metrics(MarketCategory.FINANCE, TradeType.HEDGING)["trade_count"] == 10

    def test_without_calibrator(self, ws_tmp):
        ledger = TradeLedger(state_path=str(ws_tmp / "l.json"))
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "C", "D", 0.5, 50, 1, 50, 0.5)
        ledger.record_resolution(tid, 1)
        assert ledger.get_trade(tid)["status"] == TradeStatus.RESOLVED


# =============================================================================
# PIPELINE
# =============================================================================

class TestPipeline:
    def _dec(self, **kw):
        base = {
            "market_id": "m1", "category": MarketCategory.POLITICS,
            "trade_type": TradeType.DIRECTIONAL, "predicted_probability": 0.65,
            "base_size": 100.0, "price": 0.65, "side": OrderSide.BUY, "ttl_seconds": 60,
        }
        base.update(kw)
        return base

    def test_full_flow(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        tid = execute_trade_decision(self._dec(), cal, om, led)
        assert tid is not None
        t = led.get_trade(tid)
        assert t["status"] == TradeStatus.OPEN
        assert len(t["order_ids"]) > 0

    def test_abort_zero_size(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        assert execute_trade_decision(self._dec(base_size=0.0), cal, om, led) is None

    def test_resolve_updates_calibrator(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        tid = execute_trade_decision(
            self._dec(category=MarketCategory.FINANCE, trade_type=TradeType.ARBITRAGE,
                      predicted_probability=0.7, base_size=50, price=0.7),
            cal, om, led,
        )
        resolve_trade(tid, 1, led)
        assert cal.get_metrics(MarketCategory.FINANCE, TradeType.ARBITRAGE)["trade_count"] == 1

    def test_pipeline_with_risk_manager(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        rm = RiskManager({"max_exposure_total": 200.0})
        # Primeiro trade OK (100 < 200)
        t1 = execute_trade_decision(self._dec(), cal, om, led, risk_manager=rm)
        assert t1 is not None
        # Segundo trade OK (100+100=200 <= 200)
        t2 = execute_trade_decision(self._dec(), cal, om, led, risk_manager=rm)
        assert t2 is not None
        # Terceiro bloqueado (200+100=300 > 200)
        t3 = execute_trade_decision(self._dec(), cal, om, led, risk_manager=rm)
        assert t3 is None

    def test_pipeline_with_order_book_sufficient(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        book = {"bids": [(0.64, 500)], "asks": [(0.66, 500)]}
        tid = execute_trade_decision(self._dec(), cal, om, led, order_book=book)
        assert tid is not None

    def test_pipeline_with_order_book_insufficient(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        book = {"bids": [(0.64, 500)], "asks": [(0.66, 5)]}  # Só 5 no ask
        tid = execute_trade_decision(self._dec(), cal, om, led, order_book=book)
        assert tid is None

    def test_consistency_multiple_trades(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        tids = []
        for _ in range(10):
            tid = execute_trade_decision(
                self._dec(category=MarketCategory.SCIENCE, trade_type=TradeType.STATISTICAL,
                          predicted_probability=0.6, base_size=50, price=0.6),
                cal, om, led,
            )
            tids.append(tid)
        for i, tid in enumerate(tids):
            resolve_trade(tid, i % 2, led)
        assert cal.get_metrics(MarketCategory.SCIENCE, TradeType.STATISTICAL)["trade_count"] == 10
        assert len(led.get_open_trades()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
