"""Testes para OrderReconciler."""

import uuid
import shutil
import pytest
from pathlib import Path

from src.execution.order_manager import OrderManager, OrderSide
from src.engine.trade_ledger import TradeLedger, TradeStatus
from src.execution.reconciler import OrderReconciler, ExchangeAPI


# =============================================================================
# MOCK API
# =============================================================================

class MockExchangeAPI(ExchangeAPI):
    """Mock que retorna dados configuráveis por order_id."""

    def __init__(self):
        self._statuses = {}
        self._fills = {}
        self._positions = {}

    def set_order(self, order_id, status, filled_size, fills=None):
        self._statuses[order_id] = {
            "order_id": order_id,
            "status": status,
            "filled_size": filled_size,
        }
        self._fills[order_id] = fills or []

    def get_order_status(self, order_id):
        return self._statuses.get(order_id, {
            "order_id": order_id, "status": "PENDING", "filled_size": 0.0,
        })

    def get_fills(self, order_id):
        return self._fills.get(order_id, [])

    def get_position(self, market_id):
        return self._positions.get(market_id, {"market_id": market_id, "size": 0, "avg_entry": 0})


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def ws_tmp(request):
    d = Path(__file__).parent.parent / ".test_tmp" / request.node.name
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def om():
    return OrderManager()


@pytest.fixture
def ledger(ws_tmp):
    return TradeLedger(state_path=str(ws_tmp / "ledger.json"))


@pytest.fixture
def reconciler():
    return OrderReconciler()


@pytest.fixture
def api():
    return MockExchangeAPI()


# =============================================================================
# TESTES
# =============================================================================

class TestReconcileBasic:
    def test_fully_filled(self, om, ledger, reconciler, api):
        """Ordem preenchida na API → trade marcado como FILLED."""
        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        api.set_order(oids[0], "FILLED", 50.0, [
            {"fill_id": "f1", "price": 0.65, "size": 50.0, "timestamp": "2025-01-01T00:00:00+00:00"},
        ])

        actions = reconciler.reconcile_open_trades(om, ledger, api)
        assert actions[tid] == "FULLY_FILLED"
        assert ledger.get_trade(tid)["status"] == TradeStatus.FILLED

    def test_all_cancelled_no_fills(self, om, ledger, reconciler, api):
        """Todas ordens canceladas sem fills → trade CANCELLED."""
        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        api.set_order(oids[0], "CANCELLED", 0.0)

        actions = reconciler.reconcile_open_trades(om, ledger, api)
        assert actions[tid] == "CANCELLED"
        assert ledger.get_trade(tid)["status"] == TradeStatus.CANCELLED


class TestPartialFills:
    def test_partial_fill_two_calls(self, om, ledger, reconciler, api):
        """Partial fill em duas chamadas sucessivas."""
        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 100.0, 60)
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.65, 100, 1, 100, 0.65, order_ids=oids)

        # Primeiro reconcile: 40 preenchidos
        api.set_order(oids[0], "PARTIALLY_FILLED", 40.0, [
            {"fill_id": "f1", "price": 0.65, "size": 40.0, "timestamp": "2025-01-01T00:00:00+00:00"},
        ])
        reconciler.reconcile_open_trades(om, ledger, api)
        t = ledger.get_trade(tid)
        assert t["cumulative_filled_size"] == 40.0
        assert t["status"] == TradeStatus.PARTIALLY_FILLED

        # Segundo reconcile: agora 100 preenchidos (API retorna fills acumulados)
        api.set_order(oids[0], "FILLED", 100.0, [
            {"fill_id": "f1", "price": 0.65, "size": 40.0, "timestamp": "2025-01-01T00:00:00+00:00"},
            {"fill_id": "f2", "price": 0.66, "size": 60.0, "timestamp": "2025-01-01T00:00:01+00:00"},
        ])
        reconciler.reconcile_open_trades(om, ledger, api)
        t = ledger.get_trade(tid)
        assert t["cumulative_filled_size"] == 100.0
        assert t["status"] == TradeStatus.FILLED
        assert abs(t["weighted_avg_price"] - (40 * 0.65 + 60 * 0.66) / 100) < 1e-6


class TestStateCorrection:
    def test_local_open_api_filled(self, om, ledger, reconciler, api):
        """Ordem OPEN localmente mas FILLED na API → corrigir."""
        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        assert om.check_order_status(oids[0])["status"] == "PENDING"

        api.set_order(oids[0], "FILLED", 50.0, [
            {"fill_id": "f1", "price": 0.65, "size": 50.0, "timestamp": "2025-01-01T00:00:00+00:00"},
        ])
        reconciler.reconcile_open_trades(om, ledger, api)

        # OrderManager deve ter corrigido
        assert om.check_order_status(oids[0])["status"] == "FILLED"
        # Ledger deve estar FILLED
        assert ledger.get_trade(tid)["status"] == TradeStatus.FILLED


class TestIdempotency:
    def test_triple_reconcile_idempotent(self, om, ledger, reconciler, api):
        """Rodar reconcile 3x com mesmos dados → estado idempotente."""
        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        api.set_order(oids[0], "FILLED", 50.0, [
            {"fill_id": "f1", "price": 0.65, "size": 50.0, "timestamp": "2025-01-01T00:00:00+00:00"},
        ])

        # Rodar 3x
        for _ in range(3):
            reconciler.reconcile_open_trades(om, ledger, api)

        t = ledger.get_trade(tid)
        assert t["cumulative_filled_size"] == 50.0
        assert t["weighted_avg_price"] == 0.65
        assert t["status"] == TradeStatus.FILLED

    def test_no_orders_skipped(self, ledger, om, reconciler, api):
        """Trade sem order_ids é ignorado."""
        tid = str(uuid.uuid4())
        ledger.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65)
        actions = reconciler.reconcile_open_trades(om, ledger, api)
        assert actions[tid] == "SKIPPED_NO_ORDERS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
