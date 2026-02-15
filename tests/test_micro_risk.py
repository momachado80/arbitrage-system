"""
Testes para os 3 micro-riscos:
2.1) Replay sensor (anti-replay via last_fill_ts)
2.2) fill_id/timestamp obrigatórios (contrato forte)
2.3) Warning once-per-session (persistence_ok=False)
"""

import logging
import time
import uuid
import shutil
import pytest
from pathlib import Path

from src.execution.order_manager import OrderManager, OrderSide
from src.engine.trade_ledger import TradeLedger, TradeStatus
from src.execution.reconciler import (
    OrderReconciler, ExchangeAPI, REPLAY_SKEW_SECONDS,
)
from src.risk.safety_gate import can_execute


@pytest.fixture
def ws_tmp(request):
    d = Path(__file__).parent.parent / ".test_tmp" / request.node.name
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# =============================================================================
# MOCK API
# =============================================================================

class MockExchangeAPI(ExchangeAPI):
    def __init__(self):
        self.orders = {}
        self.fills = {}

    def get_order_status(self, order_id):
        return self.orders.get(order_id, {
            "order_id": order_id, "status": "PENDING", "filled_size": 0,
        })

    def get_fills(self, order_id):
        return self.fills.get(order_id, [])

    def get_position(self, market_id):
        return {"market_id": market_id, "size": 0, "avg_entry": 0}


# =============================================================================
# 2.1) REPLAY SENSOR
# =============================================================================

class TestReplaySensor:

    def test_replay_old_fill_ignored(self, ws_tmp):
        """Fill com timestamp muito antigo (replay) → ignorado."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        # Primeiro reconcile: fill recente (t=1000)
        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "PARTIALLY_FILLED", "filled_size": 25,
        }
        api.fills[oids[0]] = [
            {"fill_id": "f1", "price": 0.65, "size": 25.0, "timestamp": "1000"},
        ]
        reconciler.reconcile_open_trades(om, led, api)

        t = led.get_trade(tid)
        assert t["cumulative_filled_size"] == 25.0

        # Segundo reconcile: f1 (dedup OK) + f_replay (timestamp=100, muito antigo)
        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "PARTIALLY_FILLED", "filled_size": 50,
        }
        api.fills[oids[0]] = [
            {"fill_id": "f1", "price": 0.65, "size": 25.0, "timestamp": "1000"},
            {"fill_id": "f_replay", "price": 0.65, "size": 25.0, "timestamp": "100"},
        ]
        reconciler.reconcile_open_trades(om, led, api)

        # f_replay deve ter sido ignorado (replay suspect)
        # Apenas f1 contou (25.0) — e f1 é re-processado mas com dedup OK
        t = led.get_trade(tid)
        assert t["cumulative_filled_size"] == 25.0  # Sem aumento — f_replay ignorado

    def test_small_out_of_order_accepted(self, ws_tmp):
        """Fills com timestamp ligeiramente fora de ordem (<= REPLAY_SKEW) → aceitos."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        # Fills com out-of-order de ~3 segundos (dentro de REPLAY_SKEW=5)
        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "FILLED", "filled_size": 50,
        }
        api.fills[oids[0]] = [
            {"fill_id": "f1", "price": 0.65, "size": 25.0, "timestamp": "1000"},
            {"fill_id": "f2", "price": 0.65, "size": 25.0, "timestamp": "997"},  # 3s antes
        ]
        reconciler.reconcile_open_trades(om, led, api)

        t = led.get_trade(tid)
        assert t["cumulative_filled_size"] == 50.0  # Ambos aceitos
        assert t["status"] == TradeStatus.FILLED

    def test_replay_idempotent(self, ws_tmp):
        """3 reconciles com mesmos fills (incluindo replay) → idempotente."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "FILLED", "filled_size": 50,
        }
        api.fills[oids[0]] = [
            {"fill_id": "f1", "price": 0.65, "size": 25.0, "timestamp": "1000"},
            {"fill_id": "f2", "price": 0.65, "size": 25.0, "timestamp": "1001"},
        ]

        for _ in range(3):
            reconciler.reconcile_open_trades(om, led, api)

        t = led.get_trade(tid)
        assert t["cumulative_filled_size"] == 50.0
        assert t["status"] == TradeStatus.FILLED

    def test_replay_warning_once_per_trade(self, ws_tmp, caplog):
        """Warning de replay emitido apenas 1x por trade."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 100.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 100, 1, 100, 0.65, order_ids=oids)

        # Fill recente + replay antigo
        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "PARTIALLY_FILLED", "filled_size": 50,
        }
        api.fills[oids[0]] = [
            {"fill_id": "f1", "price": 0.65, "size": 25.0, "timestamp": "1000"},
            {"fill_id": "f_old", "price": 0.65, "size": 25.0, "timestamp": "100"},
        ]

        with caplog.at_level(logging.WARNING, logger="src.execution.reconciler"):
            reconciler.reconcile_open_trades(om, led, api)
            # Alterar fill_id do replay e rodar novamente
            api.fills[oids[0]] = [
                {"fill_id": "f1", "price": 0.65, "size": 25.0, "timestamp": "1000"},
                {"fill_id": "f_old2", "price": 0.65, "size": 25.0, "timestamp": "50"},
            ]
            reconciler.reconcile_open_trades(om, led, api)

        replay_warnings = [
            r for r in caplog.records
            if "REPLAY_SUSPECT" in r.message
        ]
        # Apenas 1 warning por trade (once per trade)
        assert len(replay_warnings) == 1


# =============================================================================
# 2.2) FILL_ID/TIMESTAMP OBRIGATÓRIOS (CONTRATO FORTE)
# =============================================================================

class TestFillContract:

    def test_fill_without_fill_id_rejected(self, ws_tmp):
        """Fill sem fill_id → [RECONCILE:INVALID_FILL] + reconcile_ok=False."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "FILLED", "filled_size": 50,
        }
        # Fill sem fill_id!
        api.fills[oids[0]] = [
            {"price": 0.65, "size": 50.0, "timestamp": "2025-01-01T00:00:00+00:00"},
        ]

        reconciler.reconcile_open_trades(om, led, api)

        assert led.reconcile_ok is False
        # Fill não foi aplicado
        t = led.get_trade(tid)
        assert t["cumulative_filled_size"] == 0.0

    def test_fill_with_empty_fill_id_rejected(self, ws_tmp):
        """Fill com fill_id vazio → rejeitado."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "FILLED", "filled_size": 50,
        }
        api.fills[oids[0]] = [
            {"fill_id": "", "price": 0.65, "size": 50.0, "timestamp": "2025-01-01T00:00:00+00:00"},
        ]

        reconciler.reconcile_open_trades(om, led, api)
        assert led.reconcile_ok is False

    def test_fill_without_timestamp_rejected(self, ws_tmp):
        """Fill sem timestamp → [RECONCILE:INVALID_FILL] + reconcile_ok=False."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "FILLED", "filled_size": 50,
        }
        # Fill com fill_id mas SEM timestamp
        api.fills[oids[0]] = [
            {"fill_id": "f1", "price": 0.65, "size": 50.0},
        ]

        reconciler.reconcile_open_trades(om, led, api)

        assert led.reconcile_ok is False
        t = led.get_trade(tid)
        assert t["cumulative_filled_size"] == 0.0

    def test_valid_fill_accepted(self, ws_tmp):
        """Fill válido (fill_id + timestamp) → processado normalmente."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "FILLED", "filled_size": 50,
        }
        api.fills[oids[0]] = [
            {"fill_id": "f1", "price": 0.65, "size": 50.0, "timestamp": "2025-01-01T00:00:00+00:00"},
        ]

        reconciler.reconcile_open_trades(om, led, api)

        assert led.reconcile_ok is True
        t = led.get_trade(tid)
        assert t["cumulative_filled_size"] == 50.0
        assert t["status"] == TradeStatus.FILLED

    def test_epoch_timestamp_accepted(self, ws_tmp):
        """Fill com timestamp numérico (epoch) → aceito."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "FILLED", "filled_size": 50,
        }
        api.fills[oids[0]] = [
            {"fill_id": "f1", "price": 0.65, "size": 50.0, "timestamp": 1704067200.0},
        ]

        reconciler.reconcile_open_trades(om, led, api)
        assert led.reconcile_ok is True
        assert led.get_trade(tid)["cumulative_filled_size"] == 50.0

    def test_reconcile_failed_blocks_buy_via_safety_gate(self, ws_tmp):
        """reconcile_ok=False → safety_gate bloqueia BUY."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        led.mark_reconcile_failed()

        ok, reason = can_execute("BUY", trade_ledger=led)
        assert ok is False
        assert "reconcile" in reason.lower()

    def test_reconcile_failed_allows_sell(self, ws_tmp):
        """reconcile_ok=False → SELL ainda permitido."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        led.mark_reconcile_failed()

        ok, _ = can_execute("SELL", trade_ledger=led)
        assert ok is True


# =============================================================================
# 2.3) WARNING ONCE PER SESSION
# =============================================================================

class TestPersistenceWarning:

    def test_warning_emitted_once(self, ws_tmp, caplog):
        """persistence_ok=False → warning emitido 1x."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        led._persistence_ok = False

        with caplog.at_level(logging.WARNING, logger="src.engine.trade_ledger"):
            led.ensure_persistence_warning_logged()
            led.ensure_persistence_warning_logged()
            led.ensure_persistence_warning_logged()

        warnings = [
            r for r in caplog.records
            if "PERSISTENCE_DISABLED" in r.message
        ]
        assert len(warnings) == 1

    def test_no_warning_when_persistence_ok(self, ws_tmp, caplog):
        """persistence_ok=True → nenhum warning."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        assert led.persistence_ok is True

        with caplog.at_level(logging.WARNING, logger="src.engine.trade_ledger"):
            led.ensure_persistence_warning_logged()

        warnings = [
            r for r in caplog.records
            if "PERSISTENCE_DISABLED" in r.message
        ]
        assert len(warnings) == 0

    def test_warning_flag_persists(self, ws_tmp):
        """Após emitir, _persistence_warning_emitted fica True."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        led._persistence_ok = False
        assert led._persistence_warning_emitted is False

        led.ensure_persistence_warning_logged()
        assert led._persistence_warning_emitted is True


# =============================================================================
# INTEGRAÇÃO: fill_id/timestamp + replay + safety gate
# =============================================================================

class TestIntegration:

    def test_mixed_valid_and_invalid_fills(self, ws_tmp):
        """Mix de fills válidos e inválidos → válidos aplicados, flag ativada."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "FILLED", "filled_size": 50,
        }
        api.fills[oids[0]] = [
            # Fill válido
            {"fill_id": "f1", "price": 0.65, "size": 25.0, "timestamp": "2025-01-01T00:00:00+00:00"},
            # Fill sem timestamp (inválido)
            {"fill_id": "f2", "price": 0.65, "size": 25.0},
        ]

        reconciler.reconcile_open_trades(om, led, api)

        # Fill válido aplicado
        t = led.get_trade(tid)
        assert t["cumulative_filled_size"] == 25.0

        # Mas flag de segurança ativada por f2
        assert led.reconcile_ok is False

    def test_epoch_string_timestamp(self, ws_tmp):
        """Timestamp como string epoch → aceito."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "FILLED", "filled_size": 50,
        }
        api.fills[oids[0]] = [
            {"fill_id": "f1", "price": 0.65, "size": 50.0, "timestamp": "1704067200"},
        ]

        reconciler.reconcile_open_trades(om, led, api)
        assert led.reconcile_ok is True
        assert led.get_trade(tid)["cumulative_filled_size"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
