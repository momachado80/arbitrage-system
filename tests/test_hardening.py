"""
Testes para os 4 hardenings pré-produção:
A) Precision/rounding/tolerâncias
B) Deduplicação por fill_id
C) PositionManager persistência atômica
D) Kill switch operacional
"""

import json
import shutil
import uuid
import pytest
from pathlib import Path

from src.utils.precision import (
    round_price, round_shares, round_notional, is_close,
    FILL_EPS, PRICE_DECIMALS, SHARES_DECIMALS, NOTIONAL_DECIMALS,
)
from src.risk.position_manager import PositionManager, ShortNotAllowedError
from src.risk.capital_manager import CapitalManager
from src.risk.safety_gate import can_execute
from src.engine.trade_ledger import TradeLedger, TradeStatus
from src.execution.order_manager import OrderManager, OrderSide, OutcomeSide
from src.execution.reconciler import OrderReconciler, ExchangeAPI
from src.execution.notional import compute_notional
from src.execution.pipeline import execute_trade_decision
from src.engine.probability_calibrator import (
    ProbabilityCalibrator, MarketCategory, TradeType,
)


@pytest.fixture
def ws_tmp(request):
    d = Path(__file__).parent.parent / ".test_tmp" / request.node.name
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# =============================================================================
# A) PRECISION / ROUNDING / TOLERÂNCIAS
# =============================================================================

class TestPrecisionFunctions:
    def test_round_price(self):
        assert round_price(0.6666666666) == round(0.6666666666, PRICE_DECIMALS)
        assert round_price(0.5) == 0.5

    def test_round_shares(self):
        assert round_shares(100.123456789012) == round(100.123456789012, SHARES_DECIMALS)
        assert round_shares(100.0) == 100.0

    def test_round_notional(self):
        assert round_notional(65.1234567) == round(65.1234567, NOTIONAL_DECIMALS)
        assert round_notional(65.0) == 65.0

    def test_is_close(self):
        assert is_close(1.0, 1.0 + 1e-9) is True
        assert is_close(1.0, 1.0 + 1e-7) is False
        assert is_close(1.0, 1.01, eps=0.02) is True

    def test_compute_notional_rounded(self):
        # 100 * 0.65 = 65.0 (exact)
        assert compute_notional(100, 0.65) == 65.0
        # 33.3333 * 0.3 = 9.99999 (rounded to 6 decimals)
        result = compute_notional(33.3333, 0.3)
        assert result == round_notional(33.3333 * 0.3)


class TestPrecisionSellAlmostZero:
    """SELL que quase zera por float não fica negativo."""

    def test_sell_near_zero_float(self):
        pm = PositionManager()
        pm.apply_fill("m1", "YES", 100.0, 0.65, "BUY")
        # Vender 99.99999999 (quase tudo)
        pm.apply_fill("m1", "YES", 99.99999999, 0.70, "SELL")
        pos = pm.get_position("m1", "YES")
        assert pos.shares >= 0  # NUNCA negativo
        # O valor restante é tão pequeno que deve ser clampado para 0
        assert pos.shares == 0.0 or pos.shares < FILL_EPS

    def test_sell_exact_quantity_via_float(self):
        pm = PositionManager()
        pm.apply_fill("m1", "YES", 100.0, 0.65, "BUY")
        pm.apply_fill("m1", "YES", 100.0, 0.70, "SELL")
        pos = pm.get_position("m1", "YES")
        assert pos.shares == 0.0
        assert pos.avg_entry_price == 0.0


class TestPrecisionFilledEpsilon:
    """FILLED quando filled = target - tiny_eps."""

    def test_filled_with_tiny_shortfall(self, ws_tmp):
        """cumulative_filled ligeiramente menor que final_size → FILLED."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.5, 100, 1, 100, 0.5)

        # Simular fill de 99.999999999 (< 100 por ruído float)
        led.update_fill_status(tid, 99.999999999, 0.5)
        t = led.get_trade(tid)
        assert t["status"] == TradeStatus.FILLED

    def test_not_filled_with_real_shortfall(self, ws_tmp):
        """cumulative_filled significativamente menor que final_size → OPEN."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.5, 100, 1, 100, 0.5)

        led.update_fill_status(tid, 99.0, 0.5)
        t = led.get_trade(tid)
        assert t["status"] == TradeStatus.PARTIALLY_FILLED


# =============================================================================
# B) DEDUPLICAÇÃO POR fill_id
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


class TestFillDeduplication:
    def test_same_fill_id_twice_applies_once(self, ws_tmp):
        """Mesmo fill_id retornado 2x na mesma resposta → aplica 1x."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        oids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 50.0, 60)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "P", "D", 0.65, 50, 1, 50, 0.65, order_ids=oids)

        # API retorna fill duplicado
        api.orders[oids[0]] = {
            "order_id": oids[0], "status": "FILLED", "filled_size": 50,
        }
        api.fills[oids[0]] = [
            {"fill_id": "f1", "price": 0.65, "size": 50.0, "timestamp": "2025-01-01T00:00:00+00:00"},
            {"fill_id": "f1", "price": 0.65, "size": 50.0, "timestamp": "2025-01-01T00:00:00+00:00"},  # DUPLICATA
        ]

        reconciler.reconcile_open_trades(om, led, api)
        t = led.get_trade(tid)
        # Deve contar 50 (1x), não 100 (2x)
        assert t["cumulative_filled_size"] == 50.0

    def test_different_fill_ids_both_applied(self, ws_tmp):
        """Fill IDs diferentes → ambos aplicados."""
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
            {"fill_id": "f1", "price": 0.65, "size": 25.0, "timestamp": "2025-01-01T00:00:00+00:00"},
            {"fill_id": "f2", "price": 0.66, "size": 25.0, "timestamp": "2025-01-01T00:00:01+00:00"},
        ]

        reconciler.reconcile_open_trades(om, led, api)
        t = led.get_trade(tid)
        assert t["cumulative_filled_size"] == 50.0
        assert t["status"] == TradeStatus.FILLED

    def test_idempotent_with_duplicates(self, ws_tmp):
        """3 reconciles com mesmos dados + duplicatas → idempotente."""
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
            {"fill_id": "f1", "price": 0.65, "size": 25.0, "timestamp": "2025-01-01T00:00:00+00:00"},
            {"fill_id": "f1", "price": 0.65, "size": 25.0, "timestamp": "2025-01-01T00:00:00+00:00"},  # DUPLICATA
            {"fill_id": "f2", "price": 0.66, "size": 25.0, "timestamp": "2025-01-01T00:00:01+00:00"},
        ]

        for _ in range(3):
            reconciler.reconcile_open_trades(om, led, api)

        t = led.get_trade(tid)
        assert t["cumulative_filled_size"] == 50.0
        expected_avg = (25 * 0.65 + 25 * 0.66) / 50
        assert abs(t["weighted_avg_price"] - expected_avg) < 1e-6
        assert t["status"] == TradeStatus.FILLED

    def test_fill_tracking_audit_trail(self, ws_tmp):
        """Fill IDs registrados como audit trail no TradeLedger."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        tid = "trade-1"
        led.mark_fill_applied(tid, "f1")
        led.mark_fill_applied(tid, "f2")

        assert led.is_fill_applied(tid, "f1") is True
        assert led.is_fill_applied(tid, "f2") is True
        assert led.is_fill_applied(tid, "f3") is False
        assert led.is_fill_applied("other", "f1") is False


# =============================================================================
# C) POSITIONMANAGER PERSISTÊNCIA ATÔMICA
# =============================================================================

class TestPositionPersistence:
    def test_save_load_roundtrip(self, ws_tmp):
        """Persistência roundtrip: save → load mantém posições."""
        path = str(ws_tmp / "pos.json")
        pm1 = PositionManager(state_path=path)
        pm1.apply_fill("m1", "YES", 100, 0.65, "BUY")
        pm1.apply_fill("m2", "NO", 50, 0.35, "BUY")

        # Carregar nova instância do mesmo arquivo
        pm2 = PositionManager(state_path=path)
        pos1 = pm2.get_position("m1", "YES")
        pos2 = pm2.get_position("m2", "NO")
        assert pos1.shares == 100
        assert abs(pos1.avg_entry_price - 0.65) < 1e-6
        assert pos2.shares == 50
        assert abs(pos2.avg_entry_price - 0.35) < 1e-6

    def test_nonexistent_file_no_corruption(self, ws_tmp):
        """Arquivo inexistente NÃO ativa corrupção."""
        pm = PositionManager(state_path=str(ws_tmp / "new_pos.json"))
        assert pm.is_corrupted() is False
        assert pm.safety_mode is False

    def test_corrupted_file_activates_safety(self, ws_tmp):
        """Arquivo corrompido → safety_mode + corrupted."""
        path = ws_tmp / "pos.json"
        path.write_text("NOT VALID JSON {{{")

        pm = PositionManager(state_path=str(path))
        assert pm.is_corrupted() is True
        assert pm.safety_mode is True

    def test_wrong_schema_activates_safety(self, ws_tmp):
        """Schema version errado → safety."""
        path = ws_tmp / "pos.json"
        from datetime import datetime, timezone
        path.write_text(json.dumps({
            "schema_version": 999,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "positions": {},
        }))

        pm = PositionManager(state_path=str(path))
        assert pm.is_corrupted() is True

    def test_save_blocked_when_corrupted(self, ws_tmp):
        """save_state() bloqueado quando corrompido."""
        path = ws_tmp / "pos.json"
        path.write_text("CORRUPT")
        original_content = path.read_text()

        pm = PositionManager(state_path=str(path))
        assert pm.is_corrupted() is True

        # Tentativa de save — deve ser bloqueada
        pm.save_state()
        assert path.read_text() == original_content  # Inalterado

    def test_reset_all_restores_persistence(self, ws_tmp):
        """reset_all() restaura possibilidade de salvar."""
        path = ws_tmp / "pos.json"
        path.write_text("CORRUPT")

        pm = PositionManager(state_path=str(path))
        assert pm.is_corrupted() is True

        pm.reset_all()
        assert pm.is_corrupted() is False
        assert pm.safety_mode is False

        # Arquivo recriado corretamente
        data = json.loads(path.read_text())
        assert data["schema_version"] == 1
        assert data["positions"] == {}

    def test_recover_from_corruption_restores(self, ws_tmp):
        """recover_from_corruption() mantém estado in-memory e restaura save."""
        path = ws_tmp / "pos.json"
        path.write_text("CORRUPT")

        pm = PositionManager(state_path=str(path))
        assert pm.is_corrupted() is True

        # Adicionar posição in-memory (mesmo corrompido, apply_fill funciona)
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")

        pm.recover_from_corruption()
        assert pm.is_corrupted() is False

        # Posição preservada e salva
        data = json.loads(path.read_text())
        assert "m1|YES" in data["positions"]
        assert data["positions"]["m1|YES"]["shares"] == 100

    def test_created_at_immutable(self, ws_tmp):
        """created_at é definido na criação e nunca muda."""
        path = str(ws_tmp / "pos.json")
        pm = PositionManager(state_path=path)
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")

        data1 = json.loads(Path(path).read_text())
        created1 = data1["created_at"]
        updated1 = data1["updated_at"]

        # Segunda operação
        pm.apply_fill("m1", "YES", 50, 0.70, "BUY")
        data2 = json.loads(Path(path).read_text())

        assert data2["created_at"] == created1  # Imutável
        assert data2["updated_at"] >= updated1  # Atualizado

    def test_missing_created_at_activates_safety(self, ws_tmp):
        """Arquivo sem created_at → safety_mode."""
        path = ws_tmp / "pos.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "updated_at": "2024-01-01T00:00:00+00:00",
            "positions": {},
        }))

        pm = PositionManager(state_path=str(path))
        assert pm.is_corrupted() is True

    def test_negative_shares_activates_safety(self, ws_tmp):
        """Shares negativas no arquivo → safety_mode."""
        from datetime import datetime, timezone
        path = ws_tmp / "pos.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "positions": {"m1|YES": {"shares": -10, "avg_entry_price": 0.5}},
        }))

        pm = PositionManager(state_path=str(path))
        assert pm.is_corrupted() is True

    def test_no_persist_mode(self):
        """Sem state_path → in-memory, sem persistência, sem crash."""
        pm = PositionManager()
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        assert pm.get_position("m1", "YES").shares == 100
        assert pm.is_corrupted() is False


# =============================================================================
# D) KILL SWITCH OPERACIONAL
# =============================================================================

class TestKillSwitch:
    def test_buy_blocked_when_position_corrupted(self, ws_tmp):
        """position_manager corrompido → BUY bloqueado."""
        path = ws_tmp / "pos.json"
        path.write_text("CORRUPT")
        pm = PositionManager(state_path=str(path))

        ok, reason = can_execute("BUY", position_manager=pm)
        assert ok is False
        assert "position_manager" in reason

    def test_sell_allowed_when_position_corrupted(self, ws_tmp):
        """position_manager corrompido → SELL ainda permitido."""
        path = ws_tmp / "pos.json"
        path.write_text("CORRUPT")
        pm = PositionManager(state_path=str(path))

        ok, _ = can_execute("SELL", position_manager=pm)
        assert ok is True

    def test_buy_blocked_when_calibrator_safety(self, ws_tmp):
        """calibrator em safety_mode → BUY bloqueado."""
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        cal._safety_mode = True

        ok, reason = can_execute("BUY", calibrator=cal)
        assert ok is False
        assert "calibrator" in reason

    def test_buy_blocked_when_capital_corrupted(self):
        """capital_manager corrompido → BUY bloqueado."""
        cm = CapitalManager(1000)
        cm.mark_corrupted()

        ok, reason = can_execute("BUY", capital_manager=cm)
        assert ok is False
        assert "capital_manager" in reason

    def test_buy_allowed_normal_state(self, ws_tmp):
        """Estado normal → BUY permitido."""
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        cm = CapitalManager(1000)
        pm = PositionManager()

        ok, _ = can_execute("BUY", calibrator=cal, capital_manager=cm, position_manager=pm)
        assert ok is True

    def test_buy_blocked_when_ledger_persistence_failed(self, ws_tmp):
        """trade_ledger persistence failed → BUY bloqueado."""
        led = TradeLedger(state_path=str(ws_tmp / "led.json"))
        led._persistence_ok = False

        ok, reason = can_execute("BUY", trade_ledger=led)
        assert ok is False
        assert "trade_ledger" in reason

    def test_kill_switch_in_pipeline_buy(self, ws_tmp):
        """Pipeline bloqueia BUY quando position_manager corrompido."""
        path = ws_tmp / "pos.json"
        path.write_text("CORRUPT")
        pm = PositionManager(state_path=str(path))
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)

        dec = {
            "market_id": "m1", "category": MarketCategory.POLITICS,
            "trade_type": TradeType.DIRECTIONAL,
            "predicted_probability": 0.65,
            "base_size": 100, "price": 0.65,
            "side": OrderSide.BUY, "ttl_seconds": 60,
        }
        tid = execute_trade_decision(dec, cal, om, led, position_manager=pm)
        assert tid is None  # Bloqueado pelo kill switch

    def test_kill_switch_allows_sell_in_pipeline(self, ws_tmp):
        """Pipeline permite SELL mesmo com position_manager corrompido."""
        path = ws_tmp / "pos.json"
        path.write_text("CORRUPT")
        pm = PositionManager(state_path=str(path))

        # Adicionar posição in-memory para permitir SELL
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")

        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)

        sell_dec = {
            "market_id": "m1", "category": MarketCategory.POLITICS,
            "trade_type": TradeType.DIRECTIONAL,
            "action": "SELL", "outcome_side": OutcomeSide.YES,
            "shares": 50.0, "limit_price": 0.70, "ttl_seconds": 60,
        }
        tid = execute_trade_decision(sell_dec, cal, om, led, position_manager=pm)
        assert tid is not None  # SELL permitido


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
