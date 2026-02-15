"""Testes integrados para SPOT-ONLY mode."""

import uuid
import shutil
import pytest
from pathlib import Path

from src.risk.position_manager import PositionManager, ShortNotAllowedError
from src.risk.capital_manager import CapitalManager, InsufficientCapitalError
from src.risk.risk_manager import RiskManager
from src.engine.trade_ledger import TradeLedger, TradeStatus
from src.execution.order_manager import OrderManager, OrderSide, OutcomeSide
from src.execution.pipeline import execute_trade_decision, resolve_trade
from src.execution.reconciler import OrderReconciler, ExchangeAPI
from src.execution.notional import compute_notional
from src.engine.probability_calibrator import (
    ProbabilityCalibrator, MarketCategory, TradeType,
)


@pytest.fixture
def ws_tmp(request):
    d = Path(__file__).parent.parent / ".test_tmp" / request.node.name
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def pm():
    return PositionManager()


@pytest.fixture
def cm():
    return CapitalManager(initial_capital=1000.0)


@pytest.fixture
def om():
    return OrderManager()


# =============================================================================
# NOTIONAL HELPER
# =============================================================================

class TestNotional:
    def test_compute_notional(self):
        assert compute_notional(100, 0.65) == 65.0

    def test_compute_notional_zero(self):
        assert compute_notional(0, 0.65) == 0.0


# =============================================================================
# ORDER MANAGER SPOT VALIDATION
# =============================================================================

class TestOrderManagerSpot:
    def test_sell_without_position_rejected(self, pm):
        om = OrderManager()
        with pytest.raises(ShortNotAllowedError):
            om.place_limit_order(
                "m1", OrderSide.SELL, 0.65, 65.0, 60,
                outcome_side=OutcomeSide.YES,
                position_manager=pm,
            )

    def test_sell_above_position_rejected(self, pm):
        om = OrderManager()
        pm.apply_fill("m1", "YES", 50, 0.65, "BUY")
        # Tentando vender 100 shares (notional=65) mas só tem 50
        with pytest.raises(ShortNotAllowedError):
            om.place_limit_order(
                "m1", OrderSide.SELL, 0.65, 65.0, 60,
                outcome_side=OutcomeSide.YES,
                position_manager=pm,
            )

    def test_sell_within_position_ok(self, pm):
        om = OrderManager()
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        # Vender 50 shares (notional=32.50)
        ids = om.place_limit_order(
            "m1", OrderSide.SELL, 0.65, 32.50, 60,
            outcome_side=OutcomeSide.YES,
            position_manager=pm,
        )
        assert len(ids) >= 1

    def test_buy_above_capital_rejected(self, pm):
        om = OrderManager()
        cm_inst = CapitalManager(50.0)
        with pytest.raises(InsufficientCapitalError):
            om.place_limit_order(
                "m1", OrderSide.BUY, 0.65, 65.0, 60,
                outcome_side=OutcomeSide.YES,
                capital_manager=cm_inst,
            )

    def test_buy_within_capital_ok(self, pm):
        om = OrderManager()
        cm_inst = CapitalManager(100.0)
        ids = om.place_limit_order(
            "m1", OrderSide.BUY, 0.65, 65.0, 60,
            outcome_side=OutcomeSide.YES,
            capital_manager=cm_inst,
        )
        assert len(ids) >= 1

    def test_no_validation_without_outcome_side(self, pm):
        """Sem outcome_side, spot validation é ignorada (backward compat)."""
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.SELL, 0.65, 10.0, 60)
        assert len(ids) >= 1


# =============================================================================
# PIPELINE SPOT BUY
# =============================================================================

class TestPipelineSpotBuy:
    def _buy_decision(self, **kw):
        base = {
            "market_id": "m1",
            "category": MarketCategory.POLITICS,
            "trade_type": TradeType.DIRECTIONAL,
            "predicted_probability": 0.65,
            "action": "BUY",
            "outcome_side": OutcomeSide.YES,
            "shares": 100.0,
            "limit_price": 0.65,
            "side": OrderSide.BUY,
            "ttl_seconds": 60,
        }
        base.update(kw)
        return base

    def test_spot_buy_dimensions_shares_via_k(self, ws_tmp):
        """k baseline = 1.0 → shares preservadas. notional = 100*0.65 = 65."""
        cm_inst = CapitalManager(1000.0)
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(
            state_path=str(ws_tmp / "l.json"),
            calibrator=cal, capital_manager=cm_inst,
        )
        tid = execute_trade_decision(
            self._buy_decision(), cal, om, led, capital_manager=cm_inst,
        )
        assert tid is not None
        t = led.get_trade(tid)
        assert t["action"] == "BUY"
        assert t["outcome_side"] == "YES"
        assert t["shares"] == 100.0  # k=1.0 → shares inalteradas
        assert abs(t["final_size"] - 65.0) < 1e-10  # notional

    def test_spot_buy_respects_available_capital(self, ws_tmp):
        """Capital insuficiente → abortar."""
        cm_inst = CapitalManager(30.0)  # Só 30, precisa de 65
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(
            state_path=str(ws_tmp / "l.json"),
            calibrator=cal, capital_manager=cm_inst,
        )
        tid = execute_trade_decision(
            self._buy_decision(), cal, om, led, capital_manager=cm_inst,
        )
        assert tid is None

    def test_spot_buy_reserves_capital(self, ws_tmp):
        cm_inst = CapitalManager(1000.0)
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(
            state_path=str(ws_tmp / "l.json"),
            calibrator=cal, capital_manager=cm_inst,
        )
        execute_trade_decision(
            self._buy_decision(), cal, om, led, capital_manager=cm_inst,
        )
        assert abs(cm_inst.get_available_capital() - 935.0) < 1e-10


# =============================================================================
# PIPELINE SPOT SELL
# =============================================================================

class TestPipelineSpotSell:
    def _sell_decision(self, **kw):
        base = {
            "market_id": "m1",
            "category": MarketCategory.POLITICS,
            "trade_type": TradeType.DIRECTIONAL,
            "action": "SELL",
            "outcome_side": OutcomeSide.YES,
            "shares": 50.0,
            "limit_price": 0.70,
            "ttl_seconds": 60,
        }
        base.update(kw)
        return base

    def test_sell_blocked_without_position(self, ws_tmp):
        pm_inst = PositionManager()
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        tid = execute_trade_decision(
            self._sell_decision(), cal, om, led, position_manager=pm_inst,
        )
        assert tid is None

    def test_sell_with_position_ok(self, ws_tmp):
        pm_inst = PositionManager()
        pm_inst.apply_fill("m1", "YES", 100, 0.65, "BUY")
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        tid = execute_trade_decision(
            self._sell_decision(), cal, om, led, position_manager=pm_inst,
        )
        assert tid is not None
        t = led.get_trade(tid)
        assert t["action"] == "SELL"
        assert t["outcome_side"] == "YES"
        assert t["shares"] == 50.0

    def test_sell_records_in_ledger(self, ws_tmp):
        pm_inst = PositionManager()
        pm_inst.apply_fill("m1", "YES", 100, 0.65, "BUY")
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        tid = execute_trade_decision(
            self._sell_decision(), cal, om, led, position_manager=pm_inst,
        )
        t = led.get_trade(tid)
        assert t["status"] == TradeStatus.OPEN
        # notional = 50 * 0.70 = 35.0
        assert abs(t["final_size"] - 35.0) < 1e-10


# =============================================================================
# RECONCILER + POSITION TRACKING
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


class TestReconcilerPositions:
    def test_buy_fill_increases_position(self, ws_tmp):
        pm_inst = PositionManager()
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        # Trade BUY: 100 shares YES @ 0.65, notional=65
        tid = str(uuid.uuid4())
        oid = "o1"
        led.record_entry(
            tid, "m1", "P", "D", 0.65, 65, 1, 65, 0.65,
            order_ids=[oid], outcome_side="YES", action="BUY", shares=100,
        )
        om.sync_order_status(oid, "PENDING", 0)

        # Fill completo na API
        api.orders[oid] = {"order_id": oid, "status": "FILLED", "filled_size": 65}
        api.fills[oid] = [{"fill_id": "f1", "price": 0.65, "size": 65, "timestamp": "2025-01-01T00:00:00+00:00"}]

        reconciler.reconcile_open_trades(
            om, led, api,
            capital_manager=cm_inst, position_manager=pm_inst,
        )

        pos = pm_inst.get_position("m1", "YES")
        assert abs(pos.shares - 100) < 1e-6
        assert abs(pos.avg_entry_price - 0.65) < 1e-6

    def test_sell_fill_reduces_position_increases_capital(self, ws_tmp):
        pm_inst = PositionManager()
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        # Posição existente: 100 shares YES @ 0.65
        pm_inst.apply_fill("m1", "YES", 100, 0.65, "BUY")
        # Simular que capital foi bloqueado via BUY anterior
        cm_inst.reserve_for_order("prev-buy", 65.0)
        cm_inst.move_to_unresolved("prev-buy", 65.0)

        # Trade SELL: 50 shares YES @ 0.70, notional=35
        tid = str(uuid.uuid4())
        oid = "o-sell"
        led.record_entry(
            tid, "m1", "P", "D", 0.0, 35, 1, 35, 0.70,
            order_ids=[oid], outcome_side="YES", action="SELL", shares=50,
        )
        om.sync_order_status(oid, "PENDING", 0)

        avail_before = cm_inst.get_available_capital()

        # Fill completo
        api.orders[oid] = {"order_id": oid, "status": "FILLED", "filled_size": 35}
        api.fills[oid] = [{"fill_id": "f1", "price": 0.70, "size": 35, "timestamp": "2025-01-01T00:00:01+00:00"}]

        reconciler.reconcile_open_trades(
            om, led, api,
            capital_manager=cm_inst, position_manager=pm_inst,
        )

        # Posição reduzida
        pos = pm_inst.get_position("m1", "YES")
        assert abs(pos.shares - 50) < 1e-6

        # Capital disponível aumentou (proceeds - cost_basis freed)
        avail_after = cm_inst.get_available_capital()
        assert avail_after > avail_before

    def test_sell_cannot_create_negative_position(self, ws_tmp):
        """Se SELL tentar negativar posição, a exceção é capturada pelo reconciler."""
        pm_inst = PositionManager()
        # Sem posição existente
        led = TradeLedger(state_path=str(ws_tmp / "ledger.json"))
        om = OrderManager()
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        tid = str(uuid.uuid4())
        oid = "o-bad-sell"
        led.record_entry(
            tid, "m1", "P", "D", 0.0, 65, 1, 65, 0.65,
            order_ids=[oid], outcome_side="YES", action="SELL", shares=100,
        )
        om.sync_order_status(oid, "PENDING", 0)

        api.orders[oid] = {"order_id": oid, "status": "FILLED", "filled_size": 65}
        api.fills[oid] = [{"fill_id": "f1", "price": 0.65, "size": 65, "timestamp": "2025-01-01T00:00:00+00:00"}]

        # Não deve crashar — erro capturado internamente
        actions = reconciler.reconcile_open_trades(
            om, led, api, position_manager=pm_inst,
        )
        assert actions[tid] == "FULLY_FILLED"


# =============================================================================
# RISK MANAGER COM SELL TRADES
# =============================================================================

class TestRiskManagerSellFilter:
    def test_sell_trades_dont_add_exposure(self, ws_tmp):
        """Trades SELL não contam como exposição positiva."""
        led = TradeLedger(state_path=str(ws_tmp / "ledger.json"))
        rm = RiskManager({"max_exposure_total": 200})

        # BUY trade: 100 de exposição
        t1 = str(uuid.uuid4())
        led.record_entry(t1, "m1", "P", "D", 0.5, 100, 1, 100, 0.5, action="BUY")

        # SELL trade: 50 — NÃO deve contar como +50
        t2 = str(uuid.uuid4())
        led.record_entry(t2, "m1", "P", "D", 0.0, 50, 1, 50, 0.7, action="SELL")

        # Exposição deve ser 100 (só BUY), não 150
        ok, _ = rm.can_open_trade(
            {"market_id": "m2", "category": "P", "final_size": 100}, led,
        )
        # 100 + 100 = 200 <= 200 → OK
        assert ok is True


# =============================================================================
# FULL LIFECYCLE
# =============================================================================

class TestFullSpotLifecycle:
    def test_buy_fill_sell_fill_capital(self, ws_tmp):
        """
        1. BUY 100 YES @ 0.65 → notional 65, reserved
        2. Fill BUY → position 100 shares, capital moved to unresolved
        3. SELL 50 YES @ 0.70 → notional 35
        4. Fill SELL → position 50 shares, capital freed
        """
        cm_inst = CapitalManager(1000.0)
        pm_inst = PositionManager()
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(
            state_path=str(ws_tmp / "l.json"),
            calibrator=cal, capital_manager=cm_inst,
        )
        api = MockExchangeAPI()
        reconciler = OrderReconciler()

        # 1. BUY via pipeline
        buy_dec = {
            "market_id": "m1", "category": MarketCategory.POLITICS,
            "trade_type": TradeType.DIRECTIONAL,
            "predicted_probability": 0.65,
            "action": "BUY", "outcome_side": OutcomeSide.YES,
            "shares": 100.0, "limit_price": 0.65,
            "side": OrderSide.BUY, "ttl_seconds": 60,
        }
        buy_tid = execute_trade_decision(
            buy_dec, cal, om, led, capital_manager=cm_inst,
        )
        assert buy_tid is not None
        assert abs(cm_inst.get_available_capital() - 935.0) < 1e-10

        # 2. Simular fill do BUY via reconciler
        buy_trade = led.get_trade(buy_tid)
        buy_oid = buy_trade["order_ids"][0]
        api.orders[buy_oid] = {
            "order_id": buy_oid, "status": "FILLED", "filled_size": 65,
        }
        api.fills[buy_oid] = [{"fill_id": "f1", "price": 0.65, "size": 65, "timestamp": "2025-01-01T00:00:00+00:00"}]

        reconciler.reconcile_open_trades(
            om, led, api,
            capital_manager=cm_inst, position_manager=pm_inst,
        )

        pos = pm_inst.get_position("m1", "YES")
        assert abs(pos.shares - 100) < 1e-6
        snap_after_buy = cm_inst.snapshot()
        assert snap_after_buy["locked_in_unresolved_markets"] == 65.0

        # 3. SELL 50 shares via pipeline
        sell_dec = {
            "market_id": "m1", "category": MarketCategory.POLITICS,
            "trade_type": TradeType.DIRECTIONAL,
            "action": "SELL", "outcome_side": OutcomeSide.YES,
            "shares": 50.0, "limit_price": 0.70, "ttl_seconds": 60,
        }
        sell_tid = execute_trade_decision(
            sell_dec, cal, om, led, position_manager=pm_inst,
        )
        assert sell_tid is not None

        # 4. Fill do SELL
        sell_trade = led.get_trade(sell_tid)
        sell_oid = sell_trade["order_ids"][0]
        api.orders[sell_oid] = {
            "order_id": sell_oid, "status": "FILLED", "filled_size": 35,
        }
        api.fills[sell_oid] = [{"fill_id": "f2", "price": 0.70, "size": 35, "timestamp": "2025-01-01T00:00:01+00:00"}]

        reconciler.reconcile_open_trades(
            om, led, api,
            capital_manager=cm_inst, position_manager=pm_inst,
        )

        # Verificar posição reduzida
        pos = pm_inst.get_position("m1", "YES")
        assert abs(pos.shares - 50) < 1e-6

        # Verificar capital: pnl do sell = proceeds - cost_basis
        # proceeds = 50 * 0.70 = 35, cost_basis = 50 * 0.65 = 32.50
        # pnl = 2.50
        snap = cm_inst.snapshot()
        assert abs(snap["realized_pnl"] - 2.50) < 1e-6
        # unresolved reduziu: 65 - 32.50 = 32.50
        assert abs(snap["locked_in_unresolved_markets"] - 32.50) < 1e-6

    def test_outcome_side_enum_handling(self, ws_tmp):
        """OutcomeSide enum é convertido para string corretamente."""
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(state_path=str(ws_tmp / "l.json"), calibrator=cal)
        pm_inst = PositionManager()
        pm_inst.apply_fill("m1", "YES", 200, 0.50, "BUY")

        sell_dec = {
            "market_id": "m1", "category": MarketCategory.POLITICS,
            "trade_type": TradeType.DIRECTIONAL,
            "action": "SELL", "outcome_side": OutcomeSide.YES,
            "shares": 50.0, "limit_price": 0.60, "ttl_seconds": 60,
        }
        tid = execute_trade_decision(
            sell_dec, cal, om, led, position_manager=pm_inst,
        )
        assert tid is not None
        t = led.get_trade(tid)
        assert t["outcome_side"] == "YES"

    def test_backward_compat_legacy_buy(self, ws_tmp):
        """Legacy BUY sem shares/limit_price funciona normalmente."""
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
        tid = execute_trade_decision(dec, cal, om, led)
        assert tid is not None
        t = led.get_trade(tid)
        assert t["shares"] is None  # Legacy: sem shares
        assert t["outcome_side"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
