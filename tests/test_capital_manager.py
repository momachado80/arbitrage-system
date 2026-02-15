"""Testes para CapitalManager e integrações."""

import uuid
import shutil
import pytest
from pathlib import Path

from src.risk.capital_manager import CapitalManager, InsufficientCapitalError
from src.risk.risk_manager import RiskManager
from src.engine.trade_ledger import TradeLedger, TradeStatus
from src.execution.pipeline import execute_trade_decision, resolve_trade
from src.engine.probability_calibrator import (
    ProbabilityCalibrator, MarketCategory, TradeType,
)
from src.execution.order_manager import OrderManager, OrderSide
from src.execution.reconciler import OrderReconciler, ExchangeAPI


@pytest.fixture
def ws_tmp(request):
    d = Path(__file__).parent.parent / ".test_tmp" / request.node.name
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def cm():
    return CapitalManager(initial_capital=1000.0)


# =============================================================================
# 1. RESERVA REDUZ AVAILABLE_CAPITAL
# =============================================================================

class TestReserve:
    def test_reserve_reduces_available(self, cm):
        cm.reserve_for_order("t1", 200.0)
        assert cm.get_available_capital() == 800.0

    def test_reserve_idempotent(self, cm):
        cm.reserve_for_order("t1", 200.0)
        cm.reserve_for_order("t1", 200.0)  # Idempotente
        assert cm.get_available_capital() == 800.0

    def test_multiple_reserves(self, cm):
        cm.reserve_for_order("t1", 200.0)
        cm.reserve_for_order("t2", 300.0)
        assert cm.get_available_capital() == 500.0
        snap = cm.snapshot()
        assert snap["locked_in_open_orders"] == 500.0

    def test_snapshot_after_reserve(self, cm):
        cm.reserve_for_order("t1", 400.0)
        snap = cm.snapshot()
        assert snap["initial_capital"] == 1000.0
        assert snap["total_capital"] == 1000.0
        assert snap["locked_in_open_orders"] == 400.0
        assert snap["locked_in_unresolved_markets"] == 0.0
        assert snap["available_capital"] == 600.0
        assert snap["realized_pnl"] == 0.0


# =============================================================================
# 2. PARTIAL FILL NÃO LIBERA CAPITAL PREMATURAMENTE
# =============================================================================

class TestPartialFill:
    def test_partial_fill_keeps_reserve(self, cm):
        """Partial fill não altera reserva de capital — capital permanece em open_orders."""
        cm.reserve_for_order("t1", 200.0)
        # Partial fill: capital ainda está em open_orders (o CapitalManager não
        # é notificado de partial fills, apenas de full fills via move_to_unresolved)
        assert cm.get_available_capital() == 800.0
        assert cm.snapshot()["locked_in_open_orders"] == 200.0

    def test_partial_fill_with_ledger(self, ws_tmp):
        """Via TradeLedger, partial fill mantém capital em open_orders."""
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "C", "D", 0.5, 100, 1, 100, 0.5)
        # Partial fill: 50 de 100
        led.update_fill_status(tid, 50.0, 0.51)
        assert led.get_trade(tid)["status"] == TradeStatus.PARTIALLY_FILLED
        # Capital ainda em open_orders (não moveu para unresolved)
        snap = cm_inst.snapshot()
        assert snap["locked_in_open_orders"] == 100.0
        assert snap["locked_in_unresolved_markets"] == 0.0
        assert snap["available_capital"] == 900.0


# =============================================================================
# 3. CANCELAMENTO LIBERA CAPITAL
# =============================================================================

class TestCancelReleasesCapital:
    def test_cancel_releases(self, cm):
        cm.reserve_for_order("t1", 200.0)
        cm.release_order_reserve("t1")
        assert cm.get_available_capital() == 1000.0

    def test_cancel_via_ledger(self, ws_tmp):
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "C", "D", 0.5, 100, 1, 100, 0.5)
        assert cm_inst.get_available_capital() == 900.0
        led.cancel_trade(tid)
        assert cm_inst.get_available_capital() == 1000.0

    def test_release_nonexistent_idempotent(self, cm):
        cm.release_order_reserve("fake_id")  # Deve ser no-op
        assert cm.get_available_capital() == 1000.0


# =============================================================================
# 4. TRADE RESOLVIDO ATUALIZA REALIZED_PNL
# =============================================================================

class TestResolve:
    def test_resolve_win(self, cm):
        """Outcome correto: pnl = 0 (break-even, capital retorna)."""
        cm.reserve_for_order("t1", 200.0)
        cm.move_to_unresolved("t1", 200.0)
        cm.resolve_trade("t1", 0.0)  # pnl = 0 (outcome=1: final_size - final_size)
        snap = cm.snapshot()
        assert snap["locked_in_unresolved_markets"] == 0.0
        assert snap["realized_pnl"] == 0.0
        assert snap["available_capital"] == 1000.0

    def test_resolve_loss(self, cm):
        """Outcome incorreto: pnl = -final_size (perda total)."""
        cm.reserve_for_order("t1", 200.0)
        cm.move_to_unresolved("t1", 200.0)
        cm.resolve_trade("t1", -200.0)  # pnl = 0 - 200
        snap = cm.snapshot()
        assert snap["locked_in_unresolved_markets"] == 0.0
        assert snap["realized_pnl"] == -200.0
        assert snap["total_capital"] == 800.0
        assert snap["available_capital"] == 800.0

    def test_resolve_via_ledger(self, ws_tmp):
        """Resolução via TradeLedger atualiza CapitalManager."""
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "C", "D", 0.5, 100, 1, 100, 0.5)
        led.update_fill_status(tid, 100.0, 0.50)  # → FILLED → move_to_unresolved
        assert cm_inst.snapshot()["locked_in_unresolved_markets"] == 100.0
        # Resolver com outcome=0 (perda)
        led.record_resolution(tid, 0)
        snap = cm_inst.snapshot()
        assert snap["realized_pnl"] == -100.0
        assert snap["locked_in_unresolved_markets"] == 0.0
        assert snap["total_capital"] == 900.0

    def test_resolve_win_via_ledger(self, ws_tmp):
        """Resolução com outcome=1 (break-even)."""
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "C", "D", 0.5, 100, 1, 100, 0.5)
        led.update_fill_status(tid, 100.0, 0.50)
        led.record_resolution(tid, 1)
        snap = cm_inst.snapshot()
        assert snap["realized_pnl"] == 0.0
        assert snap["available_capital"] == 1000.0


# =============================================================================
# 5. NÃO PERMITE RESERVAR ACIMA DO DISPONÍVEL
# =============================================================================

class TestInsufficientCapital:
    def test_reserve_exceeds_available(self, cm):
        with pytest.raises(InsufficientCapitalError):
            cm.reserve_for_order("t1", 1500.0)

    def test_reserve_exceeds_after_partial(self, cm):
        cm.reserve_for_order("t1", 800.0)
        with pytest.raises(InsufficientCapitalError):
            cm.reserve_for_order("t2", 300.0)  # 800 + 300 > 1000

    def test_reserve_exact_available(self, cm):
        cm.reserve_for_order("t1", 1000.0)  # Exato
        assert cm.get_available_capital() == 0.0
        with pytest.raises(InsufficientCapitalError):
            cm.reserve_for_order("t2", 0.01)


# =============================================================================
# 6. DEADZONE BLOQUEIA NOVOS TRADES
# =============================================================================

class TestDeadzoneWithCapitalManager:
    def test_deadzone_blocks_via_risk_manager(self, ws_tmp):
        """locked_in_unresolved / total_capital > max_deadzone → bloqueado."""
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        rm = RiskManager({
            "max_percent_capital_deadzone": 0.3,
            "max_exposure_total": 10000,
        })
        # Criar trade e mover para unresolved (simula FILLED)
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "C", "D", 0.5, 350, 1, 350, 0.5)
        led.update_fill_status(tid, 350.0, 0.50)
        # 350/1000 = 0.35 > 0.3
        ok, reason = rm.can_open_trade(
            {"market_id": "m2", "category": "C", "final_size": 100},
            led,
            capital_manager=cm_inst,
        )
        assert ok is False
        assert "deadzone" in reason.lower()

    def test_deadzone_ok(self, ws_tmp):
        """Dentro do limite de deadzone → permitido."""
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        rm = RiskManager({
            "max_percent_capital_deadzone": 0.5,
            "max_exposure_total": 10000,
        })
        tid = str(uuid.uuid4())
        led.record_entry(tid, "m1", "C", "D", 0.5, 200, 1, 200, 0.5)
        led.update_fill_status(tid, 200.0, 0.50)
        # 200/1000 = 0.2 < 0.5
        ok, _ = rm.can_open_trade(
            {"market_id": "m2", "category": "C", "final_size": 100},
            led,
            capital_manager=cm_inst,
        )
        assert ok is True

    def test_capital_insufficient_blocks(self, ws_tmp):
        """Risk manager bloqueia quando capital insuficiente."""
        cm_inst = CapitalManager(500.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        rm = RiskManager({"max_exposure_total": 10000})
        cm_inst.reserve_for_order("existing", 400.0)
        ok, reason = rm.can_open_trade(
            {"market_id": "m1", "category": "C", "final_size": 200},
            led,
            capital_manager=cm_inst,
        )
        assert ok is False
        assert "insuficiente" in reason.lower()


# =============================================================================
# 7. IDEMPOTÊNCIA: DOUBLE RESOLVE NÃO DUPLICA PNL
# =============================================================================

class TestIdempotency:
    def test_double_resolve(self, cm):
        cm.reserve_for_order("t1", 200.0)
        cm.move_to_unresolved("t1", 200.0)
        cm.resolve_trade("t1", -200.0)
        cm.resolve_trade("t1", -200.0)  # Segunda chamada: idempotente
        assert cm.snapshot()["realized_pnl"] == -200.0

    def test_double_reserve(self, cm):
        cm.reserve_for_order("t1", 200.0)
        cm.reserve_for_order("t1", 200.0)
        assert cm.snapshot()["locked_in_open_orders"] == 200.0

    def test_double_move_to_unresolved(self, cm):
        cm.reserve_for_order("t1", 200.0)
        cm.move_to_unresolved("t1", 200.0)
        cm.move_to_unresolved("t1", 200.0)
        assert cm.snapshot()["locked_in_unresolved_markets"] == 200.0

    def test_double_release(self, cm):
        cm.reserve_for_order("t1", 200.0)
        cm.release_order_reserve("t1")
        cm.release_order_reserve("t1")  # Já liberado
        assert cm.get_available_capital() == 1000.0


# =============================================================================
# 8. SEQUÊNCIA COMPLETA: OPEN → PARTIAL → FULL FILL → RESOLVE
# =============================================================================

class TestFullSequence:
    def test_complete_lifecycle_loss(self, ws_tmp):
        """open → partial fill → full fill → resolve (loss) → capital correto."""
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        tid = str(uuid.uuid4())

        # 1. Open: reserva 200
        led.record_entry(tid, "m1", "C", "D", 0.5, 200, 1, 200, 0.5)
        assert cm_inst.get_available_capital() == 800.0
        assert cm_inst.snapshot()["locked_in_open_orders"] == 200.0

        # 2. Partial fill (100/200): capital permanece em open_orders
        led.update_fill_status(tid, 100.0, 0.51)
        assert led.get_trade(tid)["status"] == TradeStatus.PARTIALLY_FILLED
        assert cm_inst.snapshot()["locked_in_open_orders"] == 200.0
        assert cm_inst.snapshot()["locked_in_unresolved_markets"] == 0.0
        assert cm_inst.get_available_capital() == 800.0

        # 3. Full fill (200/200): move para unresolved
        led.update_fill_status(tid, 200.0, 0.50)
        assert led.get_trade(tid)["status"] == TradeStatus.FILLED
        assert cm_inst.snapshot()["locked_in_open_orders"] == 0.0
        assert cm_inst.snapshot()["locked_in_unresolved_markets"] == 200.0
        assert cm_inst.get_available_capital() == 800.0

        # 4. Resolve (loss): pnl = -200
        led.record_resolution(tid, 0)
        snap = cm_inst.snapshot()
        assert snap["locked_in_open_orders"] == 0.0
        assert snap["locked_in_unresolved_markets"] == 0.0
        assert snap["realized_pnl"] == -200.0
        assert snap["total_capital"] == 800.0
        assert snap["available_capital"] == 800.0

    def test_complete_lifecycle_win(self, ws_tmp):
        """open → full fill → resolve (win) → capital retorna."""
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        tid = str(uuid.uuid4())

        led.record_entry(tid, "m1", "C", "D", 0.5, 300, 1, 300, 0.5)
        led.update_fill_status(tid, 300.0, 0.50)
        led.record_resolution(tid, 1)  # Win: pnl = 0

        snap = cm_inst.snapshot()
        assert snap["realized_pnl"] == 0.0
        assert snap["total_capital"] == 1000.0
        assert snap["available_capital"] == 1000.0

    def test_multiple_trades_lifecycle(self, ws_tmp):
        """Múltiplos trades simultâneos com resultados mistos."""
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )

        # Trade 1: 200, win
        t1 = str(uuid.uuid4())
        led.record_entry(t1, "m1", "C", "D", 0.5, 200, 1, 200, 0.5)
        # Trade 2: 300, loss
        t2 = str(uuid.uuid4())
        led.record_entry(t2, "m2", "C", "D", 0.5, 300, 1, 300, 0.5)

        assert cm_inst.get_available_capital() == 500.0

        # Fill ambos
        led.update_fill_status(t1, 200.0, 0.50)
        led.update_fill_status(t2, 300.0, 0.50)
        assert cm_inst.snapshot()["locked_in_unresolved_markets"] == 500.0

        # Resolver t1 (win) e t2 (loss)
        led.record_resolution(t1, 1)
        led.record_resolution(t2, 0)

        snap = cm_inst.snapshot()
        assert snap["realized_pnl"] == -300.0  # 0 + (-300)
        assert snap["total_capital"] == 700.0
        assert snap["available_capital"] == 700.0


# =============================================================================
# INTEGRAÇÃO COM PIPELINE
# =============================================================================

class TestPipelineCapitalIntegration:
    def _dec(self, **kw):
        base = {
            "market_id": "m1", "category": MarketCategory.POLITICS,
            "trade_type": TradeType.DIRECTIONAL, "predicted_probability": 0.65,
            "base_size": 100.0, "price": 0.65, "side": OrderSide.BUY,
            "ttl_seconds": 60,
        }
        base.update(kw)
        return base

    def test_pipeline_reserves_capital(self, ws_tmp):
        """Pipeline reserva capital antes de colocar ordens."""
        cm_inst = CapitalManager(1000.0)
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(
            state_path=str(ws_tmp / "l.json"),
            calibrator=cal,
            capital_manager=cm_inst,
        )
        tid = execute_trade_decision(
            self._dec(), cal, om, led, capital_manager=cm_inst,
        )
        assert tid is not None
        # final_size = 100 * 1.0 (k baseline) = 100
        assert cm_inst.get_available_capital() == 900.0
        assert cm_inst.snapshot()["locked_in_open_orders"] == 100.0

    def test_pipeline_blocks_insufficient_capital(self, ws_tmp):
        """Pipeline aborta quando capital insuficiente."""
        cm_inst = CapitalManager(50.0)  # Só 50 disponível
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(
            state_path=str(ws_tmp / "l.json"),
            calibrator=cal,
            capital_manager=cm_inst,
        )
        # base_size=100, k=1.0 → final_size=100 > 50
        tid = execute_trade_decision(
            self._dec(), cal, om, led, capital_manager=cm_inst,
        )
        assert tid is None
        assert cm_inst.get_available_capital() == 50.0  # Nenhuma reserva feita

    def test_pipeline_with_risk_and_capital(self, ws_tmp):
        """Pipeline integra risk_manager + capital_manager."""
        cm_inst = CapitalManager(500.0)
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(
            state_path=str(ws_tmp / "l.json"),
            calibrator=cal,
            capital_manager=cm_inst,
        )
        rm = RiskManager({"max_exposure_total": 10000})

        # Trade 1: OK
        t1 = execute_trade_decision(
            self._dec(), cal, om, led,
            risk_manager=rm, capital_manager=cm_inst,
        )
        assert t1 is not None

        # Trade 2: OK
        t2 = execute_trade_decision(
            self._dec(), cal, om, led,
            risk_manager=rm, capital_manager=cm_inst,
        )
        assert t2 is not None

        # Trade 3: OK
        t3 = execute_trade_decision(
            self._dec(), cal, om, led,
            risk_manager=rm, capital_manager=cm_inst,
        )
        assert t3 is not None

        # Trade 4: OK
        t4 = execute_trade_decision(
            self._dec(), cal, om, led,
            risk_manager=rm, capital_manager=cm_inst,
        )
        assert t4 is not None

        # Trade 5: OK
        t5 = execute_trade_decision(
            self._dec(), cal, om, led,
            risk_manager=rm, capital_manager=cm_inst,
        )
        assert t5 is not None
        assert cm_inst.get_available_capital() == 0.0

        # Trade 6: bloqueado (sem capital)
        t6 = execute_trade_decision(
            self._dec(), cal, om, led,
            risk_manager=rm, capital_manager=cm_inst,
        )
        assert t6 is None

    def test_pipeline_full_lifecycle(self, ws_tmp):
        """Pipeline completo: open → resolve → capital correto."""
        cm_inst = CapitalManager(1000.0)
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "c.json"))
        om = OrderManager()
        led = TradeLedger(
            state_path=str(ws_tmp / "l.json"),
            calibrator=cal,
            capital_manager=cm_inst,
        )
        tid = execute_trade_decision(
            self._dec(), cal, om, led, capital_manager=cm_inst,
        )
        assert tid is not None

        # Simular fill completo e resolver
        led.update_fill_status(tid, 100.0, 0.65)
        resolve_trade(tid, 1, led)

        snap = cm_inst.snapshot()
        assert snap["realized_pnl"] == 0.0
        assert snap["available_capital"] == 1000.0


# =============================================================================
# INTEGRAÇÃO COM RECONCILER
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


class TestReconcilerCapitalIntegration:
    def test_reconcile_full_fill_moves_capital(self, ws_tmp):
        """Reconciler move capital para unresolved quando fully filled."""
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        om = OrderManager()
        reconciler = OrderReconciler()
        api = MockExchangeAPI()

        tid = str(uuid.uuid4())
        oid = "order-1"
        led.record_entry(tid, "m1", "C", "D", 0.5, 100, 1, 100, 0.5, order_ids=[oid])
        om.sync_order_status(oid, "PENDING", 0)

        # Simular fill completo na API
        api.orders[oid] = {"order_id": oid, "status": "FILLED", "filled_size": 100}
        api.fills[oid] = [{"fill_id": "f1", "price": 0.50, "size": 100, "timestamp": "2025-01-01T00:00:00+00:00"}]

        actions = reconciler.reconcile_open_trades(om, led, api, capital_manager=cm_inst)
        assert actions[tid] == "FULLY_FILLED"

        snap = cm_inst.snapshot()
        assert snap["locked_in_open_orders"] == 0.0
        assert snap["locked_in_unresolved_markets"] == 100.0

    def test_reconcile_cancelled_releases_capital(self, ws_tmp):
        """Reconciler libera capital quando trade cancelado."""
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        om = OrderManager()
        reconciler = OrderReconciler()
        api = MockExchangeAPI()

        tid = str(uuid.uuid4())
        oid = "order-2"
        led.record_entry(tid, "m1", "C", "D", 0.5, 200, 1, 200, 0.5, order_ids=[oid])
        om.sync_order_status(oid, "PENDING", 0)
        assert cm_inst.get_available_capital() == 800.0

        # API: ordem cancelada, sem fills
        api.orders[oid] = {"order_id": oid, "status": "CANCELLED", "filled_size": 0}
        api.fills[oid] = []

        actions = reconciler.reconcile_open_trades(om, led, api, capital_manager=cm_inst)
        assert actions[tid] == "CANCELLED"
        assert cm_inst.get_available_capital() == 1000.0

    def test_reconcile_partial_keeps_reserve(self, ws_tmp):
        """Partial fill mantém reserva em open_orders."""
        cm_inst = CapitalManager(1000.0)
        led = TradeLedger(
            state_path=str(ws_tmp / "ledger.json"),
            capital_manager=cm_inst,
        )
        om = OrderManager()
        reconciler = OrderReconciler()
        api = MockExchangeAPI()

        tid = str(uuid.uuid4())
        oid = "order-3"
        led.record_entry(tid, "m1", "C", "D", 0.5, 100, 1, 100, 0.5, order_ids=[oid])
        om.sync_order_status(oid, "PENDING", 0)

        # Partial fill
        api.orders[oid] = {"order_id": oid, "status": "PARTIALLY_FILLED", "filled_size": 50}
        api.fills[oid] = [{"fill_id": "f1", "price": 0.50, "size": 50, "timestamp": "2025-01-01T00:00:00+00:00"}]

        actions = reconciler.reconcile_open_trades(om, led, api, capital_manager=cm_inst)
        assert actions[tid] == "IN_PROGRESS"

        snap = cm_inst.snapshot()
        assert snap["locked_in_open_orders"] == 100.0  # Ainda reservado
        assert snap["locked_in_unresolved_markets"] == 0.0


# =============================================================================
# CONSTRUTOR E EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_negative_initial_capital_raises(self):
        with pytest.raises(ValueError):
            CapitalManager(initial_capital=-100)

    def test_zero_initial_capital(self):
        cm_inst = CapitalManager(initial_capital=0.0)
        assert cm_inst.get_available_capital() == 0.0
        with pytest.raises(InsufficientCapitalError):
            cm_inst.reserve_for_order("t1", 1.0)

    def test_move_to_unresolved_without_prior_reserve(self):
        """move_to_unresolved funciona mesmo sem reserva prévia."""
        cm_inst = CapitalManager(1000.0)
        cm_inst.move_to_unresolved("t1", 200.0)
        snap = cm_inst.snapshot()
        assert snap["locked_in_open_orders"] == 0.0
        assert snap["locked_in_unresolved_markets"] == 200.0
        assert snap["available_capital"] == 800.0

    def test_resolve_without_unresolved(self):
        """resolve_trade funciona mesmo sem registro em unresolved."""
        cm_inst = CapitalManager(1000.0)
        cm_inst.resolve_trade("t1", -100.0)
        snap = cm_inst.snapshot()
        assert snap["realized_pnl"] == -100.0
        assert snap["total_capital"] == 900.0

    def test_snapshot_is_copy(self, cm):
        """snapshot() retorna cópia, não referência interna."""
        snap1 = cm.snapshot()
        snap1["total_capital"] = 999999
        snap2 = cm.snapshot()
        assert snap2["total_capital"] == 1000.0

    def test_deadzone_resolution_buffer(self):
        cm_inst = CapitalManager(1000.0, deadzone_resolution_buffer=0.1)
        assert cm_inst.snapshot()["deadzone_resolution_buffer"] == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
