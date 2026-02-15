"""
Testes de integração do sistema completo.

Verifica que MarketDataEngine, PipelineHandler e ReconcilerLoop
funcionam corretamente em conjunto sem time.sleep para sincronização.

Usa:
- Mocks com contadores/flags para verificação
- Loop com timeout curto para aguardar processamento
- Sem dependência de timing externo
"""

import queue
import threading
import time

import pytest

from src.watcher.bridge import MarketSnapshot, MarketTrade
from src.engine.market_data import MarketDataEngine, MAX_CONSECUTIVE_ERRORS


# ========================================================================
# Mocks
# ========================================================================

class MockPipeline:
    """Mock do pipeline com contador thread-safe."""

    def __init__(self) -> None:
        self.snapshot_count: int = 0
        self.last_snapshot: object = None
        self._lock = threading.Lock()

    def process_market_update(self, event: MarketSnapshot) -> None:
        with self._lock:
            self.snapshot_count += 1
            self.last_snapshot = event


class MockReconciler:
    """Mock do reconciler com contador thread-safe."""

    def __init__(self) -> None:
        self.trade_count: int = 0
        self.last_trade: object = None
        self._lock = threading.Lock()

    def notify_trade_event(self, event: MarketTrade) -> None:
        with self._lock:
            self.trade_count += 1
            self.last_trade = event


class FailingPipeline:
    """Pipeline que sempre levanta exceção."""

    def process_market_update(self, event: object) -> None:
        raise RuntimeError("Falha proposital no pipeline")


class FailingReconciler:
    """Reconciler que sempre levanta exceção."""

    def notify_trade_event(self, event: object) -> None:
        raise RuntimeError("Falha proposital no reconciler")


# ========================================================================
# Helpers
# ========================================================================

def _wait_for(condition, timeout: float = 5.0, interval: float = 0.01) -> bool:
    """
    Aguarda condição ser verdadeira (sem time.sleep fixo).

    Args:
        condition: callable que retorna bool
        timeout: tempo máximo de espera em segundos
        interval: intervalo entre verificações

    Returns:
        True se condição atingida, False se timeout
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


def _make_snapshot(token_id: str = "token-1", bid: float = 0.65) -> MarketSnapshot:
    """Cria MarketSnapshot de teste."""
    return MarketSnapshot(
        timestamp=time.time(),
        token_id=token_id,
        best_bid_price=bid,
        best_bid_size=100.0,
        best_ask_price=0.70,
        best_ask_size=50.0,
    )


def _make_trade(
    token_id: str = "token-1",
    price: float = 0.65,
    side: str = "BUY",
) -> MarketTrade:
    """Cria MarketTrade de teste."""
    return MarketTrade(
        timestamp=time.time(),
        token_id=token_id,
        price=price,
        size=10.0,
        side=side,
    )


# ========================================================================
# Testes — Snapshot processado
# ========================================================================

class TestSnapshotProcessed:
    def test_single_snapshot(self):
        """MarketSnapshot é despachado para pipeline."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        pipeline = MockPipeline()
        reconciler = MockReconciler()

        engine = MarketDataEngine(q, pipeline, reconciler)
        engine.start()

        try:
            snap = _make_snapshot()
            q.put(snap)

            assert _wait_for(lambda: pipeline.snapshot_count == 1)
            assert pipeline.last_snapshot is snap
            assert reconciler.trade_count == 0
        finally:
            engine.stop()
            engine.join(timeout=5)

    def test_multiple_snapshots(self):
        """Múltiplos snapshots processados em sequência."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        pipeline = MockPipeline()
        reconciler = MockReconciler()

        engine = MarketDataEngine(q, pipeline, reconciler)
        engine.start()

        try:
            for i in range(10):
                q.put(_make_snapshot(bid=0.50 + i * 0.01))

            assert _wait_for(lambda: pipeline.snapshot_count == 10)
            assert reconciler.trade_count == 0
        finally:
            engine.stop()
            engine.join(timeout=5)


# ========================================================================
# Testes — Trade notifica reconciler
# ========================================================================

class TestTradeNotifiesReconciler:
    def test_single_trade(self):
        """MarketTrade é despachado para reconciler."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        pipeline = MockPipeline()
        reconciler = MockReconciler()

        engine = MarketDataEngine(q, pipeline, reconciler)
        engine.start()

        try:
            trade = _make_trade()
            q.put(trade)

            assert _wait_for(lambda: reconciler.trade_count == 1)
            assert reconciler.last_trade is trade
            assert pipeline.snapshot_count == 0
        finally:
            engine.stop()
            engine.join(timeout=5)

    def test_mixed_events(self):
        """Snapshots e trades são despachados para destinos corretos."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        pipeline = MockPipeline()
        reconciler = MockReconciler()

        engine = MarketDataEngine(q, pipeline, reconciler)
        engine.start()

        try:
            q.put(_make_snapshot())
            q.put(_make_trade())
            q.put(_make_snapshot(bid=0.66))
            q.put(_make_trade(side="SELL"))
            q.put(_make_snapshot(bid=0.67))

            assert _wait_for(lambda: pipeline.snapshot_count == 3)
            assert _wait_for(lambda: reconciler.trade_count == 2)
        finally:
            engine.stop()
            engine.join(timeout=5)


# ========================================================================
# Testes — Engine para corretamente
# ========================================================================

class TestEngineStop:
    def test_stop_and_join(self):
        """Engine encerra corretamente via stop() + join()."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        pipeline = MockPipeline()
        reconciler = MockReconciler()

        engine = MarketDataEngine(q, pipeline, reconciler)
        engine.start()

        assert engine.is_alive()

        engine.stop()
        engine.join(timeout=5)

        assert not engine.is_alive()

    def test_stop_flushes_pending(self):
        """Eventos já na fila são processados antes de parar."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        pipeline = MockPipeline()
        reconciler = MockReconciler()

        engine = MarketDataEngine(q, pipeline, reconciler)

        # Coloca eventos ANTES de iniciar
        for i in range(5):
            q.put(_make_snapshot(bid=0.50 + i * 0.01))

        engine.start()

        # Aguarda processamento
        assert _wait_for(lambda: pipeline.snapshot_count == 5)

        engine.stop()
        engine.join(timeout=5)
        assert not engine.is_alive()
        assert pipeline.snapshot_count == 5

    def test_not_daemon(self):
        """MarketDataEngine NÃO é daemon thread."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        engine = MarketDataEngine(q, MockPipeline(), MockReconciler())
        assert engine.daemon is False

    def test_events_processed_counter(self):
        """Contador de eventos processados é correto."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        pipeline = MockPipeline()
        reconciler = MockReconciler()

        engine = MarketDataEngine(q, pipeline, reconciler)
        engine.start()

        try:
            q.put(_make_snapshot())
            q.put(_make_trade())
            q.put(_make_snapshot(bid=0.66))

            assert _wait_for(lambda: engine.events_processed == 3)
        finally:
            engine.stop()
            engine.join(timeout=5)


# ========================================================================
# Testes — Resiliência a erros
# ========================================================================

class TestEngineResilience:
    def test_pipeline_error_does_not_kill_engine(self):
        """Erro no pipeline não mata a thread."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        pipeline = FailingPipeline()
        reconciler = MockReconciler()

        engine = MarketDataEngine(q, pipeline, reconciler)
        engine.start()

        try:
            # Snapshot causa erro no pipeline
            q.put(_make_snapshot())

            # Mas trade ainda funciona
            q.put(_make_trade())

            assert _wait_for(lambda: reconciler.trade_count == 1)
            assert engine.is_alive()
            assert engine.consecutive_errors == 0  # Reset após trade OK
        finally:
            engine.stop()
            engine.join(timeout=5)

    def test_consecutive_errors_tracked(self):
        """Erros consecutivos são contabilizados."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        pipeline = FailingPipeline()
        reconciler = FailingReconciler()

        engine = MarketDataEngine(q, pipeline, reconciler)
        engine.start()

        try:
            for _ in range(5):
                q.put(_make_snapshot())

            assert _wait_for(lambda: engine.consecutive_errors >= 5)
            assert engine.is_alive()  # Não morre
        finally:
            engine.stop()
            engine.join(timeout=5)

    def test_unknown_event_type_ignored(self):
        """Evento desconhecido não causa crash."""
        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        pipeline = MockPipeline()
        reconciler = MockReconciler()

        engine = MarketDataEngine(q, pipeline, reconciler)
        engine.start()

        try:
            q.put("unknown_event")
            q.put(_make_snapshot())

            assert _wait_for(lambda: pipeline.snapshot_count == 1)
            # O evento desconhecido foi ignorado, mas conta como processado
            assert _wait_for(lambda: engine.events_processed == 2)
        finally:
            engine.stop()
            engine.join(timeout=5)


# ========================================================================
# Testes — PipelineHandler e ReconcilerLoop do orquestrador
# ========================================================================

class TestOrchestratorComponents:
    def test_pipeline_handler_stores_snapshot(self):
        """PipelineHandler armazena último snapshot por token."""
        from src.orchestrator import PipelineHandler

        handler = PipelineHandler()
        snap = _make_snapshot("token-A", bid=0.55)
        handler.process_market_update(snap)

        assert handler.update_count == 1
        latest = handler.get_latest("token-A")
        assert latest is snap
        assert handler.get_latest("nonexistent") is None

    def test_pipeline_handler_multiple_tokens(self):
        """PipelineHandler isola snapshots por token."""
        from src.orchestrator import PipelineHandler

        handler = PipelineHandler()
        handler.process_market_update(_make_snapshot("A", 0.50))
        handler.process_market_update(_make_snapshot("B", 0.60))
        handler.process_market_update(_make_snapshot("A", 0.55))

        assert handler.update_count == 3
        a = handler.get_latest("A")
        b = handler.get_latest("B")
        assert a is not None and a.best_bid_price == 0.55
        assert b is not None and b.best_bid_price == 0.60

    def test_reconciler_loop_receives_trade_events(self, tmp_path):
        """ReconcilerLoop armazena trade events."""
        from src.orchestrator import ReconcilerLoop, ExchangeAPIStub
        from src.execution.reconciler import OrderReconciler
        from src.execution.order_manager import OrderManager
        from src.engine.trade_ledger import TradeLedger

        rec = ReconcilerLoop(
            reconciler=OrderReconciler(),
            order_manager=OrderManager(),
            trade_ledger=TradeLedger(
                state_path=str(tmp_path / "ledger.json"),
            ),
            exchange_api=ExchangeAPIStub(),
            interval=1,
        )

        trade = _make_trade()
        rec.notify_trade_event(trade)

        assert len(rec._trade_events) == 1
        assert rec._trade_events[0] is trade

    def test_reconciler_loop_stop(self, tmp_path):
        """ReconcilerLoop encerra via stop() + join()."""
        from src.orchestrator import ReconcilerLoop, ExchangeAPIStub
        from src.execution.reconciler import OrderReconciler
        from src.execution.order_manager import OrderManager
        from src.engine.trade_ledger import TradeLedger

        rec = ReconcilerLoop(
            reconciler=OrderReconciler(),
            order_manager=OrderManager(),
            trade_ledger=TradeLedger(
                state_path=str(tmp_path / "ledger.json"),
            ),
            exchange_api=ExchangeAPIStub(),
            interval=1,
        )
        rec.start()

        assert rec.is_alive()
        assert rec.daemon is False

        rec.stop()
        rec.join(timeout=5)
        assert not rec.is_alive()


# ========================================================================
# Testes — Integração completa
# ========================================================================

class TestFullIntegration:
    def test_end_to_end_snapshot_to_pipeline(self):
        """
        Integração: snapshot na fila → MarketDataEngine → PipelineHandler.
        """
        from src.orchestrator import PipelineHandler

        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        handler = PipelineHandler()
        reconciler = MockReconciler()

        engine = MarketDataEngine(q, handler, reconciler)
        engine.start()

        try:
            snap = _make_snapshot("t1", 0.65)
            q.put(snap)

            assert _wait_for(lambda: handler.update_count == 1)
            assert handler.get_latest("t1") is snap
        finally:
            engine.stop()
            engine.join(timeout=5)

    def test_end_to_end_trade_to_reconciler(self, tmp_path):
        """
        Integração: trade na fila → MarketDataEngine → ReconcilerLoop.
        """
        from src.orchestrator import ReconcilerLoop, ExchangeAPIStub
        from src.execution.reconciler import OrderReconciler
        from src.execution.order_manager import OrderManager
        from src.engine.trade_ledger import TradeLedger

        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        pipeline = MockPipeline()

        rec_loop = ReconcilerLoop(
            reconciler=OrderReconciler(),
            order_manager=OrderManager(),
            trade_ledger=TradeLedger(
                state_path=str(tmp_path / "ledger.json"),
            ),
            exchange_api=ExchangeAPIStub(),
            interval=60,  # Não precisa rodar reconcile neste teste
        )

        engine = MarketDataEngine(q, pipeline, rec_loop)
        engine.start()

        try:
            trade = _make_trade("t1", 0.65, "BUY")
            q.put(trade)

            assert _wait_for(lambda: len(rec_loop._trade_events) == 1)
            assert rec_loop._trade_events[0] is trade
        finally:
            engine.stop()
            engine.join(timeout=5)

    def test_end_to_end_mixed_flow(self, tmp_path):
        """
        Integração completa: snapshots + trades misturados.
        """
        from src.orchestrator import PipelineHandler, ReconcilerLoop, ExchangeAPIStub
        from src.execution.reconciler import OrderReconciler
        from src.execution.order_manager import OrderManager
        from src.engine.trade_ledger import TradeLedger

        q: queue.Queue = queue.Queue(maxsize=100)  # type: ignore[type-arg]
        handler = PipelineHandler()

        rec_loop = ReconcilerLoop(
            reconciler=OrderReconciler(),
            order_manager=OrderManager(),
            trade_ledger=TradeLedger(
                state_path=str(tmp_path / "ledger.json"),
            ),
            exchange_api=ExchangeAPIStub(),
            interval=60,
        )

        engine = MarketDataEngine(q, handler, rec_loop)
        engine.start()

        try:
            # Interleave snapshots and trades
            q.put(_make_snapshot("t1", 0.60))
            q.put(_make_trade("t1", 0.60, "BUY"))
            q.put(_make_snapshot("t2", 0.40))
            q.put(_make_trade("t2", 0.40, "SELL"))
            q.put(_make_snapshot("t1", 0.62))

            assert _wait_for(lambda: handler.update_count == 3)
            assert _wait_for(lambda: len(rec_loop._trade_events) == 2)

            # Verify correct routing
            assert handler.get_latest("t1") is not None
            assert handler.get_latest("t1").best_bid_price == 0.62
            assert handler.get_latest("t2") is not None
            assert rec_loop._trade_events[0].side == "BUY"
            assert rec_loop._trade_events[1].side == "SELL"
        finally:
            engine.stop()
            engine.join(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
