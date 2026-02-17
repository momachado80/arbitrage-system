"""
MarketDataEngine — Consumidor Determinístico
==============================================

Thread que consome update_queue e despacha eventos para pipeline e reconciler,
de forma síncrona e segura.

DESPACHO:
- MarketSnapshot → pipeline.process_market_update(event)
  Usa timestamp local. Não altera posições/capital.
- MarketTrade    → reconciler.notify_trade_event(event)
  Usa timestamp da exchange. NÃO atualiza posições ou capital.

RESILIÊNCIA:
- Captura exceções dentro do loop (nunca morre silenciosamente).
- Contador de erros consecutivos: se > MAX_CONSECUTIVE_ERRORS → log CRITICAL.
- Reset do contador após cada evento processado com sucesso.

ISOLAMENTO:
- Core nunca importa asyncio.
- Watcher nunca chama pipeline diretamente.
- Sem estado global.
"""

import logging
import queue
import threading
from typing import Any

from src.watcher.bridge import MarketSnapshot, MarketTrade

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS: int = 10


class MarketDataEngine(threading.Thread):
    """
    Consumer thread para eventos de mercado.

    Consome da update_queue e despacha:
    - MarketSnapshot → pipeline.process_market_update()
    - MarketTrade    → reconciler.notify_trade_event()

    Não é daemon. Deve ser encerrado via stop() + join().

    Uso:
        engine = MarketDataEngine(update_queue, pipeline, reconciler)
        engine.start()
        ...
        engine.stop()
        engine.join(timeout=5)
    """

    def __init__(
        self,
        update_queue: queue.Queue,  # type: ignore[type-arg]
        pipeline: Any,
        reconciler: Any,
    ) -> None:
        super().__init__(name="MarketDataEngine", daemon=False)
        self._queue = update_queue
        self._pipeline = pipeline
        self._reconciler = reconciler
        self.running: bool = True
        self._consecutive_errors: int = 0
        self._events_processed: int = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Loop principal: consome queue e despacha eventos."""
        logger.info("[ENGINE] [STARTED] MarketDataEngine")

        while self.running:
            try:
                event = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._dispatch(event)
                self._consecutive_errors = 0
                self._events_processed += 1
                try:
                    from src.metrics import inc_engine_events
                    inc_engine_events()
                except ImportError:
                    pass
            except Exception as e:
                self._consecutive_errors += 1
                logger.error(
                    "[ENGINE] [ERROR] %s: %s", type(e).__name__, e
                )
                if self._consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                    logger.critical(
                        "[ENGINE] [CRITICAL] consecutive_errors=%d — verifique pipeline/reconciler",
                        self._consecutive_errors,
                    )

        logger.info(
            "[ENGINE] [STOPPED] events_processed=%d", self._events_processed
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, event: object) -> None:
        """Roteamento por tipo de evento."""
        if isinstance(event, MarketSnapshot):
            self._pipeline.process_market_update(event)
        elif isinstance(event, MarketTrade):
            self._reconciler.notify_trade_event(event)
        else:
            logger.warning(
                "[ENGINE] [UNKNOWN_EVENT] type=%s", type(event).__name__
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Sinaliza encerramento do loop."""
        self.running = False

    @property
    def events_processed(self) -> int:
        """Contador de eventos processados com sucesso."""
        return self._events_processed

    @property
    def consecutive_errors(self) -> int:
        """Erros consecutivos atuais."""
        return self._consecutive_errors
