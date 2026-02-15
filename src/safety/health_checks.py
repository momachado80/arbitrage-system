"""
HealthChecker — Verificações periódicas integradas ao SafetyState
====================================================================

Checks:
- Exchange health (get_account_summary ou endpoint leve)
- Thread health (MarketDataEngine, ReconcilerLoop, Watcher)
- Clock sanity (salto monotônico absurdo)

Integrado ao loop do Orchestrator a cada health_check_interval_seconds.
"""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Detecção de clock: se monotonic diminui, anomalia
_CLOCK_ANOMALY_THRESHOLD = -1.0  # previous - current > 0 significa retrocesso


class HealthChecker:
    """
    Executa health checks e atualiza SafetyState.

    Nunca lança exceção — captura erros e block BUY via safety_state.
    """

    def __init__(
        self,
        exchange_api: Any,
        safety_state: Any,
        config: Any,
        market_engine: Optional[Any] = None,
        reconciler_loop: Optional[Any] = None,
        watcher_thread: Optional[Any] = None,
    ) -> None:
        self._exchange_api = exchange_api
        self._safety_state = safety_state
        self._config = config
        self._market_engine = market_engine
        self._reconciler_loop = reconciler_loop
        self._watcher_thread = watcher_thread
        self._last_monotonic: float = time.monotonic()

    def run_health_check(self) -> None:
        """
        Executa todos os checks. Não lança exceção.
        """
        self._check_exchange()
        self._check_threads()
        self._check_clock()

    def _check_exchange(self) -> None:
        """Exchange health: endpoint leve."""
        try:
            result = self._exchange_api.get_account_summary()
            if result is not None:
                self._safety_state.clear_reason("EXCHANGE_UNREACHABLE")
                return
            # get_account_summary retornou None — stub sem implementação
            # não bloquear nesse caso
        except Exception as e:
            logger.warning(f"[HEALTH:EXCHANGE] {e}")
            self._safety_state.block_buy("EXCHANGE_UNREACHABLE")

    def _check_threads(self) -> None:
        """Thread health: is_alive()."""
        dead = []
        if self._market_engine is not None and not self._market_engine.is_alive():
            dead.append("market_engine")
        if self._reconciler_loop is not None and not self._reconciler_loop.is_alive():
            dead.append("reconciler")
        if self._watcher_thread is not None and not self._watcher_thread.is_alive():
            dead.append("watcher")

        if dead:
            self._safety_state.block_buy("THREAD_DEAD")
            logger.warning(f"[HEALTH:THREAD_DEAD] {dead}")
        else:
            # Não limpar THREAD_DEAD automaticamente — requer ACK
            pass

    def _check_clock(self) -> None:
        """Clock sanity: monotonic não deve retroceder."""
        now = time.monotonic()
        delta = now - self._last_monotonic
        self._last_monotonic = now
        if delta < _CLOCK_ANOMALY_THRESHOLD:
            self._safety_state.enable_safe_mode("CLOCK_ANOMALY")
            logger.critical(
                f"[HEALTH:CLOCK_ANOMALY] monotonic retrocedeu: delta={delta}"
            )
