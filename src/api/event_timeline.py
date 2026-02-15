"""
Event Timeline — Lista em Memória de Eventos Recentes
=======================================================

Mantém últimos N eventos importantes para exibição no dashboard.
Tipos: reconcile_ok, safe_mode_on, safe_mode_off, reason_added,
reason_removed, config_changed, health_check_fail, health_check_recovered.

Apenas leitura/escrita em memória. Não modifica core.
"""

import time
from collections import deque
from typing import Any, Dict, List

EVENTS_MAX = 20


class EventTimeline:
    """Armazena eventos recentes com timestamp e texto humano."""

    def __init__(self, maxlen: int = EVENTS_MAX) -> None:
        self._events: deque = deque(maxlen=maxlen)

    def add(
        self,
        event_type: str,
        text: str,
        timestamp: float = None,
    ) -> None:
        """Adiciona evento à timeline."""
        ts = timestamp or time.time()
        human_time = time.strftime("%H:%M:%S", time.localtime(ts))
        self._events.append({
            "ts": ts,
            "timestamp": human_time,
            "type": event_type,
            "text": text,
            "message": text,  # compatibilidade
        })

    def get_recent(self, n: int = EVENTS_MAX) -> List[Dict[str, Any]]:
        """Retorna últimos n eventos, mais recente primeiro."""
        items = list(self._events)
        items.reverse()
        return items[:n]

    def clear(self) -> None:
        """Limpa todos os eventos (útil para testes)."""
        self._events.clear()
