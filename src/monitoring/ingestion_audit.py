"""
Ingestion Audit — Logs estruturados para análise posterior
===========================================================

Emitir logs JSON periódicos de auditoria da ingestão.
Sem Prometheus. Sem dependências extras.
"""

import json
import logging
import threading
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 10
GAP_THRESHOLD_SECONDS = 15
GAP_ALERT_SECONDS = 60


class IngestionAuditAggregator:
    """
    Contadores e cálculos para auditoria de ingestão.
    Janela móvel de 10 segundos.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots_total = 0
        self._snapshots_window = 0
        self._engine_events_window = 0
        self._events_discarded_window = 0
        self._ws_disconnects_total = 0
        self._gaps_total = 0
        self._last_message_ts: float | None = None
        self._last_engine_processed_ts: float | None = None
        self._window_start_ts: float = time.time()
        self._lag_samples: list[float] = []
        self._running = True
        self._prev_snapshots = 0
        self._prev_engine_events = 0
        self._prev_discarded = 0

    def tick(self) -> Dict[str, Any]:
        """
        Atualiza a partir das métricas e emite snapshot.
        Chamar a cada 10 segundos.
        """
        try:
            from src.metrics import get_ingestion_metrics
        except ImportError:
            return {}

        now = time.time()
        metrics = get_ingestion_metrics()

        with self._lock:
            snapshots_received = metrics.get("snapshots_received", 0)
            engine_events = metrics.get("engine_events", 0)
            events_discarded = metrics.get("events_discarded", 0)
            ws_disconnects = metrics.get("ws_disconnects", 0)
            last_snapshot_ts = metrics.get("last_snapshot_ts")
            last_engine_ts = metrics.get("last_engine_ts")

            self._snapshots_total = snapshots_received
            self._ws_disconnects_total = ws_disconnects
            self._last_message_ts = last_snapshot_ts
            self._last_engine_processed_ts = last_engine_ts

            elapsed = now - self._window_start_ts
            if elapsed >= WINDOW_SECONDS:
                self._snapshots_window = max(0, snapshots_received - self._prev_snapshots)
                self._engine_events_window = max(0, engine_events - self._prev_engine_events)
                self._events_discarded_window = max(0, events_discarded - self._prev_discarded)
                self._prev_snapshots = snapshots_received
                self._prev_engine_events = engine_events
                self._prev_discarded = events_discarded
                self._window_start_ts = now

            silence_seconds = 0.0
            if last_snapshot_ts is not None:
                silence_seconds = round(now - last_snapshot_ts, 1)

            engine_idle_seconds = 0.0
            if last_engine_ts is not None:
                engine_idle_seconds = round(now - last_engine_ts, 1)

            total_window = self._snapshots_window + self._engine_events_window
            discard_rate_window = 0.0
            if total_window > 0:
                discard_rate_window = round(
                    self._events_discarded_window / total_window, 4
                )

            if silence_seconds > GAP_THRESHOLD_SECONDS:
                self._gaps_total += 1
                if silence_seconds > GAP_ALERT_SECONDS:
                    logger.warning(
                        "[INGESTION_AUDIT] [GAP_ALERT] silence_seconds=%.1f",
                        silence_seconds,
                    )

            lag_p50 = 0.0
            lag_p95 = 0.0
            if self._lag_samples:
                sorted_lag = sorted(self._lag_samples)
                n = len(sorted_lag)
                lag_p50 = round(sorted_lag[int(n * 0.5)], 1)
                lag_p95 = round(sorted_lag[int(n * 0.95)] if n >= 20 else sorted_lag[-1], 1)
                self._lag_samples = self._lag_samples[-100:]

            payload = {
                "kind": "INGESTION_AUDIT",
                "snapshots_total": self._snapshots_total,
                "snapshots_window": self._snapshots_window,
                "engine_events_window": self._engine_events_window,
                "events_discarded_window": self._events_discarded_window,
                "discard_rate_window": discard_rate_window,
                "silence_seconds": silence_seconds,
                "engine_idle_seconds": engine_idle_seconds,
                "ws_disconnects_total": self._ws_disconnects_total,
                "gaps_total": self._gaps_total,
                "lag_ms_p50": lag_p50,
                "lag_ms_p95": lag_p95,
            }

        return payload

    def record_lag_ms(self, lag_ms: float) -> None:
        """Registra atraso de processamento para percentis."""
        with self._lock:
            self._lag_samples.append(lag_ms)

    def stop(self) -> None:
        self._running = False


def _run_audit_loop(aggregator: IngestionAuditAggregator) -> None:
    """Loop que emite log a cada 10 segundos."""
    while aggregator._running:
        time.sleep(WINDOW_SECONDS)
        if not aggregator._running:
            break
        payload = aggregator.tick()
        if payload:
            line = json.dumps(payload, sort_keys=True)
            logger.info("INGESTION_AUDIT %s", line)


def start_ingestion_audit(aggregator: IngestionAuditAggregator) -> threading.Thread:
    """Inicia thread de auditoria periódica."""
    t = threading.Thread(
        target=_run_audit_loop,
        args=(aggregator,),
        name="IngestionAudit",
        daemon=True,
    )
    t.start()
    return t
