"""
Métricas globais de ingestão e decisão
======================================

Thread-safe. Atualizado em tempo real.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Ingestão
INGESTION_METRICS: Dict[str, Any] = {
    "markets_subscribed": 0,
    "snapshots_received": 0,
    "engine_events": 0,
    "events_discarded": 0,
    "last_event_timestamp": None,
    "last_snapshot_ts": None,
    "last_engine_ts": None,
    "ws_disconnects": 0,
}
_INGESTION_LOCK = threading.Lock()

# Decisão (motivos de descarte)
DECISION_REASONS = (
    "blocked_by_risk",
    "blocked_by_liquidity",
    "blocked_by_spread",
    "blocked_by_latency",
    "blocked_by_confidence",
    "no_signal",
    "rejected_by_execution",
)
DECISION_COUNTERS: Dict[str, int] = {r: 0 for r in DECISION_REASONS}
_DECISION_LOCK = threading.Lock()

# Sanidade
_LAST_SNAPSHOT_TS: Optional[float] = None
_LAST_ENGINE_TS: Optional[float] = None
_SANITY_ALERTED_INGESTION = False
_SANITY_ALERTED_ENGINE = False


def set_markets_subscribed(n: int) -> None:
    with _INGESTION_LOCK:
        INGESTION_METRICS["markets_subscribed"] = n


def inc_snapshots_received() -> None:
    global _LAST_SNAPSHOT_TS
    with _INGESTION_LOCK:
        now = time.time()
        INGESTION_METRICS["snapshots_received"] = INGESTION_METRICS.get("snapshots_received", 0) + 1
        INGESTION_METRICS["last_event_timestamp"] = now
        INGESTION_METRICS["last_snapshot_ts"] = now
        _LAST_SNAPSHOT_TS = now


def inc_engine_events() -> None:
    global _LAST_ENGINE_TS
    with _INGESTION_LOCK:
        now = time.time()
        INGESTION_METRICS["engine_events"] = INGESTION_METRICS.get("engine_events", 0) + 1
        INGESTION_METRICS["last_event_timestamp"] = now
        INGESTION_METRICS["last_engine_ts"] = now
        _LAST_ENGINE_TS = now


def inc_events_discarded() -> None:
    with _INGESTION_LOCK:
        INGESTION_METRICS["events_discarded"] = INGESTION_METRICS.get("events_discarded", 0) + 1


def inc_ws_disconnects() -> None:
    with _INGESTION_LOCK:
        INGESTION_METRICS["ws_disconnects"] = INGESTION_METRICS.get("ws_disconnects", 0) + 1


def record_decision_reason(reason: str) -> None:
    if reason not in DECISION_REASONS:
        return
    with _DECISION_LOCK:
        DECISION_COUNTERS[reason] = DECISION_COUNTERS.get(reason, 0) + 1


def get_ingestion_metrics() -> Dict[str, Any]:
    with _INGESTION_LOCK:
        return dict(INGESTION_METRICS)


def get_decision_metrics() -> Dict[str, int]:
    with _DECISION_LOCK:
        return dict(DECISION_COUNTERS)


def run_sanity_checks(system_start_time: Optional[float] = None) -> None:
    """Verifica sanidade: snapshots e engine. Loga alertas se inativo."""
    global _SANITY_ALERTED_INGESTION, _SANITY_ALERTED_ENGINE
    now = time.time()
    with _INGESTION_LOCK:
        snapshots = INGESTION_METRICS.get("snapshots_received", 0)
        engine_ev = INGESTION_METRICS.get("engine_events", 0)

    if snapshots == 0:
        if _LAST_SNAPSHOT_TS is not None:
            age = now - _LAST_SNAPSHOT_TS
        elif system_start_time is not None:
            age = now - system_start_time
        else:
            age = 0
    else:
        age = now - (_LAST_SNAPSHOT_TS or now) if _LAST_SNAPSHOT_TS else 0

    if snapshots == 0 and age > 120 and not _SANITY_ALERTED_INGESTION:
        logger.critical("[INGESTION_FAIL] No market data received")
        _SANITY_ALERTED_INGESTION = True
    elif snapshots > 0:
        _SANITY_ALERTED_INGESTION = False

    engine_age = now - (_LAST_ENGINE_TS or now) if _LAST_ENGINE_TS else 999
    if snapshots > 0 and engine_ev == 0 and engine_age > 300 and not _SANITY_ALERTED_ENGINE:
        logger.warning("[ENGINE_IDLE] Engine not producing events")
        _SANITY_ALERTED_ENGINE = True
    elif engine_ev > 0:
        _SANITY_ALERTED_ENGINE = False
