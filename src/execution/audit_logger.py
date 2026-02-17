"""
Audit Logger — Registro CSV de trades simulados
================================================

Logs detalhados para auditoria de simulação realista.
Salva em /data/simulation_trades.csv (ou path configurável).
"""

import csv
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = "/data/simulation_trades.csv"
FALLBACK_LOG_PATH = ".data/simulation_trades.csv"

HEADERS = [
    "timestamp",
    "market_id",
    "side",
    "order_size",
    "executed_size",
    "execution_price",
    "mid_price_at_signal",
    "spread_at_signal",
    "liquidity_available",
    "slippage_applied",
    "latency_applied_ms",
    "pnl_after_trade",
    "capital_after_trade",
    "partial_fill",
]


class SimulationAuditLogger:
    """
    Logger thread-safe para trades de simulação.
    Cria diretório se necessário. Append mode.
    """

    def __init__(self, log_path: Optional[str] = None) -> None:
        self._path = Path(log_path or DEFAULT_LOG_PATH)
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_file(self) -> None:
        with self._lock:
            if self._initialized:
                return
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._path = Path(FALLBACK_LOG_PATH)
                self._path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self._path.exists()
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=HEADERS)
                if not file_exists:
                    w.writeheader()
            self._initialized = True

    def log_trade(self, record: Dict[str, Any]) -> None:
        """Registra um trade no CSV."""
        self._ensure_file()
        row = {h: record.get(h, "") for h in HEADERS}
        row["timestamp"] = record.get("timestamp") or datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with open(self._path, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=HEADERS).writerow(row)
            except Exception as e:
                logger.error("[AUDIT] [WRITE_FAIL] %s", str(e))

    def get_path(self) -> Path:
        return self._path
