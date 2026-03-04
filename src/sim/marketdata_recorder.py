"""
MarketData Recorder — Append-only JSONL snapshots for sim replay
=================================================================

Throttled to max 10 snapshots/sec/market.  All I/O non-fatal.
Path validated under /tmp.
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "/tmp/polymarket_state"
_FILENAME = "marketdata_snapshots.jsonl"
_MAX_SNAPSHOTS_PER_SEC = 10


def _path_safe_under_tmp(path: str) -> bool:
    try:
        real = os.path.realpath(path)
        return real.startswith("/tmp") or real.startswith("/private/tmp")
    except Exception:
        return False


class MarketDataRecorder:
    """Append-only JSONL writer for book snapshots.  Thread-safe, non-fatal."""

    def __init__(self, state_dir: Optional[str] = None) -> None:
        base = state_dir or _DEFAULT_DIR
        self._path = os.path.join(base, _FILENAME)
        self._lock = threading.Lock()
        self._safe = _path_safe_under_tmp(self._path)
        self._last_write: Dict[str, float] = {}
        self._min_interval = 1.0 / _MAX_SNAPSHOTS_PER_SEC
        self._drops = 0

        if not self._safe:
            logger.warning("[RECORDER] path outside /tmp, writes disabled: %s", self._path)
        else:
            try:
                os.makedirs(base, exist_ok=True)
            except OSError:
                pass

    @property
    def path(self) -> str:
        return self._path

    @property
    def drops(self) -> int:
        return self._drops

    def record(
        self,
        token_id: str,
        best_bid_px: float,
        best_bid_sz: float,
        best_ask_px: float,
        best_ask_sz: float,
        market_slug: Optional[str] = None,
    ) -> None:
        if not self._safe:
            return
        now_mono = time.monotonic()
        last = self._last_write.get(token_id, 0.0)
        if (now_mono - last) < self._min_interval:
            self._drops += 1
            return

        mid = (best_bid_px + best_ask_px) / 2.0 if best_bid_px > 0 and best_ask_px > 0 else 0.0
        spread_bps = ((best_ask_px - best_bid_px) / mid * 10000.0) if mid > 0 else 0.0

        record = {
            "ts_utc": time.time(),
            "ts_monotonic": now_mono,
            "token_id": token_id,
            "market_slug": market_slug,
            "best_bid_px": best_bid_px,
            "best_bid_sz": best_bid_sz,
            "best_ask_px": best_ask_px,
            "best_ask_sz": best_ask_sz,
            "mid_px": round(mid, 6),
            "spread_bps": round(spread_bps, 2),
        }

        with self._lock:
            self._last_write[token_id] = now_mono
            try:
                with open(self._path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except Exception as e:
                self._drops += 1
                logger.debug("[RECORDER] write skip (non-fatal): %s", e)


def load_recorded_snapshots(path: str) -> list:
    """Load JSONL snapshots file.  Returns list of dicts.  Non-fatal."""
    records = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        logger.debug("[RECORDER] load skip: %s", e)
    return records
