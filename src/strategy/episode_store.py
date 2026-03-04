"""
EpisodeStore — Append-only JSONL persistence for edge episodes
================================================================

Writes one JSON line per closed episode to resolved_state_dir.
All I/O is non-fatal (try/except, logger.debug).
Path is validated to be under /tmp or /private/tmp.
"""

import json
import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "/tmp/polymarket_state"
_FILENAME = "edge_episodes.jsonl"


def _path_safe_under_tmp(path: str) -> bool:
    try:
        real = os.path.realpath(path)
        return real.startswith("/tmp") or real.startswith("/private/tmp")
    except Exception:
        return False


class EpisodeStore:
    """Append-only JSONL writer for closed edge episodes. Thread-safe, non-fatal."""

    def __init__(self, state_dir: Optional[str] = None) -> None:
        base = state_dir or _DEFAULT_DIR
        self._path = os.path.join(base, _FILENAME)
        self._lock = threading.Lock()
        self._safe = _path_safe_under_tmp(self._path)
        if not self._safe:
            logger.warning("[EPISODE_STORE] path outside /tmp, writes disabled: %s", self._path)
        else:
            try:
                os.makedirs(base, exist_ok=True)
            except OSError:
                pass

    @property
    def path(self) -> str:
        return self._path

    def append(self, record: Dict[str, Any]) -> None:
        if not self._safe:
            return
        with self._lock:
            try:
                with open(self._path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except Exception as e:
                logger.debug("[EPISODE_STORE] write skip (non-fatal): %s", e)
