"""
SafetyAckStore — Persistência de Acknowledgements de Segurança
================================================================

Armazena prova auditável de que um operador reconheceu e limpou
uma razão de bloqueio no SafetyState.

PERSISTÊNCIA:
- JSON atômico: tmp → flush → fsync → os.replace → fsync dir
- File lock com fcntl (quando disponível)
- Schema versionado com created_at/updated_at UTC

Cada ACK contém:
- reason: a razão que foi reconhecida
- acked_at: timestamp ISO-8601 UTC
- operator_note: nota do operador
- preconditions: dict com o estado no momento do ack
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SAFETY_ACK_SCHEMA_VERSION: int = 1

DEFAULT_ACK_PATH: str = os.path.expanduser(
    "~/.polymarket_bot/safety_ack.json"
)

# File lock (POSIX)
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


class SafetyAckStore:
    """
    Armazena ACKs de segurança com persistência atômica.

    Uso:
        store = SafetyAckStore()
        store.record_ack("THREAD_DEAD", "Threads reiniciadas", {"threads_alive": True})
        last = store.get_last_ack("THREAD_DEAD")
    """

    def __init__(self, state_path: Optional[str] = None) -> None:
        self._state_path = Path(state_path or DEFAULT_ACK_PATH)
        self._lock = threading.RLock()
        self._acks: List[Dict[str, Any]] = []
        self._created_at: Optional[str] = None
        self._ensure_dir()
        self._load()

    def _ensure_dir(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    @staticmethod
    def _fsync_directory(dir_path: str) -> None:
        try:
            dir_fd = os.open(dir_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    def _load(self) -> None:
        if not self._state_path.exists():
            self._acks = []
            self._created_at = None
            return
        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)
            self._acks = data.get("acks", [])
            self._created_at = data.get("created_at")
        except Exception as e:
            logger.error(f"[SAFETY_ACK:LOAD_ERROR] {e}")
            self._acks = []

    def _save(self) -> None:
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            if self._created_at is None:
                self._created_at = now
            payload = {
                "schema_version": SAFETY_ACK_SCHEMA_VERSION,
                "created_at": self._created_at,
                "updated_at": now,
                "acks": list(self._acks),
            }

        json_str = json.dumps(payload, indent=2)
        state_dir = str(self._state_path.parent)
        temp_path = None

        lock_fd = None
        try:
            # File lock
            if _HAS_FCNTL:
                lock_path = str(self._state_path) + ".lock"
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Atomic write
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="ack_",
                dir=state_dir, delete=False,
            ) as f:
                temp_path = f.name
                f.write(json_str)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, str(self._state_path))
            self._fsync_directory(state_dir)
            temp_path = None
        except Exception as e:
            logger.error(f"[SAFETY_ACK:SAVE_ERROR] {e}")
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def record_ack(
        self,
        reason: str,
        operator_note: str,
        preconditions: Dict[str, Any],
    ) -> None:
        """Registra ACK de uma razão de segurança."""
        entry = {
            "reason": reason,
            "acked_at": datetime.now(timezone.utc).isoformat(),
            "operator_note": operator_note,
            "preconditions": preconditions,
        }
        with self._lock:
            self._acks.append(entry)
        logger.info(
            f"[SAFETY_ACK:RECORDED] reason={reason} note={operator_note!r}"
        )
        self._save()

    def get_last_ack(self, reason: str) -> Optional[Dict[str, Any]]:
        """Retorna o último ACK para uma razão, ou None."""
        with self._lock:
            for ack in reversed(self._acks):
                if ack.get("reason") == reason:
                    return dict(ack)
        return None

    def get_all_acks(self) -> List[Dict[str, Any]]:
        """Retorna cópia de todos os ACKs."""
        with self._lock:
            return [dict(a) for a in self._acks]
