"""
Drawdown Tracker — Pico de Capital Persistente
================================================

Persiste max_capital_seen em disco (JSON atômico).
File lock (fcntl) + merge read-modify-write.
Hash apenas sobre campos funcionais (sem timestamps).
"""

import fcntl
import hashlib
import json
import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_STATE_DIR = ".dashboard_state"


def _hash_material(schema_version: int, max_capital_seen: float) -> Dict[str, Any]:
    """Apenas campos funcionais (sem timestamps) para content_hash."""
    return {"schema_version": schema_version, "max_capital_seen": max_capital_seen}


def _compute_content_hash(schema_version: int, max_capital_seen: float) -> str:
    """Hash apenas dos campos materiais."""
    material = _hash_material(schema_version, max_capital_seen)
    canonical = json.dumps(material, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _is_finite_positive(x: float) -> bool:
    """True se x é finito, não-NaN e > 0."""
    return (
        math.isfinite(x)
        and not math.isnan(x)
        and x > 0
    )


class DrawdownTracker:
    """
    Persiste pico de capital para cálculo de drawdown.
    Lock + merge read-modify-write. Hash sem timestamps.
    Reset seguro em corrupção.
    """

    def __init__(
        self,
        state_dir: Optional[str] = None,
        filename: str = "dashboard_state.json",
        on_reset_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        state_dir = state_dir or os.environ.get(
            "DASHBOARD_STATE_DIR", DEFAULT_STATE_DIR
        )
        self._path = Path(state_dir) / filename
        self._lock = __import__("threading").RLock()
        self._max_capital_seen: float = 0.0
        self._created_at: Optional[str] = None
        self._on_reset = on_reset_callback
        self._load()

    def _ensure_dir(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _content_hash(self, schema_version: int, max_capital_seen: float) -> str:
        """Hash apenas dos campos funcionais (não inclui created_at/updated_at)."""
        return _compute_content_hash(schema_version, max_capital_seen)

    def _load(self) -> None:
        if not self._path.exists():
            return
        fd = None
        try:
            fd = os.open(str(self._path), os.O_RDONLY)
            fcntl.flock(fd, fcntl.LOCK_SH)
            with os.fdopen(fd, "r") as f:
                data = json.load(f)
            fd = None
        except Exception as e:
            logger.warning(f"[DRAWDOWN_TRACKER] Load falhou: {e}")
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._safe_reset(0.0)
            return

        try:
            if data.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Schema version mismatch")
            stored_hash = data.get("content_hash", "")
            max_val = float(data.get("max_capital_seen", 0))
            computed = self._content_hash(SCHEMA_VERSION, max_val)
            if stored_hash and stored_hash != computed:
                raise ValueError("Hash mismatch — possível corrupção")
            if not _is_finite_positive(max_val) and max_val != 0:
                raise ValueError("max_capital_seen inválido")
            self._max_capital_seen = max(0.0, max_val) if math.isfinite(max_val) else 0.0
            self._created_at = data.get("created_at")
        except Exception as e:
            logger.warning(f"[DRAWDOWN_TRACKER] Reset por corrupção/erro: {e}")
            self._safe_reset(0.0)

    def _safe_reset(self, current_capital: float) -> None:
        """Reset seguro: max = max(current, 0)."""
        safe_cap = max(0.0, current_capital) if math.isfinite(current_capital) else 0.0
        self._max_capital_seen = safe_cap
        self._created_at = None
        if self._on_reset:
            try:
                self._on_reset()
            except Exception:
                pass

    def _lock_path(self) -> Path:
        """Arquivo de lock para serialização."""
        return self._path.with_suffix(self._path.suffix + ".lock")

    def _read_existing_max(self) -> float:
        """Lê max existente do arquivo (se existir)."""
        if not self._path.exists():
            return 0.0
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            return float(data.get("max_capital_seen", 0))
        except Exception:
            return 0.0

    def _save(self, current_capital: float) -> None:
        with self._lock:
            self._ensure_dir()
            lock_path = self._lock_path()
            lock_path.touch(exist_ok=True)
            fd = None
            try:
                fd = os.open(str(lock_path), os.O_RDWR)
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    existing_max = self._read_existing_max()
                    merged_max = max(
                        self._max_capital_seen,
                        existing_max,
                        current_capital if _is_finite_positive(current_capital) else 0.0,
                    )
                    self._max_capital_seen = merged_max
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    if fd is not None:
                        os.close(fd)
                        fd = None
            except (OSError, Exception) as e:
                logger.warning(f"[DRAWDOWN_TRACKER] Merge read: {e}")
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

            now = datetime.now(timezone.utc).isoformat()
            created = self._created_at or now
            payload = {
                "schema_version": SCHEMA_VERSION,
                "created_at": created,
                "updated_at": now,
                "max_capital_seen": self._max_capital_seen,
            }
            payload["content_hash"] = self._content_hash(
                SCHEMA_VERSION, self._max_capital_seen
            )
            self._created_at = created

        json_str = json.dumps(payload, indent=2)
        state_dir = str(self._path.parent)
        self._ensure_dir()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="dd_",
                dir=state_dir,
                delete=False,
            ) as f:
                temp_path = f.name
                f.write(json_str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, str(self._path))
            temp_path = None
            try:
                dir_fd = os.open(
                    state_dir,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        except Exception as e:
            logger.error(f"[DRAWDOWN_TRACKER] Save error: {e}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def load(self) -> None:
        """Recarrega estado do disco."""
        self._load()

    def update(
        self,
        current_capital: float,
        event_timeline: Optional[Any] = None,
    ) -> bool:
        """
        Atualiza max_capital_seen se current_capital for maior (finito, > 0).
        Ignora NaN, inf, <= 0. Retorna True se houve novo pico.
        """
        if not _is_finite_positive(current_capital):
            return False
        with self._lock:
            if current_capital > self._max_capital_seen:
                prev = self._max_capital_seen
                self._max_capital_seen = current_capital
                self._save(current_capital)
                if event_timeline and prev > 0:
                    try:
                        import time as _t
                        event_timeline.add(
                            "drawdown_new_peak",
                            f"Novo pico de capital: {current_capital:.2f}",
                            _t.time(),
                        )
                    except Exception:
                        pass
                return True
            return False

    def get_drawdown(
        self,
        current_capital: float,
    ) -> Dict[str, Any]:
        """
        Retorna max_capital, current_capital, drawdown_percent.
        """
        with self._lock:
            mx = self._max_capital_seen or current_capital
        if mx <= 0:
            return {
                "capital_max": current_capital,
                "capital_current": current_capital,
                "drawdown_percent": 0.0,
                "text": "Sem histórico.",
            }
        dd = max(0.0, (mx - current_capital) / mx * 100)
        text = f"Drawdown atual: {dd:.1f}%" if dd > 0 else "No máximo histórico."
        return {
            "capital_max": round(mx, 2),
            "capital_current": round(current_capital, 2),
            "drawdown_percent": round(dd, 1),
            "text": text,
        }
