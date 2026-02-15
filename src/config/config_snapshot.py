"""
ConfigSnapshotManager — Detecção de mudança de configuração
============================================================

Persistência JSON atômica do snapshot de configuração.
Se config mudar em relação ao último snapshot → block BUY (CONFIG_CHANGED).

Não sobrescreve snapshot automaticamente quando detecta mudança.
ACK explícito (ou re-inicialização) necessário para validar nova config.
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_SNAPSHOT_SCHEMA_VERSION: int = 1
DEFAULT_SNAPSHOT_PATH: str = os.path.expanduser(
    "~/.polymarket_bot/config_snapshot.json"
)

# Campos sensíveis que devem ser redactados antes do hash
_SENSITIVE_KEYS = frozenset({
    "private_key", "api_key", "secret", "password", "token",
    "privatekey", "apikey",
})


def _redact_sensitive(obj: Any) -> Any:
    """Substitui valores sensíveis por REDACTED."""
    if isinstance(obj, dict):
        return {
            k: "REDACTED" if any(
                sk in k.lower() for sk in _SENSITIVE_KEYS
            ) else _redact_sensitive(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_sensitive(i) for i in obj]
    return obj


def _config_hash(normalized: dict) -> str:
    """SHA256 do JSON ordenado."""
    canon = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


class ConfigSnapshotManager:
    """
    Gerencia snapshot de configuração com detecção de mudança.

    Se config mudar sem ACK → block BUY via CONFIG_CHANGED.
    """

    def __init__(self, state_path: Optional[str] = None) -> None:
        self._state_path = Path(state_path or DEFAULT_SNAPSHOT_PATH)
        self._lock = threading.Lock()
        self._last_hash: Optional[str] = None
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

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

    def _load_existing(self) -> Optional[str]:
        """Retorna hash do snapshot existente ou None."""
        if not self._state_path.exists():
            return None
        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)
            return data.get("config_hash")
        except Exception as e:
            logger.warning(f"[CONFIG_SNAPSHOT] load failed: {e}")
            return None

    def _save(self, normalized: dict, config_hash: str) -> None:
        """Salva snapshot atômico."""
        now = datetime.now(timezone.utc).isoformat()
        existing_created = self._read_created_at()
        payload = {
            "schema_version": CONFIG_SNAPSHOT_SCHEMA_VERSION,
            "created_at": existing_created or now,
            "updated_at": now,
            "config_hash": config_hash,
            "normalized_config": normalized,
        }

        json_str = json.dumps(payload, indent=2)
        state_dir = str(self._state_path.parent)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="config_snap_",
                dir=state_dir, delete=False,
            ) as f:
                temp_path = f.name
                f.write(json_str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, str(self._state_path))
            self._fsync_directory(state_dir)
        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    def _read_created_at(self) -> Optional[str]:
        if not self._state_path.exists():
            return None
        try:
            with open(self._state_path, "r") as f:
                return json.load(f).get("created_at")
        except Exception:
            return None

    def initialize_or_validate(
        self,
        config: Any,
        safety_state: Any,
    ) -> None:
        """
        Inicializa ou valida snapshot.

        Se arquivo não existir: cria snapshot com hash atual.
        Se existir: compara hash. Se diferente → block_buy(CONFIG_CHANGED).
        NÃO sobrescreve quando há mudança.
        """
        normalized = config.to_snapshot_dict() if hasattr(config, "to_snapshot_dict") else dict(config)
        normalized = _redact_sensitive(normalized)
        current_hash = _config_hash(normalized)

        with self._lock:
            existing_hash = self._load_existing()

            if existing_hash is None:
                self._save(normalized, current_hash)
                self._last_hash = current_hash
                logger.info("[CONFIG_SNAPSHOT] Inicializado")
                return

            if existing_hash != current_hash:
                safety_state.block_buy("CONFIG_CHANGED")
                logger.warning(
                    "[CONFIG_SNAPSHOT] Config alterada — BUY bloqueado "
                    f"(hash anterior != atual)"
                )
                return

            self._last_hash = current_hash

    def force_update(self, config: Any) -> None:
        """
        Força atualização do snapshot (após ACK de CONFIG_CHANGED).

        Usado quando operador validou nova config.
        """
        normalized = config.to_snapshot_dict() if hasattr(config, "to_snapshot_dict") else dict(config)
        normalized = _redact_sensitive(normalized)
        current_hash = _config_hash(normalized)
        self._save(normalized, current_hash)
        with self._lock:
            self._last_hash = current_hash
        logger.info("[CONFIG_SNAPSHOT] Atualizado (force)")
