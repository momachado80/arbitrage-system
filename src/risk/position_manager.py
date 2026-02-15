"""
Position Manager — SPOT-ONLY Mode with Atomic Persistence
==========================================================

Rastreia posições por (market_id, outcome_side) em número de contratos (shares).

SPOT-ONLY:
- Apenas posições longas (shares >= 0)
- BUY aumenta shares e recalcula avg_entry_price (weighted average)
- SELL reduz shares; NUNCA permite ficar negativo (ShortNotAllowedError)
- Se SELL zera shares: avg_entry_price retorna a 0 (posição flat)

PERSISTÊNCIA (quando state_path fornecido):
- Escrita atômica: tmp → flush → fsync → os.replace → fsync_dir
- Lock de arquivo com fcntl durante sequência de escrita (POSIX)
- Versionamento de schema (POSITION_SCHEMA_VERSION = 1)
- Timestamps UTC (created_at imutável, updated_at atualizado a cada save)

CORRUPÇÃO:
- Arquivo corrompido → safety_mode=True, state_corrupted=True
- save_state() bloqueado quando corrupted (protege dados potencialmente válidos)
- Apenas reset_all() ou recover_from_corruption() restauram persistência
- safety_mode alimenta o kill switch operacional (bloqueia BUY no pipeline)

SEM PERSISTÊNCIA (state_path=None):
- Modo padrão, 100% in-memory (retrocompatível)
- save_state() é no-op

Thread-safe via RLock.
"""

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional

from src.utils.precision import round_shares, FILL_EPS

logger = logging.getLogger(__name__)

POSITION_SCHEMA_VERSION: int = 1
DEFAULT_POSITION_PATH: str = os.path.expanduser(
    "~/.polymarket_bot/position_state.json"
)

# Tentar importar fcntl (POSIX only)
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False
    logger.warning("fcntl não disponível — file lock desabilitado (Windows)")


class ShortNotAllowedError(Exception):
    """SELL rejeitada: posição insuficiente para vender (spot-only, short proibido)."""


@dataclass
class Position:
    """Estado de uma posição em (market_id, outcome_side)."""
    market_id: str
    outcome_side: str  # "YES" ou "NO"
    shares: float = 0.0
    avg_entry_price: float = 0.0
    last_updated: str = ""


class PositionManager:
    """Motor de posições spot-only com persistência atômica opcional."""

    def __init__(self, state_path: Optional[str] = None) -> None:
        self._persist = state_path is not None
        self._state_path: Optional[Path] = Path(state_path) if state_path else None
        self._positions: Dict[Tuple[str, str], Position] = {}
        self._lock = threading.RLock()
        self._safety_mode: bool = False
        self._state_corrupted: bool = False
        self._created_at: Optional[str] = None

        if self._persist:
            self._ensure_dir()
            self._load_state()
        else:
            self._created_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Directory / Validation helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        if self._state_path:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_utc_timestamp(ts: object) -> bool:
        """Valida timestamp ISO-8601 com timezone UTC (+00:00)."""
        if not isinstance(ts, str):
            return False
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                return False
            offset = dt.utcoffset()
            if offset is None or offset.total_seconds() != 0:
                return False
            return True
        except (ValueError, TypeError):
            return False

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Carrega estado do disco. Corrupção → safety_mode."""
        if self._state_path is None or not self._state_path.exists():
            self._created_at = datetime.now(timezone.utc).isoformat()
            return

        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"[POSITION:LOAD_CORRUPT] JSON inválido: {e}")
            self._safety_mode = True
            self._state_corrupted = True
            return

        # Schema version
        sv = data.get("schema_version")
        if sv != POSITION_SCHEMA_VERSION:
            logger.error(
                f"[POSITION:SCHEMA_MISMATCH] expected={POSITION_SCHEMA_VERSION} got={sv}"
            )
            self._safety_mode = True
            self._state_corrupted = True
            return

        # Validate created_at
        ca = data.get("created_at")
        if not self._validate_utc_timestamp(ca):
            logger.error("[POSITION:INVALID_CREATED_AT]")
            self._safety_mode = True
            self._state_corrupted = True
            return
        self._created_at = ca

        # Validate positions dict
        positions = data.get("positions", {})
        if not isinstance(positions, dict):
            self._safety_mode = True
            self._state_corrupted = True
            return

        for key_str, pos_data in positions.items():
            parts = key_str.split("|", 1)
            if len(parts) != 2:
                continue
            market_id, outcome_side = parts
            shares = pos_data.get("shares", 0.0)
            avg_entry = pos_data.get("avg_entry_price", 0.0)
            if shares < 0 or avg_entry < 0:
                logger.error(
                    f"[POSITION:INVALID_VALUES] {key_str}: "
                    f"shares={shares} avg={avg_entry}"
                )
                self._safety_mode = True
                self._state_corrupted = True
                return
            self._positions[(market_id, outcome_side)] = Position(
                market_id, outcome_side,
                shares, avg_entry,
                data.get("updated_at", ""),
            )

        logger.info(f"Posições carregadas: {len(self._positions)} entradas")

    def save_state(self) -> None:
        """Persiste estado atômicamente. Bloqueado se não-persistente ou corrompido."""
        if not self._persist:
            return
        if self._state_corrupted:
            logger.error(
                "[POSITION:SAVE_BLOCKED] Estado corrompido — "
                "use reset_all() ou recover_from_corruption()"
            )
            return

        # Snapshot sob lock interno
        with self._lock:
            positions_snapshot = {}
            for (mid, side), pos in self._positions.items():
                key = f"{mid}|{side}"
                positions_snapshot[key] = {
                    "shares": round_shares(pos.shares),
                    "avg_entry_price": pos.avg_entry_price,
                }

        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": POSITION_SCHEMA_VERSION,
            "created_at": self._created_at or now,
            "updated_at": now,
            "positions": positions_snapshot,
        }
        json_str = json.dumps(payload, indent=2)

        state_dir = str(self._state_path.parent)
        lock_path = str(self._state_path) + ".lock"
        lock_fd = None
        temp_path = None

        try:
            # File lock (POSIX)
            if _HAS_FCNTL:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Atomic write: tmp → flush → fsync → os.replace
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="position_",
                dir=state_dir, delete=False,
            ) as f:
                temp_path = f.name
                f.write(json_str)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, str(self._state_path))
            temp_path = None  # Replace succeeded

            # fsync directory (durabilidade POSIX)
            try:
                dir_fd = os.open(
                    state_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass

        except Exception as e:
            logger.error(f"[POSITION:SAVE_ERROR] {e}")
            self._safety_mode = True
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if lock_fd is not None:
                if _HAS_FCNTL:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

    # ------------------------------------------------------------------
    # Position operations
    # ------------------------------------------------------------------

    def get_position(self, market_id: str, outcome_side: str) -> Position:
        """Retorna cópia da posição. Se não existir, retorna posição vazia."""
        with self._lock:
            key = (market_id, outcome_side)
            pos = self._positions.get(key)
            if pos is None:
                return Position(market_id, outcome_side)
            return Position(
                pos.market_id, pos.outcome_side,
                pos.shares, pos.avg_entry_price, pos.last_updated,
            )

    def apply_fill(
        self,
        market_id: str,
        outcome_side: str,
        fill_shares: float,
        fill_price: float,
        action: str,
    ) -> None:
        """
        Aplica fill à posição.

        action: "BUY" ou "SELL"
        - BUY: aumenta shares, recalcula avg_entry_price (weighted average)
        - SELL: reduz shares; lança ShortNotAllowedError se insuficiente

        Após atualização, persiste estado (se persistência habilitada).
        """
        if action not in ("BUY", "SELL"):
            raise ValueError(f"action deve ser 'BUY' ou 'SELL', recebido: {action}")
        if fill_shares <= 0:
            return

        with self._lock:
            key = (market_id, outcome_side)
            pos = self._positions.get(key)
            if pos is None:
                pos = Position(market_id, outcome_side)

            if action == "BUY":
                total_cost = pos.shares * pos.avg_entry_price + fill_shares * fill_price
                pos.shares = round_shares(pos.shares + fill_shares)
                pos.avg_entry_price = (
                    total_cost / pos.shares if pos.shares > 0 else 0.0
                )
                logger.info(
                    f"[POSITION:BUY] {market_id}/{outcome_side} "
                    f"+{fill_shares}@{fill_price:.4f} → "
                    f"shares={pos.shares} avg={pos.avg_entry_price:.6f}"
                )

            elif action == "SELL":
                if fill_shares > pos.shares + FILL_EPS:
                    raise ShortNotAllowedError(
                        f"Short não permitido: tentando vender {fill_shares:.8f} "
                        f"shares {outcome_side} em {market_id}, "
                        f"posição atual: {pos.shares:.8f}"
                    )
                pos.shares = round_shares(max(0.0, pos.shares - fill_shares))
                if pos.shares <= FILL_EPS:
                    pos.shares = 0.0
                    pos.avg_entry_price = 0.0
                logger.info(
                    f"[POSITION:SELL] {market_id}/{outcome_side} "
                    f"-{fill_shares}@{fill_price:.4f} → shares={pos.shares}"
                )

            pos.last_updated = datetime.now(timezone.utc).isoformat()
            self._positions[key] = pos

        self.save_state()

    def can_sell(self, market_id: str, outcome_side: str, sell_shares: float) -> bool:
        """Verifica se é possível vender sem entrar em short."""
        with self._lock:
            key = (market_id, outcome_side)
            pos = self._positions.get(key)
            if pos is None:
                return sell_shares <= FILL_EPS
            return sell_shares <= pos.shares + FILL_EPS

    def get_all_positions(self) -> Dict[Tuple[str, str], Position]:
        """Retorna cópia de todas as posições."""
        with self._lock:
            return {
                k: Position(p.market_id, p.outcome_side, p.shares,
                            p.avg_entry_price, p.last_updated)
                for k, p in self._positions.items()
            }

    # ------------------------------------------------------------------
    # Safety / Corruption
    # ------------------------------------------------------------------

    @property
    def safety_mode(self) -> bool:
        return self._safety_mode

    def is_corrupted(self) -> bool:
        """Retorna True se estado corrompido (persistência bloqueada)."""
        return self._state_corrupted

    def recover_from_corruption(self) -> None:
        """Reconhece corrupção e permite save novamente (mantém estado in-memory)."""
        self._state_corrupted = False
        self._safety_mode = False
        if self._created_at is None:
            self._created_at = datetime.now(timezone.utc).isoformat()
        self.save_state()
        logger.info("[POSITION:RECOVERED] Persistência restaurada")

    def reset_all(self) -> None:
        """Limpa tudo e recria arquivo. Restaura persistência."""
        with self._lock:
            self._positions.clear()
        self._state_corrupted = False
        self._safety_mode = False
        self._created_at = datetime.now(timezone.utc).isoformat()
        self.save_state()
        logger.info("[POSITION:RESET] Estado completo reinicializado")
