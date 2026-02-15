"""
SafetyState — Estado Central de Segurança
===========================================

Armazena razões consolidadas de bloqueio de BUY de forma thread-safe.
Todos os gates, invariants e sensores publicam aqui.

buy_blocked é True se reasons não vazio.

MOTIVOS PADRONIZADOS:
- RECONCILE_STALE:      último reconcile OK muito antigo
- TRADE_DROP:           trade drop detectado desde último reconcile
- CAPITAL_MISMATCH:     saldo local diverge do externo
- INVARIANT_FAIL:       invariante de runtime violado
- THREAD_DEAD:          thread crítica morreu
- EXCHANGE_UNREACHABLE: exchange não responde
- SAFE_MODE:            modo seguro global (escalada automática)

STRIKES TO CLEAR:
- INVARIANT_FAIL: só limpa automaticamente após invariants_ok_streak >= 3
- CAPITAL_MISMATCH: só limpa automaticamente após capital_mismatch_ok_streak >= 3
- Nenhum clear automático quando SAFE_MODE ativo.

SAFE_MODE:
- Ativado por escalada (invariant fail + capital mismatch com safe mode flag).
- Desativado apenas via ack explícito com precondições saudáveis.
"""

import logging
import threading
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Motivos padronizados (strings constantes)
RECONCILE_STALE: str = "RECONCILE_STALE"
TRADE_DROP: str = "TRADE_DROP"
CAPITAL_MISMATCH: str = "CAPITAL_MISMATCH"
INVARIANT_FAIL: str = "INVARIANT_FAIL"
THREAD_DEAD: str = "THREAD_DEAD"
EXCHANGE_UNREACHABLE: str = "EXCHANGE_UNREACHABLE"
SAFE_MODE: str = "SAFE_MODE"
CONFIG_CHANGED: str = "CONFIG_CHANGED"
CLOCK_ANOMALY: str = "CLOCK_ANOMALY"

# Strikes necessários para clear automático
STRIKES_TO_CLEAR: int = 3


class SafetyState:
    """
    Estado central de segurança financeira.

    Thread-safe. Métodos idempotentes.
    buy_blocked é True se reasons não vazio.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reasons: Set[str] = set()
        self._last_reconcile_ok_ts: Optional[float] = None
        self._trade_drop_since_last_reconcile: bool = False
        self._last_capital_mismatch_ts: Optional[float] = None
        self._invariant_fail_count: int = 0
        # Streaks para "3 strikes to clear"
        self._invariants_ok_streak: int = 0
        self._capital_mismatch_ok_streak: int = 0
        # Safe mode
        self._global_safe_mode: bool = False
        # Reconcile session id
        self._last_reconcile_session_id: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def buy_blocked(self) -> bool:
        """True se há qualquer razão de bloqueio ativa."""
        with self._lock:
            return len(self._reasons) > 0

    @property
    def reasons(self) -> Set[str]:
        """Cópia do set de razões ativas."""
        with self._lock:
            return set(self._reasons)

    @property
    def last_reconcile_ok_ts(self) -> Optional[float]:
        with self._lock:
            return self._last_reconcile_ok_ts

    @property
    def trade_drop_since_last_reconcile(self) -> bool:
        with self._lock:
            return self._trade_drop_since_last_reconcile

    @property
    def last_capital_mismatch_ts(self) -> Optional[float]:
        with self._lock:
            return self._last_capital_mismatch_ts

    @property
    def invariant_fail_count(self) -> int:
        with self._lock:
            return self._invariant_fail_count

    @property
    def invariants_ok_streak(self) -> int:
        with self._lock:
            return self._invariants_ok_streak

    @property
    def capital_mismatch_ok_streak(self) -> int:
        with self._lock:
            return self._capital_mismatch_ok_streak

    @property
    def global_safe_mode(self) -> bool:
        with self._lock:
            return self._global_safe_mode

    @property
    def last_reconcile_session_id(self) -> int:
        with self._lock:
            return self._last_reconcile_session_id

    # ------------------------------------------------------------------
    # Mutações idempotentes
    # ------------------------------------------------------------------

    def block_buy(self, reason: str) -> None:
        """Adiciona razão de bloqueio. Idempotente."""
        with self._lock:
            if reason not in self._reasons:
                self._reasons.add(reason)
                logger.warning(
                    f"[SAFETY:BUY_BLOCKED] reason={reason} "
                    f"total_reasons={len(self._reasons)}"
                )

    def clear_reason(self, reason: str) -> None:
        """Remove razão de bloqueio. Idempotente."""
        with self._lock:
            self._reasons.discard(reason)

    def set_last_reconcile_ok(self, ts: float) -> None:
        """Atualiza timestamp do último reconcile bem-sucedido."""
        with self._lock:
            self._last_reconcile_ok_ts = ts

    def mark_trade_drop(self) -> None:
        """Marca que houve trade drop desde o último reconcile."""
        with self._lock:
            if not self._trade_drop_since_last_reconcile:
                self._trade_drop_since_last_reconcile = True
                self._reasons.add(TRADE_DROP)
                logger.warning("[SAFETY:TRADE_DROP] Trade drop detectado")

    def clear_trade_drop_flag(self) -> None:
        """Limpa flag de trade drop (após reconcile OK)."""
        with self._lock:
            self._trade_drop_since_last_reconcile = False
            self._reasons.discard(TRADE_DROP)

    def mark_capital_mismatch(self, ts: float) -> None:
        """Marca mismatch de capital com fonte externa."""
        with self._lock:
            self._last_capital_mismatch_ts = ts
            self._reasons.add(CAPITAL_MISMATCH)
            logger.warning(
                f"[SAFETY:CAPITAL_MISMATCH] ts={ts:.0f}"
            )

    def increment_invariant_fail(self) -> None:
        """Incrementa contador de falhas de invariant."""
        with self._lock:
            self._invariant_fail_count += 1

    # ------------------------------------------------------------------
    # Reconcile session id
    # ------------------------------------------------------------------

    def set_reconcile_session_id(self, session_id: int) -> None:
        """Atualiza o session_id do último reconcile."""
        with self._lock:
            self._last_reconcile_session_id = session_id

    # ------------------------------------------------------------------
    # Invariants streak (3 strikes to clear)
    # ------------------------------------------------------------------

    def mark_invariants_ok(self) -> None:
        """Incrementa streak de invariants OK."""
        with self._lock:
            self._invariants_ok_streak += 1

    def mark_invariants_fail(self) -> None:
        """Zera streak e bloqueia BUY."""
        with self._lock:
            self._invariants_ok_streak = 0

    def maybe_clear_invariant_fail(self) -> bool:
        """
        Limpa INVARIANT_FAIL se streak >= STRIKES_TO_CLEAR
        e SAFE_MODE não está ativo.

        Returns True se limpou.
        """
        with self._lock:
            if self._global_safe_mode:
                return False
            if self._invariants_ok_streak >= STRIKES_TO_CLEAR:
                self._reasons.discard(INVARIANT_FAIL)
                logger.info(
                    f"[SAFETY:AUTO_CLEAR] INVARIANT_FAIL cleared "
                    f"(streak={self._invariants_ok_streak})"
                )
                return True
            return False

    # ------------------------------------------------------------------
    # Capital mismatch streak (3 strikes to clear)
    # ------------------------------------------------------------------

    def mark_capital_mismatch_ok(self) -> None:
        """Incrementa streak de capital mismatch OK."""
        with self._lock:
            self._capital_mismatch_ok_streak += 1

    def mark_capital_mismatch_fail(self) -> None:
        """Zera streak de capital mismatch."""
        with self._lock:
            self._capital_mismatch_ok_streak = 0

    def maybe_clear_capital_mismatch(self) -> bool:
        """
        Limpa CAPITAL_MISMATCH se streak >= STRIKES_TO_CLEAR
        e SAFE_MODE não está ativo.

        Returns True se limpou.
        """
        with self._lock:
            if self._global_safe_mode:
                return False
            if self._capital_mismatch_ok_streak >= STRIKES_TO_CLEAR:
                self._reasons.discard(CAPITAL_MISMATCH)
                logger.info(
                    f"[SAFETY:AUTO_CLEAR] CAPITAL_MISMATCH cleared "
                    f"(streak={self._capital_mismatch_ok_streak})"
                )
                return True
            return False

    # ------------------------------------------------------------------
    # Safe Mode (escalada)
    # ------------------------------------------------------------------

    def enable_safe_mode(self, trigger_reason: str) -> None:
        """
        Ativa modo seguro global. BUY bloqueado por SAFE_MODE.

        Também bloqueia pela trigger_reason original.
        """
        with self._lock:
            if not self._global_safe_mode:
                self._global_safe_mode = True
                self._reasons.add(SAFE_MODE)
                logger.critical(
                    f"[SAFETY:SAFE_MODE_ENABLED] trigger={trigger_reason}"
                )
            # Garantir que trigger reason também está ativa
            self._reasons.add(trigger_reason)

    def disable_safe_mode(self) -> None:
        """
        Desativa modo seguro global.

        Chamado por ack_reason após verificação de precondições.
        """
        with self._lock:
            self._global_safe_mode = False
            self._reasons.discard(SAFE_MODE)
            logger.info("[SAFETY:SAFE_MODE_DISABLED]")

    # ------------------------------------------------------------------
    # Leitura para telemetria
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Snapshot para telemetria/heartbeat."""
        with self._lock:
            return {
                "buy_blocked": len(self._reasons) > 0,
                "buy_block_reasons": sorted(self._reasons),
                "last_reconcile_ok_ts": self._last_reconcile_ok_ts,
                "trade_drop_since_last_reconcile": self._trade_drop_since_last_reconcile,
                "last_capital_mismatch_ts": self._last_capital_mismatch_ts,
                "invariant_fail_count": self._invariant_fail_count,
                "invariants_ok_streak": self._invariants_ok_streak,
                "capital_mismatch_ok_streak": self._capital_mismatch_ok_streak,
                "global_safe_mode": self._global_safe_mode,
                "last_reconcile_session_id": self._last_reconcile_session_id,
            }
