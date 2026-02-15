"""
Safety Gate — Kill Switch Operacional
=======================================

Se qualquer componente crítico estiver em modo de segurança ou corrupção,
bloqueia BUY e permite apenas SELL para redução de risco.

GATES:
- Componentes corrompidos (calibrator, position_manager, capital_manager)
- Trade ledger persistence/reconcile falhou
- SafetyState central (reconcile stale, trade drop, capital mismatch, invariant fail)

REGRAS:
- BUY bloqueado se QUALQUER gate ativar
- SELL sempre permitido (reduz risco)
"""

import logging
import time
from typing import Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Gate A: Reconcile Staleness (default)
RECONCILE_STALE_SECONDS: float = 60.0


def can_execute(
    action: str,
    calibrator: Optional[Any] = None,
    capital_manager: Optional[Any] = None,
    position_manager: Optional[Any] = None,
    trade_ledger: Optional[Any] = None,
    safety_state: Optional[Any] = None,
    config: Optional[Any] = None,
) -> Tuple[bool, str]:
    """
    Verifica se a ação pode ser executada com base no estado dos componentes.

    Args:
        action: "BUY" ou "SELL"
        calibrator: ProbabilityCalibrator (opcional)
        capital_manager: CapitalManager (opcional)
        position_manager: PositionManager (opcional)
        trade_ledger: TradeLedger (opcional)
        safety_state: SafetyState central (opcional)

    Returns:
        (True, "") se permitido
        (False, reason) se bloqueado
    """
    # SELL sempre permitido — reduz risco
    if action == "SELL":
        return (True, "")

    # Para BUY: verificar todos os componentes
    reasons = []

    # --- Gates de componentes existentes ---

    if calibrator is not None:
        if getattr(calibrator, "_state_corrupted", False) or (
            callable(getattr(calibrator, "is_corrupted", None))
            and calibrator.is_corrupted()
        ):
            reasons.append("calibrator corrompido")
        elif getattr(calibrator, "safety_mode", False) or getattr(
            calibrator, "_safety_mode", False
        ):
            reasons.append("calibrator em safety_mode")

    if position_manager is not None:
        if callable(getattr(position_manager, "is_corrupted", None)):
            if position_manager.is_corrupted():
                reasons.append("position_manager corrompido")
        if getattr(position_manager, "safety_mode", False) or getattr(
            position_manager, "_safety_mode", False
        ):
            reasons.append("position_manager em safety_mode")

    if capital_manager is not None:
        if callable(getattr(capital_manager, "is_corrupted", None)):
            if capital_manager.is_corrupted():
                reasons.append("capital_manager corrompido")
        elif getattr(capital_manager, "_corrupted", False):
            reasons.append("capital_manager corrompido")

    if trade_ledger is not None:
        if not getattr(trade_ledger, "persistence_ok", True):
            reasons.append("trade_ledger persistência falhou")
        if not getattr(trade_ledger, "reconcile_ok", True):
            reasons.append("reconcile falhou (fill inválido detectado)")

    # --- Gates via SafetyState central ---

    if safety_state is not None:
        # Gate A: Reconcile Staleness
        stale_sec = (
            getattr(config, "reconcile_stale_seconds", None) or RECONCILE_STALE_SECONDS
        )
        last_rec_ts = getattr(safety_state, "last_reconcile_ok_ts", None)
        if last_rec_ts is not None:
            age = time.time() - last_rec_ts
            if age > stale_sec:
                reasons.append(
                    f"reconcile stale ({age:.0f}s > {stale_sec:.0f}s)"
                )
                safety_state.block_buy("RECONCILE_STALE")
        elif last_rec_ts is None:
            # Nunca houve reconcile OK — permitir inicialização
            pass

        # Gate B: Trade Drop
        if getattr(safety_state, "trade_drop_since_last_reconcile", False):
            reasons.append("trade drop detectado desde último reconcile")

        # Gate C: Capital Mismatch
        if getattr(safety_state, "last_capital_mismatch_ts", None) is not None:
            if "CAPITAL_MISMATCH" in getattr(safety_state, "reasons", set()):
                reasons.append("capital mismatch com fonte externa")

        # Invariant failures
        if "INVARIANT_FAIL" in getattr(safety_state, "reasons", set()):
            reasons.append("invariante de runtime violado")

        # Thread dead
        if "THREAD_DEAD" in getattr(safety_state, "reasons", set()):
            reasons.append("thread crítica morreu")

        # Exchange unreachable
        if "EXCHANGE_UNREACHABLE" in getattr(safety_state, "reasons", set()):
            reasons.append("exchange não responde")

        # Config changed
        if "CONFIG_CHANGED" in getattr(safety_state, "reasons", set()):
            reasons.append("config alterada sem ACK")

    if reasons:
        msg = "KILL_SWITCH: " + "; ".join(reasons)
        logger.warning(f"[SAFETY_GATE:BLOCKED] {msg}")
        return (False, msg)

    return (True, "")
