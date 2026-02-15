"""
AppConfig — Configuração central carregada de ENV
====================================================

Thresholds e RUN_MODE com defaults seguros.
Validação de valores e normalização de run_mode.

RUN_MODE:
- OBSERVE: não envia ordens reais
- PAPER: execução simulada (stub)
- LIVE: execução real (requer pré-condições fortes)
"""

import os
from dataclasses import dataclass
from typing import Optional

VALID_RUN_MODES = ("OBSERVE", "PAPER", "LIVE")
DEFAULT_RUN_MODE = "OBSERVE"


@dataclass(frozen=True)
class AppConfig:
    """
    Configuração imutável do sistema.

    Carregada apenas de ENV. Sem segredos expostos em logs.
    """

    run_mode: str
    reconcile_loop_interval_seconds: int
    reconcile_stale_seconds: float
    queue_maxsize: int
    trade_drop_strikes_to_clear: int
    invariant_strikes_to_clear: int
    capital_mismatch_percent_tolerance: float
    capital_mismatch_abs_tolerance: float
    health_check_interval_seconds: int
    enable_dashboard: bool = False
    dashboard_port: int = 8765

    def to_snapshot_dict(self) -> dict:
        """
        Dict normalizado para snapshot/hash (sem campos sensíveis).
        Chaves ordenadas para hash determinístico.
        """
        return {
            "run_mode": self.run_mode,
            "reconcile_loop_interval_seconds": self.reconcile_loop_interval_seconds,
            "reconcile_stale_seconds": self.reconcile_stale_seconds,
            "queue_maxsize": self.queue_maxsize,
            "trade_drop_strikes_to_clear": self.trade_drop_strikes_to_clear,
            "invariant_strikes_to_clear": self.invariant_strikes_to_clear,
            "capital_mismatch_percent_tolerance": self.capital_mismatch_percent_tolerance,
            "capital_mismatch_abs_tolerance": self.capital_mismatch_abs_tolerance,
            "health_check_interval_seconds": self.health_check_interval_seconds,
            "enable_dashboard": self.enable_dashboard,
            "dashboard_port": self.dashboard_port,
        }


def _int_env(key: str, default: int, min_val: Optional[int] = None) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    if min_val is not None and val < min_val:
        return default
    return val


def _float_env(key: str, default: float, min_val: Optional[float] = None) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    if min_val is not None and val < min_val:
        return default
    return val


def load_app_config_from_env() -> AppConfig:
    """
    Carrega AppConfig a partir de variáveis de ambiente.

    Variáveis:
    - RUN_MODE
    - RECONCILE_LOOP_INTERVAL_SECONDS
    - RECONCILE_STALE_SECONDS
    - QUEUE_MAXSIZE
    - TRADE_DROP_STRIKES_TO_CLEAR
    - INVARIANT_STRIKES_TO_CLEAR
    - CAPITAL_MISMATCH_PERCENT_TOLERANCE
    - CAPITAL_MISMATCH_ABS_TOLERANCE
    - HEALTH_CHECK_INTERVAL_SECONDS

    Raises:
        ValueError: se run_mode for inválido
    """
    run_mode_raw = os.environ.get("RUN_MODE", DEFAULT_RUN_MODE)
    run_mode = run_mode_raw.strip().upper() if run_mode_raw else DEFAULT_RUN_MODE
    if run_mode not in VALID_RUN_MODES:
        raise ValueError(
            f"RUN_MODE inválido: '{run_mode_raw}'. "
            f"Válidos: {VALID_RUN_MODES}"
        )

    reconcile_interval = _int_env(
        "RECONCILE_LOOP_INTERVAL_SECONDS", 30, min_val=1
    )
    reconcile_stale = _float_env(
        "RECONCILE_STALE_SECONDS", 60.0, min_val=1.0
    )
    queue_maxsize = _int_env("QUEUE_MAXSIZE", 100, min_val=1)
    trade_drop_strikes = _int_env(
        "TRADE_DROP_STRIKES_TO_CLEAR", 3, min_val=1
    )
    invariant_strikes = _int_env(
        "INVARIANT_STRIKES_TO_CLEAR", 3, min_val=1
    )
    cap_pct_tol = _float_env(
        "CAPITAL_MISMATCH_PERCENT_TOLERANCE", 0.001, min_val=0.0
    )
    cap_abs_tol = _float_env(
        "CAPITAL_MISMATCH_ABS_TOLERANCE", 0.01, min_val=0.0
    )
    health_interval = _int_env(
        "HEALTH_CHECK_INTERVAL_SECONDS", 30, min_val=5
    )
    enable_dashboard = os.environ.get("ENABLE_DASHBOARD", "").strip().upper() in ("1", "TRUE", "YES")
    dashboard_port = _int_env("DASHBOARD_PORT", 8765, min_val=1024)

    return AppConfig(
        run_mode=run_mode,
        reconcile_loop_interval_seconds=reconcile_interval,
        reconcile_stale_seconds=reconcile_stale,
        queue_maxsize=queue_maxsize,
        trade_drop_strikes_to_clear=trade_drop_strikes,
        invariant_strikes_to_clear=invariant_strikes,
        capital_mismatch_percent_tolerance=cap_pct_tol,
        capital_mismatch_abs_tolerance=cap_abs_tol,
        health_check_interval_seconds=health_interval,
        enable_dashboard=enable_dashboard,
        dashboard_port=dashboard_port,
    )
