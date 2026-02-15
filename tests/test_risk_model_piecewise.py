"""
Tests — Risk Model Piecewise Linear
====================================

Garante continuidade nos breakpoints e comportamento esperado.
"""

import math

import pytest

from src.api.risk_model import (
    RISK_MODEL_VERSION,
    RiskInputs,
    _piecewise_linear,
    compute_risk_score,
    risk_level_from_score,
)

RESERVED_POINTS = [
    (0.00, 0),
    (0.20, 10),
    (0.30, 15),
    (0.50, 25),
    (1.00, 25),
]


def test_piecewise_continuity_at_breakpoints_reserved():
    """x=0.20±ε não produz salto maior que pequena tolerância."""
    eps = 1e-6
    tol = 1e-4
    for x_center in [0.20, 0.30, 0.50]:
        y_lo = _piecewise_linear(x_center - eps, RESERVED_POINTS)
        y_mid = _piecewise_linear(x_center, RESERVED_POINTS)
        y_hi = _piecewise_linear(x_center + eps, RESERVED_POINTS)
        assert abs(y_mid - y_lo) < tol, f"salto em {x_center}-ε"
        assert abs(y_hi - y_mid) < tol, f"salto em {x_center}+ε"


def test_piecewise_clamp_outside_range():
    """Valores fora do range são clampados."""
    assert _piecewise_linear(-1.0, RESERVED_POINTS) == 0
    assert _piecewise_linear(2.0, RESERVED_POINTS) == 25


def test_risk_model_version_exported():
    """RISK_MODEL_VERSION existe e é 1."""
    assert RISK_MODEL_VERSION == 1


def test_financial_risk_includes_model_version():
    """Payload financial_risk deve incluir model_version."""
    inputs = RiskInputs(
        percent_reserved=0.0,
        max_concentration=0.0,
        open_positions=0,
        global_safe_mode=False,
    )
    from src.api.dashboard_metrics import compute_financial_risk_from_inputs
    cap_data = {"total_capital": 1000.0, "locked_in_open_orders": 0.0, "locked_in_unresolved_markets": 0.0}
    risk = compute_financial_risk_from_inputs(
        cap_data, 0, None, 0.0
    )
    assert risk.get("model_version") == 1
