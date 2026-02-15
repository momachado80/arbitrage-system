"""
Risk Model — Score de Risco Financeiro Acumulativo
====================================================

Modelo gradual, piecewise linear, com continuidade nos breakpoints.
Score 0-100. Versionado.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

RISK_MODEL_VERSION = 1


@dataclass(frozen=True)
class RiskInputs:
    """Inputs para cálculo de risco financeiro."""

    percent_reserved: float  # 0..1 (proporção do capital reservado)
    max_concentration: float  # 0..1 (maior exposição / total_capital)
    open_positions: int  # >= 0
    global_safe_mode: bool
    liquidity_risk: Optional[float] = None  # reservado para futuro (0..1)
    stress_risk: Optional[float] = None  # reservado para futuro (0..1)


def _piecewise_linear(x: float, points: List[Tuple[float, float]]) -> float:
    """
    Interpolação linear por faixa. points: [(x0,y0), (x1,y1), ..., (xn,yn)]
    com x crescente. Clamp fora do range.
    """
    if not points:
        return 0.0
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


# Pontos explícitos por dimensão (continuidade nos breakpoints)
RESERVED_POINTS: List[Tuple[float, float]] = [
    (0.00, 0),
    (0.20, 10),
    (0.30, 15),
    (0.50, 25),
    (1.00, 25),
]
CONCENTRATION_POINTS: List[Tuple[float, float]] = [
    (0.00, 0),
    (0.25, 10),
    (0.40, 30),
    (1.00, 30),
]
POSITIONS_POINTS: List[Tuple[float, float]] = [
    (0, 0),
    (5, 0),
    (6, 10),
    (8, 10),
    (9, 25),
    (50, 25),
]


def _score_reserved(p: float) -> float:
    return _piecewise_linear(max(0.0, min(1.0, p)), RESERVED_POINTS)


def _score_concentration(c: float) -> float:
    return _piecewise_linear(max(0.0, min(1.0, c)), CONCENTRATION_POINTS)


def _score_positions(n: int) -> float:
    return _piecewise_linear(float(max(0, n)), POSITIONS_POINTS)


def compute_risk_score(inputs: RiskInputs) -> int:
    """
    Retorna score 0..100 (acumulativo, gradual, piecewise).
    SAFE_MODE: +40 fixo.
    """
    score = 0.0
    score += _score_reserved(inputs.percent_reserved)
    score += _score_concentration(inputs.max_concentration)
    score += _score_positions(inputs.open_positions)
    if inputs.global_safe_mode:
        score += 40
    if inputs.liquidity_risk is not None and inputs.liquidity_risk > 0:
        score += min(10, inputs.liquidity_risk * 20)
    if inputs.stress_risk is not None and inputs.stress_risk > 0:
        score += min(10, inputs.stress_risk * 20)
    return max(0, min(100, int(round(score))))


def risk_level_from_score(score: int) -> Dict[str, Any]:
    """
    Níveis: Baixo (<25), Moderado (<60), Elevado (>=60).
    """
    if score < 25:
        return {"level": "Baixo", "color": "green", "score": score}
    if score < 60:
        return {"level": "Moderado", "color": "yellow", "score": score}
    return {"level": "Elevado", "color": "red", "score": score}


def build_risk_explanation(inputs: RiskInputs, score: int) -> str:
    """
    Explicação humana curta dos fatores de risco.
    """
    parts = []
    if inputs.global_safe_mode:
        parts.append("Modo de segurança ativo.")
    if inputs.percent_reserved > 0.01:
        pct = round(inputs.percent_reserved * 100)
        parts.append(f"{pct}% do capital está reservado.")
    if inputs.max_concentration > 0.05:
        pct = round(inputs.max_concentration * 100)
        parts.append(f"Concentração de {pct}% em um único mercado.")
    if inputs.open_positions > 0:
        parts.append(f"{inputs.open_positions} posições abertas.")
    if not parts:
        return "Exposição controlada."
    return " ".join(parts)
