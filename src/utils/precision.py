"""
Precision & Tolerance Module
=============================

Arredondamento monetário e tolerâncias para evitar problemas de float
em contabilidade de capital, posições e fills.

CONSTANTES CONFIGURÁVEIS:
- PRICE_DECIMALS:    casas decimais para preços (6)
- SHARES_DECIMALS:   casas decimais para shares/contratos (8)
- NOTIONAL_DECIMALS: casas decimais para valores em USD (6)
- FILL_EPS:          tolerância para comparações de fill (1e-8)

USO:
    from src.utils.precision import round_price, round_shares, FILL_EPS
"""

# ===== Constantes configuráveis =====

PRICE_DECIMALS: int = 6
SHARES_DECIMALS: int = 8
NOTIONAL_DECIMALS: int = 6
FILL_EPS: float = 1e-8


# ===== Funções de arredondamento =====

def round_price(x: float) -> float:
    """Arredonda preço para PRICE_DECIMALS casas decimais."""
    return round(x, PRICE_DECIMALS)


def round_shares(x: float) -> float:
    """Arredonda shares para SHARES_DECIMALS casas decimais."""
    return round(x, SHARES_DECIMALS)


def round_notional(x: float) -> float:
    """Arredonda notional (USD) para NOTIONAL_DECIMALS casas decimais."""
    return round(x, NOTIONAL_DECIMALS)


def is_close(a: float, b: float, eps: float = FILL_EPS) -> bool:
    """Compara dois floats com tolerância eps."""
    return abs(a - b) <= eps
