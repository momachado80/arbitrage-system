"""
Notional Helpers
================

Funções utilitárias para cálculo de notional (valor em USD) em modo spot-only.

Em mercados preditivos:
- shares = número de contratos
- price  = preço por contrato (0 < price < 1)
- notional = shares * price = custo total em USD

Exemplo:
  100 shares YES a $0.65 → notional = $65.00
"""

from src.utils.precision import round_notional


def compute_notional(shares: float, price: float) -> float:
    """
    Calcula notional (USD) = round(shares * price, NOTIONAL_DECIMALS).

    Args:
        shares: número de contratos
        price: preço por contrato

    Returns:
        Valor arredondado em USD (notional).
    """
    return round_notional(shares * price)
