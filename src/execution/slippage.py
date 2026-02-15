"""
Slippage / Impact Estimator
============================

Estima custo de preenchimento caminhando pelo order book.

Para BUY: consome asks (do mais barato ao mais caro)
Para SELL: consome bids (do mais caro ao mais barato)

O order book deve estar ordenado:
- bids: preço decrescente [(0.65, 100), (0.64, 200), ...]
- asks: preço crescente [(0.66, 100), (0.67, 200), ...]
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def estimate_fill_cost(
    order_book: Dict[str, List[Tuple[float, float]]],
    side: str,
    size: float,
) -> Dict[str, Any]:
    """
    Estima custo de preenchimento caminhando pelo order book.

    Args:
        order_book: {"bids": [(price, size), ...], "asks": [(price, size), ...]}
        side: "BUY" ou "SELL"
        size: tamanho desejado

    Returns:
        {
            "expected_avg_price": float,
            "worst_price": float,
            "slippage_vs_mid": float,
            "required_liquidity_ok": bool,
        }
    """
    if size <= 0:
        return {
            "expected_avg_price": 0.0,
            "worst_price": 0.0,
            "slippage_vs_mid": 0.0,
            "required_liquidity_ok": False,
        }

    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])

    if not bids or not asks:
        return {
            "expected_avg_price": 0.0,
            "worst_price": 0.0,
            "slippage_vs_mid": 0.0,
            "required_liquidity_ok": False,
        }

    # Mid price
    mid = (bids[0][0] + asks[0][0]) / 2.0

    # Selecionar lado do book
    if side == "BUY":
        levels = asks  # consume asks (preço crescente)
    else:
        levels = bids  # consume bids (preço decrescente)

    total_filled = 0.0
    total_cost = 0.0
    worst_price = 0.0

    for price, level_size in levels:
        can_fill = min(level_size, size - total_filled)
        total_cost += can_fill * price
        total_filled += can_fill
        worst_price = price
        if total_filled >= size:
            break

    if total_filled < size:
        logger.warning(
            f"[SLIPPAGE:INSUFFICIENT_LIQUIDITY] "
            f"side={side} needed={size} available={total_filled}"
        )
        return {
            "expected_avg_price": total_cost / total_filled if total_filled > 0 else 0.0,
            "worst_price": worst_price,
            "slippage_vs_mid": abs((total_cost / total_filled) - mid) if total_filled > 0 else 0.0,
            "required_liquidity_ok": False,
        }

    avg_price = total_cost / total_filled
    slippage = abs(avg_price - mid)

    return {
        "expected_avg_price": round(avg_price, 8),
        "worst_price": worst_price,
        "slippage_vs_mid": round(slippage, 8),
        "required_liquidity_ok": True,
    }
