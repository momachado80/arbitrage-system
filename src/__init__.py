"""
Kalshi-Polymarket Arbitrage System
==================================

Sistema de detecção de arbitragem entre mercados de previsão.

Import lazy para permitir deploy: server pode carregar sem main/collectors.
"""

try:
    from .main import ArbitrageScanner, demo_data_collection
except ImportError:
    ArbitrageScanner = None  # type: ignore[misc, assignment]
    demo_data_collection = None  # type: ignore[misc, assignment]

__all__ = [
    "ArbitrageScanner",
    "demo_data_collection",
]
