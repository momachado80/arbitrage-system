"""
Kalshi-Polymarket Arbitrage System
==================================

Sistema de detecção de arbitragem entre mercados de previsão.
"""

from .main import ArbitrageScanner, demo_data_collection

__all__ = [
    "ArbitrageScanner",
    "demo_data_collection",
]
