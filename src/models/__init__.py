"""Models module - Core data structures for the arbitrage system."""

from .core import (
    Platform,
    MarketStatus,
    Side,
    PriceLevel,
    OrderBook,
    Market,
    ExecutionSimulation,
    FeeStructure,
    ArbitrageOpportunity,
    DEFAULT_MIN_SIZE_USD,
    DEFAULT_SLIPPAGE_BUFFER,
    DEFAULT_BASIS_RISK_PENALTY,
    MIN_EDGE_THRESHOLD,
)

__all__ = [
    "Platform",
    "MarketStatus",
    "Side",
    "PriceLevel",
    "OrderBook",
    "Market",
    "ExecutionSimulation",
    "FeeStructure",
    "ArbitrageOpportunity",
    "DEFAULT_MIN_SIZE_USD",
    "DEFAULT_SLIPPAGE_BUFFER",
    "DEFAULT_BASIS_RISK_PENALTY",
    "MIN_EDGE_THRESHOLD",
]
