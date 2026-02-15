"""
Strategy Module — Statistical Reversion V1 (OBSERVE)
====================================================

Pure, deterministic, side-effect free.
"""

from src.strategy.reversion_engine import (
    ReversionConfig,
    ReversionEngine,
    ReversionMetricsCollector,
    ReversionSignal,
    ReversionState,
    create_default_config,
)

__all__ = [
    "ReversionConfig",
    "ReversionEngine",
    "ReversionMetricsCollector",
    "ReversionSignal",
    "ReversionState",
    "create_default_config",
]
