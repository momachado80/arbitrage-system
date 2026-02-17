"""Execution module — Order management, reconciliation, and trade pipeline."""

from .audit_logger import SimulationAuditLogger
from .order_manager import (
    OrderManager, Order, OrderStatus, OrderSide, OutcomeSide,
    MAX_CHILD_SIZE, DEFAULT_TTL,
)
from .pipeline import execute_trade_decision, resolve_trade
from .reconciler import OrderReconciler, ExchangeAPI
from .slippage import estimate_fill_cost
from .notional import compute_notional
from .realistic_simulator import RealisticSimulationOrderManager, RealisticSimulationExchangeAPI

__all__ = [
    "OrderManager", "Order", "OrderStatus", "OrderSide", "OutcomeSide",
    "MAX_CHILD_SIZE", "DEFAULT_TTL",
    "execute_trade_decision", "resolve_trade",
    "OrderReconciler", "ExchangeAPI",
    "estimate_fill_cost",
    "compute_notional",
    "SimulationAuditLogger",
    "RealisticSimulationOrderManager",
    "RealisticSimulationExchangeAPI",
]
