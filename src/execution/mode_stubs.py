"""
Execution stubs para RUN_MODE (OBSERVE / PAPER)
=================================================

OBSERVE: OrderManager stub que não coloca ordens.
PAPER: ExchangeAPI stub que simula fills conservadores.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.execution.order_manager import OrderManager, OrderSide, OrderStatus
from src.execution.reconciler import ExchangeAPI

# Ordem alfabética para hash determinístico


class ObserveOrderManager:
    """
    Stub para OBSERVE mode: place_limit_order retorna [] (nunca coloca ordens).

    cancel_order, cancel_all_buy_open_orders_best_effort, sync_order_status
    são no-ops. check_order_status retorna None.
    """

    def __init__(self) -> None:
        self._orders: Dict[str, Any] = {}

    def place_limit_order(
        self,
        market_id: str,
        side: Any,
        price: float,
        size: float,
        ttl_seconds: int = 60,
        outcome_side: Optional[Any] = None,
        position_manager: Optional[Any] = None,
        capital_manager: Optional[Any] = None,
    ) -> List[str]:
        """OBSERVE: não coloca ordens. Retorna lista vazia."""
        return []

    def cancel_order(self, order_id: str) -> bool:
        return False

    def cancel_all_buy_open_orders_best_effort(self) -> List[str]:
        return []

    def sync_order_status(
        self, order_id: str, status: str, filled_size: float
    ) -> None:
        pass

    def check_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        return None


class PaperExchangeAPI(ExchangeAPI):
    """
    ExchangeAPI para PAPER mode: retorna estado do OrderManager local
    e simula fills conservadores (preenche no preço da ordem).
    """

    def __init__(self, order_manager: OrderManager) -> None:
        self._om = order_manager
        self._fill_counter: int = 0
        self._simulated_fills: Dict[str, List[Dict[str, Any]]] = {}
        self._simulated_status: Dict[str, Dict[str, Any]] = {}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        if order_id in self._simulated_status:
            return self._simulated_status[order_id]
        status = self._om.check_order_status(order_id)
        if status is None:
            return {"order_id": order_id, "status": "PENDING", "filled_size": 0.0}
        return {
            "order_id": order_id,
            "status": status.get("status", "PENDING"),
            "filled_size": status.get("filled_size", 0.0),
        }

    def get_fills(self, order_id: str) -> List[Dict[str, Any]]:
        """Simula fill completo no preço da ordem (conservador)."""
        if order_id in self._simulated_fills:
            return self._simulated_fills[order_id]

        status = self._om.check_order_status(order_id)
        if status is None:
            return []

        st = status.get("status", "PENDING")
        filled = status.get("filled_size", 0.0)
        size = status.get("size", 0.0)
        price = status.get("price", 0.0)

        if filled >= size - 1e-10 or st in ("CANCELLED", "EXPIRED", "FILLED"):
            return []

        self._fill_counter += 1
        fill_size = size - filled
        fill = {
            "fill_id": f"paper_fill_{order_id}_{self._fill_counter}",
            "order_id": order_id,
            "price": price,
            "size": fill_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._simulated_fills[order_id] = [fill]
        self._simulated_status[order_id] = {
            "order_id": order_id,
            "status": "FILLED",
            "filled_size": size,
        }
        return [fill]

    def get_position(self, market_id: str) -> Dict[str, Any]:
        return {"market_id": market_id, "size": 0.0, "avg_entry": 0.0}

    def get_account_summary(self) -> Optional[Dict[str, Any]]:
        return None
