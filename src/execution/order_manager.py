"""
Order Manager Module
====================

Limit order execution engine para mercados preditivos.

FILOSOFIA:
- NUNCA usar market orders
- Todas as ordens são limite com TTL obrigatório
- Ordens grandes divididas em micro-lotes
- Toxic flow protection: cancelar ordens expostas a movimento adverso

SPOT-ONLY MODE (account_mode="SPOT_ONLY"):
- Apenas posições longas (comprar YES ou NO)
- SELL só permitido para reduzir/encerrar posição existente
- Short proibido: vender mais do que possui → ShortNotAllowedError
- BUY requer capital disponível suficiente (notional <= available)

TOXIC FLOW:
Ordens limite que ficaram expostas a movimento adverso de preço
(ex: BUY quando preço de mercado caiu abaixo do limite) indicam possível
seleção adversa de microestrutura. Essas ordens são automaticamente canceladas.

Este módulo NÃO integra com API real — apenas estrutura e interfaces.
"""

import logging
import uuid
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

MAX_CHILD_SIZE: float = 50.0
DEFAULT_TTL: int = 60


class OrderStatus(Enum):
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OutcomeSide(Enum):
    """Lado do outcome em mercados preditivos (YES/NO)."""
    YES = "YES"
    NO = "NO"


@dataclass
class Order:
    order_id: str
    market_id: str
    side: OrderSide
    price: float
    size: float
    filled_size: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: str = ""
    ttl_seconds: int = DEFAULT_TTL
    parent_order_id: Optional[str] = None

    @property
    def remaining_size(self) -> float:
        return round(self.size - self.filled_size, 10)

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED)


class OrderManager:
    """Limit order execution engine (suporta SPOT_ONLY mode)."""

    def __init__(
        self,
        max_child_size: float = MAX_CHILD_SIZE,
        account_mode: str = "SPOT_ONLY",
    ):
        self._orders: Dict[str, Order] = {}
        self._lock = threading.RLock()
        self._max_child_size = max_child_size
        self._account_mode = account_mode

    def place_limit_order(
        self,
        market_id: str,
        side: OrderSide,
        price: float,
        size: float,
        ttl_seconds: int = DEFAULT_TTL,
        outcome_side: Optional[Any] = None,
        position_manager: Optional[Any] = None,
        capital_manager: Optional[Any] = None,
    ) -> List[str]:
        """
        Coloca ordem limite. Retorna lista de order_ids (micro-lotes se necessário).

        Em SPOT_ONLY mode (quando outcome_side é fornecido):
        - SELL: valida posição via position_manager.can_sell()
        - BUY: valida capital via capital_manager.get_available_capital()
        """
        if size <= 0:
            raise ValueError(f"size deve ser > 0, recebido: {size}")
        if price <= 0 or price >= 1:
            raise ValueError(f"price deve estar em (0, 1), recebido: {price}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds deve ser > 0, recebido: {ttl_seconds}")

        # Spot-only validation
        if self._account_mode == "SPOT_ONLY" and outcome_side is not None:
            oc_str = outcome_side.value if hasattr(outcome_side, "value") else str(outcome_side)
            if side == OrderSide.SELL:
                if position_manager is not None:
                    shares_to_sell = size / price
                    if not position_manager.can_sell(market_id, oc_str, shares_to_sell):
                        logger.warning(
                            f"[ORDER:REJECTED_SHORT_NOT_ALLOWED] "
                            f"market={market_id} {oc_str} shares={shares_to_sell:.4f}"
                        )
                        from src.risk.position_manager import ShortNotAllowedError
                        raise ShortNotAllowedError(
                            f"Short não permitido: tentando vender {shares_to_sell:.4f} "
                            f"shares {oc_str} em {market_id}"
                        )
            elif side == OrderSide.BUY:
                if capital_manager is not None:
                    avail = capital_manager.get_available_capital()
                    if size > avail + 1e-10:
                        logger.warning(
                            f"[ORDER:REJECTED_INSUFFICIENT_CAPITAL] "
                            f"required={size:.2f} available={avail:.2f}"
                        )
                        from src.risk.capital_manager import InsufficientCapitalError
                        raise InsufficientCapitalError(
                            f"Capital insuficiente: requerido={size:.2f}, "
                            f"disponível={avail:.2f}"
                        )

        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if size <= self._max_child_size:
                order_id = str(uuid.uuid4())
                self._orders[order_id] = Order(
                    order_id=order_id, market_id=market_id,
                    side=side, price=price, size=size,
                    created_at=now, ttl_seconds=ttl_seconds,
                )
                logger.info(
                    f"[ORDER:CREATED] {order_id} {side.value} "
                    f"{size}@{price} market={market_id} ttl={ttl_seconds}s"
                )
                return [order_id]
            else:
                parent_id = str(uuid.uuid4())
                order_ids: List[str] = []
                remaining = size
                while remaining > 1e-10:
                    child_size = min(remaining, self._max_child_size)
                    order_id = str(uuid.uuid4())
                    self._orders[order_id] = Order(
                        order_id=order_id, market_id=market_id,
                        side=side, price=price, size=round(child_size, 10),
                        created_at=now, ttl_seconds=ttl_seconds,
                        parent_order_id=parent_id,
                    )
                    order_ids.append(order_id)
                    remaining = round(remaining - child_size, 10)
                logger.info(
                    f"[ORDER:MICRO-LOT] Parent {parent_id}: "
                    f"{len(order_ids)} ordens ({side.value} {size}@{price})"
                )
                return order_ids

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or not order.is_active:
                return False
            order.status = OrderStatus.CANCELLED
            logger.info(f"[ORDER:CANCELLED] {order_id} filled={order.filled_size}/{order.size}")
            return True

    def check_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return None
            return {
                "order_id": order.order_id,
                "market_id": order.market_id,
                "side": order.side.value,
                "price": order.price,
                "size": order.size,
                "filled_size": order.filled_size,
                "remaining_size": order.remaining_size,
                "status": order.status.value,
                "created_at": order.created_at,
                "ttl_seconds": order.ttl_seconds,
                "parent_order_id": order.parent_order_id,
            }

    def cancel_all_buy_open_orders_best_effort(self) -> List[str]:
        """
        Mitigação ativa: cancela todas as ordens BUY ativas.

        SELL não é cancelado (reduz risco).
        Idempotente, nunca lança exceção que derrube thread.

        Returns:
            Lista de order_ids cancelados.
        """
        cancelled: List[str] = []
        try:
            with self._lock:
                for order in self._orders.values():
                    if not order.is_active:
                        continue
                    if order.side != OrderSide.BUY:
                        continue
                    order.status = OrderStatus.CANCELLED
                    cancelled.append(order.order_id)
            if cancelled:
                logger.warning(
                    f"[SAFETY:MITIGATION] Canceladas {len(cancelled)} ordens BUY: "
                    f"{cancelled[:5]}{'...' if len(cancelled) > 5 else ''}"
                )
        except Exception as e:
            logger.error(f"[SAFETY:MITIGATION_ERROR] {e}")
        return cancelled

    def cancel_stale_orders(self) -> List[str]:
        now = datetime.now(timezone.utc)
        cancelled: List[str] = []
        with self._lock:
            for order in self._orders.values():
                if not order.is_active:
                    continue
                created = datetime.fromisoformat(order.created_at)
                if (now - created).total_seconds() >= order.ttl_seconds:
                    order.status = OrderStatus.EXPIRED
                    cancelled.append(order.order_id)
                    logger.info(f"[ORDER:EXPIRED] {order.order_id}")
        return cancelled

    def check_toxic_flow(
        self, order_id: str, current_price: float
    ) -> bool:
        """
        Toxic flow protection: cancela ordens limite expostas a movimento
        adverso de preço (seleção adversa de microestrutura).

        BUY em P: adverso se current_price < P (mercado caiu pelo bid)
        SELL em P: adverso se current_price > P (mercado subiu pelo ask)

        Retorna True se cancelou por toxic flow.
        """
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or not order.is_active:
                return False
            adverse = False
            if order.side == OrderSide.BUY and current_price < order.price:
                adverse = True
            elif order.side == OrderSide.SELL and current_price > order.price:
                adverse = True
            if adverse:
                order.status = OrderStatus.CANCELLED
                logger.warning(
                    f"[ORDER:TOXIC_FLOW_CANCELLED] {order.order_id} "
                    f"{order.side.value}@{order.price} current={current_price}"
                )
                return True
            return False

    def simulate_fill(self, order_id: str, fill_size: float) -> bool:
        """Simula preenchimento para testes."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or not order.is_active:
                return False
            actual_fill = min(fill_size, order.remaining_size)
            if actual_fill <= 0:
                return False
            order.filled_size = round(order.filled_size + actual_fill, 10)
            if order.filled_size >= order.size:
                order.status = OrderStatus.FILLED
                logger.info(f"[ORDER:FILLED] {order.order_id} {order.size}@{order.price}")
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                logger.info(
                    f"[ORDER:PARTIALLY_FILLED] {order.order_id} "
                    f"{order.filled_size}/{order.size}@{order.price}"
                )
            return True

    def sync_order_status(
        self, order_id: str, api_status: str, filled_size: float
    ) -> None:
        """
        Sincroniza estado de ordem a partir da exchange API (usado pelo reconciler).

        Converge estado local para o estado da API. Se divergir, loga correção.
        """
        status_map = {
            "PENDING": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return
            old_status = order.status
            new_status = status_map.get(api_status, order.status)
            order.filled_size = filled_size
            if new_status != old_status:
                order.status = new_status
                logger.info(
                    f"[RECONCILE:STATE_CORRECTED] {order_id}: "
                    f"{old_status.value} -> {new_status.value}"
                )

    def get_active_orders(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                self.check_order_status(oid)
                for oid, o in self._orders.items()
                if o.is_active
            ]

    def get_all_orders(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {oid: self.check_order_status(oid) for oid in self._orders}
