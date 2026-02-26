"""
Realistic Simulator — Execução contra book real
================================================

Simula execução contra book Polymarket em tempo real.
- Nunca executa no mid
- Consome níveis reais de liquidez
- Aplica slippage proporcional
- Latência simulada 100-300ms
"""

import logging
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.execution.order_manager import OrderSide

logger = logging.getLogger(__name__)

LATENCY_MS_MIN = 100
LATENCY_MS_MAX = 300
STRICT_REALISM = os.environ.get("STRICT_REALISM", "").strip().upper() in ("1", "TRUE", "YES")


class SimulationStats:
    """Contadores para dashboard (equity, trades, slippage, rejeições)."""

    MIN_TRADES_FOR_EXPECTANCY = 30
    MIN_TRADES_FOR_SHARPE = 30
    MIN_DAYS_FOR_SHARPE = 2

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_trades = 0
        self.rejected_orders = 0
        self.slippage_sum = 0.0
        self.slippage_count = 0
        self._first_trade_ts: Optional[float] = None

    def record_trade(self, slippage: float) -> None:
        import time
        with self._lock:
            self.total_trades += 1
            self.slippage_sum += slippage
            self.slippage_count += 1
            if self._first_trade_ts is None:
                self._first_trade_ts = time.time()

    def record_rejection(self) -> None:
        with self._lock:
            self.rejected_orders += 1

    def get_metrics(self) -> Dict[str, Any]:
        import time
        with self._lock:
            rej = self.rejected_orders
            tot = self.total_trades
            total_attempts = tot + rej
            avg_slip = self.slippage_sum / self.slippage_count if self.slippage_count > 0 else 0.0
            days_simulated = 0.0
            if self._first_trade_ts:
                days_simulated = (time.time() - self._first_trade_ts) / 86400.0
            expectancy = None
            if tot >= self.MIN_TRADES_FOR_EXPECTANCY:
                expectancy = 0.0
            sharpe = None
            if tot >= self.MIN_TRADES_FOR_SHARPE and days_simulated >= self.MIN_DAYS_FOR_SHARPE:
                sharpe = 0.0
            return {
                "total_trades": tot,
                "rejected_orders": rej,
                "rejection_rate": rej / total_attempts if total_attempts > 0 else 0.0,
                "avg_slippage": avg_slip,
                "expectancy": expectancy,
                "sharpe": sharpe,
            }


@dataclass
class SimulatedFill:
    """Fill simulado. Campos com default devem vir após os obrigatórios."""
    order_id: str
    market_id: str
    side: str
    executed_size: float
    execution_price: float
    mid_at_signal: float
    spread_at_signal: float
    liquidity_available: float
    slippage_applied: float
    latency_ms: float
    timestamp: str
    order_size: float = 0.0  # Único opcional; deve ficar no final


def _book_to_slippage_format(book: Dict) -> Dict[str, List[Tuple[float, float]]]:
    """Garante formato esperado por estimate_fill_cost."""
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if isinstance(bids, dict):
        bids = sorted(bids.items(), key=lambda x: -x[0])
    if isinstance(asks, dict):
        asks = sorted(asks.items(), key=lambda x: x[0])
    return {"bids": list(bids), "asks": list(asks)}


def _execute_against_book(
    book: Dict,
    side: str,
    size: float,
    mid: float,
    spread: float,
    strict: bool,
) -> Tuple[float, float, float, bool]:
    """
    Executa contra book. Retorna (executed_size, avg_price, slippage, full_fill).
    Nunca executa no mid. Permite partial fill quando strict=False.
    """
    ob = _book_to_slippage_format(book)
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    if not bids or not asks:
        return 0.0, 0.0, 0.0, False
    levels = asks if side == "BUY" else bids
    total_filled = 0.0
    total_cost = 0.0
    for price, level_size in levels:
        can_fill = min(level_size, size - total_filled)
        total_cost += can_fill * price
        total_filled += can_fill
        if total_filled >= size:
            break
    if total_filled <= 0:
        if strict:
            return 0.0, 0.0, 0.0, False
        return 0.0, 0.0, 0.0, False
    avg_price = total_cost / total_filled
    slippage = abs(avg_price - mid)
    full_fill = total_filled >= size - 1e-10
    if strict and not full_fill:
        return 0.0, 0.0, 0.0, False
    return total_filled, avg_price, slippage, full_fill


def _compute_mid_spread(book: Dict) -> Tuple[float, float]:
    """Mid e spread a partir do book."""
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not bids or not asks:
        return 0.0, 0.0
    if isinstance(bids[0], (list, tuple)):
        best_bid = bids[0][0] if bids else 0.0
    else:
        best_bid = max(bids.keys()) if hasattr(bids, "keys") else 0.0
    if isinstance(asks[0], (list, tuple)):
        best_ask = asks[0][0] if asks else 0.0
    else:
        best_ask = min(asks.keys()) if hasattr(asks, "keys") else 0.0
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    return mid, spread


def _liquidity_available(book: Dict, side: str, levels: int = 5) -> float:
    """Soma liquidez nos primeiros níveis."""
    ob = _book_to_slippage_format(book)
    levels_list = ob["asks"] if side == "BUY" else ob["bids"]
    return sum(s for (_, s) in levels_list[:levels])


class RealisticSimulationOrderManager:
    """
    OrderManager para REALISTIC_SIMULATION.
    Executa contra book real, aplica latência, nunca no mid.
    """

    def __init__(
        self,
        book_provider: Callable[[str], Optional[Dict]],
        audit_logger: Optional[Any] = None,
        strict_realism: bool = False,
        stats: Optional["SimulationStats"] = None,
    ) -> None:
        self._book_provider = book_provider
        self._audit = audit_logger
        self._strict = strict_realism or STRICT_REALISM
        self._stats = stats or SimulationStats()
        self._fills: Dict[str, SimulatedFill] = {}
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def place_limit_order(
        self,
        market_id: str,
        side: OrderSide,
        price: float,
        size: float,
        ttl_seconds: int = 60,
        outcome_side: Optional[Any] = None,
        position_manager: Optional[Any] = None,
        capital_manager: Optional[Any] = None,
    ) -> List[str]:
        """
        Simula execução contra book real.
        1. Obtém book atual
        2. Aplica latência 100-300ms
        3. Obtém book novamente (pós-latência)
        4. Executa contra book
        """
        order_id = f"rsim_{uuid.uuid4().hex[:12]}"
        side_str = side.value if isinstance(side, OrderSide) else str(side)

        book_at_signal = self._book_provider(market_id)
        if book_at_signal is None or (not book_at_signal.get("bids") and not book_at_signal.get("asks")):
            self._stats.record_rejection()
            if self._strict:
                raise ValueError(f"[REALISTIC_SIM] Book indisponível para {market_id}")
            logger.warning("[SIM] [REJECTED] market=%s reason=book_empty", market_id)
            return []

        mid_at_signal, spread_at_signal = _compute_mid_spread(book_at_signal)
        liq = _liquidity_available(book_at_signal, side_str)

        latency_ms = random.uniform(LATENCY_MS_MIN, LATENCY_MS_MAX)
        time.sleep(latency_ms / 1000.0)

        book_after = self._book_provider(market_id)
        if book_after is None and self._strict:
            raise ValueError(f"[REALISTIC_SIM] Book indisponível após latência para {market_id}")
        book_to_use = book_after if book_after else book_at_signal

        executed_size, exec_price, slippage, ok = _execute_against_book(
            book_to_use, side_str, size, mid_at_signal, spread_at_signal, self._strict
        )
        if not ok and executed_size <= 0:
            self._stats.record_rejection()
            logger.warning("[SIM] [REJECTED] market=%s side=%s size=%s reason=insufficient_liquidity", market_id, side_str, size)
            return []

        fill = SimulatedFill(
            order_id=order_id,
            market_id=market_id,
            side=side_str,
            executed_size=executed_size,
            execution_price=exec_price,
            mid_at_signal=mid_at_signal,
            spread_at_signal=spread_at_signal,
            liquidity_available=liq,
            slippage_applied=slippage,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            order_size=size,
        )  # order_size último (único com default)
        with self._lock:
            self._fills[order_id] = fill
            self._orders[order_id] = {"status": "FILLED", "filled_size": executed_size, "price": exec_price}
        self._stats.record_trade(slippage)
        return [order_id]

    def cancel_order(self, order_id: str) -> bool:
        return False

    def cancel_all_buy_open_orders_best_effort(self) -> List[str]:
        return []

    def sync_order_status(
        self, order_id: str, status: str, filled_size: float
    ) -> None:
        pass

    def check_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            o = self._orders.get(order_id)
            if o:
                return {
                    "order_id": order_id,
                    "status": o.get("status", "FILLED"),
                    "filled_size": o.get("filled_size", 0.0),
                    "size": o.get("filled_size", 0.0),
                    "price": o.get("price", 0.0),
                }
        return None

    def get_fill(self, order_id: str) -> Optional[SimulatedFill]:
        with self._lock:
            return self._fills.get(order_id)

    def finalize_trade_audit(
        self,
        order_ids: List[str],
        capital_manager: Optional[Any],
    ) -> None:
        """
        Auditoria atômica: chamar APÓS capital_manager atualizado.
        Ordem: capital atualizado → snapshot → audit log.
        """
        if not self._audit:
            return
        for oid in order_ids:
            fill = self.get_fill(oid)
            if fill is None:
                continue
            snap = capital_manager.snapshot() if capital_manager else {}
            cap = snap.get("total_capital", 0.0)
            pnl = snap.get("realized_pnl", 0.0)
            partial = fill.order_size > 0 and fill.executed_size < fill.order_size * 0.9999
            self._audit.log_trade({
                "timestamp": fill.timestamp,
                "market_id": fill.market_id,
                "side": fill.side,
                "order_size": fill.order_size,
                "executed_size": fill.executed_size,
                "execution_price": fill.execution_price,
                "mid_price_at_signal": fill.mid_at_signal,
                "spread_at_signal": fill.spread_at_signal,
                "liquidity_available": fill.liquidity_available,
                "slippage_applied": fill.slippage_applied,
                "latency_applied_ms": fill.latency_ms,
                "pnl_after_trade": pnl,
                "capital_after_trade": cap,
                "partial_fill": partial,
            })


class RealisticSimulationExchangeAPI:
    """ExchangeAPI que retorna fills simulados do RealisticSimulationOrderManager."""

    def __init__(self, order_manager: RealisticSimulationOrderManager) -> None:
        self._om = order_manager

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        st = self._om.check_order_status(order_id)
        if st:
            return st
        return {"order_id": order_id, "status": "PENDING", "filled_size": 0.0}

    def get_fills(self, order_id: str) -> List[Dict[str, Any]]:
        fill = self._om.get_fill(order_id)
        if fill is None:
            return []
        return [{
            "fill_id": f"rsim_fill_{order_id}",
            "order_id": order_id,
            "price": fill.execution_price,
            "size": fill.executed_size,
            "timestamp": fill.timestamp,
        }]

    def get_position(self, market_id: str) -> Dict[str, Any]:
        return {"market_id": market_id, "size": 0.0, "avg_entry": 0.0}

    def get_account_summary(self) -> Optional[Dict[str, Any]]:
        return None
