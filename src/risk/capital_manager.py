"""
Capital Manager
===============

Motor explícito de contabilidade de capital para o sistema de arbitragem.

DEFINIÇÕES:
- total_capital      = initial_capital + realized_pnl
- available_capital  = total_capital - locked_in_open_orders - locked_in_unresolved_markets
- locked_in_open_orders:       soma dos valores reservados para ordens ainda não totalmente preenchidas
- locked_in_unresolved_markets: soma dos valores de trades FILLED mas ainda não resolvidos
- unrealized_pnl:              placeholder (derivado de preço atual, calculado externamente)

REGRAS DE SEGURANÇA:
- available_capital nunca pode ficar negativo via reserve_for_order
- Todas as operações são idempotentes (não duplicam reservas nem pnl)
- Thread-safe via RLock

IDEMPOTÊNCIA:
- reserve_for_order:  ignora se trade_id já reservado
- release_order_reserve: ignora se trade_id não existe
- move_to_unresolved: ignora se trade_id já em unresolved
- resolve_trade:      ignora se trade_id já resolvido
"""

import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class InsufficientCapitalError(Exception):
    """Tentativa de reservar capital acima do disponível."""


class CapitalManager:
    """
    Motor de contabilidade de capital.

    Uso:
    ```python
    cm = CapitalManager(initial_capital=10000.0)
    cm.reserve_for_order("trade-1", 500.0)
    cm.move_to_unresolved("trade-1", 500.0)
    cm.resolve_trade("trade-1", pnl=-500.0)
    print(cm.snapshot())
    ```
    """

    def __init__(
        self,
        initial_capital: float,
        deadzone_resolution_buffer: float = 0.0,
    ):
        if initial_capital < 0:
            raise ValueError("initial_capital não pode ser negativo")
        self._initial_capital = initial_capital
        self._deadzone_buffer = deadzone_resolution_buffer
        self._realized_pnl: float = 0.0
        self._order_reserves: Dict[str, float] = {}
        self._unresolved_reserves: Dict[str, float] = {}
        self._resolved_trades: set = set()
        self._corrupted: bool = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Propriedades derivadas
    # ------------------------------------------------------------------

    @property
    def total_capital(self) -> float:
        """initial_capital + realized_pnl"""
        with self._lock:
            return self._initial_capital + self._realized_pnl

    @property
    def locked_in_open_orders(self) -> float:
        with self._lock:
            return sum(self._order_reserves.values())

    @property
    def locked_in_unresolved_markets(self) -> float:
        with self._lock:
            return sum(self._unresolved_reserves.values())

    @property
    def available_capital(self) -> float:
        with self._lock:
            return (
                self._initial_capital
                + self._realized_pnl
                - sum(self._order_reserves.values())
                - sum(self._unresolved_reserves.values())
            )

    # ------------------------------------------------------------------
    # Operações de capital
    # ------------------------------------------------------------------

    def reserve_for_order(self, trade_id: str, amount: float) -> None:
        """
        Reserva capital para uma nova ordem.

        Idempotente: se trade_id já reservado, ignora silenciosamente.
        Raises InsufficientCapitalError se amount > available_capital.
        """
        with self._lock:
            if trade_id in self._order_reserves:
                return  # Idempotente
            avail = (
                self._initial_capital
                + self._realized_pnl
                - sum(self._order_reserves.values())
                - sum(self._unresolved_reserves.values())
            )
            if amount > avail:
                raise InsufficientCapitalError(
                    f"Capital insuficiente: requerido={amount:.2f}, "
                    f"disponível={avail:.2f}"
                )
            self._order_reserves[trade_id] = amount
            logger.info(
                f"[CAPITAL:RESERVED] trade={trade_id} amount={amount:.2f} "
                f"available={avail - amount:.2f}"
            )

    def release_order_reserve(self, trade_id: str, amount: Optional[float] = None) -> None:
        """
        Libera reserva de ordem.

        Se amount é None, libera a reserva completa.
        Idempotente: ignora se trade_id não existe.
        """
        with self._lock:
            if trade_id not in self._order_reserves:
                return  # Idempotente
            if amount is None or amount >= self._order_reserves[trade_id]:
                released = self._order_reserves.pop(trade_id)
            else:
                released = amount
                self._order_reserves[trade_id] -= amount
                if self._order_reserves[trade_id] <= 0:
                    self._order_reserves.pop(trade_id, None)
            logger.info(
                f"[CAPITAL:RELEASED] trade={trade_id} amount={released:.2f}"
            )

    def move_to_unresolved(self, trade_id: str, amount: float) -> None:
        """
        Move capital de open_orders para unresolved_markets.

        Chamado quando trade é totalmente preenchido (FILLED).
        Idempotente: ignora se trade_id já está em unresolved.
        """
        with self._lock:
            if trade_id in self._unresolved_reserves:
                return  # Idempotente
            # Remove da reserva de ordens (se existir)
            self._order_reserves.pop(trade_id, None)
            self._unresolved_reserves[trade_id] = amount
            logger.info(
                f"[CAPITAL:MOVED_TO_UNRESOLVED] trade={trade_id} amount={amount:.2f}"
            )

    def resolve_trade(self, trade_id: str, pnl: float) -> None:
        """
        Resolve trade: libera capital bloqueado e registra PnL.

        Idempotente: ignora se trade_id já foi resolvido.
        """
        with self._lock:
            if trade_id in self._resolved_trades:
                return  # Idempotente
            self._realized_pnl += pnl
            self._unresolved_reserves.pop(trade_id, None)
            self._order_reserves.pop(trade_id, None)
            self._resolved_trades.add(trade_id)
            logger.info(
                f"[CAPITAL:RESOLVED] trade={trade_id} pnl={pnl:+.2f} "
                f"total_capital={self._initial_capital + self._realized_pnl:.2f}"
            )

    def process_sell_fill(self, cost_basis: float, pnl: float) -> None:
        """
        Processa SELL fill (spot-only): libera cost_basis de unresolved
        e registra PnL realizado.

        cost_basis = shares_vendidas * avg_entry_price (antes do sell)
        pnl = proceeds - cost_basis

        MVP: reduz locked_in_unresolved proporcionalmente entre trades.
        Nota: O PnL de resolução de mercado será aproximado se a posição
        foi parcialmente vendida antes da resolução.
        """
        with self._lock:
            self._realized_pnl += pnl
            # Liberar cost_basis de unresolved (distribuir entre trades)
            remaining = cost_basis
            for tid in list(self._unresolved_reserves.keys()):
                if remaining <= 1e-10:
                    break
                reduce = min(remaining, self._unresolved_reserves[tid])
                self._unresolved_reserves[tid] -= reduce
                remaining -= reduce
                if self._unresolved_reserves[tid] <= 1e-10:
                    del self._unresolved_reserves[tid]
            logger.info(
                f"[CAPITAL:SELL_FILL] cost_basis={cost_basis:.2f} pnl={pnl:+.2f}"
            )

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------

    def get_available_capital(self) -> float:
        """Retorna capital disponível para novas operações."""
        return self.available_capital

    def is_corrupted(self) -> bool:
        """Retorna True se capital manager marcado como corrompido."""
        return self._corrupted

    def mark_corrupted(self) -> None:
        """Marca como corrompido (usado pelo kill switch)."""
        self._corrupted = True

    def snapshot(self) -> Dict[str, Any]:
        """Retorna snapshot completo do estado de capital."""
        with self._lock:
            total = self._initial_capital + self._realized_pnl
            locked_orders = sum(self._order_reserves.values())
            locked_unresolved = sum(self._unresolved_reserves.values())
            avail = total - locked_orders - locked_unresolved
            return {
                "initial_capital": self._initial_capital,
                "total_capital": total,
                "realized_pnl": self._realized_pnl,
                "locked_in_open_orders": locked_orders,
                "locked_in_unresolved_markets": locked_unresolved,
                "available_capital": avail,
                "unrealized_pnl": 0.0,  # placeholder — derivado externamente
                "deadzone_resolution_buffer": self._deadzone_buffer,
                "open_order_count": len(self._order_reserves),
                "unresolved_count": len(self._unresolved_reserves),
                "resolved_count": len(self._resolved_trades),
            }
