"""
Risk Manager
============

Controle de risco pré-trade para o pipeline de execução.

LIMITES CONFIGURÁVEIS:
- max_exposure_total:          Exposição máxima total em todas as posições
- max_exposure_per_category:   Exposição máxima por categoria de mercado
- max_exposure_per_market:     Exposição máxima por market_id
- max_open_trades:             Número máximo de trades ativos
- total_capital:               Capital total disponível
- max_percent_capital_deadzone: % máximo do capital em posições abertas

Todos os limites são verificados ANTES de abrir um novo trade.
Se qualquer limite for excedido, o trade é rejeitado com reason explicativo.
"""

import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "max_exposure_total": 10000.0,
    "max_exposure_per_category": 5000.0,
    "max_exposure_per_market": 2000.0,
    "max_open_trades": 50,
    "total_capital": 10000.0,
    "max_percent_capital_deadzone": 0.5,
}


class RiskManager:
    """
    Controle de risco pré-trade.

    Uso:
    ```python
    rm = RiskManager({"max_exposure_total": 5000})
    allowed, reason = rm.can_open_trade(decision, trade_ledger)
    if not allowed:
        abort(reason)
    ```
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.max_exposure_total: float = cfg["max_exposure_total"]
        self.max_exposure_per_category: float = cfg["max_exposure_per_category"]
        self.max_exposure_per_market: float = cfg["max_exposure_per_market"]
        self.max_open_trades: int = cfg["max_open_trades"]
        self.total_capital: float = cfg["total_capital"]
        self.max_percent_capital_deadzone: float = cfg["max_percent_capital_deadzone"]

    def can_open_trade(
        self,
        decision: Dict[str, Any],
        trade_ledger: Any,
        capital_manager: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """
        Verifica se um novo trade pode ser aberto.

        decision deve conter: market_id, category, final_size.
        capital_manager: opcional, se fornecido usa contabilidade explícita.
        Retorna (allowed: bool, reason: str).
        """
        active_trades = trade_ledger.get_active_trades()
        new_size = decision.get("final_size", 0.0)
        market_id = decision.get("market_id", "")
        category = decision.get("category", "")
        # Normalizar category para string
        if hasattr(category, "value"):
            category = category.value

        # Filtrar: apenas trades BUY contribuem para exposição
        # Trades SELL reduzem exposição (spot-only)
        buy_trades = {
            tid: t for tid, t in active_trades.items()
            if t.get("action", "BUY") != "SELL"
        }

        # 1. Max open trades (conta todos os ativos)
        if len(active_trades) >= self.max_open_trades:
            reason = f"Max open trades ({self.max_open_trades}) atingido"
            logger.warning(f"[RISK:BLOCKED] {reason}")
            return (False, reason)

        # 2. Exposição total (somente BUY)
        total_exposure = sum(t["final_size"] for t in buy_trades.values())
        if total_exposure + new_size > self.max_exposure_total:
            reason = (
                f"Exposição total ({total_exposure + new_size:.2f}) "
                f"excede limite ({self.max_exposure_total:.2f})"
            )
            logger.warning(f"[RISK:BLOCKED] {reason}")
            return (False, reason)

        # 3. Exposição por categoria (somente BUY)
        cat_exposure = sum(
            t["final_size"] for t in buy_trades.values()
            if t.get("category") == category
        )
        if cat_exposure + new_size > self.max_exposure_per_category:
            reason = (
                f"Exposição categoria {category} ({cat_exposure + new_size:.2f}) "
                f"excede limite ({self.max_exposure_per_category:.2f})"
            )
            logger.warning(f"[RISK:BLOCKED] {reason}")
            return (False, reason)

        # 4. Exposição por mercado (somente BUY)
        mkt_exposure = sum(
            t["final_size"] for t in buy_trades.values()
            if t.get("market_id") == market_id
        )
        if mkt_exposure + new_size > self.max_exposure_per_market:
            reason = (
                f"Exposição mercado {market_id} ({mkt_exposure + new_size:.2f}) "
                f"excede limite ({self.max_exposure_per_market:.2f})"
            )
            logger.warning(f"[RISK:BLOCKED] {reason}")
            return (False, reason)

        # 5. Capital disponível via CapitalManager
        if capital_manager is not None:
            avail = capital_manager.get_available_capital()
            if new_size > avail:
                reason = (
                    f"Capital insuficiente (disponível={avail:.2f}, "
                    f"requerido={new_size:.2f})"
                )
                logger.warning(f"[RISK:BLOCKED] {reason}")
                return (False, reason)

        # 6. Capital deadzone
        if capital_manager is not None:
            snap = capital_manager.snapshot()
            tc = snap["total_capital"]
            if tc > 0:
                deadzone = snap["locked_in_unresolved_markets"] / tc
                if deadzone > self.max_percent_capital_deadzone:
                    reason = (
                        f"Capital deadzone ({deadzone:.1%}) "
                        f"excede limite ({self.max_percent_capital_deadzone:.1%})"
                    )
                    logger.warning(f"[RISK:BLOCKED] {reason}")
                    return (False, reason)
        else:
            if self.total_capital > 0:
                deadzone = (total_exposure + new_size) / self.total_capital
                if deadzone > self.max_percent_capital_deadzone:
                    reason = (
                        f"Capital deadzone ({deadzone:.1%}) "
                        f"excede limite ({self.max_percent_capital_deadzone:.1%})"
                    )
                    logger.warning(f"[RISK:BLOCKED] {reason}")
                    return (False, reason)

        return (True, "")
