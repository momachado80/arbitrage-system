"""
Edge Model — Mathematically Correct Edge Computation
========================================================

Fórmula correta de edge executável:

Para BUY:  net_edge = P_oracle - P_ask - fees - slippage
Para SELL: net_edge = P_bid - P_oracle - fees - slippage

Spread NÃO é subtraído separadamente — já está embutido
no preço de execução (ask para compra, bid para venda).

PROPRIEDADE FUNDAMENTAL:
    EV > 0 ↔ net_edge > 0

Sem ambiguidade.

Thread-safe via instância imutável (sem estado mutável).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EdgeConfig:
    """Custos de transação para cálculo de edge."""
    fee_rate: float = 0.02          # 2% fee por transação
    slippage_rate: float = 0.005    # 0.5% slippage estimado
    min_edge_threshold: float = 0.03  # 3% mínimo para sinal


@dataclass(frozen=True)
class EdgeResult:
    """Resultado do cálculo de edge."""
    side: Optional[str]       # "BUY", "SELL", ou None (sem edge)
    raw_edge: float           # P_oracle - P_market (midpoint)
    net_edge: float           # Edge líquido após custos
    execution_price: float    # Preço de execução (ask ou bid)
    p_oracle: float
    p_market: float           # Midpoint
    best_bid: float
    best_ask: float
    fees: float
    slippage: float
    total_cost: float
    expected_value: float     # EV por unidade apostada

    @property
    def is_actionable(self) -> bool:
        return self.side is not None and self.net_edge > 0


class EdgeModel:
    """
    Calcula edge executável com fórmula matematicamente correta.

    Regras:
    - BUY: compra no ask → net_edge = P_oracle - P_ask - fees - slippage
    - SELL: vende no bid → net_edge = P_bid - P_oracle - fees - slippage
    - Spread já embutido no preço de execução
    - EV > 0 ↔ net_edge > 0 (garantido pela fórmula)
    """

    def __init__(self, config: Optional[EdgeConfig] = None) -> None:
        self._config = config or EdgeConfig()

    @property
    def config(self) -> EdgeConfig:
        return self._config

    def compute_net_edge(
        self,
        p_oracle: float,
        best_bid: float,
        best_ask: float,
    ) -> EdgeResult:
        """
        Calcula net edge para um mercado.

        Args:
            p_oracle: probabilidade estimada pelo oracle
            best_bid: melhor preço de compra no book
            best_ask: melhor preço de venda no book

        Returns:
            EdgeResult com side, net_edge, EV e custos detalhados.
        """
        p_market = (best_bid + best_ask) / 2.0

        # Determinar direção com base no raw edge
        raw_edge = p_oracle - p_market

        fees = self._config.fee_rate
        slippage = self._config.slippage_rate
        total_cost = fees + slippage

        if raw_edge > 0:
            # BUY: execução no ask
            # Lucro esperado = P_oracle - P_ask
            net_edge = p_oracle - best_ask - total_cost
            execution_price = best_ask
            attempted_side = "BUY"
            side = "BUY" if net_edge >= self._config.min_edge_threshold else None
        elif raw_edge < 0:
            # SELL: execução no bid
            # Lucro esperado = P_bid - P_oracle
            net_edge = best_bid - p_oracle - total_cost
            execution_price = best_bid
            attempted_side = "SELL"
            side = "SELL" if net_edge >= self._config.min_edge_threshold else None
        else:
            net_edge = 0.0
            execution_price = p_market
            attempted_side = "BUY"
            side = None

        # EV always for the attempted direction (matches net_edge sign)
        ev = self.expected_value(p_oracle, execution_price, total_cost, attempted_side)

        return EdgeResult(
            side=side,
            raw_edge=raw_edge,
            net_edge=net_edge,
            execution_price=execution_price,
            p_oracle=p_oracle,
            p_market=p_market,
            best_bid=best_bid,
            best_ask=best_ask,
            fees=fees,
            slippage=slippage,
            total_cost=total_cost,
            expected_value=ev,
        )

    def expected_value(
        self,
        p_oracle: float,
        execution_price: float,
        total_cost: float,
        side: Optional[str],
    ) -> float:
        """
        Calcula Expected Value por unidade apostada.

        Para BUY (compra YES token no ask):
            EV = P_oracle * (1 - execution_price) - (1 - P_oracle) * execution_price - total_cost
            Simplifica: EV = P_oracle - execution_price - total_cost

        Para SELL (vende YES token no bid):
            EV = (1 - P_oracle) * execution_price - P_oracle * (1 - execution_price) - total_cost
            Simplifica: EV = execution_price - P_oracle - total_cost

        Garantia: EV > 0 ↔ net_edge > 0
        """
        if side == "BUY":
            return p_oracle - execution_price - total_cost
        elif side == "SELL":
            return execution_price - p_oracle - total_cost
        else:
            # Sem direção definida, calcular o melhor dos dois
            ev_buy = p_oracle - execution_price - total_cost
            ev_sell = execution_price - p_oracle - total_cost
            return max(ev_buy, ev_sell)

    def break_even_probability(
        self,
        execution_price: float,
        side: str,
    ) -> float:
        """
        Calcula P_oracle mínimo para break-even (EV = 0).

        Para BUY:  P_break = execution_price + total_cost
        Para SELL: P_break = execution_price - total_cost

        Retorna a probabilidade mínima do oracle para que o trade
        tenha EV >= 0.
        """
        total_cost = self._config.fee_rate + self._config.slippage_rate

        if side == "BUY":
            return execution_price + total_cost
        elif side == "SELL":
            return execution_price - total_cost
        else:
            return execution_price
