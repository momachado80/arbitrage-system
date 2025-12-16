"""
Execution Simulator
===================

Simula execução de ordens no livro de ordens.

PRINCÍPIO FUNDAMENTAL:
Nunca tome decisões baseadas apenas em top of book.
Sempre simule a execução caminhando pelo livro nível a nível.

Este módulo é crítico porque:
1. Calcula VWAP real para um tamanho Q
2. Determina se há liquidez suficiente
3. Conta quantos níveis seriam consumidos
4. Identifica quando uma ordem não pode ser completamente preenchida
"""

import logging
from decimal import Decimal
from typing import Optional

from ..models.core import (
    ExecutionSimulation,
    OrderBook,
    PriceLevel,
    Side,
)

logger = logging.getLogger(__name__)


class ExecutionSimulator:
    """
    Simula execução de ordens no livro.
    
    Este é o componente que transforma "preço exibido" em "preço executável".
    Toda decisão de arbitragem DEVE passar por aqui.
    """
    
    def simulate_buy(
        self,
        orderbook: OrderBook,
        size_usd: Decimal,
        include_partial: bool = True
    ) -> ExecutionSimulation:
        """
        Simula compra de contratos.
        
        Para comprar, consumimos os ASKS (ofertas de venda).
        
        Args:
            orderbook: Livro de ordens
            size_usd: Valor em USD a gastar (ex: $20)
            include_partial: Se True, aceita preenchimento parcial
        
        Returns:
            ExecutionSimulation com detalhes da execução simulada
        """
        return self._simulate_execution(
            levels=orderbook.asks,
            size_usd=size_usd,
            side=orderbook.side,
            direction="buy",
            include_partial=include_partial
        )
    
    def simulate_sell(
        self,
        orderbook: OrderBook,
        size_contracts: Decimal,
        include_partial: bool = True
    ) -> ExecutionSimulation:
        """
        Simula venda de contratos.
        
        Para vender, consumimos os BIDS (ofertas de compra).
        
        Args:
            orderbook: Livro de ordens
            size_contracts: Número de contratos a vender
            include_partial: Se True, aceita preenchimento parcial
        
        Returns:
            ExecutionSimulation com detalhes da execução simulada
        """
        return self._simulate_execution(
            levels=orderbook.bids,
            size_contracts=size_contracts,
            side=orderbook.side,
            direction="sell",
            include_partial=include_partial
        )
    
    def simulate_buy_contracts(
        self,
        orderbook: OrderBook,
        num_contracts: Decimal,
        include_partial: bool = True
    ) -> ExecutionSimulation:
        """
        Simula compra de um número específico de contratos.
        
        Diferente de simulate_buy que especifica valor em USD,
        este método especifica número de contratos desejados.
        
        Args:
            orderbook: Livro de ordens
            num_contracts: Número de contratos a comprar
            include_partial: Se True, aceita preenchimento parcial
        
        Returns:
            ExecutionSimulation com detalhes da execução simulada
        """
        return self._simulate_execution(
            levels=orderbook.asks,
            size_contracts=num_contracts,
            side=orderbook.side,
            direction="buy",
            include_partial=include_partial
        )
    
    def _simulate_execution(
        self,
        levels: list[PriceLevel],
        side: Side,
        direction: str,
        size_usd: Optional[Decimal] = None,
        size_contracts: Optional[Decimal] = None,
        include_partial: bool = True
    ) -> ExecutionSimulation:
        """
        Implementação central da simulação de execução.
        
        Caminha pelo livro nível a nível, calculando:
        - Quanto de cada nível seria consumido
        - VWAP resultante
        - Custo ou receita total
        - Se a ordem pode ser completamente preenchida
        
        Args:
            levels: Níveis de preço (asks para compra, bids para venda)
            side: Lado do mercado (YES ou NO)
            direction: "buy" ou "sell"
            size_usd: Valor em USD (para compras por valor)
            size_contracts: Número de contratos (para vendas ou compras por quantidade)
            include_partial: Se aceita preenchimento parcial
        
        Returns:
            ExecutionSimulation detalhada
        """
        if not levels:
            # Livro vazio - não consegue executar nada
            return ExecutionSimulation(
                side=side,
                direction=direction,
                requested_size=size_usd or size_contracts or Decimal("0"),
                filled_size=Decimal("0"),
                vwap=Decimal("0"),
                total_cost=Decimal("0"),
                levels_consumed=0,
                is_complete=False,
                unfilled_size=size_usd or size_contracts or Decimal("0")
            )
        
        filled_contracts = Decimal("0")
        total_cost = Decimal("0")  # Para compras: quanto gastou. Para vendas: quanto recebeu.
        levels_consumed = 0
        
        # Determinar target baseado no modo
        if size_usd is not None and direction == "buy":
            # Modo: gastar X dólares comprando
            budget_remaining = size_usd
            target_by_contracts = False
        elif size_contracts is not None:
            # Modo: comprar/vender N contratos
            contracts_remaining = size_contracts
            target_by_contracts = True
        else:
            raise ValueError("Deve especificar size_usd ou size_contracts")
        
        for level in levels:
            if target_by_contracts:
                if contracts_remaining <= 0:
                    break
                
                # Quanto deste nível podemos pegar?
                contracts_from_level = min(level.size, contracts_remaining)
                
                if direction == "buy":
                    # Comprando: pagamos price por contrato
                    cost_for_level = contracts_from_level * level.price
                else:
                    # Vendendo: recebemos price por contrato
                    cost_for_level = contracts_from_level * level.price
                
                filled_contracts += contracts_from_level
                total_cost += cost_for_level
                contracts_remaining -= contracts_from_level
                levels_consumed += 1
                
            else:
                # Modo budget (gastar X dólares)
                if budget_remaining <= 0:
                    break
                
                # A este preço, quantos contratos podemos comprar com o budget restante?
                max_contracts_by_budget = budget_remaining / level.price
                contracts_from_level = min(level.size, max_contracts_by_budget)
                
                cost_for_level = contracts_from_level * level.price
                
                filled_contracts += contracts_from_level
                total_cost += cost_for_level
                budget_remaining -= cost_for_level
                levels_consumed += 1
        
        # Calcular VWAP
        if filled_contracts > 0:
            vwap = total_cost / filled_contracts
        else:
            vwap = Decimal("0")
        
        # Determinar se preencheu tudo
        if target_by_contracts:
            requested = size_contracts
            unfilled = contracts_remaining if contracts_remaining > 0 else Decimal("0")
            is_complete = unfilled == 0
        else:
            requested = size_usd
            unfilled = budget_remaining if budget_remaining > 0 else Decimal("0")
            is_complete = unfilled < Decimal("0.01")  # Tolerância de 1 centavo
        
        return ExecutionSimulation(
            side=side,
            direction=direction,
            requested_size=requested,
            filled_size=filled_contracts,
            vwap=vwap,
            total_cost=total_cost,
            levels_consumed=levels_consumed,
            is_complete=is_complete,
            unfilled_size=unfilled
        )
    
    def calculate_cost_to_lock_payout(
        self,
        book_a: OrderBook,
        book_b: OrderBook,
        num_contracts: Decimal,
        strategy: str
    ) -> Optional[tuple[ExecutionSimulation, ExecutionSimulation, Decimal]]:
        """
        Calcula o custo para travar um payout garantido.
        
        Em arbitragem de mercados binários, travamos o payout comprando
        posições complementares. Estratégias possíveis:
        
        1. "buy_yes_a_buy_no_b": Compra YES em A, compra NO em B
           - Se evento acontece: YES paga $1, NO paga $0 → Recebe $1 de A
           - Se evento não acontece: YES paga $0, NO paga $1 → Recebe $1 de B
           - Payout garantido: $1 por par de contratos
        
        2. "buy_no_a_buy_yes_b": Compra NO em A, compra YES em B
           - Mesma lógica, invertida
        
        Args:
            book_a: Orderbook da plataforma A (para o lado especificado)
            book_b: Orderbook da plataforma B (para o lado oposto)
            num_contracts: Número de contratos a travar
            strategy: Estratégia de arbitragem
        
        Returns:
            Tupla (exec_a, exec_b, custo_total) ou None se impossível
        """
        # Determinar qual lado comprar em cada livro
        if strategy == "buy_yes_a_buy_no_b":
            # Compramos nos asks de cada livro
            exec_a = self.simulate_buy_contracts(book_a, num_contracts)
            exec_b = self.simulate_buy_contracts(book_b, num_contracts)
        elif strategy == "buy_no_a_buy_yes_b":
            exec_a = self.simulate_buy_contracts(book_a, num_contracts)
            exec_b = self.simulate_buy_contracts(book_b, num_contracts)
        else:
            logger.error(f"Estratégia desconhecida: {strategy}")
            return None
        
        # Verificar se ambas execuções são completas
        if not exec_a.is_complete or not exec_b.is_complete:
            logger.warning(
                f"Execução incompleta: A={exec_a.is_complete}, B={exec_b.is_complete}"
            )
            # Podemos retornar mesmo assim para análise
        
        total_cost = exec_a.total_cost + exec_b.total_cost
        
        return exec_a, exec_b, total_cost


def analyze_arbitrage_potential(
    kalshi_yes_book: OrderBook,
    kalshi_no_book: OrderBook,
    poly_yes_book: OrderBook,
    poly_no_book: OrderBook,
    target_contracts: Decimal,
    min_edge_percent: Decimal = Decimal("2.0")
) -> dict:
    """
    Analisa potencial de arbitragem entre dois mercados.
    
    Verifica as 4 combinações possíveis:
    1. Compra YES Kalshi + Compra NO Polymarket
    2. Compra NO Kalshi + Compra YES Polymarket
    3. Compra YES Polymarket + Compra NO Kalshi
    4. Compra NO Polymarket + Compra YES Kalshi
    
    Args:
        kalshi_yes_book: Orderbook YES da Kalshi
        kalshi_no_book: Orderbook NO da Kalshi
        poly_yes_book: Orderbook YES da Polymarket
        poly_no_book: Orderbook NO da Polymarket
        target_contracts: Número de contratos alvo
        min_edge_percent: Edge mínimo em % para considerar
    
    Returns:
        Dicionário com análise de cada combinação
    """
    simulator = ExecutionSimulator()
    results = {}
    
    # Payout garantido por par de contratos
    guaranteed_payout = target_contracts * Decimal("1.00")
    
    # Estratégia 1: YES Kalshi + NO Polymarket
    exec_k_yes = simulator.simulate_buy_contracts(kalshi_yes_book, target_contracts)
    exec_p_no = simulator.simulate_buy_contracts(poly_no_book, target_contracts)
    
    if exec_k_yes.is_complete and exec_p_no.is_complete:
        total_cost_1 = exec_k_yes.total_cost + exec_p_no.total_cost
        gross_profit_1 = guaranteed_payout - total_cost_1
        edge_percent_1 = (gross_profit_1 / total_cost_1 * 100) if total_cost_1 > 0 else Decimal("0")
        
        results["kalshi_yes_poly_no"] = {
            "kalshi_execution": exec_k_yes,
            "poly_execution": exec_p_no,
            "total_cost": total_cost_1,
            "guaranteed_payout": guaranteed_payout,
            "gross_profit": gross_profit_1,
            "edge_percent": edge_percent_1,
            "is_profitable": gross_profit_1 > 0,
            "meets_threshold": edge_percent_1 >= min_edge_percent
        }
    
    # Estratégia 2: NO Kalshi + YES Polymarket
    exec_k_no = simulator.simulate_buy_contracts(kalshi_no_book, target_contracts)
    exec_p_yes = simulator.simulate_buy_contracts(poly_yes_book, target_contracts)
    
    if exec_k_no.is_complete and exec_p_yes.is_complete:
        total_cost_2 = exec_k_no.total_cost + exec_p_yes.total_cost
        gross_profit_2 = guaranteed_payout - total_cost_2
        edge_percent_2 = (gross_profit_2 / total_cost_2 * 100) if total_cost_2 > 0 else Decimal("0")
        
        results["kalshi_no_poly_yes"] = {
            "kalshi_execution": exec_k_no,
            "poly_execution": exec_p_yes,
            "total_cost": total_cost_2,
            "guaranteed_payout": guaranteed_payout,
            "gross_profit": gross_profit_2,
            "edge_percent": edge_percent_2,
            "is_profitable": gross_profit_2 > 0,
            "meets_threshold": edge_percent_2 >= min_edge_percent
        }
    
    return results


# Exemplo de uso
if __name__ == "__main__":
    from decimal import Decimal
    
    # Criar orderbook de exemplo
    yes_book = OrderBook(
        side=Side.YES,
        bids=[
            PriceLevel(Decimal("0.55"), Decimal("100")),
            PriceLevel(Decimal("0.54"), Decimal("200")),
            PriceLevel(Decimal("0.53"), Decimal("500")),
        ],
        asks=[
            PriceLevel(Decimal("0.57"), Decimal("150")),
            PriceLevel(Decimal("0.58"), Decimal("300")),
            PriceLevel(Decimal("0.59"), Decimal("400")),
        ]
    )
    
    simulator = ExecutionSimulator()
    
    # Simular compra de $20
    result = simulator.simulate_buy(yes_book, Decimal("20"))
    print(f"\n=== Simulação de Compra $20 ===")
    print(f"Contratos preenchidos: {result.filled_size:.2f}")
    print(f"VWAP: ${result.vwap:.4f}")
    print(f"Custo total: ${result.total_cost:.2f}")
    print(f"Níveis consumidos: {result.levels_consumed}")
    print(f"Execução completa: {result.is_complete}")
    
    # Simular compra de 50 contratos
    result2 = simulator.simulate_buy_contracts(yes_book, Decimal("50"))
    print(f"\n=== Simulação de Compra 50 Contratos ===")
    print(f"Contratos preenchidos: {result2.filled_size:.2f}")
    print(f"VWAP: ${result2.vwap:.4f}")
    print(f"Custo total: ${result2.total_cost:.2f}")
    print(f"Níveis consumidos: {result2.levels_consumed}")
