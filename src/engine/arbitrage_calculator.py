"""
Arbitrage Calculator
====================

Calcula oportunidades de arbitragem líquidas de custos.

SEQUÊNCIA OBRIGATÓRIA:
1. Simular execução nos dois livros
2. Calcular custo total de entrada
3. Subtrair TODAS as taxas (Kalshi + Polymarket)
4. Aplicar penalidade de risco de base
5. Aplicar buffer de slippage
6. Só então verificar se lucro líquido > 0

Se lucro líquido <= 0, REJEITAR a oportunidade.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Union

from ..models.core import (
    ArbitrageOpportunity,
    ExecutionSimulation,
    FeeStructure,
    Market,
    OrderBook,
    Platform,
    Side,
    DEFAULT_BASIS_RISK_PENALTY,
    DEFAULT_SLIPPAGE_BUFFER,
    MIN_EDGE_THRESHOLD,
)
from .execution_simulator import ExecutionSimulator

logger = logging.getLogger(__name__)


@dataclass
class RejectionReason:
    """Razão detalhada para rejeição de uma oportunidade."""
    code: str
    message: str
    details: dict = None
    
    def __str__(self):
        return f"[{self.code}] {self.message}"


class ArbitrageCalculator:
    """
    Calcula e valida oportunidades de arbitragem.
    
    Este é o componente que decide se uma oportunidade é real ou não.
    Ele DEVE ser conservador e rejeitar a maioria das oportunidades aparentes.
    """
    
    def __init__(
        self,
        fee_structure: Optional[FeeStructure] = None,
        slippage_buffer: Decimal = DEFAULT_SLIPPAGE_BUFFER,
        basis_risk_penalty: Decimal = DEFAULT_BASIS_RISK_PENALTY,
        min_edge_threshold: Decimal = MIN_EDGE_THRESHOLD,
    ):
        """
        Inicializa o calculador.
        
        Args:
            fee_structure: Estrutura de taxas (usa default se None)
            slippage_buffer: Buffer adicional para slippage (ex: 0.005 = 0.5%)
            basis_risk_penalty: Penalidade por risco de base (ex: 0.01 = 1%)
            min_edge_threshold: Edge mínimo exigido em % (ex: 0.02 = 2%)
        """
        self.fee_structure = fee_structure or FeeStructure(Platform.KALSHI)
        self.slippage_buffer = slippage_buffer
        self.basis_risk_penalty = basis_risk_penalty
        self.min_edge_threshold = min_edge_threshold
        self.simulator = ExecutionSimulator()
    
    def calculate(
        self,
        kalshi_market: Market,
        polymarket_market: Market,
        target_usd: Decimal,
        equivalence_score: Decimal = Decimal("1.0"),
    ) -> tuple[Optional[ArbitrageOpportunity], Optional[RejectionReason]]:
        """
        Calcula oportunidade de arbitragem entre dois mercados.
        
        Args:
            kalshi_market: Mercado Kalshi com orderbooks
            polymarket_market: Mercado Polymarket com orderbooks
            target_usd: Valor alvo em USD para a operação
            equivalence_score: Score de equivalência contratual (0-1)
                              1.0 = equivalente confirmado
                              < 1.0 = penalidade adicional aplicada
        
        Returns:
            Tupla (ArbitrageOpportunity ou None, RejectionReason ou None)
        """
        # Validações básicas
        rejection = self._validate_markets(kalshi_market, polymarket_market)
        if rejection:
            return None, rejection
        
        # Calcular número de contratos baseado no target
        # Usamos o preço médio aproximado para estimar
        estimated_contracts = self._estimate_contracts(
            kalshi_market, polymarket_market, target_usd
        )
        
        if estimated_contracts <= 0:
            return None, RejectionReason(
                code="ZERO_CONTRACTS",
                message="Não foi possível estimar número de contratos"
            )
        
        # Testar as duas estratégias possíveis
        strategies = [
            ("kalshi_yes_poly_no", Side.YES, Side.NO),
            ("kalshi_no_poly_yes", Side.NO, Side.YES),
        ]
        
        best_opportunity = None
        best_edge = Decimal("-999")
        all_rejections = []
        
        for strategy_name, kalshi_side, poly_side in strategies:
            result = self._calculate_strategy(
                kalshi_market=kalshi_market,
                polymarket_market=polymarket_market,
                kalshi_side=kalshi_side,
                poly_side=poly_side,
                num_contracts=estimated_contracts,
                equivalence_score=equivalence_score,
                strategy_name=strategy_name,
            )
            
            if isinstance(result, RejectionReason):
                all_rejections.append((strategy_name, result))
                continue
            
            opportunity = result
            if opportunity.edge_percentage > best_edge:
                best_edge = opportunity.edge_percentage
                best_opportunity = opportunity
        
        if best_opportunity:
            return best_opportunity, None
        
        # Retornar a rejeição mais informativa
        if all_rejections:
            return None, all_rejections[0][1]
        
        return None, RejectionReason(
            code="NO_STRATEGY",
            message="Nenhuma estratégia viável encontrada"
        )
    
    def _validate_markets(
        self,
        kalshi_market: Market,
        polymarket_market: Market
    ) -> Optional[RejectionReason]:
        """Validações básicas dos mercados."""
        
        if not kalshi_market.has_orderbook:
            return RejectionReason(
                code="NO_KALSHI_BOOK",
                message="Mercado Kalshi sem orderbook"
            )
        
        if not polymarket_market.has_orderbook:
            return RejectionReason(
                code="NO_POLY_BOOK",
                message="Mercado Polymarket sem orderbook"
            )
        
        if not kalshi_market.is_active:
            return RejectionReason(
                code="KALSHI_INACTIVE",
                message=f"Mercado Kalshi não está ativo: {kalshi_market.status}"
            )
        
        if not polymarket_market.is_active:
            return RejectionReason(
                code="POLY_INACTIVE",
                message=f"Mercado Polymarket não está ativo: {polymarket_market.status}"
            )
        
        return None
    
    def _estimate_contracts(
        self,
        kalshi_market: Market,
        polymarket_market: Market,
        target_usd: Decimal
    ) -> Decimal:
        """
        Estima número de contratos para um valor alvo.
        
        Como arbitragem exige posições complementares (YES + NO = $1),
        o custo total aproximado é:
        custo ≈ (preço_yes_a + preço_no_b) * contratos
        
        Para arbitragem funcionar: preço_yes_a + preço_no_b < $1
        """
        # Pegar preços médios (midpoint ou best ask)
        k_yes = kalshi_market.yes_book
        p_no = polymarket_market.no_book
        
        if k_yes and k_yes.best_ask and p_no and p_no.best_ask:
            total_entry_price = k_yes.best_ask.price + p_no.best_ask.price
            if total_entry_price > 0:
                # Cada par de contratos custa aproximadamente total_entry_price
                return target_usd / total_entry_price
        
        # Fallback: assumir preço médio de 0.50 cada lado
        return target_usd / Decimal("1.0")
    
    def _calculate_strategy(
        self,
        kalshi_market: Market,
        polymarket_market: Market,
        kalshi_side: Side,
        poly_side: Side,
        num_contracts: Decimal,
        equivalence_score: Decimal,
        strategy_name: str,
    ) -> "Union[ArbitrageOpportunity, RejectionReason]":
        """
        Calcula uma estratégia específica de arbitragem.
        """
        # Obter orderbooks corretos
        kalshi_book = kalshi_market.get_book(kalshi_side)
        poly_book = polymarket_market.get_book(poly_side)
        
        if not kalshi_book or not kalshi_book.asks:
            return RejectionReason(
                code="NO_KALSHI_ASKS",
                message=f"Sem asks para {kalshi_side.value} na Kalshi"
            )
        
        if not poly_book or not poly_book.asks:
            return RejectionReason(
                code="NO_POLY_ASKS",
                message=f"Sem asks para {poly_side.value} na Polymarket"
            )
        
        # Simular execuções
        kalshi_exec = self.simulator.simulate_buy_contracts(kalshi_book, num_contracts)
        poly_exec = self.simulator.simulate_buy_contracts(poly_book, num_contracts)
        
        # Verificar se execuções são completas
        if not kalshi_exec.is_complete:
            return RejectionReason(
                code="KALSHI_INCOMPLETE",
                message=f"Liquidez insuficiente na Kalshi: "
                       f"pedido={num_contracts}, preenchido={kalshi_exec.filled_size}",
                details={"unfilled": float(kalshi_exec.unfilled_size)}
            )
        
        if not poly_exec.is_complete:
            return RejectionReason(
                code="POLY_INCOMPLETE",
                message=f"Liquidez insuficiente na Polymarket: "
                       f"pedido={num_contracts}, preenchido={poly_exec.filled_size}",
                details={"unfilled": float(poly_exec.unfilled_size)}
            )
        
        # Calcular custos
        total_cost = kalshi_exec.total_cost + poly_exec.total_cost
        
        # Payout garantido: $1 por contrato (mercados binários)
        guaranteed_payout = num_contracts * Decimal("1.00")
        
        # Lucro bruto (antes de fees)
        gross_profit = guaranteed_payout - total_cost
        
        # Se já não é lucrativo bruto, rejeitar
        if gross_profit <= 0:
            return RejectionReason(
                code="NEGATIVE_GROSS",
                message=f"Lucro bruto negativo: ${gross_profit:.4f}",
                details={
                    "total_cost": float(total_cost),
                    "guaranteed_payout": float(guaranteed_payout),
                    "kalshi_vwap": float(kalshi_exec.vwap),
                    "poly_vwap": float(poly_exec.vwap),
                }
            )
        
        # Calcular taxas
        kalshi_fees = self.fee_structure.calculate_kalshi_fee(
            contracts=num_contracts,
            price=kalshi_exec.vwap,
            is_maker=False  # Assumimos taker para ser conservador
        )
        
        poly_fees = self.fee_structure.calculate_polymarket_fee(
            contracts=num_contracts,
            price=poly_exec.vwap
        )
        
        total_fees = kalshi_fees + poly_fees
        
        # Penalidade por risco de base
        # Se equivalence_score < 1, aumentamos a penalidade
        adjusted_basis_penalty = self.basis_risk_penalty * (Decimal("2") - equivalence_score)
        basis_penalty_amount = total_cost * adjusted_basis_penalty
        
        # Buffer de slippage
        slippage_amount = total_cost * self.slippage_buffer
        
        # Lucro líquido final
        net_profit = gross_profit - total_fees - basis_penalty_amount - slippage_amount
        
        # Verificar edge mínimo
        edge_percentage = (net_profit / total_cost * 100) if total_cost > 0 else Decimal("0")
        
        if net_profit <= 0:
            return RejectionReason(
                code="NEGATIVE_NET",
                message=f"Lucro líquido negativo após custos: ${net_profit:.4f}",
                details={
                    "gross_profit": float(gross_profit),
                    "total_fees": float(total_fees),
                    "basis_penalty": float(basis_penalty_amount),
                    "slippage_buffer": float(slippage_amount),
                }
            )
        
        if edge_percentage < self.min_edge_threshold * 100:
            return RejectionReason(
                code="BELOW_THRESHOLD",
                message=f"Edge {edge_percentage:.2f}% abaixo do mínimo {self.min_edge_threshold * 100:.2f}%",
                details={
                    "edge_percentage": float(edge_percentage),
                    "min_threshold": float(self.min_edge_threshold * 100),
                }
            )
        
        # SUCESSO: Criar oportunidade de arbitragem
        return ArbitrageOpportunity(
            kalshi_market=kalshi_market,
            polymarket_market=polymarket_market,
            strategy=strategy_name,
            kalshi_execution=kalshi_exec,
            polymarket_execution=poly_exec,
            total_cost=total_cost,
            guaranteed_payout=guaranteed_payout,
            gross_profit=gross_profit,
            kalshi_fees=kalshi_fees,
            polymarket_fees=poly_fees,
            total_fees=total_fees,
            basis_risk_penalty=basis_penalty_amount,
            slippage_buffer=slippage_amount,
            net_profit=net_profit,
            detected_at=datetime.utcnow(),
            contracts=num_contracts,
        )
    
    def format_opportunity(self, opp: ArbitrageOpportunity) -> str:
        """Formata uma oportunidade para exibição."""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ARBITRAGE OPPORTUNITY                        ║
╠══════════════════════════════════════════════════════════════════╣
║ Strategy: {opp.strategy:<52} ║
║ Contracts: {opp.contracts:<51} ║
╠══════════════════════════════════════════════════════════════════╣
║ KALSHI: {opp.kalshi_market.ticker:<54} ║
║   VWAP: ${opp.kalshi_execution.vwap:.4f}                                      ║
║   Cost: ${opp.kalshi_execution.total_cost:.4f}                                      ║
╠══════════════════════════════════════════════════════════════════╣
║ POLYMARKET: {opp.polymarket_market.ticker:<50} ║
║   VWAP: ${opp.polymarket_execution.vwap:.4f}                                      ║
║   Cost: ${opp.polymarket_execution.total_cost:.4f}                                      ║
╠══════════════════════════════════════════════════════════════════╣
║ ECONOMICS:                                                       ║
║   Total Cost:       ${opp.total_cost:>10.4f}                               ║
║   Guaranteed Payout: ${opp.guaranteed_payout:>10.4f}                               ║
║   Gross Profit:     ${opp.gross_profit:>10.4f}                               ║
║   Kalshi Fees:      ${opp.kalshi_fees:>10.4f}                               ║
║   Poly Fees:        ${opp.polymarket_fees:>10.4f}                               ║
║   Risk Penalty:     ${opp.basis_risk_penalty:>10.4f}                               ║
║   Slippage Buffer:  ${opp.slippage_buffer:>10.4f}                               ║
║   ─────────────────────────────────────                          ║
║   NET PROFIT:       ${opp.net_profit:>10.4f}                               ║
║   EDGE:             {opp.edge_percentage:>10.2f}%                              ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    def format_rejection(self, reason: RejectionReason) -> str:
        """Formata uma rejeição para exibição."""
        details_str = ""
        if reason.details:
            details_str = "\n".join(f"    {k}: {v}" for k, v in reason.details.items())
        
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    OPPORTUNITY REJECTED                         ║
╠══════════════════════════════════════════════════════════════════╣
║ Code: {reason.code:<56} ║
║ Message: {reason.message[:54]:<54} ║
{f'║ Details:{chr(10)}{details_str}' if details_str else ''}
╚══════════════════════════════════════════════════════════════════╝
"""
