"""
Core Data Models for Arbitrage System
=====================================

Estas são as entidades fundamentais do sistema, conforme definido na documentação:
- Market: representa um contrato específico em uma plataforma
- OrderBook: livro de ordens normalizado
- Side: YES/NO como instrumentos financeiros
- ExecutionSimulation: resultado de simulação de execução

PRINCÍPIO CENTRAL: Nunca usar preço exibido como preço executável.
Sempre simular execução no livro real.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class Platform(Enum):
    """Plataformas suportadas pelo sistema."""
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


class MarketStatus(Enum):
    """Status operacional de um mercado."""
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"
    HALTED = "halted"
    UNKNOWN = "unknown"


class Side(Enum):
    """
    Lado econômico de uma posição.
    
    YES e NO não são apenas strings. Eles representam:
    - Uma aposta binária com payout fixo de $1
    - Uma equivalência matemática: YES price X implica NO price (1 - X)
    - Um instrumento que pode ser replicado sinteticamente
    """
    YES = "yes"
    NO = "no"
    
    def opposite(self) -> "Side":
        """Retorna o lado oposto."""
        return Side.NO if self == Side.YES else Side.YES
    
    def implied_price(self, price: Decimal) -> Decimal:
        """
        Dado um preço para este lado, calcula o preço implícito do lado oposto.
        Em mercados binários: YES + NO = 1.00
        """
        return Decimal("1.00") - price


@dataclass
class PriceLevel:
    """
    Um nível de preço no livro de ordens.
    
    Em mercados binários:
    - price: preço em dólares (0.01 a 0.99)
    - size: quantidade de contratos disponíveis
    """
    price: Decimal
    size: Decimal
    
    def __post_init__(self):
        """Validação básica."""
        if not (Decimal("0") <= self.price <= Decimal("1")):
            raise ValueError(f"Preço deve estar entre 0 e 1, recebido: {self.price}")
        if self.size < 0:
            raise ValueError(f"Size não pode ser negativo: {self.size}")


@dataclass
class OrderBook:
    """
    Livro de ordens normalizado para uma posição específica.
    
    IMPORTANTE: Este é o livro para UM LADO (YES ou NO) de um mercado.
    
    Bids: ofertas de compra (ordenadas do maior para menor preço)
    Asks: ofertas de venda (ordenadas do menor para maior preço)
    
    Em mercados binários como Kalshi:
    - Um bid de YES a X é equivalente a um ask de NO a (1-X)
    - O sistema deve sempre fazer essa conversão internamente
    """
    side: Side
    bids: list[PriceLevel] = field(default_factory=list)
    asks: list[PriceLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def best_bid(self) -> Optional[PriceLevel]:
        """Melhor preço de compra (maior bid)."""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[PriceLevel]:
        """Melhor preço de venda (menor ask)."""
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Spread entre melhor ask e melhor bid."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None
    
    @property
    def midpoint(self) -> Optional[Decimal]:
        """Ponto médio entre bid e ask."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None
    
    def total_bid_depth(self) -> Decimal:
        """Profundidade total de bids em contratos."""
        return sum(level.size for level in self.bids)
    
    def total_ask_depth(self) -> Decimal:
        """Profundidade total de asks em contratos."""
        return sum(level.size for level in self.asks)


@dataclass
class Market:
    """
    Representa um contrato específico em uma plataforma.
    
    Um mercado não é apenas um ticker. Ele contém toda a informação
    necessária para avaliar equivalência contratual.
    """
    # Identificação
    platform: Platform
    ticker: str  # Identificador único na plataforma
    
    # Conteúdo do contrato
    title: str
    description: str = ""
    resolution_rules: str = ""  # Regras completas de resolução
    resolution_source: str = ""  # Fonte oficial de verdade
    
    # Temporalidade
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    expiration_time: Optional[datetime] = None
    
    # Status
    status: MarketStatus = MarketStatus.UNKNOWN
    
    # Dados de mercado (atualizados em tempo real)
    yes_book: Optional[OrderBook] = None
    no_book: Optional[OrderBook] = None
    last_updated: Optional[datetime] = None
    
    # Metadados específicos da plataforma
    raw_data: dict = field(default_factory=dict)
    
    # Polymarket específico
    condition_id: Optional[str] = None  # Para Polymarket
    yes_token_id: Optional[str] = None
    no_token_id: Optional[str] = None
    
    @property
    def is_active(self) -> bool:
        """Verifica se o mercado está ativo para trading."""
        return self.status == MarketStatus.OPEN
    
    @property
    def has_orderbook(self) -> bool:
        """Verifica se temos dados de orderbook."""
        return self.yes_book is not None or self.no_book is not None
    
    def get_book(self, side: Side) -> Optional[OrderBook]:
        """Retorna o orderbook para um lado específico."""
        return self.yes_book if side == Side.YES else self.no_book


@dataclass
class ExecutionSimulation:
    """
    Resultado de uma simulação de execução no livro.
    
    NUNCA tome decisões baseadas apenas em top of book.
    Esta estrutura representa o custo real de executar uma ordem de tamanho Q.
    """
    side: Side
    direction: str  # "buy" ou "sell"
    requested_size: Decimal
    filled_size: Decimal
    vwap: Decimal  # Volume Weighted Average Price
    total_cost: Decimal
    levels_consumed: int
    is_complete: bool  # True se conseguiu preencher todo o tamanho
    unfilled_size: Decimal = Decimal("0")
    
    @property
    def slippage(self) -> Optional[Decimal]:
        """Slippage em relação ao melhor preço disponível."""
        # Será calculado pelo simulador que tem acesso ao best price
        return None


@dataclass
class FeeStructure:
    """
    Estrutura de taxas para uma plataforma.
    
    Kalshi: fee = 0.07 × C × P × (1-P) para taker
    Polymarket: ~0.01% flat (muito baixo)
    
    Estas taxas DEVEM ser subtraídas de qualquer cálculo de lucro.
    """
    platform: Platform
    
    # Kalshi
    kalshi_taker_coefficient: Decimal = Decimal("0.07")
    kalshi_maker_coefficient: Decimal = Decimal("0.0175")
    
    # Polymarket (flat fee por contrato)
    polymarket_fee_per_contract: Decimal = Decimal("0.0001")  # 0.01%
    
    def calculate_kalshi_fee(
        self,
        contracts: Decimal,
        price: Decimal,
        is_maker: bool = False
    ) -> Decimal:
        """
        Calcula taxa Kalshi.
        Formula: coefficient × C × P × (1-P)
        """
        coef = self.kalshi_maker_coefficient if is_maker else self.kalshi_taker_coefficient
        return coef * contracts * price * (Decimal("1") - price)
    
    def calculate_polymarket_fee(
        self,
        contracts: Decimal,
        price: Decimal
    ) -> Decimal:
        """
        Calcula taxa Polymarket.
        Atualmente ~0.01% do valor.
        """
        return contracts * price * self.polymarket_fee_per_contract


@dataclass
class ArbitrageOpportunity:
    """
    Uma oportunidade de arbitragem detectada.
    
    IMPORTANTE: Esta estrutura só deve ser criada após:
    1. Validação de equivalência contratual
    2. Simulação de execução nos dois livros
    3. Cálculo de custos completos
    4. Aplicação de penalidade de risco
    
    Se qualquer condição falhar, NÃO criar esta estrutura.
    """
    # Mercados envolvidos
    kalshi_market: Market
    polymarket_market: Market
    
    # Direção da arbitragem
    # Ex: "buy_kalshi_yes_sell_poly_yes" ou "buy_kalshi_yes_buy_poly_no"
    strategy: str
    
    # Execução simulada
    kalshi_execution: ExecutionSimulation
    polymarket_execution: ExecutionSimulation
    
    # Cálculo econômico
    total_cost: Decimal  # Custo total de entrada
    guaranteed_payout: Decimal  # Payout garantido ($1 por contrato)
    gross_profit: Decimal  # Payout - Cost
    
    # Taxas
    kalshi_fees: Decimal
    polymarket_fees: Decimal
    total_fees: Decimal
    
    # Ajustes de risco
    basis_risk_penalty: Decimal  # Penalidade por risco de base
    slippage_buffer: Decimal  # Buffer de segurança
    
    # Resultado final
    net_profit: Decimal  # Lucro líquido após tudo
    
    # Metadados
    detected_at: datetime = field(default_factory=datetime.utcnow)
    contracts: Decimal = Decimal("0")
    
    @property
    def is_profitable(self) -> bool:
        """Só é arbitragem se lucro líquido > 0."""
        return self.net_profit > Decimal("0")
    
    @property
    def edge_percentage(self) -> Decimal:
        """Edge como percentual do capital investido."""
        if self.total_cost > 0:
            return (self.net_profit / self.total_cost) * 100
        return Decimal("0")


# Constantes do sistema
DEFAULT_MIN_SIZE_USD = Decimal("20")  # Tamanho mínimo de operação
DEFAULT_SLIPPAGE_BUFFER = Decimal("0.005")  # 0.5% buffer de slippage
DEFAULT_BASIS_RISK_PENALTY = Decimal("0.01")  # 1% penalidade base por risco de base
MIN_EDGE_THRESHOLD = Decimal("0.02")  # Edge mínimo de 2% para considerar
