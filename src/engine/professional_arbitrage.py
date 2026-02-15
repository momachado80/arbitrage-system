"""
Motor de Arbitragem Profissional
Implementa todas as verificações necessárias para arbitragem real:
- Custos completos com taxas
- Liquidez e profundidade do order book
- Slippage e risco de execução
- Sincronização e idade dos dados
- Equivalência rigorosa de mercados
- Simulação de payoff
- Gestão de tamanho e exposição
"""
import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any
import json

from .arbitrage_config import ArbitrageConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Nível do order book com preço e volume."""
    price: Decimal
    size: Decimal  # Em contratos
    
    @property
    def value_usd(self) -> Decimal:
        """Valor em USD neste nível."""
        return self.price * self.size


@dataclass
class OrderBook:
    """Order book completo com bids e asks."""
    bids: List[OrderBookLevel] = field(default_factory=list)  # Compradores (preço decrescente)
    asks: List[OrderBookLevel] = field(default_factory=list)  # Vendedores (preço crescente)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def best_bid(self) -> Optional[Decimal]:
        """Melhor preço de compra."""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[Decimal]:
        """Melhor preço de venda."""
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Preço médio (mid)."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Spread bid-ask."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_percentage(self) -> Optional[Decimal]:
        """Spread como percentual do mid."""
        if self.mid_price and self.spread:
            return (self.spread / self.mid_price) * 100
        return None
    
    @property
    def age_seconds(self) -> float:
        """Idade dos dados em segundos."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()
    
    def total_bid_volume(self) -> Decimal:
        """Volume total de bids."""
        return sum(level.size for level in self.bids)
    
    def total_ask_volume(self) -> Decimal:
        """Volume total de asks."""
        return sum(level.size for level in self.asks)


@dataclass
class ExecutionEstimate:
    """Estimativa de execução para um tamanho específico."""
    target_size_usd: Decimal
    avg_price: Decimal           # Preço médio de execução (VWAP)
    total_cost: Decimal          # Custo total incluindo taxas
    slippage_percentage: Decimal # Slippage vs melhor preço
    levels_used: int             # Quantos níveis do book foram usados
    fully_fillable: bool         # Se consegue preencher todo o tamanho
    actual_size: Decimal         # Tamanho realmente executável


@dataclass 
class MarketData:
    """Dados completos de um mercado."""
    platform: str              # "polymarket" ou "kalshi"
    market_id: str
    title: str
    description: str = ""
    
    # Preços
    yes_price: Decimal = Decimal("0.5")
    no_price: Decimal = Decimal("0.5")
    
    # Order books
    yes_book: Optional[OrderBook] = None
    no_book: Optional[OrderBook] = None
    
    # Metadata
    volume_24h: Decimal = Decimal("0")
    liquidity: Decimal = Decimal("0")
    expiration: Optional[datetime] = None
    resolution_source: str = ""
    resolution_rules: str = ""
    
    # URLs
    url: str = ""
    
    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def age_seconds(self) -> float:
        """Idade dos dados em segundos."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


@dataclass
class PayoffScenario:
    """Cenário de payoff para uma arbitragem."""
    scenario_name: str
    probability: Decimal          # Probabilidade estimada do cenário
    gross_return: Decimal         # Retorno bruto
    total_fees: Decimal           # Taxas totais
    net_return: Decimal           # Retorno líquido
    roi_percentage: Decimal       # ROI percentual


@dataclass
class ArbitrageOpportunity:
    """Oportunidade de arbitragem completa e validada."""
    
    # Identificação
    id: str
    poly_market: MarketData
    kalshi_market: MarketData
    
    # Matching
    title_similarity: float
    rules_match: bool
    expiration_match: bool
    equivalence_score: float  # 0-1, quanto mais alto melhor
    
    # Estratégia
    strategy: str             # Descrição da estratégia
    action_poly: str          # "BUY YES" ou "BUY NO"
    action_kalshi: str        # "BUY YES" ou "BUY NO"
    
    # Preços e custos
    poly_entry_price: Decimal
    kalshi_entry_price: Decimal
    total_cost_per_contract: Decimal
    
    # Taxas detalhadas
    poly_fees: Decimal
    kalshi_fees: Decimal
    total_fees: Decimal
    
    # Lucro
    gross_profit: Decimal
    net_profit: Decimal
    net_profit_percentage: Decimal
    
    # Execução por tamanho
    execution_100: Optional[ExecutionEstimate] = None
    execution_500: Optional[ExecutionEstimate] = None
    execution_1000: Optional[ExecutionEstimate] = None
    
    # Liquidez
    min_available_size: Decimal = Decimal("0")
    max_recommended_size: Decimal = Decimal("0")
    liquidity_score: str = "LOW"  # LOW, MEDIUM, HIGH
    
    # Timing
    data_age_poly: float = 0.0
    data_age_kalshi: float = 0.0
    max_data_age: float = 0.0
    data_freshness: str = "FRESH"  # FRESH, WARNING, STALE
    
    # Payoff
    payoff_scenarios: List[PayoffScenario] = field(default_factory=list)
    guaranteed_min_profit: Decimal = Decimal("0")
    max_loss_if_partial: Decimal = Decimal("0")
    
    # Resolução
    resolution_date: Optional[str] = None
    days_to_resolution: Optional[int] = None
    
    # Validação
    is_valid: bool = True
    validation_warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    
    # Score final
    overall_score: float = 0.0  # 0-100


class ProfessionalArbitrageEngine:
    """Motor de arbitragem profissional."""
    
    def __init__(self, config: ArbitrageConfig = None):
        self.config = config or DEFAULT_CONFIG
        
    def parse_order_book(self, raw_book: Dict[str, Any], platform: str) -> OrderBook:
        """
        Parseia order book cru para estrutura padronizada.
        """
        bids = []
        asks = []
        
        if platform == "polymarket":
            # Polymarket CLOB format
            for bid in raw_book.get("bids", []):
                bids.append(OrderBookLevel(
                    price=Decimal(str(bid.get("price", "0"))),
                    size=Decimal(str(bid.get("size", "0")))
                ))
            for ask in raw_book.get("asks", []):
                asks.append(OrderBookLevel(
                    price=Decimal(str(ask.get("price", "0"))),
                    size=Decimal(str(ask.get("size", "0")))
                ))
        
        elif platform == "kalshi":
            # Kalshi format
            for bid in raw_book.get("yes_bids", raw_book.get("bids", [])):
                price = bid.get("price", 0)
                if isinstance(price, int):
                    price = price / 100  # Centavos para dólares
                bids.append(OrderBookLevel(
                    price=Decimal(str(price)),
                    size=Decimal(str(bid.get("quantity", bid.get("size", "0"))))
                ))
            for ask in raw_book.get("yes_asks", raw_book.get("asks", [])):
                price = ask.get("price", 0)
                if isinstance(price, int):
                    price = price / 100
                asks.append(OrderBookLevel(
                    price=Decimal(str(price)),
                    size=Decimal(str(ask.get("quantity", ask.get("size", "0"))))
                ))
        
        # Ordenar: bids decrescente, asks crescente
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        
        return OrderBook(bids=bids, asks=asks)
    
    def calculate_execution(
        self, 
        book: OrderBook, 
        target_size_usd: Decimal,
        is_buy: bool,
        platform: str
    ) -> ExecutionEstimate:
        """
        Calcula execução real considerando profundidade do book.
        Retorna preço médio (VWAP), custo total e slippage.
        """
        levels = book.asks if is_buy else book.bids
        
        if not levels:
            return ExecutionEstimate(
                target_size_usd=target_size_usd,
                avg_price=Decimal("0"),
                total_cost=Decimal("0"),
                slippage_percentage=Decimal("100"),
                levels_used=0,
                fully_fillable=False,
                actual_size=Decimal("0")
            )
        
        best_price = levels[0].price
        remaining_size = target_size_usd
        total_value = Decimal("0")
        total_contracts = Decimal("0")
        levels_used = 0
        
        for level in levels:
            if remaining_size <= 0:
                break
                
            # Valor disponível neste nível (preço * quantidade)
            level_value = level.price * level.size
            
            if level_value >= remaining_size:
                # Preenche parcialmente este nível
                contracts_to_fill = remaining_size / level.price
                total_contracts += contracts_to_fill
                total_value += remaining_size
                remaining_size = Decimal("0")
            else:
                # Usa todo este nível
                total_contracts += level.size
                total_value += level_value
                remaining_size -= level_value
            
            levels_used += 1
        
        fully_fillable = remaining_size == 0
        actual_size = target_size_usd - remaining_size
        
        if total_contracts > 0:
            avg_price = total_value / total_contracts
            slippage = ((avg_price - best_price) / best_price * 100) if is_buy else ((best_price - avg_price) / best_price * 100)
        else:
            avg_price = Decimal("0")
            slippage = Decimal("100")
        
        # Calcular taxas
        if platform == "polymarket":
            fees = actual_size * self.config.fees.poly_trading_fee
        else:  # kalshi
            fees = actual_size * self.config.fees.kalshi_taker_fee
        
        return ExecutionEstimate(
            target_size_usd=target_size_usd,
            avg_price=avg_price,
            total_cost=actual_size + fees,
            slippage_percentage=abs(slippage),
            levels_used=levels_used,
            fully_fillable=fully_fillable,
            actual_size=actual_size
        )
    
    def check_market_equivalence(
        self,
        poly_market: MarketData,
        kalshi_market: MarketData
    ) -> Tuple[bool, float, List[str]]:
        """
        Verifica se dois mercados são realmente equivalentes.
        Retorna: (é_equivalente, score, warnings)
        """
        warnings = []
        score = 1.0
        
        # 1. Similaridade de título
        poly_title = self._normalize_title(poly_market.title)
        kalshi_title = self._normalize_title(kalshi_market.title)
        
        title_sim = SequenceMatcher(None, poly_title, kalshi_title).ratio()
        
        if title_sim < self.config.matching.min_title_similarity:
            return False, title_sim, [f"Similaridade de título muito baixa: {title_sim:.1%}"]
        
        # 2. Extrair entidades (pessoas, lugares)
        poly_entities = self._extract_entities(poly_market.title)
        kalshi_entities = self._extract_entities(kalshi_market.title)
        
        # Se ambos têm entidades, devem ter pelo menos uma em comum
        if poly_entities and kalshi_entities:
            common = set(poly_entities) & set(kalshi_entities)
            if not common:
                return False, 0.0, [f"Entidades diferentes: {poly_entities} vs {kalshi_entities}"]
        
        # 3. Verificar data de expiração
        if self.config.matching.require_same_expiration:
            if poly_market.expiration and kalshi_market.expiration:
                diff_hours = abs((poly_market.expiration - kalshi_market.expiration).total_seconds() / 3600)
                if diff_hours > self.config.matching.max_expiration_diff_hours:
                    warnings.append(f"Datas de expiração diferentes: {diff_hours:.0f}h de diferença")
                    score *= 0.7
        
        # 4. Verificar termos desqualificantes
        combined_text = f"{poly_market.title} {poly_market.description} {kalshi_market.title} {kalshi_market.description}".lower()
        
        for term in self.config.matching.disqualifying_terms:
            if term in combined_text:
                # Verificar se o termo aparece de forma diferente nos dois mercados
                poly_has = term in f"{poly_market.title} {poly_market.description}".lower()
                kalshi_has = term in f"{kalshi_market.title} {kalshi_market.description}".lower()
                
                if poly_has != kalshi_has:
                    warnings.append(f"Termo '{term}' presente em apenas um mercado")
                    score *= 0.8
        
        # 5. Verificar regras de resolução se disponíveis
        if poly_market.resolution_rules and kalshi_market.resolution_rules:
            rules_sim = SequenceMatcher(
                None, 
                poly_market.resolution_rules.lower(), 
                kalshi_market.resolution_rules.lower()
            ).ratio()
            
            if rules_sim < 0.7:
                warnings.append(f"Regras de resolução diferentes: {rules_sim:.1%} similaridade")
                score *= 0.5
        
        # Score final ajustado pela similaridade de título
        final_score = score * title_sim
        
        return True, final_score, warnings
    
    def calculate_payoff_scenarios(
        self,
        poly_market: MarketData,
        kalshi_market: MarketData,
        poly_action: str,  # "YES" ou "NO"
        kalshi_action: str,
        investment_usd: Decimal
    ) -> List[PayoffScenario]:
        """
        Calcula cenários de payoff detalhados.
        """
        scenarios = []
        
        # Determinar preços de entrada
        if poly_action == "YES":
            poly_price = poly_market.yes_price
        else:
            poly_price = poly_market.no_price
            
        if kalshi_action == "YES":
            kalshi_price = kalshi_market.yes_price
        else:
            kalshi_price = kalshi_market.no_price
        
        # Investimento dividido entre as duas plataformas
        poly_invest = investment_usd / 2
        kalshi_invest = investment_usd / 2
        
        poly_contracts = poly_invest / poly_price if poly_price > 0 else Decimal("0")
        kalshi_contracts = kalshi_invest / kalshi_price if kalshi_price > 0 else Decimal("0")
        
        # Taxas de entrada
        poly_entry_fee = poly_invest * self.config.fees.poly_trading_fee
        kalshi_entry_fee = kalshi_invest * self.config.fees.kalshi_taker_fee
        total_entry_fees = poly_entry_fee + kalshi_entry_fee
        
        # Cenário 1: Evento acontece (YES = $1)
        if poly_action == "YES":
            poly_return = poly_contracts * Decimal("1")  # Ganha
            poly_profit = poly_return - poly_invest
            poly_exit_fee = poly_profit * self.config.fees.poly_trading_fee if poly_profit > 0 else Decimal("0")
        else:
            poly_return = Decimal("0")  # Perde
            poly_profit = -poly_invest
            poly_exit_fee = Decimal("0")
        
        if kalshi_action == "YES":
            kalshi_return = kalshi_contracts * Decimal("1")
            kalshi_profit = kalshi_return - kalshi_invest
            kalshi_exit_fee = kalshi_profit * self.config.fees.kalshi_trading_fee if kalshi_profit > 0 else Decimal("0")
        else:
            kalshi_return = Decimal("0")
            kalshi_profit = -kalshi_invest
            kalshi_exit_fee = Decimal("0")
        
        gross_return_yes = poly_return + kalshi_return
        total_fees_yes = total_entry_fees + poly_exit_fee + kalshi_exit_fee
        net_return_yes = gross_return_yes - investment_usd - total_fees_yes
        
        scenarios.append(PayoffScenario(
            scenario_name="Evento ACONTECE (YES vence)",
            probability=poly_market.yes_price,  # Usar probabilidade implícita
            gross_return=gross_return_yes,
            total_fees=total_fees_yes,
            net_return=net_return_yes,
            roi_percentage=(net_return_yes / investment_usd * 100) if investment_usd > 0 else Decimal("0")
        ))
        
        # Cenário 2: Evento NÃO acontece (NO = $1)
        if poly_action == "NO":
            poly_return = poly_contracts * Decimal("1")
            poly_profit = poly_return - poly_invest
            poly_exit_fee = poly_profit * self.config.fees.poly_trading_fee if poly_profit > 0 else Decimal("0")
        else:
            poly_return = Decimal("0")
            poly_profit = -poly_invest
            poly_exit_fee = Decimal("0")
        
        if kalshi_action == "NO":
            kalshi_return = kalshi_contracts * Decimal("1")
            kalshi_profit = kalshi_return - kalshi_invest
            kalshi_exit_fee = kalshi_profit * self.config.fees.kalshi_trading_fee if kalshi_profit > 0 else Decimal("0")
        else:
            kalshi_return = Decimal("0")
            kalshi_profit = -kalshi_invest
            kalshi_exit_fee = Decimal("0")
        
        gross_return_no = poly_return + kalshi_return
        total_fees_no = total_entry_fees + poly_exit_fee + kalshi_exit_fee
        net_return_no = gross_return_no - investment_usd - total_fees_no
        
        scenarios.append(PayoffScenario(
            scenario_name="Evento NÃO ACONTECE (NO vence)",
            probability=Decimal("1") - poly_market.yes_price,
            gross_return=gross_return_no,
            total_fees=total_fees_no,
            net_return=net_return_no,
            roi_percentage=(net_return_no / investment_usd * 100) if investment_usd > 0 else Decimal("0")
        ))
        
        # Cenário 3: Execução parcial (pior caso)
        # Assume que só consegue entrar em uma plataforma
        worst_case_loss = -max(poly_invest, kalshi_invest) - total_entry_fees
        
        scenarios.append(PayoffScenario(
            scenario_name="FALHA PARCIAL (só 1 perna executada)",
            probability=Decimal("0.05"),  # 5% chance estimada
            gross_return=Decimal("0"),
            total_fees=total_entry_fees,
            net_return=worst_case_loss,
            roi_percentage=(worst_case_loss / investment_usd * 100) if investment_usd > 0 else Decimal("0")
        ))
        
        return scenarios
    
    def analyze_opportunity(
        self,
        poly_raw: Dict[str, Any],
        kalshi_raw: Dict[str, Any],
        poly_book_yes: Optional[Dict] = None,
        poly_book_no: Optional[Dict] = None,
        kalshi_book_yes: Optional[Dict] = None,
        kalshi_book_no: Optional[Dict] = None
    ) -> Optional[ArbitrageOpportunity]:
        """
        Analisa uma potencial oportunidade de arbitragem de forma completa.
        """
        try:
            # 1. Parsear dados de mercado
            poly_market = self._parse_polymarket(poly_raw, poly_book_yes, poly_book_no)
            kalshi_market = self._parse_kalshi(kalshi_raw, kalshi_book_yes, kalshi_book_no)
            
            # 2. Verificar equivalência
            is_equiv, equiv_score, equiv_warnings = self.check_market_equivalence(
                poly_market, kalshi_market
            )
            
            if not is_equiv:
                return None
            
            # 3. Determinar melhor estratégia
            # Opção A: YES Poly + NO Kalshi
            cost_a = poly_market.yes_price + kalshi_market.no_price
            # Opção B: NO Poly + YES Kalshi
            cost_b = poly_market.no_price + kalshi_market.yes_price
            
            # LOG dos preços para debug
            logger.info(f"   Preços POLY: YES=${poly_market.yes_price:.3f}, NO=${poly_market.no_price:.3f}")
            logger.info(f"   Preços KALSHI: YES=${kalshi_market.yes_price:.3f}, NO=${kalshi_market.no_price:.3f}")
            logger.info(f"   Cost A (YES Poly + NO Kalshi): ${cost_a:.4f}")
            logger.info(f"   Cost B (NO Poly + YES Kalshi): ${cost_b:.4f}")
            
            if cost_a < cost_b and cost_a < Decimal("1"):
                action_poly = "YES"
                action_kalshi = "NO"
                entry_cost = cost_a
                poly_entry = poly_market.yes_price
                kalshi_entry = kalshi_market.no_price
                logger.info(f"   ✅ ARBITRAGEM: Opção A, lucro bruto ${Decimal('1') - cost_a:.4f}")
            elif cost_b < Decimal("1"):
                action_poly = "NO"
                action_kalshi = "YES"
                entry_cost = cost_b
                poly_entry = poly_market.no_price
                kalshi_entry = kalshi_market.yes_price
                logger.info(f"   ✅ ARBITRAGEM: Opção B, lucro bruto ${Decimal('1') - cost_b:.4f}")
            else:
                # Sem arbitragem (custo >= $1)
                logger.info(f"   ❌ SEM ARBITRAGEM: Custos >= $1.00")
                return None
            
            # 4. Calcular taxas
            poly_fees = poly_entry * self.config.fees.poly_trading_fee
            kalshi_fees = kalshi_entry * self.config.fees.kalshi_taker_fee
            total_fees = poly_fees + kalshi_fees
            
            # 5. Calcular lucro líquido
            gross_profit = Decimal("1") - entry_cost
            net_profit = gross_profit - total_fees
            net_profit_pct = (net_profit / entry_cost * 100) if entry_cost > 0 else Decimal("0")
            
            # 6. Verificar colchão mínimo
            logger.info(f"   Lucro bruto: ${gross_profit:.4f} | Taxas: ${total_fees:.4f} | Líquido: ${net_profit:.4f} ({net_profit_pct:.2f}%)")
            
            if net_profit <= 0:
                logger.info(f"   ❌ TAXAS > LUCRO BRUTO: Operação daria prejuízo!")
                return None
            
            if net_profit_pct < self.config.profit.min_profit_percentage:
                logger.info(f"   ❌ Lucro {net_profit_pct:.2f}% < mínimo {self.config.profit.min_profit_percentage}%")
                return None
            if net_profit < self.config.profit.min_profit_absolute / 100:  # Por contrato
                logger.info(f"   ❌ Lucro ${net_profit:.4f} < mínimo ${self.config.profit.min_profit_absolute/100:.4f}")
                return None
            
            logger.info(f"   ✅ OPORTUNIDADE VÁLIDA: +{net_profit_pct:.2f}% líquido")
            
            # 7. Calcular execuções para diferentes tamanhos
            exec_100 = None
            exec_500 = None
            exec_1000 = None
            
            poly_book = poly_market.yes_book if action_poly == "YES" else poly_market.no_book
            kalshi_book = kalshi_market.yes_book if action_kalshi == "YES" else kalshi_market.no_book
            
            if poly_book and kalshi_book:
                for size in [100, 500, 1000]:
                    poly_exec = self.calculate_execution(poly_book, Decimal(str(size/2)), True, "polymarket")
                    kalshi_exec = self.calculate_execution(kalshi_book, Decimal(str(size/2)), True, "kalshi")
                    
                    combined = ExecutionEstimate(
                        target_size_usd=Decimal(str(size)),
                        avg_price=(poly_exec.avg_price + kalshi_exec.avg_price) / 2,
                        total_cost=poly_exec.total_cost + kalshi_exec.total_cost,
                        slippage_percentage=max(poly_exec.slippage_percentage, kalshi_exec.slippage_percentage),
                        levels_used=poly_exec.levels_used + kalshi_exec.levels_used,
                        fully_fillable=poly_exec.fully_fillable and kalshi_exec.fully_fillable,
                        actual_size=min(poly_exec.actual_size, kalshi_exec.actual_size) * 2
                    )
                    
                    if size == 100:
                        exec_100 = combined
                    elif size == 500:
                        exec_500 = combined
                    else:
                        exec_1000 = combined
            
            # 8. Determinar liquidez e tamanho máximo
            min_volume = min(
                poly_market.volume_24h or Decimal("1000"),
                kalshi_market.volume_24h or Decimal("1000")
            )
            
            max_size = min(
                self.config.risk.max_size_per_trade,
                min_volume * Decimal("0.1")  # Max 10% do volume diário
            )
            
            liquidity_score = "HIGH" if min_volume > 10000 else ("MEDIUM" if min_volume > 1000 else "LOW")
            
            # 9. Verificar idade dos dados
            max_age = max(poly_market.age_seconds, kalshi_market.age_seconds)
            if max_age > self.config.timing.stale_age_seconds:
                data_freshness = "STALE"
            elif max_age > self.config.timing.warning_age_seconds:
                data_freshness = "WARNING"
            else:
                data_freshness = "FRESH"
            
            # 10. Calcular payoff
            payoffs = self.calculate_payoff_scenarios(
                poly_market, kalshi_market,
                action_poly, action_kalshi,
                Decimal("1000")
            )
            
            guaranteed_min = min(s.net_return for s in payoffs if s.probability > Decimal("0.01"))
            max_loss = min(s.net_return for s in payoffs)
            
            # 11. Calcular score final
            score = self._calculate_score(
                net_profit_pct=float(net_profit_pct),
                equiv_score=equiv_score,
                liquidity_score=liquidity_score,
                data_freshness=data_freshness,
                days_to_resolution=self._days_to_resolution(poly_market, kalshi_market)
            )
            
            # 12. Construir estratégia legível
            strategy = f"COMPRAR {action_poly} no Polymarket (${poly_entry:.3f}) + {action_kalshi} no Kalshi (${kalshi_entry:.3f})"
            
            # 13. Data de resolução
            resolution_date = None
            days_to_res = None
            if poly_market.expiration:
                resolution_date = poly_market.expiration.strftime("%Y-%m-%d")
                days_to_res = (poly_market.expiration - datetime.now(timezone.utc)).days
            
            return ArbitrageOpportunity(
                id=f"{poly_market.market_id}_{kalshi_market.market_id}",
                poly_market=poly_market,
                kalshi_market=kalshi_market,
                title_similarity=SequenceMatcher(
                    None, 
                    self._normalize_title(poly_market.title),
                    self._normalize_title(kalshi_market.title)
                ).ratio(),
                rules_match=len([w for w in equiv_warnings if "regras" in w.lower()]) == 0,
                expiration_match=len([w for w in equiv_warnings if "expiração" in w.lower()]) == 0,
                equivalence_score=equiv_score,
                strategy=strategy,
                action_poly=f"BUY {action_poly}",
                action_kalshi=f"BUY {action_kalshi}",
                poly_entry_price=poly_entry,
                kalshi_entry_price=kalshi_entry,
                total_cost_per_contract=entry_cost,
                poly_fees=poly_fees,
                kalshi_fees=kalshi_fees,
                total_fees=total_fees,
                gross_profit=gross_profit,
                net_profit=net_profit,
                net_profit_percentage=net_profit_pct,
                execution_100=exec_100,
                execution_500=exec_500,
                execution_1000=exec_1000,
                min_available_size=max_size,
                max_recommended_size=max_size,
                liquidity_score=liquidity_score,
                data_age_poly=poly_market.age_seconds,
                data_age_kalshi=kalshi_market.age_seconds,
                max_data_age=max_age,
                data_freshness=data_freshness,
                payoff_scenarios=payoffs,
                guaranteed_min_profit=guaranteed_min,
                max_loss_if_partial=max_loss,
                resolution_date=resolution_date,
                days_to_resolution=days_to_res,
                validation_warnings=equiv_warnings,
                overall_score=score
            )
            
        except Exception as e:
            logger.error(f"Erro ao analisar oportunidade: {e}")
            return None
    
    def _parse_polymarket(
        self, 
        raw: Dict[str, Any],
        book_yes: Optional[Dict] = None,
        book_no: Optional[Dict] = None
    ) -> MarketData:
        """Parseia dados do Polymarket."""
        
        # Preços
        prices = raw.get("outcomePrices", ["0.5", "0.5"])
        if isinstance(prices, str):
            prices = json.loads(prices)
        
        yes_price = Decimal(str(prices[0] if prices else "0.5"))
        no_price = Decimal("1") - yes_price
        
        # Expiração
        expiration = None
        exp_str = raw.get("endDate", raw.get("end_date_iso"))
        if exp_str:
            try:
                expiration = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
            except:
                pass
        
        # Order books
        yes_book = self.parse_order_book(book_yes or {}, "polymarket") if book_yes else None
        no_book = self.parse_order_book(book_no or {}, "polymarket") if book_no else None
        
        return MarketData(
            platform="polymarket",
            market_id=raw.get("conditionId", raw.get("id", "")),
            title=raw.get("question", ""),
            description=raw.get("description", ""),
            yes_price=yes_price,
            no_price=no_price,
            yes_book=yes_book,
            no_book=no_book,
            volume_24h=Decimal(str(raw.get("volume", raw.get("volumeNum", 0)) or 0)),
            liquidity=Decimal(str(raw.get("liquidity", 0) or 0)),
            expiration=expiration,
            resolution_source=raw.get("resolutionSource", ""),
            resolution_rules=raw.get("description", ""),
            url=f"https://polymarket.com/market/{raw.get('slug', '')}"
        )
    
    def _parse_kalshi(
        self,
        raw: Dict[str, Any],
        book_yes: Optional[Dict] = None,
        book_no: Optional[Dict] = None
    ) -> MarketData:
        """Parseia dados do Kalshi."""
        
        # Preços (em centavos)
        yes_bid = raw.get("yes_bid", 50) or 50
        yes_ask = raw.get("yes_ask", 50) or 50
        
        if isinstance(yes_bid, int):
            yes_bid = yes_bid / 100
        if isinstance(yes_ask, int):
            yes_ask = yes_ask / 100
        
        yes_price = Decimal(str((yes_bid + yes_ask) / 2))
        no_price = Decimal("1") - yes_price
        
        # Expiração
        expiration = None
        exp_str = raw.get("close_time", raw.get("expiration_time"))
        if exp_str:
            try:
                expiration = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
            except:
                pass
        
        # Order books
        yes_book = self.parse_order_book(book_yes or {}, "kalshi") if book_yes else None
        no_book = self.parse_order_book(book_no or {}, "kalshi") if book_no else None
        
        ticker = raw.get("ticker", "")
        
        return MarketData(
            platform="kalshi",
            market_id=ticker,
            title=raw.get("title", raw.get("subtitle", "")),
            description=raw.get("rules_secondary", raw.get("rules_primary", "")),
            yes_price=yes_price,
            no_price=no_price,
            yes_book=yes_book,
            no_book=no_book,
            volume_24h=Decimal(str(raw.get("volume", raw.get("volume_24h", 0)) or 0)),
            liquidity=Decimal(str(raw.get("open_interest", 0) or 0)),
            expiration=expiration,
            resolution_source=raw.get("settlement_source", ""),
            resolution_rules=raw.get("rules_primary", ""),
            url=f"https://kalshi.com/markets/{ticker}"
        )
    
    def _normalize_title(self, title: str) -> str:
        """Normaliza título para comparação."""
        title = title.lower().strip()
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title)
        
        # Remover palavras comuns
        stop_words = {'will', 'the', 'be', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'by'}
        words = [w for w in title.split() if w not in stop_words]
        
        return ' '.join(words)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extrai entidades (pessoas, lugares) do texto."""
        entities = []
        
        # Padrões para nomes próprios (palavras capitalizadas em sequência)
        words = text.split()
        current_entity = []
        
        for word in words:
            # Remove pontuação
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                current_entity.append(clean_word)
            else:
                if current_entity:
                    entities.append(' '.join(current_entity).lower())
                    current_entity = []
        
        if current_entity:
            entities.append(' '.join(current_entity).lower())
        
        # Filtrar entidades muito curtas ou comuns
        filtered = [e for e in entities if len(e) > 3 and e not in {'will', 'the', 'next', 'this'}]
        
        return filtered
    
    def _days_to_resolution(self, poly: MarketData, kalshi: MarketData) -> Optional[int]:
        """Calcula dias até resolução."""
        exp = poly.expiration or kalshi.expiration
        if exp:
            return (exp - datetime.now(timezone.utc)).days
        return None
    
    def _calculate_score(
        self,
        net_profit_pct: float,
        equiv_score: float,
        liquidity_score: str,
        data_freshness: str,
        days_to_resolution: Optional[int]
    ) -> float:
        """Calcula score final da oportunidade (0-100)."""
        
        # Base: lucro (40 pontos max)
        profit_score = min(net_profit_pct * 4, 40)
        
        # Equivalência (20 pontos max)
        equiv_points = equiv_score * 20
        
        # Liquidez (15 pontos max)
        liquidity_points = {"HIGH": 15, "MEDIUM": 10, "LOW": 5}.get(liquidity_score, 5)
        
        # Freshness (15 pontos max)
        freshness_points = {"FRESH": 15, "WARNING": 10, "STALE": 0}.get(data_freshness, 0)
        
        # Tempo até resolução (10 pontos max) - mais próximo = melhor
        if days_to_resolution is not None:
            if days_to_resolution <= 7:
                time_points = 10
            elif days_to_resolution <= 30:
                time_points = 7
            elif days_to_resolution <= 90:
                time_points = 4
            else:
                time_points = 2
        else:
            time_points = 5
        
        return profit_score + equiv_points + liquidity_points + freshness_points + time_points

