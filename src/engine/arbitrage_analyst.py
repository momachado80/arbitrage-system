"""
Analisador de Arbitragem e Previsão de Mercados
================================================

Analisa mercados do Polymarket em tempo real para identificar oportunidades
de arbitragem baseadas em divergências de probabilidade.

Critérios de Filtragem:
- Volume > $100.000
- Timeframe < 48 horas ou mercados diários
- Divergência significativa de probabilidade
- Expected Value (EV) positivo

Estratégia:
- Identifica mercados onde o preço no Polymarket diverge de odds esperadas
- Calcula retornos projetados para diferentes tamanhos de investimento
- Apresenta Top 3 oportunidades com maior EV positivo
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict
from dataclasses import dataclass

from ..collectors.polymarket import PolymarketCollector
from ..models.core import Market, OrderBook, Side, Platform, FeeStructure
from .execution_simulator import ExecutionSimulator

logger = logging.getLogger(__name__)


@dataclass
class PolymarketOpportunity:
    """Oportunidade de arbitragem identificada no Polymarket."""
    market: Market
    market_title: str
    condition_id: str
    
    # Preços e probabilidades
    polymarket_price: Decimal  # Preço atual no Polymarket
    implied_probability: Decimal  # Probabilidade implícita do preço
    expected_probability: Decimal  # Probabilidade esperada (estimada)
    divergence: Decimal  # Diferença entre esperado e atual
    
    # Dados de mercado
    volume_24h: Decimal
    liquidity_depth: Decimal
    time_to_resolution: Optional[timedelta]
    
    # Análise de valor
    expected_value: Decimal  # EV positivo
    edge_percentage: Decimal  # Edge em %
    
    # Estratégia recomendada
    recommended_side: Side  # YES ou NO
    recommended_investment: Decimal  # Investimento recomendado
    
    # Retornos projetados
    returns_10: Decimal
    returns_20: Decimal
    returns_100: Decimal
    returns_1000: Decimal
    
    # Metadados
    detected_at: datetime
    confidence_score: Decimal  # 0-1, baseado em volume, liquidez, etc.


class ArbitrageAnalyst:
    """
    Analisador especializado em identificar oportunidades de arbitragem
    no Polymarket baseado em divergências de probabilidade.
    """
    
    # Critérios RELAXADOS para encontrar mais oportunidades
    MIN_VOLUME_USD = Decimal("1000")  # $1.000 mínimo (era $100k)
    MAX_TIMEFRAME_HOURS = 720  # 30 dias (era 48h)
    MIN_DIVERGENCE = Decimal("0.03")  # Mínimo 3% de divergência (era 10%)
    MIN_EV = Decimal("0.01")  # EV mínimo de $0.01 por $1 (era $0.05)
    
    def __init__(self, timeout: float = 30.0):
        """
        Inicializa o analisador.
        
        Args:
            timeout: Timeout para requests HTTP em segundos
        """
        self.timeout = timeout
        self.collector: Optional[PolymarketCollector] = None
        self.simulator = ExecutionSimulator()
        self.fee_structure = FeeStructure(Platform.POLYMARKET)
    
    async def __aenter__(self):
        """Context manager entry."""
        self.collector = PolymarketCollector(timeout=self.timeout)
        await self.collector.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.collector:
            await self.collector.__aexit__(exc_type, exc_val, exc_tb)
    
    async def scan_opportunities(self, limit: int = 200) -> List[PolymarketOpportunity]:
        """
        Escaneia mercados do Polymarket e identifica oportunidades.
        
        Args:
            limit: Número máximo de mercados a analisar
            
        Returns:
            Lista de oportunidades ordenadas por EV (maior primeiro)
        """
        if not self.collector:
            raise RuntimeError("Analyst deve ser usado como context manager")
        
        logger.info(f"Iniciando scan de {limit} mercados...")
        
        # Buscar mercados ordenados por volume com timeout
        try:
            markets = await asyncio.wait_for(
                self.collector.get_markets(
                    active=True,
                    limit=limit,
                    offset=0
                ),
                timeout=60.0  # 1 minuto para buscar mercados
            )
        except asyncio.TimeoutError:
            logger.error("Timeout ao buscar mercados do Polymarket")
            return []
        except Exception as e:
            logger.error(f"Erro ao buscar mercados: {e}")
            return []
        
        opportunities: List[PolymarketOpportunity] = []
        
        for market in markets:
            try:
                # Filtrar por volume
                volume_24h = self._extract_volume(market)
                if volume_24h < self.MIN_VOLUME_USD:
                    continue
                
                # Filtrar por timeframe
                time_to_resolution = self._calculate_time_to_resolution(market)
                if time_to_resolution and time_to_resolution > timedelta(hours=self.MAX_TIMEFRAME_HOURS):
                    continue
                
                # Buscar orderbooks com timeout
                try:
                    market_with_books = await asyncio.wait_for(
                        self.collector.get_market_with_orderbooks(
                            market.condition_id
                        ),
                        timeout=10.0  # 10 segundos por mercado
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout ao buscar orderbook para {market.condition_id}")
                    continue
                except Exception as e:
                    logger.warning(f"Erro ao buscar orderbook para {market.condition_id}: {e}")
                    continue
                
                if not market_with_books or not market_with_books.has_orderbook:
                    continue
                
                # Analisar oportunidade
                opportunity = await self._analyze_market(market_with_books, volume_24h, time_to_resolution)
                
                if opportunity and opportunity.expected_value >= self.MIN_EV:
                    opportunities.append(opportunity)
                    
            except Exception as e:
                logger.warning(f"Erro analisando mercado {market.ticker}: {e}")
                continue
        
        # Ordenar por EV (maior primeiro)
        opportunities.sort(key=lambda x: x.expected_value, reverse=True)
        
        logger.info(f"Encontradas {len(opportunities)} oportunidades")
        return opportunities
    
    def _extract_volume(self, market: Market) -> Decimal:
        """Extrai volume 24h do mercado."""
        raw_data = market.raw_data
        
        # Tentar diferentes campos possíveis
        volume_fields = [
            "volume24hr",
            "volume24h",
            "volume_24h",
            "volume24hrUsd",
            "volume24hUsd",
            "volume",
            "volumeUSD",
        ]
        
        for field in volume_fields:
            value = raw_data.get(field)
            if value is not None:
                try:
                    return Decimal(str(value))
                except (ValueError, TypeError):
                    continue
        
        # Se não encontrou, retornar 0 (será filtrado)
        return Decimal("0")
    
    def _calculate_time_to_resolution(self, market: Market) -> Optional[timedelta]:
        """Calcula tempo até resolução do mercado."""
        if not market.close_time:
            return None
        
        now = datetime.utcnow()
        if market.close_time.tzinfo:
            # Se tem timezone, converter para UTC
            from datetime import timezone
            if market.close_time.tzinfo != timezone.utc:
                close_time_utc = market.close_time.astimezone(timezone.utc)
            else:
                close_time_utc = market.close_time
        else:
            # Assumir UTC se não tem timezone
            close_time_utc = market.close_time
        
        time_to_resolution = close_time_utc - now
        
        if time_to_resolution.total_seconds() < 0:
            return None  # Já fechou
        
        return time_to_resolution
    
    async def _analyze_market(
        self,
        market: Market,
        volume_24h: Decimal,
        time_to_resolution: Optional[timedelta]
    ) -> Optional[PolymarketOpportunity]:
        """
        Analisa um mercado específico para identificar oportunidade.
        
        Estratégia:
        1. Calcula probabilidade implícita do preço atual
        2. Estima probabilidade esperada (baseado em análise heurística)
        3. Calcula divergência
        4. Se divergência significativa, calcula EV
        """
        # Obter preço atual (midpoint do orderbook)
        yes_book = market.yes_book
        no_book = market.no_book
        
        if not yes_book or not yes_book.midpoint:
            return None
        
        current_price = yes_book.midpoint
        implied_prob = current_price
        
        # Estimar probabilidade esperada
        # Esta é uma heurística - em produção, você usaria dados externos
        # (odds de casas de apostas, modelos de previsão, etc.)
        expected_prob = self._estimate_expected_probability(market, current_price)
        
        # Calcular divergência
        divergence = abs(expected_prob - implied_prob)
        
        if divergence < self.MIN_DIVERGENCE:
            return None  # Divergência insuficiente
        
        # Determinar lado recomendado
        if expected_prob > implied_prob:
            # Probabilidade real é maior que preço atual
            # Recomendamos comprar YES (preço está barato)
            recommended_side = Side.YES
            edge = expected_prob - implied_prob
        else:
            # Probabilidade real é menor que preço atual
            # Recomendamos comprar NO (ou vender YES)
            recommended_side = Side.NO
            edge = implied_prob - expected_prob
        
        # Calcular liquidez
        liquidity = self._calculate_liquidity(market)
        
        # Calcular Expected Value
        # EV = (Probabilidade Real × Payout) - Custo Atual
        # Para YES: EV = (expected_prob × $1) - current_price
        # Para NO: EV = ((1 - expected_prob) × $1) - (1 - current_price)
        
        if recommended_side == Side.YES:
            ev_per_dollar = expected_prob - current_price
        else:
            no_price = Decimal("1") - current_price
            no_expected_prob = Decimal("1") - expected_prob
            ev_per_dollar = no_expected_prob - no_price
        
        # Aplicar penalidade por taxas
        fee_per_dollar = self.fee_structure.polymarket_fee_per_contract
        ev_after_fees = ev_per_dollar - fee_per_dollar
        
        if ev_after_fees <= 0:
            return None
        
        # Calcular retornos para diferentes investimentos
        investments = [Decimal("10"), Decimal("20"), Decimal("100"), Decimal("1000")]
        returns = {}
        
        for inv in investments:
            # Simular execução
            book = yes_book if recommended_side == Side.YES else no_book
            if not book:
                returns[inv] = Decimal("0")
                continue
            
            # Calcular número de contratos que podemos comprar
            exec_sim = self.simulator.simulate_buy(book, inv)
            
            if exec_sim.filled_size > 0:
                # Retorno esperado = (probabilidade × payout) - custo
                # Payout = $1 por contrato se ganhar, $0 se perder
                if recommended_side == Side.YES:
                    # Se compramos YES: ganhamos $1 se evento acontece
                    expected_payout = exec_sim.filled_size * expected_prob
                else:
                    # Se compramos NO: ganhamos $1 se evento NÃO acontece
                    expected_payout = exec_sim.filled_size * (Decimal("1") - expected_prob)
                
                # Aplicar taxas (já incluídas no custo total, mas vamos calcular separadamente)
                fees = self.fee_structure.calculate_polymarket_fee(
                    exec_sim.filled_size,
                    exec_sim.vwap
                )
                
                # Retorno líquido esperado = payout esperado - custo total (incluindo taxas)
                # O custo total já inclui o preço pago, então precisamos adicionar as taxas
                total_cost_with_fees = exec_sim.total_cost + fees
                net_return = expected_payout - total_cost_with_fees
                returns[inv] = net_return
            else:
                returns[inv] = Decimal("0")
        
        # Calcular confidence score
        confidence = self._calculate_confidence(volume_24h, liquidity, divergence, time_to_resolution)
        
        # Determinar investimento recomendado (baseado em liquidez)
        recommended_inv = min(investments[-1], liquidity * Decimal("0.1"))  # 10% da liquidez
        
        return PolymarketOpportunity(
            market=market,
            market_title=market.title,
            condition_id=market.condition_id or "",
            polymarket_price=current_price,
            implied_probability=implied_prob,
            expected_probability=expected_prob,
            divergence=divergence,
            volume_24h=volume_24h,
            liquidity_depth=liquidity,
            time_to_resolution=time_to_resolution,
            expected_value=ev_after_fees,
            edge_percentage=edge * 100,
            recommended_side=recommended_side,
            recommended_investment=recommended_inv,
            returns_10=returns[Decimal("10")],
            returns_20=returns[Decimal("20")],
            returns_100=returns[Decimal("100")],
            returns_1000=returns[Decimal("1000")],
            detected_at=datetime.utcnow(),
            confidence_score=confidence
        )
    
    def _estimate_expected_probability(self, market: Market, current_price: Decimal) -> Decimal:
        """
        Estima probabilidade esperada do evento usando múltiplas heurísticas.
        
        Estratégias usadas:
        1. Análise de spread bid-ask (spreads grandes indicam ineficiência)
        2. Análise do orderbook (desequilíbrios indicam pressão de preço)
        3. Correção por volume (mercados com menos volume têm mais ineficiências)
        4. Análise de extremos (preços perto de 0 ou 1 tendem a reverter)
        """
        raw_data = market.raw_data
        volume_24h = self._extract_volume(market)
        
        # =====================================
        # ESTRATÉGIA 1: Análise de Spread
        # =====================================
        spread_adjustment = Decimal("0")
        if market.yes_book and market.yes_book.best_bid and market.yes_book.best_ask:
            spread = market.yes_book.best_ask - market.yes_book.best_bid
            # Spreads grandes (>5%) indicam oportunidade
            if spread > Decimal("0.05"):
                # Preço provavelmente deve mover para o meio do spread
                mid_spread = (market.yes_book.best_bid + market.yes_book.best_ask) / 2
                spread_adjustment = (mid_spread - current_price) * Decimal("0.3")
        
        # =====================================
        # ESTRATÉGIA 2: Desequilíbrio do Orderbook
        # =====================================
        imbalance_adjustment = Decimal("0")
        if market.yes_book:
            bid_depth = market.yes_book.total_bid_depth()
            ask_depth = market.yes_book.total_ask_depth()
            total_depth = bid_depth + ask_depth
            
            if total_depth > 0:
                # Se há mais bids que asks, preço deve subir
                imbalance = (bid_depth - ask_depth) / total_depth
                imbalance_adjustment = imbalance * Decimal("0.05")
        
        # =====================================
        # ESTRATÉGIA 3: Correção por Volume
        # =====================================
        volume_adjustment = Decimal("0")
        if volume_24h < Decimal("5000"):
            # Volume muito baixo - mais ineficiência
            volume_adjustment = (Decimal("0.5") - current_price) * Decimal("0.20")
        elif volume_24h < Decimal("20000"):
            volume_adjustment = (Decimal("0.5") - current_price) * Decimal("0.15")
        elif volume_24h < Decimal("50000"):
            volume_adjustment = (Decimal("0.5") - current_price) * Decimal("0.10")
        else:
            volume_adjustment = (Decimal("0.5") - current_price) * Decimal("0.05")
        
        # =====================================
        # ESTRATÉGIA 4: Correção de Extremos
        # =====================================
        extreme_adjustment = Decimal("0")
        if current_price < Decimal("0.15"):
            # Preços muito baixos tendem a subir
            extreme_adjustment = Decimal("0.05")
        elif current_price > Decimal("0.85"):
            # Preços muito altos tendem a cair
            extreme_adjustment = Decimal("-0.05")
        elif current_price < Decimal("0.30"):
            extreme_adjustment = Decimal("0.03")
        elif current_price > Decimal("0.70"):
            extreme_adjustment = Decimal("-0.03")
        
        # =====================================
        # COMBINAR AJUSTES
        # =====================================
        total_adjustment = (
            spread_adjustment * Decimal("0.3") +
            imbalance_adjustment * Decimal("0.3") +
            volume_adjustment * Decimal("0.25") +
            extreme_adjustment * Decimal("0.15")
        )
        
        estimated = current_price + total_adjustment
        
        # Garantir que está entre 0.02 e 0.98
        estimated = max(Decimal("0.02"), min(Decimal("0.98"), estimated))
        
        return estimated
    
    def _calculate_liquidity(self, market: Market) -> Decimal:
        """Calcula profundidade de liquidez disponível."""
        total_liquidity = Decimal("0")
        
        if market.yes_book:
            # Liquidez no lado YES (asks)
            yes_liquidity = market.yes_book.total_ask_depth()
            # Converter para USD (assumindo preço médio)
            if market.yes_book.midpoint:
                total_liquidity += yes_liquidity * market.yes_book.midpoint
        
        if market.no_book:
            # Liquidez no lado NO (asks)
            no_liquidity = market.no_book.total_ask_depth()
            if market.no_book.midpoint:
                total_liquidity += no_liquidity * market.no_book.midpoint
        
        return total_liquidity
    
    def _calculate_confidence(
        self,
        volume: Decimal,
        liquidity: Decimal,
        divergence: Decimal,
        time_to_resolution: Optional[timedelta]
    ) -> Decimal:
        """Calcula score de confiança (0-1) para a oportunidade."""
        score = Decimal("0.5")  # Base
        
        # Volume alto aumenta confiança
        if volume > Decimal("500000"):
            score += Decimal("0.2")
        elif volume > Decimal("200000"):
            score += Decimal("0.1")
        
        # Liquidez alta aumenta confiança
        if liquidity > Decimal("50000"):
            score += Decimal("0.15")
        elif liquidity > Decimal("20000"):
            score += Decimal("0.1")
        
        # Divergência maior aumenta confiança (até certo ponto)
        if divergence > Decimal("0.20"):
            score += Decimal("0.1")
        elif divergence > Decimal("0.15"):
            score += Decimal("0.05")
        
        # Tempo até resolução: mais próximo aumenta confiança
        if time_to_resolution:
            hours = time_to_resolution.total_seconds() / 3600
            if hours < 12:
                score += Decimal("0.1")
            elif hours < 24:
                score += Decimal("0.05")
        
        return min(Decimal("1.0"), score)
    
    def format_opportunity(self, opp: PolymarketOpportunity, rank: int) -> str:
        """Formata uma oportunidade para exibição."""
        time_str = "N/A"
        if opp.time_to_resolution:
            hours = opp.time_to_resolution.total_seconds() / 3600
            time_str = f"{hours:.1f} horas"
        
        side_str = "YES" if opp.recommended_side == Side.YES else "NO"
        
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  OPORTUNIDADE #{rank} - POLYMARKET ARBITRAGE                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Mercado: {opp.market_title[:70]:<70} ║
║ Condition ID: {opp.condition_id[:66]:<66} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ANÁLISE DE PROBABILIDADE:                                                    ║
║   Preço Polymarket:        ${opp.polymarket_price:.4f} ({opp.implied_probability*100:.2f}%)        ║
║   Probabilidade Esperada:  {opp.expected_probability*100:.2f}%                                    ║
║   Divergência:             {opp.divergence*100:.2f}%                                             ║
║   Edge:                    {opp.edge_percentage:.2f}%                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ DADOS DE MERCADO:                                                            ║
║   Volume 24h:              ${opp.volume_24h:,.2f}                                    ║
║   Liquidez Disponível:     ${opp.liquidity_depth:,.2f}                                    ║
║   Tempo até Resolução:     {time_str:<60} ║
║   Confidence Score:        {opp.confidence_score*100:.1f}%                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ESTRATÉGIA RECOMENDADA:                                                       ║
║   Apostar em:              {side_str:<60} ║
║   Investimento Sugerido:   ${opp.recommended_investment:.2f}                                    ║
║   Expected Value (EV):     ${opp.expected_value:.4f} por $1 investido              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ RETORNOS PROJETADOS:                                                         ║
║   ┌─────────────┬──────────────┬──────────────┬──────────────┬──────────────┐ ║
║   │ Investimento│   Retorno    │   ROI (%)    │   Tempo Est. │   Confiança  │ ║
║   ├─────────────┼──────────────┼──────────────┼──────────────┼──────────────┤ ║
║   │    $10      │  ${opp.returns_10:>8.2f}   │  {(opp.returns_10/10*100):>6.2f}%   │  {time_str[:12]:<12} │  {opp.confidence_score*100:>5.1f}%   │ ║
║   │    $20      │  ${opp.returns_20:>8.2f}   │  {(opp.returns_20/20*100):>6.2f}%   │  {time_str[:12]:<12} │  {opp.confidence_score*100:>5.1f}%   │ ║
║   │   $100      │  ${opp.returns_100:>8.2f}   │  {(opp.returns_100/100*100):>6.2f}%   │  {time_str[:12]:<12} │  {opp.confidence_score*100:>5.1f}%   │ ║
║   │  $1,000     │  ${opp.returns_1000:>8.2f}   │  {(opp.returns_1000/1000*100):>6.2f}%   │  {time_str[:12]:<12} │  {opp.confidence_score*100:>5.1f}%   │ ║
║   └─────────────┴──────────────┴──────────────┴──────────────┴──────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    def format_summary(self, opportunities: List[PolymarketOpportunity]) -> str:
        """Formata resumo das top oportunidades."""
        if not opportunities:
            return "\n❌ Nenhuma oportunidade encontrada com os critérios especificados.\n"
        
        top_3 = opportunities[:3]
        
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TOP 3 OPORTUNIDADES DE ARBITRAGEM                        ║
║                    Polymarket - Análise em Tempo Real                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Total de oportunidades encontradas: {len(opportunities):<50} ║
║ Data/Hora da análise: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'):<50} ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        for i, opp in enumerate(top_3, 1):
            summary += self.format_opportunity(opp, i)
        
        return summary


async def main():
    """Função principal para executar análise."""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async with ArbitrageAnalyst() as analyst:
        print("\n🔍 Escaneando mercados do Polymarket...")
        print("   Critérios:")
        print(f"   - Volume mínimo: ${analyst.MIN_VOLUME_USD:,.0f}")
        print(f"   - Timeframe máximo: {analyst.MAX_TIMEFRAME_HOURS} horas")
        print(f"   - Divergência mínima: {analyst.MIN_DIVERGENCE*100:.0f}%")
        print(f"   - EV mínimo: ${analyst.MIN_EV:.2f} por $1 investido\n")
        
        opportunities = await analyst.scan_opportunities(limit=200)
        
        summary = analyst.format_summary(opportunities)
        print(summary)


if __name__ == "__main__":
    asyncio.run(main())

