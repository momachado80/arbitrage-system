"""
Large Scale Scanner
===================

Scanner otimizado para escanear TODOS os mercados de ambas plataformas.

Estratégia:
1. Coleta todos os mercados (paginado)
2. Extrai entidades específicas (pessoas, empresas, eventos, datas)
3. Só compara mercados com entidades em comum
4. Filtra por liquidez
5. Retorna oportunidades encontradas

Este scanner é projetado para rodar em background continuamente.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional, Tuple

from src.collectors import KalshiCollector, PolymarketCollector
from src.models import Market, Side
from src.engine.entity_matcher import EntityBasedMatcher, EntityMatch
from src.engine.contract_equivalence import (
    ContractEquivalenceAnalyzer,
    EquivalenceResult,
    EquivalenceLevel,
)
from src.engine.arbitrage_calculator import ArbitrageCalculator

logger = logging.getLogger(__name__)


@dataclass
class PairAnalysis:
    """Resultado da análise de um par."""
    kalshi: Market
    polymarket: Market
    shared_entities: set
    entity_type: str
    entity_confidence: float
    has_arbitrage: bool = False
    edge_percentage: float = 0.0
    strategy: str = ""
    net_profit: float = 0.0
    rejection_reason: str = ""


@dataclass 
class FullScanResult:
    """Resultado de um scan completo."""
    timestamp: datetime
    kalshi_total: int
    polymarket_total: int
    entities_matched: int
    pairs_analyzed: int
    pairs_with_liquidity: int
    opportunities_found: int
    opportunities: List[Dict]
    top_pairs: List[Dict]  # Melhores pares mesmo sem arbitragem
    scan_duration_seconds: float
    entity_stats: Dict[str, int]  # Contagem por tipo de entidade


class LargeScaleScanner:
    """
    Scanner de larga escala para todos os mercados.
    Usa matching baseado em entidades específicas.
    """
    
    def __init__(
        self,
        target_usd: Decimal = Decimal("100"),
        min_similarity: float = 0.25,
        min_edge: Decimal = Decimal("0.01"),
        max_markets_per_platform: int = 5000,
    ):
        self.target_usd = target_usd
        self.min_similarity = min_similarity
        self.min_edge = min_edge
        self.max_markets = max_markets_per_platform
        
        self.entity_matcher = EntityBasedMatcher(min_shared_entities=1)
        self.equivalence_analyzer = ContractEquivalenceAnalyzer(
            min_title_similarity=min_similarity,
            require_date_match=False,
        )
        self.calculator = ArbitrageCalculator(min_edge_threshold=min_edge)
    
    async def full_scan(self) -> FullScanResult:
        """
        Executa scan completo de todos os mercados.
        """
        start_time = datetime.utcnow()
        logger.info("=== INICIANDO SCAN COMPLETO ===")
        
        # 1. Coletar todos os mercados
        logger.info("Fase 1: Coletando mercados...")
        kalshi_markets = await self._collect_all_kalshi()
        poly_markets = await self._collect_all_polymarket()
        
        logger.info(f"Coletados: {len(kalshi_markets)} Kalshi, {len(poly_markets)} Polymarket")
        
        # 2. Encontrar matches por entidades específicas
        logger.info("Fase 2: Matching por entidades...")
        entity_matches = self.entity_matcher.find_matches(
            kalshi_markets, 
            poly_markets,
            min_confidence=self.min_similarity
        )
        
        logger.info(f"Matches por entidade: {len(entity_matches)}")
        
        # 3. Analisar pares encontrados
        logger.info("Fase 3: Analisando arbitragem...")
        analyses = []
        entity_type_counts = {}
        
        for match in entity_matches:
            # Contar por tipo de entidade
            entity_type_counts[match.entity_type] = entity_type_counts.get(match.entity_type, 0) + 1
            
            analysis = self._analyze_entity_match(match)
            if analysis:
                analyses.append(analysis)
        
        # 4. Filtrar resultados
        opportunities = []
        top_pairs = []
        pairs_with_liquidity = 0
        
        for analysis in analyses:
            # Contar pares com liquidez
            if analysis.kalshi.yes_book or analysis.kalshi.no_book:
                pairs_with_liquidity += 1
            
            # Coletar oportunidades
            if analysis.has_arbitrage:
                opportunities.append(self._analysis_to_dict(analysis))
            
            # Coletar top pares (mesmo sem arbitragem)
            if analysis.entity_confidence >= 0.3:
                top_pairs.append(self._analysis_to_dict(analysis))
        
        # Ordenar
        opportunities.sort(key=lambda x: x["edge_percentage"], reverse=True)
        top_pairs.sort(key=lambda x: x["confidence"], reverse=True)
        top_pairs = top_pairs[:50]  # Top 50
        
        # Calcular duração
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"=== SCAN COMPLETO em {duration:.1f}s ===")
        logger.info(f"Oportunidades: {len(opportunities)} | Top pares: {len(top_pairs)}")
        
        return FullScanResult(
            timestamp=end_time,
            kalshi_total=len(kalshi_markets),
            polymarket_total=len(poly_markets),
            entities_matched=len(entity_matches),
            pairs_analyzed=len(analyses),
            pairs_with_liquidity=pairs_with_liquidity,
            opportunities_found=len(opportunities),
            opportunities=opportunities,
            top_pairs=top_pairs,
            scan_duration_seconds=duration,
            entity_stats=entity_type_counts,
        )
    
    def _analyze_entity_match(self, match: EntityMatch) -> Optional[PairAnalysis]:
        """Analisa um match de entidades para arbitragem."""
        analysis = PairAnalysis(
            kalshi=match.kalshi,
            polymarket=match.polymarket,
            shared_entities=match.shared_entities,
            entity_type=match.entity_type,
            entity_confidence=match.confidence,
        )
        
        # Tentar calcular arbitragem
        try:
            equiv_score = Decimal(str(match.confidence))
            
            opportunity, rejection = self.calculator.calculate(
                kalshi_market=match.kalshi,
                polymarket_market=match.polymarket,
                target_usd=self.target_usd,
                equivalence_score=equiv_score,
            )
            
            if opportunity:
                analysis.has_arbitrage = True
                analysis.edge_percentage = float(opportunity.edge_percentage)
                analysis.strategy = opportunity.strategy
                analysis.net_profit = float(opportunity.net_profit)
            elif rejection:
                analysis.rejection_reason = rejection.code
                
        except Exception as e:
            analysis.rejection_reason = str(e)
        
        return analysis
    
    def _analysis_to_dict(self, analysis: PairAnalysis) -> Dict:
        """Converte análise para dicionário."""
        return {
            "kalshi_ticker": analysis.kalshi.ticker,
            "kalshi_title": analysis.kalshi.title,
            "polymarket_ticker": analysis.polymarket.ticker,
            "polymarket_title": analysis.polymarket.title,
            "shared_entities": list(analysis.shared_entities),
            "entity_type": analysis.entity_type,
            "confidence": analysis.entity_confidence,
            "has_arbitrage": analysis.has_arbitrage,
            "edge_percentage": analysis.edge_percentage,
            "strategy": analysis.strategy,
            "net_profit": analysis.net_profit,
            "rejection_reason": analysis.rejection_reason,
        }
    
    async def _collect_all_kalshi(self) -> List[Market]:
        """Coleta todos os mercados Kalshi com paginação."""
        all_markets = []
        
        async with KalshiCollector() as collector:
            cursor = None
            
            while len(all_markets) < self.max_markets:
                try:
                    markets, next_cursor = await collector.get_markets(
                        status="open",
                        limit=200,  # Máximo por página
                        cursor=cursor,
                    )
                    
                    if not markets:
                        break
                    
                    # Buscar orderbooks em paralelo (batches de 20)
                    for i in range(0, len(markets), 20):
                        batch = markets[i:i+20]
                        tasks = []
                        
                        for market in batch:
                            tasks.append(self._fetch_kalshi_orderbook(collector, market))
                        
                        await asyncio.gather(*tasks, return_exceptions=True)
                    
                    all_markets.extend(markets)
                    logger.info(f"Kalshi: {len(all_markets)} mercados coletados...")
                    
                    if not next_cursor:
                        break
                    cursor = next_cursor
                    
                except Exception as e:
                    logger.error(f"Erro coletando Kalshi: {e}")
                    break
        
        return all_markets
    
    async def _fetch_kalshi_orderbook(self, collector: KalshiCollector, market: Market):
        """Busca orderbook de um mercado Kalshi."""
        try:
            yes_book, no_book = await collector.get_orderbook(market.ticker)
            market.yes_book = yes_book
            market.no_book = no_book
        except Exception as e:
            logger.debug(f"Erro orderbook Kalshi {market.ticker}: {e}")
    
    async def _collect_all_polymarket(self) -> List[Market]:
        """Coleta todos os mercados Polymarket com paginação."""
        all_markets = []
        
        async with PolymarketCollector() as collector:
            offset = 0
            
            while len(all_markets) < self.max_markets:
                try:
                    markets = await collector.get_markets(
                        active=True,
                        limit=100,
                        offset=offset,
                    )
                    
                    if not markets:
                        break
                    
                    # Buscar orderbooks em paralelo
                    for i in range(0, len(markets), 20):
                        batch = markets[i:i+20]
                        tasks = []
                        
                        for market in batch:
                            tasks.append(self._fetch_poly_orderbook(collector, market))
                        
                        await asyncio.gather(*tasks, return_exceptions=True)
                    
                    all_markets.extend(markets)
                    logger.info(f"Polymarket: {len(all_markets)} mercados coletados...")
                    
                    offset += 100
                    
                    if len(markets) < 100:
                        break
                        
                except Exception as e:
                    logger.error(f"Erro coletando Polymarket: {e}")
                    break
        
        return all_markets
    
    async def _fetch_poly_orderbook(self, collector: PolymarketCollector, market: Market):
        """Busca orderbook de um mercado Polymarket."""
        try:
            if market.yes_token_id:
                yes_book = await collector.get_orderbook(market.yes_token_id)
                if yes_book:
                    yes_book.side = Side.YES
                market.yes_book = yes_book
            
            if market.no_token_id:
                no_book = await collector.get_orderbook(market.no_token_id)
                if no_book:
                    no_book.side = Side.NO
                market.no_book = no_book
        except Exception as e:
            logger.debug(f"Erro orderbook Poly {market.ticker}: {e}")
