"""
Large Scale Scanner
===================

Scanner otimizado para escanear TODOS os mercados de ambas plataformas.

Estratégia:
1. Coleta todos os mercados (paginado)
2. Smart matching por categoria (sports, politics, economics, crypto)
3. Análise semântica com DeepSeek para confirmar
4. Calcula arbitragem nos matches confirmados
5. Retorna oportunidades encontradas

Este scanner é projetado para rodar em background continuamente.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional, Tuple

from src.collectors import KalshiCollector, PolymarketCollector
from src.models import Market, Side
from src.engine.smart_matcher import SmartMarketMatcher, SmartMatch
from src.engine.arbitrage_calculator import ArbitrageCalculator

logger = logging.getLogger(__name__)

# Flag para habilitar análise semântica
SEMANTIC_ENABLED = bool(os.environ.get("DEEPSEEK_API_KEY"))


@dataclass
class PairAnalysis:
    """Resultado da análise de um par."""
    kalshi: Market
    polymarket: Market
    category: str
    event_type: str
    match_score: float
    match_reason: str
    # Campos de análise semântica
    is_same_event: bool = False
    semantic_confidence: float = 0.0
    semantic_reasoning: str = ""
    event_description: str = ""
    # Campos de arbitragem
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
    smart_matches: int
    semantic_analyzed: int
    same_event_confirmed: int
    pairs_analyzed: int
    pairs_with_liquidity: int
    opportunities_found: int
    opportunities: List[Dict]
    top_pairs: List[Dict]  # Melhores pares mesmo sem arbitragem
    scan_duration_seconds: float
    category_stats: Dict[str, int]  # Contagem por categoria
    semantic_enabled: bool


class LargeScaleScanner:
    """
    Scanner de larga escala para todos os mercados.
    Usa smart matching por categoria + análise semântica.
    """
    
    def __init__(
        self,
        target_usd: Decimal = Decimal("100"),
        min_similarity: float = 0.5,  # Score mínimo do smart match
        min_edge: Decimal = Decimal("0.01"),
        max_markets_per_platform: int = 50000,
        use_semantic: bool = True,
        max_semantic_checks: int = 500,
    ):
        self.target_usd = target_usd
        self.min_similarity = min_similarity
        self.min_edge = min_edge
        self.max_markets = max_markets_per_platform
        self.use_semantic = use_semantic and SEMANTIC_ENABLED
        self.max_semantic_checks = max_semantic_checks
        
        # Smart matcher por categorias
        self.smart_matcher = SmartMarketMatcher(min_match_score=min_similarity)
        self.calculator = ArbitrageCalculator(min_edge_threshold=min_edge)
        
        # Inicializar analisador semântico se disponível
        self.semantic_analyzer = None
        if self.use_semantic:
            try:
                from src.engine.semantic_analyzer import SemanticAnalyzer
                self.semantic_analyzer = SemanticAnalyzer()
                logger.info("Análise semântica HABILITADA")
            except Exception as e:
                logger.warning(f"Análise semântica não disponível: {e}")
                self.use_semantic = False
    
    async def full_scan(self) -> FullScanResult:
        """
        Executa scan completo de todos os mercados.
        """
        start_time = datetime.utcnow()
        logger.info("=== INICIANDO SCAN COMPLETO ===")
        logger.info(f"Análise semântica: {'HABILITADA' if self.use_semantic else 'DESABILITADA'}")
        
        # 1. Coletar todos os mercados
        logger.info("Fase 1: Coletando mercados...")
        kalshi_markets = await self._collect_all_kalshi()
        poly_markets = await self._collect_all_polymarket()
        
        logger.info(f"Coletados: {len(kalshi_markets)} Kalshi, {len(poly_markets)} Polymarket")
        
        # 2. Smart matching por categorias
        logger.info("Fase 2: Smart matching por categorias...")
        smart_matches = self.smart_matcher.find_matches(kalshi_markets, poly_markets)
        
        logger.info(f"Smart matches: {len(smart_matches)}")
        
        # Contar por categoria
        category_counts = {}
        for match in smart_matches:
            category_counts[match.category] = category_counts.get(match.category, 0) + 1
        
        # 3. Análise semântica (se habilitada)
        semantic_analyzed = 0
        same_event_confirmed = 0
        semantic_results = {}
        
        if self.use_semantic and self.semantic_analyzer and smart_matches:
            logger.info(f"Fase 3: Análise semântica de até {self.max_semantic_checks} candidatos...")
            candidates = smart_matches[:self.max_semantic_checks]
            
            for match in candidates:
                try:
                    result = await self.semantic_analyzer.analyze_pair(match.kalshi, match.polymarket)
                    semantic_analyzed += 1
                    
                    key = (match.kalshi.ticker, match.polymarket.ticker)
                    semantic_results[key] = result
                    
                    if result.is_same_event:
                        same_event_confirmed += 1
                        logger.info(f"✓ MESMO EVENTO: {match.kalshi.title[:50]} <-> {match.polymarket.title[:50]}")
                    
                except Exception as e:
                    logger.error(f"Erro análise semântica: {e}")
            
            logger.info(f"Análise semântica: {semantic_analyzed} analisados, {same_event_confirmed} confirmados")
        
        # 4. Analisar pares encontrados
        logger.info("Fase 4: Analisando arbitragem...")
        analyses = []
        
        for match in smart_matches:
            # Buscar resultado semântico se disponível
            key = (match.kalshi.ticker, match.polymarket.ticker)
            semantic_result = semantic_results.get(key)
            
            analysis = self._analyze_smart_match(match, semantic_result)
            if analysis:
                analyses.append(analysis)
        
        # 5. Filtrar resultados
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
            
            # Coletar top pares - priorizar os confirmados semanticamente
            if analysis.is_same_event:
                top_pairs.insert(0, self._analysis_to_dict(analysis))
            elif analysis.match_score >= 0.5:
                top_pairs.append(self._analysis_to_dict(analysis))
        
        # Ordenar
        opportunities.sort(key=lambda x: x["edge_percentage"], reverse=True)
        confirmed = [p for p in top_pairs if p.get("is_same_event")]
        others = [p for p in top_pairs if not p.get("is_same_event")]
        others.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        top_pairs = confirmed + others[:100 - len(confirmed)]
        
        # Calcular duração
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"=== SCAN COMPLETO em {duration:.1f}s ===")
        logger.info(f"Oportunidades: {len(opportunities)} | Top pares: {len(top_pairs)}")
        
        return FullScanResult(
            timestamp=end_time,
            kalshi_total=len(kalshi_markets),
            polymarket_total=len(poly_markets),
            smart_matches=len(smart_matches),
            semantic_analyzed=semantic_analyzed,
            same_event_confirmed=same_event_confirmed,
            pairs_analyzed=len(analyses),
            pairs_with_liquidity=pairs_with_liquidity,
            opportunities_found=len(opportunities),
            opportunities=opportunities,
            top_pairs=top_pairs,
            scan_duration_seconds=duration,
            category_stats=category_counts,
            semantic_enabled=self.use_semantic,
        )
    
    def _analyze_smart_match(self, match: SmartMatch, semantic_result=None) -> Optional[PairAnalysis]:
        """Analisa um smart match para arbitragem."""
        analysis = PairAnalysis(
            kalshi=match.kalshi,
            polymarket=match.polymarket,
            category=match.category,
            event_type=match.event_type,
            match_score=match.match_score,
            match_reason=match.match_reason,
        )
        
        # Adicionar informação semântica se disponível
        if semantic_result:
            analysis.is_same_event = semantic_result.is_same_event
            analysis.semantic_confidence = semantic_result.confidence
            analysis.semantic_reasoning = semantic_result.reasoning
            analysis.event_description = semantic_result.event_description
        
        # Tentar calcular arbitragem
        try:
            # Se temos confirmação semântica, usar confiança mais alta
            if analysis.is_same_event and analysis.semantic_confidence >= 0.7:
                equiv_score = Decimal(str(analysis.semantic_confidence))
            else:
                equiv_score = Decimal(str(match.match_score))
            
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
            "category": analysis.category,
            "event_type": analysis.event_type,
            "match_score": analysis.match_score,
            "match_reason": analysis.match_reason,
            # Campos semânticos
            "is_same_event": analysis.is_same_event,
            "semantic_confidence": analysis.semantic_confidence,
            "semantic_reasoning": analysis.semantic_reasoning,
            "event_description": analysis.event_description,
            # Campos de arbitragem
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
