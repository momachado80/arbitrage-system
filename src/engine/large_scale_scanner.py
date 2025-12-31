"""
Large Scale Scanner - OPTIMIZED VERSION
========================================

Scanner otimizado:
- Coleta TODOS os mercados (até 50.000 por plataforma)
- NÃO busca orderbooks durante coleta inicial (muito lento)
- Smart matching por entidades
- Apenas 50 análises semânticas (para evitar timeout)
- Busca orderbooks APENAS para pares confirmados
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional, Tuple

from src.collectors import KalshiCollector, PolymarketCollector
from src.models import Market, Side
from src.engine.smart_matcher import SmartMarketMatcher as SmartMatcher
from src.engine.semantic_analyzer import SemanticAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PairAnalysis:
    """Resultado da análise de um par."""
    kalshi: Market
    polymarket: Market
    category: str
    event_type: str = ""
    match_score: float = 0.0
    match_reason: str = ""
    # Campos semânticos
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
    opportunities_found: int
    top_pairs: List[Dict]
    scan_duration_seconds: float


class LargeScaleScanner:
    """
    Scanner de larga escala OTIMIZADO.
    - Coleta mercados SEM orderbooks (rápido)
    - Smart matching por entidades
    - Análise semântica limitada a 50 pares
    """
    
    def __init__(
        self,
        target_usd: Decimal = Decimal("100"),
        min_similarity: float = 0.35,
        min_edge: Decimal = Decimal("0.01"),
        max_markets_per_platform: int = 50000,
        use_semantic: bool = True,
        max_semantic_checks: int = 50,  # REDUZIDO para evitar timeout
    ):
        self.target_usd = target_usd
        self.min_similarity = min_similarity
        self.min_edge = min_edge
        self.max_markets = max_markets_per_platform
        self.use_semantic = use_semantic
        self.max_semantic_checks = max_semantic_checks
        
        self.smart_matcher = SmartMatcher(min_match_score=min_similarity)
        self.semantic_analyzer = SemanticAnalyzer() if use_semantic else None
    
    async def full_scan(self) -> FullScanResult:
        """Executa scan completo otimizado."""
        start_time = datetime.utcnow()
        logger.info("=== INICIANDO SCAN OTIMIZADO ===")
        
        # 1. Coletar mercados SEM orderbooks (rápido)
        logger.info("Fase 1: Coletando mercados...")
        kalshi_markets = await self._collect_all_kalshi()
        poly_markets = await self._collect_all_polymarket()
        
        logger.info(f"Coletados: {len(kalshi_markets)} Kalshi, {len(poly_markets)} Polymarket")
        
        # 2. Smart matching por entidades
        logger.info("Fase 2: Smart matching por categorias...")
        matches = self.smart_matcher.find_matches(kalshi_markets, poly_markets)
        logger.info(f"Smart matches: {len(matches)}")
        
        # 3. Análise semântica (limitada a max_semantic_checks)
        top_pairs = []
        same_event_count = 0
        semantic_analyzed = 0
        
        if self.use_semantic and self.semantic_analyzer:
            logger.info(f"Fase 3: Análise semântica de até {self.max_semantic_checks} candidatos...")
            
            # Ordenar por score e pegar top N
            sorted_matches = sorted(matches, key=lambda x: x.match_score, reverse=True)[:self.max_semantic_checks]
            
            for match in sorted_matches:
                semantic_analyzed += 1
                
                analysis = PairAnalysis(
                    kalshi=match.kalshi,
                    polymarket=match.polymarket,
                    category=match.category,
                    event_type=match.event_type,
                    match_score=match.match_score,
                    match_reason=match.match_reason,
                )
                
                # Análise semântica
                try:
                    semantic_result = await self.semantic_analyzer.analyze_pair(
                        match.kalshi.title,
                        match.polymarket.title,
                        match.kalshi.description or "",
                        match.polymarket.description or "",
                    )
                    
                    analysis.is_same_event = semantic_result.get("is_same_event", False)
                    analysis.semantic_confidence = semantic_result.get("confidence", 0)
                    analysis.semantic_reasoning = semantic_result.get("reasoning", "")
                    analysis.event_description = semantic_result.get("event_description", "")
                    
                    if analysis.is_same_event:
                        same_event_count += 1
                        logger.info(f"✓ MESMO EVENTO: {match.kalshi.title[:50]} <-> {match.polymarket.title[:50]}")
                        # Inserir confirmados no início
                        top_pairs.insert(0, self._analysis_to_dict(analysis))
                    else:
                        top_pairs.append(self._analysis_to_dict(analysis))
                        
                except Exception as e:
                    logger.warning(f"Erro análise semântica: {e}")
                    top_pairs.append(self._analysis_to_dict(analysis))
            
            logger.info(f"Análise semântica: {semantic_analyzed} analisados, {same_event_count} confirmados")
        else:
            # Sem análise semântica
            for match in matches[:100]:
                analysis = PairAnalysis(
                    kalshi=match.kalshi,
                    polymarket=match.polymarket,
                    category=match.category,
                    event_type=match.event_type,
                    match_score=match.match_score,
                    match_reason=match.match_reason,
                )
                top_pairs.append(self._analysis_to_dict(analysis))
        
        # 4. Buscar orderbooks APENAS para pares confirmados
        logger.info("Fase 4: Buscando orderbooks para pares confirmados...")
        # TODO: Implementar busca de orderbooks para pares confirmados
        
        # Calcular duração
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"=== SCAN COMPLETO em {duration:.1f}s ===")
        logger.info(f"Confirmados: {same_event_count} | Analisados: {semantic_analyzed}")
        
        return FullScanResult(
            timestamp=end_time,
            kalshi_total=len(kalshi_markets),
            polymarket_total=len(poly_markets),
            smart_matches=len(matches),
            semantic_analyzed=semantic_analyzed,
            same_event_confirmed=same_event_count,
            opportunities_found=0,
            top_pairs=top_pairs,
            scan_duration_seconds=duration,
        )
    
    async def _collect_all_kalshi(self) -> List[Market]:
        """Coleta todos os mercados Kalshi SEM orderbooks."""
        all_markets = []
        
        async with KalshiCollector() as collector:
            cursor = None
            
            while len(all_markets) < self.max_markets:
                try:
                    markets, next_cursor = await collector.get_markets(
                        status="open",
                        limit=200,
                        cursor=cursor,
                    )
                    
                    if not markets:
                        break
                    
                    all_markets.extend(markets)
                    
                    if len(all_markets) % 200 == 0:
                        logger.info(f"Kalshi: {len(all_markets)} mercados coletados...")
                    
                    if not next_cursor:
                        break
                    cursor = next_cursor
                    
                except Exception as e:
                    logger.error(f"Erro coletando Kalshi: {e}")
                    break
        
        return all_markets
    
    async def _collect_all_polymarket(self) -> List[Market]:
        """Coleta todos os mercados Polymarket SEM orderbooks."""
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
                    
                    # NÃO buscar orderbooks aqui - muito lento!
                    all_markets.extend(markets)
                    
                    if len(all_markets) % 1000 == 0:
                        logger.info(f"Polymarket: {len(all_markets)} mercados coletados...")
                    
                    offset += 100
                    
                    if len(markets) < 100:
                        break
                        
                except Exception as e:
                    logger.error(f"Erro coletando Polymarket: {e}")
                    break
        
        logger.info(f"Polymarket: {len(all_markets)} mercados coletados (final)")
        return all_markets
    
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
            "is_same_event": analysis.is_same_event,
            "semantic_confidence": analysis.semantic_confidence,
            "semantic_reasoning": analysis.semantic_reasoning,
            "event_description": analysis.event_description,
            "has_arbitrage": analysis.has_arbitrage,
            "edge_percentage": analysis.edge_percentage,
            "strategy": analysis.strategy,
            "net_profit": analysis.net_profit,
            "rejection_reason": analysis.rejection_reason,
        }
