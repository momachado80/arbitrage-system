"""
Large Scale Scanner - ROBUST VERSION
=====================================

Scanner robusto que não crasha:
- Imports opcionais com fallback
- Timeout em cada operação
- Reset automático da flag em caso de erro
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional

from src.collectors import KalshiCollector, PolymarketCollector
from src.models import Market, Side

# Import opcional do SmartMatcher
try:
    from src.engine.smart_matcher import SmartMarketMatcher
    SMART_MATCHER_AVAILABLE = True
except ImportError:
    SMART_MATCHER_AVAILABLE = False
    SmartMarketMatcher = None

# Import opcional do SemanticAnalyzer
try:
    from src.engine.semantic_analyzer import SemanticAnalyzer
    SEMANTIC_AVAILABLE = bool(os.environ.get("DEEPSEEK_API_KEY"))
except ImportError:
    SEMANTIC_AVAILABLE = False
    SemanticAnalyzer = None

logger = logging.getLogger(__name__)


@dataclass
class PairAnalysis:
    """Resultado da análise de um par."""
    kalshi: Market
    polymarket: Market
    category: str = ""
    event_type: str = ""
    match_score: float = 0.0
    match_reason: str = ""
    is_same_event: bool = False
    semantic_confidence: float = 0.0
    semantic_reasoning: str = ""
    event_description: str = ""
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
    """Scanner de larga escala robusto."""
    
    def __init__(
        self,
        target_usd: Decimal = Decimal("100"),
        min_similarity: float = 0.35,
        min_edge: Decimal = Decimal("0.01"),
        max_markets_per_platform: int = 1000,  # REDUZIDO para evitar timeout
        use_semantic: bool = True,
        max_semantic_checks: int = 20,  # REDUZIDO para evitar timeout
    ):
        self.target_usd = target_usd
        self.min_similarity = min_similarity
        self.min_edge = min_edge
        self.max_markets = max_markets_per_platform
        self.use_semantic = use_semantic and SEMANTIC_AVAILABLE
        self.max_semantic_checks = max_semantic_checks
        
        # Inicializar matcher se disponível
        if SMART_MATCHER_AVAILABLE:
            self.smart_matcher = SmartMarketMatcher(min_match_score=min_similarity)
        else:
            self.smart_matcher = None
            
        # Inicializar analyzer se disponível
        if self.use_semantic and SemanticAnalyzer:
            try:
                self.semantic_analyzer = SemanticAnalyzer()
            except Exception as e:
                logger.warning(f"Erro inicializando SemanticAnalyzer: {e}")
                self.semantic_analyzer = None
                self.use_semantic = False
        else:
            self.semantic_analyzer = None
    
    async def full_scan(self) -> FullScanResult:
        """Executa scan completo com tratamento de erros."""
        start_time = datetime.utcnow()
        logger.info("=== INICIANDO SCAN ROBUSTO ===")
        
        kalshi_markets = []
        poly_markets = []
        matches = []
        top_pairs = []
        same_event_count = 0
        semantic_analyzed = 0
        
        try:
            # 1. Coletar mercados Kalshi
            logger.info("Fase 1a: Coletando Kalshi...")
            try:
                kalshi_markets = await asyncio.wait_for(
                    self._collect_all_kalshi(),
                    timeout=120  # 2 minutos max
                )
            except asyncio.TimeoutError:
                logger.error("Timeout coletando Kalshi")
            except Exception as e:
                logger.error(f"Erro coletando Kalshi: {e}")
            
            logger.info(f"Kalshi: {len(kalshi_markets)} mercados")
            
            # 2. Coletar mercados Polymarket
            logger.info("Fase 1b: Coletando Polymarket...")
            try:
                poly_markets = await asyncio.wait_for(
                    self._collect_all_polymarket(),
                    timeout=180  # 3 minutos max
                )
            except asyncio.TimeoutError:
                logger.error("Timeout coletando Polymarket")
            except Exception as e:
                logger.error(f"Erro coletando Polymarket: {e}")
            
            logger.info(f"Polymarket: {len(poly_markets)} mercados")
            
            # 3. Smart matching
            if self.smart_matcher and kalshi_markets and poly_markets:
                logger.info("Fase 2: Smart matching...")
                try:
                    matches = self.smart_matcher.find_matches(kalshi_markets, poly_markets)
                    logger.info(f"Matches encontrados: {len(matches)}")
                except Exception as e:
                    logger.error(f"Erro no smart matching: {e}")
            
            # 4. Análise semântica (se disponível)
            if self.use_semantic and self.semantic_analyzer and matches:
                logger.info(f"Fase 3: Análise semântica (max {self.max_semantic_checks})...")
                
                sorted_matches = sorted(matches, key=lambda x: x.match_score, reverse=True)
                
                for match in sorted_matches[:self.max_semantic_checks]:
                    try:
                        semantic_analyzed += 1
                        
                        analysis = PairAnalysis(
                            kalshi=match.kalshi,
                            polymarket=match.polymarket,
                            category=match.category,
                            event_type=match.event_type,
                            match_score=match.match_score,
                            match_reason=match.match_reason,
                        )
                        
                        # Análise semântica com timeout
                        try:
                            result = await asyncio.wait_for(
                                self.semantic_analyzer.analyze_pair(
                                    match.kalshi.title,
                                    match.polymarket.title,
                                    match.kalshi.description or "",
                                    match.polymarket.description or "",
                                ),
                                timeout=15  # 15 segundos por análise
                            )
                            
                            analysis.is_same_event = result.get("is_same_event", False)
                            analysis.semantic_confidence = result.get("confidence", 0)
                            analysis.semantic_reasoning = result.get("reasoning", "")
                            analysis.event_description = result.get("event_description", "")
                            
                            if analysis.is_same_event:
                                same_event_count += 1
                                logger.info(f"✓ MESMO EVENTO: {match.kalshi.title[:40]}...")
                                
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout análise semântica")
                        except Exception as e:
                            logger.warning(f"Erro análise: {e}")
                        
                        top_pairs.append(self._analysis_to_dict(analysis))
                        
                    except Exception as e:
                        logger.warning(f"Erro processando match: {e}")
                
                logger.info(f"Semântica: {semantic_analyzed} analisados, {same_event_count} confirmados")
            
            # Se não tem análise semântica, converter matches direto
            elif matches:
                logger.info("Convertendo matches (sem análise semântica)...")
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
            
            # Ordenar: confirmados primeiro
            top_pairs.sort(key=lambda x: (x.get("is_same_event", False), x.get("match_score", 0)), reverse=True)
            
        except Exception as e:
            logger.error(f"Erro geral no scan: {e}")
            import traceback
            traceback.print_exc()
        
        # Calcular duração
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"=== SCAN COMPLETO em {duration:.1f}s ===")
        
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
        """Coleta mercados Kalshi."""
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
                        logger.info(f"Kalshi: {len(all_markets)}...")
                    
                    if not next_cursor:
                        break
                    cursor = next_cursor
                    
                except Exception as e:
                    logger.error(f"Erro página Kalshi: {e}")
                    break
        
        return all_markets
    
    async def _collect_all_polymarket(self) -> List[Market]:
        """Coleta mercados Polymarket SEM orderbooks."""
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
                    
                    all_markets.extend(markets)
                    
                    if len(all_markets) % 500 == 0:
                        logger.info(f"Polymarket: {len(all_markets)}...")
                    
                    offset += 100
                    
                    if len(markets) < 100:
                        break
                        
                except Exception as e:
                    logger.error(f"Erro página Polymarket: {e}")
                    break
        
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
