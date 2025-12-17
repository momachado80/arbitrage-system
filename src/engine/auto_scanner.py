"""
Automatic Arbitrage Scanner
============================

Sistema automatizado que:
1. Coleta mercados das duas plataformas
2. Encontra pares similares automaticamente
3. Analisa arbitragem para cada par
4. Ranqueia oportunidades por edge
5. Retorna resultados estruturados

Este é o componente que roda continuamente buscando oportunidades.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from src.collectors import KalshiCollector, PolymarketCollector
from src.engine.arbitrage_calculator import ArbitrageCalculator, RejectionReason
from src.engine.market_matcher import MarketMatcher, MarketPair, similarity_to_equivalence_score
from src.models import Market, ArbitrageOpportunity, Side

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Resultado de um scan automático."""
    timestamp: datetime
    kalshi_markets_scanned: int
    polymarket_markets_scanned: int
    pairs_analyzed: int
    opportunities_found: int
    opportunities: List[dict]
    rejections_summary: dict
    scan_duration_seconds: float
    

@dataclass
class OpportunityDetail:
    """Detalhes de uma oportunidade encontrada."""
    kalshi_ticker: str
    kalshi_title: str
    polymarket_ticker: str
    polymarket_title: str
    similarity_score: float
    match_reason: str
    equivalence_level: str  # NOVO: Nível de equivalência
    equivalence_confidence: float  # NOVO: Confiança na análise
    risks: List[str]  # NOVO: Riscos identificados
    strategy: str
    contracts: float
    total_cost: float
    guaranteed_payout: float
    gross_profit: float
    total_fees: float
    net_profit: float
    edge_percentage: float
    kalshi_vwap: float
    polymarket_vwap: float
    

class AutoScanner:
    """
    Scanner automático de arbitragem.
    
    Executa o ciclo completo:
    Coleta -> Matching -> Análise -> Ranking
    """
    
    def __init__(
        self,
        target_usd: Decimal = Decimal("20"),
        min_similarity: float = 0.15,  # Reduzido para encontrar mais pares
        min_edge: Decimal = Decimal("0.02"),
        max_markets_per_platform: int = 50,
    ):
        """
        Args:
            target_usd: Valor alvo para cada operação
            min_similarity: Similaridade mínima para considerar par
            min_edge: Edge mínimo para reportar oportunidade
            max_markets_per_platform: Máximo de mercados a buscar por plataforma
        """
        self.target_usd = target_usd
        self.min_similarity = min_similarity
        self.min_edge = min_edge
        self.max_markets = max_markets_per_platform
        
        self.matcher = MarketMatcher(min_similarity=min_similarity)
        self.calculator = ArbitrageCalculator(min_edge_threshold=min_edge)
    
    async def scan(self) -> ScanResult:
        """
        Executa um scan completo.
        
        Returns:
            ScanResult com todas as oportunidades encontradas
        """
        start_time = datetime.utcnow()
        
        logger.info("Iniciando scan automático...")
        
        # Coletar mercados
        kalshi_markets, poly_markets = await self._collect_markets()
        
        logger.info(f"Coletados: {len(kalshi_markets)} Kalshi, {len(poly_markets)} Polymarket")
        
        # Encontrar pares similares
        pairs = self.matcher.find_pairs(
            kalshi_markets, 
            poly_markets,
            top_n=30
        )
        
        logger.info(f"Pares similares encontrados: {len(pairs)}")
        
        # Analisar cada par
        opportunities = []
        rejections = {}
        
        for pair in pairs:
            result = await self._analyze_pair(pair)
            
            if isinstance(result, OpportunityDetail):
                opportunities.append(result)
                logger.info(f"✓ Oportunidade: {pair.kalshi.ticker} <-> {pair.polymarket.ticker} | Edge: {result.edge_percentage:.2f}%")
            elif isinstance(result, RejectionReason):
                code = result.code
                rejections[code] = rejections.get(code, 0) + 1
        
        # Ordenar por edge
        opportunities.sort(key=lambda x: x.edge_percentage, reverse=True)
        
        # Calcular duração
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Scan completo em {duration:.1f}s | Oportunidades: {len(opportunities)}")
        
        return ScanResult(
            timestamp=end_time,
            kalshi_markets_scanned=len(kalshi_markets),
            polymarket_markets_scanned=len(poly_markets),
            pairs_analyzed=len(pairs),
            opportunities_found=len(opportunities),
            opportunities=[self._opportunity_to_dict(o) for o in opportunities],
            rejections_summary=rejections,
            scan_duration_seconds=duration,
        )
    
    async def _collect_markets(self) -> tuple[List[Market], List[Market]]:
        """Coleta mercados das duas plataformas."""
        kalshi_markets = []
        poly_markets = []
        
        async with KalshiCollector() as kalshi:
            markets, _ = await kalshi.get_markets(limit=self.max_markets, status="open")
            
            # Buscar orderbooks para cada mercado
            for market in markets:
                try:
                    yes_book, no_book = await kalshi.get_orderbook(market.ticker)
                    market.yes_book = yes_book
                    market.no_book = no_book
                    kalshi_markets.append(market)
                except Exception as e:
                    logger.warning(f"Erro ao buscar orderbook Kalshi {market.ticker}: {e}")
        
        async with PolymarketCollector() as poly:
            markets = await poly.get_markets(limit=self.max_markets, active=True)
            
            # Buscar orderbooks
            for market in markets:
                try:
                    if market.yes_token_id:
                        yes_book = await poly.get_orderbook(market.yes_token_id)
                        if yes_book:
                            yes_book.side = Side.YES
                        market.yes_book = yes_book
                    
                    if market.no_token_id:
                        no_book = await poly.get_orderbook(market.no_token_id)
                        if no_book:
                            no_book.side = Side.NO
                        market.no_book = no_book
                    
                    poly_markets.append(market)
                except Exception as e:
                    logger.warning(f"Erro ao buscar orderbook Polymarket {market.ticker}: {e}")
        
        return kalshi_markets, poly_markets
    
    async def _analyze_pair(self, pair: MarketPair) -> OpportunityDetail | RejectionReason:
        """Analisa um par de mercados usando penalidade de risco do par."""
        # Usar penalidade de risco calculada pela análise de equivalência
        basis_risk_penalty = pair.risk_penalty
        
        # Converter para equivalence_score (inverso da penalidade)
        # Penalidade 0.005 = score 1.0, penalidade 0.05 = score 0.5
        equivalence_score = max(Decimal("0.1"), Decimal("1.0") - (basis_risk_penalty * 10))
        
        opportunity, rejection = self.calculator.calculate(
            kalshi_market=pair.kalshi,
            polymarket_market=pair.polymarket,
            target_usd=self.target_usd,
            equivalence_score=equivalence_score,
        )
        
        if opportunity:
            return OpportunityDetail(
                kalshi_ticker=pair.kalshi.ticker,
                kalshi_title=pair.kalshi.title,
                polymarket_ticker=pair.polymarket.ticker,
                polymarket_title=pair.polymarket.title,
                similarity_score=pair.similarity_score,
                match_reason=pair.match_reason,
                equivalence_level=pair.equivalence.level.value,
                equivalence_confidence=float(pair.equivalence.confidence),
                risks=pair.equivalence.risks,
                strategy=opportunity.strategy,
                contracts=float(opportunity.contracts),
                total_cost=float(opportunity.total_cost),
                guaranteed_payout=float(opportunity.guaranteed_payout),
                gross_profit=float(opportunity.gross_profit),
                total_fees=float(opportunity.total_fees),
                net_profit=float(opportunity.net_profit),
                edge_percentage=float(opportunity.edge_percentage),
                kalshi_vwap=float(opportunity.kalshi_execution.vwap),
                polymarket_vwap=float(opportunity.polymarket_execution.vwap),
            )
        else:
            return rejection
    
    def _opportunity_to_dict(self, opp: OpportunityDetail) -> dict:
        """Converte OpportunityDetail para dicionário."""
        return {
            "kalshi_ticker": opp.kalshi_ticker,
            "kalshi_title": opp.kalshi_title,
            "polymarket_ticker": opp.polymarket_ticker,
            "polymarket_title": opp.polymarket_title,
            "similarity_score": opp.similarity_score,
            "match_reason": opp.match_reason,
            "equivalence_level": opp.equivalence_level,
            "equivalence_confidence": opp.equivalence_confidence,
            "risks": opp.risks,
            "strategy": opp.strategy,
            "contracts": opp.contracts,
            "total_cost": opp.total_cost,
            "guaranteed_payout": opp.guaranteed_payout,
            "gross_profit": opp.gross_profit,
            "total_fees": opp.total_fees,
            "net_profit": opp.net_profit,
            "edge_percentage": opp.edge_percentage,
            "kalshi_vwap": opp.kalshi_vwap,
            "polymarket_vwap": opp.polymarket_vwap,
        }


async def run_continuous_scanner(
    interval_seconds: int = 60,
    callback=None
):
    """
    Executa scanner continuamente.
    
    Args:
        interval_seconds: Intervalo entre scans
        callback: Função a chamar com resultados (opcional)
    """
    scanner = AutoScanner()
    
    while True:
        try:
            result = await scanner.scan()
            
            if callback:
                callback(result)
            else:
                print(f"\n{'='*60}")
                print(f"SCAN COMPLETO - {result.timestamp.isoformat()}")
                print(f"{'='*60}")
                print(f"Mercados: {result.kalshi_markets_scanned} Kalshi, {result.polymarket_markets_scanned} Poly")
                print(f"Pares analisados: {result.pairs_analyzed}")
                print(f"Oportunidades: {result.opportunities_found}")
                print(f"Duração: {result.scan_duration_seconds:.1f}s")
                
                if result.opportunities:
                    print(f"\n🎯 TOP OPORTUNIDADES:")
                    for i, opp in enumerate(result.opportunities[:5], 1):
                        print(f"  {i}. Edge: {opp['edge_percentage']:.2f}% | "
                              f"Lucro: ${opp['net_profit']:.4f} | "
                              f"{opp['kalshi_ticker']} <-> {opp['polymarket_ticker']}")
                
                if result.rejections_summary:
                    print(f"\nRejeições: {result.rejections_summary}")
            
            await asyncio.sleep(interval_seconds)
            
        except Exception as e:
            logger.error(f"Erro no scan: {e}")
            await asyncio.sleep(interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_continuous_scanner(interval_seconds=120))
