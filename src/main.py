"""
Kalshi-Polymarket Arbitrage System
==================================

Sistema de detecção e avaliação de arbitragem entre Kalshi e Polymarket.

OBJETIVO REAL:
Identificar oportunidades de arbitragem econômica REAL, líquida de custos,
ajustada ao risco, executável operacionalmente, e robusta contra diferenças
contratuais, microestrutura e latência.

PRINCÍPIO FUNDAMENTAL:
Se houver dúvida, rejeite.
Se algo parecer bom demais, rejeite.
Se não conseguir explicar claramente o edge, rejeite.

USO:
    python -m src.main scan --target 20

    Ou como módulo:
    from src.main import ArbitrageScanner
    scanner = ArbitrageScanner()
    await scanner.scan()
"""

import asyncio
import logging
from decimal import Decimal
from typing import Optional

from .collectors import KalshiCollector, PolymarketCollector
from .engine import ArbitrageCalculator, RejectionReason
from .models import (
    ArbitrageOpportunity,
    Market,
    Platform,
    DEFAULT_MIN_SIZE_USD,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class ArbitrageScanner:
    """
    Scanner principal de arbitragem.
    
    Coordena:
    1. Coleta de dados das duas plataformas
    2. Matching de mercados equivalentes
    3. Cálculo de oportunidades
    4. Filtragem e rejeição
    """
    
    def __init__(
        self,
        target_usd: Decimal = DEFAULT_MIN_SIZE_USD,
        min_edge_percent: Decimal = Decimal("2.0"),
    ):
        """
        Inicializa o scanner.
        
        Args:
            target_usd: Valor alvo para cada operação
            min_edge_percent: Edge mínimo exigido em %
        """
        self.target_usd = target_usd
        self.min_edge_percent = min_edge_percent
        self.calculator = ArbitrageCalculator(
            min_edge_threshold=min_edge_percent / 100
        )
        
        # Cache de mercados
        self._kalshi_markets: list[Market] = []
        self._polymarket_markets: list[Market] = []
        
        # Pares mapeados manualmente (para MVP)
        # Formato: [(kalshi_ticker, polymarket_condition_id, equivalence_score), ...]
        self._manual_pairs: list[tuple[str, str, Decimal]] = []
    
    def add_market_pair(
        self,
        kalshi_ticker: str,
        polymarket_condition_id: str,
        equivalence_score: Decimal = Decimal("1.0")
    ):
        """
        Adiciona um par de mercados manualmente mapeado.
        
        Este é o método recomendado para o MVP.
        Matching automático será implementado depois.
        
        Args:
            kalshi_ticker: Ticker do mercado Kalshi
            polymarket_condition_id: Condition ID do mercado Polymarket
            equivalence_score: Score de equivalência (0-1)
                              1.0 = equivalente confirmado
                              0.9 = provavelmente equivalente
                              < 0.8 = ambíguo (penalidade alta)
        """
        self._manual_pairs.append((kalshi_ticker, polymarket_condition_id, equivalence_score))
        logger.info(f"Par adicionado: {kalshi_ticker} <-> {polymarket_condition_id} (score={equivalence_score})")
    
    async def scan(self) -> list[ArbitrageOpportunity]:
        """
        Executa scan completo de arbitragem.
        
        Returns:
            Lista de oportunidades encontradas (provavelmente vazia ou pequena)
        """
        opportunities = []
        rejections = []
        
        async with KalshiCollector() as kalshi, PolymarketCollector() as polymarket:
            for kalshi_ticker, poly_condition, equiv_score in self._manual_pairs:
                logger.info(f"\nAnalisando par: {kalshi_ticker} <-> {poly_condition}")
                
                try:
                    # Buscar mercados com orderbooks
                    kalshi_market = await kalshi.get_market_with_orderbook(kalshi_ticker)
                    poly_market = await polymarket.get_market_with_orderbooks(poly_condition)
                    
                    if not kalshi_market:
                        logger.warning(f"Mercado Kalshi não encontrado: {kalshi_ticker}")
                        continue
                    
                    if not poly_market:
                        logger.warning(f"Mercado Polymarket não encontrado: {poly_condition}")
                        continue
                    
                    # Calcular arbitragem
                    opportunity, rejection = self.calculator.calculate(
                        kalshi_market=kalshi_market,
                        polymarket_market=poly_market,
                        target_usd=self.target_usd,
                        equivalence_score=equiv_score,
                    )
                    
                    if opportunity:
                        opportunities.append(opportunity)
                        logger.info(f"✓ OPORTUNIDADE ENCONTRADA: edge={opportunity.edge_percentage:.2f}%")
                        print(self.calculator.format_opportunity(opportunity))
                    else:
                        rejections.append((kalshi_ticker, poly_condition, rejection))
                        logger.info(f"✗ Rejeitado: {rejection}")
                
                except Exception as e:
                    logger.error(f"Erro analisando {kalshi_ticker}: {e}")
                    continue
        
        # Sumário
        print(f"\n{'='*60}")
        print(f"SUMÁRIO DO SCAN")
        print(f"{'='*60}")
        print(f"Pares analisados: {len(self._manual_pairs)}")
        print(f"Oportunidades encontradas: {len(opportunities)}")
        print(f"Rejeições: {len(rejections)}")
        
        if rejections:
            print(f"\nRejeições por código:")
            codes = {}
            for _, _, r in rejections:
                if r:
                    codes[r.code] = codes.get(r.code, 0) + 1
            for code, count in sorted(codes.items(), key=lambda x: -x[1]):
                print(f"  {code}: {count}")
        
        return opportunities
    
    async def scan_single_pair(
        self,
        kalshi_ticker: str,
        polymarket_condition_id: str,
        equivalence_score: Decimal = Decimal("1.0")
    ) -> tuple[Optional[ArbitrageOpportunity], Optional[RejectionReason]]:
        """
        Analisa um único par de mercados.
        
        Útil para testes e análises pontuais.
        """
        async with KalshiCollector() as kalshi, PolymarketCollector() as polymarket:
            kalshi_market = await kalshi.get_market_with_orderbook(kalshi_ticker)
            poly_market = await polymarket.get_market_with_orderbooks(polymarket_condition_id)
            
            if not kalshi_market or not poly_market:
                return None, RejectionReason(
                    code="MARKET_NOT_FOUND",
                    message="Um ou ambos mercados não encontrados"
                )
            
            return self.calculator.calculate(
                kalshi_market=kalshi_market,
                polymarket_market=poly_market,
                target_usd=self.target_usd,
                equivalence_score=equivalence_score,
            )


async def demo_data_collection():
    """
    Demo: Coleta e exibe dados das duas plataformas.
    
    Este é o primeiro passo - verificar que conseguimos coletar dados.
    """
    print("\n" + "="*60)
    print("DEMO: Coleta de Dados")
    print("="*60)
    
    # Kalshi
    print("\n--- KALSHI ---")
    async with KalshiCollector() as kalshi:
        markets, _ = await kalshi.get_markets(limit=3)
        for m in markets:
            print(f"\nTicker: {m.ticker}")
            print(f"Title: {m.title}")
            
            yes_book, no_book = await kalshi.get_orderbook(m.ticker)
            if yes_book and yes_book.best_bid and yes_book.best_ask:
                print(f"YES: Bid={yes_book.best_bid.price} Ask={yes_book.best_ask.price} Spread={yes_book.spread}")
            if no_book and no_book.best_bid and no_book.best_ask:
                print(f"NO:  Bid={no_book.best_bid.price} Ask={no_book.best_ask.price} Spread={no_book.spread}")
    
    # Polymarket
    print("\n--- POLYMARKET ---")
    async with PolymarketCollector() as polymarket:
        markets = await polymarket.get_markets(limit=3)
        for m in markets:
            print(f"\nTicker: {m.ticker}")
            print(f"Title: {m.title[:60]}...")
            print(f"Condition ID: {m.condition_id}")
            
            if m.yes_token_id:
                yes_book = await polymarket.get_orderbook(m.yes_token_id)
                if yes_book and yes_book.best_bid and yes_book.best_ask:
                    print(f"YES: Bid={yes_book.best_bid.price} Ask={yes_book.best_ask.price} Spread={yes_book.spread}")


async def main():
    """
    Função principal.
    
    Para o MVP, executa demo de coleta de dados.
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        await demo_data_collection()
    else:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║         KALSHI-POLYMARKET ARBITRAGE SYSTEM - MVP                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Este sistema detecta oportunidades de arbitragem REAIS entre   ║
║  Kalshi e Polymarket.                                           ║
║                                                                  ║
║  PRINCÍPIO: Rejeitar a maioria das oportunidades aparentes.     ║
║  Um sistema que encontra muitas arbitragens está ERRADO.        ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  USO:                                                            ║
║                                                                  ║
║  1. Demo de coleta de dados:                                    ║
║     python -m src.main demo                                     ║
║                                                                  ║
║  2. Como módulo Python:                                         ║
║     from src.main import ArbitrageScanner                       ║
║     scanner = ArbitrageScanner(target_usd=Decimal("20"))        ║
║     scanner.add_market_pair("KALSHI_TICKER", "POLY_CONDITION")  ║
║     opportunities = await scanner.scan()                        ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  PRÓXIMOS PASSOS:                                                ║
║                                                                  ║
║  1. Identificar pares de mercados equivalentes manualmente      ║
║  2. Adicionar pares ao scanner                                  ║
║  3. Executar scan e analisar rejeições                          ║
║  4. Operar em modo paper extensivo                              ║
║  5. Calibrar parâmetros baseado em resultados                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)


if __name__ == "__main__":
    asyncio.run(main())
