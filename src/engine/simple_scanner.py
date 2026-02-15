"""
Scanner Simples de Oportunidades - Polymarket
=============================================

Busca oportunidades reais baseadas em:
1. Mercados com spread alto (diferença bid-ask > 3%)
2. Mercados onde YES + NO < 1 (arbitragem real)
3. Mercados com preços extremos (< 0.10 ou > 0.90)
"""

import asyncio
import aiohttp
import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SimpleOpportunity:
    """Oportunidade de arbitragem simplificada."""
    market_id: str
    market_title: str
    opportunity_type: str  # 'spread', 'arbitrage', 'value'
    
    yes_price: float
    no_price: float
    spread: float
    
    edge: float
    expected_return: float
    confidence: float
    
    recommendation: str  # 'BUY YES', 'BUY NO', 'ARBITRAGE'
    reason: str
    
    volume_24h: float
    liquidity: float
    
    returns_10: float
    returns_100: float
    returns_1000: float
    
    detected_at: datetime


class SimpleScanner:
    """
    Scanner simples que busca oportunidades REAIS no Polymarket.
    """
    
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        import ssl
        import certifi
        
        # Criar contexto SSL que aceita certificados
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=connector
        )
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def scan(self, limit: int = 100) -> List[SimpleOpportunity]:
        """
        Escaneia mercados e retorna oportunidades.
        """
        if not self.session:
            raise RuntimeError("Use como context manager")
        
        opportunities = []
        
        # 1. Buscar mercados ativos da API Gamma
        markets = await self._fetch_markets(limit)
        logger.info(f"Buscando {len(markets)} mercados...")
        
        # 2. Para cada mercado, buscar orderbook e analisar
        for i, market in enumerate(markets):
            try:
                if i % 20 == 0:
                    logger.info(f"Analisando mercado {i+1}/{len(markets)}...")
                
                opp = await self._analyze_market(market)
                if opp:
                    opportunities.append(opp)
                    
            except Exception as e:
                logger.debug(f"Erro analisando mercado: {e}")
                continue
            
            # Rate limiting
            if i % 10 == 0:
                await asyncio.sleep(0.1)
        
        # Ordenar por expected_return
        opportunities.sort(key=lambda x: x.expected_return, reverse=True)
        
        logger.info(f"Encontradas {len(opportunities)} oportunidades!")
        return opportunities
    
    async def _fetch_markets(self, limit: int) -> List[Dict]:
        """Busca mercados ativos da API Gamma."""
        try:
            url = f"{self.GAMMA_API}/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": limit,
                "order": "volume24hr",
                "ascending": "false"
            }
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"Erro ao buscar mercados: {resp.status}")
                    return []
        except Exception as e:
            logger.error(f"Erro ao buscar mercados: {e}")
            return []
    
    async def _fetch_orderbook(self, token_id: str) -> Optional[Dict]:
        """Busca orderbook do CLOB."""
        try:
            url = f"{self.CLOB_API}/book"
            params = {"token_id": token_id}
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.debug(f"Erro ao buscar orderbook: {e}")
            return None
    
    async def _fetch_price_data(self, token_id: str) -> Optional[Dict]:
        """Busca dados de preço do CLOB (mais confiável que orderbook)."""
        try:
            result = {}
            
            # Buscar midpoint
            mid_url = f"{self.CLOB_API}/midpoint?token_id={token_id}"
            async with self.session.get(mid_url) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    import json
                    data = json.loads(text)
                    mid_str = data.get("mid", "0")
                    result["midpoint"] = float(mid_str) if mid_str else 0
            
            # Buscar preço de compra (bid)
            buy_url = f"{self.CLOB_API}/price?token_id={token_id}&side=BUY"
            async with self.session.get(buy_url) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    import json
                    data = json.loads(text)
                    price_str = data.get("price", "0")
                    result["bid"] = float(price_str) if price_str else 0
            
            # Buscar preço de venda (ask)
            sell_url = f"{self.CLOB_API}/price?token_id={token_id}&side=SELL"
            async with self.session.get(sell_url) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    import json
                    data = json.loads(text)
                    price_str = data.get("price", "0")
                    result["ask"] = float(price_str) if price_str else 0
            
            if result.get("midpoint") or result.get("bid") or result.get("ask"):
                return result
            return None
        except Exception as e:
            logger.debug(f"Erro ao buscar preços: {e}")
            return None
    
    async def _analyze_market(self, market: Dict) -> Optional[SimpleOpportunity]:
        """Analisa um mercado e retorna oportunidade se houver."""
        
        # Extrair dados básicos
        market_id = market.get("conditionId", market.get("id", ""))
        market_title = market.get("question", market.get("title", "Unknown"))
        
        # Pegar tokens - clobTokenIds pode ser string JSON ou lista
        import json as json_module
        
        clob_tokens_raw = market.get("clobTokenIds", "[]")
        if isinstance(clob_tokens_raw, str):
            try:
                clob_tokens = json_module.loads(clob_tokens_raw)
            except:
                clob_tokens = []
        else:
            clob_tokens = clob_tokens_raw
        
        if len(clob_tokens) < 2:
            return None
        
        yes_token = clob_tokens[0]
        no_token = clob_tokens[1]
        
        if not yes_token or not no_token:
            return None
        
        # Buscar dados de preço (mais confiável)
        yes_prices = await self._fetch_price_data(yes_token)
        no_prices = await self._fetch_price_data(no_token)
        
        if not yes_prices:
            return None
        
        # Extrair preços do YES
        yes_bid = yes_prices.get("bid", 0)
        yes_ask = yes_prices.get("ask", 0)
        midpoint = yes_prices.get("midpoint", 0)
        
        # Se não tem midpoint, calcular
        if not midpoint and yes_bid and yes_ask:
            midpoint = (yes_bid + yes_ask) / 2
        
        # Se ainda não tem dados suficientes, pular
        if not midpoint or midpoint <= 0:
            return None
        
        # Se não tem bid/ask, usar midpoint para estimar
        if not yes_bid:
            yes_bid = max(0.001, midpoint - 0.01)
        if not yes_ask:
            yes_ask = min(0.999, midpoint + 0.01)
        
        # Calcular spread
        spread = yes_ask - yes_bid
        
        # NO price
        no_price = 1.0 - midpoint
        
        # Volume
        volume_24h = float(market.get("volume24hr", 0) or 0)
        
        # Liquidez estimada baseada em volume
        liquidity = volume_24h * 0.1  # Estimativa: 10% do volume diário
        
        # Para arbitragem, precisamos do preço do NO também
        no_book = no_prices
        
        # ========================
        # DETECTAR OPORTUNIDADES
        # ========================
        
        opportunity_type = None
        edge = 0.0
        expected_return = 0.0
        recommendation = ""
        reason = ""
        confidence = 0.0
        
        # Ignorar mercados com spread muito extremo (>90%) - sem liquidez real
        if spread > 0.90:
            return None
        
        # 1. ARBITRAGEM YES + NO < 1 (a melhor oportunidade)
        if no_book and isinstance(no_book, dict):
            no_ask = no_book.get("ask", 0)
            if no_ask and 0 < no_ask < 0.99:  # Ignorar asks de $0.99+
                total_cost = yes_ask + no_ask
                if total_cost < 0.98:  # Menos de $0.98 para garantir $1
                    opportunity_type = "arbitrage"
                    edge = 1.0 - total_cost
                    expected_return = (edge / total_cost) * 100
                    recommendation = "ARBITRAGE"
                    reason = f"Comprar YES (${yes_ask:.2f}) + NO (${no_ask:.2f}) = ${total_cost:.2f} < $1.00"
                    confidence = 0.95
        
        # 2. SPREAD RAZOÁVEL (5-30%) - Oportunidade de market making
        if not opportunity_type and 0.05 < spread < 0.30:
            opportunity_type = "spread"
            edge = spread / 2
            expected_return = edge * 100
            recommendation = "MARKET MAKE"
            reason = f"Spread de {spread*100:.1f}% - Oportunidade de market making"
            confidence = min(0.7, volume_24h / 100000) if volume_24h > 0 else 0.3
        
        # 3. PREÇO BOM PARA COMPRAR YES (0.20-0.40)
        if not opportunity_type and 0.20 < midpoint < 0.40:
            opportunity_type = "value_yes"
            edge = 0.03 + (0.40 - midpoint) * 0.1
            expected_return = edge * 100
            recommendation = "BUY YES"
            reason = f"Preço de ${midpoint:.2f} pode ter potencial de alta"
            confidence = 0.4 + min(0.3, volume_24h / 500000) if volume_24h > 0 else 0.4
        
        # 4. PREÇO BOM PARA COMPRAR NO (0.60-0.80)
        if not opportunity_type and 0.60 < midpoint < 0.80:
            opportunity_type = "value_no"
            edge = 0.03 + (midpoint - 0.60) * 0.1
            expected_return = edge * 100
            recommendation = "BUY NO"
            reason = f"Preço de ${midpoint:.2f} pode ter potencial de queda"
            confidence = 0.4 + min(0.3, volume_24h / 500000) if volume_24h > 0 else 0.4
        
        # 5. SPREAD PEQUENO MAS APROVEITÁVEL (2-5%)
        if not opportunity_type and 0.02 < spread < 0.05:
            opportunity_type = "tight_spread"
            edge = spread / 3
            expected_return = edge * 100
            recommendation = "LIMIT ORDER"
            reason = f"Spread de {spread*100:.1f}% - Aproveite com ordens limitadas"
            confidence = min(0.5, volume_24h / 50000) if volume_24h > 0 else 0.3
        
        # 6. MERCADO COM ALTO VOLUME (qualquer preço)
        if not opportunity_type and volume_24h > 500000:
            opportunity_type = "high_volume"
            edge = 0.02
            expected_return = 2.0
            recommendation = "MONITOR"
            reason = f"Alto volume (${volume_24h:,.0f}) - Mercado muito ativo"
            confidence = 0.5
        
        # Se não encontrou oportunidade
        if not opportunity_type:
            return None
        
        # Calcular retornos projetados
        returns_10 = 10 * expected_return / 100
        returns_100 = 100 * expected_return / 100
        returns_1000 = 1000 * expected_return / 100
        
        return SimpleOpportunity(
            market_id=market_id,
            market_title=market_title[:100],
            opportunity_type=opportunity_type,
            yes_price=midpoint,
            no_price=no_price,
            spread=spread,
            edge=edge,
            expected_return=expected_return,
            confidence=confidence,
            recommendation=recommendation,
            reason=reason,
            volume_24h=volume_24h,
            liquidity=liquidity,
            returns_10=returns_10,
            returns_100=returns_100,
            returns_1000=returns_1000,
            detected_at=datetime.utcnow()
        )
    
    def _get_best_bid(self, book: Dict) -> Optional[float]:
        """Extrai melhor bid do orderbook."""
        bids = book.get("bids", [])
        if bids:
            return float(bids[0].get("price", 0))
        return None
    
    def _get_best_ask(self, book: Dict) -> Optional[float]:
        """Extrai melhor ask do orderbook."""
        asks = book.get("asks", [])
        if asks:
            return float(asks[0].get("price", 0))
        return None
    
    def _calculate_liquidity(self, book: Dict) -> float:
        """Calcula liquidez total do orderbook."""
        total = 0.0
        
        for bid in book.get("bids", [])[:10]:
            total += float(bid.get("size", 0)) * float(bid.get("price", 0))
        
        for ask in book.get("asks", [])[:10]:
            total += float(ask.get("size", 0)) * float(ask.get("price", 0))
        
        return total


async def run_scan(limit: int = 100) -> List[SimpleOpportunity]:
    """Executa scan e retorna oportunidades."""
    async with SimpleScanner(timeout=60.0) as scanner:
        return await scanner.scan(limit)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    opportunities = asyncio.run(run_scan(limit))
    
    print(f"\n{'='*80}")
    print(f"  TOP OPORTUNIDADES DE ARBITRAGEM - POLYMARKET")
    print(f"{'='*80}\n")
    
    for i, opp in enumerate(opportunities[:10], 1):
        print(f"#{i} - {opp.opportunity_type.upper()}")
        print(f"   Mercado: {opp.market_title}")
        print(f"   Preço YES: ${opp.yes_price:.3f} | NO: ${opp.no_price:.3f}")
        print(f"   Spread: {opp.spread*100:.1f}%")
        print(f"   Edge: {opp.edge*100:.2f}%")
        print(f"   Retorno Esperado: {opp.expected_return:.1f}%")
        print(f"   Recomendação: {opp.recommendation}")
        print(f"   Razão: {opp.reason}")
        print(f"   Volume 24h: ${opp.volume_24h:,.0f}")
        print(f"   Confiança: {opp.confidence*100:.0f}%")
        print(f"   Retornos: $10→${opp.returns_10:.2f} | $100→${opp.returns_100:.2f} | $1000→${opp.returns_1000:.2f}")
        print()

