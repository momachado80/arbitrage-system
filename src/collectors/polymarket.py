"""
Polymarket Data Collector
=========================

Coleta dados de mercados e orderbooks da Polymarket via API REST.

APIs principais:
- CLOB API: https://clob.polymarket.com (orderbooks, preços)
- Gamma API: https://gamma-api.polymarket.com (metadados de mercados)

Identificadores Polymarket:
- condition_id: ID único do mercado
- token_id: ID dos tokens YES/NO (números longos)
- market (hash): Identificador do livro de ordens

IMPORTANTE:
- Polymarket usa tokens ERC1155 no Polygon
- Cada mercado tem 2 tokens: YES e NO
- O CLOB permite consultas públicas sem autenticação
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional

import httpx

from ..models.core import (
    Market,
    MarketStatus,
    OrderBook,
    Platform,
    PriceLevel,
    Side,
)

logger = logging.getLogger(__name__)


class PolymarketCollector:
    """
    Coletor de dados da Polymarket.
    
    Usa duas APIs:
    1. Gamma API - metadados de mercados, eventos, categorias
    2. CLOB API - orderbooks em tempo real
    """
    
    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"
    
    def __init__(self, timeout: float = 30.0):
        """
        Inicializa o coletor.
        
        Args:
            timeout: Timeout para requests HTTP em segundos
        """
        self.timeout = timeout
        self._gamma_client: Optional[httpx.AsyncClient] = None
        self._clob_client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Context manager entry."""
        self._gamma_client = httpx.AsyncClient(
            base_url=self.GAMMA_URL,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        self._clob_client = httpx.AsyncClient(
            base_url=self.CLOB_URL,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._gamma_client:
            await self._gamma_client.aclose()
        if self._clob_client:
            await self._clob_client.aclose()
    
    @property
    def gamma_client(self) -> httpx.AsyncClient:
        """Retorna o cliente Gamma API."""
        if self._gamma_client is None:
            raise RuntimeError("PolymarketCollector deve ser usado como context manager")
        return self._gamma_client
    
    @property
    def clob_client(self) -> httpx.AsyncClient:
        """Retorna o cliente CLOB API."""
        if self._clob_client is None:
            raise RuntimeError("PolymarketCollector deve ser usado como context manager")
        return self._clob_client
    
    async def get_markets(
        self,
        active: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        """
        Busca lista de mercados da Gamma API.
        
        Args:
            active: Se True, retorna apenas mercados ativos
            limit: Número máximo de resultados
            offset: Offset para paginação
        
        Returns:
            Lista de mercados
        """
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
        }
        
        try:
            response = await self.gamma_client.get("/markets", params=params)
            response.raise_for_status()
            data = response.json()
            
            markets = []
            # A resposta pode ser uma lista direta ou um objeto com chave
            market_list = data if isinstance(data, list) else data.get("markets", data.get("data", []))
            
            for m in market_list:
                market = self._parse_gamma_market(m)
                if market:
                    markets.append(market)
            
            logger.info(f"Polymarket: Fetched {len(markets)} markets")
            return markets
            
        except httpx.HTTPError as e:
            logger.error(f"Polymarket Gamma API error: {e}")
            raise
    
    async def get_market_by_condition(self, condition_id: str) -> Optional[Market]:
        """
        Busca um mercado específico pelo condition_id.
        
        Args:
            condition_id: ID único do mercado na Polymarket
        
        Returns:
            Market ou None se não encontrado
        """
        try:
            response = await self.gamma_client.get(f"/markets/{condition_id}")
            response.raise_for_status()
            data = response.json()
            
            return self._parse_gamma_market(data)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Polymarket market not found: {condition_id}")
                return None
            raise
        except httpx.HTTPError as e:
            logger.error(f"Polymarket API error: {e}")
            raise
    
    async def get_orderbook(self, token_id: str) -> Optional[OrderBook]:
        """
        Busca orderbook de um token específico via CLOB API.
        
        IMPORTANTE: Cada token_id representa um lado (YES ou NO).
        Você precisa chamar esta função para cada token.
        
        Args:
            token_id: ID do token (YES ou NO)
        
        Returns:
            OrderBook para este token
        """
        try:
            response = await self.clob_client.get(
                "/book",
                params={"token_id": token_id}
            )
            response.raise_for_status()
            data = response.json()
            
            return self._parse_clob_orderbook(data, token_id)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Polymarket orderbook not found for token: {token_id}")
                return None
            raise
        except httpx.HTTPError as e:
            logger.error(f"Polymarket CLOB API error: {e}")
            raise
    
    async def get_market_with_orderbooks(
        self,
        condition_id: str
    ) -> Optional[Market]:
        """
        Busca mercado com orderbooks completos.
        
        Combina dados da Gamma API (metadados) com CLOB API (orderbooks).
        """
        market = await self.get_market_by_condition(condition_id)
        if not market:
            return None
        
        # Buscar orderbooks para YES e NO tokens
        if market.yes_token_id:
            yes_book = await self.get_orderbook(market.yes_token_id)
            if yes_book:
                yes_book.side = Side.YES
            market.yes_book = yes_book
        
        if market.no_token_id:
            no_book = await self.get_orderbook(market.no_token_id)
            if no_book:
                no_book.side = Side.NO
            market.no_book = no_book
        
        market.last_updated = datetime.utcnow()
        return market
    
    async def search_markets(self, query: str, limit: int = 20) -> list[Market]:
        """
        Busca mercados por texto.
        
        Args:
            query: Termo de busca
            limit: Número máximo de resultados
        
        Returns:
            Lista de mercados que correspondem à busca
        """
        try:
            response = await self.gamma_client.get(
                "/markets",
                params={"_q": query, "limit": limit, "active": "true"}
            )
            response.raise_for_status()
            data = response.json()
            
            markets = []
            market_list = data if isinstance(data, list) else data.get("markets", data.get("data", []))
            
            for m in market_list:
                market = self._parse_gamma_market(m)
                if market:
                    markets.append(market)
            
            return markets
            
        except httpx.HTTPError as e:
            logger.error(f"Polymarket search error: {e}")
            raise
    
    def _parse_gamma_market(self, data: dict) -> Optional[Market]:
        """
        Converte dados da Gamma API para o modelo Market.
        """
        try:
            condition_id = data.get("conditionId") or data.get("condition_id")
            if not condition_id:
                return None
            
            # Extrair token IDs
            # Polymarket pode ter estruturas diferentes
            tokens = data.get("tokens", [])
            yes_token_id = None
            no_token_id = None
            
            for token in tokens:
                outcome = token.get("outcome", "").upper()
                token_id = token.get("token_id") or token.get("tokenId")
                if outcome == "YES":
                    yes_token_id = token_id
                elif outcome == "NO":
                    no_token_id = token_id
            
            # Se tokens não vieram em array, tentar campos diretos
            if not yes_token_id:
                clob_tokens = data.get("clobTokenIds")
                if clob_tokens:
                    # Pode vir como string JSON, lista, ou outro formato
                    if isinstance(clob_tokens, str):
                        try:
                            import json
                            clob_tokens = json.loads(clob_tokens)
                        except:
                            clob_tokens = None
                    if isinstance(clob_tokens, list) and len(clob_tokens) >= 1:
                        yes_token_id = clob_tokens[0] if clob_tokens[0] and len(str(clob_tokens[0])) > 10 else None
            
            if not no_token_id:
                clob_tokens = data.get("clobTokenIds")
                if clob_tokens:
                    if isinstance(clob_tokens, str):
                        try:
                            import json
                            clob_tokens = json.loads(clob_tokens)
                        except:
                            clob_tokens = None
                    if isinstance(clob_tokens, list) and len(clob_tokens) >= 2:
                        no_token_id = clob_tokens[1] if clob_tokens[1] and len(str(clob_tokens[1])) > 10 else None
            
            # Parse status
            active = data.get("active", data.get("enableOrderBook", False))
            closed = data.get("closed", False)
            resolved = data.get("resolved", False)
            
            if resolved:
                status = MarketStatus.SETTLED
            elif closed:
                status = MarketStatus.CLOSED
            elif active:
                status = MarketStatus.OPEN
            else:
                status = MarketStatus.UNKNOWN
            
            # Parse timestamps
            def parse_time(ts) -> Optional[datetime]:
                if not ts:
                    return None
                try:
                    if isinstance(ts, str):
                        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    elif isinstance(ts, (int, float)):
                        return datetime.fromtimestamp(ts)
                except (ValueError, TypeError):
                    return None
                return None
            
            # Construir ticker legível (slug ou question truncada)
            slug = data.get("slug", "")
            question = data.get("question", data.get("title", ""))
            ticker = slug if slug else condition_id[:16]
            
            return Market(
                platform=Platform.POLYMARKET,
                ticker=ticker,
                title=question,
                description=data.get("description", ""),
                resolution_rules=data.get("resolutionSource", ""),
                resolution_source=data.get("resolutionSource", ""),
                close_time=parse_time(data.get("endDate") or data.get("end_date_iso")),
                status=status,
                condition_id=condition_id,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                raw_data=data,
            )
            
        except Exception as e:
            logger.error(f"Error parsing Polymarket market: {e}")
            return None
    
    def _parse_clob_orderbook(
        self,
        data: dict,
        token_id: str
    ) -> OrderBook:
        """
        Converte orderbook do CLOB para formato normalizado.
        
        O CLOB da Polymarket retorna:
        {
            "bids": [{"price": "0.55", "size": "100"}, ...],
            "asks": [{"price": "0.56", "size": "50"}, ...]
        }
        """
        timestamp = datetime.utcnow()
        
        def parse_levels(levels_raw: list) -> list[PriceLevel]:
            """Parse níveis de preço."""
            levels = []
            for level in levels_raw:
                try:
                    price = Decimal(str(level.get("price", "0")))
                    size = Decimal(str(level.get("size", "0")))
                    
                    # Polymarket usa preços em formato decimal (0.55 = 55 cents)
                    if size > 0:
                        levels.append(PriceLevel(price=price, size=size))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid price level: {level}, error: {e}")
            return levels
        
        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])
        
        bids = parse_levels(bids_raw)
        asks = parse_levels(asks_raw)
        
        # Ordenar: bids do maior para menor, asks do menor para maior
        bids = sorted(bids, key=lambda x: x.price, reverse=True)
        asks = sorted(asks, key=lambda x: x.price)
        
        # Side será definido pelo chamador (YES ou NO)
        return OrderBook(
            side=Side.YES,  # Placeholder, será atualizado
            bids=bids,
            asks=asks,
            timestamp=timestamp
        )


async def test_polymarket_collector():
    """Teste básico do coletor Polymarket."""
    async with PolymarketCollector() as collector:
        # Buscar alguns mercados ativos
        markets = await collector.get_markets(limit=5)
        print(f"\n=== Polymarket Markets ({len(markets)}) ===")
        
        for market in markets[:3]:
            print(f"\nTicker: {market.ticker}")
            print(f"Title: {market.title[:80]}...")
            print(f"Status: {market.status}")
            print(f"Condition ID: {market.condition_id}")
            print(f"YES Token: {market.yes_token_id}")
            print(f"NO Token: {market.no_token_id}")
            
            # Buscar orderbook do YES token
            if market.yes_token_id:
                yes_book = await collector.get_orderbook(market.yes_token_id)
                if yes_book:
                    print(f"YES Book - Best Bid: {yes_book.best_bid}, Best Ask: {yes_book.best_ask}")
                    print(f"  Spread: {yes_book.spread}")


if __name__ == "__main__":
    asyncio.run(test_polymarket_collector())
