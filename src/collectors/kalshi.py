"""
Kalshi Data Collector
=====================

Coleta dados de mercados e orderbooks da Kalshi via API REST.

API Base: https://api.elections.kalshi.com/trade-api/v2

Endpoints principais:
- GET /markets - Lista mercados
- GET /markets/{ticker} - Detalhes de um mercado
- GET /markets/{ticker}/orderbook - Livro de ordens

IMPORTANTE: 
- Kalshi retorna apenas BIDS (não asks)
- Em mercados binários: YES bid at X = NO ask at (100-X)
- O sistema deve fazer essa conversão internamente
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


class KalshiCollector:
    """
    Coletor de dados da Kalshi.
    
    Responsabilidades:
    1. Buscar lista de mercados ativos
    2. Buscar detalhes de mercados específicos
    3. Buscar orderbooks completos
    4. Normalizar dados para o formato canônico
    """
    
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(self, timeout: float = 30.0):
        """
        Inicializa o coletor.
        
        Args:
            timeout: Timeout para requests HTTP em segundos
        """
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Retorna o cliente HTTP, garantindo que está inicializado."""
        if self._client is None:
            raise RuntimeError(
                "KalshiCollector deve ser usado como context manager: "
                "async with KalshiCollector() as collector:"
            )
        return self._client
    
    async def get_markets(
        self,
        status: str = "open",
        limit: int = 100,
        cursor: Optional[str] = None,
        series_ticker: Optional[str] = None,
    ) -> tuple[list[Market], Optional[str]]:
        """
        Busca lista de mercados.
        
        Args:
            status: Filtro por status ("open", "closed", etc.)
            limit: Número máximo de resultados (max 1000)
            cursor: Cursor para paginação
            series_ticker: Filtrar por série específica
        
        Returns:
            Tupla (lista de mercados, próximo cursor para paginação)
        """
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        
        try:
            response = await self.client.get("/markets", params=params)
            response.raise_for_status()
            data = response.json()
            
            markets = []
            for m in data.get("markets", []):
                market = self._parse_market(m)
                if market:
                    markets.append(market)
            
            next_cursor = data.get("cursor")
            logger.info(f"Kalshi: Fetched {len(markets)} markets")
            return markets, next_cursor
            
        except httpx.HTTPError as e:
            logger.error(f"Kalshi API error fetching markets: {e}")
            raise
    
    async def get_market(self, ticker: str) -> Optional[Market]:
        """
        Busca detalhes de um mercado específico.
        
        Args:
            ticker: Ticker do mercado (ex: "KXHIGHNY-24JAN01-T60")
        
        Returns:
            Market ou None se não encontrado
        """
        try:
            response = await self.client.get(f"/markets/{ticker}")
            response.raise_for_status()
            data = response.json()
            
            market_data = data.get("market", {})
            return self._parse_market(market_data)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Kalshi market not found: {ticker}")
                return None
            raise
        except httpx.HTTPError as e:
            logger.error(f"Kalshi API error fetching market {ticker}: {e}")
            raise
    
    async def get_orderbook(
        self,
        ticker: str,
        depth: int = 100
    ) -> tuple[Optional[OrderBook], Optional[OrderBook]]:
        """
        Busca orderbook de um mercado.
        
        IMPORTANTE: Kalshi retorna apenas BIDS para YES e NO.
        Este método converte para o formato normalizado com bids E asks.
        
        Args:
            ticker: Ticker do mercado
            depth: Profundidade máxima do livro
        
        Returns:
            Tupla (YES orderbook, NO orderbook)
        """
        try:
            response = await self.client.get(
                f"/markets/{ticker}/orderbook",
                params={"depth": depth}
            )
            response.raise_for_status()
            data = response.json()
            
            orderbook_data = data.get("orderbook", {})
            
            yes_book = self._parse_orderbook(orderbook_data, Side.YES)
            no_book = self._parse_orderbook(orderbook_data, Side.NO)
            
            logger.debug(f"Kalshi orderbook for {ticker}: "
                        f"YES bids={len(yes_book.bids) if yes_book else 0}, "
                        f"NO bids={len(no_book.bids) if no_book else 0}")
            
            return yes_book, no_book
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Kalshi orderbook not found: {ticker}")
                return None, None
            raise
        except httpx.HTTPError as e:
            logger.error(f"Kalshi API error fetching orderbook {ticker}: {e}")
            raise
    
    async def get_market_with_orderbook(self, ticker: str) -> Optional[Market]:
        """
        Busca mercado com orderbook já populado.
        
        Combina get_market e get_orderbook em uma operação.
        """
        market = await self.get_market(ticker)
        if not market:
            return None
        
        yes_book, no_book = await self.get_orderbook(ticker)
        market.yes_book = yes_book
        market.no_book = no_book
        market.last_updated = datetime.utcnow()
        
        return market
    
    def _parse_market(self, data: dict) -> Optional[Market]:
        """
        Converte dados da API Kalshi para o modelo Market.
        """
        try:
            ticker = data.get("ticker")
            if not ticker:
                return None
            
            # Parse status
            status_str = data.get("status", "unknown")
            status_map = {
                "open": MarketStatus.OPEN,
                "closed": MarketStatus.CLOSED,
                "settled": MarketStatus.SETTLED,
                "halted": MarketStatus.HALTED,
            }
            status = status_map.get(status_str, MarketStatus.UNKNOWN)
            
            # Parse timestamps
            def parse_time(ts_str: Optional[str]) -> Optional[datetime]:
                if not ts_str:
                    return None
                try:
                    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    return None
            
            return Market(
                platform=Platform.KALSHI,
                ticker=ticker,
                title=data.get("title", ""),
                description=data.get("subtitle", ""),
                resolution_rules=data.get("rules_primary", ""),
                resolution_source=data.get("settlement_source_url", ""),
                open_time=parse_time(data.get("open_time")),
                close_time=parse_time(data.get("close_time")),
                expiration_time=parse_time(data.get("expiration_time")),
                status=status,
                raw_data=data,
            )
            
        except Exception as e:
            logger.error(f"Error parsing Kalshi market: {e}")
            return None
    
    def _parse_orderbook(
        self,
        data: dict,
        side: Side
    ) -> OrderBook:
        """
        Converte orderbook da Kalshi para formato normalizado.
        
        CONVERSÃO CRÍTICA:
        - Kalshi retorna: yes_bids, no_bids (em dólares)
        - Para YES: bids são os yes_bids, asks são derivados de no_bids
        - Para NO: bids são os no_bids, asks são derivados de yes_bids
        
        Matematicamente:
        - YES ask at price X = NO bid at price (1-X)
        - NO ask at price Y = YES bid at price (1-Y)
        """
        timestamp = datetime.utcnow()
        
        # Kalshi retorna yes_dollars e no_dollars como [[price_str, qty], ...]
        # Preços já vêm em dólares (ex: "0.5600" = 56 cents)
        yes_levels_raw = data.get("yes_dollars", []) or data.get("yes", [])
        no_levels_raw = data.get("no_dollars", []) or data.get("no", [])
        
        def parse_levels(raw_levels: list) -> list[PriceLevel]:
            """Parse níveis de preço."""
            levels = []
            for level in raw_levels:
                if len(level) >= 2:
                    try:
                        # Pode vir como string ou número
                        price = Decimal(str(level[0]))
                        size = Decimal(str(level[1]))
                        
                        # Se preço veio em centavos (ex: 56), converter para dólares
                        if price > 1:
                            price = price / 100
                        
                        if size > 0:
                            levels.append(PriceLevel(price=price, size=size))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid price level: {level}, error: {e}")
            return levels
        
        yes_bids = parse_levels(yes_levels_raw)
        no_bids = parse_levels(no_levels_raw)
        
        if side == Side.YES:
            # Para YES orderbook:
            # - Bids são os yes_bids (ordenados do maior para menor)
            # - Asks são derivados de no_bids: se alguém quer comprar NO a X,
            #   isso é equivalente a vender YES a (1-X)
            bids = sorted(yes_bids, key=lambda x: x.price, reverse=True)
            
            # Converter no_bids em yes_asks
            asks = []
            for no_bid in no_bids:
                implied_yes_ask = Decimal("1") - no_bid.price
                asks.append(PriceLevel(price=implied_yes_ask, size=no_bid.size))
            asks = sorted(asks, key=lambda x: x.price)  # Menor para maior
            
        else:  # Side.NO
            # Para NO orderbook:
            # - Bids são os no_bids
            # - Asks são derivados de yes_bids
            bids = sorted(no_bids, key=lambda x: x.price, reverse=True)
            
            asks = []
            for yes_bid in yes_bids:
                implied_no_ask = Decimal("1") - yes_bid.price
                asks.append(PriceLevel(price=implied_no_ask, size=yes_bid.size))
            asks = sorted(asks, key=lambda x: x.price)
        
        return OrderBook(
            side=side,
            bids=bids,
            asks=asks,
            timestamp=timestamp
        )


async def test_kalshi_collector():
    """Teste básico do coletor Kalshi."""
    async with KalshiCollector() as collector:
        # Buscar alguns mercados abertos
        markets, cursor = await collector.get_markets(limit=5)
        print(f"\n=== Kalshi Markets ({len(markets)}) ===")
        
        for market in markets[:3]:
            print(f"\nTicker: {market.ticker}")
            print(f"Title: {market.title}")
            print(f"Status: {market.status}")
            
            # Buscar orderbook
            yes_book, no_book = await collector.get_orderbook(market.ticker)
            
            if yes_book:
                print(f"YES Book - Best Bid: {yes_book.best_bid}, Best Ask: {yes_book.best_ask}")
                print(f"  Spread: {yes_book.spread}")
            
            if no_book:
                print(f"NO Book - Best Bid: {no_book.best_bid}, Best Ask: {no_book.best_ask}")


if __name__ == "__main__":
    asyncio.run(test_kalshi_collector())
