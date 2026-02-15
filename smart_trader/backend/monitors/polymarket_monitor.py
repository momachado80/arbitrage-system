"""
Monitor de Polymarket Price Action
===================================
Monitora preços, volume e momentum dos mercados do Polymarket.
"""
import asyncio
import aiohttp
import logging
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

try:
    from models.signal import MercadoPolymarket, Categoria, CATEGORIA_KEYWORDS
except ImportError:
    from ..models.signal import MercadoPolymarket, Categoria, CATEGORIA_KEYWORDS

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """Ponto de preço histórico"""
    timestamp: datetime
    price: Decimal
    volume: Decimal = Decimal("0")


@dataclass
class MarketMomentum:
    """Momentum calculado para um mercado"""
    market_id: str
    current_price: Decimal
    price_1h_ago: Decimal
    price_24h_ago: Decimal
    price_7d_avg: Decimal
    volume_1h: Decimal
    volume_24h: Decimal
    change_1h: float  # Percentual
    change_24h: float
    momentum_score: float  # 0-100
    is_trending: bool = False


class PolymarketMonitor:
    """
    Monitor de preços e volume do Polymarket.
    
    Funcionalidades:
    - Busca mercados ativos
    - Calcula momentum de preço
    - Detecta mercados trending
    - Correlaciona com eventos
    """
    
    # APIs
    CLOB_API = "https://clob.polymarket.com"
    GAMMA_API = "https://gamma-api.polymarket.com"
    
    CHECK_INTERVAL = 30  # Segundos
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache de mercados
        self.markets: Dict[str, MercadoPolymarket] = {}
        self.price_history: Dict[str, List[PricePoint]] = {}
        self.momentum_cache: Dict[str, MarketMomentum] = {}
        
        # Callbacks
        self.callbacks: List[callable] = []
        
        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(ssl=False)
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=connector,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; SmartEventTrader/1.0)",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate"
            }
        )
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    def on_momentum(self, callback: callable):
        """Registra callback para mercados com alto momentum"""
        self.callbacks.append(callback)
    
    async def _notify_callbacks(self, momentum: MarketMomentum):
        """Notifica callbacks"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(momentum)
                else:
                    callback(momentum)
            except Exception as e:
                logger.error(f"Erro no callback: {e}")
    
    # =========================================================================
    # BUSCA DE MERCADOS
    # =========================================================================
    
    async def fetch_markets(self, limit: int = 100) -> List[MercadoPolymarket]:
        """
        Busca mercados ativos do Polymarket.
        """
        mercados = []
        
        try:
            url = f"{self.CLOB_API}/markets"
            params = {"limit": limit}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    markets_data = data.get("data", []) if isinstance(data, dict) else data
                    
                    for m in markets_data:
                        if not m.get("active") or m.get("closed"):
                            continue
                        
                        try:
                            # Parse preços
                            tokens = m.get("tokens", [])
                            yes_price = Decimal("0.5")
                            no_price = Decimal("0.5")
                            
                            for token in tokens:
                                outcome = token.get("outcome", "").upper()
                                price = Decimal(str(token.get("price", "0.5")))
                                if outcome == "YES":
                                    yes_price = price
                                elif outcome == "NO":
                                    no_price = price
                            
                            # Cria mercado
                            mercado = MercadoPolymarket(
                                id=m.get("condition_id", "")[:16],
                                titulo=m.get("question", m.get("description", "Unknown")),
                                link=f"https://polymarket.com/event/{m.get('condition_id', '')}",
                                preco_atual=yes_price,
                                liquidity=Decimal(str(m.get("liquidity", 0) or 0)),
                            )
                            
                            mercados.append(mercado)
                            
                        except Exception as e:
                            logger.debug(f"Erro ao parsear mercado: {e}")
                            continue
                else:
                    logger.warning(f"CLOB API retornou {response.status}")
        
        except Exception as e:
            logger.error(f"Erro ao buscar mercados: {e}")
        
        return mercados
    
    async def fetch_market_prices(self, market_id: str) -> Optional[Dict[str, Decimal]]:
        """
        Busca preços atuais de um mercado específico.
        """
        try:
            url = f"{self.CLOB_API}/prices"
            params = {"token_ids": market_id}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = {}
                    
                    for token_id, price_data in data.items():
                        if isinstance(price_data, dict):
                            prices[token_id] = Decimal(str(price_data.get("price", 0.5)))
                        else:
                            prices[token_id] = Decimal(str(price_data))
                    
                    return prices
        
        except Exception as e:
            logger.debug(f"Erro ao buscar preços: {e}")
        
        return None
    
    # =========================================================================
    # CÁLCULO DE MOMENTUM
    # =========================================================================
    
    def _update_price_history(self, market_id: str, price: Decimal):
        """Atualiza histórico de preços de um mercado"""
        if market_id not in self.price_history:
            self.price_history[market_id] = []
        
        now = datetime.now(timezone.utc)
        self.price_history[market_id].append(PricePoint(
            timestamp=now,
            price=price
        ))
        
        # Mantém apenas últimas 24 horas
        cutoff = now - timedelta(hours=24)
        self.price_history[market_id] = [
            p for p in self.price_history[market_id]
            if p.timestamp >= cutoff
        ]
    
    def _calculate_momentum(self, market_id: str, current_price: Decimal) -> MarketMomentum:
        """
        Calcula momentum de um mercado.
        
        Momentum Score (0-100):
        - Mudança de preço 1h: peso 40%
        - Mudança de preço 24h: peso 30%
        - Desvio da média 7d: peso 30%
        """
        history = self.price_history.get(market_id, [])
        now = datetime.now(timezone.utc)
        
        # Preços em diferentes períodos
        price_1h_ago = current_price
        price_24h_ago = current_price
        prices_7d = [current_price]
        
        for point in history:
            age = now - point.timestamp
            
            if age <= timedelta(hours=1):
                price_1h_ago = point.price
            if age <= timedelta(hours=24):
                price_24h_ago = point.price
            prices_7d.append(point.price)
        
        price_7d_avg = sum(prices_7d) / len(prices_7d) if prices_7d else current_price
        
        # Calcula mudanças percentuais
        change_1h = float((current_price - price_1h_ago) / max(price_1h_ago, Decimal("0.01"))) * 100
        change_24h = float((current_price - price_24h_ago) / max(price_24h_ago, Decimal("0.01"))) * 100
        deviation_7d = float((current_price - price_7d_avg) / max(price_7d_avg, Decimal("0.01"))) * 100
        
        # Calcula score (0-100)
        # Normaliza mudanças para escala 0-100
        score_1h = min(100, abs(change_1h) * 2)  # 50% change = 100 score
        score_24h = min(100, abs(change_24h))    # 100% change = 100 score
        score_dev = min(100, abs(deviation_7d) * 2)
        
        momentum_score = (score_1h * 0.4 + score_24h * 0.3 + score_dev * 0.3)
        
        return MarketMomentum(
            market_id=market_id,
            current_price=current_price,
            price_1h_ago=price_1h_ago,
            price_24h_ago=price_24h_ago,
            price_7d_avg=price_7d_avg,
            volume_1h=Decimal("0"),  # TODO: buscar volume real
            volume_24h=Decimal("0"),
            change_1h=change_1h,
            change_24h=change_24h,
            momentum_score=momentum_score,
            is_trending=momentum_score >= 50
        )
    
    # =========================================================================
    # BUSCA POR KEYWORDS
    # =========================================================================
    
    async def search_markets_by_keywords(
        self,
        keywords: List[str],
        categoria: Optional[Categoria] = None
    ) -> List[MercadoPolymarket]:
        """
        Busca mercados que contém keywords específicas.
        Útil para correlacionar com eventos de notícias.
        """
        all_markets = await self.fetch_markets(limit=500)
        matched = []
        
        for market in all_markets:
            title_lower = market.titulo.lower()
            
            for keyword in keywords:
                if keyword.lower() in title_lower:
                    # Classifica categoria se não especificada
                    if categoria:
                        market_categoria = self._classify_market(market.titulo)
                        if market_categoria != categoria:
                            continue
                    
                    matched.append(market)
                    break
        
        return matched
    
    def _classify_market(self, title: str) -> Categoria:
        """Classifica mercado por título"""
        title_lower = title.lower()
        scores = {}
        
        for cat, keywords in CATEGORIA_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in title_lower)
            if score > 0:
                scores[cat] = score
        
        if scores:
            return max(scores, key=scores.get)
        return Categoria.POLITICO
    
    async def find_related_markets(
        self,
        event_title: str,
        event_keywords: List[str]
    ) -> List[MercadoPolymarket]:
        """
        Encontra mercados relacionados a um evento/notícia.
        """
        # Extrai entidades do título do evento
        words = event_title.lower().split()
        important_words = [
            w for w in words
            if len(w) > 3 and w not in ["will", "the", "and", "for", "with", "that", "this"]
        ]
        
        # Combina com keywords
        search_terms = list(set(important_words + [k.lower() for k in event_keywords]))[:10]
        
        return await self.search_markets_by_keywords(search_terms)
    
    # =========================================================================
    # CONVERSÃO PARA MODELO
    # =========================================================================
    
    async def get_market_details(
        self,
        market_id: str
    ) -> Optional[MercadoPolymarket]:
        """
        Obtém detalhes completos de um mercado com momentum.
        """
        try:
            # Busca preço atual
            prices = await self.fetch_market_prices(market_id)
            
            if not prices:
                return None
            
            current_price = list(prices.values())[0] if prices else Decimal("0.5")
            
            # Atualiza histórico
            self._update_price_history(market_id, current_price)
            
            # Calcula momentum
            momentum = self._calculate_momentum(market_id, current_price)
            self.momentum_cache[market_id] = momentum
            
            # Retorna ou cria mercado
            if market_id in self.markets:
                market = self.markets[market_id]
                market.preco_atual = current_price
                market.preco_24h_atras = momentum.price_24h_ago
                market.movimento_percent = momentum.change_24h
            else:
                market = MercadoPolymarket(
                    id=market_id,
                    preco_atual=current_price,
                    preco_24h_atras=momentum.price_24h_ago,
                    movimento_percent=momentum.change_24h
                )
            
            return market
        
        except Exception as e:
            logger.error(f"Erro ao obter detalhes do mercado: {e}")
            return None
    
    # =========================================================================
    # LOOP PRINCIPAL
    # =========================================================================
    
    async def scan_once(self) -> List[MarketMomentum]:
        """Executa um scan de todos os mercados"""
        trending = []
        
        # Busca mercados
        markets = await self.fetch_markets(limit=200)
        logger.info(f"📊 Monitorando {len(markets)} mercados Polymarket")
        
        for market in markets:
            # Atualiza cache
            self.markets[market.id] = market
            
            # Atualiza histórico de preços
            self._update_price_history(market.id, market.preco_atual)
            
            # Calcula momentum
            momentum = self._calculate_momentum(market.id, market.preco_atual)
            self.momentum_cache[market.id] = momentum
            
            if momentum.is_trending:
                trending.append(momentum)
                await self._notify_callbacks(momentum)
        
        # Ordena por score
        trending.sort(key=lambda m: m.momentum_score, reverse=True)
        
        if trending:
            logger.info(f"📈 {len(trending)} mercados trending detectados")
        
        return trending[:20]  # Top 20
    
    async def start(self):
        """Inicia monitoramento contínuo"""
        self._running = True
        logger.info(f"📊 Polymarket Monitor iniciado (intervalo: {self.check_interval}s)")
        
        while self._running:
            try:
                await self.scan_once()
            except Exception as e:
                logger.error(f"Erro no scan Polymarket: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def stop(self):
        """Para monitoramento"""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("📊 Polymarket Monitor parado")
    
    def run_background(self):
        """Inicia em background"""
        self._task = asyncio.create_task(self.start())
        return self._task
    
    # =========================================================================
    # MÉTODOS PÚBLICOS
    # =========================================================================
    
    def get_trending_markets(self, min_score: float = 50) -> List[MarketMomentum]:
        """Retorna mercados trending"""
        return [
            m for m in self.momentum_cache.values()
            if m.momentum_score >= min_score
        ]
    
    def get_market_momentum(self, market_id: str) -> Optional[MarketMomentum]:
        """Retorna momentum de um mercado específico"""
        return self.momentum_cache.get(market_id)


# Teste
async def test_polymarket_monitor():
    """Testa o monitor de Polymarket"""
    async with PolymarketMonitor() as monitor:
        trending = await monitor.scan_once()
        
        print(f"\n📊 {len(trending)} mercados trending:\n")
        
        for i, m in enumerate(trending[:5], 1):
            print(f"{i}. Score: {m.momentum_score:.0f}")
            print(f"   Preço atual: {float(m.current_price):.2f}")
            print(f"   Mudança 24h: {m.change_24h:+.1f}%")
            print()


if __name__ == "__main__":
    asyncio.run(test_polymarket_monitor())

