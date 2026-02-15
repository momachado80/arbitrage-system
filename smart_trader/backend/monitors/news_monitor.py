"""
Monitor de Breaking News
=========================
Integra com NewsAPI + web scraping de Reuters/AP/Bloomberg
Detecta notícias trending e classifica por categoria
"""
import asyncio
import aiohttp
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from models.signal import Evento, Categoria, CATEGORIA_KEYWORDS
except ImportError:
    from ..models.signal import Evento, Categoria, CATEGORIA_KEYWORDS

logger = logging.getLogger(__name__)


@dataclass
class NewsSource:
    """Configuração de fonte de notícias"""
    name: str
    api_url: str
    api_key: Optional[str] = None
    scrape_url: Optional[str] = None
    enabled: bool = True


@dataclass
class TrendingEvent:
    """Evento trending detectado"""
    evento: Evento
    strength: float = 0.0  # 0-100
    sources_count: int = 0
    velocity: float = 0.0  # Menções por minuto


class NewsMonitor:
    """
    Monitor de breaking news em tempo real.
    
    Fontes:
    - NewsAPI (100 requests/dia free)
    - Reuters RSS
    - AP News RSS
    - Google News scraping
    """
    
    # Configurações
    TRENDING_THRESHOLD = 100  # Menções mínimas para considerar trending
    VELOCITY_THRESHOLD = 10  # Menções/minuto para alerta
    CHECK_INTERVAL = 30  # Segundos entre verificações
    
    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        check_interval: int = 30
    ):
        self.newsapi_key = newsapi_key
        self.check_interval = check_interval
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache de eventos recentes
        self.recent_events: Dict[str, Evento] = {}
        self.event_mentions: Dict[str, List[datetime]] = defaultdict(list)
        
        # Callbacks para novos eventos
        self.callbacks: List[callable] = []
        
        # Running state
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def __aenter__(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SmartEventTrader/1.0)",
            "Accept": "application/json"
        }
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=headers
        )
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    def on_event(self, callback: callable):
        """Registra callback para novos eventos"""
        self.callbacks.append(callback)
    
    async def _notify_callbacks(self, event: TrendingEvent):
        """Notifica callbacks sobre novo evento"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Erro no callback: {e}")
    
    # =========================================================================
    # FONTE 1: NewsAPI
    # =========================================================================
    
    async def fetch_newsapi(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Evento]:
        """
        Busca notícias do NewsAPI.
        Free tier: 100 requests/dia, até 100 artigos/request
        """
        eventos = []
        
        if not self.newsapi_key:
            logger.warning("NewsAPI key não configurada")
            return eventos
        
        try:
            # Top headlines
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                "apiKey": self.newsapi_key,
                "language": "en",
                "pageSize": 100
            }
            
            if query:
                params["q"] = query
            if category:
                params["category"] = category
            else:
                params["category"] = "general"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("articles", [])
                    
                    for article in articles:
                        evento = Evento(
                            titulo=article.get("title", ""),
                            descricao=article.get("description", ""),
                            fonte=article.get("source", {}).get("name", "NewsAPI"),
                            timestamp=self._parse_timestamp(article.get("publishedAt")),
                            url=article.get("url", ""),
                            raw_data=article
                        )
                        
                        # Classificar categoria
                        evento.categoria = self._classify_categoria(
                            evento.titulo + " " + evento.descricao
                        )
                        
                        eventos.append(evento)
                else:
                    logger.warning(f"NewsAPI retornou {response.status}")
        
        except Exception as e:
            logger.error(f"Erro ao buscar NewsAPI: {e}")
        
        return eventos
    
    # =========================================================================
    # FONTE 2: RSS Feeds (Reuters, AP)
    # =========================================================================
    
    async def fetch_rss_feeds(self) -> List[Evento]:
        """
        Busca notícias de feeds RSS.
        Não tem limite de requests.
        """
        eventos = []
        
        feeds = [
            ("Reuters", "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best"),
            ("AP News", "https://rsshub.app/apnews/topics/apf-topnews"),
            ("BBC", "http://feeds.bbci.co.uk/news/world/rss.xml"),
        ]
        
        for name, url in feeds:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        text = await response.text()
                        eventos.extend(self._parse_rss(text, name))
            except Exception as e:
                logger.debug(f"Erro ao buscar RSS {name}: {e}")
        
        return eventos
    
    def _parse_rss(self, xml_text: str, source: str) -> List[Evento]:
        """Parse simples de RSS XML"""
        eventos = []
        
        # Parse básico de RSS (sem dependência de lxml)
        import re
        
        items = re.findall(r'<item>(.*?)</item>', xml_text, re.DOTALL)
        
        for item in items[:20]:  # Limita a 20 itens
            title_match = re.search(r'<title>(.*?)</title>', item, re.DOTALL)
            desc_match = re.search(r'<description>(.*?)</description>', item, re.DOTALL)
            link_match = re.search(r'<link>(.*?)</link>', item, re.DOTALL)
            date_match = re.search(r'<pubDate>(.*?)</pubDate>', item, re.DOTALL)
            
            title = title_match.group(1).strip() if title_match else ""
            title = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', title)
            
            desc = desc_match.group(1).strip() if desc_match else ""
            desc = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', desc)
            desc = re.sub(r'<[^>]+>', '', desc)  # Remove HTML tags
            
            link = link_match.group(1).strip() if link_match else ""
            date_str = date_match.group(1).strip() if date_match else ""
            
            if title:
                evento = Evento(
                    titulo=title,
                    descricao=desc[:500],
                    fonte=source,
                    url=link,
                    timestamp=self._parse_rss_date(date_str)
                )
                evento.categoria = self._classify_categoria(title + " " + desc)
                eventos.append(evento)
        
        return eventos
    
    # =========================================================================
    # FONTE 3: Google News (scraping)
    # =========================================================================
    
    async def fetch_google_news(self, query: str = "breaking news") -> List[Evento]:
        """
        Busca Google News via scraping.
        Útil para detectar trending topics.
        """
        eventos = []
        
        try:
            url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    
                    # Parse básico de títulos
                    titles = re.findall(r'<a[^>]*class="[^"]*DY5T1d[^"]*"[^>]*>(.*?)</a>', text)
                    
                    for title in titles[:20]:
                        title = re.sub(r'<[^>]+>', '', title)
                        if title:
                            evento = Evento(
                                titulo=title,
                                fonte="Google News",
                                timestamp=datetime.now(timezone.utc)
                            )
                            evento.categoria = self._classify_categoria(title)
                            eventos.append(evento)
        
        except Exception as e:
            logger.debug(f"Erro ao buscar Google News: {e}")
        
        return eventos
    
    # =========================================================================
    # CLASSIFICAÇÃO E ANÁLISE
    # =========================================================================
    
    def _classify_categoria(self, text: str) -> Categoria:
        """
        Classifica texto em categoria usando keywords.
        Retorna categoria com mais matches.
        """
        text_lower = text.lower()
        scores = {}
        
        for categoria, keywords in CATEGORIA_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[categoria] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return Categoria.POLITICO  # Default
    
    def _extract_keywords_matched(self, text: str, categoria: Categoria) -> List[str]:
        """Extrai keywords que fizeram match"""
        text_lower = text.lower()
        keywords = CATEGORIA_KEYWORDS.get(categoria, [])
        return [kw for kw in keywords if kw.lower() in text_lower]
    
    def _parse_timestamp(self, ts_str: Optional[str]) -> datetime:
        """Parse de timestamp ISO"""
        if not ts_str:
            return datetime.now(timezone.utc)
        
        try:
            # Remove timezone info e parse
            ts_str = ts_str.replace("Z", "+00:00")
            return datetime.fromisoformat(ts_str)
        except:
            return datetime.now(timezone.utc)
    
    def _parse_rss_date(self, date_str: str) -> datetime:
        """Parse de data RSS (RFC 2822)"""
        from email.utils import parsedate_to_datetime
        try:
            return parsedate_to_datetime(date_str)
        except:
            return datetime.now(timezone.utc)
    
    # =========================================================================
    # DETECÇÃO DE TRENDING
    # =========================================================================
    
    def _calculate_event_key(self, evento: Evento) -> str:
        """Gera chave única para evento (para dedupe)"""
        # Normaliza título
        title_normalized = evento.titulo.lower()
        title_normalized = re.sub(r'[^\w\s]', '', title_normalized)
        words = title_normalized.split()[:5]  # Primeiras 5 palavras
        return "_".join(words)
    
    def _calculate_trending_strength(self, event_key: str) -> Tuple[float, int, float]:
        """
        Calcula força de trending de um evento.
        
        Returns:
            (strength: 0-100, mentions: count, velocity: mentions/min)
        """
        mentions = self.event_mentions.get(event_key, [])
        
        if not mentions:
            return 0.0, 0, 0.0
        
        # Conta menções nos últimos 5 minutos
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        recent = [m for m in mentions if m >= cutoff]
        
        mentions_count = len(recent)
        
        # Calcula velocidade (menções por minuto)
        if len(recent) >= 2:
            time_span = (recent[-1] - recent[0]).total_seconds() / 60
            velocity = mentions_count / max(time_span, 0.1)
        else:
            velocity = mentions_count
        
        # Calcula strength (0-100)
        # 100 = 500+ menções ou 50+ menções/min
        strength = min(100, (mentions_count / 500) * 100)
        strength = max(strength, min(100, (velocity / 50) * 100))
        
        return strength, mentions_count, velocity
    
    async def _check_trending(self, eventos: List[Evento]) -> List[TrendingEvent]:
        """
        Analisa eventos e detecta trending.
        """
        trending = []
        now = datetime.now(timezone.utc)
        
        for evento in eventos:
            event_key = self._calculate_event_key(evento)
            
            # Registra menção
            self.event_mentions[event_key].append(now)
            
            # Limpa menções antigas (> 1 hora)
            cutoff = now - timedelta(hours=1)
            self.event_mentions[event_key] = [
                m for m in self.event_mentions[event_key] if m >= cutoff
            ]
            
            # Calcula trending
            strength, mentions, velocity = self._calculate_trending_strength(event_key)
            
            if strength >= 20 or mentions >= self.TRENDING_THRESHOLD:
                # Atualiza evento no cache
                if event_key not in self.recent_events:
                    self.recent_events[event_key] = evento
                else:
                    # Atualiza menções
                    self.recent_events[event_key].mentions = mentions
                
                evento.mentions = mentions
                evento.keywords_matched = self._extract_keywords_matched(
                    evento.titulo + " " + evento.descricao,
                    evento.categoria
                )
                
                trending_event = TrendingEvent(
                    evento=evento,
                    strength=strength,
                    sources_count=len(set(
                        e.fonte for e in [evento] + 
                        [self.recent_events.get(event_key)] if e
                    )),
                    velocity=velocity
                )
                trending.append(trending_event)
        
        # Ordena por strength
        trending.sort(key=lambda t: t.strength, reverse=True)
        
        return trending[:10]  # Top 10
    
    # =========================================================================
    # LOOP PRINCIPAL
    # =========================================================================
    
    async def scan_once(self) -> List[TrendingEvent]:
        """Executa um scan de todas as fontes"""
        all_eventos = []
        
        # Busca de todas as fontes em paralelo
        tasks = [
            self.fetch_newsapi(),
            self.fetch_rss_feeds(),
            self.fetch_google_news()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_eventos.extend(result)
            elif isinstance(result, Exception):
                logger.debug(f"Erro em fonte: {result}")
        
        logger.info(f"📰 Coletados {len(all_eventos)} eventos de notícias")
        
        # Detecta trending
        trending = await self._check_trending(all_eventos)
        
        # Notifica callbacks
        for event in trending:
            if event.strength >= 50:  # Só notifica se strength >= 50
                await self._notify_callbacks(event)
        
        return trending
    
    async def start(self):
        """Inicia monitoramento contínuo"""
        self._running = True
        logger.info(f"📰 News Monitor iniciado (intervalo: {self.check_interval}s)")
        
        while self._running:
            try:
                await self.scan_once()
            except Exception as e:
                logger.error(f"Erro no scan de news: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def stop(self):
        """Para monitoramento"""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("📰 News Monitor parado")
    
    def run_background(self):
        """Inicia em background"""
        self._task = asyncio.create_task(self.start())
        return self._task


# Função utilitária para testar
async def test_news_monitor():
    """Testa o monitor de notícias"""
    async with NewsMonitor() as monitor:
        trending = await monitor.scan_once()
        
        print(f"\n📰 {len(trending)} eventos trending detectados:\n")
        
        for i, t in enumerate(trending[:5], 1):
            print(f"{i}. [{t.evento.categoria.value}] {t.evento.titulo[:60]}...")
            print(f"   Fonte: {t.evento.fonte} | Strength: {t.strength:.0f}% | Velocity: {t.velocity:.1f}/min")
            print()


if __name__ == "__main__":
    asyncio.run(test_news_monitor())

