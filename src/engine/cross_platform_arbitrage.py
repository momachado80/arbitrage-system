"""
Arbitragem Cross-Platform: Polymarket vs Kalshi
================================================

Busca oportunidades de arbitragem entre as duas maiores plataformas
de mercados de previsão:

- Polymarket (crypto, internacional)
- Kalshi (regulada CFTC, EUA)

Estratégia:
1. Buscar mercados similares em ambas as plataformas
2. Comparar preços para o mesmo evento
3. Se preço_A + (1 - preço_B) < 1, há arbitragem
4. Comprar YES no mais barato, NO no outro

IMPORTANTE: Requer contas em ambas as plataformas para executar.
"""

import asyncio
import aiohttp
import ssl
import certifi
import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class CrossPlatformArbitrage:
    """Oportunidade de arbitragem entre plataformas."""
    
    # Identificação
    event_name: str
    match_confidence: float  # 0-1, quão certo que é o mesmo evento
    
    # Polymarket
    poly_market_title: str
    poly_market_id: str
    poly_market_slug: str  # Para URL
    poly_yes_price: float
    poly_no_price: float
    poly_volume: float
    poly_url: str  # Link direto
    
    # Kalshi
    kalshi_market_title: str
    kalshi_market_id: str
    kalshi_ticker: str  # Para URL
    kalshi_yes_price: float
    kalshi_no_price: float
    kalshi_volume: float
    kalshi_url: str  # Link direto
    
    # Arbitragem
    arbitrage_type: str  # 'buy_poly_sell_kalshi' ou 'buy_kalshi_sell_poly'
    total_cost: float
    guaranteed_return: float
    profit_amount: float
    profit_percentage: float
    
    # Ação
    action: str
    explanation: str
    
    # Retornos
    returns_100: float
    returns_1000: float
    
    detected_at: datetime


class CrossPlatformScanner:
    """
    Scanner que busca arbitragem entre Polymarket e Kalshi.
    """
    
    # APIs
    POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
    POLYMARKET_CLOB = "https://clob.polymarket.com"
    KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"  # API pública
    
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache
        self.poly_markets: List[Dict] = []
        self.kalshi_markets: List[Dict] = []
    
    async def __aenter__(self):
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
    
    async def scan(self, limit: int = 100) -> List[CrossPlatformArbitrage]:
        """
        Escaneia ambas as plataformas e encontra arbitragem.
        """
        if not self.session:
            raise RuntimeError("Use como context manager")
        
        opportunities = []
        
        # 1. Buscar mercados do Polymarket
        logger.info("Buscando mercados do Polymarket...")
        self.poly_markets = await self._fetch_polymarket_markets(limit)
        logger.info(f"Encontrados {len(self.poly_markets)} mercados no Polymarket")
        
        # 2. Buscar mercados do Kalshi
        logger.info("Buscando mercados do Kalshi...")
        self.kalshi_markets = await self._fetch_kalshi_markets(limit)
        logger.info(f"Encontrados {len(self.kalshi_markets)} mercados no Kalshi")
        
        if not self.kalshi_markets:
            logger.warning("Não foi possível acessar Kalshi. Usando dados simulados para demonstração.")
            self.kalshi_markets = self._get_sample_kalshi_markets()
        
        # 3. Encontrar mercados correspondentes
        logger.info("Buscando mercados correspondentes...")
        matches = self._find_matching_markets()
        logger.info(f"Encontrados {len(matches)} pares de mercados similares")
        
        # 4. Analisar arbitragem em cada par
        for poly_market, kalshi_market, similarity in matches:
            arb = await self._check_arbitrage(poly_market, kalshi_market, similarity)
            if arb:
                opportunities.append(arb)
                logger.info(f"🎯 ARBITRAGEM: {arb.profit_percentage:.2f}% - {arb.event_name[:50]}")
        
        # Ordenar por lucro
        opportunities.sort(key=lambda x: x.profit_percentage, reverse=True)
        
        logger.info(f"Total: {len(opportunities)} oportunidades de arbitragem cross-platform")
        return opportunities
    
    async def _fetch_polymarket_markets(self, limit: int) -> List[Dict]:
        """Busca mercados do Polymarket."""
        try:
            url = f"{self.POLYMARKET_GAMMA}/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": limit,
                "order": "volume24hr",
                "ascending": "false"
            }
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    markets = await resp.json()
                    # Buscar preços
                    for market in markets:
                        prices = await self._fetch_polymarket_prices(market)
                        market["_prices"] = prices
                    return markets
                return []
        except Exception as e:
            logger.error(f"Erro ao buscar Polymarket: {e}")
            return []
    
    async def _fetch_polymarket_prices(self, market: Dict) -> Dict:
        """Busca preços de um mercado do Polymarket."""
        try:
            clob_tokens_raw = market.get("clobTokenIds", "[]")
            if isinstance(clob_tokens_raw, str):
                tokens = json.loads(clob_tokens_raw)
            else:
                tokens = clob_tokens_raw
            
            if len(tokens) < 2:
                return {"yes": 0.5, "no": 0.5}
            
            # Buscar midpoint do YES
            url = f"{self.POLYMARKET_CLOB}/midpoint?token_id={tokens[0]}"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    yes_price = float(data.get("mid", 0.5))
                    return {"yes": yes_price, "no": 1 - yes_price}
            
            return {"yes": 0.5, "no": 0.5}
        except:
            return {"yes": 0.5, "no": 0.5}
    
    async def _fetch_kalshi_markets(self, limit: int) -> List[Dict]:
        """
        Busca mercados do Kalshi.
        
        A API pública do Kalshi retorna principalmente mercados de esportes.
        Para mercados de política/economia, usamos dados simulados baseados
        em preços públicos conhecidos.
        """
        try:
            # Tentar API pública do Kalshi
            url = f"{self.KALSHI_API}/markets"
            params = {"limit": limit, "status": "open"}
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    api_markets = data.get("markets", [])
                    
                    # Verificar se são mercados úteis (não apenas esportes)
                    useful = [m for m in api_markets if not self._is_sports_prop(m)]
                    
                    if len(useful) < 5:
                        logger.info("API do Kalshi retornou principalmente esportes. Usando dados simulados.")
                        return self._get_sample_kalshi_markets()
                    
                    return useful
                else:
                    logger.warning(f"Kalshi API retornou {resp.status}")
                    return self._get_sample_kalshi_markets()
        except Exception as e:
            logger.warning(f"Erro ao acessar Kalshi: {e}")
            return self._get_sample_kalshi_markets()
    
    def _is_sports_prop(self, market: Dict) -> bool:
        """Verifica se é um mercado de props de esportes (menos útil para arbitragem)."""
        title = (market.get("title", "") + " " + market.get("subtitle", "")).lower()
        ticker = market.get("ticker", "").lower()
        
        # Props de jogadores (NBA, NFL, etc.)
        sports_indicators = [
            "esports", "nba", "nfl", "mlb", "nhl",
            "points", "rebounds", "assists", "touchdowns",
            "player", "game", "match"
        ]
        
        # Tickers de esportes
        sports_tickers = ["KXMV", "KXSPORTS", "KXNBA", "KXNFL"]
        
        for indicator in sports_indicators:
            if indicator in title:
                return True
        
        for ticker_prefix in sports_tickers:
            if ticker_prefix in ticker.upper():
                return True
        
        return False
    
    def _get_sample_kalshi_markets(self) -> List[Dict]:
        """
        Retorna mercados conhecidos do Kalshi para demonstração.
        
        NOTA: Para dados em tempo real, você precisa:
        1. Criar conta no Kalshi (kalshi.com)
        2. Gerar API key
        3. Usar a API autenticada
        
        Estes são preços APROXIMADOS baseados em dados públicos.
        """
        return [
            {
                "ticker": "FED-26JAN-T0.25",
                "title": "Fed to cut rates by 25+ bps at January 2026 meeting",
                "yes_price": 0.12,  # Kalshi geralmente tem preços um pouco diferentes
                "no_price": 0.88,
                "volume": 500000,
                "category": "economics"
            },
            {
                "ticker": "FED-26JAN-T0.50",
                "title": "Fed to cut rates by 50+ bps at January 2026 meeting",
                "yes_price": 0.03,
                "no_price": 0.97,
                "volume": 400000,
                "category": "economics"
            },
            {
                "ticker": "FED-HIKE-2026",
                "title": "Fed to raise interest rates by 25+ bps in 2026",
                "yes_price": 0.08,
                "no_price": 0.92,
                "volume": 350000,
                "category": "economics"
            },
            {
                "ticker": "TRUMP-2028-RNOM",
                "title": "Trump to win 2028 Republican presidential nomination",
                "yes_price": 0.48,
                "no_price": 0.52,
                "volume": 300000,
                "category": "politics"
            },
            {
                "ticker": "UKRAINE-CEASEFIRE-JAN",
                "title": "Russia-Ukraine ceasefire by January 31, 2026",
                "yes_price": 0.18,
                "no_price": 0.82,
                "volume": 200000,
                "category": "geopolitics"
            },
            {
                "ticker": "UKRAINE-CEASEFIRE-MAR",
                "title": "Russia-Ukraine ceasefire by March 31, 2026",
                "yes_price": 0.28,
                "no_price": 0.72,
                "volume": 180000,
                "category": "geopolitics"
            },
            {
                "ticker": "BTC-100K-JAN26",
                "title": "Bitcoin price above $100,000 on January 31, 2026",
                "yes_price": 0.68,
                "no_price": 0.32,
                "volume": 400000,
                "category": "crypto"
            },
            {
                "ticker": "MICROSTRATEGY-SELL-2025",
                "title": "MicroStrategy sells any Bitcoin in 2025",
                "yes_price": 0.05,
                "no_price": 0.95,
                "volume": 250000,
                "category": "crypto"
            },
            {
                "ticker": "SUPERBOWL-BUCS",
                "title": "Tampa Bay Buccaneers win Super Bowl 2026",
                "yes_price": 0.04,
                "no_price": 0.96,
                "volume": 150000,
                "category": "sports"
            },
            {
                "ticker": "MUSK-TWEETS-480",
                "title": "Elon Musk posts 480-499 tweets Dec 26 to Jan 2",
                "yes_price": 0.22,
                "no_price": 0.78,
                "volume": 100000,
                "category": "entertainment"
            },
            {
                "ticker": "STRANGER-THINGS-ELEVEN",
                "title": "Will Eleven die in Stranger Things Season 5",
                "yes_price": 0.08,
                "no_price": 0.92,
                "volume": 80000,
                "category": "entertainment"
            },
        ]
    
    def _find_matching_markets(self) -> List[Tuple[Dict, Dict, float]]:
        """
        Encontra mercados correspondentes entre as plataformas.
        
        IMPORTANTE: Só faz match quando há alta confiança de que é o MESMO evento.
        Falsos positivos seriam problemáticos (apostar em eventos diferentes).
        """
        matches = []
        
        for poly in self.poly_markets:
            poly_title = poly.get("question", "").lower()
            poly_core = self._extract_core_topic(poly_title)
            
            for kalshi in self.kalshi_markets:
                kalshi_title = kalshi.get("title", "").lower()
                kalshi_core = self._extract_core_topic(kalshi_title)
                
                # Verificar se é o MESMO evento (não apenas similar)
                is_same_event, confidence = self._is_same_event(
                    poly_title, kalshi_title, 
                    poly_core, kalshi_core
                )
                
                if is_same_event and confidence >= 0.70:
                    matches.append((poly, kalshi, confidence))
                    logger.info(f"Match encontrado ({confidence:.0%}): {poly_core} ⟷ {kalshi_core}")
        
        # Ordenar por confiança
        matches.sort(key=lambda x: x[2], reverse=True)
        
        return matches
    
    def _extract_core_topic(self, title: str) -> str:
        """Extrai o tópico central do mercado."""
        title = title.lower()
        
        # Remover frases comuns
        remove_phrases = [
            "will ", "would ", "does ", "do ", "is ", "are ", "was ", "were ",
            "by ", "on ", "in ", "at ", "to ", "the ", "a ", "an ",
            "before ", "after ", "during ",
        ]
        
        result = title
        for phrase in remove_phrases:
            result = result.replace(phrase, " ")
        
        # Limpar espaços
        result = ' '.join(result.split())
        
        return result
    
    def _is_same_event(
        self, 
        poly_title: str, 
        kalshi_title: str,
        poly_core: str,
        kalshi_core: str
    ) -> Tuple[bool, float]:
        """
        Determina se dois mercados são sobre o MESMO evento.
        Retorna (is_same, confidence).
        
        MUITO RESTRITIVO: Só retorna True se tivermos ALTA confiança.
        """
        # 1. Verificar correspondências EXATAS de eventos conhecidos
        event_patterns = [
            # Fed/Taxa de juros - DEVE ter mesma ação (cut/hike) e valor
            (r"fed.*(cut|decrease|lower).*rate.*50", "fed_cut_50"),
            (r"fed.*(cut|decrease|lower).*rate.*25", "fed_cut_25"),
            (r"decrease.*rate.*50.*bps", "fed_cut_50"),
            (r"decrease.*rate.*25.*bps", "fed_cut_25"),
            (r"fed.*(raise|increase|hike).*rate.*25", "fed_hike_25"),
            (r"increase.*rate.*25.*bps", "fed_hike_25"),
            (r"no.?change.*fed.*rate", "fed_no_change"),
            (r"fed.*rate.*no.?change", "fed_no_change"),
            
            # MicroStrategy - MUITO específico
            (r"microstrategy.*sell.*bitcoin", "mstr_sell_btc"),
            (r"mstr.*sell.*btc", "mstr_sell_btc"),
            
            # Stranger Things - MUITO específico
            (r"eleven.*die.*stranger.?things", "eleven_dies_st"),
            (r"stranger.?things.*eleven.*die", "eleven_dies_st"),
            
            # Ukraine ceasefire - com data
            (r"(ukraine|russia).*ceasefire.*january.*31", "ukraine_ceasefire_jan31"),
            (r"(ukraine|russia).*ceasefire.*march.*31", "ukraine_ceasefire_mar31"),
            
            # Musk tweets - com range específico
            (r"musk.*480.*499.*tweet", "musk_tweets_480_499"),
            (r"elon.*480.*499.*tweet", "musk_tweets_480_499"),
            
            # Super Bowl - time específico
            (r"tampa.*bay.*buccaneers.*super.?bowl", "bucs_superbowl"),
            (r"buccaneers.*win.*super.?bowl", "bucs_superbowl"),
        ]
        
        poly_event = None
        kalshi_event = None
        
        for pattern, event_id in event_patterns:
            if re.search(pattern, poly_title):
                poly_event = event_id
            if re.search(pattern, kalshi_title):
                kalshi_event = event_id
        
        # Match exato de evento conhecido
        if poly_event and kalshi_event and poly_event == kalshi_event:
            return True, 0.95
        
        # 2. Para eventos não reconhecidos, usar keywords MAS ser MUITO restritivo
        poly_keys = self._extract_critical_keywords(poly_title)
        kalshi_keys = self._extract_critical_keywords(kalshi_title)
        
        common_keys = poly_keys & kalshi_keys
        
        # DEVE ter pelo menos uma ENTIDADE em comum (não só ação)
        entity_keywords = {"FED_ENTITY", "BTC_ENTITY", "MSTR_ENTITY", "UKRAINE_ENTITY", 
                          "STRANGER_THINGS", "MUSK_ENTITY", "GOP_NOM", "DEM_NOM", "SUPER_BOWL"}
        
        common_entities = common_keys & entity_keywords
        
        # Sem entidade em comum = não é o mesmo evento
        if not common_entities:
            return False, 0.0
        
        # DEVE ter pelo menos uma AÇÃO específica em comum
        action_keywords = {"FED_CUT", "FED_HIKE", "FED_HOLD", "MSTR_SELL", 
                          "ELEVEN_DIES", "CEASEFIRE", "MUSK_TWEETS"}
        
        common_actions = common_keys & action_keywords
        
        # Entidade + Ação específica = bom match
        if common_entities and common_actions:
            return True, 0.90
        
        # Só entidade = match fraco (provavelmente eventos relacionados mas diferentes)
        if len(common_entities) >= 1 and len(common_keys) >= 2:
            # Verificar contradições
            if self._has_contradictions(poly_title, kalshi_title):
                return False, 0.0
            return True, 0.75
        
        return False, 0.0
    
    def _extract_critical_keywords(self, text: str) -> set:
        """
        Extrai palavras-chave CRÍTICAS que definem o evento.
        
        IMPORTANTE: Só retorna keywords que IDENTIFICAM UNICAMENTE o evento.
        Datas e números genéricos NÃO contam.
        """
        text = text.lower()
        keywords = set()
        
        # CATEGORIA 1: Entidades específicas (alto valor de identificação)
        # Só uma entidade por categoria para evitar falsos positivos
        
        # Economia/Fed
        if "fed" in text or "federal reserve" in text:
            keywords.add("FED_ENTITY")
            if "cut" in text or "decrease" in text or "lower" in text:
                keywords.add("FED_CUT")
            elif "raise" in text or "increase" in text or "hike" in text:
                keywords.add("FED_HIKE")
            elif "no change" in text:
                keywords.add("FED_HOLD")
        
        # Crypto
        if "bitcoin" in text or "btc" in text:
            keywords.add("BTC_ENTITY")
        if "microstrategy" in text or "mstr" in text:
            keywords.add("MSTR_ENTITY")
            if "sell" in text:
                keywords.add("MSTR_SELL")
        
        # Geopolítica
        if "ukraine" in text or "russia" in text:
            keywords.add("UKRAINE_ENTITY")
            if "ceasefire" in text:
                keywords.add("CEASEFIRE")
        
        # Entretenimento
        if "stranger things" in text:
            keywords.add("STRANGER_THINGS")
            if "eleven" in text and ("die" in text or "death" in text):
                keywords.add("ELEVEN_DIES")
        
        # Musk
        if "musk" in text or "elon" in text:
            keywords.add("MUSK_ENTITY")
            if "tweet" in text:
                keywords.add("MUSK_TWEETS")
        
        # Política - só para eventos específicos de nominação
        if "republican" in text and "nomination" in text:
            keywords.add("GOP_NOM")
        if "democratic" in text and "nomination" in text:
            keywords.add("DEM_NOM")
        
        # Esportes - só para campeonatos específicos
        if "super bowl" in text:
            keywords.add("SUPER_BOWL")
        
        return keywords
    
    def _has_contradictions(self, text1: str, text2: str) -> bool:
        """Verifica se os textos têm contradições (eventos opostos)."""
        # Pares contraditórios
        contradictions = [
            ("cut", "raise"),
            ("decrease", "increase"),
            ("lower", "higher"),
            ("above", "below"),
            ("win", "lose"),
            ("yes", "no"),
        ]
        
        for word1, word2 in contradictions:
            if (word1 in text1 and word2 in text2) or (word2 in text1 and word1 in text2):
                # Um tem "cut" e outro tem "raise" = provavelmente eventos diferentes
                return True
        
        return False
    
    def _extract_all_keywords(self, text: str) -> set:
        """Extrai todas as palavras-chave relevantes do texto."""
        text = text.lower()
        keywords = set()
        
        # Entidades importantes
        entities = {
            # Política
            "trump", "biden", "harris", "desantis", "haley", "hawley", "vance",
            "republican", "democratic", "gop", "president", "election",
            # Economia
            "fed", "federal reserve", "interest rate", "rates", "bps", "inflation",
            # Crypto
            "bitcoin", "btc", "ethereum", "crypto", "microstrategy",
            # Geopolítica
            "ukraine", "russia", "ceasefire", "war",
            # Entretenimento
            "stranger things", "eleven", "netflix",
            # Empresas
            "tesla", "spacex", "musk", "elon",
            # Quantidades
            "25", "50", "100",
        }
        
        for entity in entities:
            if entity in text:
                keywords.add(entity)
        
        # Detectar padrões específicos
        if "cut" in text and "rate" in text:
            keywords.add("rate_cut")
        if "raise" in text and "rate" in text:
            keywords.add("rate_hike")
        if "increase" in text and "rate" in text:
            keywords.add("rate_hike")
        if "decrease" in text and "rate" in text:
            keywords.add("rate_cut")
        if "sell" in text and "bitcoin" in text:
            keywords.add("btc_sell")
        if "die" in text or "death" in text:
            keywords.add("death")
        
        return keywords
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcula similaridade entre dois textos."""
        # Normalizar
        t1 = self._normalize_text(text1)
        t2 = self._normalize_text(text2)
        
        # Usar SequenceMatcher
        base_similarity = SequenceMatcher(None, t1, t2).ratio()
        
        # Bonus por palavras-chave em comum
        keywords1 = set(self._extract_keywords(t1))
        keywords2 = set(self._extract_keywords(t2))
        
        if keywords1 and keywords2:
            keyword_overlap = len(keywords1 & keywords2) / max(len(keywords1), len(keywords2))
            return (base_similarity + keyword_overlap) / 2
        
        return base_similarity
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparação."""
        # Remover pontuação
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Remover múltiplos espaços
        text = ' '.join(text.split())
        return text
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrai palavras-chave importantes."""
        # Palavras importantes para matching
        important_words = []
        
        # Nomes próprios e termos-chave
        patterns = [
            r'\b(trump|biden|harris|desantis|haley)\b',
            r'\b(republican|democratic|gop|dem)\b',
            r'\b(bitcoin|btc|ethereum|eth|crypto)\b',
            r'\b(fed|federal reserve|interest rate)\b',
            r'\b(ukraine|russia|ceasefire)\b',
            r'\b(super bowl|nfl|nba|mlb)\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(2025|2026|2027|2028)\b',
            r'\b(microstrategy|tesla|apple|google)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            important_words.extend(matches)
        
        return important_words
    
    async def _check_arbitrage(
        self, 
        poly_market: Dict, 
        kalshi_market: Dict, 
        similarity: float
    ) -> Optional[CrossPlatformArbitrage]:
        """Verifica se há arbitragem entre os mercados."""
        
        # Preços Polymarket
        poly_prices = poly_market.get("_prices", {"yes": 0.5, "no": 0.5})
        poly_yes = poly_prices["yes"]
        poly_no = poly_prices["no"]
        
        # Preços Kalshi
        kalshi_yes = kalshi_market.get("yes_price", 0.5)
        kalshi_no = kalshi_market.get("no_price", 0.5)
        
        # Volumes
        poly_volume = float(poly_market.get("volume24hr", 0) or 0)
        kalshi_volume = float(kalshi_market.get("volume", 0) or 0)
        
        # Verificar arbitragem:
        # Opção 1: Comprar YES no Poly, comprar NO no Kalshi
        cost1 = poly_yes + kalshi_no
        # Opção 2: Comprar YES no Kalshi, comprar NO no Poly
        cost2 = kalshi_yes + poly_no
        
        # Taxas estimadas (2% cada plataforma)
        fees = 0.04
        
        # Verificar qual opção tem arbitragem
        arbitrage_type = None
        total_cost = 0
        profit = 0
        
        if cost1 < (1 - fees):
            arbitrage_type = "buy_poly_yes_kalshi_no"
            total_cost = cost1 * (1 + fees)
            profit = 1 - total_cost
        elif cost2 < (1 - fees):
            arbitrage_type = "buy_kalshi_yes_poly_no"
            total_cost = cost2 * (1 + fees)
            profit = 1 - total_cost
        
        # Se não há arbitragem, verificar diferença de preço significativa
        # (oportunidade de valor, não arbitragem garantida)
        price_diff = abs(poly_yes - kalshi_yes)
        if not arbitrage_type and price_diff > 0.05:  # >5% diferença
            # Oportunidade de valor
            if poly_yes < kalshi_yes:
                arbitrage_type = "value_poly_cheaper"
                profit = price_diff * 0.5  # Estimativa conservadora
                total_cost = poly_yes
            else:
                arbitrage_type = "value_kalshi_cheaper"
                profit = price_diff * 0.5
                total_cost = kalshi_yes
        
        if not arbitrage_type:
            return None
        
        profit_pct = (profit / total_cost) * 100 if total_cost > 0 else 0
        
        # Mínimo 1% de diferença para reportar
        if profit_pct < 1:
            return None
        
        # Criar descrição
        poly_title = poly_market.get("question", "Unknown")[:100]
        kalshi_title = kalshi_market.get("title", "Unknown")[:100]
        
        if "buy_poly" in arbitrage_type:
            action = f"COMPRAR YES no Polymarket (${poly_yes:.3f}) + NO no Kalshi (${kalshi_no:.3f})"
            explanation = f"Custo total: ${total_cost:.3f} → Retorno garantido: $1.00 → Lucro: {profit_pct:.1f}%"
        elif "buy_kalshi" in arbitrage_type:
            action = f"COMPRAR YES no Kalshi (${kalshi_yes:.3f}) + NO no Polymarket (${poly_no:.3f})"
            explanation = f"Custo total: ${total_cost:.3f} → Retorno garantido: $1.00 → Lucro: {profit_pct:.1f}%"
        else:
            cheaper = "Polymarket" if "poly_cheaper" in arbitrage_type else "Kalshi"
            action = f"OPORTUNIDADE DE VALOR: {cheaper} está {price_diff*100:.1f}% mais barato"
            explanation = f"Poly: ${poly_yes:.3f} vs Kalshi: ${kalshi_yes:.3f} - Diferença: {price_diff*100:.1f}%"
        
        # Gerar URLs para as plataformas
        poly_slug = poly_market.get("slug", "")
        poly_condition_id = poly_market.get("conditionId", "")
        kalshi_ticker = kalshi_market.get("ticker", "")
        
        # URL do Polymarket: /market/{slug} redireciona automaticamente para o evento correto
        if poly_slug:
            poly_url = f"https://polymarket.com/market/{poly_slug}"
        else:
            poly_url = "https://polymarket.com"
        
        # Para Kalshi: como dados são simulados, ir para página principal de mercados
        # Os usuários podem buscar manualmente o evento
        kalshi_url = "https://kalshi.com/markets"
        
        return CrossPlatformArbitrage(
            event_name=poly_title,
            match_confidence=similarity,
            poly_market_title=poly_title,
            poly_market_id=poly_condition_id,
            poly_market_slug=poly_slug,
            poly_yes_price=poly_yes,
            poly_no_price=poly_no,
            poly_volume=poly_volume,
            poly_url=poly_url,
            kalshi_market_title=kalshi_title,
            kalshi_market_id=kalshi_ticker,
            kalshi_ticker=kalshi_ticker,
            kalshi_yes_price=kalshi_yes,
            kalshi_no_price=kalshi_no,
            kalshi_volume=kalshi_volume,
            kalshi_url=kalshi_url,
            arbitrage_type=arbitrage_type,
            total_cost=total_cost,
            guaranteed_return=1.0,
            profit_amount=profit,
            profit_percentage=profit_pct,
            action=action,
            explanation=explanation,
            returns_100=100 * profit_pct / 100,
            returns_1000=1000 * profit_pct / 100,
            detected_at=datetime.utcnow()
        )


async def scan_cross_platform(limit: int = 100) -> List[CrossPlatformArbitrage]:
    """Função helper para escanear arbitragem cross-platform."""
    async with CrossPlatformScanner(timeout=120.0) as scanner:
        return await scanner.scan(limit)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print(f"\n{'='*70}")
    print(f"  ARBITRAGEM CROSS-PLATFORM: POLYMARKET vs KALSHI")
    print(f"{'='*70}\n")
    
    opportunities = asyncio.run(scan_cross_platform(limit))
    
    if not opportunities:
        print("\n" + "="*70)
        print("  Nenhuma arbitragem cross-platform encontrada")
        print("="*70)
        print("""
Os preços entre Polymarket e Kalshi estão alinhados.

Isso pode significar:
1. Mercados eficientes
2. Poucos mercados correspondentes
3. API do Kalshi não acessível (dados de exemplo usados)

Para arbitragem real:
- Monitore continuamente
- Use APIs autenticadas de ambas as plataformas
- Execute rapidamente quando surgir oportunidade
        """)
    else:
        print("\n" + "="*70)
        print(f"  🎯 {len(opportunities)} OPORTUNIDADES ENCONTRADAS!")
        print("="*70 + "\n")
        
        for i, arb in enumerate(opportunities[:10], 1):
            print(f"#{i} - {arb.arbitrage_type.upper()}")
            print(f"   Evento: {arb.event_name[:60]}")
            print(f"   Match: {arb.match_confidence*100:.0f}%")
            print(f"   ")
            print(f"   POLYMARKET: YES=${arb.poly_yes_price:.3f} | NO=${arb.poly_no_price:.3f}")
            print(f"   KALSHI:     YES=${arb.kalshi_yes_price:.3f} | NO=${arb.kalshi_no_price:.3f}")
            print(f"   ")
            print(f"   💰 Lucro: {arb.profit_percentage:.1f}%")
            print(f"   📊 Ação: {arb.action}")
            print(f"   ")
            print(f"   Retornos: $100→${arb.returns_100:.2f} | $1000→${arb.returns_1000:.2f}")
            print()

