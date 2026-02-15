"""
Kalshi Authenticated Collector
==============================

Coleta dados REAIS do Kalshi usando autenticação.
Acessa TODOS os mercados (política, economia, etc.), não apenas esportes.
"""

import asyncio
import aiohttp
import ssl
import certifi
import logging
import hashlib
import hmac
import base64
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KalshiCredentials:
    """Credenciais do Kalshi."""
    email: str
    password: str


class KalshiAuthenticatedCollector:
    """
    Coletor autenticado do Kalshi.
    
    Suporta autenticação via:
    - API Key (recomendado)
    - Email/Senha (login tradicional)
    """
    
    # URLs da API do Kalshi (atualizado janeiro 2026)
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"  # Para testes
    
    def __init__(
        self, 
        api_key: str = None,
        email: str = None, 
        password: str = None,
        use_demo: bool = False,
        timeout: float = 60.0
    ):
        self.api_key = api_key
        self.email = email
        self.password = password
        self.base_url = self.DEMO_URL if use_demo else self.BASE_URL
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.token: Optional[str] = None
        self.member_id: Optional[str] = None
        self.authenticated = False
    
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
    
    async def login(self, api_key: str = None, email: str = None, password: str = None) -> bool:
        """
        Autentica na API do Kalshi.
        
        Prioridade:
        1. API Key (se fornecida)
        2. Email/Senha (login tradicional)
        
        Retorna True se autenticado com sucesso.
        """
        api_key = api_key or self.api_key
        email = email or self.email
        password = password or self.password
        
        # Método 1: API Key (mais simples e recomendado)
        if api_key:
            logger.info(f"Autenticando com API Key: {api_key[:8]}...")
            self.api_key = api_key
            self.authenticated = True
            
            # Testar se a key funciona buscando mercados
            try:
                markets, _ = await self.get_markets(limit=1)
                if markets:
                    logger.info(f"✅ API Key válida! Acesso confirmado.")
                    return True
                else:
                    # Pode ter funcionado mas sem mercados
                    logger.info(f"✅ API Key aceita (verificar acesso a mercados)")
                    return True
            except Exception as e:
                logger.error(f"❌ API Key inválida ou sem permissão: {e}")
                self.authenticated = False
                return False
        
        # Método 2: Login com email/senha
        if not email or not password:
            logger.error("API Key ou Email/Senha são obrigatórios")
            return False
        
        try:
            url = f"{self.base_url}/login"
            payload = {
                "email": email,
                "password": password
            }
            
            logger.info(f"Tentando login no Kalshi com {email[:3]}***")
            
            async with self.session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.token = data.get("token")
                    self.member_id = data.get("member_id")
                    self.authenticated = True
                    logger.info(f"✅ Login bem-sucedido! Member ID: {self.member_id}")
                    return True
                elif resp.status == 401:
                    logger.error("❌ Credenciais inválidas")
                    return False
                elif resp.status == 403:
                    error_data = await resp.json()
                    logger.error(f"❌ Acesso negado: {error_data}")
                    return False
                else:
                    error_text = await resp.text()
                    logger.error(f"❌ Erro no login: {resp.status} - {error_text[:200]}")
                    return False
                    
        except Exception as e:
            logger.error(f"Erro ao fazer login: {e}")
            return False
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Retorna headers com autenticação."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Prioridade: API Key > Token (login)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        return headers
    
    async def get_markets(
        self, 
        limit: int = 100,
        cursor: str = None,
        event_ticker: str = None,
        series_ticker: str = None,
        status: str = "open",
        tickers: List[str] = None
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Busca mercados do Kalshi.
        
        Retorna: (lista de mercados, próximo cursor para paginação)
        """
        if not self.authenticated:
            logger.warning("Não autenticado. Tentando buscar sem auth...")
        
        try:
            url = f"{self.base_url}/markets"
            params = {
                "limit": min(limit, 200),  # Kalshi limita a 200
                "status": status
            }
            
            if cursor:
                params["cursor"] = cursor
            if event_ticker:
                params["event_ticker"] = event_ticker
            if series_ticker:
                params["series_ticker"] = series_ticker
            if tickers:
                params["tickers"] = ",".join(tickers)
            
            headers = self._get_auth_headers()
            
            async with self.session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    markets = data.get("markets", [])
                    next_cursor = data.get("cursor")
                    
                    logger.info(f"Kalshi retornou {len(markets)} mercados")
                    
                    # Log de categorias para debug
                    categories = {}
                    for m in markets:
                        cat = m.get("category", "unknown")
                        categories[cat] = categories.get(cat, 0) + 1
                    
                    if categories:
                        logger.info(f"Categorias: {categories}")
                    
                    return markets, next_cursor
                else:
                    error_text = await resp.text()
                    logger.error(f"Erro ao buscar mercados: {resp.status} - {error_text[:200]}")
                    return [], None
                    
        except Exception as e:
            logger.error(f"Erro ao buscar mercados: {e}")
            return [], None
    
    async def get_all_markets(
        self, 
        max_markets: int = 500,
        status: str = "open",
        include_events: bool = True
    ) -> List[Dict]:
        """
        Busca todos os mercados usando paginação.
        Se include_events=True, também busca mercados aninhados em eventos.
        """
        all_markets = []
        
        # Método 1: Buscar via endpoint /markets (retorna principalmente esportes)
        cursor = None
        while len(all_markets) < max_markets // 2:
            batch_size = min(200, max_markets - len(all_markets))
            markets, cursor = await self.get_markets(
                limit=batch_size,
                cursor=cursor,
                status=status
            )
            
            if not markets:
                break
            
            all_markets.extend(markets)
            
            if not cursor:
                break
            
            await asyncio.sleep(0.1)
        
        # Método 2: Buscar via endpoint /events com mercados aninhados
        # Este método retorna mercados de política, economia, etc.
        if include_events:
            event_markets = await self.get_markets_from_events(
                max_markets=max_markets,
                categories=["Politics", "Economics", "Financials", "World", "Elections"]
            )
            
            # Adicionar apenas mercados únicos
            existing_tickers = {m.get("ticker") for m in all_markets}
            for m in event_markets:
                if m.get("ticker") not in existing_tickers:
                    all_markets.append(m)
                    existing_tickers.add(m.get("ticker"))
        
        return all_markets
    
    async def get_markets_from_events(
        self, 
        max_markets: int = 300,
        categories: List[str] = None
    ) -> List[Dict]:
        """
        Busca mercados via endpoint de eventos.
        Este é o método correto para obter mercados de política/economia.
        Suporta paginação para buscar grandes volumes.
        """
        all_markets = []
        cursor = None
        
        # Categorias padrão se não especificado
        if categories is None:
            categories = [
                "Politics", "Economics", "Financials", "World", "Elections",
                "Entertainment", "Science and Technology", "Climate and Weather",
                "Social", "Companies", "Crypto", "Health"
            ]
        
        try:
            while len(all_markets) < max_markets:
                url = f"{self.base_url}/events"
                params = {
                    "limit": 200,
                    "with_nested_markets": "true"
                }
                
                if cursor:
                    params["cursor"] = cursor
                
                headers = self._get_auth_headers()
                
                async with self.session.get(url, params=params, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Erro ao buscar eventos: {resp.status} - {error_text[:200]}")
                        break
                    
                    data = await resp.json()
                    events = data.get("events", [])
                    cursor = data.get("cursor")
                    
                    if not events:
                        break
                    
                    # Extrair mercados dos eventos
                    for event in events:
                        event_category = event.get("category", "")
                        event_title = event.get("title", "")
                        
                        # Filtrar por categoria
                        category_match = False
                        for cat in categories:
                            if cat.lower() in event_category.lower() or event_category.lower() in cat.lower():
                                category_match = True
                                break
                        
                        if categories and not category_match:
                            continue
                        
                        for market in event.get("markets", []):
                            # Adicionar metadados do evento
                            market["_category"] = event_category
                            market["_event_title"] = event_title
                            all_markets.append(market)
                            
                            if len(all_markets) >= max_markets:
                                break
                        
                        if len(all_markets) >= max_markets:
                            break
                    
                    # Se não há mais páginas, parar
                    if not cursor:
                        break
                    
                    # Pequena pausa para não sobrecarregar
                    await asyncio.sleep(0.1)
            
            logger.info(f"Extraídos {len(all_markets)} mercados de eventos")
            
            # Log de categorias
            cat_counts = {}
            for m in all_markets:
                cat = m.get("_category", "unknown")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            
            if cat_counts:
                logger.info(f"Categorias: {cat_counts}")
            
            return all_markets
            
        except Exception as e:
            logger.error(f"Erro ao buscar mercados de eventos: {e}")
            return all_markets  # Retornar o que já foi coletado
    
    async def get_events(self, limit: int = 100) -> List[Dict]:
        """Busca eventos (grupos de mercados)."""
        try:
            url = f"{self.base_url}/events"
            params = {"limit": min(limit, 200), "status": "open"}
            headers = self._get_auth_headers()
            
            async with self.session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    events = data.get("events", [])
                    logger.info(f"Kalshi retornou {len(events)} eventos")
                    return events
                else:
                    logger.error(f"Erro ao buscar eventos: {resp.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Erro ao buscar eventos: {e}")
            return []
    
    async def get_market(self, ticker: str) -> Optional[Dict]:
        """Busca um mercado específico pelo ticker."""
        try:
            url = f"{self.base_url}/markets/{ticker}"
            headers = self._get_auth_headers()
            
            async with self.session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("market")
                else:
                    logger.error(f"Mercado {ticker} não encontrado: {resp.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Erro ao buscar mercado {ticker}: {e}")
            return None
    
    async def get_orderbook(self, ticker: str, depth: int = 10) -> Optional[Dict]:
        """Busca o order book de um mercado."""
        try:
            url = f"{self.base_url}/markets/{ticker}/orderbook"
            params = {"depth": depth}
            headers = self._get_auth_headers()
            
            async with self.session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Erro ao buscar orderbook: {e}")
            return None


async def test_kalshi_auth(api_key: str = None, email: str = None, password: str = None) -> Dict:
    """
    Testa a autenticação do Kalshi.
    Suporta API Key ou Email/Senha.
    Retorna informações sobre mercados disponíveis.
    """
    async with KalshiAuthenticatedCollector(api_key=api_key, email=email, password=password) as collector:
        # Tentar login
        success = await collector.login()
        
        if not success:
            return {
                "authenticated": False,
                "error": "Falha na autenticação. Verifique API Key ou credenciais.",
                "markets": [],
                "categories": {}
            }
        
        # Buscar mercados
        markets = await collector.get_all_markets(max_markets=200)
        
        # Categorizar mercados
        categories = {}
        sample_markets = {}
        
        for m in markets:
            cat = m.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            
            # Guardar um exemplo de cada categoria
            if cat not in sample_markets:
                sample_markets[cat] = {
                    "ticker": m.get("ticker"),
                    "title": m.get("title", m.get("subtitle", "?")),
                    "yes_bid": m.get("yes_bid"),
                    "yes_ask": m.get("yes_ask")
                }
        
        return {
            "authenticated": True,
            "member_id": collector.member_id,
            "api_key_used": bool(api_key),
            "total_markets": len(markets),
            "categories": categories,
            "sample_markets": sample_markets,
            "markets": markets
        }


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 3:
        print("Uso: python kalshi_authenticated.py <email> <password>")
        sys.exit(1)
    
    email = sys.argv[1]
    password = sys.argv[2]
    
    print("\n" + "="*60)
    print("  TESTE DE AUTENTICAÇÃO KALSHI")
    print("="*60 + "\n")
    
    result = asyncio.run(test_kalshi_auth(email, password))
    
    if result["authenticated"]:
        print(f"✅ Autenticado com sucesso!")
        print(f"   Member ID: {result['member_id']}")
        print(f"   Total de mercados: {result['total_markets']}")
        print()
        print("📊 Categorias de mercados:")
        for cat, count in sorted(result["categories"].items(), key=lambda x: -x[1]):
            print(f"   • {cat}: {count} mercados")
        print()
        print("📝 Exemplos por categoria:")
        for cat, sample in result["sample_markets"].items():
            print(f"   [{cat}] {sample['title'][:50]}")
            print(f"      Ticker: {sample['ticker']}, YES: {sample['yes_bid']}")
    else:
        print(f"❌ {result['error']}")

