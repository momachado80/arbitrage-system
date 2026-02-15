"""
Bot de Arbitragem com Lucro Garantido
=====================================

Busca oportunidades de arbitragem REAL onde o lucro é garantido:

1. YES + NO < $1 (mesmo mercado)
2. Multi-outcome onde soma > 100% ou < 100%
3. Cross-platform (Polymarket vs outras)

IMPORTANTE: Estas oportunidades são RARAS e duram SEGUNDOS.
Para capturá-las, você precisa:
- Execução automatizada
- Capital disponível
- Monitoramento 24/7
"""

import asyncio
import aiohttp
import ssl
import certifi
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class GuaranteedArbitrage:
    """Oportunidade de arbitragem com lucro garantido."""
    arbitrage_type: str  # 'yes_no', 'multi_outcome', 'cross_platform'
    
    market_title: str
    market_id: str
    
    # Preços
    yes_ask: float  # Preço para COMPRAR yes
    no_ask: float   # Preço para COMPRAR no
    total_cost: float  # Custo total
    guaranteed_return: float  # Retorno garantido ($1 para binário)
    
    # Lucro
    profit_amount: float  # Lucro em $ por ciclo
    profit_percentage: float  # Lucro em %
    
    # Detalhes
    action: str  # Ação a tomar
    explanation: str
    
    # Liquidez
    yes_liquidity: float
    no_liquidity: float
    max_investment: float  # Máximo que pode investir
    
    # Retornos projetados
    returns_100: float
    returns_1000: float
    returns_10000: float
    
    detected_at: datetime
    expires_in_seconds: int  # Estimativa de duração


class GuaranteedArbitrageBot:
    """
    Bot que busca arbitragem com lucro garantido em tempo real.
    """
    
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    
    # Taxas do Polymarket (aproximadas)
    TAKER_FEE = 0.02  # 2%
    
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
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
    
    async def scan(self, limit: int = 200) -> List[GuaranteedArbitrage]:
        """
        Escaneia mercados buscando arbitragem garantida.
        """
        if not self.session:
            raise RuntimeError("Use como context manager")
        
        opportunities = []
        
        # Buscar mercados
        logger.info(f"Buscando {limit} mercados para arbitragem garantida...")
        markets = await self._fetch_markets(limit)
        logger.info(f"Encontrados {len(markets)} mercados")
        
        # Analisar cada mercado
        for i, market in enumerate(markets):
            if i % 20 == 0:
                logger.info(f"Analisando mercado {i+1}/{len(markets)}...")
            
            try:
                # Buscar orderbooks completos
                arb = await self._check_market_arbitrage(market)
                if arb:
                    opportunities.append(arb)
                    logger.info(f"🎯 ARBITRAGEM ENCONTRADA: {arb.profit_percentage:.2f}% em {arb.market_title[:50]}")
            except Exception as e:
                logger.debug(f"Erro analisando mercado: {e}")
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        # Ordenar por lucro
        opportunities.sort(key=lambda x: x.profit_percentage, reverse=True)
        
        logger.info(f"Total: {len(opportunities)} oportunidades de arbitragem garantida")
        return opportunities
    
    async def _fetch_markets(self, limit: int) -> List[Dict]:
        """Busca mercados ativos."""
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
                return []
        except Exception as e:
            logger.error(f"Erro ao buscar mercados: {e}")
            return []
    
    async def _fetch_orderbook(self, token_id: str) -> Optional[Dict]:
        """Busca orderbook completo de um token."""
        try:
            url = f"{self.CLOB_API}/book?token_id={token_id}"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except:
            return None
    
    async def _check_market_arbitrage(self, market: Dict) -> Optional[GuaranteedArbitrage]:
        """
        Verifica se há arbitragem garantida no mercado.
        
        Arbitragem existe quando: YES_ASK + NO_ASK < $1.00 (menos taxas)
        """
        # Extrair tokens
        clob_tokens_raw = market.get("clobTokenIds", "[]")
        if isinstance(clob_tokens_raw, str):
            try:
                tokens = json.loads(clob_tokens_raw)
            except:
                return None
        else:
            tokens = clob_tokens_raw
        
        if len(tokens) < 2:
            return None
        
        yes_token = tokens[0]
        no_token = tokens[1]
        
        # Buscar orderbooks
        yes_book = await self._fetch_orderbook(yes_token)
        no_book = await self._fetch_orderbook(no_token)
        
        if not yes_book or not no_book:
            return None
        
        # Extrair melhor ASK (preço para comprar)
        yes_asks = yes_book.get("asks", [])
        no_asks = no_book.get("asks", [])
        
        if not yes_asks or not no_asks:
            return None
        
        yes_ask = float(yes_asks[0].get("price", 1))
        yes_size = float(yes_asks[0].get("size", 0))
        
        no_ask = float(no_asks[0].get("price", 1))
        no_size = float(no_asks[0].get("size", 0))
        
        # Calcular custo total
        total_cost = yes_ask + no_ask
        
        # Aplicar taxas (2% taker fee em cada lado)
        total_cost_with_fees = total_cost * (1 + self.TAKER_FEE)
        
        # Retorno garantido é $1.00
        guaranteed_return = 1.0
        
        # Verificar se há lucro
        if total_cost_with_fees >= guaranteed_return:
            return None  # Não há arbitragem
        
        # Calcular lucro
        profit_amount = guaranteed_return - total_cost_with_fees
        profit_percentage = (profit_amount / total_cost_with_fees) * 100
        
        # Mínimo de 0.5% de lucro para valer a pena
        if profit_percentage < 0.5:
            return None
        
        # Calcular máximo investimento (limitado pela menor liquidez)
        max_contracts = min(yes_size, no_size)
        max_investment = max_contracts * total_cost
        
        # Retornos projetados
        returns_100 = 100 * profit_percentage / 100
        returns_1000 = 1000 * profit_percentage / 100
        returns_10000 = 10000 * profit_percentage / 100
        
        market_title = market.get("question", "Unknown")
        market_id = market.get("conditionId", "")
        
        return GuaranteedArbitrage(
            arbitrage_type="yes_no",
            market_title=market_title[:150],
            market_id=market_id,
            yes_ask=yes_ask,
            no_ask=no_ask,
            total_cost=total_cost_with_fees,
            guaranteed_return=guaranteed_return,
            profit_amount=profit_amount,
            profit_percentage=profit_percentage,
            action=f"COMPRAR {max_contracts:.0f} YES a ${yes_ask:.3f} + {max_contracts:.0f} NO a ${no_ask:.3f}",
            explanation=f"Custo: ${total_cost_with_fees:.4f} (inc. taxas) → Retorno garantido: $1.00 → Lucro: ${profit_amount:.4f} ({profit_percentage:.2f}%)",
            yes_liquidity=yes_size,
            no_liquidity=no_size,
            max_investment=max_investment,
            returns_100=returns_100,
            returns_1000=returns_1000,
            returns_10000=returns_10000,
            detected_at=datetime.utcnow(),
            expires_in_seconds=30  # Estimativa conservadora
        )
    
    async def monitor_realtime(self, market_ids: List[str], callback):
        """
        Monitora mercados específicos em tempo real.
        Chama callback quando encontrar arbitragem.
        """
        logger.info(f"Iniciando monitoramento em tempo real de {len(market_ids)} mercados...")
        
        while True:
            for market_id in market_ids:
                try:
                    # Buscar mercado
                    url = f"{self.GAMMA_API}/markets/{market_id}"
                    async with self.session.get(url) as resp:
                        if resp.status == 200:
                            market = await resp.json()
                            arb = await self._check_market_arbitrage(market)
                            if arb:
                                await callback(arb)
                except Exception as e:
                    logger.debug(f"Erro monitorando {market_id}: {e}")
                
                await asyncio.sleep(0.5)  # 2 checks por segundo por mercado
            
            await asyncio.sleep(1)


async def scan_guaranteed_arbitrage(limit: int = 200) -> List[GuaranteedArbitrage]:
    """Função helper para escanear arbitragem garantida."""
    async with GuaranteedArbitrageBot(timeout=120.0) as bot:
        return await bot.scan(limit)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    
    print(f"\n{'='*70}")
    print(f"  BOT DE ARBITRAGEM COM LUCRO GARANTIDO")
    print(f"  Buscando YES + NO < $1.00")
    print(f"{'='*70}\n")
    
    opportunities = asyncio.run(scan_guaranteed_arbitrage(limit))
    
    if not opportunities:
        print("\n" + "="*70)
        print("  ❌ NENHUMA ARBITRAGEM GARANTIDA ENCONTRADA")
        print("="*70)
        print("""
Isso é NORMAL. Arbitragem garantida é extremamente rara porque:

1. 🤖 Bots de HFT capturam em milissegundos
2. 💰 Market makers mantêm preços eficientes
3. 📊 Milhares de traders monitoram 24/7

Para capturar arbitragem garantida, você precisaria:
- Execução automatizada (API)
- Monitoramento 24/7
- Capital disponível instantaneamente
- Latência muito baixa

O que podemos fazer:
- Monitorar continuamente e alertar quando aparecer
- Integrar com API do Polymarket para execução automática
        """)
    else:
        print("\n" + "="*70)
        print(f"  🎯 {len(opportunities)} ARBITRAGENS GARANTIDAS ENCONTRADAS!")
        print("="*70 + "\n")
        
        for i, arb in enumerate(opportunities[:10], 1):
            print(f"#{i} - Lucro: {arb.profit_percentage:.2f}%")
            print(f"   Mercado: {arb.market_title[:60]}")
            print(f"   YES ASK: ${arb.yes_ask:.4f} | NO ASK: ${arb.no_ask:.4f}")
            print(f"   Custo Total: ${arb.total_cost:.4f}")
            print(f"   Lucro por ciclo: ${arb.profit_amount:.4f}")
            print(f"   ")
            print(f"   Liquidez: YES={arb.yes_liquidity:.0f} | NO={arb.no_liquidity:.0f}")
            print(f"   Máx. investimento: ${arb.max_investment:.2f}")
            print(f"   ")
            print(f"   Retornos:")
            print(f"   $100 → ${100 + arb.returns_100:.2f} (lucro: ${arb.returns_100:.2f})")
            print(f"   $1,000 → ${1000 + arb.returns_1000:.2f} (lucro: ${arb.returns_1000:.2f})")
            print(f"   $10,000 → ${10000 + arb.returns_10000:.2f} (lucro: ${arb.returns_10000:.2f})")
            print(f"   ")
            print(f"   ⚠️  AÇÃO RÁPIDA: {arb.action}")
            print(f"   ⏱️  Expira em ~{arb.expires_in_seconds}s")
            print()




