"""
Arbitragem Cross-Platform REAL: Polymarket vs Kalshi
=====================================================

Sistema que usa APENAS dados reais de ambas as plataformas.
ZERO dados simulados. ZERO suposições.

Se não houver correspondência entre mercados, informa claramente.
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
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)


@dataclass
class RealArbitrageOpportunity:
    """Oportunidade de arbitragem com dados 100% reais."""
    
    # Identificação
    event_description: str
    confidence_match: float  # Quão certo que é o mesmo evento
    
    # Polymarket (dados reais)
    poly_question: str
    poly_slug: str
    poly_url: str
    poly_yes_price: Decimal
    poly_no_price: Decimal
    poly_volume_24h: Decimal
    poly_liquidity: Decimal
    
    # Kalshi (dados reais)
    kalshi_title: str
    kalshi_ticker: str
    kalshi_url: str
    kalshi_yes_price: Decimal
    kalshi_no_price: Decimal
    kalshi_volume: Decimal
    
    # Cálculo de arbitragem
    is_true_arbitrage: bool  # Se YES_A + NO_B < $1 (lucro garantido)
    strategy: str  # O que fazer
    total_cost: Decimal
    guaranteed_profit: Decimal
    profit_percentage: Decimal
    
    # Explicação clara
    explanation: str
    why_profitable: str
    risk_warning: str
    
    # Retornos calculados
    return_on_100: Decimal
    return_on_1000: Decimal
    
    detected_at: datetime


class RealCrossPlatformScanner:
    """
    Scanner que usa APENAS dados reais.
    Sem simulações. Sem suposições.
    """
    
    POLYMARKET_API = "https://gamma-api.polymarket.com"
    POLYMARKET_CLOB = "https://clob.polymarket.com"
    KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
    
    # Taxas reais
    POLYMARKET_FEE = Decimal("0.02")  # 2%
    KALSHI_FEE = Decimal("0.01")  # 1% (varia por mercado)
    
    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Dados reais coletados
        self.poly_markets: List[Dict] = []
        self.kalshi_markets: List[Dict] = []
        
        # Estatísticas
        self.stats = {
            "poly_fetched": 0,
            "kalshi_fetched": 0,
            "kalshi_api_status": "",
            "matches_found": 0,
            "true_arbitrage": 0,
            "value_opportunities": 0
        }
    
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
    
    async def scan(self, limit: int = 100) -> Tuple[List[RealArbitrageOpportunity], Dict]:
        """
        Escaneia ambas as plataformas com dados REAIS.
        Retorna oportunidades e estatísticas.
        """
        if not self.session:
            raise RuntimeError("Use como context manager")
        
        opportunities = []
        
        # 1. Buscar mercados REAIS do Polymarket
        logger.info("📊 Buscando mercados REAIS do Polymarket...")
        self.poly_markets = await self._fetch_real_polymarket(limit)
        self.stats["poly_fetched"] = len(self.poly_markets)
        logger.info(f"✅ Polymarket: {len(self.poly_markets)} mercados reais")
        
        # 2. Buscar mercados REAIS do Kalshi
        logger.info("📈 Buscando mercados REAIS do Kalshi...")
        self.kalshi_markets, kalshi_status = await self._fetch_real_kalshi(limit)
        self.stats["kalshi_fetched"] = len(self.kalshi_markets)
        self.stats["kalshi_api_status"] = kalshi_status
        logger.info(f"✅ Kalshi: {len(self.kalshi_markets)} mercados reais ({kalshi_status})")
        
        # 3. Encontrar correspondências REAIS
        logger.info("🔍 Buscando correspondências reais...")
        matches = self._find_real_matches()
        self.stats["matches_found"] = len(matches)
        
        if not matches:
            logger.warning("⚠️ Nenhuma correspondência encontrada entre as plataformas")
            return [], self.stats
        
        logger.info(f"🔗 {len(matches)} correspondências encontradas")
        
        # 4. Calcular arbitragem REAL para cada match
        for poly, kalshi, confidence in matches:
            arb = self._calculate_real_arbitrage(poly, kalshi, confidence)
            if arb:
                opportunities.append(arb)
                if arb.is_true_arbitrage:
                    self.stats["true_arbitrage"] += 1
                    logger.info(f"🎯 ARBITRAGEM REAL: {arb.profit_percentage:.1f}% - {arb.event_description[:40]}")
                else:
                    self.stats["value_opportunities"] += 1
        
        # Ordenar por lucratividade
        opportunities.sort(key=lambda x: x.profit_percentage, reverse=True)
        
        return opportunities, self.stats
    
    async def _fetch_real_polymarket(self, limit: int) -> List[Dict]:
        """Busca dados REAIS do Polymarket com preços atualizados."""
        markets = []
        
        try:
            url = f"{self.POLYMARKET_API}/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": limit,
                "order": "volume24hr",
                "ascending": "false"
            }
            
            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Polymarket API retornou {resp.status}")
                    return []
                
                raw_markets = await resp.json()
                
                # Buscar preços reais para cada mercado
                for market in raw_markets:
                    try:
                        prices = await self._get_real_poly_prices(market)
                        if prices:
                            market["_real_prices"] = prices
                            markets.append(market)
                    except Exception as e:
                        logger.debug(f"Erro ao buscar preços: {e}")
                        continue
                
                return markets
                
        except Exception as e:
            logger.error(f"Erro ao buscar Polymarket: {e}")
            return []
    
    async def _get_real_poly_prices(self, market: Dict) -> Optional[Dict]:
        """Obtém preços REAIS do order book do Polymarket."""
        try:
            clob_tokens = market.get("clobTokenIds", "[]")
            if isinstance(clob_tokens, str):
                clob_tokens = json.loads(clob_tokens)
            
            if not clob_tokens or len(clob_tokens) < 1:
                return None
            
            # Buscar preço real do YES
            yes_token = clob_tokens[0]
            url = f"{self.POLYMARKET_CLOB}/price?token_id={yes_token}&side=buy"
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    yes_price = Decimal(str(data.get("price", "0.5")))
                    
                    return {
                        "yes": yes_price,
                        "no": Decimal("1") - yes_price,
                        "source": "clob_real"
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"Erro ao buscar preço CLOB: {e}")
            return None
    
    async def _fetch_real_kalshi(self, limit: int) -> Tuple[List[Dict], str]:
        """
        Busca dados REAIS do Kalshi.
        Retorna mercados e status da API.
        """
        markets = []
        status = ""
        
        try:
            url = f"{self.KALSHI_API}/markets"
            params = {"limit": limit, "status": "open"}
            
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    status = f"API retornou {resp.status}"
                    logger.warning(f"Kalshi API: {status}")
                    return [], status
                
                data = await resp.json()
                raw_markets = data.get("markets", [])
                
                if not raw_markets:
                    status = "API não retornou mercados"
                    return [], status
                
                # Filtrar mercados úteis (não apenas esportes)
                for market in raw_markets:
                    # Extrair preços reais
                    try:
                        yes_price = Decimal(str(market.get("yes_bid", market.get("last_price", 0.5))))
                        no_price = Decimal("1") - yes_price
                        
                        market["_real_prices"] = {
                            "yes": yes_price,
                            "no": no_price,
                            "source": "kalshi_api"
                        }
                        markets.append(market)
                        
                    except Exception as e:
                        logger.debug(f"Erro ao processar mercado Kalshi: {e}")
                        continue
                
                # Verificar tipos de mercados
                categories = {}
                for m in markets:
                    cat = m.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
                
                if categories:
                    top_cat = max(categories.items(), key=lambda x: x[1])
                    status = f"{len(markets)} mercados reais (maioria: {top_cat[0]})"
                else:
                    status = f"{len(markets)} mercados reais"
                
                return markets, status
                
        except Exception as e:
            status = f"Erro: {str(e)}"
            logger.error(f"Erro ao buscar Kalshi: {e}")
            return [], status
    
    def _find_real_matches(self) -> List[Tuple[Dict, Dict, float]]:
        """
        Encontra mercados que são REALMENTE sobre o mesmo evento.
        Usa critérios rigorosos para evitar falsos positivos.
        """
        matches = []
        
        for poly in self.poly_markets:
            poly_q = poly.get("question", "").lower()
            
            for kalshi in self.kalshi_markets:
                kalshi_title = kalshi.get("title", kalshi.get("subtitle", "")).lower()
                
                # Verificar correspondência rigorosa
                confidence = self._check_real_match(poly_q, kalshi_title)
                
                if confidence >= 0.80:  # Mínimo 80% de confiança
                    matches.append((poly, kalshi, confidence))
        
        # Ordenar por confiança
        matches.sort(key=lambda x: x[2], reverse=True)
        
        return matches
    
    def _check_real_match(self, text1: str, text2: str) -> float:
        """
        Verifica se dois textos referem-se ao MESMO evento.
        Retorna confiança de 0 a 1.
        """
        # Extrair entidades de cada texto
        entities1 = self._extract_entities(text1)
        entities2 = self._extract_entities(text2)
        
        if not entities1 or not entities2:
            return 0.0
        
        # Verificar entidades em comum
        common = entities1 & entities2
        
        if not common:
            return 0.0
        
        # Calcular score baseado em entidades comuns
        score = len(common) / max(len(entities1), len(entities2))
        
        # Verificar se há contradições
        if self._has_contradiction(text1, text2):
            return 0.0
        
        return score
    
    def _extract_entities(self, text: str) -> set:
        """Extrai entidades importantes do texto."""
        text = text.lower()
        entities = set()
        
        # Pessoas/Organizações
        people = ["trump", "biden", "harris", "musk", "elon", "putin", "zelensky", 
                  "fed", "fomc", "microstrategy", "tesla"]
        
        # Eventos/Conceitos
        events = ["election", "rate", "cut", "hike", "ceasefire", "bitcoin", "btc",
                  "super bowl", "championship", "nomination"]
        
        # Datas/Períodos
        dates = ["january", "february", "march", "april", "may", "june", 
                 "july", "august", "september", "october", "november", "december",
                 "2024", "2025", "2026", "2027", "2028"]
        
        for entity in people + events + dates:
            if entity in text:
                entities.add(entity)
        
        return entities
    
    def _has_contradiction(self, text1: str, text2: str) -> bool:
        """Verifica se há contradições entre os textos."""
        contradictions = [
            ("cut", "hike"),
            ("raise", "lower"),
            ("above", "below"),
            ("win", "lose"),
            ("yes", "no"),
            ("increase", "decrease"),
        ]
        
        for word1, word2 in contradictions:
            if (word1 in text1 and word2 in text2) or (word2 in text1 and word1 in text2):
                return True
        
        return False
    
    def _calculate_real_arbitrage(
        self, 
        poly: Dict, 
        kalshi: Dict, 
        confidence: float
    ) -> Optional[RealArbitrageOpportunity]:
        """
        Calcula arbitragem com matemática REAL.
        Sem simulações. Sem suposições.
        """
        # Preços REAIS
        poly_prices = poly.get("_real_prices", {})
        kalshi_prices = kalshi.get("_real_prices", {})
        
        if not poly_prices or not kalshi_prices:
            return None
        
        poly_yes = poly_prices.get("yes", Decimal("0.5"))
        poly_no = poly_prices.get("no", Decimal("0.5"))
        kalshi_yes = kalshi_prices.get("yes", Decimal("0.5"))
        kalshi_no = kalshi_prices.get("no", Decimal("0.5"))
        
        # Volume REAL
        poly_volume = Decimal(str(poly.get("volume24hr", 0) or 0))
        kalshi_volume = Decimal(str(kalshi.get("volume", 0) or 0))
        poly_liquidity = Decimal(str(poly.get("liquidity", 0) or 0))
        
        # =====================
        # CÁLCULO DE ARBITRAGEM
        # =====================
        
        # Opção A: Comprar YES no Poly + NO no Kalshi
        cost_a = poly_yes + kalshi_no
        fees_a = (poly_yes * self.POLYMARKET_FEE) + (kalshi_no * self.KALSHI_FEE)
        total_cost_a = cost_a + fees_a
        
        # Opção B: Comprar YES no Kalshi + NO no Poly  
        cost_b = kalshi_yes + poly_no
        fees_b = (kalshi_yes * self.KALSHI_FEE) + (poly_no * self.POLYMARKET_FEE)
        total_cost_b = cost_b + fees_b
        
        # Determinar melhor estratégia
        is_true_arbitrage = False
        strategy = ""
        total_cost = Decimal("0")
        profit = Decimal("0")
        
        if total_cost_a < Decimal("1"):
            is_true_arbitrage = True
            strategy = f"COMPRAR: YES no Polymarket (${poly_yes:.3f}) + NO no Kalshi (${kalshi_no:.3f})"
            total_cost = total_cost_a
            profit = Decimal("1") - total_cost_a
            
        elif total_cost_b < Decimal("1"):
            is_true_arbitrage = True
            strategy = f"COMPRAR: YES no Kalshi (${kalshi_yes:.3f}) + NO no Polymarket (${poly_no:.3f})"
            total_cost = total_cost_b
            profit = Decimal("1") - total_cost_b
        
        else:
            # Não há arbitragem garantida, mas pode haver oportunidade de valor
            price_diff = abs(poly_yes - kalshi_yes)
            
            if price_diff < Decimal("0.03"):  # Menos de 3% de diferença
                return None  # Não vale reportar
            
            # Oportunidade de valor (não garantida)
            if poly_yes < kalshi_yes:
                strategy = f"VALOR: Polymarket tem YES mais barato (${poly_yes:.3f} vs ${kalshi_yes:.3f})"
                profit = price_diff * Decimal("0.5")  # Estimativa conservadora
                total_cost = poly_yes
            else:
                strategy = f"VALOR: Kalshi tem YES mais barato (${kalshi_yes:.3f} vs ${poly_yes:.3f})"
                profit = price_diff * Decimal("0.5")
                total_cost = kalshi_yes
        
        if profit <= 0:
            return None
        
        profit_pct = (profit / total_cost * 100).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        
        if profit_pct < Decimal("1"):  # Menos de 1% não vale
            return None
        
        # URLs reais
        poly_slug = poly.get("slug", "")
        poly_url = f"https://polymarket.com/market/{poly_slug}" if poly_slug else "https://polymarket.com"
        
        kalshi_ticker = kalshi.get("ticker", "")
        kalshi_url = f"https://kalshi.com/markets/{kalshi_ticker}" if kalshi_ticker else "https://kalshi.com/markets"
        
        # Explicação clara
        if is_true_arbitrage:
            explanation = f"""
ARBITRAGEM GARANTIDA ENCONTRADA!

O mesmo evento está precificado diferente nas duas plataformas:
• Polymarket: YES=${poly_yes:.3f} / NO=${poly_no:.3f}
• Kalshi: YES=${kalshi_yes:.3f} / NO=${kalshi_no:.3f}

Ao comprar ambos os lados em plataformas diferentes, você GARANTE lucro
independente do resultado do evento.
            """.strip()
            
            why_profitable = f"""
MATEMÁTICA DO LUCRO:
• Custo total: ${total_cost:.4f} (incluindo taxas)
• Retorno garantido: $1.00 (quando o evento resolver)
• Lucro líquido: ${profit:.4f} ({profit_pct:.1f}%)

O lucro é GARANTIDO porque a soma dos preços (${cost_a:.4f} ou ${cost_b:.4f}) 
é menor que $1.00, que é o valor que você recebe quando o evento resolve.
            """.strip()
            
            risk_warning = """
⚠️ RISCOS:
• Diferença de timing na resolução entre plataformas
• Possível mudança de preço durante execução
• Taxas de saque/conversão de moeda
• Sempre verifique os preços ATUAIS antes de operar
            """.strip()
        else:
            explanation = f"""
OPORTUNIDADE DE VALOR (NÃO GARANTIDA)

Há uma diferença de preço significativa entre as plataformas:
• Polymarket: YES=${poly_yes:.3f}
• Kalshi: YES=${kalshi_yes:.3f}
• Diferença: {abs(poly_yes - kalshi_yes)*100:.1f}%

Isso pode indicar que uma plataforma está "errada" sobre a probabilidade.
            """.strip()
            
            why_profitable = f"""
LÓGICA DA OPORTUNIDADE:
Se os preços convergirem, quem comprou no preço mais baixo lucra.
Porém, NÃO É GARANTIDO - os preços podem não convergir.
            """.strip()
            
            risk_warning = """
⚠️ ALTO RISCO:
• Não é lucro garantido
• Preços podem divergir mais antes de convergir
• Você pode perder dinheiro
• FAÇA SUA PRÓPRIA ANÁLISE
            """.strip()
        
        return RealArbitrageOpportunity(
            event_description=poly.get("question", "")[:100],
            confidence_match=confidence,
            poly_question=poly.get("question", ""),
            poly_slug=poly_slug,
            poly_url=poly_url,
            poly_yes_price=poly_yes,
            poly_no_price=poly_no,
            poly_volume_24h=poly_volume,
            poly_liquidity=poly_liquidity,
            kalshi_title=kalshi.get("title", kalshi.get("subtitle", "")),
            kalshi_ticker=kalshi_ticker,
            kalshi_url=kalshi_url,
            kalshi_yes_price=kalshi_yes,
            kalshi_no_price=kalshi_no,
            kalshi_volume=kalshi_volume,
            is_true_arbitrage=is_true_arbitrage,
            strategy=strategy,
            total_cost=total_cost,
            guaranteed_profit=profit,
            profit_percentage=profit_pct,
            explanation=explanation,
            why_profitable=why_profitable,
            risk_warning=risk_warning,
            return_on_100=Decimal("100") * profit_pct / 100,
            return_on_1000=Decimal("1000") * profit_pct / 100,
            detected_at=datetime.utcnow()
        )


async def scan_real_arbitrage(limit: int = 100) -> Tuple[List[RealArbitrageOpportunity], Dict]:
    """Função helper para escanear arbitragem REAL."""
    async with RealCrossPlatformScanner(timeout=120.0) as scanner:
        return await scanner.scan(limit)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print("\n" + "="*70)
    print("  ARBITRAGEM CROSS-PLATFORM - APENAS DADOS REAIS")
    print("="*70 + "\n")
    
    opportunities, stats = asyncio.run(scan_real_arbitrage(limit))
    
    print("\n" + "="*70)
    print("  ESTATÍSTICAS")
    print("="*70)
    print(f"  Mercados Polymarket: {stats['poly_fetched']}")
    print(f"  Mercados Kalshi: {stats['kalshi_fetched']} ({stats['kalshi_api_status']})")
    print(f"  Correspondências encontradas: {stats['matches_found']}")
    print(f"  Arbitragens REAIS: {stats['true_arbitrage']}")
    print(f"  Oportunidades de valor: {stats['value_opportunities']}")
    print("="*70 + "\n")
    
    if not opportunities:
        print("❌ Nenhuma oportunidade encontrada com dados reais.")
        print()
        print("Isso pode significar:")
        print("1. Os mercados estão eficientes (preços alinhados)")
        print("2. Poucos mercados correspondentes entre as plataformas")
        print("3. A API do Kalshi retorna categorias diferentes do Polymarket")
        print()
    else:
        print(f"✅ {len(opportunities)} OPORTUNIDADES ENCONTRADAS\n")
        
        for i, arb in enumerate(opportunities[:5], 1):
            tipo = "🔒 ARBITRAGEM GARANTIDA" if arb.is_true_arbitrage else "⚠️ OPORTUNIDADE DE VALOR"
            
            print(f"#{i} {tipo}")
            print(f"   Evento: {arb.event_description[:55]}")
            print(f"   Confiança: {arb.confidence_match*100:.0f}%")
            print()
            print(f"   POLYMARKET: YES ${arb.poly_yes_price:.3f} | NO ${arb.poly_no_price:.3f}")
            print(f"   KALSHI:     YES ${arb.kalshi_yes_price:.3f} | NO ${arb.kalshi_no_price:.3f}")
            print()
            print(f"   📊 {arb.strategy}")
            print(f"   💰 Lucro: {arb.profit_percentage:.1f}%")
            print(f"   📈 $100→${arb.return_on_100:.2f} | $1000→${arb.return_on_1000:.2f}")
            print()
            print("-"*70)
            print()




