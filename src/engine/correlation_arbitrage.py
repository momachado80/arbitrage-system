"""
Arbitragem em Mercados Correlacionados
======================================

Identifica oportunidades de arbitragem entre mercados que têm
relação lógica entre si.

Exemplos de correlações:
1. Eleições: "Trump vence" vs "Republicano vence" (Trump ⊆ Republicano)
2. Datas: "Evento até Jan" vs "Evento até Mar" (Jan ⊆ Mar)
3. Valores: "BTC > $100k" vs "BTC > $150k" ($150k ⊂ $100k)
4. Esportes: "Time A vence campeonato" implica "Time A passa de fase"
"""

import asyncio
import aiohttp
import ssl
import certifi
import json
import logging
import re
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CorrelatedOpportunity:
    """Oportunidade de arbitragem entre mercados correlacionados."""
    opportunity_type: str  # 'subset', 'mutex', 'complement', 'temporal'
    
    market1_title: str
    market1_id: str
    market1_yes_price: float
    market1_no_price: float
    
    market2_title: str
    market2_id: str
    market2_yes_price: float
    market2_no_price: float
    
    correlation_type: str  # Descrição da correlação
    inconsistency: str  # Descrição da inconsistência
    
    arbitrage_action: str  # Ação recomendada
    expected_profit: float  # Lucro esperado em %
    confidence: float  # 0-1
    
    explanation: str  # Explicação detalhada
    
    detected_at: datetime


class CorrelationArbitrageScanner:
    """
    Scanner que identifica arbitragem entre mercados correlacionados.
    """
    
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.markets_cache: List[Dict] = []
        self.prices_cache: Dict[str, Dict] = {}
    
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
    
    async def scan(self, limit: int = 200) -> List[CorrelatedOpportunity]:
        """
        Escaneia mercados e encontra oportunidades de arbitragem correlacionada.
        """
        if not self.session:
            raise RuntimeError("Use como context manager")
        
        opportunities = []
        
        # 1. Buscar mercados
        logger.info(f"Buscando {limit} mercados...")
        self.markets_cache = await self._fetch_markets(limit)
        logger.info(f"Encontrados {len(self.markets_cache)} mercados")
        
        # 2. Buscar preços de todos os mercados
        logger.info("Buscando preços...")
        await self._fetch_all_prices()
        
        # 3. Agrupar mercados por categoria/tema
        groups = self._group_markets()
        
        # 4. Analisar cada grupo
        for group_name, markets in groups.items():
            if len(markets) < 2:
                continue
            
            logger.info(f"Analisando grupo '{group_name}' com {len(markets)} mercados...")
            
            # Encontrar correlações dentro do grupo
            group_opps = await self._analyze_group(group_name, markets)
            opportunities.extend(group_opps)
        
        # 5. Análises específicas
        
        # Análise de datas (eventos temporais)
        temporal_opps = await self._analyze_temporal_correlations()
        opportunities.extend(temporal_opps)
        
        # Análise de valores numéricos
        numeric_opps = await self._analyze_numeric_correlations()
        opportunities.extend(numeric_opps)
        
        # Análise de subconjuntos (candidatos vs partido)
        subset_opps = await self._analyze_subset_correlations()
        opportunities.extend(subset_opps)
        
        # Ordenar por lucro esperado
        opportunities.sort(key=lambda x: x.expected_profit, reverse=True)
        
        logger.info(f"Encontradas {len(opportunities)} oportunidades de arbitragem correlacionada")
        return opportunities
    
    async def _fetch_markets(self, limit: int) -> List[Dict]:
        """Busca mercados da API."""
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
    
    async def _fetch_all_prices(self):
        """Busca preços de todos os mercados."""
        for market in self.markets_cache:
            clob_tokens_raw = market.get("clobTokenIds", "[]")
            if isinstance(clob_tokens_raw, str):
                try:
                    tokens = json.loads(clob_tokens_raw)
                except:
                    continue
            else:
                tokens = clob_tokens_raw
            
            if len(tokens) < 2:
                continue
            
            market_id = market.get("conditionId", "")
            
            # Buscar preço YES
            yes_price = await self._fetch_midpoint(tokens[0])
            no_price = await self._fetch_midpoint(tokens[1])
            
            if yes_price is None:
                yes_price = 0.5
            if no_price is None:
                no_price = 1 - yes_price
            
            self.prices_cache[market_id] = {
                "yes": yes_price,
                "no": no_price,
                "title": market.get("question", ""),
                "volume": float(market.get("volume24hr", 0) or 0)
            }
            
            # Rate limiting
            await asyncio.sleep(0.05)
    
    async def _fetch_midpoint(self, token_id: str) -> Optional[float]:
        """Busca midpoint de um token."""
        try:
            url = f"{self.CLOB_API}/midpoint?token_id={token_id}"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    data = json.loads(text)
                    mid_str = data.get("mid", "0")
                    return float(mid_str) if mid_str else None
                return None
        except:
            return None
    
    def _group_markets(self) -> Dict[str, List[Dict]]:
        """Agrupa mercados por tema/categoria."""
        groups = defaultdict(list)
        
        for market in self.markets_cache:
            question = market.get("question", "").lower()
            category = market.get("category", "").lower()
            
            # Agrupar por palavras-chave
            keywords = [
                ("fed", "federal reserve"),
                ("trump", "trump"),
                ("bitcoin", "btc"),
                ("election", "president"),
                ("ukraine", "russia"),
                ("ceasefire", "ceasefire"),
                ("elon", "musk"),
                ("nba", "basketball"),
                ("nfl", "football"),
                ("crypto", "token"),
            ]
            
            for key, group_name in keywords:
                if key in question:
                    groups[group_name].append(market)
                    break
            else:
                # Usar categoria como grupo
                if category:
                    groups[category].append(market)
        
        return dict(groups)
    
    async def _analyze_group(self, group_name: str, markets: List[Dict]) -> List[CorrelatedOpportunity]:
        """Analisa um grupo de mercados relacionados."""
        opportunities = []
        
        # Comparar cada par de mercados
        for i, m1 in enumerate(markets):
            for m2 in markets[i+1:]:
                opp = self._check_correlation(m1, m2, group_name)
                if opp:
                    opportunities.append(opp)
        
        return opportunities
    
    def _check_correlation(self, m1: Dict, m2: Dict, group: str) -> Optional[CorrelatedOpportunity]:
        """Verifica se há correlação explorável entre dois mercados."""
        
        id1 = m1.get("conditionId", "")
        id2 = m2.get("conditionId", "")
        
        if id1 not in self.prices_cache or id2 not in self.prices_cache:
            return None
        
        p1 = self.prices_cache[id1]
        p2 = self.prices_cache[id2]
        
        title1 = p1["title"]
        title2 = p2["title"]
        yes1, no1 = p1["yes"], p1["no"]
        yes2, no2 = p2["yes"], p2["no"]
        
        # =============================================
        # VERIFICAR CORRELAÇÕES LÓGICAS
        # =============================================
        
        # 1. MUTUAMENTE EXCLUSIVOS
        # Se eventos são mutuamente exclusivos, YES1 + YES2 <= 1
        if self._are_mutually_exclusive(title1, title2):
            total_yes = yes1 + yes2
            if total_yes > 1.05:  # Mais de 105% - inconsistência
                profit = (total_yes - 1) * 100
                return CorrelatedOpportunity(
                    opportunity_type="mutex",
                    market1_title=title1[:100],
                    market1_id=id1,
                    market1_yes_price=yes1,
                    market1_no_price=no1,
                    market2_title=title2[:100],
                    market2_id=id2,
                    market2_yes_price=yes2,
                    market2_no_price=no2,
                    correlation_type="Eventos mutuamente exclusivos",
                    inconsistency=f"YES1 ({yes1:.1%}) + YES2 ({yes2:.1%}) = {total_yes:.1%} > 100%",
                    arbitrage_action=f"Vender NO em ambos: Pagar ${1-yes1:.2f} + ${1-yes2:.2f} = ${2-total_yes:.2f} para garantir $1",
                    expected_profit=profit,
                    confidence=0.8,
                    explanation=f"Se {title1[:50]} e {title2[:50]} não podem acontecer juntos, a soma das probabilidades deveria ser ≤ 100%",
                    detected_at=datetime.utcnow()
                )
        
        # 2. SUBCONJUNTO (A implica B)
        # Se A acontecer implica B acontecer, então P(A) <= P(B)
        # Exemplo: "Ceasefire by Jan" implica "Ceasefire by Mar" 
        #          Então P(Jan) <= P(Mar)
        #          Se P(Jan) > P(Mar), há inconsistência
        implies = self._check_implication(title1, title2)
        if implies == 1:  # title1 implica title2 (A→B)
            # Se A implica B, então P(A) <= P(B)
            # Inconsistência se P(A) > P(B)
            if yes1 > yes2 + 0.05:
                profit = (yes1 - yes2) * 100
                return CorrelatedOpportunity(
                    opportunity_type="subset",
                    market1_title=title1[:100],
                    market1_id=id1,
                    market1_yes_price=yes1,
                    market1_no_price=no1,
                    market2_title=title2[:100],
                    market2_id=id2,
                    market2_yes_price=yes2,
                    market2_no_price=no2,
                    correlation_type="Implicação lógica (A → B)",
                    inconsistency=f"P(A)={yes1:.1%} > P(B)={yes2:.1%}, mas A→B exige P(A)≤P(B)",
                    arbitrage_action=f"Vender YES em '{title1[:30]}...' a ${yes1:.2f} e Comprar YES em '{title2[:30]}...' a ${yes2:.2f}",
                    expected_profit=profit,
                    confidence=0.85,
                    explanation=f"'{title1[:40]}' implica '{title2[:40]}'. Se A acontece, B necessariamente acontece. Logo P(A) ≤ P(B).",
                    detected_at=datetime.utcnow()
                )
        elif implies == -1:  # title2 implica title1 (B→A)
            # Se B implica A, então P(B) <= P(A)
            # Inconsistência se P(B) > P(A)
            if yes2 > yes1 + 0.05:
                profit = (yes2 - yes1) * 100
                return CorrelatedOpportunity(
                    opportunity_type="subset",
                    market1_title=title2[:100],
                    market1_id=id2,
                    market1_yes_price=yes2,
                    market1_no_price=no2,
                    market2_title=title1[:100],
                    market2_id=id1,
                    market2_yes_price=yes1,
                    market2_no_price=no1,
                    correlation_type="Implicação lógica (A → B)",
                    inconsistency=f"P(A)={yes2:.1%} > P(B)={yes1:.1%}, mas A→B exige P(A)≤P(B)",
                    arbitrage_action=f"Vender YES em '{title2[:30]}...' a ${yes2:.2f} e Comprar YES em '{title1[:30]}...' a ${yes1:.2f}",
                    expected_profit=profit,
                    confidence=0.85,
                    explanation=f"'{title2[:40]}' implica '{title1[:40]}'. Se A acontece, B necessariamente acontece. Logo P(A) ≤ P(B).",
                    detected_at=datetime.utcnow()
                )
        
        return None
    
    def _are_mutually_exclusive(self, title1: str, title2: str) -> bool:
        """Verifica se dois eventos são mutuamente exclusivos."""
        t1 = title1.lower()
        t2 = title2.lower()
        
        # Mesmo candidato em diferentes posições
        candidates = ["trump", "biden", "harris", "desantis", "haley", "newsom"]
        for cand in candidates:
            if cand in t1 and cand in t2:
                # Verificar se são posições diferentes
                if ("republican" in t1 and "democratic" in t2) or \
                   ("democratic" in t1 and "republican" in t2):
                    return True
        
        # Ranges numéricos não sobrepostos
        # Ex: "420-439 tweets" vs "440-459 tweets"
        range1 = self._extract_range(t1)
        range2 = self._extract_range(t2)
        if range1 and range2:
            # Verificar se não há sobreposição
            if range1[1] < range2[0] or range2[1] < range1[0]:
                # Verificar se são do mesmo tema
                if self._same_topic(t1, t2):
                    return True
        
        return False
    
    def _check_implication(self, title1: str, title2: str) -> int:
        """
        Verifica se há implicação lógica entre os títulos.
        Retorna: 1 se title1 implica title2, -1 se title2 implica title1, 0 se não há
        
        IMPORTANTE: Só retorna implicação se for logicamente válida!
        """
        t1 = title1.lower()
        t2 = title2.lower()
        
        # VERIFICAR SE NÃO SÃO RANGES MUTUAMENTE EXCLUSIVOS
        range1 = self._extract_range(t1)
        range2 = self._extract_range(t2)
        if range1 and range2:
            # Ranges são mutuamente exclusivos, não há implicação
            return 0
        
        # Datas: "ceasefire by January" implica "ceasefire by March"
        # Se acontece ATÉ janeiro, necessariamente acontece ATÉ março (posterior)
        # Então: "by Jan" → "by Mar", e P(Jan) <= P(Mar)
        if "by " in t1 and "by " in t2:
            date1 = self._extract_date(t1)
            date2 = self._extract_date(t2)
            if date1 and date2 and self._same_topic(t1, t2):
                # Converter datas para comparação
                month_order = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
                    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'june': 6,
                    'july': 7, 'august': 8, 'september': 9, 'october': 10, 
                    'november': 11, 'december': 12
                }
                
                m1 = None
                m2 = None
                for month, num in month_order.items():
                    if month in date1:
                        m1 = num
                    if month in date2:
                        m2 = num
                
                if m1 and m2:
                    # Extrair anos se disponíveis
                    year1 = self._extract_year(t1)
                    year2 = self._extract_year(t2)
                    
                    # Criar valor comparável (ano*12 + mês)
                    if year1 and year2:
                        v1 = year1 * 12 + m1
                        v2 = year2 * 12 + m2
                    else:
                        # Se não tem ano, considerar virada de ano (dez->jan)
                        if m1 == 12 and m2 <= 3:
                            v1 = 2025 * 12 + m1  # Assumir dezembro é 2025
                            v2 = 2026 * 12 + m2  # Janeiro é 2026
                        elif m2 == 12 and m1 <= 3:
                            v2 = 2025 * 12 + m2
                            v1 = 2026 * 12 + m1
                        else:
                            v1 = m1
                            v2 = m2
                    
                    # Data mais cedo implica data mais tarde
                    if v1 < v2:
                        return 1  # t1 (data mais cedo) implica t2 (data mais tarde)
                    elif v2 < v1:
                        return -1
        
        # Valores com > ou <: "BTC > $150k" implica "BTC > $100k"
        if (">" in t1 or "above" in t1) and (">" in t2 or "above" in t2):
            val1 = self._extract_threshold(t1)
            val2 = self._extract_threshold(t2)
            if val1 and val2 and self._same_topic(t1, t2):
                if val1 > val2:
                    return 1  # > val1 implica > val2
                elif val2 > val1:
                    return -1
        
        if ("<" in t1 or "below" in t1) and ("<" in t2 or "below" in t2):
            val1 = self._extract_threshold(t1)
            val2 = self._extract_threshold(t2)
            if val1 and val2 and self._same_topic(t1, t2):
                if val1 < val2:
                    return 1  # < val1 implica < val2
                elif val2 < val1:
                    return -1
        
        # Candidato específico vs partido
        # "Trump wins Republican nomination" implica "Republican wins presidency" NÃO!
        # "Trump wins presidency" implica "Republican wins presidency" SIM!
        if "trump" in t1 and "republican" in t2:
            if "president" in t1 and "president" in t2 and "win" in t1 and "win" in t2:
                if "nomination" not in t1 and "nomination" not in t2:
                    return 1
        if "trump" in t2 and "republican" in t1:
            if "president" in t1 and "president" in t2 and "win" in t1 and "win" in t2:
                if "nomination" not in t1 and "nomination" not in t2:
                    return -1
        
        return 0
    
    def _extract_range(self, text: str) -> Optional[Tuple[int, int]]:
        """Extrai range numérico do texto."""
        # Pattern: 420-439 ou 420 to 439
        match = re.search(r'(\d+)\s*[-–to]+\s*(\d+)', text)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extrai data do texto."""
        # Patterns comuns
        patterns = [
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}',
            r'\d{4}-\d{2}-\d{2}',
            r'by\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).lower()
        return None
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extrai ano do texto."""
        match = re.search(r'20\d{2}', text)
        if match:
            return int(match.group(0))
        return None
    
    def _extract_threshold(self, text: str) -> Optional[float]:
        """Extrai valor de threshold do texto."""
        # Pattern: >$100k, >100000, >$1B
        match = re.search(r'[><]\s*\$?([\d,.]+)\s*(k|m|b|thousand|million|billion)?', text, re.IGNORECASE)
        if match:
            value = float(match.group(1).replace(',', ''))
            multiplier = match.group(2)
            if multiplier:
                mult_lower = multiplier.lower()
                if mult_lower in ['k', 'thousand']:
                    value *= 1000
                elif mult_lower in ['m', 'million']:
                    value *= 1000000
                elif mult_lower in ['b', 'billion']:
                    value *= 1000000000
            return value
        return None
    
    def _same_topic(self, t1: str, t2: str) -> bool:
        """Verifica se dois textos são sobre o mesmo tópico."""
        # Extrair palavras-chave significativas
        keywords1 = set(re.findall(r'\b[a-z]{4,}\b', t1.lower()))
        keywords2 = set(re.findall(r'\b[a-z]{4,}\b', t2.lower()))
        
        # Remover palavras comuns
        common_words = {'will', 'the', 'from', 'this', 'that', 'with', 'have', 'been', 'would', 'could', 'should'}
        keywords1 -= common_words
        keywords2 -= common_words
        
        # Verificar sobreposição
        overlap = len(keywords1 & keywords2)
        total = min(len(keywords1), len(keywords2))
        
        if total == 0:
            return False
        
        return overlap / total > 0.5
    
    async def _analyze_temporal_correlations(self) -> List[CorrelatedOpportunity]:
        """Analisa correlações temporais (evento até data X vs até data Y)."""
        opportunities = []
        
        # Agrupar mercados por tema temporal
        temporal_groups = defaultdict(list)
        
        for market in self.markets_cache:
            question = market.get("question", "").lower()
            date = self._extract_date(question)
            
            if date:
                # Extrair tema (remover a data)
                topic = re.sub(r'(by\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)(\s+\d{1,2}(,?\s+\d{4})?)?', '', question, flags=re.IGNORECASE)
                topic = re.sub(r'\d{4}-\d{2}-\d{2}', '', topic)
                topic = ' '.join(topic.split())[:50]
                
                temporal_groups[topic].append({
                    "market": market,
                    "date": date
                })
        
        # Analisar cada grupo temporal
        for topic, markets in temporal_groups.items():
            if len(markets) < 2:
                continue
            
            # Ordenar por data
            # (simplificado - em produção seria mais robusto)
            for i, m1 in enumerate(markets):
                for m2 in markets[i+1:]:
                    opp = self._check_correlation(m1["market"], m2["market"], "temporal")
                    if opp:
                        opportunities.append(opp)
        
        return opportunities
    
    async def _analyze_numeric_correlations(self) -> List[CorrelatedOpportunity]:
        """Analisa correlações numéricas (>$100k vs >$150k)."""
        opportunities = []
        
        # Agrupar por tema com threshold
        numeric_groups = defaultdict(list)
        
        for market in self.markets_cache:
            question = market.get("question", "").lower()
            threshold = self._extract_threshold(question)
            
            if threshold:
                # Extrair tema
                topic = re.sub(r'[><]\s*\$?[\d,.]+\s*(k|m|b|thousand|million|billion)?', '', question, flags=re.IGNORECASE)
                topic = ' '.join(topic.split())[:50]
                
                numeric_groups[topic].append({
                    "market": market,
                    "threshold": threshold
                })
        
        for topic, markets in numeric_groups.items():
            if len(markets) < 2:
                continue
            
            for i, m1 in enumerate(markets):
                for m2 in markets[i+1:]:
                    opp = self._check_correlation(m1["market"], m2["market"], "numeric")
                    if opp:
                        opportunities.append(opp)
        
        return opportunities
    
    async def _analyze_subset_correlations(self) -> List[CorrelatedOpportunity]:
        """Analisa correlações de subconjunto (candidato vs partido)."""
        opportunities = []
        
        # Buscar mercados de eleição
        election_markets = [m for m in self.markets_cache 
                          if any(word in m.get("question", "").lower() 
                                for word in ["president", "election", "win", "nominee"])]
        
        for i, m1 in enumerate(election_markets):
            for m2 in election_markets[i+1:]:
                opp = self._check_correlation(m1, m2, "election")
                if opp:
                    opportunities.append(opp)
        
        return opportunities


async def run_correlation_scan(limit: int = 200) -> List[CorrelatedOpportunity]:
    """Executa scan de correlação."""
    async with CorrelationArbitrageScanner(timeout=120.0) as scanner:
        return await scanner.scan(limit)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    
    opportunities = asyncio.run(run_correlation_scan(limit))
    
    print(f"\n{'='*80}")
    print(f"  ARBITRAGEM EM MERCADOS CORRELACIONADOS")
    print(f"{'='*80}\n")
    
    if not opportunities:
        print("Nenhuma oportunidade de arbitragem correlacionada encontrada.")
        print("\nIsso significa que os preços estão consistentes entre mercados relacionados.")
    else:
        for i, opp in enumerate(opportunities[:10], 1):
            print(f"#{i} - {opp.opportunity_type.upper()} - Lucro esperado: {opp.expected_profit:.1f}%")
            print(f"   Mercado 1: {opp.market1_title}")
            print(f"   Preço YES: ${opp.market1_yes_price:.3f}")
            print(f"   ")
            print(f"   Mercado 2: {opp.market2_title}")
            print(f"   Preço YES: ${opp.market2_yes_price:.3f}")
            print(f"   ")
            print(f"   Correlação: {opp.correlation_type}")
            print(f"   Inconsistência: {opp.inconsistency}")
            print(f"   Ação: {opp.arbitrage_action}")
            print(f"   Confiança: {opp.confidence*100:.0f}%")
            print(f"   ")
            print(f"   Explicação: {opp.explanation}")
            print()

