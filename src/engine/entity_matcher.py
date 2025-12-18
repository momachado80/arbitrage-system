"""
Entity-Based Market Matcher
===========================

Matching baseado em entidades específicas, não categorias genéricas.

Só pareia mercados que compartilham:
- Mesma pessoa (Trump, Musk, Biden, etc.)
- Mesma empresa (Tesla, Google, OpenAI, etc.)
- Mesmo evento específico (Super Bowl 2025, Fed January meeting, etc.)
- Mesmo ativo (Bitcoin, Ethereum, S&P 500, etc.)
- Mesma data específica (December 31 2025, Q1 2025, etc.)

Isso evita matches falsos como "unemployment" com "Jake Paul".
"""

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime

from src.models import Market

logger = logging.getLogger(__name__)


@dataclass
class EntityMatch:
    """Resultado de matching por entidade."""
    kalshi: Market
    polymarket: Market
    shared_entities: Set[str]
    entity_type: str  # person, company, event, asset, date
    confidence: float


class EntityBasedMatcher:
    """
    Matcher baseado em entidades específicas.
    
    Só pareia mercados que compartilham entidades concretas,
    não palavras genéricas.
    """
    
    # Pessoas específicas
    PEOPLE = {
        # Políticos US
        'trump', 'donald trump', 'biden', 'joe biden', 'harris', 'kamala harris',
        'desantis', 'ron desantis', 'newsom', 'gavin newsom', 'vance', 'jd vance',
        'pence', 'mike pence', 'obama', 'barack obama', 'pelosi', 'nancy pelosi',
        'mcconnell', 'mitch mcconnell', 'schumer', 'chuck schumer',
        'aoc', 'alexandria ocasio-cortez', 'bernie sanders', 'sanders',
        
        # Políticos internacionais
        'putin', 'vladimir putin', 'xi jinping', 'xi', 'zelensky', 'volodymyr zelensky',
        'netanyahu', 'benjamin netanyahu', 'kim jong un', 'kim jong-un',
        'modi', 'narendra modi', 'macron', 'emmanuel macron', 'scholz', 'olaf scholz',
        'sunak', 'rishi sunak', 'starmer', 'keir starmer', 'trudeau', 'justin trudeau',
        'bolsonaro', 'jair bolsonaro', 'lula', 'milei', 'javier milei',
        
        # Tech leaders
        'musk', 'elon musk', 'bezos', 'jeff bezos', 'zuckerberg', 'mark zuckerberg',
        'altman', 'sam altman', 'nadella', 'satya nadella', 'cook', 'tim cook',
        'pichai', 'sundar pichai', 'huang', 'jensen huang',
        
        # Celebridades/Atletas
        'taylor swift', 'beyonce', 'drake', 'kanye', 'rihanna',
        'lebron', 'lebron james', 'messi', 'ronaldo', 'cristiano ronaldo',
        'mahomes', 'patrick mahomes', 'brady', 'tom brady',
        'jake paul', 'logan paul', 'mike tyson',
        
        # Outros
        'powell', 'jerome powell', 'yellen', 'janet yellen',
        'pope francis', 'pope', 'dalai lama',
    }
    
    # Empresas/Organizações
    COMPANIES = {
        # Big Tech
        'apple', 'google', 'alphabet', 'microsoft', 'amazon', 'meta', 'facebook',
        'nvidia', 'tesla', 'netflix', 'twitter', 'x corp',
        
        # AI Companies
        'openai', 'anthropic', 'deepmind', 'xai', 'x.ai', 'mistral',
        'stability ai', 'midjourney', 'cohere', 'inflection',
        
        # Other Tech
        'spacex', 'neuralink', 'boring company', 'palantir', 'snowflake',
        'uber', 'lyft', 'airbnb', 'doordash', 'instacart',
        'tiktok', 'bytedance', 'snap', 'snapchat', 'pinterest',
        'salesforce', 'oracle', 'ibm', 'intel', 'amd', 'qualcomm',
        'coinbase', 'binance', 'kraken', 'ftx',
        
        # Finance
        'jpmorgan', 'goldman sachs', 'morgan stanley', 'blackrock',
        'citadel', 'bridgewater', 'berkshire', 'visa', 'mastercard',
        
        # Other
        'disney', 'warner', 'paramount', 'sony', 'nintendo',
        'boeing', 'lockheed', 'raytheon', 'northrop',
        'pfizer', 'moderna', 'johnson & johnson', 'merck',
    }
    
    # Ativos específicos
    ASSETS = {
        # Crypto
        'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol',
        'cardano', 'ada', 'dogecoin', 'doge', 'xrp', 'ripple',
        'bnb', 'binance coin', 'polkadot', 'dot', 'avalanche', 'avax',
        'chainlink', 'link', 'polygon', 'matic', 'litecoin', 'ltc',
        
        # Indices
        's&p 500', 's&p500', 'sp500', 'spy', 'dow jones', 'djia',
        'nasdaq', 'qqq', 'russell 2000', 'vix',
        
        # Commodities
        'gold', 'silver', 'oil', 'crude oil', 'wti', 'brent',
        'natural gas', 'copper', 'wheat', 'corn',
    }
    
    # AI Models específicos
    AI_MODELS = {
        'gpt-4', 'gpt-5', 'gpt4', 'gpt5', 'chatgpt', 'gpt',
        'claude', 'claude 3', 'claude 4', 'opus', 'sonnet',
        'gemini', 'gemini pro', 'gemini ultra', 'bard',
        'grok', 'grok 2', 'grok 3',
        'llama', 'llama 2', 'llama 3', 'llama 4',
        'mistral', 'mixtral',
        'copilot', 'dall-e', 'dalle', 'midjourney', 'stable diffusion', 'sora',
    }
    
    # Eventos específicos
    EVENTS = {
        # Esportes
        'super bowl', 'superbowl', 'world series', 'nba finals', 'stanley cup',
        'world cup', 'champions league', 'premier league', 'euro 2024', 'euro 2028',
        'olympics', 'olympic games', 'wimbledon', 'us open', 'french open',
        'masters', 'pga championship', 'ufc',
        
        # Política
        'presidential election', 'midterm', 'primary', 'caucus',
        'inauguration', 'state of the union', 'debate',
        'impeachment', 'supreme court',
        
        # Economia
        'fomc', 'fed meeting', 'rate decision', 'rate cut', 'rate hike',
        'cpi', 'inflation report', 'jobs report', 'nonfarm payroll',
        'gdp', 'earnings', 'ipo',
        
        # Tech
        'wwdc', 'google i/o', 'ces', 'mwc',
        
        # Awards
        'oscars', 'academy awards', 'grammy', 'grammys', 'emmy', 'emmys',
        'golden globes', 'tony awards', 'billboard',
    }
    
    # Padrões de data
    DATE_PATTERNS = [
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s*\d{4}\b',
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s*\d{4}\b',
        r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
        r'\b(q[1-4])\s*\d{4}\b',  # Q1 2025
        r'\b(end of |by end of )?\d{4}\b',  # 2025, end of 2025
        r'\b(december|january|february|march)\s+\d{4}\b',  # December 2025
    ]
    
    def __init__(self, min_shared_entities: int = 1):
        """
        Args:
            min_shared_entities: Mínimo de entidades em comum para considerar match
        """
        self.min_shared_entities = min_shared_entities
        
        # Criar lookup sets para busca rápida
        self.all_entities = {}
        for entity in self.PEOPLE:
            self.all_entities[entity.lower()] = 'person'
        for entity in self.COMPANIES:
            self.all_entities[entity.lower()] = 'company'
        for entity in self.ASSETS:
            self.all_entities[entity.lower()] = 'asset'
        for entity in self.AI_MODELS:
            self.all_entities[entity.lower()] = 'ai_model'
        for entity in self.EVENTS:
            self.all_entities[entity.lower()] = 'event'
    
    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """
        Extrai entidades de um texto.
        
        Returns:
            Dict com tipo -> set de entidades encontradas
        """
        text_lower = text.lower()
        found = defaultdict(set)
        
        # Buscar entidades conhecidas
        for entity, entity_type in self.all_entities.items():
            if entity in text_lower:
                # Verificar se é palavra completa (não parte de outra)
                pattern = r'\b' + re.escape(entity) + r'\b'
                if re.search(pattern, text_lower):
                    found[entity_type].add(entity)
        
        # Extrair datas
        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(match)
                found['date'].add(match.strip())
        
        # Extrair anos específicos (2024, 2025, 2026, etc.)
        years = re.findall(r'\b(202[4-9]|203[0-9])\b', text_lower)
        for year in years:
            found['year'].add(year)
        
        return dict(found)
    
    def get_shared_entities(
        self, 
        entities1: Dict[str, Set[str]], 
        entities2: Dict[str, Set[str]]
    ) -> Tuple[Set[str], str]:
        """
        Encontra entidades compartilhadas entre dois mercados.
        
        Returns:
            Tuple (set de entidades compartilhadas, tipo principal)
        """
        shared = set()
        primary_type = 'unknown'
        
        # Prioridade: person > company > ai_model > asset > event > date > year
        priority_order = ['person', 'company', 'ai_model', 'asset', 'event', 'date', 'year']
        
        for entity_type in priority_order:
            set1 = entities1.get(entity_type, set())
            set2 = entities2.get(entity_type, set())
            common = set1 & set2
            
            if common:
                shared.update(common)
                if primary_type == 'unknown':
                    primary_type = entity_type
        
        return shared, primary_type
    
    def find_matches(
        self,
        kalshi_markets: List[Market],
        poly_markets: List[Market],
        min_confidence: float = 0.3
    ) -> List[EntityMatch]:
        """
        Encontra matches entre mercados baseado em entidades.
        
        Args:
            kalshi_markets: Lista de mercados Kalshi
            poly_markets: Lista de mercados Polymarket
            min_confidence: Confiança mínima para incluir match
            
        Returns:
            Lista de EntityMatch ordenada por confiança
        """
        matches = []
        
        # Pré-extrair entidades de todos os mercados
        logger.info(f"Extraindo entidades de {len(kalshi_markets)} Kalshi + {len(poly_markets)} Poly...")
        
        kalshi_entities = {}
        for market in kalshi_markets:
            text = f"{market.title} {market.description or ''}"
            kalshi_entities[market.ticker] = {
                'market': market,
                'entities': self.extract_entities(text)
            }
        
        poly_entities = {}
        for market in poly_markets:
            text = f"{market.title} {market.description or ''}"
            poly_entities[market.ticker] = {
                'market': market,
                'entities': self.extract_entities(text)
            }
        
        # Criar índice invertido: entidade -> mercados
        kalshi_by_entity = defaultdict(list)
        for ticker, data in kalshi_entities.items():
            for entity_type, entities in data['entities'].items():
                for entity in entities:
                    kalshi_by_entity[entity].append(ticker)
        
        poly_by_entity = defaultdict(list)
        for ticker, data in poly_entities.items():
            for entity_type, entities in data['entities'].items():
                for entity in entities:
                    poly_by_entity[entity].append(ticker)
        
        # Encontrar entidades em comum
        common_entities = set(kalshi_by_entity.keys()) & set(poly_by_entity.keys())
        logger.info(f"Entidades em comum: {len(common_entities)}")
        
        # Para cada entidade em comum, criar pares
        seen_pairs = set()
        
        for entity in common_entities:
            kalshi_tickers = kalshi_by_entity[entity]
            poly_tickers = poly_by_entity[entity]
            
            for k_ticker in kalshi_tickers:
                for p_ticker in poly_tickers:
                    pair_key = (k_ticker, p_ticker)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    
                    k_data = kalshi_entities[k_ticker]
                    p_data = poly_entities[p_ticker]
                    
                    shared, primary_type = self.get_shared_entities(
                        k_data['entities'],
                        p_data['entities']
                    )
                    
                    if len(shared) >= self.min_shared_entities:
                        # Calcular confiança baseada em entidades compartilhadas
                        confidence = self._calculate_confidence(
                            k_data['entities'],
                            p_data['entities'],
                            shared,
                            primary_type
                        )
                        
                        if confidence >= min_confidence:
                            matches.append(EntityMatch(
                                kalshi=k_data['market'],
                                polymarket=p_data['market'],
                                shared_entities=shared,
                                entity_type=primary_type,
                                confidence=confidence
                            ))
        
        # Ordenar por confiança
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        logger.info(f"Matches encontrados: {len(matches)}")
        return matches
    
    def _calculate_confidence(
        self,
        entities1: Dict[str, Set[str]],
        entities2: Dict[str, Set[str]],
        shared: Set[str],
        primary_type: str
    ) -> float:
        """Calcula confiança do match."""
        
        # Base: número de entidades compartilhadas
        base_score = min(len(shared) / 3.0, 1.0)  # 3+ entidades = score máximo
        
        # Bônus por tipo de entidade (pessoas/empresas são mais específicas)
        type_bonus = {
            'person': 0.3,
            'company': 0.25,
            'ai_model': 0.25,
            'asset': 0.2,
            'event': 0.15,
            'date': 0.1,
            'year': 0.05,
        }
        bonus = type_bonus.get(primary_type, 0)
        
        # Bônus se compartilham mesmo ano
        years1 = entities1.get('year', set())
        years2 = entities2.get('year', set())
        if years1 & years2:
            bonus += 0.15
        
        # Penalidade se um tem muito mais entidades que outro (assimetria)
        total1 = sum(len(s) for s in entities1.values())
        total2 = sum(len(s) for s in entities2.values())
        if total1 > 0 and total2 > 0:
            ratio = min(total1, total2) / max(total1, total2)
            if ratio < 0.3:
                bonus -= 0.2
        
        return min(base_score + bonus, 1.0)
