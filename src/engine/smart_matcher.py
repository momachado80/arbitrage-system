"""
Smart Market Matcher
====================

Matching inteligente focado nas categorias onde Kalshi e Polymarket
têm mercados equivalentes:

1. SPORTS: Times esportivos (NFL, NBA, MLB, NHL, etc.)
2. POLITICS: Eleições, congressional control, etc.
3. ECONOMICS: Fed rates, CPI, inflation, GDP, jobs
4. CRYPTO: Bitcoin price targets, ETF approvals
5. EVENTS: Awards (Oscars, Grammys), specific dated events

A estratégia é:
1. Normalizar títulos removendo variações de formato
2. Extrair "evento base" de cada mercado
3. Comparar eventos base para encontrar equivalências
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from difflib import SequenceMatcher

from src.models import Market

logger = logging.getLogger(__name__)


@dataclass
class NormalizedMarket:
    """Mercado com informações normalizadas."""
    original: Market
    category: str  # sports, politics, economics, crypto, events
    event_type: str  # nfl_game, election, fed_rate, btc_price, etc.
    normalized_title: str
    key_entities: Set[str]  # times, candidatos, valores, etc.
    date_info: Optional[str]  # data relevante se houver


@dataclass
class SmartMatch:
    """Match inteligente entre mercados."""
    kalshi: Market
    polymarket: Market
    category: str
    event_type: str
    match_score: float  # 0-1
    match_reason: str


class SmartMarketMatcher:
    """
    Matcher inteligente que entende a estrutura dos mercados.
    """
    
    # Padrões de normalização
    TEAM_ALIASES = {
        # NFL
        'chiefs': 'kansas city chiefs', 'kc': 'kansas city chiefs',
        'eagles': 'philadelphia eagles', 'philly': 'philadelphia eagles',
        'bills': 'buffalo bills',
        'ravens': 'baltimore ravens',
        '49ers': 'san francisco 49ers', 'niners': 'san francisco 49ers',
        'cowboys': 'dallas cowboys',
        'lions': 'detroit lions',
        'packers': 'green bay packers',
        'dolphins': 'miami dolphins',
        'jets': 'new york jets',
        'giants': 'new york giants',
        'patriots': 'new england patriots', 'pats': 'new england patriots',
        'steelers': 'pittsburgh steelers',
        'bengals': 'cincinnati bengals',
        'browns': 'cleveland browns',
        'texans': 'houston texans',
        'colts': 'indianapolis colts',
        'jaguars': 'jacksonville jaguars', 'jags': 'jacksonville jaguars',
        'titans': 'tennessee titans',
        'broncos': 'denver broncos',
        'chargers': 'los angeles chargers', 'la chargers': 'los angeles chargers',
        'raiders': 'las vegas raiders',
        'seahawks': 'seattle seahawks',
        'cardinals': 'arizona cardinals',
        'rams': 'los angeles rams', 'la rams': 'los angeles rams',
        'falcons': 'atlanta falcons',
        'panthers': 'carolina panthers',
        'saints': 'new orleans saints',
        'buccaneers': 'tampa bay buccaneers', 'bucs': 'tampa bay buccaneers',
        'bears': 'chicago bears',
        'vikings': 'minnesota vikings',
        'commanders': 'washington commanders',
        
        # NBA
        'lakers': 'los angeles lakers', 'la lakers': 'los angeles lakers',
        'celtics': 'boston celtics',
        'warriors': 'golden state warriors', 'gsw': 'golden state warriors',
        'nuggets': 'denver nuggets',
        'heat': 'miami heat',
        'bucks': 'milwaukee bucks',
        'suns': 'phoenix suns',
        'sixers': 'philadelphia 76ers', '76ers': 'philadelphia 76ers',
        'knicks': 'new york knicks',
        'nets': 'brooklyn nets',
        'clippers': 'los angeles clippers', 'la clippers': 'los angeles clippers',
        'mavericks': 'dallas mavericks', 'mavs': 'dallas mavericks',
        'thunder': 'oklahoma city thunder', 'okc': 'oklahoma city thunder',
        'timberwolves': 'minnesota timberwolves', 'wolves': 'minnesota timberwolves',
        'pelicans': 'new orleans pelicans',
        'grizzlies': 'memphis grizzlies',
        'kings': 'sacramento kings',
        'spurs': 'san antonio spurs',
        'rockets': 'houston rockets',
        'jazz': 'utah jazz',
        'blazers': 'portland trail blazers', 'trail blazers': 'portland trail blazers',
        'cavaliers': 'cleveland cavaliers', 'cavs': 'cleveland cavaliers',
        'pistons': 'detroit pistons',
        'pacers': 'indiana pacers',
        'bulls': 'chicago bulls',
        'hawks': 'atlanta hawks',
        'hornets': 'charlotte hornets',
        'magic': 'orlando magic',
        'raptors': 'toronto raptors',
        'wizards': 'washington wizards',
    }
    
    # Padrões de categorias
    CATEGORY_PATTERNS = {
        'sports': [
            r'\b(nfl|nba|mlb|nhl|ncaa|college football|super bowl|world series|stanley cup|nba finals)\b',
            r'\b(win|beat|defeat|vs\.?|versus|game|match|playoffs?|championship)\b',
            r'\b(chiefs|eagles|lakers|celtics|yankees|dodgers|' + '|'.join(TEAM_ALIASES.keys()) + r')\b',
        ],
        'politics': [
            r'\b(election|president|congress|senate|house|governor|vote|electoral|nominee|primary)\b',
            r'\b(trump|biden|harris|democrat|republican|gop|dnc|rnc)\b',
            r'\b(win.*election|become.*president|control.*senate|control.*house)\b',
        ],
        'economics': [
            r'\b(fed|federal reserve|interest rate|rate cut|rate hike|fomc)\b',
            r'\b(cpi|inflation|gdp|unemployment|jobs report|pce)\b',
            r'\b(recession|economic|economy)\b',
        ],
        'crypto': [
            r'\b(bitcoin|btc|ethereum|eth|crypto|solana|sol)\b',
            r'\b(price.*\$[\d,]+|\$[\d,]+.*price|reach.*\$|hit.*\$)\b',
            r'\b(etf|spot.*etf|bitcoin.*etf)\b',
        ],
        'events': [
            r'\b(oscar|grammy|emmy|golden globe|academy award)\b',
            r'\b(ipo|launch|release|announce)\b',
        ],
    }
    
    # Padrões de extração de eventos
    EVENT_EXTRACTORS = {
        'sports_game': r'(?:will\s+)?(?:the\s+)?(.+?)\s+(?:win|beat|defeat|vs\.?|versus)\s+(.+?)(?:\s+on|\s+in|\?|$)',
        'sports_championship': r'(?:will\s+)?(?:the\s+)?(.+?)\s+win\s+(?:the\s+)?(.+?)(?:\s+in|\s+\d{4}|\?|$)',
        'election': r'(?:will\s+)?(.+?)\s+(?:win|become|be elected)\s+(?:the\s+)?(.+?)(?:\s+in|\s+\d{4}|\?|$)',
        'price_target': r'(?:will\s+)?(.+?)\s+(?:reach|hit|exceed|be above|be below)\s+\$?([\d,\.]+)',
        'fed_rate': r'(?:will\s+)?(?:the\s+)?fed(?:eral reserve)?\s+(.+?)\s+(?:rate|rates)',
        'binary_event': r'(?:will\s+)?(.+?)\s+(?:happen|occur|take place|be announced)',
    }
    
    def __init__(self, min_match_score: float = 0.7):
        self.min_match_score = min_match_score
    
    def normalize_market(self, market: Market) -> NormalizedMarket:
        """Normaliza um mercado para matching."""
        title_lower = market.title.lower().strip()
        
        # Detectar categoria
        category = self._detect_category(title_lower)
        
        # Normalizar título
        normalized = self._normalize_title(title_lower)
        
        # Extrair tipo de evento e entidades
        event_type, entities = self._extract_event_info(normalized, category)
        
        # Extrair data se houver
        date_info = self._extract_date(title_lower)
        
        return NormalizedMarket(
            original=market,
            category=category,
            event_type=event_type,
            normalized_title=normalized,
            key_entities=entities,
            date_info=date_info,
        )
    
    def _detect_category(self, title: str) -> str:
        """Detecta categoria do mercado."""
        scores = {}
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, title, re.IGNORECASE):
                    score += 1
            scores[category] = score
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'other'
    
    def _normalize_title(self, title: str) -> str:
        """Normaliza título do mercado."""
        normalized = title.lower()
        
        # Remover prefixos comuns
        normalized = re.sub(r'^(will\s+)?(the\s+)?', '', normalized)
        
        # Expandir aliases de times
        for alias, full_name in self.TEAM_ALIASES.items():
            normalized = re.sub(r'\b' + re.escape(alias) + r'\b', full_name, normalized)
        
        # Normalizar números/preços
        normalized = re.sub(r'\$\s*([\d,]+)', r'$\1', normalized)
        normalized = re.sub(r'(\d),(\d{3})', r'\1\2', normalized)
        
        # Remover pontuação extra
        normalized = re.sub(r'[?!.]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _extract_event_info(self, title: str, category: str) -> Tuple[str, Set[str]]:
        """Extrai tipo de evento e entidades chave."""
        entities = set()
        event_type = 'unknown'
        
        if category == 'sports':
            # Buscar times
            for alias, full_name in self.TEAM_ALIASES.items():
                if full_name in title or alias in title:
                    entities.add(full_name)
            
            # Detectar tipo
            if re.search(r'super bowl|championship|finals|world series', title):
                event_type = 'championship'
            elif re.search(r'win|beat|vs|versus', title):
                event_type = 'game'
            elif re.search(r'playoffs?', title):
                event_type = 'playoffs'
        
        elif category == 'politics':
            # Buscar candidatos/partidos
            politicians = re.findall(r'\b(trump|biden|harris|desantis|newsom|obama)\b', title)
            entities.update(politicians)
            
            if re.search(r'president|presidential', title):
                event_type = 'presidential'
            elif re.search(r'senate', title):
                event_type = 'senate'
            elif re.search(r'house', title):
                event_type = 'house'
            elif re.search(r'governor', title):
                event_type = 'governor'
        
        elif category == 'economics':
            if re.search(r'fed|fomc|rate', title):
                event_type = 'fed_rate'
            elif re.search(r'cpi|inflation', title):
                event_type = 'inflation'
            elif re.search(r'gdp', title):
                event_type = 'gdp'
            elif re.search(r'unemployment|jobs', title):
                event_type = 'employment'
        
        elif category == 'crypto':
            cryptos = re.findall(r'\b(bitcoin|btc|ethereum|eth|solana|sol)\b', title)
            entities.update([c.replace('btc', 'bitcoin').replace('eth', 'ethereum') for c in cryptos])
            
            # Extrair preço alvo
            prices = re.findall(r'\$?([\d,]+)(?:k|K)?', title)
            for p in prices:
                p_clean = p.replace(',', '')
                if p_clean.isdigit() and int(p_clean) > 1000:
                    entities.add(f'${p_clean}')
            
            event_type = 'price_target'
        
        return event_type, entities
    
    def _extract_date(self, title: str) -> Optional[str]:
        """Extrai data relevante do título."""
        # Padrões de data
        patterns = [
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s+\d{1,2},?\s*\d{4}',
            r'\b(\d{4})\b',
            r'(q[1-4])\s*\d{4}',
            r'(end of|by|before|after)\s+(january|february|march|april|may|june|july|august|september|october|november|december|\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(0).lower()
        
        return None
    
    def find_matches(
        self,
        kalshi_markets: List[Market],
        poly_markets: List[Market],
    ) -> List[SmartMatch]:
        """
        Encontra matches entre mercados das duas plataformas.
        Usa índice invertido por entidades para evitar O(n*m).
        """
        matches = []
        
        # Normalizar todos os mercados
        logger.info(f"Normalizando {len(kalshi_markets)} Kalshi + {len(poly_markets)} Poly mercados...")
        
        kalshi_normalized = {m.ticker: self.normalize_market(m) for m in kalshi_markets}
        poly_normalized = {m.ticker: self.normalize_market(m) for m in poly_markets}
        
        # Criar índice invertido: entidade -> mercados Polymarket
        poly_by_entity = {}
        for ticker, nm in poly_normalized.items():
            for entity in nm.key_entities:
                if entity not in poly_by_entity:
                    poly_by_entity[entity] = []
                poly_by_entity[entity].append(ticker)
        
        logger.info(f"Índice de entidades criado: {len(poly_by_entity)} entidades únicas")
        
        # Para cada mercado Kalshi, buscar Polymarket com entidades em comum
        seen_pairs = set()
        
        for k_ticker, k_norm in kalshi_normalized.items():
            if not k_norm.key_entities:
                continue
            
            # Encontrar mercados Poly que compartilham entidades
            candidate_poly_tickers = set()
            for entity in k_norm.key_entities:
                if entity in poly_by_entity:
                    candidate_poly_tickers.update(poly_by_entity[entity])
            
            # Avaliar cada candidato
            for p_ticker in candidate_poly_tickers:
                pair_key = (k_ticker, p_ticker)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                p_norm = poly_normalized[p_ticker]
                
                # Pular se categorias são muito diferentes
                if k_norm.category != p_norm.category and k_norm.category != 'other' and p_norm.category != 'other':
                    continue
                
                # Calcular score de match
                score, reason = self._calculate_match_score(k_norm, p_norm)
                
                if score >= self.min_match_score:
                    matches.append(SmartMatch(
                        kalshi=k_norm.original,
                        polymarket=p_norm.original,
                        category=k_norm.category if k_norm.category != 'other' else p_norm.category,
                        event_type=k_norm.event_type or p_norm.event_type,
                        match_score=score,
                        match_reason=reason,
                    ))
        
        # Ordenar por score
        matches.sort(key=lambda m: m.match_score, reverse=True)
        
        logger.info(f"Matches encontrados: {len(matches)}")
        return matches
    
    def _calculate_match_score(
        self,
        k: NormalizedMarket,
        p: NormalizedMarket,
    ) -> Tuple[float, str]:
        """Calcula score de match entre dois mercados normalizados."""
        scores = []
        reasons = []
        
        # 1. Overlap de entidades (peso alto)
        if k.key_entities and p.key_entities:
            overlap = len(k.key_entities & p.key_entities)
            total = len(k.key_entities | p.key_entities)
            entity_score = overlap / total if total > 0 else 0
            
            if entity_score > 0:
                scores.append(entity_score * 1.5)  # Peso 1.5
                reasons.append(f"Entidades: {k.key_entities & p.key_entities}")
        
        # 2. Similaridade de título normalizado
        title_sim = SequenceMatcher(None, k.normalized_title, p.normalized_title).ratio()
        scores.append(title_sim)
        if title_sim > 0.5:
            reasons.append(f"Títulos similares ({title_sim:.0%})")
        
        # 3. Match de data (se ambos têm)
        if k.date_info and p.date_info:
            if k.date_info == p.date_info:
                scores.append(1.0)
                reasons.append(f"Mesma data: {k.date_info}")
            else:
                # Datas diferentes = penalidade forte
                scores.append(-0.5)
                reasons.append(f"Datas diferentes: {k.date_info} vs {p.date_info}")
        
        # 4. Mesmo tipo de evento
        if k.event_type == p.event_type and k.event_type != 'unknown':
            scores.append(0.3)
            reasons.append(f"Tipo: {k.event_type}")
        
        # Calcular média ponderada
        if not scores:
            return 0.0, "Sem similaridade"
        
        final_score = max(0, min(1, sum(scores) / len(scores)))
        return final_score, "; ".join(reasons) if reasons else "Match genérico"
