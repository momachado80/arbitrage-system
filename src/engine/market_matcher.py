"""
Automatic Market Matcher
========================

Sistema de matching automático de mercados entre Kalshi e Polymarket.

Usa múltiplas estratégias para identificar mercados equivalentes:
1. Similaridade textual (título/descrição)
2. Palavras-chave em comum
3. Entidades nomeadas (pessoas, organizações, datas)
4. Categorias/tags similares

Cada par recebe um score de equivalência (0-1) que é usado
para ajustar a penalidade de risco na análise de arbitragem.
"""

import re
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Tuple, Optional
from difflib import SequenceMatcher

from src.models import Market, Platform

logger = logging.getLogger(__name__)


@dataclass
class MarketPair:
    """Par de mercados potencialmente equivalentes."""
    kalshi: Market
    polymarket: Market
    similarity_score: float  # 0-1
    matched_keywords: List[str]
    match_reason: str


class MarketMatcher:
    """
    Encontra mercados equivalentes entre Kalshi e Polymarket.
    
    IMPORTANTE: Este é um matching heurístico. Para arbitragem real,
    os pares devem ser validados manualmente quanto à equivalência
    contratual (regras de resolução, fonte de verdade, etc.)
    """
    
    # Palavras a ignorar no matching
    STOP_WORDS = {
        'will', 'the', 'be', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
        'of', 'and', 'or', 'is', 'are', 'was', 'were', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'but', 'if', 'then',
        'than', 'so', 'as', 'by', 'with', 'from', 'about', 'into',
        'before', 'after', 'above', 'below', 'between', 'under', 'over',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
        'only', 'own', 'same', 'than', 'too', 'very', 'just', 'can',
        'could', 'may', 'might', 'must', 'shall', 'should', 'would',
    }
    
    # Entidades importantes para matching
    IMPORTANT_ENTITIES = {
        # Pessoas
        'trump', 'biden', 'harris', 'desantis', 'newsom', 'musk', 'bezos',
        'zuckerberg', 'powell', 'yellen', 'xi', 'putin', 'zelensky',
        # Organizações
        'fed', 'federal reserve', 'sec', 'ftc', 'fda', 'epa', 'nasa',
        'spacex', 'tesla', 'apple', 'google', 'meta', 'amazon', 'microsoft',
        'openai', 'anthropic', 'nvidia', 'bitcoin', 'ethereum',
        # Eventos
        'election', 'rate cut', 'rate hike', 'recession', 'gdp', 'inflation',
        'unemployment', 'superbowl', 'oscar', 'grammy', 'world cup',
        # Locais
        'usa', 'china', 'russia', 'ukraine', 'israel', 'gaza', 'taiwan',
        'california', 'texas', 'florida', 'new york', 'washington',
    }
    
    def __init__(self, min_similarity: float = 0.4):
        """
        Args:
            min_similarity: Score mínimo para considerar um par (0-1)
        """
        self.min_similarity = min_similarity
    
    def find_pairs(
        self,
        kalshi_markets: List[Market],
        poly_markets: List[Market],
        top_n: int = 20
    ) -> List[MarketPair]:
        """
        Encontra pares de mercados potencialmente equivalentes.
        
        Args:
            kalshi_markets: Lista de mercados Kalshi
            poly_markets: Lista de mercados Polymarket
            top_n: Número máximo de pares a retornar
        
        Returns:
            Lista de MarketPair ordenada por similarity_score
        """
        pairs = []
        
        for kalshi in kalshi_markets:
            for poly in poly_markets:
                score, keywords, reason = self._calculate_similarity(kalshi, poly)
                
                if score >= self.min_similarity:
                    pairs.append(MarketPair(
                        kalshi=kalshi,
                        polymarket=poly,
                        similarity_score=score,
                        matched_keywords=keywords,
                        match_reason=reason
                    ))
        
        # Ordenar por score (maior primeiro)
        pairs.sort(key=lambda p: p.similarity_score, reverse=True)
        
        # Remover duplicatas (mesmo mercado aparecendo em múltiplos pares)
        seen_kalshi = set()
        seen_poly = set()
        unique_pairs = []
        
        for pair in pairs:
            if pair.kalshi.ticker not in seen_kalshi and pair.polymarket.ticker not in seen_poly:
                unique_pairs.append(pair)
                seen_kalshi.add(pair.kalshi.ticker)
                seen_poly.add(pair.polymarket.ticker)
        
        return unique_pairs[:top_n]
    
    def _calculate_similarity(
        self,
        kalshi: Market,
        poly: Market
    ) -> Tuple[float, List[str], str]:
        """
        Calcula similaridade entre dois mercados.
        
        Returns:
            Tupla (score, keywords_matched, reason)
        """
        kalshi_text = self._normalize_text(f"{kalshi.title} {kalshi.description}")
        poly_text = self._normalize_text(f"{poly.title} {poly.description}")
        
        # Extrair palavras-chave
        kalshi_keywords = self._extract_keywords(kalshi_text)
        poly_keywords = self._extract_keywords(poly_text)
        
        # Palavras em comum
        common_keywords = kalshi_keywords & poly_keywords
        
        # Entidades importantes em comum
        kalshi_entities = self._extract_entities(kalshi_text)
        poly_entities = self._extract_entities(poly_text)
        common_entities = kalshi_entities & poly_entities
        
        # Similaridade de sequência (fuzzy matching)
        sequence_sim = SequenceMatcher(None, kalshi_text, poly_text).ratio()
        
        # Calcular score composto
        keyword_score = len(common_keywords) / max(len(kalshi_keywords | poly_keywords), 1)
        entity_score = len(common_entities) / max(len(kalshi_entities | poly_entities), 1) if (kalshi_entities or poly_entities) else 0
        
        # Peso maior para entidades (são mais específicas)
        final_score = (
            0.3 * sequence_sim +
            0.3 * keyword_score +
            0.4 * entity_score
        )
        
        # Boost se tiver entidades importantes em comum
        if common_entities:
            final_score = min(1.0, final_score * 1.3)
        
        # Determinar razão do match
        if common_entities:
            reason = f"Entidades: {', '.join(list(common_entities)[:3])}"
        elif common_keywords:
            reason = f"Keywords: {', '.join(list(common_keywords)[:3])}"
        else:
            reason = f"Similaridade textual: {sequence_sim:.0%}"
        
        return final_score, list(common_keywords | common_entities), reason
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparação."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_keywords(self, text: str) -> set:
        """Extrai palavras-chave significativas."""
        words = text.split()
        keywords = set()
        
        for word in words:
            if len(word) > 2 and word not in self.STOP_WORDS:
                keywords.add(word)
        
        return keywords
    
    def _extract_entities(self, text: str) -> set:
        """Extrai entidades importantes do texto."""
        entities = set()
        
        for entity in self.IMPORTANT_ENTITIES:
            if entity in text:
                entities.add(entity)
        
        return entities


def similarity_to_equivalence_score(similarity: float) -> Decimal:
    """
    Converte similarity score em equivalence score para o calculador.
    
    Similarity 0.8+ -> Equivalence 1.0 (confirmado)
    Similarity 0.6-0.8 -> Equivalence 0.9 (provável)
    Similarity 0.4-0.6 -> Equivalence 0.7 (ambíguo)
    Similarity <0.4 -> Não considerar
    """
    if similarity >= 0.8:
        return Decimal("1.0")
    elif similarity >= 0.6:
        return Decimal("0.9")
    elif similarity >= 0.4:
        return Decimal("0.7")
    else:
        return Decimal("0.5")
