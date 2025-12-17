"""
Advanced Market Matcher
=======================

Sistema avançado de matching de mercados entre Kalshi e Polymarket.

Usa análise de equivalência contratual completa:
1. Similaridade textual (título/descrição)
2. Regras de resolução
3. Fontes de verdade
4. Datas de expiração
5. Entidades nomeadas
6. Condições específicas

Cada par recebe um score de equivalência que determina:
- Se o par deve ser considerado para arbitragem
- Qual penalidade de risco aplicar
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from src.models import Market, Platform
from src.engine.contract_equivalence import (
    ContractEquivalenceAnalyzer,
    EquivalenceResult,
    EquivalenceLevel,
    equivalence_to_penalty,
)

logger = logging.getLogger(__name__)


@dataclass
class MarketPair:
    """Par de mercados potencialmente equivalentes."""
    kalshi: Market
    polymarket: Market
    equivalence: EquivalenceResult
    
    @property
    def similarity_score(self) -> float:
        """Score de similaridade para compatibilidade."""
        return float(self.equivalence.score)
    
    @property
    def matched_keywords(self) -> List[str]:
        """Keywords em comum."""
        return self.equivalence.details.get("kalshi_entities", [])
    
    @property
    def match_reason(self) -> str:
        """Razão do match."""
        return self.equivalence.recommendation
    
    @property
    def is_valid_for_arbitrage(self) -> bool:
        """Verifica se o par é válido para arbitragem."""
        return self.equivalence.level in [
            EquivalenceLevel.IDENTICAL,
            EquivalenceLevel.HIGH_CONFIDENCE,
            EquivalenceLevel.MEDIUM_CONFIDENCE,
        ]
    
    @property
    def risk_penalty(self) -> Decimal:
        """Penalidade de risco baseada na equivalência."""
        return equivalence_to_penalty(self.equivalence)


class MarketMatcher:
    """
    Encontra mercados equivalentes entre Kalshi e Polymarket.
    
    Usa análise de equivalência contratual completa para
    determinar se dois mercados são suficientemente equivalentes
    para arbitragem segura.
    """
    
    # Entidades importantes para pré-filtragem rápida
    KEY_ENTITIES = {
        # Política
        'trump', 'biden', 'harris', 'musk', 'putin', 'xi', 'zelensky',
        # Economia
        'fed', 'fomc', 'rate cut', 'rate hike', 'inflation', 'recession',
        # Crypto
        'bitcoin', 'btc', 'ethereum', 'eth', 'crypto',
        # Empresas Tech
        'spacex', 'tesla', 'google', 'openai', 'anthropic', 'meta', 'apple', 'microsoft', 'nvidia', 'amazon',
        # IA - Modelos e empresas
        'gemini', 'gpt', 'chatgpt', 'claude', 'grok', 'llama', 'ai model', 'best ai', 'artificial intelligence',
        'xai', 'deepmind', 'midjourney', 'stable diffusion', 'copilot',
        # Esportes
        'super bowl', 'superbowl', 'oscars', 'grammy', 'world cup', 'nfl', 'nba', 'champions league',
        # Geopolítica
        'china', 'russia', 'ukraine', 'israel', 'gaza', 'taiwan', 'war', 'invasion',
        # Outros
        'pope', 'nasa', 'mars', 'moon', 'spacex', 'starship',
        'election', 'president', 'congress', 'senate',
    }
    
    def __init__(
        self,
        min_similarity: float = 0.2,  # Reduzido para ser mais permissivo
        require_date_match: bool = False,  # Não exigir match de data
    ):
        """
        Args:
            min_similarity: Score mínimo para considerar um par (0-1)
            require_date_match: Se deve exigir datas compatíveis
        """
        self.min_similarity = min_similarity
        self.analyzer = ContractEquivalenceAnalyzer(
            min_title_similarity=min_similarity,
            require_date_match=require_date_match,
        )
    
    def _extract_entities(self, text: str) -> set:
        """Extrai entidades do texto para pré-filtragem."""
        text = text.lower()
        entities = set()
        for entity in self.KEY_ENTITIES:
            if entity in text:
                entities.add(entity)
        return entities
    
    def _quick_match(self, kalshi: Market, poly: Market) -> bool:
        """
        Pré-filtragem rápida para evitar análise completa de pares irrelevantes.
        Retorna True se vale a pena fazer análise completa.
        """
        # Extrair entidades de ambos
        kalshi_text = f"{kalshi.title} {kalshi.description or ''}".lower()
        poly_text = f"{poly.title} {poly.description or ''}".lower()
        
        kalshi_entities = self._extract_entities(kalshi_text)
        poly_entities = self._extract_entities(poly_text)
        
        # Se têm pelo menos uma entidade em comum, vale a pena analisar
        if kalshi_entities & poly_entities:
            return True
        
        # Também verificar similaridade textual básica de palavras
        kalshi_words = set(kalshi_text.split())
        poly_words = set(poly_text.split())
        
        # Filtrar palavras muito curtas
        kalshi_words = {w for w in kalshi_words if len(w) > 3}
        poly_words = {w for w in poly_words if len(w) > 3}
        
        common_words = kalshi_words & poly_words
        
        # Se têm 3+ palavras em comum, vale analisar
        if len(common_words) >= 3:
            return True
        
        return False
    
    def find_pairs(
        self,
        kalshi_markets: List[Market],
        poly_markets: List[Market],
        top_n: int = 50
    ) -> List[MarketPair]:
        """
        Encontra pares de mercados potencialmente equivalentes.
        
        Args:
            kalshi_markets: Lista de mercados Kalshi
            poly_markets: Lista de mercados Polymarket
            top_n: Número máximo de pares a retornar
        
        Returns:
            Lista de MarketPair ordenada por equivalence score
        """
        pairs = []
        total_comparisons = len(kalshi_markets) * len(poly_markets)
        quick_matches = 0
        
        logger.info(f"Iniciando matching: {len(kalshi_markets)} Kalshi x {len(poly_markets)} Poly = {total_comparisons} comparações possíveis")
        
        for kalshi in kalshi_markets:
            for poly in poly_markets:
                # Pré-filtragem rápida
                if not self._quick_match(kalshi, poly):
                    continue
                
                quick_matches += 1
                
                # Análise de equivalência completa
                equivalence = self.analyzer.analyze(
                    kalshi_title=kalshi.title,
                    kalshi_description=kalshi.description or "",
                    kalshi_rules=kalshi.resolution_rules,
                    kalshi_end_date=kalshi.close_time or kalshi.expiration_time,
                    poly_title=poly.title,
                    poly_description=poly.description or "",
                    poly_rules=poly.resolution_rules,
                    poly_end_date=poly.close_time,
                )
                
                # Incluir se atinge score mínimo
                if float(equivalence.score) >= self.min_similarity:
                    pair = MarketPair(
                        kalshi=kalshi,
                        polymarket=poly,
                        equivalence=equivalence,
                    )
                    pairs.append(pair)
                    logger.info(
                        f"Par encontrado: {kalshi.ticker} <-> {poly.ticker} | "
                        f"Score: {equivalence.score:.2f} | "
                        f"Nível: {equivalence.level.value}"
                    )
        
        logger.info(f"Pré-filtragem: {quick_matches} candidatos de {total_comparisons} comparações")
        logger.info(f"Total de pares com score >= {self.min_similarity}: {len(pairs)}")
        
        # Ordenar por score (maior primeiro)
        pairs.sort(key=lambda p: float(p.equivalence.score), reverse=True)
        
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


def similarity_to_equivalence_score(similarity: float) -> Decimal:
    """
    Converte similarity score em equivalence score para o calculador.
    
    DEPRECATED: Use MarketPair.risk_penalty diretamente.
    Mantido para compatibilidade.
    """
    if similarity >= 0.8:
        return Decimal("1.0")
    elif similarity >= 0.6:
        return Decimal("0.9")
    elif similarity >= 0.4:
        return Decimal("0.7")
    else:
        return Decimal("0.5")
