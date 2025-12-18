"""
Market Indexer
==============

Indexa mercados por categoria/entidade para matching eficiente.

Em vez de comparar N × M mercados (bilhões de comparações),
agrupa por tema e só compara dentro do mesmo grupo.

Exemplo:
- Grupo "AI": mercados sobre IA de ambas plataformas
- Grupo "Elections": mercados sobre eleições
- Grupo "Crypto": mercados sobre Bitcoin, Ethereum, etc.

Isso reduz comparações de bilhões para milhares.
"""

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from datetime import datetime

from src.models import Market, Platform

logger = logging.getLogger(__name__)


@dataclass
class MarketIndex:
    """Índice de mercados agrupados por categoria."""
    kalshi_by_category: Dict[str, List[Market]] = field(default_factory=lambda: defaultdict(list))
    polymarket_by_category: Dict[str, List[Market]] = field(default_factory=lambda: defaultdict(list))
    kalshi_total: int = 0
    polymarket_total: int = 0
    indexed_at: Optional[datetime] = None
    
    def get_matching_categories(self) -> List[str]:
        """Retorna categorias que existem em ambas plataformas."""
        kalshi_cats = set(self.kalshi_by_category.keys())
        poly_cats = set(self.polymarket_by_category.keys())
        return list(kalshi_cats & poly_cats)
    
    def get_category_stats(self) -> Dict[str, Dict[str, int]]:
        """Estatísticas por categoria."""
        stats = {}
        for cat in self.get_matching_categories():
            stats[cat] = {
                "kalshi": len(self.kalshi_by_category[cat]),
                "polymarket": len(self.polymarket_by_category[cat]),
                "potential_pairs": len(self.kalshi_by_category[cat]) * len(self.polymarket_by_category[cat])
            }
        return stats


class MarketIndexer:
    """
    Indexa mercados por categoria para matching eficiente.
    """
    
    # Categorias e suas palavras-chave
    CATEGORIES = {
        "ai_models": {
            "keywords": [
                "ai", "artificial intelligence", "machine learning", "llm",
                "gpt", "chatgpt", "gemini", "claude", "grok", "llama",
                "openai", "anthropic", "google ai", "deepmind", "xai",
                "copilot", "midjourney", "stable diffusion", "dall-e", "sora",
                "best ai", "top ai", "ai model", "language model",
            ],
            "priority": 1,
        },
        "us_politics": {
            "keywords": [
                "trump", "biden", "harris", "desantis", "newsom", "pence",
                "vance", "walz", "pelosi", "mcconnell", "schumer",
                "president", "election", "electoral", "democrat", "republican",
                "congress", "senate", "house of representatives", "speaker",
                "impeach", "inauguration", "primary", "nomination",
            ],
            "priority": 1,
        },
        "fed_economy": {
            "keywords": [
                "fed", "federal reserve", "fomc", "rate cut", "rate hike",
                "interest rate", "powell", "inflation", "cpi", "pce",
                "recession", "gdp", "unemployment", "jobs report", "nonfarm",
                "treasury", "yield", "bond", "quantitative",
            ],
            "priority": 1,
        },
        "crypto": {
            "keywords": [
                "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
                "solana", "sol", "cardano", "ada", "dogecoin", "doge",
                "ripple", "xrp", "binance", "coinbase", "defi",
                "nft", "blockchain", "halving", "etf",
            ],
            "priority": 1,
        },
        "tech_companies": {
            "keywords": [
                "tesla", "apple", "google", "meta", "amazon", "microsoft",
                "nvidia", "spacex", "twitter", "x corp", "netflix",
                "stock", "ipo", "earnings", "market cap", "ceo",
                "musk", "bezos", "zuckerberg", "cook", "nadella", "pichai",
            ],
            "priority": 2,
        },
        "sports": {
            "keywords": [
                "super bowl", "superbowl", "nfl", "nba", "mlb", "nhl",
                "world cup", "fifa", "champions league", "premier league",
                "olympics", "tennis", "golf", "ufc", "boxing",
                "mvp", "championship", "playoff", "finals",
            ],
            "priority": 2,
        },
        "entertainment": {
            "keywords": [
                "oscars", "grammy", "emmys", "golden globes", "academy award",
                "movie", "film", "album", "artist", "actor", "actress",
                "box office", "streaming", "spotify", "billboard",
                "taylor swift", "beyonce", "drake",
            ],
            "priority": 3,
        },
        "world_politics": {
            "keywords": [
                "china", "russia", "ukraine", "israel", "gaza", "taiwan",
                "putin", "xi", "zelensky", "netanyahu", "kim jong",
                "war", "invasion", "nato", "un", "sanctions",
                "brexit", "eu", "european union",
            ],
            "priority": 2,
        },
        "science_space": {
            "keywords": [
                "nasa", "spacex", "mars", "moon", "starship", "rocket",
                "asteroid", "satellite", "iss", "astronaut",
                "climate", "temperature", "emissions", "carbon",
                "fda", "vaccine", "drug", "trial", "approval",
            ],
            "priority": 3,
        },
        "other": {
            "keywords": [],  # Catch-all
            "priority": 99,
        },
    }
    
    def __init__(self, min_liquidity: float = 0.0):
        """
        Args:
            min_liquidity: Liquidez mínima para incluir mercado (0 = todos)
        """
        self.min_liquidity = min_liquidity
    
    def index_markets(
        self, 
        kalshi_markets: List[Market], 
        poly_markets: List[Market]
    ) -> MarketIndex:
        """
        Indexa mercados de ambas plataformas por categoria.
        
        Args:
            kalshi_markets: Lista de mercados Kalshi
            poly_markets: Lista de mercados Polymarket
            
        Returns:
            MarketIndex com mercados agrupados
        """
        index = MarketIndex()
        index.indexed_at = datetime.utcnow()
        
        # Indexar Kalshi
        for market in kalshi_markets:
            if self._has_liquidity(market):
                categories = self._categorize_market(market)
                for cat in categories:
                    index.kalshi_by_category[cat].append(market)
                index.kalshi_total += 1
        
        # Indexar Polymarket
        for market in poly_markets:
            if self._has_liquidity(market):
                categories = self._categorize_market(market)
                for cat in categories:
                    index.polymarket_by_category[cat].append(market)
                index.polymarket_total += 1
        
        # Log estatísticas
        matching_cats = index.get_matching_categories()
        total_potential = sum(
            len(index.kalshi_by_category[c]) * len(index.polymarket_by_category[c])
            for c in matching_cats
        )
        
        logger.info(
            f"Indexação completa: {index.kalshi_total} Kalshi, {index.polymarket_total} Poly | "
            f"{len(matching_cats)} categorias em comum | "
            f"{total_potential:,} comparações potenciais"
        )
        
        return index
    
    def _has_liquidity(self, market: Market) -> bool:
        """Verifica se mercado tem liquidez mínima."""
        if self.min_liquidity == 0:
            return True
        
        # Verificar se tem orderbook com ordens
        if market.yes_book and market.yes_book.bids:
            return True
        if market.no_book and market.no_book.bids:
            return True
        
        return False
    
    def _categorize_market(self, market: Market) -> List[str]:
        """
        Determina categorias de um mercado.
        
        Returns:
            Lista de categorias (pode pertencer a múltiplas)
        """
        text = f"{market.title} {market.description or ''}".lower()
        
        categories = []
        
        for cat_name, cat_info in self.CATEGORIES.items():
            if cat_name == "other":
                continue
                
            for keyword in cat_info["keywords"]:
                if keyword in text:
                    categories.append(cat_name)
                    break
        
        # Se não encontrou categoria, usa "other"
        if not categories:
            categories = ["other"]
        
        return categories
    
    def get_pairs_to_compare(
        self, 
        index: MarketIndex,
        max_per_category: int = 1000
    ) -> List[tuple]:
        """
        Gera lista de pares para comparar.
        
        Args:
            index: Índice de mercados
            max_per_category: Máximo de comparações por categoria
            
        Returns:
            Lista de tuplas (kalshi_market, poly_market)
        """
        pairs = []
        
        for category in index.get_matching_categories():
            kalshi_markets = index.kalshi_by_category[category]
            poly_markets = index.polymarket_by_category[category]
            
            # Limitar para não explodir
            count = 0
            for k in kalshi_markets:
                for p in poly_markets:
                    pairs.append((k, p, category))
                    count += 1
                    if count >= max_per_category:
                        break
                if count >= max_per_category:
                    break
        
        logger.info(f"Total de pares para comparar: {len(pairs)}")
        return pairs
