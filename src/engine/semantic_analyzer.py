"""
Semantic Market Analyzer
========================

Usa Claude para analisar semanticamente se dois mercados
são sobre o MESMO EVENTO, não apenas mencionam a mesma pessoa/empresa.

Exemplo:
- "Musk visit Mars" vs "Musk win election" → DIFERENTES (mesmo pessoa, eventos diferentes)
- "Trump wins 2024" vs "Trump president after Nov 2024" → MESMO EVENTO
- "Bitcoin 100k by Dec" vs "BTC reaches $100,000 December 2024" → MESMO EVENTO
"""

import os
import json
import logging
import asyncio
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import httpx

from src.models import Market

logger = logging.getLogger(__name__)


@dataclass
class SemanticMatch:
    """Resultado de análise semântica."""
    kalshi: Market
    polymarket: Market
    is_same_event: bool
    confidence: float  # 0-1
    reasoning: str
    event_description: str  # Descrição do evento em comum
    key_differences: List[str]  # Diferenças identificadas


class SemanticAnalyzer:
    """
    Analisa semanticamente se dois mercados são sobre o mesmo evento.
    Usa Claude API para análise de linguagem natural.
    """
    
    # Cache para evitar chamadas repetidas
    _cache: Dict[str, SemanticMatch] = {}
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Anthropic API key. Se não fornecida, usa ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY não configurada - análise semântica desabilitada")
        
        self.client = httpx.AsyncClient(timeout=30.0)
        self.api_url = "https://api.anthropic.com/v1/messages"
    
    def _cache_key(self, title1: str, title2: str) -> str:
        """Gera chave de cache para um par de títulos."""
        combined = f"{title1.lower().strip()}|||{title2.lower().strip()}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def analyze_pair(
        self,
        kalshi: Market,
        polymarket: Market,
    ) -> SemanticMatch:
        """
        Analisa se dois mercados são sobre o mesmo evento.
        
        Returns:
            SemanticMatch com resultado da análise
        """
        # Verificar cache
        cache_key = self._cache_key(kalshi.title, polymarket.title)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Se não tem API key, retorna match básico
        if not self.api_key:
            return SemanticMatch(
                kalshi=kalshi,
                polymarket=polymarket,
                is_same_event=False,
                confidence=0.0,
                reasoning="API key não configurada",
                event_description="",
                key_differences=["Análise semântica não disponível"],
            )
        
        # Chamar Claude para análise
        result = await self._call_claude(kalshi, polymarket)
        
        # Cachear resultado
        self._cache[cache_key] = result
        
        return result
    
    async def analyze_batch(
        self,
        pairs: List[Tuple[Market, Market]],
        max_concurrent: int = 5,
    ) -> List[SemanticMatch]:
        """
        Analisa múltiplos pares em paralelo.
        
        Args:
            pairs: Lista de tuplas (kalshi_market, poly_market)
            max_concurrent: Máximo de chamadas paralelas
            
        Returns:
            Lista de SemanticMatch
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_limit(pair):
            async with semaphore:
                return await self.analyze_pair(pair[0], pair[1])
        
        tasks = [analyze_with_limit(pair) for pair in pairs]
        return await asyncio.gather(*tasks)
    
    async def _call_claude(
        self,
        kalshi: Market,
        polymarket: Market,
    ) -> SemanticMatch:
        """Chama Claude API para análise semântica."""
        
        prompt = f"""Analise estes dois mercados de previsão e determine se são sobre o MESMO EVENTO específico.

MERCADO 1 (Kalshi):
Título: {kalshi.title}
Descrição: {kalshi.description or 'N/A'}

MERCADO 2 (Polymarket):
Título: {polymarket.title}
Descrição: {polymarket.description or 'N/A'}

IMPORTANTE: Dois mercados sobre a MESMA PESSOA ou EMPRESA não são necessariamente o mesmo evento!
- "Musk vai a Marte" vs "Musk ganha eleição" = EVENTOS DIFERENTES (mesma pessoa, eventos diferentes)
- "Bitcoin atinge 100k" vs "BTC chega a $100,000" = MESMO EVENTO (mesma coisa com palavras diferentes)
- "Trump ganha 2024" vs "Trump presidente após Nov 2024" = MESMO EVENTO (mesmo resultado)

Responda APENAS em JSON válido:
{{
    "is_same_event": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explicação breve",
    "event_description": "descrição do evento se for o mesmo, ou vazio",
    "key_differences": ["diferença 1", "diferença 2"] ou []
}}"""

        try:
            response = await self.client.post(
                self.api_url,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-3-haiku-20240307",  # Mais rápido e barato
                    "max_tokens": 300,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Erro Claude API: {response.status_code} - {response.text}")
                return self._fallback_match(kalshi, polymarket, f"API error: {response.status_code}")
            
            data = response.json()
            content = data["content"][0]["text"]
            
            # Parse JSON da resposta
            # Limpar possíveis marcadores de código
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            result = json.loads(content)
            
            return SemanticMatch(
                kalshi=kalshi,
                polymarket=polymarket,
                is_same_event=result.get("is_same_event", False),
                confidence=float(result.get("confidence", 0.0)),
                reasoning=result.get("reasoning", ""),
                event_description=result.get("event_description", ""),
                key_differences=result.get("key_differences", []),
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Erro parsing JSON: {e}")
            return self._fallback_match(kalshi, polymarket, f"JSON parse error: {e}")
        except Exception as e:
            logger.error(f"Erro na análise semântica: {e}")
            return self._fallback_match(kalshi, polymarket, str(e))
    
    def _fallback_match(
        self,
        kalshi: Market,
        polymarket: Market,
        error: str,
    ) -> SemanticMatch:
        """Retorna match de fallback em caso de erro."""
        return SemanticMatch(
            kalshi=kalshi,
            polymarket=polymarket,
            is_same_event=False,
            confidence=0.0,
            reasoning=f"Erro: {error}",
            event_description="",
            key_differences=["Análise não disponível"],
        )
    
    async def close(self):
        """Fecha o cliente HTTP."""
        await self.client.aclose()


class SmartMarketMatcher:
    """
    Matcher inteligente que combina:
    1. Pré-filtragem por entidades (rápido)
    2. Análise semântica com Claude (preciso)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        from src.engine.entity_matcher import EntityBasedMatcher
        
        self.entity_matcher = EntityBasedMatcher(min_shared_entities=1)
        self.semantic_analyzer = SemanticAnalyzer(api_key=api_key)
    
    async def find_equivalent_markets(
        self,
        kalshi_markets: List[Market],
        poly_markets: List[Market],
        max_semantic_checks: int = 100,
    ) -> List[SemanticMatch]:
        """
        Encontra mercados verdadeiramente equivalentes.
        
        1. Pré-filtra por entidades em comum
        2. Analisa semanticamente os top candidatos
        3. Retorna apenas matches confirmados
        
        Args:
            kalshi_markets: Mercados Kalshi
            poly_markets: Mercados Polymarket
            max_semantic_checks: Máximo de análises semânticas
            
        Returns:
            Lista de SemanticMatch confirmados
        """
        # 1. Pré-filtragem por entidades
        logger.info("Fase 1: Pré-filtragem por entidades...")
        entity_matches = self.entity_matcher.find_matches(
            kalshi_markets,
            poly_markets,
            min_confidence=0.3,
        )
        
        logger.info(f"Candidatos por entidade: {len(entity_matches)}")
        
        # 2. Limitar para análise semântica
        candidates = entity_matches[:max_semantic_checks]
        
        if not candidates:
            return []
        
        # 3. Análise semântica
        logger.info(f"Fase 2: Análise semântica de {len(candidates)} candidatos...")
        pairs = [(m.kalshi, m.polymarket) for m in candidates]
        semantic_results = await self.semantic_analyzer.analyze_batch(pairs)
        
        # 4. Filtrar apenas matches confirmados
        confirmed = [
            match for match in semantic_results
            if match.is_same_event and match.confidence >= 0.7
        ]
        
        logger.info(f"Matches semânticos confirmados: {len(confirmed)}")
        
        return confirmed
    
    async def close(self):
        """Fecha recursos."""
        await self.semantic_analyzer.close()
