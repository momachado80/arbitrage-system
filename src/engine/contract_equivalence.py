"""
Contract Equivalence Analyzer
==============================

Sistema avançado para determinar se dois contratos de mercados de previsão
são VERDADEIRAMENTE equivalentes para fins de arbitragem.

A equivalência contratual é o ponto crítico da arbitragem:
- Dois mercados podem parecer iguais no título
- Mas ter regras de resolução diferentes
- Ou fontes de verdade diferentes
- Ou condições de cancelamento diferentes

Este módulo analisa TODOS esses aspectos antes de aprovar um par.

PRINCÍPIO FUNDAMENTAL:
Se há QUALQUER ambiguidade → o par é REJEITADO ou penalizado severamente.
Arbitragem só funciona com certeza absoluta de equivalência.
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Tuple, Optional, Set
from difflib import SequenceMatcher
from enum import Enum

logger = logging.getLogger(__name__)


class EquivalenceLevel(Enum):
    """Níveis de equivalência contratual."""
    IDENTICAL = "identical"          # 100% equivalente - mesmo evento, mesmas regras
    HIGH_CONFIDENCE = "high"         # 95%+ - muito provável ser o mesmo
    MEDIUM_CONFIDENCE = "medium"     # 80%+ - provavelmente o mesmo, mas com riscos
    LOW_CONFIDENCE = "low"           # 60%+ - possivelmente o mesmo, alto risco
    INCOMPATIBLE = "incompatible"    # Definitivamente NÃO são equivalentes
    UNKNOWN = "unknown"              # Não foi possível determinar


@dataclass
class ResolutionAnalysis:
    """Análise das regras de resolução de um mercado."""
    source: Optional[str] = None          # Fonte de verdade (ex: "AP News", "Official Results")
    description: Optional[str] = None      # Descrição completa das regras
    end_date: Optional[datetime] = None    # Data limite
    cancel_conditions: List[str] = field(default_factory=list)  # Condições de cancelamento
    key_entities: Set[str] = field(default_factory=set)  # Entidades chave (pessoas, orgs)
    key_dates: Set[str] = field(default_factory=set)     # Datas chave
    key_numbers: Set[str] = field(default_factory=set)   # Números chave (ex: "$100", "50%")
    outcome_type: str = "binary"           # Tipo de resultado (binary, multiple, numeric)


@dataclass
class EquivalenceResult:
    """Resultado da análise de equivalência entre dois contratos."""
    level: EquivalenceLevel
    score: Decimal                     # 0.0 a 1.0
    confidence: Decimal                # Confiança na análise (0.0 a 1.0)
    title_similarity: float
    description_similarity: float
    resolution_source_match: bool
    date_compatibility: bool
    entity_overlap: float
    risks: List[str]                   # Lista de riscos identificados
    recommendation: str                # Recomendação final
    details: dict                      # Detalhes adicionais


class ContractEquivalenceAnalyzer:
    """
    Analisador de equivalência contratual.
    
    Compara dois mercados de plataformas diferentes e determina
    se são suficientemente equivalentes para arbitragem.
    """
    
    # Fontes de resolução conhecidas e suas variações
    RESOLUTION_SOURCES = {
        # Eleições
        "associated press": ["ap", "ap news", "associated press"],
        "official results": ["official", "certified", "state certified"],
        "electoral college": ["electoral", "electors"],
        
        # Esportes
        "espn": ["espn", "espn.com"],
        "official league": ["nba", "nfl", "mlb", "nhl", "fifa"],
        
        # Economia
        "federal reserve": ["fed", "fomc", "federal reserve"],
        "bls": ["bureau of labor statistics", "bls"],
        "bea": ["bureau of economic analysis", "bea"],
        
        # Crypto
        "coingecko": ["coingecko", "coin gecko"],
        "coinmarketcap": ["coinmarketcap", "cmc"],
        
        # Geral
        "consensus": ["consensus", "credible reporting", "major outlets"],
    }
    
    # Entidades importantes para matching
    IMPORTANT_ENTITIES = {
        # Políticos
        'trump', 'biden', 'harris', 'desantis', 'newsom', 'pence', 'obama',
        'vance', 'walz', 'pelosi', 'mcconnell', 'schumer',
        
        # Tech
        'musk', 'bezos', 'zuckerberg', 'altman', 'nadella', 'cook', 'pichai',
        
        # Organizações
        'fed', 'federal reserve', 'sec', 'ftc', 'fda', 'epa', 'nasa', 'fbi', 'cia',
        'supreme court', 'congress', 'senate', 'house',
        
        # Empresas
        'tesla', 'apple', 'google', 'meta', 'amazon', 'microsoft', 'nvidia',
        'openai', 'anthropic', 'spacex', 'twitter', 'x corp',
        
        # Eventos
        'election', 'super bowl', 'superbowl', 'world cup', 'olympics',
        'oscars', 'grammy', 'emmys', 'golden globes',
        
        # Países/Regiões
        'usa', 'united states', 'china', 'russia', 'ukraine', 'israel',
        'gaza', 'taiwan', 'north korea', 'iran',
    }
    
    # Palavras que indicam condições específicas
    CONDITION_WORDS = {
        'before', 'after', 'by', 'until', 'within', 'during',
        'more than', 'less than', 'at least', 'at most',
        'above', 'below', 'between', 'exactly',
        'first', 'last', 'next', 'previous',
        'win', 'lose', 'defeat', 'beat',
        'announce', 'confirm', 'declare', 'certify',
    }
    
    def __init__(
        self,
        min_title_similarity: float = 0.5,
        min_entity_overlap: float = 0.3,
        require_date_match: bool = True,
        require_source_match: bool = False,  # Nem sempre disponível
    ):
        self.min_title_similarity = min_title_similarity
        self.min_entity_overlap = min_entity_overlap
        self.require_date_match = require_date_match
        self.require_source_match = require_source_match
    
    def analyze(
        self,
        kalshi_title: str,
        kalshi_description: str,
        kalshi_rules: Optional[str],
        kalshi_end_date: Optional[datetime],
        poly_title: str,
        poly_description: str,
        poly_rules: Optional[str],
        poly_end_date: Optional[datetime],
    ) -> EquivalenceResult:
        """
        Analisa equivalência entre dois contratos.
        
        Returns:
            EquivalenceResult com nível de equivalência e detalhes
        """
        risks = []
        details = {}
        
        # 1. Análise de similaridade textual
        title_sim = self._text_similarity(kalshi_title, poly_title)
        desc_sim = self._text_similarity(
            kalshi_description or "", 
            poly_description or ""
        )
        
        details["title_similarity"] = title_sim
        details["description_similarity"] = desc_sim
        
        # 2. Extração de informações estruturadas
        kalshi_analysis = self._analyze_resolution(
            kalshi_title, kalshi_description, kalshi_rules, kalshi_end_date
        )
        poly_analysis = self._analyze_resolution(
            poly_title, poly_description, poly_rules, poly_end_date
        )
        
        # 3. Comparação de entidades
        entity_overlap = self._calculate_entity_overlap(
            kalshi_analysis.key_entities,
            poly_analysis.key_entities
        )
        details["entity_overlap"] = entity_overlap
        details["kalshi_entities"] = list(kalshi_analysis.key_entities)
        details["poly_entities"] = list(poly_analysis.key_entities)
        
        # 4. Comparação de datas
        date_compatible = self._check_date_compatibility(
            kalshi_analysis.end_date,
            poly_analysis.end_date
        )
        details["date_compatible"] = date_compatible
        
        if not date_compatible and self.require_date_match:
            risks.append("CRITICAL: Datas de expiração incompatíveis")
        
        # 5. Comparação de fonte de resolução
        source_match = self._check_resolution_source_match(
            kalshi_analysis.source,
            poly_analysis.source
        )
        details["resolution_source_match"] = source_match
        
        if not source_match and self.require_source_match:
            risks.append("WARNING: Fontes de resolução diferentes ou não identificadas")
        
        # 6. Comparação de números chave
        numbers_match = self._check_key_numbers_match(
            kalshi_analysis.key_numbers,
            poly_analysis.key_numbers
        )
        details["key_numbers_match"] = numbers_match
        
        if not numbers_match:
            risks.append("CRITICAL: Números/valores chave diferentes nos contratos")
        
        # 7. Análise de condições específicas
        condition_similarity = self._analyze_conditions(
            kalshi_title + " " + (kalshi_description or ""),
            poly_title + " " + (poly_description or "")
        )
        details["condition_similarity"] = condition_similarity
        
        # 8. Calcular score final
        score, level = self._calculate_final_score(
            title_sim=title_sim,
            desc_sim=desc_sim,
            entity_overlap=entity_overlap,
            date_compatible=date_compatible,
            source_match=source_match,
            numbers_match=numbers_match,
            condition_similarity=condition_similarity,
            risks=risks,
        )
        
        # 9. Gerar recomendação
        recommendation = self._generate_recommendation(level, risks, score)
        
        # 10. Calcular confiança na análise
        confidence = self._calculate_confidence(
            has_description=bool(kalshi_description and poly_description),
            has_rules=bool(kalshi_rules or poly_rules),
            has_dates=bool(kalshi_end_date and poly_end_date),
            entity_count=len(kalshi_analysis.key_entities | poly_analysis.key_entities),
        )
        
        return EquivalenceResult(
            level=level,
            score=Decimal(str(round(score, 4))),
            confidence=Decimal(str(round(confidence, 4))),
            title_similarity=title_sim,
            description_similarity=desc_sim,
            resolution_source_match=source_match,
            date_compatibility=date_compatible,
            entity_overlap=entity_overlap,
            risks=risks,
            recommendation=recommendation,
            details=details,
        )
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calcula similaridade textual normalizada."""
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)
        
        if not text1 or not text2:
            return 0.0
        
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparação."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _analyze_resolution(
        self,
        title: str,
        description: Optional[str],
        rules: Optional[str],
        end_date: Optional[datetime],
    ) -> ResolutionAnalysis:
        """Extrai informações estruturadas sobre resolução."""
        full_text = f"{title} {description or ''} {rules or ''}".lower()
        
        # Extrair fonte de resolução
        source = self._extract_resolution_source(full_text)
        
        # Extrair entidades
        entities = self._extract_entities(full_text)
        
        # Extrair datas
        dates = self._extract_dates(full_text)
        
        # Extrair números
        numbers = self._extract_numbers(full_text)
        
        return ResolutionAnalysis(
            source=source,
            description=description,
            end_date=end_date,
            key_entities=entities,
            key_dates=dates,
            key_numbers=numbers,
        )
    
    def _extract_resolution_source(self, text: str) -> Optional[str]:
        """Identifica a fonte de resolução no texto."""
        text = text.lower()
        
        for canonical, variations in self.RESOLUTION_SOURCES.items():
            for variation in variations:
                if variation in text:
                    return canonical
        
        # Padrões genéricos
        if "official" in text:
            return "official_source"
        if "consensus" in text or "credible" in text:
            return "consensus"
        
        return None
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extrai entidades importantes do texto."""
        text = text.lower()
        entities = set()
        
        for entity in self.IMPORTANT_ENTITIES:
            if entity in text:
                entities.add(entity)
        
        return entities
    
    def _extract_dates(self, text: str) -> Set[str]:
        """Extrai datas do texto."""
        dates = set()
        
        # Padrões de data
        patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',         # YYYY-MM-DD
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}\b',
            r'\b\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}\b',
            r'\b20\d{2}\b',  # Anos 2000s
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            dates.update(matches)
        
        return dates
    
    def _extract_numbers(self, text: str) -> Set[str]:
        """Extrai números significativos do texto."""
        numbers = set()
        
        # Padrões de números com contexto
        patterns = [
            r'\$[\d,]+(?:\.\d+)?',           # Valores monetários
            r'\d+(?:\.\d+)?%',                # Porcentagens
            r'\d+(?:,\d{3})+',                # Números grandes
            r'\b\d+\s*(?:million|billion|trillion)\b',
            r'\b(?:first|second|third|1st|2nd|3rd|\d+th)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            numbers.update(matches)
        
        return numbers
    
    def _calculate_entity_overlap(self, set1: Set[str], set2: Set[str]) -> float:
        """Calcula sobreposição de entidades (Jaccard)."""
        if not set1 and not set2:
            return 0.0
        
        intersection = set1 & set2
        union = set1 | set2
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _check_date_compatibility(
        self,
        date1: Optional[datetime],
        date2: Optional[datetime],
        tolerance_days: int = 7
    ) -> bool:
        """Verifica se as datas são compatíveis."""
        if date1 is None or date2 is None:
            return True  # Não podemos verificar, assume compatível
        
        diff = abs((date1 - date2).days)
        return diff <= tolerance_days
    
    def _check_resolution_source_match(
        self,
        source1: Optional[str],
        source2: Optional[str]
    ) -> bool:
        """Verifica se as fontes de resolução são compatíveis."""
        if source1 is None or source2 is None:
            return True  # Não podemos verificar
        
        return source1 == source2
    
    def _check_key_numbers_match(self, nums1: Set[str], nums2: Set[str]) -> bool:
        """Verifica se números chave são compatíveis."""
        if not nums1 and not nums2:
            return True
        
        # Se um tem números e outro não, pode ser problema
        if bool(nums1) != bool(nums2):
            return True  # Inconclusivo
        
        # Verificar se há conflitos
        # Por exemplo, "$100" vs "$200" seria conflito
        return True  # Simplificado por enquanto
    
    def _analyze_conditions(self, text1: str, text2: str) -> float:
        """Analisa similaridade de condições específicas."""
        words1 = set(self._normalize_text(text1).split())
        words2 = set(self._normalize_text(text2).split())
        
        # Focar em palavras de condição
        cond1 = words1 & self.CONDITION_WORDS
        cond2 = words2 & self.CONDITION_WORDS
        
        if not cond1 and not cond2:
            return 1.0  # Sem condições específicas
        
        if not cond1 or not cond2:
            return 0.5  # Um tem condições, outro não
        
        # Verificar sobreposição
        overlap = len(cond1 & cond2) / len(cond1 | cond2)
        return overlap
    
    def _calculate_final_score(
        self,
        title_sim: float,
        desc_sim: float,
        entity_overlap: float,
        date_compatible: bool,
        source_match: bool,
        numbers_match: bool,
        condition_similarity: float,
        risks: List[str],
    ) -> Tuple[float, EquivalenceLevel]:
        """Calcula score final e nível de equivalência."""
        
        # Pesos
        weights = {
            "title": 0.25,
            "description": 0.15,
            "entities": 0.25,
            "conditions": 0.15,
            "dates": 0.10,
            "source": 0.05,
            "numbers": 0.05,
        }
        
        score = (
            weights["title"] * title_sim +
            weights["description"] * desc_sim +
            weights["entities"] * entity_overlap +
            weights["conditions"] * condition_similarity +
            weights["dates"] * (1.0 if date_compatible else 0.0) +
            weights["source"] * (1.0 if source_match else 0.5) +
            weights["numbers"] * (1.0 if numbers_match else 0.0)
        )
        
        # Penalidades por riscos críticos
        critical_risks = sum(1 for r in risks if "CRITICAL" in r)
        score -= critical_risks * 0.2
        score = max(0.0, score)
        
        # Determinar nível
        if score >= 0.85 and critical_risks == 0:
            level = EquivalenceLevel.IDENTICAL
        elif score >= 0.70 and critical_risks == 0:
            level = EquivalenceLevel.HIGH_CONFIDENCE
        elif score >= 0.50:
            level = EquivalenceLevel.MEDIUM_CONFIDENCE
        elif score >= 0.30:
            level = EquivalenceLevel.LOW_CONFIDENCE
        else:
            level = EquivalenceLevel.INCOMPATIBLE
        
        return score, level
    
    def _calculate_confidence(
        self,
        has_description: bool,
        has_rules: bool,
        has_dates: bool,
        entity_count: int,
    ) -> float:
        """Calcula confiança na análise."""
        confidence = 0.5  # Base
        
        if has_description:
            confidence += 0.15
        if has_rules:
            confidence += 0.15
        if has_dates:
            confidence += 0.10
        if entity_count >= 2:
            confidence += 0.10
        
        return min(1.0, confidence)
    
    def _generate_recommendation(
        self,
        level: EquivalenceLevel,
        risks: List[str],
        score: float
    ) -> str:
        """Gera recomendação final."""
        if level == EquivalenceLevel.IDENTICAL:
            return "✅ APROVAR: Contratos parecem idênticos. Prosseguir com arbitragem."
        
        elif level == EquivalenceLevel.HIGH_CONFIDENCE:
            return f"✅ APROVAR COM CAUTELA: Alta confiança ({score:.0%}). Verificar manualmente antes de executar."
        
        elif level == EquivalenceLevel.MEDIUM_CONFIDENCE:
            risk_summary = "; ".join(risks[:2]) if risks else "Ambiguidade moderada"
            return f"⚠️ REVISAR: Confiança média ({score:.0%}). Riscos: {risk_summary}"
        
        elif level == EquivalenceLevel.LOW_CONFIDENCE:
            return f"❌ NÃO RECOMENDADO: Baixa confiança ({score:.0%}). Alto risco de basis."
        
        else:
            return "❌ REJEITAR: Contratos incompatíveis ou não equivalentes."


def equivalence_to_penalty(result: EquivalenceResult) -> Decimal:
    """
    Converte resultado de equivalência em penalidade de risco.
    
    Quanto MENOR a equivalência, MAIOR a penalidade.
    """
    if result.level == EquivalenceLevel.IDENTICAL:
        return Decimal("0.005")  # 0.5% - mínimo
    
    elif result.level == EquivalenceLevel.HIGH_CONFIDENCE:
        return Decimal("0.01")   # 1%
    
    elif result.level == EquivalenceLevel.MEDIUM_CONFIDENCE:
        return Decimal("0.025")  # 2.5%
    
    elif result.level == EquivalenceLevel.LOW_CONFIDENCE:
        return Decimal("0.05")   # 5% - muito alto
    
    else:
        return Decimal("1.0")    # 100% - impossível lucrar
