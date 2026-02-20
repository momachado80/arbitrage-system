"""
Universe Builder — Market Selection and Filtering
=====================================================

Seleciona mercados automaticamente com base em critérios quantitativos.

FILTROS:
1. Liquidez mínima (bid_size + ask_size > threshold)
2. Spread máximo (ask - bid < threshold)
3. Volume mínimo (volume_24h > threshold)
4. Prazo mínimo até resolução (days_to_close > threshold)
5. Disponibilidade de fonte oracle compatível

INTEGRAÇÃO:
- Respeita POLYMARKET_TOKENS como override absoluto
- Roda periodicamente via update()
- Atualiza lista ativa de token_ids
- Thread-safe

Thread-safe via RLock.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_MIN_LIQUIDITY: float = 50.0        # $ mínimo de liquidez no book
DEFAULT_MAX_SPREAD: float = 0.10            # 10% spread máximo
DEFAULT_MIN_VOLUME_24H: float = 100.0       # $ volume mínimo 24h
DEFAULT_MIN_DAYS_TO_CLOSE: float = 1.0      # Mínimo 1 dia até resolução
DEFAULT_UPDATE_INTERVAL: float = 300.0      # 5 minutos entre updates
UNIVERSE_MAX_MARKETS: int = 50              # Máximo de mercados monitorados


@dataclass
class MarketInfo:
    """Informação de um mercado para filtragem."""
    token_id: str
    question: str = ""
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    volume_24h: float = 0.0
    days_to_close: float = 0.0
    oracle_supported: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UniverseFilters:
    """Configuração dos filtros do universe."""
    min_liquidity: float = DEFAULT_MIN_LIQUIDITY
    max_spread: float = DEFAULT_MAX_SPREAD
    min_volume_24h: float = DEFAULT_MIN_VOLUME_24H
    min_days_to_close: float = DEFAULT_MIN_DAYS_TO_CLOSE
    max_markets: int = UNIVERSE_MAX_MARKETS


class UniverseBuilder:
    """
    Seleciona e mantém lista de mercados ativos.

    POLYMARKET_TOKENS (override manual) sempre tem prioridade.
    Quando vazio, usa filtragem automática.
    """

    def __init__(
        self,
        filters: Optional[UniverseFilters] = None,
        manual_tokens: Optional[List[str]] = None,
        oracle_provider: Optional[Any] = None,
        update_interval: float = DEFAULT_UPDATE_INTERVAL,
    ) -> None:
        self._filters = filters or UniverseFilters()
        self._manual_tokens = manual_tokens or []
        self._oracle_provider = oracle_provider
        self._update_interval = update_interval
        self._lock = threading.RLock()

        # Estado
        self._active_tokens: List[str] = list(self._manual_tokens)
        self._all_candidates: List[MarketInfo] = []
        self._last_update_ts: float = 0.0
        self._update_count: int = 0
        self._filter_stats: Dict[str, int] = {}

    def set_candidates(self, markets: List[MarketInfo]) -> None:
        """
        Define candidatos disponíveis (de API ou mock).

        Chamado pelo orchestrator quando novos dados de mercado
        estão disponíveis.
        """
        with self._lock:
            self._all_candidates = markets

    def update(self, force: bool = False) -> List[str]:
        """
        Atualiza lista de tokens ativos.

        Se manual_tokens definidos, retorna override.
        Caso contrário, aplica filtros nos candidatos.

        Args:
            force: ignorar intervalo de update

        Returns:
            Lista de token_ids ativos.
        """
        with self._lock:
            # Override absoluto: POLYMARKET_TOKENS manual
            if self._manual_tokens:
                self._active_tokens = list(self._manual_tokens)
                return self._active_tokens

            # Throttle updates
            now = time.time()
            if not force and (now - self._last_update_ts) < self._update_interval:
                return self._active_tokens

            self._last_update_ts = now
            self._update_count += 1

            # Aplicar filtros
            passed = self._apply_filters(self._all_candidates)

            # Limitar ao máximo
            if len(passed) > self._filters.max_markets:
                # Ordenar por liquidez (melhor primeiro)
                passed.sort(
                    key=lambda m: (m.bid_size + m.ask_size) * (m.best_bid + m.best_ask) / 2,
                    reverse=True,
                )
                passed = passed[:self._filters.max_markets]

            self._active_tokens = [m.token_id for m in passed]

            logger.info(
                "[UNIVERSE] Update #%d: candidates=%d passed=%d active=%d",
                self._update_count, len(self._all_candidates),
                len(passed), len(self._active_tokens),
            )

            return self._active_tokens

    def _apply_filters(self, markets: List[MarketInfo]) -> List[MarketInfo]:
        """Aplica todos os filtros sequencialmente."""
        self._filter_stats = {
            "total_candidates": len(markets),
            "passed_liquidity": 0,
            "passed_spread": 0,
            "passed_volume": 0,
            "passed_days": 0,
            "passed_oracle": 0,
            "final": 0,
        }

        result = []
        for m in markets:
            # Filter 1: Liquidez
            liquidity = (m.bid_size + m.ask_size) * (
                (m.best_bid + m.best_ask) / 2 if m.best_bid > 0 else 0
            )
            if liquidity < self._filters.min_liquidity:
                continue
            self._filter_stats["passed_liquidity"] += 1

            # Filter 2: Spread
            if m.best_ask > 0 and m.best_bid > 0:
                spread = m.best_ask - m.best_bid
                if spread > self._filters.max_spread:
                    continue
            self._filter_stats["passed_spread"] += 1

            # Filter 3: Volume 24h
            if m.volume_24h < self._filters.min_volume_24h:
                continue
            self._filter_stats["passed_volume"] += 1

            # Filter 4: Dias até resolução
            if m.days_to_close < self._filters.min_days_to_close:
                continue
            self._filter_stats["passed_days"] += 1

            # Filter 5: Oracle suportado
            if self._oracle_provider is not None:
                if not self._oracle_provider.supports(m.token_id):
                    continue
            self._filter_stats["passed_oracle"] += 1

            result.append(m)

        self._filter_stats["final"] = len(result)
        return result

    def get_active_tokens(self) -> List[str]:
        """Retorna tokens ativos atuais."""
        with self._lock:
            return list(self._active_tokens)

    def add_manual_token(self, token_id: str) -> None:
        """Adiciona token ao override manual."""
        with self._lock:
            if token_id not in self._manual_tokens:
                self._manual_tokens.append(token_id)
                self._active_tokens = list(self._manual_tokens)

    def remove_manual_token(self, token_id: str) -> None:
        """Remove token do override manual."""
        with self._lock:
            self._manual_tokens = [t for t in self._manual_tokens if t != token_id]
            if self._manual_tokens:
                self._active_tokens = list(self._manual_tokens)

    def is_manual_override(self) -> bool:
        """True se usando tokens manuais."""
        with self._lock:
            return len(self._manual_tokens) > 0

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo do universe."""
        with self._lock:
            return {
                "active_tokens": len(self._active_tokens),
                "manual_override": self.is_manual_override(),
                "total_candidates": len(self._all_candidates),
                "update_count": self._update_count,
                "last_update_ts": self._last_update_ts,
                "filter_stats": dict(self._filter_stats),
                "filters": {
                    "min_liquidity": self._filters.min_liquidity,
                    "max_spread": self._filters.max_spread,
                    "min_volume_24h": self._filters.min_volume_24h,
                    "min_days_to_close": self._filters.min_days_to_close,
                    "max_markets": self._filters.max_markets,
                },
            }
