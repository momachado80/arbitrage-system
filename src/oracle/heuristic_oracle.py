"""
Heuristic Micro Oracle — Microstructure-Based Probability Estimation
======================================================================

Oracle V1: mensurável, sem LLM, baseado em microestrutura de mercado.

Inputs:
- Book imbalance: (bid_size - ask_size) / (bid_size + ask_size)
- Volume spike: ratio volume_recent / volume_baseline
- Recent price acceleration: delta midpoint / lookback
- Spread compression: spread_baseline / spread_current

Output:
    p_oracle = P_market + alpha * imbalance + beta * acceleration
                        + gamma * volume_signal + delta * spread_signal

Alpha, beta, gamma, delta são pequenos e configuráveis.

Objetivo: testar se microestrutura gera edge detectável.

Thread-safe via RLock.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.oracle.probability_oracle import OracleEstimate, OracleProvider

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_ALPHA: float = 0.05     # Peso do book imbalance
DEFAULT_BETA: float = 0.03      # Peso da aceleração de preço
DEFAULT_GAMMA: float = 0.02     # Peso do volume spike
DEFAULT_DELTA: float = 0.01     # Peso da compressão de spread
LOOKBACK_WINDOW: int = 50       # Snapshots para baseline
MIN_SAMPLES: int = 5            # Mínimo de amostras para estimativa


@dataclass
class MicroConfig:
    """Configuração dos pesos do oracle heurístico."""
    alpha: float = DEFAULT_ALPHA     # Book imbalance weight
    beta: float = DEFAULT_BETA       # Price acceleration weight
    gamma: float = DEFAULT_GAMMA     # Volume spike weight
    delta: float = DEFAULT_DELTA     # Spread compression weight
    lookback: int = LOOKBACK_WINDOW
    min_samples: int = MIN_SAMPLES
    max_adjustment: float = 0.10     # Cap máximo do ajuste total


@dataclass
class MicroSnapshot:
    """Dados de um snapshot para cálculo de features."""
    timestamp: float
    mid: float
    spread: float
    bid_size: float
    ask_size: float
    volume: float = 0.0


class HeuristicMicroOracle(OracleProvider):
    """
    Oracle baseado em microestrutura de mercado.

    Mantém histórico de snapshots por token e calcula features
    de microestrutura para ajustar P_market marginalmente.

    Não tenta prever o outcome — apenas detecta pressão direcional
    na microestrutura do book.

    Thread-safe.
    """

    def __init__(self, config: Optional[MicroConfig] = None) -> None:
        self._config = config or MicroConfig()
        self._lock = threading.RLock()
        self._history: Dict[str, Deque[MicroSnapshot]] = {}
        self._supported_tokens: Optional[set] = None  # None = all supported

    @property
    def name(self) -> str:
        return "heuristic_micro"

    def supports(self, token_id: str) -> bool:
        if self._supported_tokens is None:
            return True
        return token_id in self._supported_tokens

    def set_supported_tokens(self, tokens: List[str]) -> None:
        """Restringe a tokens específicos."""
        with self._lock:
            self._supported_tokens = set(tokens)

    def ingest_snapshot(
        self,
        token_id: str,
        mid: float,
        spread: float,
        bid_size: float,
        ask_size: float,
        volume: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Ingere um snapshot para o histórico.

        Chamado pelo orchestrator a cada MarketSnapshot.
        """
        with self._lock:
            if token_id not in self._history:
                self._history[token_id] = deque(maxlen=self._config.lookback * 2)
            self._history[token_id].append(MicroSnapshot(
                timestamp=timestamp or time.time(),
                mid=mid,
                spread=spread,
                bid_size=bid_size,
                ask_size=ask_size,
                volume=volume,
            ))

    def estimate(
        self,
        token_id: str,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[OracleEstimate]:
        """
        Estima probabilidade baseada em microestrutura.

        p_oracle = P_market + alpha * imbalance + beta * acceleration
                            + gamma * volume_signal + delta * spread_signal
        """
        if market_data is None:
            return None

        mid = market_data.get("mid", 0.0)
        spread = market_data.get("spread", 0.0)
        bid_size = market_data.get("bid_size", 0.0)
        ask_size = market_data.get("ask_size", 0.0)
        volume = market_data.get("volume", 0.0)

        if mid <= 0:
            return None

        # Ingest current snapshot
        self.ingest_snapshot(
            token_id, mid, spread, bid_size, ask_size, volume,
        )

        with self._lock:
            history = self._history.get(token_id)
            if history is None or len(history) < self._config.min_samples:
                # Not enough data — return market price with low confidence
                return OracleEstimate(
                    probability=mid,
                    confidence=0.1,
                    source=self.name,
                    metadata={"reason": "insufficient_data", "samples": len(history) if history else 0},
                    timestamp=time.time(),
                )

            # Compute features
            imbalance = self._compute_imbalance(bid_size, ask_size)
            acceleration = self._compute_acceleration(history)
            volume_signal = self._compute_volume_signal(history, volume)
            spread_signal = self._compute_spread_signal(history, spread)

            # Adjustment = weighted sum of features
            adjustment = (
                self._config.alpha * imbalance
                + self._config.beta * acceleration
                + self._config.gamma * volume_signal
                + self._config.delta * spread_signal
            )

            # Cap adjustment
            adjustment = max(-self._config.max_adjustment,
                             min(self._config.max_adjustment, adjustment))

            p_oracle = mid + adjustment
            # Clamp to valid probability range
            p_oracle = max(0.001, min(0.999, p_oracle))

            # Confidence based on signal strength and data quality
            signal_strength = abs(adjustment) / self._config.max_adjustment
            data_quality = min(1.0, len(history) / self._config.lookback)
            confidence = 0.3 + 0.5 * signal_strength * data_quality

            return OracleEstimate(
                probability=p_oracle,
                confidence=round(confidence, 3),
                source=self.name,
                metadata={
                    "imbalance": round(imbalance, 4),
                    "acceleration": round(acceleration, 6),
                    "volume_signal": round(volume_signal, 4),
                    "spread_signal": round(spread_signal, 4),
                    "adjustment": round(adjustment, 6),
                    "samples": len(history),
                },
                timestamp=time.time(),
            )

    @staticmethod
    def _compute_imbalance(bid_size: float, ask_size: float) -> float:
        """
        Book imbalance: (bid_size - ask_size) / (bid_size + ask_size)

        Range: [-1, 1]
        Positive = more bids (buying pressure)
        Negative = more asks (selling pressure)
        """
        total = bid_size + ask_size
        if total <= 0:
            return 0.0
        return (bid_size - ask_size) / total

    @staticmethod
    def _compute_acceleration(history: Deque[MicroSnapshot]) -> float:
        """
        Price acceleration: (mid_now - mid_lookback) / lookback_count

        Positive = price trending up
        Negative = price trending down
        Normalized by count to keep small.
        """
        if len(history) < 2:
            return 0.0
        current = history[-1].mid
        oldest = history[0].mid
        n = len(history)
        return (current - oldest) / n

    @staticmethod
    def _compute_volume_signal(
        history: Deque[MicroSnapshot],
        current_volume: float,
    ) -> float:
        """
        Volume spike: current / baseline_average - 1.

        Range: [-1, inf)
        > 0 means higher than average (unusual activity)
        """
        if len(history) < 2:
            return 0.0
        volumes = [s.volume for s in history if s.volume > 0]
        if not volumes:
            return 0.0
        avg_vol = sum(volumes) / len(volumes)
        if avg_vol <= 0:
            return 0.0
        return (current_volume / avg_vol) - 1.0

    @staticmethod
    def _compute_spread_signal(
        history: Deque[MicroSnapshot],
        current_spread: float,
    ) -> float:
        """
        Spread compression: baseline_avg_spread / current_spread - 1.

        > 0 means spread is tighter than usual (more liquidity)
        < 0 means spread is wider than usual (less liquidity)
        """
        if len(history) < 2 or current_spread <= 0:
            return 0.0
        spreads = [s.spread for s in history if s.spread > 0]
        if not spreads:
            return 0.0
        avg_spread = sum(spreads) / len(spreads)
        if avg_spread <= 0:
            return 0.0
        return (avg_spread / current_spread) - 1.0

    def get_diagnostics(self, token_id: str) -> Dict[str, Any]:
        """Retorna diagnósticos para um token."""
        with self._lock:
            history = self._history.get(token_id)
            if not history:
                return {"token_id": token_id, "samples": 0}
            return {
                "token_id": token_id,
                "samples": len(history),
                "latest_mid": history[-1].mid,
                "latest_spread": history[-1].spread,
                "config": {
                    "alpha": self._config.alpha,
                    "beta": self._config.beta,
                    "gamma": self._config.gamma,
                    "delta": self._config.delta,
                    "lookback": self._config.lookback,
                },
            }
