"""
Probability Oracle — External Probability Estimation
========================================================

ABC com providers plugáveis para estimar P_oracle de fontes externas.

COMPONENTES:
1. OracleProvider (ABC)         — interface para cada fonte
2. MockOracleProvider           — determinístico para testes
3. CompositeOracleProvider      — combina múltiplos providers
4. OraclePerformanceTracker     — Brier score rolling, calibração por bucket
5. EdgeDecayTracker             — mede half-life do edge
6. ProbabilityOracle            — orquestrador principal

MÉTRICAS CONTÍNUAS:
- Brier Score rolling por token
- Calibration error por bucket de confiança
- Log estruturado de P_oracle, P_market, edge, outcome
- Edge capturado vs edge previsto
- Edge half-life

Thread-safe via RLock em todos os componentes.
"""

import abc
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Defaults
BRIER_WINDOW: int = 200
EDGE_DECAY_WINDOW: int = 100
CALIBRATION_BUCKETS: int = 10
PERFORMANCE_HISTORY_MAX: int = 1000
ORACLE_CACHE_TTL: float = 30.0  # seconds


# =========================================================================
# Data structures
# =========================================================================

@dataclass(frozen=True)
class OracleEstimate:
    """Resultado de uma estimativa do oracle."""
    probability: float
    confidence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class OracleOutcome:
    """Registro de outcome para tracking de performance."""
    token_id: str
    timestamp: float
    p_oracle: float
    p_market: float
    edge: float
    confidence: float
    source: str
    outcome: Optional[float] = None  # 1.0 = YES resolved, 0.0 = NO
    resolved_at: Optional[float] = None


@dataclass
class EdgeDecayPoint:
    """Ponto de tracking de decay do edge."""
    signal_ts: float
    token_id: str
    initial_edge: float
    initial_p_market: float
    p_oracle: float
    first_adjustment_ts: Optional[float] = None
    edge_at_adjustment: Optional[float] = None
    half_life_seconds: Optional[float] = None


# =========================================================================
# ABC — OracleProvider
# =========================================================================

class OracleProvider(abc.ABC):
    """
    Interface para providers de probabilidade externa.

    Cada provider implementa estimate() que retorna OracleEstimate.
    Providers são plugáveis e testáveis independentemente.
    """

    @abc.abstractmethod
    def estimate(
        self,
        token_id: str,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[OracleEstimate]:
        """
        Estima probabilidade para um token.

        Args:
            token_id: identificador do token/mercado
            market_data: dados opcionais do mercado (bid, ask, volume, etc.)

        Returns:
            OracleEstimate ou None se não disponível.
        """
        ...

    @abc.abstractmethod
    def supports(self, token_id: str) -> bool:
        """Retorna True se este provider suporta o token."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Nome do provider para logging/metrics."""
        ...


# =========================================================================
# MockOracleProvider — determinístico para testes
# =========================================================================

class MockOracleProvider(OracleProvider):
    """
    Provider mock determinístico.

    Aceita um mapeamento token_id → (probability, confidence).
    Perfeito para testes e backtesting.
    """

    def __init__(
        self,
        estimates: Optional[Dict[str, Tuple[float, float]]] = None,
        default_prob: float = 0.5,
        default_conf: float = 0.5,
    ) -> None:
        self._estimates = estimates or {}
        self._default_prob = default_prob
        self._default_conf = default_conf

    def estimate(
        self,
        token_id: str,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[OracleEstimate]:
        if token_id in self._estimates:
            p, c = self._estimates[token_id]
        else:
            p, c = self._default_prob, self._default_conf

        return OracleEstimate(
            probability=p,
            confidence=c,
            source=self.name,
            metadata={"mock": True, "token_id": token_id},
            timestamp=time.time(),
        )

    def supports(self, token_id: str) -> bool:
        return True

    @property
    def name(self) -> str:
        return "mock"

    def set_estimate(self, token_id: str, prob: float, conf: float) -> None:
        """Atualiza estimativa para um token (útil em testes)."""
        self._estimates[token_id] = (prob, conf)

    def remove_estimate(self, token_id: str) -> None:
        """Remove estimativa específica."""
        self._estimates.pop(token_id, None)


# =========================================================================
# CompositeOracleProvider — combina múltiplos providers
# =========================================================================

class CompositeOracleProvider(OracleProvider):
    """
    Combina múltiplos providers via média ponderada por confiança.

    Se nenhum provider suporta o token, retorna None.
    """

    def __init__(self, providers: Optional[List[OracleProvider]] = None) -> None:
        self._providers: List[OracleProvider] = providers or []

    def add_provider(self, provider: OracleProvider) -> None:
        """Adiciona provider ao composite."""
        self._providers.append(provider)

    def estimate(
        self,
        token_id: str,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[OracleEstimate]:
        estimates: List[OracleEstimate] = []
        for p in self._providers:
            if not p.supports(token_id):
                continue
            try:
                est = p.estimate(token_id, market_data)
                if est is not None and 0.0 <= est.probability <= 1.0:
                    estimates.append(est)
            except Exception as e:
                logger.warning(
                    "[ORACLE] Provider %s failed for %s: %s",
                    p.name, token_id[:12], e,
                )

        if not estimates:
            return None

        if len(estimates) == 1:
            return estimates[0]

        # Média ponderada por confiança
        total_w = sum(e.confidence for e in estimates)
        if total_w <= 0:
            return estimates[0]

        weighted_prob = sum(e.probability * e.confidence for e in estimates) / total_w
        # Confiança do composite = média das confiancas * fator de concordância
        avg_conf = total_w / len(estimates)
        # Concordância: 1.0 se todos iguais, menor se divergentes
        variance = sum(
            (e.probability - weighted_prob) ** 2 * e.confidence
            for e in estimates
        ) / total_w
        agreement = max(0.0, 1.0 - 4.0 * variance)  # 4x variance penalty
        final_conf = avg_conf * agreement

        sources = [e.source for e in estimates]

        return OracleEstimate(
            probability=max(0.001, min(0.999, weighted_prob)),
            confidence=max(0.01, min(1.0, final_conf)),
            source=f"composite({','.join(sources)})",
            metadata={
                "providers_used": len(estimates),
                "sources": sources,
                "agreement": round(agreement, 3),
                "variance": round(variance, 6),
            },
            timestamp=time.time(),
        )

    def supports(self, token_id: str) -> bool:
        return any(p.supports(token_id) for p in self._providers)

    @property
    def name(self) -> str:
        return "composite"


# =========================================================================
# OraclePerformanceTracker — Brier score + calibração
# =========================================================================

class OraclePerformanceTracker:
    """
    Mede performance do Oracle continuamente.

    Métricas:
    - Brier Score rolling (últimos N outcomes)
    - Calibration error por bucket de confiança
    - Edges capturados vs previstos
    - Performance por token

    Thread-safe.
    """

    def __init__(
        self,
        brier_window: int = BRIER_WINDOW,
        calibration_buckets: int = CALIBRATION_BUCKETS,
    ) -> None:
        self._lock = threading.RLock()
        self._brier_window = brier_window
        self._calibration_buckets = calibration_buckets

        # Outcomes com resultado conhecido
        self._outcomes: Deque[OracleOutcome] = deque(maxlen=PERFORMANCE_HISTORY_MAX)
        # Outcomes pendentes (sem resolução)
        self._pending: Dict[str, OracleOutcome] = {}

        # Brier rolling
        self._brier_scores: Deque[float] = deque(maxlen=brier_window)

        # Edge tracking
        self._predicted_edges: Deque[float] = deque(maxlen=brier_window)
        self._captured_edges: Deque[float] = deque(maxlen=brier_window)

        # Performance por token
        self._token_brier: Dict[str, Deque[float]] = {}

    def record_prediction(
        self,
        token_id: str,
        p_oracle: float,
        p_market: float,
        edge: float,
        confidence: float,
        source: str,
    ) -> None:
        """Registra predição do oracle (outcome ainda desconhecido)."""
        with self._lock:
            outcome = OracleOutcome(
                token_id=token_id,
                timestamp=time.time(),
                p_oracle=p_oracle,
                p_market=p_market,
                edge=edge,
                confidence=confidence,
                source=source,
            )
            # Usar token_id como chave (sobrescreve anterior se não resolvido)
            self._pending[token_id] = outcome

    def record_outcome(self, token_id: str, outcome: float) -> None:
        """
        Registra outcome real (0.0 ou 1.0) para atualizar Brier score.

        Args:
            token_id: token que resolveu
            outcome: 1.0 para YES, 0.0 para NO
        """
        with self._lock:
            pending = self._pending.pop(token_id, None)
            if pending is None:
                return

            pending.outcome = outcome
            pending.resolved_at = time.time()
            self._outcomes.append(pending)

            # Brier = (p_oracle - outcome)^2
            brier = (pending.p_oracle - outcome) ** 2
            self._brier_scores.append(brier)

            # Token-specific Brier
            if token_id not in self._token_brier:
                self._token_brier[token_id] = deque(maxlen=self._brier_window)
            self._token_brier[token_id].append(brier)

            # Edge capturado
            self._predicted_edges.append(abs(pending.edge))
            # Edge capturado = lucro real baseado no outcome
            if pending.edge > 0:
                captured = outcome - pending.p_market
            else:
                captured = pending.p_market - outcome
            self._captured_edges.append(captured)

            logger.info(
                "[ORACLE_PERF] Outcome %s: p_oracle=%.3f p_market=%.3f "
                "outcome=%.1f brier=%.4f",
                token_id[:12], pending.p_oracle, pending.p_market,
                outcome, brier,
            )

    def get_brier_score(self) -> Optional[float]:
        """Brier score rolling (menor = melhor, 0 = perfeito)."""
        with self._lock:
            if not self._brier_scores:
                return None
            return sum(self._brier_scores) / len(self._brier_scores)

    def get_brier_score_for_token(self, token_id: str) -> Optional[float]:
        """Brier score rolling para um token específico."""
        with self._lock:
            scores = self._token_brier.get(token_id)
            if not scores:
                return None
            return sum(scores) / len(scores)

    def get_calibration_error(self) -> Dict[str, Any]:
        """
        Calibration error por bucket de confiança.

        Retorna dict com:
        - buckets: lista de {range, predicted_avg, outcome_avg, count, error}
        - mean_absolute_calibration_error: float
        """
        with self._lock:
            if not self._outcomes:
                return {"buckets": [], "mean_absolute_calibration_error": None}

            bucket_size = 1.0 / self._calibration_buckets
            buckets: List[Dict[str, Any]] = []

            for i in range(self._calibration_buckets):
                lo = i * bucket_size
                hi = (i + 1) * bucket_size
                items = [
                    o for o in self._outcomes
                    if o.outcome is not None and lo <= o.p_oracle < hi
                ]
                if not items:
                    continue

                pred_avg = sum(o.p_oracle for o in items) / len(items)
                outcome_avg = sum(o.outcome for o in items) / len(items)
                error = abs(pred_avg - outcome_avg)

                buckets.append({
                    "range": f"{lo:.1f}-{hi:.1f}",
                    "predicted_avg": round(pred_avg, 3),
                    "outcome_avg": round(outcome_avg, 3),
                    "count": len(items),
                    "error": round(error, 4),
                })

            mace = (
                sum(b["error"] * b["count"] for b in buckets)
                / sum(b["count"] for b in buckets)
                if buckets else None
            )

            return {
                "buckets": buckets,
                "mean_absolute_calibration_error": (
                    round(mace, 4) if mace is not None else None
                ),
            }

    def get_edge_efficiency(self) -> Dict[str, Any]:
        """Edge previsto vs capturado."""
        with self._lock:
            if not self._predicted_edges:
                return {
                    "avg_predicted_edge": None,
                    "avg_captured_edge": None,
                    "capture_ratio": None,
                    "sample_count": 0,
                }

            avg_pred = sum(self._predicted_edges) / len(self._predicted_edges)
            avg_cap = sum(self._captured_edges) / len(self._captured_edges)
            ratio = avg_cap / avg_pred if avg_pred > 0 else None

            return {
                "avg_predicted_edge": round(avg_pred, 4),
                "avg_captured_edge": round(avg_cap, 4),
                "capture_ratio": round(ratio, 3) if ratio is not None else None,
                "sample_count": len(self._predicted_edges),
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Resumo completo de performance do Oracle."""
        brier = self.get_brier_score()
        cal = self.get_calibration_error()
        edge_eff = self.get_edge_efficiency()

        with self._lock:
            total_predictions = len(self._outcomes) + len(self._pending)
            resolved = len(self._outcomes)
            pending = len(self._pending)

        return {
            "brier_score": round(brier, 4) if brier is not None else None,
            "calibration": cal,
            "edge_efficiency": edge_eff,
            "total_predictions": total_predictions,
            "resolved": resolved,
            "pending": pending,
            "brier_interpretation": (
                "excellent" if brier is not None and brier < 0.1
                else "good" if brier is not None and brier < 0.2
                else "moderate" if brier is not None and brier < 0.3
                else "poor" if brier is not None
                else "insufficient_data"
            ),
        }


# =========================================================================
# EdgeDecayTracker — mede half-life do edge
# =========================================================================

class EdgeDecayTracker:
    """
    Mede quanto tempo o mercado leva para absorver o edge detectado.

    Para cada sinal:
    1. Registra timestamp + edge inicial
    2. Monitora preço subsequente
    3. Quando edge cai para metade, calcula half-life
    4. Agrega half-life médio

    Thread-safe.
    """

    def __init__(self, window: int = EDGE_DECAY_WINDOW) -> None:
        self._lock = threading.RLock()
        self._active: Dict[str, EdgeDecayPoint] = {}
        self._completed: Deque[EdgeDecayPoint] = deque(maxlen=window)
        self._half_lives: Deque[float] = deque(maxlen=window)

    def register_signal(
        self,
        token_id: str,
        initial_edge: float,
        p_market: float,
        p_oracle: float,
    ) -> None:
        """Registra novo sinal para tracking de decay."""
        with self._lock:
            self._active[token_id] = EdgeDecayPoint(
                signal_ts=time.time(),
                token_id=token_id,
                initial_edge=initial_edge,
                initial_p_market=p_market,
                p_oracle=p_oracle,
            )

    def update_market_price(self, token_id: str, current_p_market: float) -> None:
        """
        Atualiza preço de mercado e verifica se edge decaiu.

        Chamado a cada snapshot para tokens com sinais ativos.
        """
        with self._lock:
            point = self._active.get(token_id)
            if point is None:
                return

            current_edge = point.p_oracle - current_p_market
            initial_edge = point.initial_edge

            # Detectar primeiro ajuste significativo (>10% do edge)
            if (
                point.first_adjustment_ts is None
                and abs(initial_edge) > 0
                and abs(current_edge) < abs(initial_edge) * 0.9
            ):
                point.first_adjustment_ts = time.time()
                point.edge_at_adjustment = current_edge

            # Detectar half-life (edge caiu para metade)
            if (
                abs(initial_edge) > 0
                and abs(current_edge) <= abs(initial_edge) * 0.5
            ):
                now = time.time()
                half_life = now - point.signal_ts
                point.half_life_seconds = half_life
                self._half_lives.append(half_life)
                self._completed.append(point)
                del self._active[token_id]

                logger.info(
                    "[EDGE_DECAY] %s half-life=%.1fs initial_edge=%.4f",
                    token_id[:12], half_life, initial_edge,
                )

            # Timeout: se edge não decaiu em 5 minutos, finalizar
            elif time.time() - point.signal_ts > 300:
                self._completed.append(point)
                del self._active[token_id]

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de decay do edge."""
        with self._lock:
            if not self._half_lives:
                return {
                    "avg_half_life_seconds": None,
                    "median_half_life_seconds": None,
                    "min_half_life_seconds": None,
                    "max_half_life_seconds": None,
                    "sample_count": 0,
                    "active_tracking": len(self._active),
                    "capturable": None,
                }

            sorted_hl = sorted(self._half_lives)
            n = len(sorted_hl)
            median = sorted_hl[n // 2]
            avg = sum(sorted_hl) / n

            return {
                "avg_half_life_seconds": round(avg, 1),
                "median_half_life_seconds": round(median, 1),
                "min_half_life_seconds": round(sorted_hl[0], 1),
                "max_half_life_seconds": round(sorted_hl[-1], 1),
                "sample_count": n,
                "active_tracking": len(self._active),
                # Capturável = half-life > 5s (tempo mínimo de execução)
                "capturable": avg > 5.0 if n > 0 else None,
            }


# =========================================================================
# ProbabilityOracle — orquestrador principal
# =========================================================================

class ProbabilityOracle:
    """
    Orquestrador principal do Oracle.

    Combina provider, performance tracking e edge decay tracking.
    Cache de estimativas para evitar chamadas redundantes.

    Thread-safe.
    """

    def __init__(
        self,
        provider: Optional[OracleProvider] = None,
        performance_tracker: Optional[OraclePerformanceTracker] = None,
        edge_decay_tracker: Optional[EdgeDecayTracker] = None,
        cache_ttl: float = ORACLE_CACHE_TTL,
    ) -> None:
        self._provider = provider or MockOracleProvider()
        self._performance = performance_tracker or OraclePerformanceTracker()
        self._edge_decay = edge_decay_tracker or EdgeDecayTracker()
        self._cache_ttl = cache_ttl
        self._lock = threading.RLock()

        # Cache: token_id → (estimate, timestamp)
        self._cache: Dict[str, Tuple[OracleEstimate, float]] = {}

        # Contadores
        self._total_queries: int = 0
        self._cache_hits: int = 0

    def get_estimate(
        self,
        token_id: str,
        market_data: Optional[Dict[str, Any]] = None,
        p_market: Optional[float] = None,
    ) -> Optional[OracleEstimate]:
        """
        Obtém estimativa de probabilidade para um token.

        Usa cache se disponível e válido.
        Registra no performance tracker se p_market fornecido.
        """
        with self._lock:
            self._total_queries += 1

            # Cache check
            cached = self._cache.get(token_id)
            if cached is not None:
                est, cached_ts = cached
                if time.time() - cached_ts < self._cache_ttl:
                    self._cache_hits += 1
                    return est

        # Provider call (fora do lock)
        if not self._provider.supports(token_id):
            return None

        try:
            estimate = self._provider.estimate(token_id, market_data)
        except Exception as e:
            logger.warning(
                "[ORACLE] Provider error for %s: %s", token_id[:12], e,
            )
            return None

        if estimate is None:
            return None

        with self._lock:
            # Cache update
            self._cache[token_id] = (estimate, time.time())

            # Performance tracking
            if p_market is not None:
                edge = estimate.probability - p_market
                self._performance.record_prediction(
                    token_id=token_id,
                    p_oracle=estimate.probability,
                    p_market=p_market,
                    edge=edge,
                    confidence=estimate.confidence,
                    source=estimate.source,
                )

        return estimate

    def record_outcome(self, token_id: str, outcome: float) -> None:
        """Registra outcome para Brier score tracking."""
        self._performance.record_outcome(token_id, outcome)

    def register_edge_signal(
        self,
        token_id: str,
        edge: float,
        p_market: float,
        p_oracle: float,
    ) -> None:
        """Registra sinal para edge decay tracking."""
        self._edge_decay.register_signal(token_id, edge, p_market, p_oracle)

    def update_decay_tracking(self, token_id: str, current_p_market: float) -> None:
        """Atualiza tracking de decay do edge com preço atual."""
        self._edge_decay.update_market_price(token_id, current_p_market)

    def get_brier_score(self) -> Optional[float]:
        """Brier score rolling atual."""
        return self._performance.get_brier_score()

    def get_adaptive_sizing_multiplier(self) -> float:
        """
        Multiplier adaptativo baseado no Brier score.

        Brier < 0.15 → 1.0 (full size)
        Brier 0.15-0.25 → 0.7
        Brier 0.25-0.35 → 0.4
        Brier > 0.35 → 0.1 (minimal)
        Sem dados → 0.3 (conservador)
        """
        brier = self._performance.get_brier_score()
        if brier is None:
            return 0.3  # Conservador até ter dados

        if brier < 0.15:
            return 1.0
        elif brier < 0.25:
            return 0.7
        elif brier < 0.35:
            return 0.4
        else:
            return 0.1

    def should_block_oracle(self) -> bool:
        """
        Retorna True se o Oracle deve ser bloqueado.

        Bloqueia se Brier > 0.4 com sample > 20 (pior que aleatório).
        """
        with self._lock:
            brier = self._performance.get_brier_score()
            resolved = self._performance.get_performance_summary()["resolved"]
            if brier is not None and resolved >= 20 and brier > 0.4:
                return True
            return False

    def get_full_metrics(self) -> Dict[str, Any]:
        """Retorna métricas completas do Oracle."""
        with self._lock:
            return {
                "total_queries": self._total_queries,
                "cache_hits": self._cache_hits,
                "cache_hit_rate": (
                    round(self._cache_hits / self._total_queries, 3)
                    if self._total_queries > 0 else 0.0
                ),
                "performance": self._performance.get_performance_summary(),
                "edge_decay": self._edge_decay.get_metrics(),
                "adaptive_sizing_multiplier": self.get_adaptive_sizing_multiplier(),
                "oracle_blocked": self.should_block_oracle(),
                "provider": self._provider.name,
            }

    @property
    def performance_tracker(self) -> OraclePerformanceTracker:
        """Acesso ao tracker de performance."""
        return self._performance

    @property
    def edge_decay_tracker(self) -> EdgeDecayTracker:
        """Acesso ao tracker de edge decay."""
        return self._edge_decay
