"""
Reversion Engine — Statistical Overreaction Detection V1
==========================================================

Detecta overreação estatística em trajetória de probabilidade e
prevê reversão parcial da média. OBSERVE mode only.

- Determinístico, audível, sem side-effects
- Não envia ordens
- Emite forecast + trade intent signal
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.06
DEFAULT_Z_ENTRY = 2.3
DEFAULT_Z_EXIT = 1.0
DEFAULT_K_REVERT = 0.25
DEFAULT_MIN_SPREAD = 0.002
DEFAULT_MAX_SPREAD = 0.05
DEFAULT_MIN_LIQUIDITY = 100.0
DEFAULT_MAX_TTL = 3600
DEFAULT_EPSILON = 1e-9


@dataclass(frozen=True)
class ReversionConfig:
    """Configuração do engine de reversão."""

    alpha: float
    z_entry_threshold: float
    z_exit_threshold: float
    k_revert: float
    min_spread: float
    max_spread: float
    min_liquidity: float
    max_ttl_seconds: int
    epsilon: float


@dataclass(frozen=True)
class ReversionState:
    """Estado persistente EWMA."""

    ewma_mean: float
    ewma_var: float
    last_mid: float
    last_timestamp: float


@dataclass(frozen=True)
class ReversionSignal:
    """Sinal estruturado (forecast + intent)."""

    z_score: float
    shock: float
    p_est: float
    edge_estimate: float
    side: Optional[str]
    confidence: float
    rationale: tuple  # frozen → tuple


def _finite(x: float) -> bool:
    return math.isfinite(x) and not math.isnan(x)


def _neutral_signal(
    z: float,
    shock: float,
    mid: float,
    rationale: List[str],
) -> ReversionSignal:
    """Sinal neutro (side=None). Sem propagação de NaN."""
    mid_safe = mid if _finite(mid) else 0.5
    p_est = mid_safe - 0.25 * shock
    p_est = max(0.01, min(0.99, p_est))
    edge = p_est - mid_safe if _finite(mid) else 0.0
    return ReversionSignal(
        z_score=z if _finite(z) else 0.0,
        shock=shock if _finite(shock) else 0.0,
        p_est=p_est,
        edge_estimate=edge,
        side=None,
        confidence=0.0,
        rationale=tuple(rationale),
    )


class ReversionEngine:
    """
    Engine estatístico de reversão à média parcial.
    Determinístico, side-effect free.
    """

    def __init__(self, config: ReversionConfig) -> None:
        self._config = config

    def update_state(
        self,
        state: ReversionState,
        mid: float,
        timestamp: Optional[float] = None,
    ) -> ReversionState:
        """
        Atualiza EWMA mean e variance.
        Mantém estado se mid não finito.
        """
        if not _finite(mid):
            logger.warning(
                "[REVERSION_ENGINE] update_state: mid não finito, mantendo estado",
                extra={"mid": mid},
            )
            return state
        alpha = self._config.alpha
        eps = self._config.epsilon
        mu_prev = state.ewma_mean
        var_prev = state.ewma_var
        mu_t = alpha * mid + (1 - alpha) * mu_prev
        var_t = alpha * (mid - mu_prev) ** 2 + (1 - alpha) * var_prev
        var_t = max(var_t, eps)
        ts = timestamp if timestamp is not None else time.time()
        return ReversionState(
            ewma_mean=mu_t,
            ewma_var=var_t,
            last_mid=mid,
            last_timestamp=ts,
        )

    def generate_signal(
        self,
        state: ReversionState,
        mid: float,
        spread: float,
        liquidity: float,
        operational_confidence: float,
        safe_to_buy: bool,
        market_id: str = "",
    ) -> ReversionSignal:
        """
        Gera sinal estruturado. Neutro se inputs inválidos.
        BUY só se safe_to_buy. SELL sempre permitido.
        """
        eps = self._config.epsilon
        rationale: List[str] = []

        if not _finite(mid) or not _finite(spread) or not _finite(liquidity):
            rationale.append("input não finito")
            return _neutral_signal(0.0, 0.0, mid if _finite(mid) else 0.5, rationale)
        if spread < 0 or liquidity < 0 or operational_confidence < 0:
            rationale.append("input negativo")
            return _neutral_signal(0.0, 0.0, mid, rationale)
        if operational_confidence > 1.0:
            rationale.append("operational_confidence > 1")
            return _neutral_signal(0.0, 0.0, mid, rationale)

        cfg = self._config
        if spread < cfg.min_spread:
            rationale.append(f"spread {spread:.4f} < min {cfg.min_spread}")
            return _neutral_signal(0.0, 0.0, mid, rationale)
        if spread > cfg.max_spread:
            rationale.append(f"spread {spread:.4f} > max {cfg.max_spread}")
            return _neutral_signal(0.0, 0.0, mid, rationale)
        if liquidity < cfg.min_liquidity:
            rationale.append(f"liquidity {liquidity:.0f} < min {cfg.min_liquidity}")
            return _neutral_signal(0.0, 0.0, mid, rationale)

        shock = mid - state.ewma_mean
        sigma = math.sqrt(max(state.ewma_var, eps))
        z = shock / max(sigma, eps)
        rationale.append(f"z={z:.3f}")
        rationale.append(f"spread={spread:.4f}")
        rationale.append(f"liquidity={liquidity:.0f}")

        side: Optional[str] = None
        if z <= -cfg.z_entry_threshold and safe_to_buy:
            side = "BUY"
            rationale.append(f"entry BUY (z <= -{cfg.z_entry_threshold})")
        elif z >= cfg.z_entry_threshold:
            side = "SELL"
            rationale.append(f"entry SELL (z >= {cfg.z_entry_threshold})")
        else:
            rationale.append("threshold não atingido")

        if not safe_to_buy and z <= -cfg.z_entry_threshold:
            rationale.append("BUY bloqueado por SafetyState")
            side = None

        p_est = mid - cfg.k_revert * shock
        p_est = max(0.01, min(0.99, p_est))
        edge_estimate = p_est - mid

        base_conf = min(1.0, abs(z) / 4.0)
        if operational_confidence < 0.9:
            conf_scale = operational_confidence / 0.9
            confidence = base_conf * conf_scale
        else:
            confidence = base_conf

        logger.info(
            "[REVERSION_ENGINE] signal",
            extra={
                "market_id": market_id,
                "z_score": round(z, 4),
                "side": side,
                "p_est": round(p_est, 4),
                "mid": round(mid, 4),
                "spread": round(spread, 4),
                "liquidity": round(liquidity, 2),
                "operational_confidence": round(operational_confidence, 2),
            },
        )
        return ReversionSignal(
            z_score=z,
            shock=shock,
            p_est=p_est,
            edge_estimate=edge_estimate,
            side=side,
            confidence=confidence,
            rationale=tuple(rationale),
        )


class ReversionMetricsCollector:
    """
    Coletor de métricas para scorecard e dashboard.
    Não modifica core. Apenas agrega dados de sinais.
    """

    def __init__(self, window_seconds: int = 86400) -> None:
        self._window = window_seconds
        self._z_history: deque = deque(maxlen=10000)
        self._signals: deque = deque(maxlen=1000)
        self._last_signal_ts: float = 0.0

    def record_signal(
        self,
        z_score: float,
        side: Optional[str],
        p_est: float,
        mid: float,
        timestamp: float,
    ) -> None:
        """Registra sinal (chamado pelo integrador, não pelo engine)."""
        self._z_history.append((timestamp, z_score))
        if side is not None:
            self._signals.append({
                "ts": timestamp,
                "z": z_score,
                "side": side,
                "p_est": p_est,
                "mid": mid,
            })
            self._last_signal_ts = timestamp

    def get_metrics(self) -> dict:
        """Retorna métricas agregadas para /technical."""
        now = time.time()
        cut = now - self._window
        z_recent = [z for (t, z) in self._z_history if t >= cut]
        sig_recent = [s for s in self._signals if s["ts"] >= cut]
        n = len(z_recent)
        if n == 0:
            return {
                "current_z": None,
                "rolling_z_mean": None,
                "rolling_z_std": None,
                "signal_count_last_24h": 0,
                "hit_rate_simulated": None,
                "average_reversion_magnitude": None,
                "average_time_to_reversion": None,
            }
        mean_z = sum(z_recent) / n
        var_z = sum((z - mean_z) ** 2 for z in z_recent) / n
        std_z = math.sqrt(var_z) if var_z > 0 else 0.0
        current_z = z_recent[-1] if z_recent else None
        avg_revert = None
        avg_ttr = None
        hit_rate = None
        if sig_recent:
            mags = [abs(s["p_est"] - s["mid"]) for s in sig_recent]
            avg_revert = sum(mags) / len(mags) if mags else None
            hit_rate = 0.5
            avg_ttr = 0.0
        return {
            "current_z": round(current_z, 4) if current_z is not None else None,
            "rolling_z_mean": round(mean_z, 4),
            "rolling_z_std": round(std_z, 4),
            "signal_count_last_24h": len(sig_recent),
            "hit_rate_simulated": hit_rate,
            "average_reversion_magnitude": round(avg_revert, 4) if avg_revert is not None else None,
            "average_time_to_reversion": avg_ttr,
        }


def create_default_config() -> ReversionConfig:
    """Config padrão para OBSERVE mode."""
    return ReversionConfig(
        alpha=DEFAULT_ALPHA,
        z_entry_threshold=DEFAULT_Z_ENTRY,
        z_exit_threshold=DEFAULT_Z_EXIT,
        k_revert=DEFAULT_K_REVERT,
        min_spread=DEFAULT_MIN_SPREAD,
        max_spread=DEFAULT_MAX_SPREAD,
        min_liquidity=DEFAULT_MIN_LIQUIDITY,
        max_ttl_seconds=DEFAULT_MAX_TTL,
        epsilon=DEFAULT_EPSILON,
    )
