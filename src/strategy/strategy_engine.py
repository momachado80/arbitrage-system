"""
Strategy Engine — Quantitative Trading Strategy
==================================================

Integra o ReversionEngine com o pipeline de dados de mercado.

TESE QUANTITATIVA:
- Detecta overreação estatística em preços de mercados preditivos
- Usa EWMA para calcular z-score do midpoint
- Gera sinais BUY quando z <= -threshold (preço caiu demais)
- Gera sinais SELL quando z >= +threshold (preço subiu demais)
- Position sizing via ProbabilityCalibrator (Brier score)

PIPELINE:
1. Recebe MarketSnapshot do PipelineHandler
2. Calcula midpoint = (best_bid + best_ask) / 2
3. Atualiza EWMA state
4. Gera ReversionSignal
5. Se sinal acionável → DecisionPipeline

MÉTRICAS:
- Histórico de z-scores por token
- Contagem de sinais gerados
- Performance simulada (hit rate, edge médio)

Thread-safe via RLock.
"""

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.strategy.reversion_engine import (
    ReversionConfig,
    ReversionEngine,
    ReversionMetricsCollector,
    ReversionSignal,
    ReversionState,
    create_default_config,
)
from src.watcher.bridge import MarketSnapshot

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_HISTORY_MAXLEN: int = 5000
DEFAULT_SIGNAL_HISTORY_MAXLEN: int = 500
MIN_SNAPSHOTS_BEFORE_SIGNAL: int = 10
WARMUP_SNAPSHOTS: int = 30


@dataclass
class MarketState:
    """Estado interno para cada token monitorado."""
    token_id: str
    reversion_state: ReversionState
    snapshot_count: int = 0
    last_mid: float = 0.0
    last_spread: float = 0.0
    last_best_bid: Optional[float] = None
    last_best_ask: Optional[float] = None
    last_update_ts: float = 0.0


@dataclass
class StrategySignal:
    """Sinal estruturado para o DecisionPipeline."""
    timestamp: float
    token_id: str
    side: str           # "BUY" ou "SELL"
    z_score: float
    midpoint: float
    spread: float
    edge_estimate: float
    confidence: float
    p_est: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    rationale: Tuple[str, ...]


class StrategyEngine:
    """
    Motor de estratégia quantitativa.

    Recebe snapshots, mantém estado EWMA por token,
    gera sinais acionáveis para o DecisionPipeline.
    """

    def __init__(
        self,
        config: Optional[ReversionConfig] = None,
        safety_state: Optional[Any] = None,
        min_liquidity: float = 50.0,
    ) -> None:
        self._config = config or create_default_config()
        self._engine = ReversionEngine(self._config)
        self._safety_state = safety_state
        self._min_liquidity = min_liquidity
        self._lock = threading.RLock()

        # Estado por token
        self._markets: Dict[str, MarketState] = {}

        # Históricos
        self._z_history: Dict[str, deque] = {}
        self._mid_history: Dict[str, deque] = {}
        self._signal_history: deque = deque(maxlen=DEFAULT_SIGNAL_HISTORY_MAXLEN)

        # Métricas
        self._metrics_collector = ReversionMetricsCollector()
        self._total_signals: int = 0
        self._total_updates: int = 0

    def process_snapshot(self, snapshot: MarketSnapshot) -> Optional[StrategySignal]:
        """
        Processa MarketSnapshot e retorna StrategySignal se acionável.

        Returns None se:
        - Dados insuficientes (warmup)
        - Sem spread válido
        - Z-score abaixo do threshold
        - Safety bloqueando BUY
        """
        bid = snapshot.best_bid_price
        ask = snapshot.best_ask_price

        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return None
        if ask <= bid:
            return None

        mid = (bid + ask) / 2.0
        spread = ask - bid
        token_id = snapshot.token_id
        now = snapshot.timestamp or time.time()

        with self._lock:
            self._total_updates += 1
            state = self._get_or_create_market(token_id, mid, now)
            state.snapshot_count += 1
            state.last_mid = mid
            state.last_spread = spread
            state.last_best_bid = bid
            state.last_best_ask = ask
            state.last_update_ts = now

            # Histórico de midpoints
            if token_id not in self._mid_history:
                self._mid_history[token_id] = deque(maxlen=DEFAULT_HISTORY_MAXLEN)
            self._mid_history[token_id].append((now, mid))

            # Atualizar EWMA
            new_reversion_state = self._engine.update_state(
                state.reversion_state, mid, now
            )
            state.reversion_state = new_reversion_state

            # Warmup: precisa de snapshots suficientes
            if state.snapshot_count < MIN_SNAPSHOTS_BEFORE_SIGNAL:
                return None

            # Verificar safety
            safe_to_buy = True
            if self._safety_state is not None:
                safe_to_buy = not self._safety_state.buy_blocked

            # Estimar liquidez do book
            bid_size = snapshot.best_bid_size or 0.0
            ask_size = snapshot.best_ask_size or 0.0
            liquidity = (bid_size + ask_size) * mid

            # Confidence operacional
            op_confidence = self._compute_operational_confidence(state)

            # Gerar sinal
            signal = self._engine.generate_signal(
                state=new_reversion_state,
                mid=mid,
                spread=spread,
                liquidity=liquidity,
                operational_confidence=op_confidence,
                safe_to_buy=safe_to_buy,
                market_id=token_id,
            )

            # Histórico de z-scores
            if token_id not in self._z_history:
                self._z_history[token_id] = deque(maxlen=DEFAULT_HISTORY_MAXLEN)
            z = signal.z_score if math.isfinite(signal.z_score) else 0.0
            self._z_history[token_id].append((now, z))

            # Registrar no metrics collector
            self._metrics_collector.record_signal(
                z_score=z,
                side=signal.side,
                p_est=signal.p_est,
                mid=mid,
                timestamp=now,
            )

            # Se sinal acionável, retornar
            if signal.side is not None:
                self._total_signals += 1
                strat_signal = StrategySignal(
                    timestamp=now,
                    token_id=token_id,
                    side=signal.side,
                    z_score=signal.z_score,
                    midpoint=mid,
                    spread=spread,
                    edge_estimate=signal.edge_estimate,
                    confidence=signal.confidence,
                    p_est=signal.p_est,
                    best_bid=bid,
                    best_ask=ask,
                    rationale=signal.rationale,
                )
                self._signal_history.append({
                    "ts": now,
                    "token_id": token_id,
                    "side": signal.side,
                    "z": round(signal.z_score, 4),
                    "mid": round(mid, 4),
                    "spread": round(spread, 4),
                    "edge": round(signal.edge_estimate, 4),
                    "confidence": round(signal.confidence, 3),
                })
                logger.info(
                    "[STRATEGY] Signal: %s %s z=%.3f mid=%.4f spread=%.4f edge=%.4f conf=%.2f",
                    signal.side, token_id[:12], signal.z_score,
                    mid, spread, signal.edge_estimate, signal.confidence,
                )
                return strat_signal

            return None

    def _get_or_create_market(
        self, token_id: str, initial_mid: float, ts: float
    ) -> MarketState:
        """Obtém ou cria MarketState para um token."""
        if token_id not in self._markets:
            self._markets[token_id] = MarketState(
                token_id=token_id,
                reversion_state=ReversionState(
                    ewma_mean=initial_mid,
                    ewma_var=1e-6,
                    last_mid=initial_mid,
                    last_timestamp=ts,
                ),
            )
        return self._markets[token_id]

    def _compute_operational_confidence(self, state: MarketState) -> float:
        """
        Confiança operacional baseada em:
        - Número de snapshots (warmup)
        - Recência do último update
        """
        # Ramp up durante warmup
        if state.snapshot_count < WARMUP_SNAPSHOTS:
            base = state.snapshot_count / WARMUP_SNAPSHOTS
        else:
            base = 1.0

        # Penalizar se dados antigos
        age = time.time() - state.last_update_ts
        if age > 60:
            staleness_penalty = max(0.5, 1.0 - (age - 60) / 300)
        else:
            staleness_penalty = 1.0

        return min(1.0, base * staleness_penalty)

    # ------------------------------------------------------------------
    # Leitura de métricas
    # ------------------------------------------------------------------

    def get_market_data(self) -> List[Dict[str, Any]]:
        """Retorna dados de mercado atuais para todos os tokens monitorados."""
        with self._lock:
            result = []
            for token_id, state in self._markets.items():
                z_hist = self._z_history.get(token_id, deque())
                current_z = z_hist[-1][1] if z_hist else None
                result.append({
                    "token_id": token_id,
                    "midpoint": round(state.last_mid, 4) if state.last_mid else None,
                    "spread": round(state.last_spread, 4) if state.last_spread else None,
                    "best_bid": state.last_best_bid,
                    "best_ask": state.last_best_ask,
                    "z_score": round(current_z, 4) if current_z is not None else None,
                    "snapshot_count": state.snapshot_count,
                    "last_update": state.last_update_ts,
                    "ewma_mean": round(state.reversion_state.ewma_mean, 4),
                    "ewma_var": round(state.reversion_state.ewma_var, 6),
                })
            return result

    def get_z_history(self, token_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna histórico de z-scores para um token."""
        with self._lock:
            hist = self._z_history.get(token_id, deque())
            entries = list(hist)[-limit:]
            return [{"ts": ts, "z": round(z, 4)} for ts, z in entries]

    def get_mid_history(self, token_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna histórico de midpoints para um token."""
        with self._lock:
            hist = self._mid_history.get(token_id, deque())
            entries = list(hist)[-limit:]
            return [{"ts": ts, "mid": round(m, 4)} for ts, m in entries]

    def get_signal_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna últimos sinais gerados."""
        with self._lock:
            entries = list(self._signal_history)
            return entries[-limit:]

    def get_strategy_summary(self) -> Dict[str, Any]:
        """Retorna resumo da estratégia."""
        with self._lock:
            markets_count = len(self._markets)
            total_snapshots = sum(
                s.snapshot_count for s in self._markets.values()
            )
            signals_24h = sum(
                1 for s in self._signal_history
                if s["ts"] > time.time() - 86400
            )
            buy_signals = sum(
                1 for s in self._signal_history
                if s["side"] == "BUY" and s["ts"] > time.time() - 86400
            )
            sell_signals = sum(
                1 for s in self._signal_history
                if s["side"] == "SELL" and s["ts"] > time.time() - 86400
            )

            # Reversion metrics
            rev_metrics = self._metrics_collector.get_metrics()

            return {
                "strategy": "Mean Reversion (EWMA Z-Score)",
                "markets_monitored": markets_count,
                "total_snapshots_processed": total_snapshots,
                "total_signals_generated": self._total_signals,
                "signals_last_24h": signals_24h,
                "buy_signals_24h": buy_signals,
                "sell_signals_24h": sell_signals,
                "config": {
                    "alpha": self._config.alpha,
                    "z_entry_threshold": self._config.z_entry_threshold,
                    "z_exit_threshold": self._config.z_exit_threshold,
                    "k_revert": self._config.k_revert,
                    "min_spread": self._config.min_spread,
                    "max_spread": self._config.max_spread,
                    "min_liquidity": self._config.min_liquidity,
                },
                "reversion_metrics": rev_metrics,
            }

    @property
    def metrics_collector(self) -> ReversionMetricsCollector:
        """Acesso ao coletor de métricas para dashboard."""
        return self._metrics_collector
