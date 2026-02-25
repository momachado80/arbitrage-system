"""
Oracle Metrics — Persistent Measurement and Reporting
========================================================

Métricas obrigatórias:
- rolling_brier_score: Brier score rolling por janela
- calibration_bins: erro de calibração por bucket de probabilidade
- edge_capture_ratio: edge capturado vs edge previsto
- edge_decay_distribution: distribuição de half-life do edge
- profit_per_unit_edge: PnL realizado por unidade de edge

Persistência leve: data/oracle_metrics.jsonl
Sem banco. Um JSON por linha, append-only.

Thread-safe via RLock.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_METRICS_PATH = os.path.join(
    os.environ.get("DATA_DIR", "data"), "oracle_metrics.jsonl",
)
ROLLING_WINDOW = 200
CALIBRATION_BINS = 10


@dataclass
class MetricEvent:
    """Um evento de métrica para persistência."""
    event_type: str           # "prediction", "outcome", "edge_signal", "decay"
    timestamp: float
    token_id: str
    data: Dict[str, Any]


class OracleMetricsStore:
    """
    Coleta, agrega e persiste métricas do Oracle.

    Métricas computadas:
    1. rolling_brier_score - Brier score rolling (janela configurável)
    2. calibration_bins - erro por bucket de probabilidade
    3. edge_capture_ratio - captura real vs edge previsto
    4. edge_decay_distribution - distribuição de half-life
    5. profit_per_unit_edge - PnL / edge previsto

    Persistência:
    - Append-only para data/oracle_metrics.jsonl
    - Cada evento é um JSON por linha
    - Diretório criado automaticamente se não existir

    Thread-safe.
    """

    def __init__(
        self,
        metrics_path: Optional[str] = None,
        rolling_window: int = ROLLING_WINDOW,
    ) -> None:
        self._metrics_path = metrics_path or DEFAULT_METRICS_PATH
        self._rolling_window = rolling_window
        self._lock = threading.RLock()

        # Rolling buffers
        self._brier_scores: Deque[float] = deque(maxlen=rolling_window)
        self._predictions: Deque[Dict[str, Any]] = deque(maxlen=rolling_window * 2)
        self._outcomes: Deque[Dict[str, Any]] = deque(maxlen=rolling_window)
        self._edge_signals: Deque[Dict[str, Any]] = deque(maxlen=rolling_window)
        self._decay_half_lives: Deque[float] = deque(maxlen=rolling_window)
        self._pnl_per_edge: Deque[Dict[str, float]] = deque(maxlen=rolling_window)

        # Per-token Brier
        self._token_brier: Dict[str, Deque[float]] = {}

        # Pending predictions (waiting for outcomes)
        self._pending: Dict[str, Dict[str, Any]] = {}

        # Ensure directory exists
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Cria diretório para métricas se não existir."""
        d = os.path.dirname(self._metrics_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _persist(self, event: MetricEvent) -> None:
        """Append evento ao JSONL (fire-and-forget)."""
        try:
            with open(self._metrics_path, "a") as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")
        except Exception as e:
            logger.warning("[ORACLE_METRICS] Persist failed: %s", e)

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        token_id: str,
        p_oracle: float,
        p_market: float,
        confidence: float,
        source: str,
        edge: Optional[float] = None,
    ) -> None:
        """Registra predição do oracle."""
        now = time.time()
        entry = {
            "token_id": token_id,
            "timestamp": now,
            "p_oracle": p_oracle,
            "p_market": p_market,
            "confidence": confidence,
            "source": source,
            "edge": edge or (p_oracle - p_market),
        }
        with self._lock:
            self._predictions.append(entry)
            self._pending[token_id] = entry

        self._persist(MetricEvent(
            event_type="prediction",
            timestamp=now,
            token_id=token_id,
            data=entry,
        ))

    def record_outcome(
        self,
        token_id: str,
        outcome: float,
        realized_pnl: Optional[float] = None,
    ) -> None:
        """
        Registra outcome real e calcula Brier.

        Args:
            token_id: token resolvido
            outcome: 1.0 = YES, 0.0 = NO
            realized_pnl: PnL real do trade (se executado)
        """
        now = time.time()
        with self._lock:
            pending = self._pending.pop(token_id, None)
            if pending is None:
                return

            p_oracle = pending["p_oracle"]
            p_market = pending["p_market"]
            edge = pending["edge"]

            # Brier score
            brier = (p_oracle - outcome) ** 2
            self._brier_scores.append(brier)

            # Per-token Brier
            if token_id not in self._token_brier:
                self._token_brier[token_id] = deque(maxlen=self._rolling_window)
            self._token_brier[token_id].append(brier)

            # Edge capture
            if edge > 0:
                captured = outcome - p_market
            else:
                captured = p_market - outcome

            # PnL per unit edge
            if realized_pnl is not None and abs(edge) > 0.001:
                self._pnl_per_edge.append({
                    "pnl": realized_pnl,
                    "edge": abs(edge),
                    "ratio": realized_pnl / abs(edge),
                })

            outcome_entry = {
                "token_id": token_id,
                "timestamp": now,
                "p_oracle": p_oracle,
                "p_market": p_market,
                "outcome": outcome,
                "brier": brier,
                "predicted_edge": edge,
                "captured_edge": captured,
                "realized_pnl": realized_pnl,
            }
            self._outcomes.append(outcome_entry)

        self._persist(MetricEvent(
            event_type="outcome",
            timestamp=now,
            token_id=token_id,
            data=outcome_entry,
        ))

    def record_edge_signal(
        self,
        token_id: str,
        net_edge: float,
        side: str,
        p_oracle: float,
        p_market: float,
        execution_price: float,
    ) -> None:
        """Registra sinal de edge detectado."""
        now = time.time()
        entry = {
            "token_id": token_id,
            "timestamp": now,
            "net_edge": net_edge,
            "side": side,
            "p_oracle": p_oracle,
            "p_market": p_market,
            "execution_price": execution_price,
        }
        with self._lock:
            self._edge_signals.append(entry)

        self._persist(MetricEvent(
            event_type="edge_signal",
            timestamp=now,
            token_id=token_id,
            data=entry,
        ))

    def record_decay(
        self,
        token_id: str,
        half_life_seconds: float,
        initial_edge: float,
    ) -> None:
        """Registra half-life de decay de edge."""
        now = time.time()
        with self._lock:
            self._decay_half_lives.append(half_life_seconds)

        self._persist(MetricEvent(
            event_type="decay",
            timestamp=now,
            token_id=token_id,
            data={
                "half_life_seconds": half_life_seconds,
                "initial_edge": initial_edge,
            },
        ))

    # ------------------------------------------------------------------
    # Computed metrics
    # ------------------------------------------------------------------

    def rolling_brier_score(self) -> Optional[float]:
        """Brier score rolling (menor = melhor, 0 = perfeito)."""
        with self._lock:
            if not self._brier_scores:
                return None
            return sum(self._brier_scores) / len(self._brier_scores)

    def brier_score_for_token(self, token_id: str) -> Optional[float]:
        """Brier score rolling para token específico."""
        with self._lock:
            scores = self._token_brier.get(token_id)
            if not scores:
                return None
            return sum(scores) / len(scores)

    def calibration_bins(self) -> Dict[str, Any]:
        """
        Calibração por bucket de probabilidade.

        Retorna bins com predicted_avg, outcome_avg, count, error.
        """
        with self._lock:
            if not self._outcomes:
                return {"bins": [], "mean_calibration_error": None}

            bin_size = 1.0 / CALIBRATION_BINS
            bins = []

            for i in range(CALIBRATION_BINS):
                lo = i * bin_size
                hi = (i + 1) * bin_size
                items = [
                    o for o in self._outcomes
                    if lo <= o["p_oracle"] < hi
                ]
                if not items:
                    continue

                pred_avg = sum(o["p_oracle"] for o in items) / len(items)
                outcome_avg = sum(o["outcome"] for o in items) / len(items)
                error = abs(pred_avg - outcome_avg)

                bins.append({
                    "range": f"{lo:.1f}-{hi:.1f}",
                    "predicted_avg": round(pred_avg, 4),
                    "outcome_avg": round(outcome_avg, 4),
                    "count": len(items),
                    "error": round(error, 4),
                })

            mce = (
                sum(b["error"] * b["count"] for b in bins) / sum(b["count"] for b in bins)
                if bins else None
            )
            return {
                "bins": bins,
                "mean_calibration_error": round(mce, 4) if mce is not None else None,
            }

    def edge_capture_ratio(self) -> Optional[float]:
        """
        Ratio: edge capturado / edge previsto.

        > 1.0 = capturando mais do que previsto
        < 1.0 = capturando menos (slippage, timing)
        """
        with self._lock:
            if not self._outcomes:
                return None
            predicted = sum(abs(o["predicted_edge"]) for o in self._outcomes)
            captured = sum(o["captured_edge"] for o in self._outcomes)
            if predicted <= 0:
                return None
            return captured / predicted

    def edge_decay_distribution(self) -> Dict[str, Any]:
        """Distribuição de half-life do edge decay."""
        with self._lock:
            if not self._decay_half_lives:
                return {
                    "avg_seconds": None,
                    "median_seconds": None,
                    "min_seconds": None,
                    "max_seconds": None,
                    "sample_count": 0,
                }
            hl = sorted(self._decay_half_lives)
            n = len(hl)
            return {
                "avg_seconds": round(sum(hl) / n, 1),
                "median_seconds": round(hl[n // 2], 1),
                "min_seconds": round(hl[0], 1),
                "max_seconds": round(hl[-1], 1),
                "sample_count": n,
            }

    def profit_per_unit_edge(self) -> Optional[float]:
        """PnL realizado por unidade de edge previsto."""
        with self._lock:
            if not self._pnl_per_edge:
                return None
            ratios = [p["ratio"] for p in self._pnl_per_edge]
            return sum(ratios) / len(ratios)

    # ------------------------------------------------------------------
    # Aggregate report
    # ------------------------------------------------------------------

    def get_full_report(self) -> Dict[str, Any]:
        """Relatório completo de métricas."""
        brier = self.rolling_brier_score()
        return {
            "rolling_brier_score": round(brier, 4) if brier is not None else None,
            "calibration_bins": self.calibration_bins(),
            "edge_capture_ratio": (
                round(self.edge_capture_ratio(), 4)
                if self.edge_capture_ratio() is not None else None
            ),
            "edge_decay_distribution": self.edge_decay_distribution(),
            "profit_per_unit_edge": (
                round(self.profit_per_unit_edge(), 4)
                if self.profit_per_unit_edge() is not None else None
            ),
            "total_predictions": len(self._predictions),
            "total_outcomes": len(self._outcomes),
            "pending_predictions": len(self._pending),
            "total_edge_signals": len(self._edge_signals),
            "brier_interpretation": (
                "excellent" if brier is not None and brier < 0.1
                else "good" if brier is not None and brier < 0.2
                else "moderate" if brier is not None and brier < 0.3
                else "poor" if brier is not None
                else "insufficient_data"
            ),
        }

    def get_signals_last_24h(self) -> List[Dict[str, Any]]:
        """Retorna sinais das últimas 24h."""
        cutoff = time.time() - 86400
        with self._lock:
            return [s for s in self._edge_signals if s["timestamp"] > cutoff]
