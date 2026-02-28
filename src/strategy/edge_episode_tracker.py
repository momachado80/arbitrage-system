"""
Edge Episode Tracker — Prova de Edge Empírica
================================================

Registra episódios de edge (início/fim) com duração e expectativa líquida.
Config via env. Sem persistência (só memória). Expõe agregados.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EDGE_ENTER_BPS: float = float(os.environ.get("EDGE_ENTER_BPS", "30"))
EDGE_EXIT_BPS: float = float(os.environ.get("EDGE_EXIT_BPS", "10"))
MAX_EPISODE_SECONDS: float = float(os.environ.get("MAX_EPISODE_SECONDS", "300"))
LAT_MS: float = float(os.environ.get("LAT_MS", "500"))
ORDER_SIZE: float = float(os.environ.get("ORDER_SIZE", "10"))
FEE_BPS: float = float(os.environ.get("FEE_BPS", "0"))


@dataclass
class EdgeEpisode:
    token_id: str
    ts_start: float
    ts_end: float
    duration_ms: float
    side: str
    mid_start: float
    mid_end: float
    p_oracle_start: float
    spread_start: float
    edge_bps_start: float
    expected_net_edge_bps_start: float
    capacity_est: float
    dropped_reason: Optional[str] = None


@dataclass
class OpenEpisode:
    token_id: str
    ts_start: float
    side: str
    mid_start: float
    p_oracle_start: float
    spread_start: float
    edge_bps_start: float
    expected_net_edge_bps_start: float
    capacity_est: float


class EdgeEpisodeTracker:
    """
    Detecta episódios de edge (entrada/saída) e registra estatísticas.
    Thread-safe. Sem I/O. Apenas memória.
    """

    def __init__(
        self,
        edge_enter_bps: float = EDGE_ENTER_BPS,
        edge_exit_bps: float = EDGE_EXIT_BPS,
        max_episode_seconds: float = MAX_EPISODE_SECONDS,
        order_size: float = ORDER_SIZE,
        fee_bps: float = FEE_BPS,
    ) -> None:
        self._edge_enter_bps = edge_enter_bps
        self._edge_exit_bps = edge_exit_bps
        self._max_episode_seconds = max_episode_seconds
        self._order_size = order_size
        self._fee_bps = fee_bps
        self._lock = threading.RLock()
        self._open: Dict[str, OpenEpisode] = {}
        self._closed: List[EdgeEpisode] = []
        self._max_closed: int = 5000

    def update(
        self,
        token_id: str,
        ts: float,
        best_bid: float,
        bid_size: Optional[float],
        best_ask: float,
        ask_size: Optional[float],
        p_oracle: float,
    ) -> None:
        if best_bid is None or best_ask is None or best_bid <= 0 or best_ask <= 0:
            return
        if best_ask <= best_bid:
            return

        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        bid_sz = bid_size if bid_size is not None else 0.0
        ask_sz = ask_size if ask_size is not None else 0.0

        raw_edge = p_oracle - mid
        if raw_edge > 0:
            side = "BUY"
            edge_bps = (p_oracle - best_ask) * 10000.0
            capacity_est = ask_sz
        elif raw_edge < 0:
            side = "SELL"
            edge_bps = (best_bid - p_oracle) * 10000.0
            capacity_est = bid_sz
        else:
            return

        if self._order_size > capacity_est:
            dropped_reason = "insufficient_top_liquidity"
            self._maybe_close(token_id, ts, mid, dropped_reason)
            return

        slippage_bps = 0.0
        expected_net_edge_bps = edge_bps - slippage_bps - self._fee_bps

        with self._lock:
            if token_id in self._open:
                ep = self._open[token_id]
                if (
                    expected_net_edge_bps <= self._edge_exit_bps
                    or (ts - ep.ts_start) >= self._max_episode_seconds
                ):
                    duration_ms = (ts - ep.ts_start) * 1000.0
                    closed_ep = EdgeEpisode(
                        token_id=ep.token_id,
                        ts_start=ep.ts_start,
                        ts_end=ts,
                        duration_ms=duration_ms,
                        side=ep.side,
                        mid_start=ep.mid_start,
                        mid_end=mid,
                        p_oracle_start=ep.p_oracle_start,
                        spread_start=ep.spread_start,
                        edge_bps_start=ep.edge_bps_start,
                        expected_net_edge_bps_start=ep.expected_net_edge_bps_start,
                        capacity_est=ep.capacity_est,
                        dropped_reason=None,
                    )
                    self._closed.append(closed_ep)
                    if len(self._closed) > self._max_closed:
                        self._closed = self._closed[-self._max_closed:]
                    del self._open[token_id]
            elif expected_net_edge_bps >= self._edge_enter_bps:
                self._open[token_id] = OpenEpisode(
                    token_id=token_id,
                    ts_start=ts,
                    side=side,
                    mid_start=mid,
                    p_oracle_start=p_oracle,
                    spread_start=spread,
                    edge_bps_start=edge_bps,
                    expected_net_edge_bps_start=expected_net_edge_bps,
                    capacity_est=capacity_est,
                )

    def _maybe_close(self, token_id: str, ts: float, mid: float, dropped_reason: Optional[str]) -> None:
        with self._lock:
            self._maybe_close_impl(token_id, ts, mid, dropped_reason)

    def _maybe_close_impl(
        self, token_id: str, ts: float, mid: float, dropped_reason: Optional[str]
    ) -> None:
        if token_id not in self._open:
            return
        ep = self._open[token_id]
        duration_ms = (ts - ep.ts_start) * 1000.0
        closed_ep = EdgeEpisode(
            token_id=ep.token_id,
            ts_start=ep.ts_start,
            ts_end=ts,
            duration_ms=duration_ms,
            side=ep.side,
            mid_start=ep.mid_start,
            mid_end=mid,
            p_oracle_start=ep.p_oracle_start,
            spread_start=ep.spread_start,
            edge_bps_start=ep.edge_bps_start,
            expected_net_edge_bps_start=ep.expected_net_edge_bps_start,
            capacity_est=ep.capacity_est,
            dropped_reason=dropped_reason,
        )
        self._closed.append(closed_ep)
        if len(self._closed) > self._max_closed:
            self._closed = self._closed[-self._max_closed:]
        del self._open[token_id]

    def get_aggregates(self) -> Dict[str, Any]:
        with self._lock:
            closed = list(self._closed)
            open_count = len(self._open)

        if not closed:
            return {
                "edge_episodes_total": 0,
                "edge_episodes_open": open_count,
                "edge_episodes_closed": 0,
                "edge_duration_ms_p50": None,
                "edge_duration_ms_p90": None,
                "expected_net_edge_bps_p50": None,
                "expected_net_edge_bps_p90": None,
                "episodes_survive_500ms": 0.0,
                "episodes_survive_2000ms": 0.0,
                "episodes_survive_5000ms": 0.0,
                "top_5_markets_by_ev": [],
            }

        durations = [e.duration_ms for e in closed]
        edges = [e.expected_net_edge_bps_start for e in closed]
        durations_sorted = sorted(durations)
        edges_sorted = sorted(edges)
        n = len(closed)

        def _p(arr: List[float], p: float) -> Optional[float]:
            if not arr:
                return None
            idx = min(n - 1, max(0, int(n * p / 100.0)))
            return round(arr[idx], 2)

        p50_dur = _p(durations_sorted, 50)
        p90_dur = _p(durations_sorted, 90)
        p50_edge = _p(edges_sorted, 50)
        p90_edge = _p(edges_sorted, 90)

        survive_500 = sum(1 for d in durations if d >= 500) / n if n else 0.0
        survive_2000 = sum(1 for d in durations if d >= 2000) / n if n else 0.0
        survive_5000 = sum(1 for d in durations if d >= 5000) / n if n else 0.0

        ev_by_token: Dict[str, float] = {}
        for e in closed:
            if e.dropped_reason:
                continue
            ev = e.expected_net_edge_bps_start * (e.duration_ms / 1000.0)
            ev_by_token[e.token_id] = ev_by_token.get(e.token_id, 0) + ev

        top_5 = sorted(
            [{"token_id": k, "ev": round(v, 2)} for k, v in ev_by_token.items()],
            key=lambda x: -x["ev"],
        )[:5]

        return {
            "edge_episodes_total": n + open_count,
            "edge_episodes_open": open_count,
            "edge_episodes_closed": n,
            "edge_duration_ms_p50": p50_dur,
            "edge_duration_ms_p90": p90_dur,
            "expected_net_edge_bps_p50": p50_edge,
            "expected_net_edge_bps_p90": p90_edge,
            "episodes_survive_500ms": round(survive_500, 4),
            "episodes_survive_2000ms": round(survive_2000, 4),
            "episodes_survive_5000ms": round(survive_5000, 4),
            "top_5_markets_by_ev": top_5,
        }
