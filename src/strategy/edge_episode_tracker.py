"""
Edge Episode Tracker — Prova de Edge Empírica
================================================

Registra episódios de edge (início/fim) com duração e expectativa líquida.
Config via env. Expõe agregados. Opcionalmente persiste via EpisodeStore.

Métricas enriquecidas (v2):
- Timestamps UTC (ISO) + monotonic para duração precisa
- spread_bps_proxy, slippage_proxy_bps, expected_net_edge_bps_at_entry
- edge_auc_bps_ms (integral trapezoidal do edge ao longo do episódio)
- expected_net_edge_bps_after_latency (edge medido após EXEC_LATENCY_MS)
- p50/p90/p99 de duração e AUC
- episodes_per_hour
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Config via env (lidos uma vez no import) ---
EDGE_ENTER_BPS: float = float(os.environ.get("EDGE_ENTER_BPS", "30"))
EDGE_EXIT_BPS: float = float(os.environ.get("EDGE_EXIT_BPS", "10"))
MAX_EPISODE_SECONDS: float = float(os.environ.get("MAX_EPISODE_SECONDS", "300"))
LAT_MS: float = float(os.environ.get("LAT_MS", "500"))
ORDER_SIZE: float = float(os.environ.get("ORDER_SIZE", "10"))
FEE_BPS: float = float(os.environ.get("FEE_BPS", "0"))

EXEC_LATENCY_MS: float = float(os.environ.get("EXEC_LATENCY_MS", "500"))
SLIPPAGE_BPS_BASE: float = float(os.environ.get("SLIPPAGE_BPS_BASE", "0"))
SLIPPAGE_BPS_PER_SPREAD: float = float(os.environ.get("SLIPPAGE_BPS_PER_SPREAD", "0"))
MIN_NET_EDGE_BPS: float = float(os.environ.get("MIN_NET_EDGE_BPS", "0"))


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
    start_time_utc: Optional[str] = None
    end_time_utc: Optional[str] = None
    start_monotonic: Optional[float] = None
    end_monotonic: Optional[float] = None
    spread_bps_proxy: Optional[float] = None
    slippage_proxy_bps: Optional[float] = None
    expected_net_edge_bps_at_entry: Optional[float] = None
    expected_net_edge_bps_after_latency: Optional[float] = None
    edge_auc_bps_ms: Optional[float] = None


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
    start_time_utc: str = ""
    start_monotonic: float = 0.0
    spread_bps_proxy: Optional[float] = None
    slippage_proxy_bps: Optional[float] = None
    expected_net_edge_bps_at_entry: Optional[float] = None
    edge_samples: List[Tuple[float, float]] = field(default_factory=list)
    edge_after_latency: Optional[float] = None


def _compute_auc(samples: List[Tuple[float, float]]) -> Optional[float]:
    """Trapezoidal AUC of edge_bps over monotonic time (result in bps*ms)."""
    if len(samples) < 2:
        return None
    total = 0.0
    for i in range(1, len(samples)):
        t0, e0 = samples[i - 1]
        t1, e1 = samples[i]
        dt_ms = (t1 - t0) * 1000.0
        if dt_ms <= 0:
            continue
        total += 0.5 * (e0 + e1) * dt_ms
    return round(total, 2)


class EdgeEpisodeTracker:
    """
    Detecta episódios de edge (entrada/saída) e registra estatísticas.
    Thread-safe. Opcionalmente persiste via on_close callback.
    """

    def __init__(
        self,
        edge_enter_bps: float = EDGE_ENTER_BPS,
        edge_exit_bps: float = EDGE_EXIT_BPS,
        max_episode_seconds: float = MAX_EPISODE_SECONDS,
        order_size: float = ORDER_SIZE,
        fee_bps: float = FEE_BPS,
        exec_latency_ms: float = EXEC_LATENCY_MS,
        slippage_bps_base: float = SLIPPAGE_BPS_BASE,
        slippage_bps_per_spread: float = SLIPPAGE_BPS_PER_SPREAD,
        min_net_edge_bps: float = MIN_NET_EDGE_BPS,
        on_close: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._edge_enter_bps = edge_enter_bps
        self._edge_exit_bps = edge_exit_bps
        self._max_episode_seconds = max_episode_seconds
        self._order_size = order_size
        self._fee_bps = fee_bps
        self._exec_latency_ms = exec_latency_ms
        self._slippage_bps_base = slippage_bps_base
        self._slippage_bps_per_spread = slippage_bps_per_spread
        self._min_net_edge_bps = min_net_edge_bps
        self._on_close = on_close
        self._lock = threading.RLock()
        self._open: Dict[str, OpenEpisode] = {}
        self._closed: List[EdgeEpisode] = []
        self._max_closed: int = 5000
        self._tracker_start_mono: float = time.monotonic()
        self._tracker_start_time: float = time.time()

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

        mono_now = time.monotonic()
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
            self._maybe_close(token_id, ts, mid, dropped_reason, mono_now, edge_bps)
            return

        slippage_bps = 0.0
        expected_net_edge_bps = edge_bps - slippage_bps - self._fee_bps

        spread_bps_proxy = (spread / mid * 10000.0) if mid > 0 else None

        with self._lock:
            if token_id in self._open:
                ep = self._open[token_id]
                ep.edge_samples.append((mono_now, edge_bps))
                if len(ep.edge_samples) > 2000:
                    ep.edge_samples = ep.edge_samples[-2000:]

                if ep.edge_after_latency is None:
                    elapsed_ms = (mono_now - ep.start_monotonic) * 1000.0
                    if elapsed_ms >= self._exec_latency_ms:
                        ep.edge_after_latency = expected_net_edge_bps

                if (
                    expected_net_edge_bps <= self._edge_exit_bps
                    or (ts - ep.ts_start) >= self._max_episode_seconds
                ):
                    self._close_episode(ep, ts, mid, None, mono_now)
                    del self._open[token_id]
            elif expected_net_edge_bps >= self._edge_enter_bps:
                slip_proxy = (
                    self._slippage_bps_base + self._slippage_bps_per_spread * spread_bps_proxy
                    if spread_bps_proxy is not None
                    else None
                )
                net_at_entry = (
                    edge_bps - self._fee_bps - slip_proxy
                    if slip_proxy is not None
                    else None
                )
                now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                ep = OpenEpisode(
                    token_id=token_id,
                    ts_start=ts,
                    side=side,
                    mid_start=mid,
                    p_oracle_start=p_oracle,
                    spread_start=spread,
                    edge_bps_start=edge_bps,
                    expected_net_edge_bps_start=expected_net_edge_bps,
                    capacity_est=capacity_est,
                    start_time_utc=now_utc,
                    start_monotonic=mono_now,
                    spread_bps_proxy=spread_bps_proxy,
                    slippage_proxy_bps=slip_proxy,
                    expected_net_edge_bps_at_entry=net_at_entry,
                    edge_samples=[(mono_now, edge_bps)],
                )
                self._open[token_id] = ep

    def _close_episode(
        self,
        ep: OpenEpisode,
        ts: float,
        mid: float,
        dropped_reason: Optional[str],
        mono_now: float,
    ) -> None:
        """Build closed episode record and persist (lock must be held)."""
        duration_ms = (mono_now - ep.start_monotonic) * 1000.0
        end_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        auc = _compute_auc(ep.edge_samples)

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
            start_time_utc=ep.start_time_utc,
            end_time_utc=end_utc,
            start_monotonic=ep.start_monotonic,
            end_monotonic=mono_now,
            spread_bps_proxy=ep.spread_bps_proxy,
            slippage_proxy_bps=ep.slippage_proxy_bps,
            expected_net_edge_bps_at_entry=ep.expected_net_edge_bps_at_entry,
            expected_net_edge_bps_after_latency=ep.edge_after_latency,
            edge_auc_bps_ms=auc,
        )
        self._closed.append(closed_ep)
        if len(self._closed) > self._max_closed:
            self._closed = self._closed[-self._max_closed:]

        if self._on_close is not None:
            try:
                self._on_close(self._episode_to_dict(closed_ep))
            except Exception as e:
                logger.debug("[EDGE_TRACKER] on_close callback failed: %s", e)

    @staticmethod
    def _episode_to_dict(ep: EdgeEpisode) -> Dict[str, Any]:
        return {
            "token_id": ep.token_id,
            "ts_start": ep.ts_start,
            "ts_end": ep.ts_end,
            "duration_ms": round(ep.duration_ms, 2),
            "side": ep.side,
            "mid_start": ep.mid_start,
            "mid_end": ep.mid_end,
            "p_oracle_start": ep.p_oracle_start,
            "spread_start": ep.spread_start,
            "edge_bps_start": round(ep.edge_bps_start, 2),
            "expected_net_edge_bps_start": round(ep.expected_net_edge_bps_start, 2),
            "capacity_est": ep.capacity_est,
            "dropped_reason": ep.dropped_reason,
            "start_time_utc": ep.start_time_utc,
            "end_time_utc": ep.end_time_utc,
            "start_monotonic": ep.start_monotonic,
            "end_monotonic": ep.end_monotonic,
            "spread_bps_proxy": round(ep.spread_bps_proxy, 2) if ep.spread_bps_proxy is not None else None,
            "slippage_proxy_bps": round(ep.slippage_proxy_bps, 2) if ep.slippage_proxy_bps is not None else None,
            "expected_net_edge_bps_at_entry": round(ep.expected_net_edge_bps_at_entry, 2) if ep.expected_net_edge_bps_at_entry is not None else None,
            "expected_net_edge_bps_after_latency": round(ep.expected_net_edge_bps_after_latency, 2) if ep.expected_net_edge_bps_after_latency is not None else None,
            "edge_auc_bps_ms": ep.edge_auc_bps_ms,
        }

    def _maybe_close(
        self,
        token_id: str,
        ts: float,
        mid: float,
        dropped_reason: Optional[str],
        mono_now: float,
        edge_bps: float = 0.0,
    ) -> None:
        with self._lock:
            if token_id not in self._open:
                return
            ep = self._open[token_id]
            ep.edge_samples.append((mono_now, edge_bps))
            self._close_episode(ep, ts, mid, dropped_reason, mono_now)
            del self._open[token_id]

    def get_aggregates(self) -> Dict[str, Any]:
        with self._lock:
            closed = list(self._closed)
            open_count = len(self._open)

        n = len(closed)
        elapsed_hours = (time.monotonic() - self._tracker_start_mono) / 3600.0

        empty: Dict[str, Any] = {
            "edge_episodes_total": open_count,
            "edge_episodes_open": open_count,
            "edge_episodes_closed": 0,
            "edge_duration_ms_p50": None,
            "edge_duration_ms_p90": None,
            "edge_duration_ms_p99": None,
            "expected_net_edge_bps_p50": None,
            "expected_net_edge_bps_p90": None,
            "episodes_survive_500ms": 0.0,
            "episodes_survive_2000ms": 0.0,
            "episodes_survive_5000ms": 0.0,
            "episodes_per_hour": 0.0,
            "top_5_markets_by_ev": [],
            "edge_auc_bps_ms_p50": None,
            "edge_auc_bps_ms_p90": None,
            "top_5_markets_by_edge_auc": [],
            "exec_latency_ms": self._exec_latency_ms,
            "fee_bps": self._fee_bps,
            "slippage_bps_base": self._slippage_bps_base,
            "slippage_bps_per_spread": self._slippage_bps_per_spread,
        }
        if not closed:
            return empty

        durations = [e.duration_ms for e in closed]
        edges = [e.expected_net_edge_bps_start for e in closed]
        aucs = [e.edge_auc_bps_ms for e in closed if e.edge_auc_bps_ms is not None]
        durations_sorted = sorted(durations)
        edges_sorted = sorted(edges)
        aucs_sorted = sorted(aucs)

        def _p(arr: List[float], p: float) -> Optional[float]:
            if not arr:
                return None
            sz = len(arr)
            idx = min(sz - 1, max(0, int(sz * p / 100.0)))
            return round(arr[idx], 2)

        survive_500 = sum(1 for d in durations if d >= 500) / n
        survive_2000 = sum(1 for d in durations if d >= 2000) / n
        survive_5000 = sum(1 for d in durations if d >= 5000) / n
        eps_per_hour = round(n / elapsed_hours, 2) if elapsed_hours > 0 else 0.0

        ev_by_token: Dict[str, float] = {}
        auc_by_token: Dict[str, float] = {}
        for e in closed:
            if e.dropped_reason:
                continue
            ev = e.expected_net_edge_bps_start * (e.duration_ms / 1000.0)
            ev_by_token[e.token_id] = ev_by_token.get(e.token_id, 0) + ev
            if e.edge_auc_bps_ms is not None:
                auc_by_token[e.token_id] = auc_by_token.get(e.token_id, 0) + e.edge_auc_bps_ms

        top_5_ev = sorted(
            [{"token_id": k, "ev": round(v, 2)} for k, v in ev_by_token.items()],
            key=lambda x: -x["ev"],
        )[:5]
        top_5_auc = sorted(
            [{"token_id": k, "total_auc": round(v, 2)} for k, v in auc_by_token.items()],
            key=lambda x: -x["total_auc"],
        )[:5]

        return {
            "edge_episodes_total": n + open_count,
            "edge_episodes_open": open_count,
            "edge_episodes_closed": n,
            "edge_duration_ms_p50": _p(durations_sorted, 50),
            "edge_duration_ms_p90": _p(durations_sorted, 90),
            "edge_duration_ms_p99": _p(durations_sorted, 99),
            "expected_net_edge_bps_p50": _p(edges_sorted, 50),
            "expected_net_edge_bps_p90": _p(edges_sorted, 90),
            "episodes_survive_500ms": round(survive_500, 4),
            "episodes_survive_2000ms": round(survive_2000, 4),
            "episodes_survive_5000ms": round(survive_5000, 4),
            "episodes_per_hour": eps_per_hour,
            "top_5_markets_by_ev": top_5_ev,
            "edge_auc_bps_ms_p50": _p(aucs_sorted, 50),
            "edge_auc_bps_ms_p90": _p(aucs_sorted, 90),
            "top_5_markets_by_edge_auc": top_5_auc,
            "exec_latency_ms": self._exec_latency_ms,
            "fee_bps": self._fee_bps,
            "slippage_bps_base": self._slippage_bps_base,
            "slippage_bps_per_spread": self._slippage_bps_per_spread,
        }
