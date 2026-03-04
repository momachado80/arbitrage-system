"""
Execution Simulator — Gold-standard replay engine
===================================================

Deterministic given seed + input.  Replays recorded snapshots against
episode signals to simulate taker/maker fills with realistic latency,
slippage, fees, and maker-queue models.

Pure function: no globals, no side effects, no background tasks.
"""

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config (read once per sim run, passed via SimConfig)
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    lat_signal_to_order_ms: float = 50.0
    lat_order_to_exchange_ms: float = 150.0
    lat_exchange_to_ack_ms: float = 50.0
    fee_taker_bps: float = 0.0
    fee_maker_bps: float = 0.0
    slippage_from_spread_mult: float = 0.5
    slippage_bps_floor: float = 2.0
    maker_fill_prob_per_second: float = 0.15
    sim_order_size: float = 10.0
    seed: int = 42

    @property
    def total_latency_ms(self) -> float:
        return (
            self.lat_signal_to_order_ms
            + self.lat_order_to_exchange_ms
            + self.lat_exchange_to_ack_ms
        )

    @staticmethod
    def from_env(**overrides) -> "SimConfig":
        def _f(key: str, default: float) -> float:
            return float(overrides.get(key, os.environ.get(key.upper(), str(default))))
        def _i(key: str, default: int) -> int:
            return int(overrides.get(key, os.environ.get(key.upper(), str(default))))
        return SimConfig(
            lat_signal_to_order_ms=_f("lat_signal_to_order_ms", 50),
            lat_order_to_exchange_ms=_f("lat_order_to_exchange_ms", 150),
            lat_exchange_to_ack_ms=_f("lat_exchange_to_ack_ms", 50),
            fee_taker_bps=_f("fee_taker_bps", 0),
            fee_maker_bps=_f("fee_maker_bps", 0),
            slippage_from_spread_mult=_f("slippage_from_spread_mult", 0.5),
            slippage_bps_floor=_f("slippage_bps_floor", 2.0),
            maker_fill_prob_per_second=_f("maker_fill_prob_per_second", 0.15),
            sim_order_size=_f("sim_order_size", 10),
            seed=_i("seed", 42),
        )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SimTrade:
    trade_id: str
    market_key: str
    side: str
    style: str  # "taker" or "maker"
    detect_time_utc: str
    place_time_utc: str
    ack_time_utc: str
    fill_time_utc: Optional[str] = None
    fill_qty: float = 0.0
    avg_fill_px: float = 0.0
    fees_bps: float = 0.0
    slippage_bps: float = 0.0
    pnl_bps: float = 0.0
    alpha_decay_loss_bps: float = 0.0
    episode_id: Optional[str] = None
    dropped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "market_key": self.market_key,
            "side": self.side,
            "style": self.style,
            "detect_time_utc": self.detect_time_utc,
            "place_time_utc": self.place_time_utc,
            "ack_time_utc": self.ack_time_utc,
            "fill_time_utc": self.fill_time_utc,
            "fill_qty": round(self.fill_qty, 4),
            "avg_fill_px": round(self.avg_fill_px, 6),
            "fees_bps": round(self.fees_bps, 2),
            "slippage_bps": round(self.slippage_bps, 2),
            "pnl_bps": round(self.pnl_bps, 2),
            "alpha_decay_loss_bps": round(self.alpha_decay_loss_bps, 2),
            "episode_id": self.episode_id,
            "dropped_reason": self.dropped_reason,
        }


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

def _ts_to_utc_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _find_snapshot_at(snapshots_by_market: Dict[str, list], market_key: str, target_ts: float) -> Optional[dict]:
    """Find the snapshot closest to (but not after) target_ts for a market."""
    snaps = snapshots_by_market.get(market_key, [])
    if not snaps:
        return None
    best = None
    for s in snaps:
        if s["ts_utc"] <= target_ts:
            best = s
        else:
            break
    return best


def _compute_slippage_bps(snap: dict, cfg: SimConfig) -> float:
    spread_bps = snap.get("spread_bps", 0.0) or 0.0
    return max(spread_bps * cfg.slippage_from_spread_mult, cfg.slippage_bps_floor)


def simulate(
    episodes: List[Dict[str, Any]],
    snapshots: List[Dict[str, Any]],
    cfg: SimConfig,
    style: str = "both",
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    market_key_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run deterministic execution simulation.

    Returns dict with 'trades' list and 'summary' statistics.
    """
    rng = random.Random(cfg.seed)

    # Index snapshots by market, sorted by time
    snaps_by_market: Dict[str, List[dict]] = {}
    for s in snapshots:
        mk = s.get("token_id", "")
        if market_key_filter and mk != market_key_filter:
            continue
        snaps_by_market.setdefault(mk, []).append(s)
    for mk in snaps_by_market:
        snaps_by_market[mk].sort(key=lambda x: x.get("ts_utc", 0))

    # Filter and sort episodes by start time
    filtered_eps = []
    for ep in episodes:
        ts_start = ep.get("ts_start", 0)
        if from_ts and ts_start < from_ts:
            continue
        if to_ts and ts_start > to_ts:
            continue
        mk = ep.get("token_id", "")
        if market_key_filter and mk != market_key_filter:
            continue
        filtered_eps.append(ep)
    filtered_eps.sort(key=lambda x: x.get("ts_start", 0))

    styles_to_run = []
    if style in ("taker", "both"):
        styles_to_run.append("taker")
    if style in ("maker", "both"):
        styles_to_run.append("maker")

    trades: List[SimTrade] = []
    trade_counter = 0
    total_latency_s = cfg.total_latency_ms / 1000.0

    for ep in filtered_eps:
        net_edge = ep.get("expected_net_edge_bps_after_latency")
        if net_edge is None:
            net_edge = ep.get("expected_net_edge_bps_at_entry")
        if net_edge is None or net_edge <= 0:
            for sty in styles_to_run:
                trade_counter += 1
                trades.append(SimTrade(
                    trade_id=f"sim_{trade_counter:06d}",
                    market_key=ep.get("token_id", ""),
                    side=ep.get("side", "BUY"),
                    style=sty,
                    detect_time_utc=ep.get("start_time_utc", ""),
                    place_time_utc="",
                    ack_time_utc="",
                    dropped_reason="non_positive_edge",
                    episode_id=ep.get("token_id", "") + "_" + str(ep.get("ts_start", "")),
                ))
            continue

        detect_ts = ep.get("ts_start", 0.0)
        place_ts = detect_ts + cfg.lat_signal_to_order_ms / 1000.0
        ack_ts = place_ts + cfg.lat_order_to_exchange_ms / 1000.0 + cfg.lat_exchange_to_ack_ms / 1000.0
        mk = ep.get("token_id", "")
        side = ep.get("side", "BUY")

        snap_at_place = _find_snapshot_at(snaps_by_market, mk, place_ts)
        snap_at_detect = _find_snapshot_at(snaps_by_market, mk, detect_ts)

        for sty in styles_to_run:
            trade_counter += 1
            tid = f"sim_{trade_counter:06d}"
            ep_id = mk + "_" + str(detect_ts)

            if snap_at_place is None:
                trades.append(SimTrade(
                    trade_id=tid, market_key=mk, side=side, style=sty,
                    detect_time_utc=_ts_to_utc_str(detect_ts) if detect_ts > 0 else "",
                    place_time_utc=_ts_to_utc_str(place_ts) if place_ts > 0 else "",
                    ack_time_utc=_ts_to_utc_str(ack_ts) if ack_ts > 0 else "",
                    dropped_reason="no_snapshot_at_placement",
                    episode_id=ep_id,
                ))
                continue

            slippage = _compute_slippage_bps(snap_at_place, cfg)
            mid_at_detect = snap_at_detect["mid_px"] if snap_at_detect else snap_at_place["mid_px"]
            mid_at_place = snap_at_place["mid_px"]
            alpha_decay = abs(mid_at_detect - mid_at_place) * 10000.0 if mid_at_detect > 0 else 0.0

            if sty == "taker":
                fee = cfg.fee_taker_bps
                if side == "BUY":
                    fill_px = snap_at_place.get("best_ask_px", mid_at_place)
                else:
                    fill_px = snap_at_place.get("best_bid_px", mid_at_place)
                pnl = net_edge - slippage - fee - alpha_decay
                trades.append(SimTrade(
                    trade_id=tid, market_key=mk, side=side, style="taker",
                    detect_time_utc=_ts_to_utc_str(detect_ts) if detect_ts > 0 else "",
                    place_time_utc=_ts_to_utc_str(place_ts) if place_ts > 0 else "",
                    ack_time_utc=_ts_to_utc_str(ack_ts) if ack_ts > 0 else "",
                    fill_time_utc=_ts_to_utc_str(ack_ts) if ack_ts > 0 else "",
                    fill_qty=cfg.sim_order_size,
                    avg_fill_px=fill_px,
                    fees_bps=fee,
                    slippage_bps=slippage,
                    pnl_bps=pnl,
                    alpha_decay_loss_bps=alpha_decay,
                    episode_id=ep_id,
                ))
            else:  # maker
                fee = cfg.fee_maker_bps
                if side == "BUY":
                    queue_depth = snap_at_place.get("best_bid_sz", 0)
                    limit_px = snap_at_place.get("best_bid_px", mid_at_place)
                else:
                    queue_depth = snap_at_place.get("best_ask_sz", 0)
                    limit_px = snap_at_place.get("best_ask_px", mid_at_place)

                ep_duration_ms = ep.get("duration_ms", 0)
                time_at_best_s = max(0, ep_duration_ms / 1000.0 - total_latency_s)

                filled = False
                if queue_depth > 0 and time_at_best_s > 0:
                    prob_no_fill = (1.0 - cfg.maker_fill_prob_per_second) ** time_at_best_s
                    fill_prob = 1.0 - prob_no_fill
                    filled = rng.random() < fill_prob
                    fill_time_offset_s = rng.uniform(0, time_at_best_s) if filled else 0

                if filled:
                    fill_ts = ack_ts + fill_time_offset_s
                    maker_slip = max(0.0, slippage * 0.3)
                    pnl = net_edge - maker_slip - fee - alpha_decay
                    trades.append(SimTrade(
                        trade_id=tid, market_key=mk, side=side, style="maker",
                        detect_time_utc=_ts_to_utc_str(detect_ts) if detect_ts > 0 else "",
                        place_time_utc=_ts_to_utc_str(place_ts) if place_ts > 0 else "",
                        ack_time_utc=_ts_to_utc_str(ack_ts) if ack_ts > 0 else "",
                        fill_time_utc=_ts_to_utc_str(fill_ts),
                        fill_qty=cfg.sim_order_size,
                        avg_fill_px=limit_px,
                        fees_bps=fee,
                        slippage_bps=maker_slip,
                        pnl_bps=pnl,
                        alpha_decay_loss_bps=alpha_decay,
                        episode_id=ep_id,
                    ))
                else:
                    trades.append(SimTrade(
                        trade_id=tid, market_key=mk, side=side, style="maker",
                        detect_time_utc=_ts_to_utc_str(detect_ts) if detect_ts > 0 else "",
                        place_time_utc=_ts_to_utc_str(place_ts) if place_ts > 0 else "",
                        ack_time_utc=_ts_to_utc_str(ack_ts) if ack_ts > 0 else "",
                        dropped_reason="maker_no_fill",
                        episode_id=ep_id,
                    ))

    summary = _build_summary(trades)
    return {"trades": [t.to_dict() for t in trades], "summary": summary}


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    n = len(sorted_vals)
    if n == 0:
        return None
    idx = min(n - 1, max(0, int(n * p / 100.0)))
    return round(sorted_vals[idx], 2)


def _build_summary(trades: List[SimTrade]) -> Dict[str, Any]:
    total = len(trades)
    filled = [t for t in trades if t.fill_time_utc is not None]
    skipped = [t for t in trades if t.dropped_reason is not None]
    taker_filled = [t for t in filled if t.style == "taker"]
    maker_filled = [t for t in filled if t.style == "maker"]

    pnl_vals = sorted(t.pnl_bps for t in filled)
    slip_vals = sorted(t.slippage_bps for t in filled)
    fill_times_ms = []
    for t in filled:
        if t.fill_time_utc and t.detect_time_utc:
            try:
                ft = datetime.fromisoformat(t.fill_time_utc.replace("Z", "+00:00"))
                dt = datetime.fromisoformat(t.detect_time_utc.replace("Z", "+00:00"))
                fill_times_ms.append((ft - dt).total_seconds() * 1000.0)
            except Exception:
                pass
    fill_times_sorted = sorted(fill_times_ms)

    # Top markets by expected pnl/hour proxy
    pnl_by_market: Dict[str, List[float]] = {}
    for t in filled:
        pnl_by_market.setdefault(t.market_key, []).append(t.pnl_bps)
    top_markets = sorted(
        [
            {"market_key": mk, "mean_pnl_bps": round(sum(v) / len(v), 2), "trades": len(v)}
            for mk, v in pnl_by_market.items()
        ],
        key=lambda x: -x["mean_pnl_bps"],
    )[:10]

    return {
        "total_trades": total,
        "filled": len(filled),
        "skipped": len(skipped),
        "fill_rate": round(len(filled) / total, 4) if total > 0 else 0.0,
        "pnl_bps_p50": _percentile(pnl_vals, 50),
        "pnl_bps_p90": _percentile(pnl_vals, 90),
        "pnl_bps_p99": _percentile(pnl_vals, 99),
        "slippage_bps_p50": _percentile(slip_vals, 50),
        "slippage_bps_p90": _percentile(slip_vals, 90),
        "time_to_fill_ms_p50": _percentile(fill_times_sorted, 50),
        "time_to_fill_ms_p90": _percentile(fill_times_sorted, 90),
        "taker_filled": len(taker_filled),
        "maker_filled": len(maker_filled),
        "top_markets_by_pnl": top_markets,
    }


# ---------------------------------------------------------------------------
# JSONL persistence (append-only, non-fatal)
# ---------------------------------------------------------------------------

def _path_safe_under_tmp(path: str) -> bool:
    try:
        real = os.path.realpath(path)
        return real.startswith("/tmp") or real.startswith("/private/tmp")
    except Exception:
        return False


def save_sim_trades_jsonl(trades: List[Dict[str, Any]], state_dir: Optional[str] = None) -> Optional[str]:
    base = state_dir or "/tmp/polymarket_state"
    path = os.path.join(base, "sim_trades.jsonl")
    if not _path_safe_under_tmp(path):
        logger.warning("[SIM] path outside /tmp, skip write: %s", path)
        return None
    try:
        os.makedirs(base, exist_ok=True)
        with open(path, "a") as f:
            for t in trades:
                f.write(json.dumps(t, default=str) + "\n")
        return path
    except Exception as e:
        logger.debug("[SIM] trade write skip (non-fatal): %s", e)
        return None
