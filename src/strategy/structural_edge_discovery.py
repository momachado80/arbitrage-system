"""
Structural Edge Discovery
=========================

Pure-function module that detects markets exhibiting persistent, executable,
net-positive edge across multiple time windows.

No side effects. All I/O (flagged.json) is handled by the caller (tracker).
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


def _parse_utc(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    n = len(sorted_vals)
    if n == 0:
        return None
    idx = min(n - 1, max(0, int(n * p / 100.0)))
    return sorted_vals[idx]


def compute_structural_edge_candidates(
    closed_episode_summaries: List[Dict[str, Any]],
    exec_latency_ms: float,
    window_minutes: int = 60,
    lookback_windows: int = 3,
    min_episodes_per_window: int = 20,
    min_survival_at_latency: float = 0.40,
    min_median_net_edge_bps: float = 5.0,
    persistent_windows_required: int = 2,
    spike_multiplier: float = 4.0,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    if not closed_episode_summaries:
        return {"structural_edge_candidates": []}

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    window_td = timedelta(minutes=window_minutes)
    boundaries = []
    for i in range(lookback_windows):
        w_end = now_utc - (window_td * i)
        w_start = w_end - window_td
        boundaries.append((w_start, w_end))

    oldest = boundaries[-1][0] if boundaries else now_utc

    # Assign episodes to (market_key, window_index) buckets — O(n)
    Bucket = Dict[str, List[Dict[str, Any]]]
    windows: List[Bucket] = [{} for _ in range(lookback_windows)]

    for ep in closed_episode_summaries:
        end_dt = _parse_utc(ep.get("end_time_utc"))
        if end_dt is None or end_dt < oldest:
            continue
        token_id = ep.get("token_id", "")
        slug = ep.get("market_slug")
        market_key = slug if slug else token_id
        if not market_key:
            continue
        for wi, (w_start, w_end) in enumerate(boundaries):
            if w_start <= end_dt < w_end:
                windows[wi].setdefault(market_key, []).append(ep)
                break

    all_markets = set()
    for bucket in windows:
        all_markets.update(bucket.keys())

    if not all_markets:
        return {"structural_edge_candidates": []}

    hours_per_window = window_minutes / 60.0

    # Per-market per-window stats
    def _window_stats(episodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        n = len(episodes)
        if n == 0:
            return None
        survive = sum(
            1 for e in episodes if e.get("duration_ms", 0) >= exec_latency_ms
        )
        survival = survive / n

        net_edges = sorted(
            e["expected_net_edge_bps_after_latency"]
            for e in episodes
            if e.get("expected_net_edge_bps_after_latency") is not None
        )
        median_edge = _percentile(net_edges, 50)
        p90_edge = _percentile(net_edges, 90)
        eps_per_hour = n / hours_per_window if hours_per_window > 0 else 0.0

        return {
            "episodes_closed": n,
            "survival_at_latency": survival,
            "median_net_edge_after_latency_bps": median_edge,
            "p90_net_edge_after_latency_bps": p90_edge,
            "episodes_per_hour": eps_per_hour,
        }

    def _passes_criteria(st: Dict[str, Any]) -> bool:
        if st["episodes_closed"] < min_episodes_per_window:
            return False
        if st["survival_at_latency"] < min_survival_at_latency:
            return False
        med = st["median_net_edge_after_latency_bps"]
        if med is None or med < min_median_net_edge_bps:
            return False
        p90 = st["p90_net_edge_after_latency_bps"]
        if med > 0 and p90 is not None and p90 > med * spike_multiplier:
            return False
        return True

    candidates: List[Dict[str, Any]] = []

    for mk in all_markets:
        stats_per_window: List[Optional[Dict[str, Any]]] = []
        for wi in range(lookback_windows):
            eps_in_window = windows[wi].get(mk, [])
            stats_per_window.append(_window_stats(eps_in_window))

        windows_passed = sum(
            1 for st in stats_per_window if st is not None and _passes_criteria(st)
        )
        if windows_passed < persistent_windows_required:
            continue

        last_w = stats_per_window[0]
        if last_w is None or not _passes_criteria(last_w):
            continue

        sample_ep = None
        for ep in (windows[0].get(mk, [])):
            sample_ep = ep
            break

        token_id = sample_ep.get("token_id", mk) if sample_ep else mk
        slug = sample_ep.get("market_slug") if sample_ep else None

        notes: List[str] = []
        if windows_passed >= lookback_windows:
            notes.append("persistent")
        p90_val = last_w["p90_net_edge_after_latency_bps"]
        med_val = last_w["median_net_edge_after_latency_bps"]
        if p90_val is not None and med_val is not None and med_val > 0:
            if p90_val <= med_val * 2.0:
                notes.append("stable")

        score_raw = (
            last_w["episodes_per_hour"]
            * last_w["survival_at_latency"]
            * max(0.0, med_val if med_val is not None else 0.0)
        )

        candidates.append({
            "market_key": mk,
            "market_slug": slug,
            "token_id": token_id,
            "score_raw": score_raw,
            "windows_passed": windows_passed,
            "episodes_last_window": last_w["episodes_closed"],
            "episodes_per_hour_last_window": round(last_w["episodes_per_hour"], 2),
            "survival_at_latency_last_window": round(last_w["survival_at_latency"], 4),
            "median_net_edge_after_latency_bps_last_window": (
                round(med_val, 2) if med_val is not None else None
            ),
            "p90_net_edge_after_latency_bps_last_window": (
                round(p90_val, 2) if p90_val is not None else None
            ),
            "notes": notes,
        })

    if not candidates:
        return {"structural_edge_candidates": []}

    candidates.sort(key=lambda x: -x["score_raw"])
    max_raw = candidates[0]["score_raw"]
    for c in candidates:
        c["score"] = round(c["score_raw"] / max_raw, 4) if max_raw > 0 else 0.0
        del c["score_raw"]

    return {"structural_edge_candidates": candidates}
