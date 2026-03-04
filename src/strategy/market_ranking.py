"""
Market Profitability Ranking
============================

Pure-function module that scores markets by estimated real arbitrage profitability.
Uses only data already collected by EdgeEpisodeTracker (closed episodes).

Score model:
    profitability_score =
        opportunity_density × latency_survival × edge_strength × log(1 + edge_intensity)

All inputs are per-market aggregations of EdgeEpisode fields.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MarketStats:
    token_id: str
    episode_count: int
    durations_ms: List[float]
    edge_after_latency_values: List[float]
    auc_values: List[float]


def compute_market_ranking(
    episodes: List[Any],
    elapsed_hours: float,
    exec_latency_ms: float,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Rank markets by profitability score. O(n) over episodes."""
    if not episodes or elapsed_hours <= 0:
        return []

    by_market: Dict[str, MarketStats] = {}
    for ep in episodes:
        tid = ep.token_id
        if tid not in by_market:
            by_market[tid] = MarketStats(
                token_id=tid,
                episode_count=0,
                durations_ms=[],
                edge_after_latency_values=[],
                auc_values=[],
            )
        ms = by_market[tid]
        ms.episode_count += 1
        ms.durations_ms.append(ep.duration_ms)
        if ep.expected_net_edge_bps_after_latency is not None:
            ms.edge_after_latency_values.append(ep.expected_net_edge_bps_after_latency)
        if ep.edge_auc_bps_ms is not None:
            ms.auc_values.append(ep.edge_auc_bps_ms)

    scored: List[Dict[str, Any]] = []
    for ms in by_market.values():
        if not ms.edge_after_latency_values or not ms.auc_values:
            continue

        opportunity_density = ms.episode_count / elapsed_hours

        survive_count = sum(1 for d in ms.durations_ms if d >= exec_latency_ms)
        latency_survival = survive_count / len(ms.durations_ms)
        if latency_survival <= 0:
            continue

        edge_strength = sum(ms.edge_after_latency_values) / len(ms.edge_after_latency_values)
        if edge_strength <= 0:
            continue

        edge_intensity = sum(ms.auc_values) / len(ms.auc_values)
        if edge_intensity < 0:
            edge_intensity = 0.0

        raw_score = (
            opportunity_density
            * latency_survival
            * edge_strength
            * math.log(1.0 + edge_intensity)
        )

        scored.append({
            "token_id": ms.token_id,
            "profitability_score_raw": raw_score,
            "episodes_per_hour": round(opportunity_density, 2),
            "latency_survival": round(latency_survival, 4),
            "mean_edge_after_latency": round(edge_strength, 2),
            "edge_auc": round(edge_intensity, 2),
            "episode_count": ms.episode_count,
        })

    if not scored:
        return []

    scored.sort(key=lambda x: -x["profitability_score_raw"])

    max_raw = scored[0]["profitability_score_raw"]
    for entry in scored:
        entry["profitability_score"] = (
            round(entry["profitability_score_raw"] / max_raw, 4) if max_raw > 0 else 0.0
        )
        del entry["profitability_score_raw"]

    return scored[:top_n]
