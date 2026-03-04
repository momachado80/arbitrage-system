"""Tests for market profitability ranking."""

from src.strategy.market_ranking import compute_market_ranking, MarketStats
from src.strategy.edge_episode_tracker import EdgeEpisode


def _make_episode(
    token_id: str,
    duration_ms: float = 1000.0,
    edge_after_latency: float = 20.0,
    auc: float = 500.0,
    dropped: str = None,
) -> EdgeEpisode:
    return EdgeEpisode(
        token_id=token_id,
        ts_start=0.0,
        ts_end=duration_ms / 1000.0,
        duration_ms=duration_ms,
        side="BUY",
        mid_start=0.50,
        mid_end=0.48,
        p_oracle_start=0.55,
        spread_start=0.02,
        edge_bps_start=50.0,
        expected_net_edge_bps_start=50.0,
        capacity_est=100,
        dropped_reason=dropped,
        expected_net_edge_bps_after_latency=edge_after_latency,
        edge_auc_bps_ms=auc,
    )


def test_empty_episodes_returns_empty():
    assert compute_market_ranking([], elapsed_hours=1.0, exec_latency_ms=500) == []


def test_zero_elapsed_hours_returns_empty():
    eps = [_make_episode("tok1")]
    assert compute_market_ranking(eps, elapsed_hours=0, exec_latency_ms=500) == []


def test_ranking_sorted_descending():
    episodes = [
        _make_episode("tokA", duration_ms=2000, edge_after_latency=30.0, auc=800.0),
        _make_episode("tokA", duration_ms=1500, edge_after_latency=25.0, auc=700.0),
        _make_episode("tokB", duration_ms=600, edge_after_latency=10.0, auc=200.0),
        _make_episode("tokC", duration_ms=3000, edge_after_latency=40.0, auc=1200.0),
        _make_episode("tokC", duration_ms=2500, edge_after_latency=35.0, auc=1000.0),
        _make_episode("tokC", duration_ms=1800, edge_after_latency=28.0, auc=900.0),
    ]
    ranking = compute_market_ranking(episodes, elapsed_hours=1.0, exec_latency_ms=500)
    scores = [r["profitability_score"] for r in ranking]
    for i in range(1, len(scores)):
        assert scores[i] <= scores[i - 1], f"Not sorted: {scores}"


def test_scores_normalized_0_to_1():
    episodes = [
        _make_episode("tokA", duration_ms=2000, edge_after_latency=30.0, auc=800.0),
        _make_episode("tokB", duration_ms=1000, edge_after_latency=15.0, auc=300.0),
        _make_episode("tokC", duration_ms=3000, edge_after_latency=40.0, auc=1200.0),
    ]
    ranking = compute_market_ranking(episodes, elapsed_hours=1.0, exec_latency_ms=500)
    for r in ranking:
        assert 0.0 <= r["profitability_score"] <= 1.0, (
            f"Score {r['profitability_score']} out of [0,1]"
        )
    assert ranking[0]["profitability_score"] == 1.0


def test_top_market_has_score_1():
    episodes = [
        _make_episode("tokA", duration_ms=2000, edge_after_latency=30.0, auc=800.0),
        _make_episode("tokB", duration_ms=600, edge_after_latency=10.0, auc=200.0),
    ]
    ranking = compute_market_ranking(episodes, elapsed_hours=1.0, exec_latency_ms=500)
    assert ranking[0]["profitability_score"] == 1.0


def test_missing_latency_data_skips_market():
    episodes = [
        _make_episode("tokA", duration_ms=2000, edge_after_latency=30.0, auc=800.0),
        EdgeEpisode(
            token_id="tokB",
            ts_start=0, ts_end=1, duration_ms=1000,
            side="BUY", mid_start=0.5, mid_end=0.48,
            p_oracle_start=0.55, spread_start=0.02,
            edge_bps_start=50, expected_net_edge_bps_start=50,
            capacity_est=100,
            expected_net_edge_bps_after_latency=None,
            edge_auc_bps_ms=None,
        ),
    ]
    ranking = compute_market_ranking(episodes, elapsed_hours=1.0, exec_latency_ms=500)
    token_ids = [r["token_id"] for r in ranking]
    assert "tokB" not in token_ids


def test_negative_edge_skips_market():
    episodes = [
        _make_episode("tokA", duration_ms=2000, edge_after_latency=30.0, auc=800.0),
        _make_episode("tokB", duration_ms=1000, edge_after_latency=-5.0, auc=100.0),
    ]
    ranking = compute_market_ranking(episodes, elapsed_hours=1.0, exec_latency_ms=500)
    token_ids = [r["token_id"] for r in ranking]
    assert "tokB" not in token_ids


def test_top_n_limit():
    episodes = [_make_episode(f"tok_{i}", duration_ms=2000, auc=float(i * 100)) for i in range(20)]
    ranking = compute_market_ranking(episodes, elapsed_hours=1.0, exec_latency_ms=500, top_n=5)
    assert len(ranking) <= 5


def test_output_fields_present():
    episodes = [_make_episode("tokA", duration_ms=2000, edge_after_latency=30.0, auc=800.0)]
    ranking = compute_market_ranking(episodes, elapsed_hours=1.0, exec_latency_ms=500)
    assert len(ranking) == 1
    entry = ranking[0]
    expected_fields = {
        "token_id", "profitability_score", "episodes_per_hour",
        "latency_survival", "mean_edge_after_latency", "edge_auc", "episode_count",
    }
    assert expected_fields.issubset(set(entry.keys()))


def test_all_episodes_too_short_for_latency():
    episodes = [
        _make_episode("tokA", duration_ms=100, edge_after_latency=30.0, auc=800.0),
        _make_episode("tokA", duration_ms=200, edge_after_latency=25.0, auc=600.0),
    ]
    ranking = compute_market_ranking(episodes, elapsed_hours=1.0, exec_latency_ms=500)
    assert len(ranking) == 0
