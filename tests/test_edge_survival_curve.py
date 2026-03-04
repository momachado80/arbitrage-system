"""
Tests for EdgeEpisodeTracker.compute_edge_survival_curve().

Uses direct injection of EdgeEpisode objects to control duration_ms
precisely, since the tracker measures duration via time.monotonic().
"""

from src.strategy.edge_episode_tracker import EdgeEpisode, EdgeEpisodeTracker


def _inject_episodes(tracker, durations_ms):
    """Inject closed episodes with exact durations into the tracker."""
    for i, dur in enumerate(durations_ms):
        ep = EdgeEpisode(
            token_id=f"tok{i}", ts_start=1000.0, ts_end=1000.0 + dur / 1000.0,
            duration_ms=dur, side="BUY", mid_start=0.5, mid_end=0.5,
            p_oracle_start=0.55, spread_start=0.01, edge_bps_start=50.0,
            expected_net_edge_bps_start=50.0, capacity_est=100.0,
        )
        tracker._closed.append(ep)


def test_empty_returns_empty_dict():
    t = EdgeEpisodeTracker()
    curve = t.compute_edge_survival_curve()
    assert curve == {}


def test_values_between_0_and_1():
    t = EdgeEpisodeTracker()
    _inject_episodes(t, [100, 500, 1000, 3000, 8000])
    curve = t.compute_edge_survival_curve()
    assert len(curve) > 0
    for key, val in curve.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


def test_monotonic_decreasing():
    t = EdgeEpisodeTracker()
    _inject_episodes(t, [30, 80, 200, 600, 1500, 4000, 12000, 35000])
    curve = t.compute_edge_survival_curve()
    values = list(curve.values())
    for i in range(1, len(values)):
        assert values[i] <= values[i - 1], (
            f"Not monotonic: {list(curve.keys())[i-1]}={values[i-1]} > "
            f"{list(curve.keys())[i]}={values[i]}"
        )


def test_all_long_episodes():
    t = EdgeEpisodeTracker()
    _inject_episodes(t, [50000, 60000, 70000])
    curve = t.compute_edge_survival_curve()
    for val in curve.values():
        assert val == 1.0


def test_all_short_episodes():
    t = EdgeEpisodeTracker()
    _inject_episodes(t, [10, 20, 30])
    curve = t.compute_edge_survival_curve()
    for val in curve.values():
        assert val == 0.0
