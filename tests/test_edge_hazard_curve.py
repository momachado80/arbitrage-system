"""Tests for EdgeEpisodeTracker.compute_edge_hazard_curve()."""

from src.strategy.edge_episode_tracker import EdgeEpisodeTracker, EdgeEpisode


def _make_tracker() -> EdgeEpisodeTracker:
    return EdgeEpisodeTracker(
        edge_enter_bps=10,
        edge_exit_bps=5,
        max_episode_seconds=300,
        order_size=1,
        fee_bps=0,
    )


def _inject_episodes(tracker: EdgeEpisodeTracker, durations_ms: list) -> None:
    for i, d in enumerate(durations_ms):
        ep = EdgeEpisode(
            token_id=f"tok_{i}",
            ts_start=0.0,
            ts_end=d / 1000.0,
            duration_ms=d,
            side="BUY",
            mid_start=0.50,
            mid_end=0.48,
            p_oracle_start=0.55,
            spread_start=0.02,
            edge_bps_start=50.0,
            expected_net_edge_bps_start=50.0,
            capacity_est=100,
        )
        tracker._closed.append(ep)


def test_empty_returns_empty():
    tracker = _make_tracker()
    assert tracker.compute_edge_hazard_curve() == {}


def test_hazard_values_between_0_and_1():
    tracker = _make_tracker()
    _inject_episodes(tracker, [50, 120, 300, 600, 1500, 3000, 80, 400, 1100, 2500])
    hazard = tracker.compute_edge_hazard_curve()
    assert len(hazard) > 0
    for v in hazard.values():
        assert 0.0 <= v <= 1.0, f"Hazard value {v} out of [0,1]"


def test_hazard_sum_leq_1():
    """Sum of all hazard probabilities weighted by survivors should not exceed
    total episodes; equivalently, hazard * alive <= deaths, so sum(deaths) <= n."""
    tracker = _make_tracker()
    _inject_episodes(tracker, [50, 120, 300, 600, 1500, 3000, 80, 400, 1100, 2500])
    hazard = tracker.compute_edge_hazard_curve()
    total = sum(hazard.values())
    assert total <= len(hazard), f"Hazard sum {total} > number of intervals"


def test_all_die_in_first_bucket():
    tracker = _make_tracker()
    _inject_episodes(tracker, [10, 20, 30, 50, 80, 90])
    hazard = tracker.compute_edge_hazard_curve()
    assert hazard.get("0-100ms") == 1.0


def test_all_survive_past_all_buckets():
    tracker = _make_tracker()
    _inject_episodes(tracker, [6000, 7000, 8000, 10000])
    hazard = tracker.compute_edge_hazard_curve()
    for v in hazard.values():
        assert v == 0.0, f"Expected 0.0 hazard, got {v}"


def test_uniform_distribution():
    """One episode dying in each interval should give predictable hazard."""
    tracker = _make_tracker()
    _inject_episodes(tracker, [50, 150, 350, 700, 1500, 3000])
    hazard = tracker.compute_edge_hazard_curve()
    assert "0-100ms" in hazard
    assert abs(hazard["0-100ms"] - 1 / 6) < 0.01
