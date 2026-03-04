"""Tests for EdgeEpisodeTracker.compute_edge_decay_curve()."""

from src.strategy.edge_episode_tracker import EdgeEpisodeTracker, EdgeEpisode, OpenEpisode


def _make_tracker() -> EdgeEpisodeTracker:
    return EdgeEpisodeTracker(
        edge_enter_bps=10,
        edge_exit_bps=5,
        max_episode_seconds=300,
        order_size=1,
        fee_bps=0,
    )


def _inject_episode_with_samples(
    tracker: EdgeEpisodeTracker,
    token_id: str,
    samples: list,
    duration_ms: float = 5000,
) -> None:
    """Inject an OpenEpisode, populate its edge_samples, then close it
    via the internal _extract_decay_and_latency + append a synthetic closed ep."""
    if not samples:
        return
    start_mono = samples[0][0]
    ep = OpenEpisode(
        token_id=token_id,
        ts_start=0.0,
        side="BUY",
        mid_start=0.50,
        p_oracle_start=0.55,
        spread_start=0.02,
        edge_bps_start=samples[0][1],
        expected_net_edge_bps_start=samples[0][1],
        capacity_est=100,
        start_time_utc="2025-01-01T00:00:00Z",
        start_monotonic=start_mono,
        edge_samples=list(samples),
    )
    with tracker._lock:
        tracker._extract_decay_and_latency(ep)
        closed = EdgeEpisode(
            token_id=token_id,
            ts_start=0.0,
            ts_end=duration_ms / 1000.0,
            duration_ms=duration_ms,
            side="BUY",
            mid_start=0.50,
            mid_end=0.48,
            p_oracle_start=0.55,
            spread_start=0.02,
            edge_bps_start=samples[0][1],
            expected_net_edge_bps_start=samples[0][1],
            capacity_est=100,
        )
        tracker._closed.append(closed)


def test_empty_returns_empty():
    tracker = _make_tracker()
    assert tracker.compute_edge_decay_curve() == {}


def test_decay_values_between_0_and_1():
    tracker = _make_tracker()
    base_mono = 1000.0
    samples = [
        (base_mono + 0.0, 100.0),
        (base_mono + 0.05, 95.0),
        (base_mono + 0.15, 80.0),
        (base_mono + 0.30, 65.0),
        (base_mono + 0.60, 50.0),
        (base_mono + 1.20, 30.0),
        (base_mono + 2.50, 15.0),
        (base_mono + 5.50, 5.0),
    ]
    _inject_episode_with_samples(tracker, "tok1", samples, duration_ms=6000)
    curve = tracker.compute_edge_decay_curve()
    assert len(curve) > 0
    for v in curve.values():
        assert 0.0 <= v <= 1.0001, f"Decay ratio {v} out of range"


def test_decay_monotonic_decreasing():
    tracker = _make_tracker()
    base_mono = 1000.0
    samples = [
        (base_mono + 0.0, 100.0),
        (base_mono + 0.05, 90.0),
        (base_mono + 0.15, 75.0),
        (base_mono + 0.30, 60.0),
        (base_mono + 0.60, 45.0),
        (base_mono + 1.20, 30.0),
        (base_mono + 2.50, 15.0),
        (base_mono + 5.50, 5.0),
    ]
    _inject_episode_with_samples(tracker, "tok1", samples, duration_ms=6000)
    curve = tracker.compute_edge_decay_curve()
    vals = list(curve.values())
    for i in range(1, len(vals)):
        assert vals[i] <= vals[i - 1] + 1e-9, (
            f"Decay not monotonic: {vals[i]} > {vals[i-1]}"
        )


def test_bucket_0ms_is_1():
    """The ratio at t=0 should always be 1.0 (edge / edge at detection)."""
    tracker = _make_tracker()
    base_mono = 1000.0
    samples = [
        (base_mono + 0.0, 80.0),
        (base_mono + 0.5, 60.0),
        (base_mono + 1.0, 40.0),
    ]
    _inject_episode_with_samples(tracker, "tok1", samples, duration_ms=1500)
    curve = tracker.compute_edge_decay_curve()
    assert "0ms" in curve
    assert abs(curve["0ms"] - 1.0) < 1e-6


def test_multiple_episodes_averaged():
    tracker = _make_tracker()
    base = 1000.0
    samples_a = [
        (base, 100.0),
        (base + 0.5, 50.0),
        (base + 1.0, 20.0),
    ]
    _inject_episode_with_samples(tracker, "tokA", samples_a, duration_ms=1500)

    base2 = 2000.0
    samples_b = [
        (base2, 100.0),
        (base2 + 0.5, 70.0),
        (base2 + 1.0, 40.0),
    ]
    _inject_episode_with_samples(tracker, "tokB", samples_b, duration_ms=1500)
    curve = tracker.compute_edge_decay_curve()
    assert "500ms" in curve
    assert 0.0 <= curve["500ms"] <= 1.0
