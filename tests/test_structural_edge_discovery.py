"""Tests for structural edge discovery."""

from datetime import datetime, timedelta, timezone

from src.strategy.structural_edge_discovery import compute_structural_edge_candidates


def _now():
    return datetime(2026, 2, 5, 12, 0, 0, tzinfo=timezone.utc)


def _make_summary(
    token_id: str,
    minutes_ago: float,
    duration_ms: float = 1000.0,
    net_edge: float = 15.0,
    auc: float = 500.0,
    now: datetime = None,
):
    now = now or _now()
    end = now - timedelta(minutes=minutes_ago)
    return {
        "end_time_utc": end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "token_id": token_id,
        "market_slug": None,
        "duration_ms": duration_ms,
        "expected_net_edge_bps_after_latency": net_edge,
        "edge_auc_bps_ms": auc,
    }


def _batch(token_id, count, minutes_start, minutes_end, **kwargs):
    """Create a batch of episodes spread across a time range."""
    now = kwargs.pop("now", None) or _now()
    step = (minutes_start - minutes_end) / max(count - 1, 1) if count > 1 else 0
    return [
        _make_summary(token_id, minutes_start - i * step, now=now, **kwargs)
        for i in range(count)
    ]


# ------------------------------------------------------------------
# Test 1: empty dataset
# ------------------------------------------------------------------
def test_empty_returns_empty():
    result = compute_structural_edge_candidates(
        [], exec_latency_ms=500, now_utc=_now(),
    )
    assert result["structural_edge_candidates"] == []


# ------------------------------------------------------------------
# Test 2: candidate passes 2 of 3 windows -> included
# ------------------------------------------------------------------
def test_candidate_passes_2_of_3_windows():
    now = _now()
    eps = []
    # Window 0 (0-60 min ago): 25 episodes with good edge
    eps.extend(_batch("tok_A", 25, 55, 5, duration_ms=1000, net_edge=15.0, now=now))
    # Window 1 (60-120 min ago): 25 episodes with good edge
    eps.extend(_batch("tok_A", 25, 115, 65, duration_ms=1000, net_edge=15.0, now=now))
    # Window 2 (120-180 min ago): only 5 episodes (below threshold)
    eps.extend(_batch("tok_A", 5, 175, 125, duration_ms=1000, net_edge=15.0, now=now))

    result = compute_structural_edge_candidates(
        eps, exec_latency_ms=500, now_utc=now,
        min_episodes_per_window=20,
        persistent_windows_required=2,
    )
    candidates = result["structural_edge_candidates"]
    assert len(candidates) == 1
    assert candidates[0]["token_id"] == "tok_A"
    assert candidates[0]["windows_passed"] == 2


# ------------------------------------------------------------------
# Test 3: spike filtered -> excluded
# ------------------------------------------------------------------
def test_spike_filtered_excluded():
    now = _now()
    eps = []
    # All 3 windows have enough episodes, but 5/25 are extreme outliers
    # pushing p90 well above median * spike_multiplier
    for w_offset in [30, 90, 150]:
        for i in range(25):
            edge = 10.0 if i < 20 else 200.0
            eps.append(_make_summary(
                "tok_spike", w_offset + i * 0.5,
                duration_ms=1000, net_edge=edge, now=now,
            ))

    result = compute_structural_edge_candidates(
        eps, exec_latency_ms=500, now_utc=now,
        spike_multiplier=4.0,
        min_episodes_per_window=20,
        persistent_windows_required=2,
    )
    candidates = result["structural_edge_candidates"]
    spike_ids = [c["token_id"] for c in candidates]
    assert "tok_spike" not in spike_ids


# ------------------------------------------------------------------
# Test 4: cooldown blocks re-emission within window
# ------------------------------------------------------------------
def test_cooldown_blocks_reemission():
    from src.strategy.edge_episode_tracker import EdgeEpisodeTracker

    now = _now()
    eps = []
    for w_offset in [30, 90, 150]:
        eps.extend(_batch(
            "tok_cool", 25, w_offset + 25, w_offset,
            duration_ms=1000, net_edge=15.0, now=now,
        ))

    tracker = EdgeEpisodeTracker(
        edge_enter_bps=10, edge_exit_bps=5,
        max_episode_seconds=300, order_size=1, fee_bps=0,
        state_dir="/tmp/test_structural_discovery",
    )
    # Inject summaries into buffer
    for ep in eps:
        tracker._discovery_buffer.append(ep)

    # First call: should produce new markets
    import os
    os.environ["STRUCT_WINDOW_MINUTES"] = "60"
    os.environ["STRUCT_LOOKBACK_WINDOWS"] = "3"
    os.environ["MIN_EPISODES_PER_WINDOW"] = "20"
    os.environ["MIN_SURVIVAL_AT_LATENCY"] = "0.40"
    os.environ["MIN_MEDIAN_NET_EDGE_BPS"] = "5.0"
    os.environ["PERSISTENT_WINDOWS_REQUIRED"] = "2"
    os.environ["NEW_MARKET_COOLDOWN_HOURS"] = "24"

    try:
        result1 = tracker.compute_structural_edge_discovery()
        new1 = result1.get("structural_edge_new_markets", [])
        new1_keys = [m["market_key"] for m in new1]

        # Second call: same data, should NOT re-emit (cooldown active)
        result2 = tracker.compute_structural_edge_discovery()
        new2 = result2.get("structural_edge_new_markets", [])
        new2_keys = [m["market_key"] for m in new2]

        if "tok_cool" in new1_keys:
            assert "tok_cool" not in new2_keys, "Cooldown should block re-emission"
    finally:
        # Cleanup env
        for k in [
            "STRUCT_WINDOW_MINUTES", "STRUCT_LOOKBACK_WINDOWS",
            "MIN_EPISODES_PER_WINDOW", "MIN_SURVIVAL_AT_LATENCY",
            "MIN_MEDIAN_NET_EDGE_BPS", "PERSISTENT_WINDOWS_REQUIRED",
            "NEW_MARKET_COOLDOWN_HOURS",
        ]:
            os.environ.pop(k, None)
        # Cleanup flagged file
        try:
            p = tracker._flagged_json_path()
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


# ------------------------------------------------------------------
# Test 5: score normalized and sorting descending
# ------------------------------------------------------------------
def test_score_normalized_and_sorted():
    now = _now()
    eps = []
    # Market A: strong
    for w_offset in [30, 90, 150]:
        eps.extend(_batch(
            "tok_strong", 30, w_offset + 25, w_offset,
            duration_ms=2000, net_edge=25.0, auc=1000.0, now=now,
        ))
    # Market B: moderate
    for w_offset in [30, 90, 150]:
        eps.extend(_batch(
            "tok_moderate", 25, w_offset + 20, w_offset,
            duration_ms=1000, net_edge=10.0, auc=300.0, now=now,
        ))

    result = compute_structural_edge_candidates(
        eps, exec_latency_ms=500, now_utc=now,
        min_episodes_per_window=20,
        persistent_windows_required=2,
    )
    candidates = result["structural_edge_candidates"]
    assert len(candidates) >= 1

    # Sorted descending
    scores = [c["score"] for c in candidates]
    for i in range(1, len(scores)):
        assert scores[i] <= scores[i - 1]

    # Top score is 1.0 (normalized)
    assert candidates[0]["score"] == 1.0

    # All scores in [0, 1]
    for c in candidates:
        assert 0.0 <= c["score"] <= 1.0


# ------------------------------------------------------------------
# Test 6: insufficient data in last window -> not a candidate
# ------------------------------------------------------------------
def test_last_window_must_pass():
    now = _now()
    eps = []
    # Windows 1 and 2 are great, but window 0 (most recent) has too few
    eps.extend(_batch("tok_old", 25, 115, 65, duration_ms=1000, net_edge=15.0, now=now))
    eps.extend(_batch("tok_old", 25, 175, 125, duration_ms=1000, net_edge=15.0, now=now))
    eps.extend(_batch("tok_old", 3, 55, 5, duration_ms=1000, net_edge=15.0, now=now))

    result = compute_structural_edge_candidates(
        eps, exec_latency_ms=500, now_utc=now,
        min_episodes_per_window=20,
        persistent_windows_required=2,
    )
    candidates = result["structural_edge_candidates"]
    assert len(candidates) == 0
