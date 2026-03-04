"""Test that simulator handles empty inputs safely."""

from src.sim.execution_simulator import SimConfig, simulate


def test_empty_episodes_and_snapshots():
    result = simulate([], [], SimConfig(seed=42))
    assert result["trades"] == []
    assert result["summary"]["total_trades"] == 0
    assert result["summary"]["fill_rate"] == 0.0


def test_episodes_but_no_snapshots():
    eps = [
        {
            "token_id": "tok_X",
            "ts_start": 1000.0,
            "side": "BUY",
            "start_time_utc": "2026-01-01T00:00:00.000000Z",
            "duration_ms": 5000,
            "expected_net_edge_bps_after_latency": 15.0,
        },
    ]
    result = simulate(eps, [], SimConfig(seed=42))
    for t in result["trades"]:
        assert t["dropped_reason"] == "no_snapshot_at_placement"


def test_snapshots_but_no_episodes():
    snaps = [
        {
            "ts_utc": 1000.0, "token_id": "tok_Y",
            "best_bid_px": 0.48, "best_bid_sz": 50,
            "best_ask_px": 0.52, "best_ask_sz": 50,
            "mid_px": 0.50, "spread_bps": 800.0,
        },
    ]
    result = simulate([], snaps, SimConfig(seed=42))
    assert result["trades"] == []


def test_all_episodes_zero_edge():
    eps = [
        {
            "token_id": "tok_Z",
            "ts_start": 1000.0,
            "side": "BUY",
            "start_time_utc": "2026-01-01T00:00:00.000000Z",
            "duration_ms": 5000,
            "expected_net_edge_bps_after_latency": 0.0,
        },
        {
            "token_id": "tok_Z",
            "ts_start": 2000.0,
            "side": "SELL",
            "start_time_utc": "2026-01-01T00:16:40.000000Z",
            "duration_ms": 3000,
            "expected_net_edge_bps_after_latency": -5.0,
        },
    ]
    snaps = [
        {
            "ts_utc": 999.0, "token_id": "tok_Z",
            "best_bid_px": 0.48, "best_bid_sz": 50,
            "best_ask_px": 0.52, "best_ask_sz": 50,
            "mid_px": 0.50, "spread_bps": 800.0,
        },
    ]
    result = simulate(eps, snaps, SimConfig(seed=42))
    for t in result["trades"]:
        assert t["dropped_reason"] == "non_positive_edge"
    assert result["summary"]["filled"] == 0
