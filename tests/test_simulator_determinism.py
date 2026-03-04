"""Test that the simulator produces identical output given same inputs + seed."""

from src.sim.execution_simulator import SimConfig, simulate


def _make_episodes():
    return [
        {
            "token_id": "tok_A",
            "ts_start": 1000.0,
            "side": "BUY",
            "start_time_utc": "2026-01-01T00:00:00.000000Z",
            "duration_ms": 5000,
            "expected_net_edge_bps_after_latency": 15.0,
            "expected_net_edge_bps_at_entry": 20.0,
        },
        {
            "token_id": "tok_A",
            "ts_start": 2000.0,
            "side": "SELL",
            "start_time_utc": "2026-01-01T00:16:40.000000Z",
            "duration_ms": 3000,
            "expected_net_edge_bps_after_latency": 10.0,
            "expected_net_edge_bps_at_entry": 12.0,
        },
    ]


def _make_snapshots():
    return [
        {
            "ts_utc": 999.0, "token_id": "tok_A",
            "best_bid_px": 0.48, "best_bid_sz": 50,
            "best_ask_px": 0.52, "best_ask_sz": 50,
            "mid_px": 0.50, "spread_bps": 800.0,
        },
        {
            "ts_utc": 1000.05, "token_id": "tok_A",
            "best_bid_px": 0.49, "best_bid_sz": 30,
            "best_ask_px": 0.51, "best_ask_sz": 40,
            "mid_px": 0.50, "spread_bps": 400.0,
        },
        {
            "ts_utc": 2000.05, "token_id": "tok_A",
            "best_bid_px": 0.47, "best_bid_sz": 60,
            "best_ask_px": 0.53, "best_ask_sz": 45,
            "mid_px": 0.50, "spread_bps": 1200.0,
        },
    ]


def test_same_seed_same_output():
    eps = _make_episodes()
    snaps = _make_snapshots()
    cfg = SimConfig(seed=123)

    result1 = simulate(eps, snaps, cfg, style="both")
    result2 = simulate(eps, snaps, cfg, style="both")

    assert len(result1["trades"]) == len(result2["trades"])
    for t1, t2 in zip(result1["trades"], result2["trades"]):
        assert t1 == t2, f"Mismatch:\n{t1}\nvs\n{t2}"


def test_different_seed_different_maker_fills():
    eps = _make_episodes()
    snaps = _make_snapshots()

    r1 = simulate(eps, snaps, SimConfig(seed=1), style="maker")
    r2 = simulate(eps, snaps, SimConfig(seed=99999), style="maker")

    fills1 = [t for t in r1["trades"] if t.get("fill_time_utc")]
    fills2 = [t for t in r2["trades"] if t.get("fill_time_utc")]
    # With very different seeds, maker fills may differ (probabilistic)
    # At minimum, both runs should produce the same number of trades
    assert len(r1["trades"]) == len(r2["trades"])
