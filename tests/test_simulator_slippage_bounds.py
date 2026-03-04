"""Test that slippage respects bounds and floor."""

from src.sim.execution_simulator import SimConfig, simulate


def _make_episodes():
    return [
        {
            "token_id": "tok_A",
            "ts_start": 1000.0,
            "side": "BUY",
            "start_time_utc": "2026-01-01T00:00:00.000000Z",
            "duration_ms": 5000,
            "expected_net_edge_bps_after_latency": 20.0,
            "expected_net_edge_bps_at_entry": 25.0,
        },
    ]


def _make_snapshots(spread_bps: float):
    mid = 0.50
    half_spread = spread_bps / 10000.0 / 2.0
    return [
        {
            "ts_utc": 999.0, "token_id": "tok_A",
            "best_bid_px": mid - half_spread, "best_bid_sz": 50,
            "best_ask_px": mid + half_spread, "best_ask_sz": 50,
            "mid_px": mid, "spread_bps": spread_bps,
        },
    ]


def test_slippage_always_non_negative():
    cfg = SimConfig(slippage_bps_floor=2.0, slippage_from_spread_mult=0.5, seed=42)
    for spread in [0, 50, 100, 500, 1000]:
        result = simulate(_make_episodes(), _make_snapshots(spread), cfg, style="taker")
        for t in result["trades"]:
            if t.get("fill_time_utc"):
                assert t["slippage_bps"] >= 0, f"Negative slippage: {t['slippage_bps']}"


def test_slippage_respects_floor():
    cfg = SimConfig(slippage_bps_floor=5.0, slippage_from_spread_mult=0.5, seed=42)
    result = simulate(_make_episodes(), _make_snapshots(0), cfg, style="taker")
    for t in result["trades"]:
        if t.get("fill_time_utc"):
            assert t["slippage_bps"] >= 5.0, (
                f"Slippage {t['slippage_bps']} below floor 5.0"
            )


def test_slippage_scales_with_spread():
    cfg = SimConfig(slippage_bps_floor=0.0, slippage_from_spread_mult=0.5, seed=42)
    r_narrow = simulate(_make_episodes(), _make_snapshots(100), cfg, style="taker")
    r_wide = simulate(_make_episodes(), _make_snapshots(1000), cfg, style="taker")

    slip_narrow = [t["slippage_bps"] for t in r_narrow["trades"] if t.get("fill_time_utc")]
    slip_wide = [t["slippage_bps"] for t in r_wide["trades"] if t.get("fill_time_utc")]

    if slip_narrow and slip_wide:
        assert slip_wide[0] > slip_narrow[0], "Wider spread should produce more slippage"
