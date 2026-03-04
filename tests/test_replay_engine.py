"""Tests for ReplayEngine — deterministic event replay."""

from src.sim.replay_engine import ReplayEngine


def _make_events(n=100):
    return [
        {"ts_utc": float(i), "token_id": f"mk_{i % 3}", "mid_px": 0.5 + i * 0.0001}
        for i in range(n)
    ]


def test_processes_all_events():
    events = _make_events(50)
    engine = ReplayEngine(events)
    collected = []
    count = engine.run(lambda ev: collected.append(ev))
    assert count == 50
    assert len(collected) == 50
    assert engine.processed_count == 50


def test_deterministic_order():
    events = _make_events(100)
    collected1 = []
    collected2 = []
    ReplayEngine(events).run(lambda ev: collected1.append(ev["ts_utc"]))
    ReplayEngine(events).run(lambda ev: collected2.append(ev["ts_utc"]))
    assert collected1 == collected2


def test_empty_events():
    engine = ReplayEngine([])
    count = engine.run(lambda ev: None)
    assert count == 0
    assert engine.processed_count == 0


def test_filter_by_market():
    events = _make_events(30)
    engine = ReplayEngine(events)
    collected = []
    count = engine.run_filtered(lambda ev: collected.append(ev), market_key="mk_0")
    assert all(e["token_id"] == "mk_0" for e in collected)
    assert count == len(collected)


def test_filter_by_time():
    events = _make_events(100)
    collected = []
    engine = ReplayEngine(events)
    engine.run_filtered(lambda ev: collected.append(ev), from_ts=50, to_ts=70)
    assert all(50 <= e["ts_utc"] <= 70 for e in collected)


def test_handles_100k_events():
    events = _make_events(100_000)
    engine = ReplayEngine(events)
    count = engine.run(lambda ev: None)
    assert count == 100_000
