"""Test that MarketDataRecorder enforces /tmp path safety."""

from src.sim.marketdata_recorder import MarketDataRecorder, _path_safe_under_tmp


def test_path_under_tmp_is_safe():
    assert _path_safe_under_tmp("/tmp/polymarket/data.jsonl") is True


def test_path_under_private_tmp_is_safe():
    assert _path_safe_under_tmp("/private/tmp/polymarket/data.jsonl") is True


def test_path_outside_tmp_rejected():
    assert _path_safe_under_tmp("/home/user/data.jsonl") is False
    assert _path_safe_under_tmp("/var/log/data.jsonl") is False
    assert _path_safe_under_tmp("/etc/data.jsonl") is False


def test_recorder_disables_writes_outside_tmp():
    recorder = MarketDataRecorder(state_dir="/home/fakeuser/unsafe")
    recorder.record(
        token_id="tok_test",
        best_bid_px=0.48,
        best_bid_sz=50,
        best_ask_px=0.52,
        best_ask_sz=50,
    )
    # Should not crash, just silently skip
    assert recorder.drops == 0  # not counted as throttle drop


def test_recorder_writes_under_tmp():
    import os
    import json
    test_dir = "/tmp/test_recorder_safety"
    recorder = MarketDataRecorder(state_dir=test_dir)
    recorder.record(
        token_id="tok_safe",
        best_bid_px=0.49,
        best_bid_sz=30,
        best_ask_px=0.51,
        best_ask_sz=40,
    )
    assert os.path.exists(recorder.path)
    with open(recorder.path, "r") as f:
        lines = f.readlines()
    assert len(lines) >= 1
    data = json.loads(lines[-1])
    assert data["token_id"] == "tok_safe"
    # Cleanup
    try:
        os.remove(recorder.path)
        os.rmdir(test_dir)
    except Exception:
        pass
