"""
Guardrail: EpisodeStore JSONL path must stay under /tmp.
Write failures must not crash the process.
"""

import json
import os
import tempfile

from src.strategy.episode_store import EpisodeStore, _path_safe_under_tmp


def test_path_under_tmp():
    store = EpisodeStore(state_dir="/tmp/polymarket_state")
    assert store.path.endswith("edge_episodes.jsonl")
    assert _path_safe_under_tmp(store.path)


def test_rejects_outside_tmp():
    assert not _path_safe_under_tmp("/etc/edge_episodes.jsonl")
    assert not _path_safe_under_tmp("/home/user/edge_episodes.jsonl")
    assert not _path_safe_under_tmp("/var/data/edge_episodes.jsonl")


def test_writes_disabled_outside_tmp():
    store = EpisodeStore(state_dir="/etc/should_not_write")
    store.append({"test": True})
    assert not os.path.exists(store.path)


def test_append_writes_jsonl():
    with tempfile.TemporaryDirectory(prefix="/tmp/test_ep_store_") as d:
        store = EpisodeStore(state_dir=d)
        store.append({"token_id": "abc", "duration_ms": 123.4})
        store.append({"token_id": "def", "duration_ms": 456.7})
        with open(store.path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["token_id"] == "abc"


def test_write_failure_nonfatal():
    store = EpisodeStore(state_dir="/tmp/polymarket_test_nonexist_ep")
    store._safe = True
    store._path = "/tmp/polymarket_test_nonexist_ep/sub/deep/edge_episodes.jsonl"
    store.append({"test": True})
