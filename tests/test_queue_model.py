"""Tests for QueueModel — deterministic queue position and fill logic."""

import random
from src.sim.queue_model import QueueModel


def test_deterministic_with_same_seed():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    qm1 = QueueModel(rng=rng1)
    qm2 = QueueModel(rng=rng2)

    pos1 = qm1.initial_position(100)
    pos2 = qm2.initial_position(100)
    assert pos1 == pos2


def test_initial_position_within_visible_size():
    rng = random.Random(99)
    qm = QueueModel(rng=rng)
    for _ in range(100):
        pos = qm.initial_position(50.0)
        assert 0 <= pos <= 50.0


def test_zero_visible_size_returns_inf():
    qm = QueueModel(rng=random.Random(1))
    assert qm.initial_position(0) == float("inf")
    assert qm.initial_position(-5) == float("inf")


def test_is_filled_basic():
    qm = QueueModel()
    assert qm.is_filled(queue_position=10, traded_volume=15) is True
    assert qm.is_filled(queue_position=10, traded_volume=5) is False
    assert qm.is_filled(queue_position=10, traded_volume=10) is True


def test_probabilistic_fill_deterministic():
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    qm1 = QueueModel(rng=rng1)
    qm2 = QueueModel(rng=rng2)

    results1 = [qm1.probabilistic_fill(0.15, 5.0) for _ in range(20)]
    results2 = [qm2.probabilistic_fill(0.15, 5.0) for _ in range(20)]
    assert results1 == results2


def test_probabilistic_fill_zero_time():
    qm = QueueModel(rng=random.Random(1))
    assert qm.probabilistic_fill(0.15, 0.0) is False


def test_probabilistic_fill_zero_prob():
    qm = QueueModel(rng=random.Random(1))
    assert qm.probabilistic_fill(0.0, 10.0) is False


def test_fill_time_offset_within_bounds():
    rng = random.Random(42)
    qm = QueueModel(rng=rng)
    for _ in range(100):
        offset = qm.fill_time_offset(5.0)
        assert 0 <= offset <= 5.0


def test_fill_time_offset_zero():
    qm = QueueModel(rng=random.Random(1))
    assert qm.fill_time_offset(0) == 0.0
