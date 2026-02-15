"""
Testes para o módulo de calibração de probabilidades.

Cobertura: Brier, EWMA, N_MIN, baselines, k, persistência, recovery,
classificação, deterioração, validação, integração, concorrência, schema,
crescimento, deep copy, negativos, fail-closed, timestamps, dir fsync,
state_corrupted, validação UTC, corruption mode, os.replace, is_corrupted.
"""

import json
import logging
import math
import os
import shutil
import threading
import time
import pytest
from pathlib import Path

from src.engine.probability_calibrator import (
    ProbabilityCalibrator,
    MarketCategory,
    TradeType,
    CategoryTypeMetrics,
    calculate_lambda,
    calculate_brier_score,
    clamp,
    SCHEMA_VERSION,
    HALF_LIFE_FAST,
    HALF_LIFE_SLOW,
    E_MAX,
    ALPHA,
    K_MIN,
    K_MIN_GLOBAL,
    N_MIN,
    THETA,
    BASELINE_BRIER_BY_CATEGORY,
    TAG_TO_CATEGORY,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def ws_tmp(request):
    tmp_dir = Path(__file__).parent.parent / ".test_tmp" / request.node.name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def calibrator(ws_tmp):
    return ProbabilityCalibrator(state_path=str(ws_tmp / "calibrator.json"))


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

class TestCalculateLambda:
    def test_lambda_formula(self):
        assert abs(calculate_lambda(20) - math.exp(math.log(0.5) / 20)) < 1e-10

    def test_lambda_half_life_20(self):
        assert 0.96 < calculate_lambda(20) < 0.97

    def test_lambda_half_life_100(self):
        assert 0.99 < calculate_lambda(100) < 0.995

    def test_lambda_invalid(self):
        with pytest.raises(ValueError):
            calculate_lambda(0)
        with pytest.raises(ValueError):
            calculate_lambda(-5)


class TestCalculateBrierScore:
    def test_perfect_prediction_yes(self):
        assert calculate_brier_score(1.0, 1) == 0.0

    def test_perfect_prediction_no(self):
        assert calculate_brier_score(0.0, 0) == 0.0

    def test_worst_prediction_yes(self):
        assert calculate_brier_score(0.0, 1) == 1.0

    def test_worst_prediction_no(self):
        assert calculate_brier_score(1.0, 0) == 1.0

    def test_random_prediction(self):
        assert calculate_brier_score(0.5, 0) == 0.25
        assert calculate_brier_score(0.5, 1) == 0.25

    def test_partial_confidence(self):
        assert abs(calculate_brier_score(0.7, 1) - 0.09) < 1e-10


class TestClamp:
    def test_value_in_range(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_value_below_min(self):
        assert clamp(-0.1, 0.0, 1.0) == 0.0

    def test_value_above_max(self):
        assert clamp(1.5, 0.0, 1.0) == 1.0


# =============================================================================
# BRIER E CLAMP
# =============================================================================

class TestBrierAndClamp:
    def test_brier_clamped_at_e_max(self, calibrator):
        calibrator.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.1, 1)
        m = calibrator.get_metrics(MarketCategory.POLITICS, TradeType.DIRECTIONAL)
        bl = BASELINE_BRIER_BY_CATEGORY["POLITICS"]
        lam = calculate_lambda(HALF_LIFE_FAST)
        assert abs(m["bs_fast"] - (lam * bl + (1 - lam) * E_MAX)) < 1e-6

    def test_brier_not_clamped_when_below(self, calibrator):
        calibrator.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.7, 1)
        m = calibrator.get_metrics(MarketCategory.POLITICS, TradeType.DIRECTIONAL)
        bl = BASELINE_BRIER_BY_CATEGORY["POLITICS"]
        lam = calculate_lambda(HALF_LIFE_FAST)
        assert abs(m["bs_fast"] - (lam * bl + (1 - lam) * 0.09)) < 1e-6


# =============================================================================
# EWMA
# =============================================================================

class TestEWMAFormula:
    def test_ewma_formula_exact(self, calibrator):
        cat, tt = MarketCategory.FINANCE, TradeType.ARBITRAGE
        bl = BASELINE_BRIER_BY_CATEGORY["FINANCE"]
        lf, ls = calculate_lambda(HALF_LIFE_FAST), calculate_lambda(HALF_LIFE_SLOW)
        calibrator.record_trade_result(cat, tt, 0.6, 1)
        ef1 = lf * bl + (1 - lf) * 0.16
        es1 = ls * bl + (1 - ls) * 0.16
        m = calibrator.get_metrics(cat, tt)
        assert abs(m["bs_fast"] - ef1) < 1e-9 and abs(m["bs_slow"] - es1) < 1e-9
        calibrator.record_trade_result(cat, tt, 0.8, 0)
        ef2 = lf * ef1 + (1 - lf) * 0.5
        es2 = ls * es1 + (1 - ls) * 0.5
        m = calibrator.get_metrics(cat, tt)
        assert abs(m["bs_fast"] - ef2) < 1e-9 and abs(m["bs_slow"] - es2) < 1e-9

    def test_fast_reacts_faster_than_slow(self, calibrator):
        cat, tt = MarketCategory.SPORTS, TradeType.DIRECTIONAL
        bl = BASELINE_BRIER_BY_CATEGORY["SPORTS"]
        for _ in range(10):
            calibrator.record_trade_result(cat, tt, 0.9, 0)
        m = calibrator.get_metrics(cat, tt)
        assert abs(m["bs_fast"] - bl) > abs(m["bs_slow"] - bl)


# =============================================================================
# N_MIN
# =============================================================================

class TestNMin:
    def test_k_is_1_when_below_n_min(self, calibrator):
        for _ in range(N_MIN - 1):
            calibrator.record_trade_result(MarketCategory.CRYPTO, TradeType.DIRECTIONAL, 0.9, 0)
        assert calibrator.get_sizing_multiplier(MarketCategory.CRYPTO, TradeType.DIRECTIONAL) == 1.0

    def test_k_can_change_after_n_min(self, calibrator):
        for _ in range(N_MIN):
            calibrator.record_trade_result(MarketCategory.CRYPTO, TradeType.DIRECTIONAL, 0.9, 0)
        assert calibrator.get_sizing_multiplier(MarketCategory.CRYPTO, TradeType.DIRECTIONAL) < 1.0


# =============================================================================
# BASELINES
# =============================================================================

class TestBaselines:
    def test_different_baselines(self):
        assert BASELINE_BRIER_BY_CATEGORY["FINANCE"] == 0.18
        assert BASELINE_BRIER_BY_CATEGORY["CRYPTO"] == 0.22

    def test_metrics_initialized_with_baseline(self, calibrator):
        calibrator.record_trade_result(MarketCategory.CRYPTO, TradeType.ARBITRAGE, 0.5, 1)
        m = calibrator.get_metrics(MarketCategory.CRYPTO, TradeType.ARBITRAGE)
        bl = BASELINE_BRIER_BY_CATEGORY["CRYPTO"]
        lam = calculate_lambda(HALF_LIFE_SLOW)
        assert abs(m["bs_slow"] - (lam * bl + (1 - lam) * 0.25)) < 1e-6


# =============================================================================
# MULTIPLICADOR K
# =============================================================================

class TestSizingMultiplier:
    def test_k_formula(self, calibrator):
        cat, tt = MarketCategory.POLITICS, TradeType.DIRECTIONAL
        bl = BASELINE_BRIER_BY_CATEGORY["POLITICS"]
        for i in range(N_MIN + 10):
            calibrator.record_trade_result(cat, tt, 0.5, i % 2)
        m = calibrator.get_metrics(cat, tt)
        bs = m["bs_fast"]
        expected = 1.0 if bs <= bl else clamp(1.0 - ALPHA * ((bs - bl) / bl), K_MIN, 1.0)
        assert abs(calibrator.get_sizing_multiplier(cat, tt) - expected) < 1e-6

    def test_k_is_1_when_bs_fast_below_baseline(self, calibrator):
        for _ in range(N_MIN + 5):
            calibrator.record_trade_result(MarketCategory.FINANCE, TradeType.ARBITRAGE, 0.9, 1)
        assert calibrator.get_sizing_multiplier(MarketCategory.FINANCE, TradeType.ARBITRAGE) == 1.0

    def test_k_never_below_k_min(self, calibrator):
        for _ in range(50):
            calibrator.record_trade_result(MarketCategory.ENTERTAINMENT, TradeType.STATISTICAL, 1.0, 0)
        assert calibrator.get_sizing_multiplier(MarketCategory.ENTERTAINMENT, TradeType.STATISTICAL) >= K_MIN


# =============================================================================
# PERSISTÊNCIA
# =============================================================================

class TestPersistence:
    def test_save_and_load(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        c1 = ProbabilityCalibrator(state_path=str(sp))
        for i in range(20):
            c1.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.6, i % 2)
        m1 = c1.get_metrics(MarketCategory.POLITICS, TradeType.DIRECTIONAL)
        c2 = ProbabilityCalibrator(state_path=str(sp))
        m2 = c2.get_metrics(MarketCategory.POLITICS, TradeType.DIRECTIONAL)
        assert m2["trade_count"] == m1["trade_count"]
        assert abs(m2["bs_fast"] - m1["bs_fast"]) < 1e-9

    def test_file_is_valid_json_with_schema_version(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        c = ProbabilityCalibrator(state_path=str(sp))
        c.record_trade_result(MarketCategory.CRYPTO, TradeType.ARBITRAGE, 0.5, 1)
        with open(sp) as f:
            assert json.load(f)["schema_version"] == SCHEMA_VERSION

    def test_atomic_write(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        c = ProbabilityCalibrator(state_path=str(sp))
        for i in range(10):
            c.record_trade_result(MarketCategory.SPORTS, TradeType.HEDGING, 0.5, i % 2)
            with open(sp) as f:
                assert json.load(f)["schema_version"] == SCHEMA_VERSION

    def test_persisted_state_only_contains_essential_fields(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        c = ProbabilityCalibrator(state_path=str(sp))
        for i in range(5):
            c.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, i % 2)
        with open(sp) as f:
            data = json.load(f)
        for key, entry in data["metrics"].items():
            assert set(entry.keys()) == {"bs_fast", "bs_slow", "trade_count"}


# =============================================================================
# RECUPERAÇÃO E SAFETY MODE
# =============================================================================

class TestRecovery:
    def test_no_safety_mode_when_file_missing(self, ws_tmp):
        c = ProbabilityCalibrator(state_path=str(ws_tmp / "nope.json"))
        assert c.safety_mode is False
        assert c.state_corrupted is False

    def test_safety_mode_when_file_corrupted(self, ws_tmp):
        sp = ws_tmp / "bad.json"
        sp.write_text("not json {{{")
        c = ProbabilityCalibrator(state_path=str(sp))
        assert c.safety_mode is True
        assert "JSON" in c.safety_mode_reason

    def test_safety_mode_when_invalid_schema(self, ws_tmp):
        sp = ws_tmp / "bad.json"
        sp.write_text(json.dumps({"foo": "bar"}))
        c = ProbabilityCalibrator(state_path=str(sp))
        assert c.safety_mode is True

    def test_safety_mode_returns_k_min(self, ws_tmp):
        sp = ws_tmp / "bad.json"
        sp.write_text("corrupted")
        c = ProbabilityCalibrator(state_path=str(sp))
        assert c.get_sizing_multiplier(MarketCategory.POLITICS, TradeType.DIRECTIONAL) == K_MIN

    def test_exit_safety_mode_after_n_min_trades(self, ws_tmp):
        sp = ws_tmp / "bad.json"
        sp.write_text("corrupted")
        c = ProbabilityCalibrator(state_path=str(sp))
        assert c.safety_mode is True
        for i in range(N_MIN):
            c.record_trade_result(MarketCategory.FINANCE, TradeType.HEDGING, 0.5, i % 2)
        assert c.safety_mode is False


# =============================================================================
# CLASSIFICAÇÃO
# =============================================================================

class TestCategoryClassification:
    def test_classify_by_tag_politics(self, calibrator):
        assert calibrator.classify_market_category(["Elections"]) == MarketCategory.POLITICS

    def test_classify_by_tag_crypto(self, calibrator):
        assert calibrator.classify_market_category(["Bitcoin"]) == MarketCategory.CRYPTO

    def test_classify_by_tag_sports(self, calibrator):
        assert calibrator.classify_market_category(["NBA"]) == MarketCategory.SPORTS

    def test_classify_by_title_keywords(self, calibrator):
        assert calibrator.classify_market_category(["x"], title="Will Bitcoin hit $100k?") == MarketCategory.CRYPTO

    def test_classify_by_description_keywords(self, calibrator):
        assert calibrator.classify_market_category(["x"], title="A", description="presidential election results") == MarketCategory.POLITICS

    def test_fallback_to_other(self, calibrator):
        assert calibrator.classify_market_category(["unknown"], title="Foo bar baz", description="Unrelated") == MarketCategory.OTHER

    def test_category_override(self, ws_tmp):
        c = ProbabilityCalibrator(state_path=str(ws_tmp / "cal.json"), category_overrides={"special_tag": "FINANCE"})
        assert c.classify_market_category(["special_tag", "Elections"]) == MarketCategory.FINANCE


# =============================================================================
# DETERIORAÇÃO
# =============================================================================

class TestDeterioration:
    def test_deterioration_flag(self, calibrator):
        cat, tt = MarketCategory.WEATHER, TradeType.DIRECTIONAL
        for _ in range(20):
            calibrator.record_trade_result(cat, tt, 0.8, 1)
        for _ in range(10):
            calibrator.record_trade_result(cat, tt, 0.9, 0)
        m = calibrator.get_metrics(cat, tt)
        if m["bs_fast"] - m["bs_slow"] > THETA:
            assert m["recent_deterioration"] is True

    def test_deterioration_does_not_affect_k(self, calibrator):
        cat, tt = MarketCategory.SCIENCE, TradeType.STATISTICAL
        bl = BASELINE_BRIER_BY_CATEGORY["SCIENCE"]
        for i in range(N_MIN + 5):
            calibrator.record_trade_result(cat, tt, 0.5, i % 2)
        m = calibrator.get_metrics(cat, tt)
        bs = m["bs_fast"]
        expected = 1.0 if bs <= bl else clamp(1.0 - ALPHA * ((bs - bl) / bl), K_MIN, 1.0)
        assert abs(calibrator.get_sizing_multiplier(cat, tt) - expected) < 1e-6


# =============================================================================
# VALIDAÇÃO
# =============================================================================

class TestValidation:
    def test_invalid_outcome(self, calibrator):
        with pytest.raises(ValueError, match="actual_outcome"):
            calibrator.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, 2)

    def test_probability_clamped(self, calibrator):
        calibrator.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 1.5, 1)
        calibrator.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, -0.1, 0)
        assert calibrator.get_metrics(MarketCategory.POLITICS, TradeType.DIRECTIONAL)["trade_count"] == 2


# =============================================================================
# INTEGRAÇÃO
# =============================================================================

class TestIntegration:
    def test_full_workflow(self, ws_tmp):
        import random
        sp = ws_tmp / "cal.json"
        c = ProbabilityCalibrator(state_path=str(sp))
        cat = c.classify_market_category(["Elections"], "Presidential race")
        assert cat == MarketCategory.POLITICS
        random.seed(123)
        for _ in range(25):
            c.record_trade_result(cat, TradeType.DIRECTIONAL, 0.6 + random.random() * 0.2, 1 if random.random() < 0.7 else 0)
        k = c.get_sizing_multiplier(cat, TradeType.DIRECTIONAL)
        assert K_MIN <= k <= 1.0
        c2 = ProbabilityCalibrator(state_path=str(sp))
        assert abs(c2.get_sizing_multiplier(cat, TradeType.DIRECTIONAL) - k) < 1e-9


# =============================================================================
# CONCORRÊNCIA
# =============================================================================

class TestConcurrency:
    def test_concurrent_record_and_read_no_deadlock(self, ws_tmp):
        import random
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "cal.json"))
        cats, tts = list(MarketCategory), list(TradeType)
        errors: list = []
        barrier = threading.Barrier(8)

        def writer(tid):
            try:
                barrier.wait(timeout=5)
                rng = random.Random(tid)
                for _ in range(20):
                    cal.record_trade_result(rng.choice(cats), rng.choice(tts), rng.random(), rng.choice([0, 1]))
            except Exception as e:
                errors.append(e)

        def reader(tid):
            try:
                barrier.wait(timeout=5)
                for _ in range(30):
                    cal.get_sizing_multiplier(cats[tid % len(cats)], tts[tid % len(tts)])
                    cal.get_all_metrics()
                    cal.get_summary()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(4)]
        threads += [threading.Thread(target=reader, args=(i + 4,), daemon=True) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert not [t for t in threads if t.is_alive()], "Deadlock"
        assert not errors, f"Erros: {errors}"

    def test_concurrent_writes_preserve_consistency(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "cal.json"))
        cat, tt = MarketCategory.FINANCE, TradeType.ARBITRAGE
        per_thread, n_threads = 25, 4
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait(timeout=5)
            for i in range(per_thread):
                cal.record_trade_result(cat, tt, 0.6, i % 2)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert cal.get_metrics(cat, tt)["trade_count"] == per_thread * n_threads

    def test_save_state_does_not_hold_lock_during_io(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "cal.json"))
        for i in range(20):
            cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, i % 2)
        read_count = 0
        stop = threading.Event()

        def saver():
            for _ in range(50):
                if stop.is_set():
                    break
                cal.save_state()

        def reader():
            nonlocal read_count
            while not stop.is_set():
                cal.get_sizing_multiplier(MarketCategory.POLITICS, TradeType.DIRECTIONAL)
                read_count += 1

        ts = threading.Thread(target=saver, daemon=True)
        tr = threading.Thread(target=reader, daemon=True)
        tr.start(); ts.start()
        ts.join(timeout=10); stop.set(); tr.join(timeout=2)
        assert not ts.is_alive()
        assert read_count > 10

    def test_concurrent_save_state_produces_valid_json(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        cal = ProbabilityCalibrator(state_path=str(sp))
        for i in range(20):
            cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, i % 2)
        errors: list = []
        barrier = threading.Barrier(2)

        def saver():
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    cal.save_state()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=saver, daemon=True) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert not errors
        with open(sp) as f:
            assert json.load(f)["schema_version"] == SCHEMA_VERSION


# =============================================================================
# SCHEMA VERSIONING
# =============================================================================

class TestSchemaVersioning:
    def test_wrong_schema_version(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({"schema_version": 999, "created_at": "2026-01-01T00:00:00+00:00", "metrics": {}}))
        assert ProbabilityCalibrator(state_path=str(sp)).safety_mode is True

    def test_missing_schema_version(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({"metrics": {}, "created_at": "2026-01-01T00:00:00+00:00"}))
        assert ProbabilityCalibrator(state_path=str(sp)).safety_mode is True

    def test_correct_schema_loads(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "created_at": "2026-01-01T00:00:00+00:00",
            "metrics": {"POLITICS|DIRECTIONAL": {"bs_fast": 0.18, "bs_slow": 0.19, "trade_count": 20}}
        }))
        c = ProbabilityCalibrator(state_path=str(sp))
        assert c.safety_mode is False
        assert c.get_metrics(MarketCategory.POLITICS, TradeType.DIRECTIONAL)["trade_count"] == 20


# =============================================================================
# CRESCIMENTO DE ESTADO
# =============================================================================

class TestStateGrowth:
    def test_state_size_does_not_grow(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        c = ProbabilityCalibrator(state_path=str(sp))
        for i in range(10):
            c.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, i % 2)
        size_10 = sp.stat().st_size
        for i in range(990):
            c.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, i % 2)
        assert sp.stat().st_size < size_10 * 1.2


# =============================================================================
# DEEP COPY
# =============================================================================

class TestDeepCopy:
    def test_get_metrics_returns_copy(self, calibrator):
        calibrator.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.6, 1)
        m1 = calibrator.get_metrics(MarketCategory.POLITICS, TradeType.DIRECTIONAL)
        orig = m1["bs_fast"]
        m1["bs_fast"] = 999.0
        assert calibrator.get_metrics(MarketCategory.POLITICS, TradeType.DIRECTIONAL)["bs_fast"] == orig

    def test_get_all_metrics_returns_copy(self, calibrator):
        calibrator.record_trade_result(MarketCategory.CRYPTO, TradeType.ARBITRAGE, 0.5, 1)
        all_m = calibrator.get_all_metrics()
        orig = all_m["CRYPTO/ARBITRAGE"]["bs_fast"]
        all_m["CRYPTO/ARBITRAGE"]["bs_fast"] = 999.0
        assert calibrator.get_all_metrics()["CRYPTO/ARBITRAGE"]["bs_fast"] == orig


# =============================================================================
# VALORES NEGATIVOS
# =============================================================================

class TestNegativeValueValidation:
    def test_negative_bs_fast(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "created_at": "2026-01-01T00:00:00+00:00",
            "metrics": {"POLITICS|DIRECTIONAL": {"bs_fast": -0.1, "bs_slow": 0.2, "trade_count": 10}}
        }))
        assert ProbabilityCalibrator(state_path=str(sp)).safety_mode is True

    def test_negative_trade_count(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "created_at": "2026-01-01T00:00:00+00:00",
            "metrics": {"CRYPTO|ARBITRAGE": {"bs_fast": 0.2, "bs_slow": 0.2, "trade_count": -5}}
        }))
        assert ProbabilityCalibrator(state_path=str(sp)).safety_mode is True

    def test_valid_values(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "created_at": "2026-01-01T00:00:00+00:00",
            "metrics": {"FINANCE|HEDGING": {"bs_fast": 0.15, "bs_slow": 0.16, "trade_count": 50}}
        }))
        c = ProbabilityCalibrator(state_path=str(sp))
        assert c.safety_mode is False
        assert c.get_metrics(MarketCategory.FINANCE, TradeType.HEDGING)["trade_count"] == 50


# =============================================================================
# SAFETY MODE FAIL-CLOSED
# =============================================================================

class TestSafetyModeFailClosed:
    def test_overrides_good_metrics(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "cal.json"))
        for _ in range(N_MIN + 10):
            cal.record_trade_result(MarketCategory.FINANCE, TradeType.ARBITRAGE, 0.9, 1)
        assert cal.get_sizing_multiplier(MarketCategory.FINANCE, TradeType.ARBITRAGE) == 1.0
        cal._safety_mode = True
        assert cal.get_sizing_multiplier(MarketCategory.FINANCE, TradeType.ARBITRAGE) == K_MIN_GLOBAL

    def test_overrides_n_min(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "cal.json"))
        for _ in range(N_MIN - 1):
            cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, 0)
        assert cal.get_sizing_multiplier(MarketCategory.POLITICS, TradeType.DIRECTIONAL) == 1.0
        cal._safety_mode = True
        assert cal.get_sizing_multiplier(MarketCategory.POLITICS, TradeType.DIRECTIONAL) == K_MIN_GLOBAL

    def test_applies_to_all_categories(self, ws_tmp):
        cal = ProbabilityCalibrator(state_path=str(ws_tmp / "cal.json"))
        cal._safety_mode = True
        for cat in MarketCategory:
            for tt in TradeType:
                assert cal.get_sizing_multiplier(cat, tt) == K_MIN_GLOBAL


# =============================================================================
# TIMESTAMPS
# =============================================================================

class TestTimestamps:
    def test_first_save_timestamps_equal(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        cal = ProbabilityCalibrator(state_path=str(sp))
        cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, 1)
        with open(sp) as f:
            data = json.load(f)
        assert data["created_at"] == data["updated_at"]

    def test_updated_at_changes_created_at_stays(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        cal = ProbabilityCalibrator(state_path=str(sp))
        cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, 1)
        with open(sp) as f:
            d1 = json.load(f)
        time.sleep(0.01)
        cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.6, 0)
        with open(sp) as f:
            d2 = json.load(f)
        assert d2["created_at"] == d1["created_at"]
        assert d2["updated_at"] >= d1["updated_at"]

    def test_created_at_preserved_across_reload(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        c1 = ProbabilityCalibrator(state_path=str(sp))
        c1.record_trade_result(MarketCategory.CRYPTO, TradeType.ARBITRAGE, 0.5, 1)
        with open(sp) as f:
            orig = json.load(f)["created_at"]
        time.sleep(0.01)
        c2 = ProbabilityCalibrator(state_path=str(sp))
        c2.record_trade_result(MarketCategory.CRYPTO, TradeType.ARBITRAGE, 0.6, 0)
        with open(sp) as f:
            assert json.load(f)["created_at"] == orig

    def test_missing_created_at(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "updated_at": "2026-01-01T00:00:00+00:00", "metrics": {}}))
        assert ProbabilityCalibrator(state_path=str(sp)).safety_mode is True

    def test_timestamps_are_utc(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        cal = ProbabilityCalibrator(state_path=str(sp))
        cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, 1)
        with open(sp) as f:
            data = json.load(f)
        for field in ("created_at", "updated_at"):
            assert "+00:00" in data[field]

    def test_reset_all_resets_created_at(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        cal = ProbabilityCalibrator(state_path=str(sp))
        cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, 1)
        with open(sp) as f:
            orig = json.load(f)["created_at"]
        time.sleep(0.01)
        cal.reset_all()
        with open(sp) as f:
            assert json.load(f)["created_at"] != orig


# =============================================================================
# DIR FSYNC
# =============================================================================

class TestDirFsync:
    def test_save_state_executes_without_error(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        cal = ProbabilityCalibrator(state_path=str(sp))
        for i in range(5):
            cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, i % 2)
        assert cal.safety_mode is False
        with open(sp) as f:
            assert json.load(f)["schema_version"] == SCHEMA_VERSION

    def test_multiple_saves_with_dir_fsync(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        cal = ProbabilityCalibrator(state_path=str(sp))
        cal.record_trade_result(MarketCategory.CRYPTO, TradeType.ARBITRAGE, 0.5, 1)
        for _ in range(20):
            cal.save_state()
        assert cal.safety_mode is False


# =============================================================================
# STATE CORRUPTED
# =============================================================================

class TestStateCorrupted:
    def test_corrupted_file_sets_state_corrupted(self, ws_tmp):
        sp = ws_tmp / "bad.json"
        sp.write_text("not valid json")
        cal = ProbabilityCalibrator(state_path=str(sp))
        assert cal.safety_mode is True
        assert cal.state_corrupted is True

    def test_save_state_refuses_when_corrupted(self, ws_tmp):
        sp = ws_tmp / "bad.json"
        original = "this is corrupted!!!"
        sp.write_text(original)
        cal = ProbabilityCalibrator(state_path=str(sp))
        for i in range(5):
            cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, i % 2)
        assert sp.read_text() == original

    def test_save_state_explicit_call_also_refused(self, ws_tmp):
        sp = ws_tmp / "bad.json"
        sp.write_text("{invalid")
        cal = ProbabilityCalibrator(state_path=str(sp))
        cal.save_state()
        assert sp.read_text() == "{invalid"

    def test_reset_all_restores_persistence(self, ws_tmp):
        sp = ws_tmp / "bad.json"
        sp.write_text("corrupted!!!")
        cal = ProbabilityCalibrator(state_path=str(sp))
        assert cal.state_corrupted is True
        cal.reset_all()
        assert cal.state_corrupted is False
        assert cal.safety_mode is False
        with open(sp) as f:
            assert json.load(f)["schema_version"] == SCHEMA_VERSION

    def test_recover_from_corruption_restores_persistence(self, ws_tmp):
        sp = ws_tmp / "bad.json"
        sp.write_text("corrupted!!!")
        cal = ProbabilityCalibrator(state_path=str(sp))
        cal.recover_from_corruption()
        assert cal.state_corrupted is False
        with open(sp) as f:
            assert json.load(f)["schema_version"] == SCHEMA_VERSION

    def test_valid_file_does_not_set_corrupted(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "created_at": "2026-01-01T00:00:00+00:00", "metrics": {}
        }))
        cal = ProbabilityCalibrator(state_path=str(sp))
        assert cal.state_corrupted is False


# =============================================================================
# CORRUPTION MODE — FAIL-CLOSED NO SIZING (NOVO)
# =============================================================================

class TestCorruptionMode:
    """
    Valida que state_corrupted implica K_MIN_GLOBAL em get_sizing_multiplier(),
    trades em memória continuam, e warning é emitido uma vez por sessão.
    """

    def test_corrupted_always_returns_k_min_global(self, ws_tmp):
        """state_corrupted → K_MIN_GLOBAL independente de métricas."""
        sp = ws_tmp / "bad.json"
        sp.write_text("corrupted")
        cal = ProbabilityCalibrator(state_path=str(sp))
        assert cal.is_corrupted() is True

        # Mesmo após N_MIN trades perfeitos:
        for _ in range(N_MIN + 10):
            cal.record_trade_result(MarketCategory.FINANCE, TradeType.ARBITRAGE, 0.9, 1)

        # safety_mode pode ter sido limpo, mas state_corrupted persiste
        assert cal.state_corrupted is True
        assert cal.get_sizing_multiplier(MarketCategory.FINANCE, TradeType.ARBITRAGE) == K_MIN_GLOBAL

    def test_trade_count_continues_in_memory(self, ws_tmp):
        """record_trade_result funciona em memória durante corrupção."""
        sp = ws_tmp / "bad.json"
        sp.write_text("corrupted")
        cal = ProbabilityCalibrator(state_path=str(sp))

        for _ in range(25):
            cal.record_trade_result(MarketCategory.CRYPTO, TradeType.DIRECTIONAL, 0.7, 1)

        m = cal.get_metrics(MarketCategory.CRYPTO, TradeType.DIRECTIONAL)
        assert m is not None
        assert m["trade_count"] == 25

    def test_is_corrupted_method(self, ws_tmp):
        """is_corrupted() retorna bool correto."""
        sp_good = ws_tmp / "good.json"
        cal_good = ProbabilityCalibrator(state_path=str(sp_good))
        assert cal_good.is_corrupted() is False

        sp_bad = ws_tmp / "bad.json"
        sp_bad.write_text("corrupted")
        cal_bad = ProbabilityCalibrator(state_path=str(sp_bad))
        assert cal_bad.is_corrupted() is True

    def test_warning_emitted_once(self, ws_tmp, caplog):
        """Warning sobre corrupção emitido apenas uma vez por sessão."""
        sp = ws_tmp / "bad.json"
        sp.write_text("corrupted")
        cal = ProbabilityCalibrator(state_path=str(sp))

        with caplog.at_level(logging.WARNING, logger="src.engine.probability_calibrator"):
            caplog.clear()
            cal.get_sizing_multiplier(MarketCategory.POLITICS, TradeType.DIRECTIONAL)
            cal.get_sizing_multiplier(MarketCategory.CRYPTO, TradeType.ARBITRAGE)
            cal.get_sizing_multiplier(MarketCategory.FINANCE, TradeType.HEDGING)

        corruption_warnings = [
            r for r in caplog.records
            if "corrompido" in r.message and r.levelno == logging.WARNING
        ]
        assert len(corruption_warnings) == 1

    def test_corrupted_overrides_all_categories(self, ws_tmp):
        """state_corrupted → K_MIN_GLOBAL para toda combinação categoria/tipo."""
        sp = ws_tmp / "bad.json"
        sp.write_text("corrupted")
        cal = ProbabilityCalibrator(state_path=str(sp))
        for cat in MarketCategory:
            for tt in TradeType:
                assert cal.get_sizing_multiplier(cat, tt) == K_MIN_GLOBAL


# =============================================================================
# os.replace E MESMO DIRETÓRIO (NOVO)
# =============================================================================

class TestOsReplace:
    def test_tmp_and_final_in_same_directory(self, ws_tmp, monkeypatch):
        """tmp file é criado no mesmo diretório do arquivo final."""
        sp = ws_tmp / "subdir" / "cal.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        cal = ProbabilityCalibrator(state_path=str(sp))

        original_replace = os.replace
        captured_paths: list = []

        def mock_replace(src, dst):
            captured_paths.append((src, dst))
            return original_replace(src, dst)

        monkeypatch.setattr(os, "replace", mock_replace)
        cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, 1)

        assert len(captured_paths) > 0
        for src, dst in captured_paths:
            assert os.path.dirname(os.path.abspath(src)) == os.path.dirname(os.path.abspath(dst)), (
                f"tmp={src} e final={dst} não estão no mesmo diretório"
            )

    def test_no_shutil_import(self):
        """Módulo calibrator não importa shutil (usa os.replace)."""
        import src.engine.probability_calibrator as mod
        # shutil não deve existir no namespace do módulo
        assert not hasattr(mod, "shutil"), "shutil não deveria ser importado pelo módulo"


# =============================================================================
# UTC VALIDATION
# =============================================================================

class TestUTCValidation:
    def test_created_at_without_timezone(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "created_at": "2026-01-01T00:00:00", "metrics": {}
        }))
        cal = ProbabilityCalibrator(state_path=str(sp))
        assert cal.safety_mode is True
        assert cal.state_corrupted is True

    def test_created_at_with_non_utc_timezone(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "created_at": "2026-01-01T00:00:00+05:30", "metrics": {}
        }))
        cal = ProbabilityCalibrator(state_path=str(sp))
        assert cal.safety_mode is True

    def test_created_at_with_z_suffix(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "created_at": "2026-01-01T00:00:00Z", "metrics": {}
        }))
        assert ProbabilityCalibrator(state_path=str(sp)).safety_mode is False

    def test_updated_at_always_has_utc_offset(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        cal = ProbabilityCalibrator(state_path=str(sp))
        cal.record_trade_result(MarketCategory.POLITICS, TradeType.DIRECTIONAL, 0.5, 1)
        with open(sp) as f:
            assert json.load(f)["updated_at"].endswith("+00:00")

    def test_valid_utc_timestamp(self, ws_tmp):
        sp = ws_tmp / "cal.json"
        sp.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "created_at": "2026-02-05T12:30:45.123456+00:00", "metrics": {}
        }))
        assert ProbabilityCalibrator(state_path=str(sp)).safety_mode is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
