"""
Tests — Reversion Engine V1
===========================

Garante:
- z aumenta quando desvio aumenta
- Zero shock → sem sinal
- Grande shock positivo → SELL
- Grande shock negativo → BUY (se safe_to_buy)
- p_est sempre clampado
- Variância nunca negativa
- Sem propagação de NaN
- Spread filter bloqueia sinal
- Liquidity filter bloqueia sinal
- Comportamento determinístico
- Não modifica core financeiro
"""

import math
import time

from src.strategy.reversion_engine import (
    ReversionConfig,
    ReversionEngine,
    ReversionMetricsCollector,
    ReversionState,
    create_default_config,
)


def _config() -> ReversionConfig:
    return create_default_config()


def _initial_state(mid: float = 0.5) -> ReversionState:
    return ReversionState(
        ewma_mean=mid,
        ewma_var=0.01,
        last_mid=mid,
        last_timestamp=0.0,
    )


# --- z increases when deviation increases ---


def test_z_increases_when_deviation_increases():
    """z aumenta quando o desvio do mid em relação à média aumenta."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = _initial_state(0.5)
    spread = 0.01
    liq = 200.0

    s1 = engine.generate_signal(state, 0.52, spread, liq, 0.95, True)
    s2 = engine.generate_signal(state, 0.60, spread, liq, 0.95, True)
    s3 = engine.generate_signal(state, 0.70, spread, liq, 0.95, True)

    assert s1.z_score < s2.z_score < s3.z_score


# --- Zero shock → no signal ---


def test_zero_shock_no_signal():
    """mid = ewma_mean → shock=0, z≈0, side=None."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = _initial_state(0.5)
    sig = engine.generate_signal(state, 0.5, 0.01, 200.0, 0.95, True)
    assert abs(sig.shock) < 1e-6
    assert abs(sig.z_score) < 1.0
    assert sig.side is None


# --- Large positive shock → SELL ---


def test_large_positive_shock_sell():
    """Grande desvio positivo → side=SELL."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = ReversionState(ewma_mean=0.3, ewma_var=0.0001, last_mid=0.3, last_timestamp=0.0)
    mid = 0.6
    sigma = math.sqrt(0.0001)
    z_expected = (mid - 0.3) / sigma
    assert z_expected >= 2.3
    sig = engine.generate_signal(state, mid, 0.01, 200.0, 0.95, True)
    assert sig.side == "SELL"
    assert sig.z_score >= 2.3


# --- Large negative shock → BUY ---


def test_large_negative_shock_buy_when_safe():
    """Grande desvio negativo + safe_to_buy → side=BUY."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = ReversionState(ewma_mean=0.7, ewma_var=0.0001, last_mid=0.7, last_timestamp=0.0)
    mid = 0.4
    sig = engine.generate_signal(state, mid, 0.01, 200.0, 0.95, safe_to_buy=True)
    assert sig.side == "BUY"
    assert sig.z_score <= -2.3


def test_large_negative_shock_no_buy_when_unsafe():
    """Grande desvio negativo + safe_to_buy=False → side=None (fail closed)."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = ReversionState(ewma_mean=0.7, ewma_var=0.0001, last_mid=0.7, last_timestamp=0.0)
    mid = 0.4
    sig = engine.generate_signal(state, mid, 0.01, 200.0, 0.95, safe_to_buy=False)
    assert sig.side is None


# --- p_est always clamped ---


def test_p_est_always_clamped():
    """p_est sempre em [0.01, 0.99]."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    for mid, ewma in [(0.99, 0.5), (0.01, 0.5), (0.0, 0.5), (1.0, 0.5)]:
        state = ReversionState(ewma_mean=ewma, ewma_var=0.01, last_mid=ewma, last_timestamp=0.0)
        sig = engine.generate_signal(state, mid, 0.01, 200.0, 0.95, True)
        assert 0.01 <= sig.p_est <= 0.99


# --- Variance never negative ---


def test_variance_never_negative():
    """update_state garante var >= epsilon."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = ReversionState(ewma_mean=0.5, ewma_var=0.0, last_mid=0.5, last_timestamp=0.0)
    new_state = engine.update_state(state, 0.5)
    assert new_state.ewma_var >= cfg.epsilon


# --- No NaN propagation ---


def test_no_nan_propagation():
    """mid=NaN ou inf → retorna sinal neutro, sem NaN no output."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = _initial_state(0.5)
    for bad_mid in [float("nan"), float("inf"), float("-inf")]:
        sig = engine.generate_signal(state, bad_mid, 0.01, 200.0, 0.95, True)
        assert sig.side is None
        assert math.isfinite(sig.p_est)
        assert math.isfinite(sig.edge_estimate)
        assert math.isfinite(sig.z_score)


def test_update_state_preserves_state_on_nan():
    """update_state com mid NaN mantém estado inalterado."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = _initial_state(0.5)
    out = engine.update_state(state, float("nan"))
    assert out.ewma_mean == state.ewma_mean
    assert out.ewma_var == state.ewma_var


# --- Spread filter ---


def test_spread_filter_blocks_signal():
    """spread < min_spread ou spread > max_spread → neutro."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = ReversionState(ewma_mean=0.3, ewma_var=0.0001, last_mid=0.3, last_timestamp=0.0)
    mid = 0.6
    sig_low = engine.generate_signal(state, mid, 0.001, 200.0, 0.95, True)
    sig_high = engine.generate_signal(state, mid, 0.06, 200.0, 0.95, True)
    assert sig_low.side is None
    assert sig_high.side is None


# --- Liquidity filter ---


def test_liquidity_filter_blocks_signal():
    """liquidity < min_liquidity → neutro."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = ReversionState(ewma_mean=0.3, ewma_var=0.0001, last_mid=0.3, last_timestamp=0.0)
    mid = 0.6
    sig = engine.generate_signal(state, mid, 0.01, 50.0, 0.95, True)
    assert sig.side is None


# --- Deterministic ---


def test_deterministic_same_inputs_same_output():
    """Mesmo stream de input → mesmo output."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = ReversionState(ewma_mean=0.5, ewma_var=0.01, last_mid=0.5, last_timestamp=0.0)
    inputs = [(0.52, 0.01, 200.0), (0.55, 0.012, 250.0), (0.48, 0.01, 200.0)]
    outputs1 = [
        engine.generate_signal(state, mid, sp, liq, 0.9, True)
        for mid, sp, liq in inputs
    ]
    outputs2 = [
        engine.generate_signal(state, mid, sp, liq, 0.9, True)
        for mid, sp, liq in inputs
    ]
    for a, b in zip(outputs1, outputs2):
        assert a.z_score == b.z_score
        assert a.side == b.side
        assert a.p_est == b.p_est
        assert a.confidence == b.confidence


# --- Rationale ---


def test_rationale_includes_z_spread_liquidity_threshold():
    """rationale inclui z, spread, liquidity e threshold."""
    cfg = _config()
    engine = ReversionEngine(cfg)
    state = ReversionState(ewma_mean=0.3, ewma_var=0.0001, last_mid=0.3, last_timestamp=0.0)
    sig = engine.generate_signal(state, 0.6, 0.02, 300.0, 0.95, True)
    rationale_text = " ".join(sig.rationale)
    assert "z=" in rationale_text
    assert "spread=" in rationale_text
    assert "liquidity=" in rationale_text
    assert "entry" in rationale_text or "threshold" in rationale_text.lower()


# --- Metrics collector ---


def test_metrics_collector_aggregates():
    """ReversionMetricsCollector agrega sinais sem tocar core."""
    coll = ReversionMetricsCollector(window_seconds=86400)
    now = time.time()
    coll.record_signal(1.5, "SELL", 0.45, 0.50, now - 1.0)
    coll.record_signal(-2.0, "BUY", 0.55, 0.50, now - 0.5)
    m = coll.get_metrics()
    assert m["signal_count_last_24h"] == 2
    assert m["rolling_z_mean"] is not None
    assert m["rolling_z_std"] is not None
    assert m["average_reversion_magnitude"] is not None


def test_metrics_collector_empty_returns_defaults():
    """Métricas vazias retornam defaults."""
    coll = ReversionMetricsCollector(window_seconds=86400)
    m = coll.get_metrics()
    assert m["signal_count_last_24h"] == 0
    assert m["current_z"] is None


# --- Default config ---


def test_default_config_values():
    """Config padrão tem valores OBSERVE corretos."""
    cfg = create_default_config()
    assert cfg.alpha == 0.06
    assert cfg.z_entry_threshold == 2.3
    assert cfg.z_exit_threshold == 1.0
    assert cfg.k_revert == 0.25
    assert cfg.min_spread == 0.002
    assert cfg.max_spread == 0.05
    assert cfg.min_liquidity == 100.0
    assert cfg.max_ttl_seconds == 3600
    assert cfg.epsilon == 1e-9
