"""
Tests — Risk Refinement Pass
==============================

Garante:
- Risco gradual sem saltos abruptos
- SAFE_MODE domina o score
- Drawdown persistente e reset em corrupção
- Narrativa alinhada (confidence vs risk)
- Estrutura do payload mantida
"""

import os
import time

import pytest
from fastapi.testclient import TestClient

from src.api.drawdown_tracker import (
    DrawdownTracker,
    _compute_content_hash,
    _hash_material,
)
from src.api.risk_model import (
    RiskInputs,
    build_risk_explanation,
    compute_risk_score,
    risk_level_from_score,
)
from src.api.dashboard_server import create_dashboard_app
from src.api.dashboard_metrics import apply_narrative_alignment
from src.config.config import AppConfig
from src.risk.capital_manager import CapitalManager
from src.risk.position_manager import PositionManager
from src.safety.safety_state import SafetyState
from src.engine.trade_ledger import TradeLedger
from src.engine.probability_calibrator import ProbabilityCalibrator
from src.safety.audit_report import build_audit_report


def _make_system(tmp_path, capital: float = 1000.0) -> dict:
    tmpdir = str(tmp_path)
    cfg = AppConfig(
        run_mode="OBSERVE",
        reconcile_loop_interval_seconds=30,
        reconcile_stale_seconds=60,
        queue_maxsize=100,
        trade_drop_strikes_to_clear=3,
        invariant_strikes_to_clear=3,
        capital_mismatch_percent_tolerance=0.001,
        capital_mismatch_abs_tolerance=0.01,
        health_check_interval_seconds=30,
    )
    cm = CapitalManager(initial_capital=capital)
    cal = ProbabilityCalibrator(state_path=f"{tmpdir}/cal.json")
    pm = PositionManager(state_path=f"{tmpdir}/pos.json")
    tl = TradeLedger(
        state_path=f"{tmpdir}/ledger.json",
        calibrator=cal,
        capital_manager=cm,
    )
    ss = SafetyState()
    return {
        "system_start_time": time.time(),
        "state_dir": tmpdir,
        "safety_state": ss,
        "capital_manager": cm,
        "position_manager": pm,
        "trade_ledger": tl,
        "config": cfg,
        "config_snapshot": None,
        "ack_store": None,
        "reconciler_loop": None,
        "market_engine": None,
        "exchange_api": None,
        "build_audit_report": build_audit_report,
        "watcher_thread": None,
    }


# --- risk_model ---


def test_risk_score_gradual_scaling_no_abrupt_jump():
    """Score sobe gradualmente; sem salto de Baixo para Elevado por um passo pequeno."""
    # percent_reserved: 0.19 -> 0.21 não deve pular de Baixo para Elevado
    base = RiskInputs(
        percent_reserved=0.0,
        max_concentration=0.0,
        open_positions=0,
        global_safe_mode=False,
    )
    s0 = compute_risk_score(base)
    assert risk_level_from_score(s0)["level"] == "Baixo"

    # 0.19
    r1 = RiskInputs(
        percent_reserved=0.19,
        max_concentration=0.0,
        open_positions=0,
        global_safe_mode=False,
    )
    s1 = compute_risk_score(r1)
    l1 = risk_level_from_score(s1)["level"]

    # 0.21
    r2 = RiskInputs(
        percent_reserved=0.21,
        max_concentration=0.0,
        open_positions=0,
        global_safe_mode=False,
    )
    s2 = compute_risk_score(r2)
    l2 = risk_level_from_score(s2)["level"]

    assert s2 > s1
    assert s2 - s1 < 5
    assert l1 == l2 or (l1, l2) in [("Baixo", "Moderado")]


def test_risk_safe_mode_dominates():
    """global_safe_mode True adiciona +40 fixo ao score."""
    base = RiskInputs(
        percent_reserved=0.0,
        max_concentration=0.0,
        open_positions=0,
        global_safe_mode=False,
    )
    safe = RiskInputs(
        percent_reserved=0.0,
        max_concentration=0.0,
        open_positions=0,
        global_safe_mode=True,
    )
    assert compute_risk_score(safe) >= 40
    assert compute_risk_score(safe) == compute_risk_score(base) + 40


# --- drawdown_tracker ---


def test_drawdown_tracker_persists_max_across_restart(tmp_path):
    """Tracker persiste max; reinstanciar preserva."""
    state_dir = str(tmp_path / "dd_state")
    os.makedirs(state_dir, exist_ok=True)

    t1 = DrawdownTracker(state_dir=state_dir)
    t1.update(1500.0)
    dd1 = t1.get_drawdown(1400.0)
    assert dd1["capital_max"] == 1500.0
    assert dd1["drawdown_percent"] > 0

    t2 = DrawdownTracker(state_dir=state_dir)
    dd2 = t2.get_drawdown(1400.0)
    assert dd2["capital_max"] == 1500.0


def test_drawdown_tracker_resets_on_corruption(tmp_path):
    """Arquivo corrompido causa reset seguro sem exception."""
    state_dir = str(tmp_path / "dd_corrupt")
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, "dashboard_state.json")

    t1 = DrawdownTracker(state_dir=state_dir)
    t1.update(2000.0)
    assert t1.get_drawdown(1000.0)["capital_max"] == 2000.0

    with open(path, "w") as f:
        f.write("{ invalid json")
    t2 = DrawdownTracker(state_dir=state_dir)
    dd = t2.get_drawdown(1500.0)
    assert dd["capital_max"] == 1500.0


# --- narrative alignment ---


def test_narrative_alignment_confidence_high_risk_high():
    """Quando confidence alta e risk elevado, frase de harmonização."""
    base = "Monitorando mercados normalmente."
    conf = {"score": 85, "level": "Alta"}
    risk = {"level": "Elevado"}
    out = apply_narrative_alignment(base, conf, risk)
    assert "exposição financeira elevada" in out


def test_content_hash_ignores_timestamps(tmp_path):
    """content_hash não inclui timestamps — mesmo max gera mesmo hash."""
    state_dir = str(tmp_path / "dd_hash")
    os.makedirs(state_dir, exist_ok=True)
    h1 = _compute_content_hash(schema_version=1, max_capital_seen=1500.0)
    h2 = _compute_content_hash(schema_version=1, max_capital_seen=1500.0)
    assert h1 == h2
    h3 = _compute_content_hash(schema_version=1, max_capital_seen=1501.0)
    assert h1 != h3
    material = _hash_material(1, 1500.0)
    assert "created_at" not in material
    assert "updated_at" not in material


def test_drawdown_tracker_merge_on_concurrent_updates(tmp_path):
    """Dois updates sequenciais com leitura intermediária: max final é o maior."""
    state_dir = str(tmp_path / "dd_merge")
    os.makedirs(state_dir, exist_ok=True)
    t1 = DrawdownTracker(state_dir=state_dir)
    t1.update(1000.0)
    t1.update(1200.0)
    dd1 = t1.get_drawdown(1100.0)
    assert dd1["capital_max"] == 1200.0
    t2 = DrawdownTracker(state_dir=state_dir)
    t2.update(1500.0)
    dd2 = t2.get_drawdown(1400.0)
    assert dd2["capital_max"] == 1500.0


def test_drawdown_tracker_ignores_nan_inf_negative(tmp_path):
    """Update com NaN/inf/negativo não altera max."""
    state_dir = str(tmp_path / "dd_sanity")
    os.makedirs(state_dir, exist_ok=True)
    t = DrawdownTracker(state_dir=state_dir)
    t.update(2000.0)
    assert t.get_drawdown(1000.0)["capital_max"] == 2000.0
    t.update(float("nan"))
    assert t.get_drawdown(1000.0)["capital_max"] == 2000.0
    t.update(float("inf"))
    assert t.get_drawdown(1000.0)["capital_max"] == 2000.0
    t.update(-100.0)
    assert t.get_drawdown(1000.0)["capital_max"] == 2000.0
    t.update(0.0)
    assert t.get_drawdown(1000.0)["capital_max"] == 2000.0


def test_timeline_risk_level_debounce(tmp_path):
    """Score muda mas level não muda → não registra risk_level_changed."""
    system = _make_system(tmp_path)
    app = create_dashboard_app(system)
    client = TestClient(app)
    r1 = client.get("/api/dashboard")
    assert r1.status_code == 200
    d1 = r1.json()
    risk_events_before = [
        e for e in d1.get("recent_events_v2", [])
        if "Nível de risco" in (e.get("text") or "")
    ]
    r2 = client.get("/api/dashboard")
    d2 = r2.json()
    risk_events_after = [
        e for e in d2.get("recent_events_v2", [])
        if "Nível de risco" in (e.get("text") or "")
    ]
    assert len(risk_events_after) == len(risk_events_before), (
        "Sem mudança de level não deve adicionar risk_level_changed"
    )


def test_payload_structure_unchanged_for_dashboard(tmp_path):
    """Payload /api/dashboard mantém chaves esperadas."""
    system = _make_system(tmp_path)
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/dashboard")
    assert r.status_code == 200
    d = r.json()
    for key in (
        "run_mode", "human_status", "buy_blocked", "capital",
        "positions", "exposure_summary", "recent_events", "explanation_text",
        "confidence_score", "financial_risk", "concentration", "drawdown",
        "uptime_seconds",
    ):
        assert key in d
    assert "score" in d["financial_risk"]
    assert "level" in d["financial_risk"]
    assert "description" in d["financial_risk"]
    assert "model_version" in d["financial_risk"]
