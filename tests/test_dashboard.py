"""
Tests — Dashboard API
========================

Cobrindo:
- GET /api/status retorna buy_blocked corretamente
- Human reasons traduzidas
- POST /api/ack chama ack_reason
- Não permite ACK inválido
- /api/capital retorna valores corretos
- /api/positions funciona
"""

import tempfile
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.dashboard_server import (
    HUMAN_REASON_MAP,
    create_dashboard_app,
)
from src.api.dashboard_metrics import (
    build_explanation_text,
    compute_confidence_score,
)
from src.config.config import AppConfig, load_app_config_from_env
from src.risk.capital_manager import CapitalManager
from src.risk.position_manager import PositionManager
from src.safety.safety_state import SafetyState, CONFIG_CHANGED, RECONCILE_STALE
from src.engine.trade_ledger import TradeLedger
from src.engine.probability_calibrator import ProbabilityCalibrator
from src.safety.audit_report import build_audit_report
from src.safety.safety_ack import SafetyAckStore


def _make_system(
    buy_blocked: bool = False,
    reasons: set = None,
    capital: float = 1000.0,
    positions: list = None,
) -> dict:
    """Sistema mínimo para testes do dashboard."""
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
    tmpdir = tempfile.mkdtemp(prefix="test_dashboard_")
    cal = ProbabilityCalibrator(state_path=f"{tmpdir}/cal.json")
    pm = PositionManager(state_path=f"{tmpdir}/pos.json")
    tl = TradeLedger(
        state_path=f"{tmpdir}/ledger.json",
        calibrator=cal,
        capital_manager=cm,
    )
    ss = SafetyState()
    for r in reasons or []:
        ss.block_buy(r)

    for p in (positions or []):
        pm.apply_fill(
            p["market_id"],
            p["side"],
            p["shares"],
            p["avg_price"],
            "BUY",
        )

    ack_store = SafetyAckStore(state_path=f"{tmpdir}/ack.json")

    system = {
        "system_start_time": time.time(),
        "safety_state": ss,
        "capital_manager": cm,
        "position_manager": pm,
        "trade_ledger": tl,
        "config": cfg,
        "config_snapshot": None,
        "ack_store": ack_store,
        "reconciler_loop": None,
        "market_engine": None,
        "exchange_api": None,
        "build_audit_report": build_audit_report,
        "watcher_thread": None,
    }
    return system


def test_api_status_returns_buy_blocked():
    """GET /api/status retorna buy_blocked corretamente."""
    system = _make_system(buy_blocked=True, reasons={CONFIG_CHANGED})
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["buy_blocked"] is True
    assert CONFIG_CHANGED in data["reasons"]


def test_api_status_buy_not_blocked():
    """GET /api/status retorna buy_blocked=False quando sem reasons."""
    system = _make_system(buy_blocked=False, reasons=set())
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["buy_blocked"] is False
    assert data["reasons"] == []


def test_human_reasons_translated():
    """Human reasons estão traduzidas (não exibe nomes técnicos)."""
    system = _make_system(reasons={RECONCILE_STALE, CONFIG_CHANGED})
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    human = data["human_reasons"]
    assert len(human) == 2
    assert HUMAN_REASON_MAP["RECONCILE_STALE"] in human
    assert HUMAN_REASON_MAP["CONFIG_CHANGED"] in human
    assert "RECONCILE_STALE" not in human


def test_api_ack_calls_ack_reason():
    """POST /api/ack chama ack_reason quando reason ativa."""
    system = _make_system(reasons={CONFIG_CHANGED})
    app = create_dashboard_app(system)
    client = TestClient(app)

    with patch("src.orchestrator.ack_reason") as mock_ack:
        mock_ack.return_value = True
        r = client.post(
            "/api/ack",
            json={"reason": CONFIG_CHANGED, "note": "validado"},
        )
        assert r.status_code == 200
        mock_ack.assert_called_once()
        args = mock_ack.call_args
        assert args[0][0] == CONFIG_CHANGED
        assert args[0][1] == "validado"
        assert args[0][2] == system


def test_api_ack_rejects_invalid_reason():
    """Não permite ACK de reason que não está ativa."""
    system = _make_system(reasons={RECONCILE_STALE})
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.post(
        "/api/ack",
        json={"reason": CONFIG_CHANGED, "note": "x"},
    )
    assert r.status_code == 400
    assert "não está ativa" in r.json()["detail"]


def test_api_ack_rejects_empty_reason():
    """Não permite ACK sem reason."""
    system = _make_system(reasons={CONFIG_CHANGED})
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.post("/api/ack", json={"reason": "", "note": "x"})
    assert r.status_code == 400
    assert "obrigatório" in r.json()["detail"]


def test_api_capital_returns_correct_values():
    """/api/capital retorna valores corretos."""
    system = _make_system(capital=2500.0)
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/capital")
    assert r.status_code == 200
    data = r.json()
    assert data["total_capital"] == 2500.0
    assert data["available_capital"] == 2500.0
    assert data["locked_in_open_orders"] == 0.0
    assert data["locked_in_unresolved_markets"] == 0.0
    assert data["realized_pnl"] == 0.0


def test_api_positions_works():
    """/api/positions funciona."""
    system = _make_system(
        positions=[
            {"market_id": "m1", "side": "YES", "shares": 100.0, "avg_price": 0.5},
            {"market_id": "m2", "side": "NO", "shares": 50.0, "avg_price": 0.3},
        ]
    )
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/positions")
    assert r.status_code == 200
    data = r.json()
    pos = data["positions"]
    assert len(pos) == 2
    by_market = {p["market"]: p for p in pos}
    assert "m1" in by_market
    assert by_market["m1"]["side"] == "YES"
    assert by_market["m1"]["shares"] == 100.0
    assert by_market["m1"]["avg_entry_price"] == 0.5
    assert "m2" in by_market
    assert by_market["m2"]["side"] == "NO"


def test_api_audit_returns_report():
    """GET /api/audit retorna relatório de auditoria."""
    system = _make_system()
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/audit")
    assert r.status_code == 200
    data = r.json()
    assert "report" in data
    assert "AUDIT REPORT" in data["report"]
    assert "SAFETY STATE" in data["report"]
    assert "CAPITAL" in data["report"]


def test_get_root_returns_html():
    """GET / retorna dashboard HTML."""
    system = _make_system()
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Trading Robot" in r.text or "Painel do Robo" in r.text


def test_api_dashboard_payload_structure():
    """GET /api/dashboard retorna payload com estrutura esperada."""
    system = _make_system(capital=1500.0)
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/dashboard")
    assert r.status_code == 200
    d = r.json()
    assert "run_mode" in d
    assert "human_status" in d
    assert "buy_blocked" in d
    assert "global_safe_mode" in d
    assert "last_reconcile_age_seconds" in d
    assert "invariants_ok_streak" in d
    assert "capital_mismatch_ok_streak" in d
    assert "capital" in d
    assert "positions" in d
    assert "exposure_summary" in d
    assert "recent_events" in d
    assert "explanation_text" in d
    assert "capital_history" in d
    assert "financial_risk" in d
    assert "concentration" in d
    assert "drawdown" in d
    assert d["capital"]["total_capital"] == 1500.0
    assert d["financial_risk"]["level"] in ("Baixo", "Moderado", "Elevado")
    assert "capital_max" in d["drawdown"]
    assert "drawdown_percent" in d["drawdown"]


def test_technical_route_returns_html():
    """GET /technical retorna HTML da página técnica."""
    system = _make_system()
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/technical")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Visão Técnica" in r.text or "technical" in r.text.lower()


def test_explanation_text_changes_with_state():
    """explanation_text varia com o estado (saudável vs buy_blocked vs SAFE_MODE)."""
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
    ss_ok = SafetyState()
    ss_blocked = SafetyState()
    ss_blocked.block_buy(RECONCILE_STALE)
    ss_safe = SafetyState()
    ss_safe.enable_safe_mode("INVARIANT_FAIL")

    text_ok = build_explanation_text(ss_ok, cfg)
    text_blocked = build_explanation_text(ss_blocked, cfg)
    text_safe = build_explanation_text(ss_safe, cfg)

    assert "mercados normalmente" in text_ok or "posição" in text_ok
    assert "mercados normalmente" not in text_blocked
    assert "pausadas" in text_blocked or "confirmar" in text_blocked
    assert "segurança" in text_safe.lower() or "Modo" in text_safe


def test_exposure_summary_calculation():
    """exposure_summary calcula % do capital por mercado."""
    system = _make_system(
        capital=1000.0,
        positions=[
            {"market_id": "m1", "side": "YES", "shares": 100.0, "avg_price": 0.5},
            {"market_id": "m2", "side": "NO", "shares": 200.0, "avg_price": 0.5},
        ],
    )
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/dashboard")
    assert r.status_code == 200
    d = r.json()
    exp = d["exposure_summary"]
    # m1: 100*0.5=50, m2: 200*0.5=100. Total exp=150. Total cap=1000.
    assert len(exp) >= 1
    total_exp = sum(e["exposure"] for e in exp)
    assert abs(total_exp - 150.0) < 0.01
    for e in exp:
        assert "market" in e
        assert "exposure" in e
        assert "percent" in e


def test_confidence_score_levels():
    """Confidence score retorna level e color corretos para diferentes estados."""
    system = _make_system(reasons=set())
    system["config"] = system["config"]
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/dashboard")
    assert r.status_code == 200
    conf = r.json().get("confidence_score", {})
    assert "score" in conf
    assert "level" in conf
    assert "color" in conf
    assert conf["score"] >= 0 and conf["score"] <= 100
    assert conf["level"] in ("Alta", "Moderada", "Baixa")
    assert conf["color"] in ("green", "yellow", "red")


def test_uptime_present():
    """Payload inclui uptime_seconds."""
    system = _make_system()
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/dashboard")
    assert r.status_code == 200
    d = r.json()
    assert "uptime_seconds" in d
    assert d["uptime_seconds"] is not None
    assert d["uptime_seconds"] >= 0


def test_timeline_events_structure():
    """recent_events tem estrutura com timestamp e text."""
    system = _make_system()
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/dashboard")
    assert r.status_code == 200
    d = r.json()
    assert "recent_events" in d
    assert "recent_events_v2" in d
    for ev in d.get("recent_events_v2", []):
        assert "timestamp" in ev
        assert "text" in ev


def test_capital_history_updates():
    """capital_history acumula snapshots a cada refresh."""
    system = _make_system(capital=500.0)
    app = create_dashboard_app(system)
    client = TestClient(app)

    r1 = client.get("/api/dashboard")
    d1 = r1.json()
    hist1 = d1.get("capital_history", [])
    assert len(hist1) >= 1
    r2 = client.get("/api/dashboard")
    d2 = r2.json()
    hist2 = d2.get("capital_history", [])
    assert len(hist2) >= len(hist1)
    assert hist2[-1]["total_capital"] == 500.0


def test_explanation_changes_when_buy_blocked():
    """explanation_text mostra mensagem de compras pausadas quando buy_blocked."""
    system = _make_system(reasons={RECONCILE_STALE})
    app = create_dashboard_app(system)
    client = TestClient(app)

    r = client.get("/api/dashboard")
    assert r.status_code == 200
    text = r.json().get("explanation_text", "")
    assert "pausad" in text.lower() or "confirmar" in text.lower() or "reconcili" in text.lower()
