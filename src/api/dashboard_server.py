"""
Dashboard Server — Painel Web para o Sistema de Trading
==========================================================

FastAPI + Jinja2 + Tailwind + Chart.js + Heroicons.

Apenas leitura + ACK. Sem alteração de config. Sem envio de ordens.
Uso standalone:
    uvicorn src.api.dashboard_server:app --host 0.0.0.0 --port 8765

Render (via src.server):
    uvicorn src.server:app --host 0.0.0.0 --port $PORT
"""

import os
import tempfile
import time
from collections import deque
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.api.dashboard_metrics import (
    HUMAN_REASON_MAP,
    apply_narrative_alignment,
    build_explanation_text,
    compute_confidence_score,
    compute_concentration,
    compute_drawdown_from_tracker,
    compute_financial_risk_from_inputs,
    get_human_status,
    get_status_color,
    humanize_reasons,
)
from src.api.drawdown_tracker import DrawdownTracker
from src.api.event_timeline import EventTimeline

# Re-export para compatibilidade com testes
__all__ = ["HUMAN_REASON_MAP", "create_dashboard_app"]

CAPITAL_HISTORY_MAX = 100
EVENTS_MAX = 20


def _get_decision_metrics() -> Dict[str, int]:
    """Métricas de decisão (motivos de descarte)."""
    try:
        from src.metrics import get_decision_metrics
        return get_decision_metrics()
    except ImportError:
        return {}


def _get_ingestion_metrics() -> Dict[str, Any]:
    """Métricas de ingestão (snapshots, engine events)."""
    try:
        from src.metrics import get_ingestion_metrics
        return get_ingestion_metrics()
    except ImportError:
        return {}


def _get_reversion_metrics(system_context: Dict[str, Any]) -> Dict[str, Any]:
    """Métricas do Reversion Engine para /technical. Não afeta core."""
    collector = system_context.get("reversion_metrics_collector")
    if collector is None or not hasattr(collector, "get_metrics"):
        return {
            "current_z": None,
            "rolling_z_mean": None,
            "rolling_z_std": None,
            "signal_count_last_24h": 0,
            "hit_rate_simulated": None,
            "average_reversion_magnitude": None,
            "average_time_to_reversion": None,
        }
    try:
        return collector.get_metrics()
    except Exception:
        return {
            "current_z": None,
            "rolling_z_mean": None,
            "rolling_z_std": None,
            "signal_count_last_24h": 0,
            "hit_rate_simulated": None,
            "average_reversion_magnitude": None,
            "average_time_to_reversion": None,
        }


def _last_reconcile_age(safety_state: Any) -> Optional[float]:
    """Idade em segundos do último reconcile OK. None se nunca."""
    if safety_state is None:
        return None
    ts = safety_state.last_reconcile_ok_ts
    if ts is None:
        return None
    return time.time() - ts


def _build_exposure_summary(
    position_manager: Any,
    capital_manager: Any,
) -> List[Dict[str, Any]]:
    """Resume exposição por mercado (% do capital)."""
    if position_manager is None or capital_manager is None:
        return []
    try:
        total = capital_manager.total_capital
        if total <= 0:
            return []
        all_pos = position_manager.get_all_positions()
        by_market: Dict[str, float] = {}
        for (_mid, _side), pos in all_pos.items():
            shares = getattr(pos, "shares", 0.0)
            avg = getattr(pos, "avg_entry_price", 0.0)
            exp = shares * avg if shares and avg else 0
            if exp > 0:
                key = f"{pos.market_id}|{pos.outcome_side}"
                by_market[key] = by_market.get(key, 0) + exp
        return [
            {
                "market": k.replace("|", " "),
                "exposure": round(v, 2),
                "percent": round(100 * v / total, 1),
            }
            for k, v in sorted(by_market.items(), key=lambda x: -x[1])
        ]
    except Exception:
        return []


def _derive_events(
    safety_state: Any,
    prev: Dict[str, Any],
    humanize_reason_map: Dict[str, str],
    event_timeline: EventTimeline,
) -> List[Dict[str, Any]]:
    """Deriva eventos a partir de mudanças de estado e registra na timeline."""
    events = []
    now = time.time()
    if safety_state is None:
        return events

    rec_ts = safety_state.last_reconcile_ok_ts
    if rec_ts is not None and (prev.get("last_reconcile_ts") or 0) < rec_ts:
        ev = {"ts": now, "type": "reconcile_ok", "message": "Reconciliação confirmada"}
        events.append(ev)
        event_timeline.add("reconcile_ok", "Reconciliação confirmada", now)

    safe = safety_state.global_safe_mode
    prev_safe = prev.get("global_safe_mode", False)
    if safe and not prev_safe:
        ev = {"ts": now, "type": "safe_mode_on", "message": "Modo de segurança ativado"}
        events.append(ev)
        event_timeline.add("safe_mode_on", "Modo de segurança ativado", now)
    elif not safe and prev_safe:
        ev = {"ts": now, "type": "safe_mode_off", "message": "Modo de segurança desativado"}
        events.append(ev)
        event_timeline.add("safe_mode_off", "Modo de segurança desativado", now)

    reasons = safety_state.reasons
    prev_reasons = prev.get("reasons", set())
    for r in reasons - prev_reasons:
        msg = humanize_reason_map.get(r, r)
        ev = {"ts": now, "type": "reason_added", "message": msg}
        events.append(ev)
        event_timeline.add("reason_added", msg, now)
    for r in prev_reasons - reasons:
        msg = humanize_reason_map.get(r, r)
        ev = {"ts": now, "type": "reason_removed", "message": msg}
        events.append(ev)
        event_timeline.add("reason_removed", msg, now)

    return events


def _build_universe_info(universe_manager: Any) -> Dict[str, Any]:
    """Build universe info for /api/performance."""
    try:
        from src.metrics import get_ingestion_metrics, get_universe_metrics
        metrics = get_ingestion_metrics()
        universe = get_universe_metrics()

        last_ts = metrics.get("last_event_timestamp")
        now = time.time()
        last_age = round(now - last_ts, 1) if last_ts else None

        # Compute snapshots per minute
        snapshots = metrics.get("snapshots_received", 0)
        start_ts = universe.get("universe_last_refresh_timestamp")
        if start_ts and snapshots > 0:
            elapsed = now - start_ts
            spm = round(snapshots / max(elapsed / 60.0, 0.1), 1) if elapsed > 0 else 0
        else:
            spm = 0

        info: Dict[str, Any] = {
            "source": universe.get("universe_source", "none"),
            "markets_subscribed": metrics.get("markets_subscribed", 0),
            "snapshots_received": snapshots,
            "snapshots_per_minute": spm,
            "last_snapshot_age_seconds": last_age,
            "last_refresh_timestamp": universe.get("universe_last_refresh_timestamp"),
            "universe_error": universe.get("universe_error"),
            "ws_disconnects_total": universe.get("ws_disconnects_total", 0),
        }

        if universe_manager is not None and hasattr(universe_manager, "get_market_infos"):
            info["top_markets"] = universe_manager.get_market_infos()

        return info
    except Exception:
        return {"source": "unknown", "markets_subscribed": 0}


def create_dashboard_app(system_context: Dict[str, Any]) -> FastAPI:
    """
    Cria app FastAPI do dashboard.

    system_context: dict do create_system (safety_state, capital_manager,
        position_manager, trade_ledger, config, etc.) + build_audit_report.
    """
    app = FastAPI(title="Dashboard Robot", docs_url=None, redoc_url=None)

    _dir = os.path.dirname(os.path.abspath(__file__))
    templates = Jinja2Templates(directory=os.path.join(_dir, "templates"))

    safety_state = system_context.get("safety_state")
    capital_manager = system_context.get("capital_manager")
    position_manager = system_context.get("position_manager")
    trade_ledger = system_context.get("trade_ledger")
    app_config = system_context.get("config")

    # Estado em memória (por instância do app)
    capital_history: deque = deque(maxlen=CAPITAL_HISTORY_MAX)
    event_timeline: EventTimeline = EventTimeline(maxlen=EVENTS_MAX)

    state_dir = system_context.get("state_dir")
    if not state_dir or state_dir == "mock":
        state_dir = tempfile.mkdtemp(prefix="dashboard_dd_")
    else:
        state_dir = str(state_dir)

    def _on_drawdown_reset() -> None:
        try:
            event_timeline.add(
                "dashboard_state_reset",
                "Estado do dashboard reiniciado por segurança",
                time.time(),
            )
        except Exception:
            pass

    drawdown_tracker: DrawdownTracker = DrawdownTracker(
        state_dir=state_dir,
        on_reset_callback=_on_drawdown_reset,
    )

    _prev_state: Dict[str, Any] = {}
    _first_build: bool = True

    def _build_audit_report() -> str:
        fn = system_context.get("build_audit_report")
        if callable(fn):
            return fn(
                capital_manager=capital_manager,
                position_manager=position_manager,
                trade_ledger=trade_ledger,
                safety_state=safety_state,
            )
        return "Relatório de auditoria não disponível."

    def _ack_reason(reason: str, note: str) -> bool:
        try:
            from src.orchestrator import ack_reason

            watcher = system_context.get("watcher_thread")
            return ack_reason(reason, note, system_context, watcher_thread=watcher)
        except Exception:
            return False

    def _build_dashboard_payload() -> Dict[str, Any]:
        """Constrói JSON consolidado para o dashboard 2.1."""
        run_mode = app_config.run_mode if app_config else "OBSERVE"
        buy_blocked = safety_state.buy_blocked if safety_state else False
        global_safe_mode = safety_state.global_safe_mode if safety_state else False
        reasons = sorted(safety_state.reasons) if safety_state else []
        human_status = get_human_status(safety_state)
        status_color = get_status_color(safety_state)

        # Capital
        if capital_manager:
            s = capital_manager.snapshot()
            cap_data = {
                "total_capital": s.get("total_capital", 0.0),
                "available_capital": s.get("available_capital", 0.0),
                "locked_in_open_orders": s.get("locked_in_open_orders", 0.0),
                "locked_in_unresolved_markets": s.get(
                    "locked_in_unresolved_markets", 0.0
                ),
                "realized_pnl": s.get("realized_pnl", 0.0),
            }
            ts = time.time()
            snap = {"ts": ts, "total_capital": cap_data["total_capital"], "realized_pnl": cap_data["realized_pnl"]}
            if not capital_history:
                capital_history.append(snap)
            else:
                capital_history.append(snap)
        else:
            cap_data = {
                "total_capital": 0.0,
                "available_capital": 0.0,
                "locked_in_open_orders": 0.0,
                "locked_in_unresolved_markets": 0.0,
                "realized_pnl": 0.0,
            }
            if not capital_history:
                capital_history.append({"ts": time.time(), "total_capital": 0.0, "realized_pnl": 0.0})

        # Positions
        positions = []
        if position_manager:
            all_pos = position_manager.get_all_positions()
            for (_mid, _side), pos in all_pos.items():
                shares = getattr(pos, "shares", 0.0)
                avg = getattr(pos, "avg_entry_price", 0.0)
                est = shares * avg if shares and avg else None
                positions.append({
                    "market": pos.market_id,
                    "side": pos.outcome_side,
                    "shares": shares,
                    "avg_entry_price": avg,
                    "estimated_value": est,
                })

        # Exposure
        exposure_summary = _build_exposure_summary(
            position_manager, capital_manager
        )
        total_cap = cap_data.get("total_capital") or 0.0
        exposure_pct = (
            sum(e["exposure"] for e in exposure_summary) / total_cap * 100
            if total_cap > 0 else 0.0
        )

        base_explanation = build_explanation_text(
            safety_state,
            app_config,
            position_manager=position_manager,
            capital_manager=capital_manager,
            exposure_percent=exposure_pct,
        )

        # Eventos derivados de mudanças (não no primeiro build)
        nonlocal _first_build
        if not _first_build:
            _derive_events(
                safety_state, _prev_state, HUMAN_REASON_MAP, event_timeline
            )
        _first_build = False

        _prev_state.update({
            "last_reconcile_ts": (
                safety_state.last_reconcile_ok_ts if safety_state else None
            ),
            "global_safe_mode": global_safe_mode,
            "reasons": set(safety_state.reasons) if safety_state else set(),
        })

        # Confidence score
        ctx = {
            "safety_state": safety_state,
            "config": app_config,
            "app_config": app_config,
        }
        confidence = compute_confidence_score(ctx)

        # Risco financeiro (modelo gradual)
        max_concentration = (
            max((e.get("percent", 0) for e in exposure_summary), default=0.0)
            / 100.0
        )
        financial_risk = compute_financial_risk_from_inputs(
            cap_data,
            len([p for p in positions if (p.get("shares") or 0) > 0]),
            safety_state,
            max_concentration,
        )

        # risk_level_changed
        prev_risk = _prev_state.get("risk_level")
        if prev_risk is not None and prev_risk != financial_risk.get("level"):
            event_timeline.add(
                "risk_level_changed",
                f"Nível de risco: {prev_risk} → {financial_risk.get('level', '?')}",
                time.time(),
            )
        _prev_state["risk_level"] = financial_risk.get("level")

        # Explicação com harmonização narrativa
        explanation = apply_narrative_alignment(
            base_explanation, confidence, financial_risk
        )

        # Concentração (maior exposição)
        concentration = compute_concentration(exposure_summary, total_cap)

        # Drawdown persistente
        current_cap = cap_data.get("total_capital", 0.0)
        drawdown_tracker.update(current_cap, event_timeline)
        drawdown = compute_drawdown_from_tracker(drawdown_tracker, current_cap)

        # Simulação realista (apenas RUN_MODE=REALISTIC_SIMULATION)
        simulation_metrics = None
        if run_mode == "REALISTIC_SIMULATION":
            stats = system_context.get("simulation_stats")
            if stats is not None and hasattr(stats, "get_metrics"):
                try:
                    simulation_metrics = stats.get_metrics()
                except Exception:
                    simulation_metrics = {}

        # Uptime
        start_time = system_context.get("system_start_time")
        uptime_seconds = (
            time.time() - start_time
            if start_time is not None
            else None
        )

        # Recent events formato {timestamp, text}
        recent = event_timeline.get_recent(20)
        recent_events_payload = [
            {"timestamp": e.get("timestamp", ""), "text": e.get("text", e.get("message", ""))}
            for e in recent
        ]
        recent_events_legacy = [
            {"ts": e["ts"], "message": e.get("text", e.get("message", ""))}
            for e in recent
        ]

        return {
            "run_mode": run_mode,
            "human_status": human_status,
            "status_color": status_color,
            "buy_blocked": buy_blocked,
            "global_safe_mode": global_safe_mode,
            "last_reconcile_age_seconds": _last_reconcile_age(safety_state),
            "invariants_ok_streak": (
                safety_state.invariants_ok_streak if safety_state else 0
            ),
            "capital_mismatch_ok_streak": (
                safety_state.capital_mismatch_ok_streak if safety_state else 0
            ),
            "capital": cap_data,
            "positions": positions,
            "exposure_summary": exposure_summary,
            "recent_events": recent_events_legacy,
            "recent_events_v2": recent_events_payload,
            "explanation_text": explanation,
            "reasons": reasons,
            "human_reasons": humanize_reasons(
                set(safety_state.reasons) if safety_state else set()
            ),
            "capital_history": list(capital_history),
            "confidence_score": confidence,
            "financial_risk": financial_risk,
            "concentration": concentration,
            "drawdown": drawdown,
            "uptime_seconds": uptime_seconds,
            "simulation_metrics": simulation_metrics,
            "decision_metrics": _get_decision_metrics(),
            "ingestion_metrics": _get_ingestion_metrics(),
        }

    # -------------------------------------------------------------------------
    # GET /
    # -------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={},
        )

    # -------------------------------------------------------------------------
    # GET /technical
    # -------------------------------------------------------------------------
    @app.get("/technical", response_class=HTMLResponse)
    async def technical_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="technical.html",
            context={},
        )

    # -------------------------------------------------------------------------
    # GET /api/dashboard
    # -------------------------------------------------------------------------
    @app.get("/api/dashboard")
    async def api_dashboard():
        return _build_dashboard_payload()

    # -------------------------------------------------------------------------
    # GET /api/status (retrocompatibilidade)
    # -------------------------------------------------------------------------
    @app.get("/api/status")
    async def api_status():
        p = _build_dashboard_payload()
        return {
            "run_mode": p["run_mode"],
            "buy_blocked": p["buy_blocked"],
            "global_safe_mode": p["global_safe_mode"],
            "human_status": p["human_status"],
            "last_reconcile_age_seconds": p["last_reconcile_age_seconds"],
            "reasons": p["reasons"],
            "human_reasons": p["human_reasons"],
            "invariants_ok_streak": p["invariants_ok_streak"],
            "capital_mismatch_ok_streak": p["capital_mismatch_ok_streak"],
            "activity_text": p["explanation_text"],
        }

    # -------------------------------------------------------------------------
    # GET /api/capital
    # -------------------------------------------------------------------------
    @app.get("/api/capital")
    async def api_capital():
        p = _build_dashboard_payload()
        return p["capital"]

    # -------------------------------------------------------------------------
    # GET /api/positions
    # -------------------------------------------------------------------------
    @app.get("/api/positions")
    async def api_positions():
        p = _build_dashboard_payload()
        return {"positions": p["positions"]}

    # -------------------------------------------------------------------------
    # GET /api/audit
    # -------------------------------------------------------------------------
    @app.get("/api/audit")
    async def api_audit():
        return {"report": _build_audit_report()}

    # -------------------------------------------------------------------------
    # GET /api/technical (dados para página técnica)
    # -------------------------------------------------------------------------
    @app.get("/api/technical")
    async def api_technical():
        report = _build_audit_report()
        p = _build_dashboard_payload()
        system = system_context
        threads = {}
        if system.get("market_engine"):
            t = system["market_engine"]
            threads["market_engine"] = t.is_alive()
        if system.get("reconciler_loop"):
            t = system["reconciler_loop"]
            threads["reconciler"] = t.is_alive()
        if system.get("watcher_thread"):
            threads["watcher"] = system["watcher_thread"].is_alive()

        fr = p.get("financial_risk") or {}
        risk_model_inputs = {
            "model_version": fr.get("model_version"),
            "percent_reserved": fr.get("percent_reserved"),
            "max_concentration": fr.get("max_concentration"),
            "open_positions": fr.get("open_positions"),
            "global_safe_mode": fr.get("global_safe_mode"),
            "risk_score": fr.get("score"),
        }
        reversion_metrics = _get_reversion_metrics(system_context)
        return {
            "audit_report": report,
            "run_mode": p["run_mode"],
            "raw_reasons": p["reasons"],
            "invariants_ok_streak": p["invariants_ok_streak"],
            "capital_mismatch_ok_streak": p["capital_mismatch_ok_streak"],
            "session_id": (
                safety_state.last_reconcile_session_id
                if safety_state else 0
            ),
            "capital_breakdown": p["capital"],
            "thread_status": threads,
            "risk_model_inputs": risk_model_inputs,
            "reversion_metrics": reversion_metrics,
        }

    # -------------------------------------------------------------------------
    # GET /api/strategy
    # -------------------------------------------------------------------------
    @app.get("/api/strategy")
    async def api_strategy():
        """Dados da estratégia: sinais, mercados, performance."""
        strategy_engine = system_context.get("strategy_engine")
        decision_pipeline = system_context.get("decision_pipeline")

        result: Dict[str, Any] = {
            "markets": [],
            "signals": [],
            "decisions": [],
            "strategy_summary": {},
            "pipeline_summary": {},
        }

        if strategy_engine is not None:
            try:
                result["markets"] = strategy_engine.get_market_data()
                result["signals"] = strategy_engine.get_signal_history(50)
                result["strategy_summary"] = strategy_engine.get_strategy_summary()
            except Exception as e:
                result["error_strategy"] = str(e)

        if decision_pipeline is not None:
            try:
                result["decisions"] = decision_pipeline.get_decision_history(50)
                result["pipeline_summary"] = decision_pipeline.get_pipeline_summary()
            except Exception as e:
                result["error_pipeline"] = str(e)

        return result

    # -------------------------------------------------------------------------
    # GET /api/market/{token_id}/history
    # -------------------------------------------------------------------------
    @app.get("/api/market/{token_id}/history")
    async def api_market_history(token_id: str):
        """Histórico de midpoints e z-scores para um token."""
        strategy_engine = system_context.get("strategy_engine")
        if strategy_engine is None:
            return {"mid_history": [], "z_history": []}
        return {
            "mid_history": strategy_engine.get_mid_history(token_id, 200),
            "z_history": strategy_engine.get_z_history(token_id, 200),
        }

    # -------------------------------------------------------------------------
    # GET /strategy (pagina frontend)
    # -------------------------------------------------------------------------
    @app.get("/strategy", response_class=HTMLResponse)
    async def strategy_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="strategy.html",
            context={},
        )

    # -------------------------------------------------------------------------
    # GET /api/oracle
    # -------------------------------------------------------------------------
    @app.get("/api/oracle")
    async def api_oracle():
        """Métricas completas do ProbabilityOracle."""
        oracle = system_context.get("probability_oracle")
        if oracle is None:
            return {"error": "Oracle not available", "metrics": {}}
        try:
            return {"metrics": oracle.get_full_metrics()}
        except Exception as e:
            return {"error": str(e), "metrics": {}}

    # -------------------------------------------------------------------------
    # GET /api/backtest
    # -------------------------------------------------------------------------
    @app.get("/api/backtest")
    async def api_backtest():
        """Executa backtest rápido com cenários de exemplo."""
        try:
            from src.backtest.simple_backtest import SimpleBacktest, BacktestScenario
            bt = SimpleBacktest()
            # Cenários demonstrativos
            scenarios = [
                BacktestScenario(token_id="demo_1", p_market=0.40, p_oracle=0.55, outcome=1.0, spread=0.02),
                BacktestScenario(token_id="demo_2", p_market=0.60, p_oracle=0.45, outcome=0.0, spread=0.02),
                BacktestScenario(token_id="demo_3", p_market=0.50, p_oracle=0.52, outcome=1.0, spread=0.02),
                BacktestScenario(token_id="demo_4", p_market=0.70, p_oracle=0.80, outcome=1.0, spread=0.03),
                BacktestScenario(token_id="demo_5", p_market=0.30, p_oracle=0.15, outcome=0.0, spread=0.02),
            ]
            result = bt.run_scenarios(scenarios)
            return {
                "total_trades": result.total_trades,
                "total_pnl": result.total_pnl,
                "win_rate": result.win_rate,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "brier_score": result.brier_score,
                "equity_curve": result.equity_curve,
                "trades": result.trades,
            }
        except Exception as e:
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # GET /api/universe
    # -------------------------------------------------------------------------
    @app.get("/api/universe")
    async def api_universe():
        """Estado do UniverseBuilder."""
        ub = system_context.get("universe_builder")
        if ub is None:
            return {"error": "Universe builder not available"}
        try:
            return ub.get_summary()
        except Exception as e:
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # GET /performance (executive dashboard page)
    # -------------------------------------------------------------------------
    @app.get("/performance", response_class=HTMLResponse)
    async def performance_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="performance.html",
            context={},
        )

    # -------------------------------------------------------------------------
    # GET /api/performance (data for executive dashboard)
    # -------------------------------------------------------------------------
    @app.get("/api/performance")
    async def api_performance():
        """Dados para a página de performance executiva."""
        oracle = system_context.get("probability_oracle")
        edge_validator = system_context.get("edge_validator")
        oracle_metrics_store = system_context.get("oracle_metrics_store")
        universe_manager = system_context.get("universe_manager")

        # 1. Oracle quality
        oracle_quality = {"label": "Sem dados", "level": "unknown", "brier": None}
        brier = None
        if oracle is not None:
            try:
                brier = oracle.get_brier_score()
            except Exception:
                pass
        if oracle_metrics_store is not None:
            try:
                brier = brier or oracle_metrics_store.rolling_brier_score()
            except Exception:
                pass
        if brier is not None:
            # Baseline: random oracle = 0.25
            if brier < 0.15:
                oracle_quality = {"label": "Excelente", "level": "excellent", "brier": round(brier, 4)}
            elif brier < 0.25:
                oracle_quality = {"label": "Boa", "level": "good", "brier": round(brier, 4)}
            else:
                oracle_quality = {"label": "Fraca", "level": "poor", "brier": round(brier, 4)}

        # 2. Finding edge?
        edge_count_24h = 0
        if edge_validator is not None:
            try:
                report = edge_validator.generate_report()
                edge_count_24h = report.get("total_opportunities", 0)
            except Exception:
                pass
        elif oracle_metrics_store is not None:
            try:
                signals = oracle_metrics_store.get_signals_last_24h()
                edge_count_24h = len(signals)
            except Exception:
                pass

        # 3. Capturing edge?
        profit_per_edge = None
        if oracle_metrics_store is not None:
            try:
                profit_per_edge = oracle_metrics_store.profit_per_unit_edge()
            except Exception:
                pass

        # 4. Market reaction speed
        avg_decay = None
        if oracle is not None:
            try:
                decay_metrics = oracle.edge_decay_tracker.get_metrics()
                avg_decay = decay_metrics.get("avg_half_life_seconds")
            except Exception:
                pass

        # 5. System status
        oracle_blocked = False
        if oracle is not None:
            try:
                oracle_blocked = oracle.should_block_oracle()
            except Exception:
                pass

        if oracle_blocked:
            system_status = {"label": "Oracle instavel", "level": "danger"}
        elif edge_count_24h == 0:
            system_status = {"label": "Sem edge detectavel", "level": "warning"}
        else:
            system_status = {"label": "Operando", "level": "ok"}

        # Validation report
        validation_report = {}
        if edge_validator is not None:
            try:
                validation_report = edge_validator.generate_report()
            except Exception:
                pass

        # Oracle full metrics
        oracle_full = {}
        if oracle_metrics_store is not None:
            try:
                oracle_full = oracle_metrics_store.get_full_report()
            except Exception:
                pass

        # Universe info
        universe_info = _build_universe_info(universe_manager)

        return {
            "oracle_quality": oracle_quality,
            "edge_finding": {
                "count_24h": edge_count_24h,
                "avg_per_day": edge_count_24h,
            },
            "edge_capturing": {
                "profit_per_unit_edge": round(profit_per_edge, 4) if profit_per_edge is not None else None,
            },
            "market_reaction": {
                "avg_decay_seconds": round(avg_decay, 1) if avg_decay is not None else None,
            },
            "system_status": system_status,
            "validation_report": validation_report,
            "oracle_metrics": oracle_full,
            "universe": universe_info,
        }

    # -------------------------------------------------------------------------
    # GET /api/metrics_summary (remote monitoring endpoint)
    # -------------------------------------------------------------------------
    @app.get("/api/metrics_summary")
    async def api_metrics_summary():
        """Endpoint para verificação remota de métricas."""
        oracle = system_context.get("probability_oracle")
        oracle_metrics_store = system_context.get("oracle_metrics_store")
        edge_validator = system_context.get("edge_validator")

        result: Dict[str, Any] = {
            "timestamp": time.time(),
            "oracle_blocked": False,
            "brier_score": None,
            "edge_signals_24h": 0,
            "system_ok": True,
        }

        if oracle is not None:
            try:
                result["oracle_blocked"] = oracle.should_block_oracle()
                result["brier_score"] = oracle.get_brier_score()
            except Exception:
                pass

        if oracle_metrics_store is not None:
            try:
                signals = oracle_metrics_store.get_signals_last_24h()
                result["edge_signals_24h"] = len(signals)
            except Exception:
                pass

        if edge_validator is not None:
            try:
                report = edge_validator.generate_report()
                result["validation"] = report
            except Exception:
                pass

        result["system_ok"] = not result.get("oracle_blocked", False)
        return result

    # -------------------------------------------------------------------------
    # POST /api/ack
    # -------------------------------------------------------------------------
    class AckBody(BaseModel):
        reason: str
        note: str = ""

    @app.post("/api/ack")
    async def api_ack(body: AckBody):
        reason = (body.reason or "").strip()
        if not reason:
            raise HTTPException(
                status_code=400,
                detail="reason obrigatório",
            )

        if safety_state is None:
            raise HTTPException(
                status_code=400,
                detail="safety_state não disponível",
            )

        if reason not in safety_state.reasons:
            raise HTTPException(
                status_code=400,
                detail=f"reason '{reason}' não está ativa",
            )

        from src.safety.safety_state import SAFE_MODE

        if reason == SAFE_MODE:
            other = safety_state.reasons - {SAFE_MODE}
            if other:
                raise HTTPException(
                    status_code=400,
                    detail="Não é possível confirmar Modo Seguro com outros alertas ativos",
                )

        ok = _ack_reason(reason, body.note)
        if not ok:
            raise HTTPException(
                status_code=400,
                detail="ACK rejeitado: pré-condições não atendidas",
            )
        return {"ok": True, "reason": reason}

    return app


# -----------------------------------------------------------------------------
# Mock system para rodar standalone
# -----------------------------------------------------------------------------

def _create_mock_system() -> Dict[str, Any]:
    """Sistema mínimo para dashboard sem orchestrator (demo)."""
    from src.risk.capital_manager import CapitalManager
    from src.risk.position_manager import PositionManager
    from src.engine.trade_ledger import TradeLedger
    from src.engine.probability_calibrator import ProbabilityCalibrator
    from src.safety.safety_state import SafetyState
    from src.config.config import load_app_config_from_env
    from src.safety.audit_report import build_audit_report

    cfg = load_app_config_from_env()
    cm = CapitalManager(initial_capital=1000.0)
    tmpdir = tempfile.mkdtemp(prefix="dashboard_demo_")
    cal = ProbabilityCalibrator(state_path=os.path.join(tmpdir, "cal.json"))
    pm = PositionManager(state_path=os.path.join(tmpdir, "pos.json"))
    tl = TradeLedger(
        state_path=os.path.join(tmpdir, "ledger.json"),
        calibrator=cal,
        capital_manager=cm,
    )
    ss = SafetyState()

    return {
        "system_start_time": time.time(),
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
        "resolved_state_dir": tmpdir,
        "resolved_data_dir": tmpdir,
        "state_dir": tmpdir,
        "boot_error": None,
        "strategy_engine": None,
        "decision_pipeline": None,
        "probability_oracle": None,
        "universe_builder": None,
        "oracle_metrics_store": None,
        "edge_validator": None,
        "universe_manager": None,
    }


app = create_dashboard_app(_create_mock_system())
