"""
Servidor unificado — FastAPI + Dashboard + /health para Render
=============================================================

Um único processo: uvicorn serve FastAPI com dashboard e /health.
Orchestrator roda em thread em background.

Uso:
    uvicorn src.server:app --host 0.0.0.0 --port $PORT

Render:
    Start Command: python -m uvicorn src.server:app --host 0.0.0.0 --port $PORT
    Health Check Path: /health

BOOT SEQUENCE:
1. Module load: cria FastAPI app + empty _system dict (< 1s)
2. Startup event: create_system() popula _system in-place
3. Background thread: run_system() inicia watcher + engine
4. Se AUTO_UNIVERSE=true: descobre mercados via Gamma API
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Python version: apenas log, nunca crash
# -----------------------------------------------------------------------------

def _log_python_version() -> None:
    """
    Loga versão Python. Nunca levanta exceção.
    >= 3.11: aceito; < 3.11: log error; > 3.12: log warning (não testado oficialmente).
    """
    try:
        v = sys.version_info
        ver = (v.major, v.minor)
        if ver < (3, 11):
            logger.error(
                "[BOOT] Python %s.%s.%s é anterior a 3.11; podem ocorrer incompatibilidades",
                v.major, v.minor, v.micro,
            )
        elif ver >= (3, 12):
            logger.warning(
                "[BOOT] Python %s.%s.%s (>= 3.12) — versão não testada oficialmente",
                v.major, v.minor, v.micro,
            )
    except Exception as e:
        logger.warning("[BOOT] Não foi possível verificar versão Python: %s", e)


from src.seed import set_global_seed

_seed = int(os.environ.get("RANDOM_SEED", "42"))
set_global_seed(_seed)

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Global state — mutable dict, populated in startup event
# -----------------------------------------------------------------------

_system: Dict[str, Any] = {}
_boot_error: Optional[str] = None
_git_sha: str = "unknown"
_build_time_utc: str = "unknown"
_process_start_time: float = time.time()
_process_start_time_utc: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Presets para testes controlados de ingestão WS (TOKEN_SET=1|3|5)
_TOKEN_SET_PRESETS: dict = {
    1: [
        "38397507750621893057346880033441136112987238933685677349709401910643842844855",
    ],
    3: [
        "38397507750621893057346880033441136112987238933685677349709401910643842844855",
        "14793609061774012318755418128997240420901961440229138466682192794558490666550",
        "5708561660601459805512817131601230493971589760294984590237789749933853841330",
    ],
    5: [
        "38397507750621893057346880033441136112987238933685677349709401910643842844855",
        "14793609061774012318755418128997240420901961440229138466682192794558490666550",
        "5708561660601459805512817131601230493971589760294984590237789749933853841330",
        "39317885422026394259056328144566743331998444273202427934141325790266108570112",
        "8067407495040644813204108294851401001772374258077273118510788800804436836793",
    ],
}

_token_set_active: Optional[int] = None


def _resolve_tokens_fast() -> list:
    global _token_set_active
    token_set = os.environ.get("TOKEN_SET", "").strip()
    if token_set in ("1", "3", "5"):
        preset = _TOKEN_SET_PRESETS[int(token_set)]
        _token_set_active = int(token_set)
        logger.info("[BOOT] TOKEN_SET=%s — preset com %d tokens", token_set, len(preset))
        return list(preset)
    _token_set_active = None
    raw = os.environ.get("POLYMARKET_TOKENS", "").strip()
    if not raw:
        return []
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if tokens:
        logger.info("[BOOT] POLYMARKET_TOKENS: %d tokens", len(tokens))
    return tokens


def _is_auto_universe() -> bool:
    return os.environ.get("AUTO_UNIVERSE", "true").strip().lower() in ("true", "1", "yes")


# -----------------------------------------------------------------------
# Module-level: ONLY create FastAPI app (fast, no system creation)
# -----------------------------------------------------------------------

logger.info("[BOOT] Module load start")

_manual_tokens = _resolve_tokens_fast()
_auto_universe = _is_auto_universe()
_universe_source = "manual" if _manual_tokens else ("auto" if _auto_universe else "none")

logger.info(
    "[BOOT] tokens=%d auto_universe=%s universe_source=%s",
    len(_manual_tokens), _auto_universe, _universe_source,
)

try:
    from src.metrics import set_universe_source
    set_universe_source(_universe_source)
except Exception:
    pass

# Dashboard: paths iguais ao orchestrator para evitar mkdtemp no import
# Constantes locais (orchestrator.DEFAULT_*); sem I/O
_IMPORT_STATE_DIR = "/tmp/polymarket_state"
_IMPORT_DATA_DIR = "/tmp/polymarket_data"
_system["state_dir"] = _IMPORT_STATE_DIR
_system["resolved_state_dir"] = _IMPORT_STATE_DIR
_system["resolved_data_dir"] = _IMPORT_DATA_DIR

# Dashboard gets reference to _system dict — will see updates after startup populates it
try:
    from src.api.dashboard_server import create_dashboard_app
    app = create_dashboard_app(_system)
    logger.info("[BOOT] Dashboard app created")
except Exception as e:
    logger.error("[BOOT] Dashboard creation FAILED: %s", e)
    app = FastAPI(title="Trading System - Fallback")

    @app.get("/")
    async def fallback_root():
        return {"status": "error", "message": str(e)}


# -----------------------------------------------------------------------
# Health checkpoint — persistência mínima em /tmp (não fatal)
# -----------------------------------------------------------------------

_HEALTH_CHECKPOINT_INTERVAL_SEC = 60
_health_checkpoint_baseline: Dict[str, Any] = {}
_health_checkpoint_last_session: Dict[str, Any] = {}
_health_checkpoint_loaded: bool = False
_health_checkpoint_lock = threading.Lock()
_last_checkpoint_write_ts: Optional[float] = None
_last_checkpoint_write_utc: Optional[str] = None


def _get_checkpoint_path() -> str:
    """Path do checkpoint. Garantido sob /tmp (fallback)."""
    base = _system.get("resolved_state_dir") or _IMPORT_STATE_DIR
    return os.path.join(base, "health_checkpoint.json")


def _checkpoint_path_safe_under_tmp(path: str) -> bool:
    """Garantia: path resolvido está sob /tmp (ou /private/tmp no macOS)."""
    try:
        real = os.path.realpath(path)
        return real.startswith("/tmp") or real.startswith("/private/tmp")
    except Exception:
        return False


def _load_health_checkpoint() -> None:
    """Carrega baseline do checkpoint (não fatal)."""
    global _health_checkpoint_baseline
    try:
        path = _get_checkpoint_path()
        if not _checkpoint_path_safe_under_tmp(path):
            logger.warning("[CHECKPOINT] path fora de /tmp, skip load: %s", path)
            return
        if os.path.isfile(path):
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                with _health_checkpoint_lock:
                    _health_checkpoint_baseline = dict(data.get("counters", {}))
                logger.info("[CHECKPOINT] loaded baseline: %d keys", len(_health_checkpoint_baseline))
    except Exception as e:
        logger.debug("[CHECKPOINT] load skip (non-fatal): %s", e)


def _maybe_write_health_checkpoint(metrics: Dict[str, Any], universe: Dict[str, Any]) -> None:
    """Escreve checkpoint se 60s desde última escrita (não fatal)."""
    global _last_checkpoint_write_ts, _last_checkpoint_write_utc
    global _health_checkpoint_baseline, _health_checkpoint_last_session, _health_checkpoint_loaded
    if not _health_checkpoint_loaded:
        _load_health_checkpoint()
        _health_checkpoint_loaded = True
    now = time.time()
    with _health_checkpoint_lock:
        if _last_checkpoint_write_ts is not None and (now - _last_checkpoint_write_ts) < _HEALTH_CHECKPOINT_INTERVAL_SEC:
            return
    try:
        path = _get_checkpoint_path()
        if not _checkpoint_path_safe_under_tmp(path):
            logger.warning("[CHECKPOINT] path fora de /tmp, skip write: %s", path)
            return
        base = os.path.dirname(path)
        os.makedirs(base, exist_ok=True)
        keys = (
            "total_ws_messages", "total_book_events", "snapshots_received", "total_filtered_events",
            "enqueue_snapshot_attempted", "enqueue_snapshot_ok", "enqueue_snapshot_dropped_full",
            "enqueue_snapshot_error", "ws_disconnects_total", "gaps_total",
        )
        current: Dict[str, Any] = {}
        for k in keys:
            v = metrics.get(k) if k in metrics else universe.get(k)
            if v is not None and isinstance(v, (int, float)):
                current[k] = v
        tracker = _system.get("edge_episode_tracker")
        if tracker:
            try:
                agg = tracker.get_aggregates()
                for k in ("edge_episodes_total", "edge_episodes_closed"):
                    v = agg.get(k)
                    if v is not None:
                        current[k] = v if isinstance(v, (int, float)) else 0
            except Exception:
                pass
        delta = {
            k: current.get(k, 0) - _health_checkpoint_last_session.get(k, 0)
            for k in set(keys) | {"edge_episodes_total", "edge_episodes_closed"}
        }
        counters: Dict[str, Any] = {
            k: _health_checkpoint_baseline.get(k, 0) + delta.get(k, 0)
            for k in set(_health_checkpoint_baseline) | set(delta)
        }
        payload = {
            "counters": counters,
            "last_write_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=0)
        with _health_checkpoint_lock:
            _last_checkpoint_write_ts = now
            _last_checkpoint_write_utc = payload["last_write_utc"]
            _health_checkpoint_baseline = dict(counters)
            _health_checkpoint_last_session = dict(current)
    except Exception as e:
        logger.debug("[CHECKPOINT] write skip (non-fatal): %s", e)


# -----------------------------------------------------------------------
# Health check — ALWAYS works, exposes boot errors
# -----------------------------------------------------------------------

def _token_ids_sample() -> List[str]:
    """Primeiros 3 token_ids truncados (para /health)."""
    tokens = _system.get("token_ids") or _manual_tokens
    return [t[:16] + "..." if len(t) > 16 else t for t in tokens[:3]]


def _edge_episode_health(system: Dict[str, Any]) -> Dict[str, Any]:
    """Métricas do Edge Episode Tracker para /health."""
    tracker = system.get("edge_episode_tracker")
    if tracker is None:
        return {}
    try:
        agg = tracker.get_aggregates()
        return {
            "edge_episodes_total": agg.get("edge_episodes_total"),
            "edge_episodes_open": agg.get("edge_episodes_open"),
            "edge_episodes_closed": agg.get("edge_episodes_closed"),
            "edge_duration_ms_p50": agg.get("edge_duration_ms_p50"),
            "edge_duration_ms_p90": agg.get("edge_duration_ms_p90"),
            "edge_duration_ms_p99": agg.get("edge_duration_ms_p99"),
            "expected_net_edge_bps_p50": agg.get("expected_net_edge_bps_p50"),
            "expected_net_edge_bps_p90": agg.get("expected_net_edge_bps_p90"),
            "episodes_survive_500ms": agg.get("episodes_survive_500ms"),
            "episodes_survive_2000ms": agg.get("episodes_survive_2000ms"),
            "episodes_survive_5000ms": agg.get("episodes_survive_5000ms"),
            "episodes_per_hour": agg.get("episodes_per_hour"),
            "top_5_markets_by_ev": agg.get("top_5_markets_by_ev", []),
            "edge_auc_bps_ms_p50": agg.get("edge_auc_bps_ms_p50"),
            "edge_auc_bps_ms_p90": agg.get("edge_auc_bps_ms_p90"),
            "top_5_markets_by_edge_auc": agg.get("top_5_markets_by_edge_auc", []),
            "exec_latency_ms": agg.get("exec_latency_ms"),
            "fee_bps": agg.get("fee_bps"),
            "slippage_bps_base": agg.get("slippage_bps_base"),
            "slippage_bps_per_spread": agg.get("slippage_bps_per_spread"),
            "edge_survival_curve": tracker.compute_edge_survival_curve(),
            "edge_decay_curve": tracker.compute_edge_decay_curve(),
            "edge_hazard_curve": tracker.compute_edge_hazard_curve(),
            "expected_edge_after_latency": tracker.compute_expected_edge_after_latency(),
            "market_profitability_ranking": tracker.compute_market_ranking(),
            "structural_edge_candidates_top": _structural_top3(tracker),
        }
    except Exception:
        return {}


def _structural_top3(tracker) -> list:
    """Top 3 structural candidates (lightweight summary for /health)."""
    try:
        disc = tracker.compute_structural_edge_discovery()
        cands = disc.get("structural_edge_candidates", [])
        return [
            {"market_key": c["market_key"], "score": c["score"]}
            for c in cands[:3]
        ]
    except Exception:
        return []


@app.get("/health")
def health():
    try:
        from src.config.config import load_app_config_from_env
        from src.metrics import get_ingestion_metrics, get_universe_metrics
        cfg = load_app_config_from_env()
        metrics = get_ingestion_metrics()
        universe = get_universe_metrics()

        last_ts = metrics.get("last_event_timestamp")
        last_age = round(time.time() - last_ts, 1) if last_ts else None

        _maybe_write_health_checkpoint(metrics, universe)

        with _health_checkpoint_lock:
            ck_age = round(time.time() - _last_checkpoint_write_ts, 1) if _last_checkpoint_write_ts else None

        return {
            "status": "ok",
            "process_start_time_utc": _process_start_time_utc,
            "uptime_seconds": int(time.time() - _process_start_time),
            "run_mode": cfg.run_mode,
            "has_system": bool(_system.get("ws_client")),
            "boot_error": _boot_error,
            "resolved_state_dir": _system.get("resolved_state_dir"),
            "resolved_data_dir": _system.get("resolved_data_dir"),
            "git_sha": _git_sha,
            "build_time_utc": _build_time_utc,
            "token_count": metrics.get("markets_subscribed", 0),
            "markets_subscribed": metrics.get("markets_subscribed", 0),
            "snapshots_received": metrics.get("snapshots_received", 0),
            "total_ws_messages": metrics.get("total_ws_messages", 0),
            "total_book_events": metrics.get("total_book_events", 0),
            "total_filtered_events": metrics.get("total_filtered_events", 0),
            "enqueue_snapshot_attempted": metrics.get("enqueue_snapshot_attempted", 0),
            "enqueue_snapshot_ok": metrics.get("enqueue_snapshot_ok", 0),
            "enqueue_snapshot_dropped_full": metrics.get("enqueue_snapshot_dropped_full", 0),
            "enqueue_snapshot_error": metrics.get("enqueue_snapshot_error", 0),
            "last_queue_size": metrics.get("last_queue_size"),
            "last_snapshot_age_seconds": last_age,
            "ws_disconnects_total": universe.get("ws_disconnects_total", 0),
            "gaps_total": universe.get("gaps_total", 0),
            "universe_last_refresh_timestamp": universe.get("universe_last_refresh_timestamp"),
            "universe_source": (
                "manual" if _token_set_active is not None
                else universe.get("universe_source", "none")
            ),
            "universe_error": universe.get("universe_error"),
            "degraded_components": _system.get("degraded_components", []),
            "token_set_active": _token_set_active,
            "token_ids_sample": _token_ids_sample(),
            "checkpoint_age_seconds": ck_age,
            "last_checkpoint_write_utc": _last_checkpoint_write_utc,
            **_edge_episode_health(_system),
        }
    except Exception as ex:
        return {
            "status": "ok",
            "boot_error": _boot_error,
            "health_error": str(ex),
            "process_start_time_utc": _process_start_time_utc,
            "uptime_seconds": int(time.time() - _process_start_time),
            "git_sha": _git_sha,
            "build_time_utc": _build_time_utc,
            "resolved_state_dir": _system.get("resolved_state_dir"),
            "resolved_data_dir": _system.get("resolved_data_dir"),
            "checkpoint_age_seconds": None,
            "last_checkpoint_write_utc": None,
        }


# -----------------------------------------------------------------------
# Analytics endpoint
# -----------------------------------------------------------------------

@app.get("/analytics")
def analytics():
    tracker = _system.get("edge_episode_tracker")
    if tracker is None:
        return {"edge_survival_curve": {}, "aggregates": {}}
    try:
        cross_arb = _cross_market_analytics()
        result = {
            "edge_survival_curve": tracker.compute_edge_survival_curve(),
            "edge_decay_curve": tracker.compute_edge_decay_curve(),
            "edge_hazard_curve": tracker.compute_edge_hazard_curve(),
            "expected_edge_after_latency": tracker.compute_expected_edge_after_latency(),
            "market_profitability_ranking": tracker.compute_market_ranking(),
            **_structural_analytics(tracker),
            "sim_summary": _last_sim_summary,
            **cross_arb,
            "cross_market_trade_candidates": _build_trade_candidates(
                cross_arb.get("cross_market_arbitrage_opportunities", [])
            ),
            "aggregates": tracker.get_aggregates(),
        }
        result["market_regime"] = _compute_market_regime(result)
        result["shadow_trading_summary"] = _shadow_trading_summary()
        result["risk_engine_state"] = _risk_engine_state()
        result["opportunity_ranking"] = _opportunity_ranking(result)
        return result
    except Exception as ex:
        return {
            "error": str(ex),
            "edge_survival_curve": {},
            "edge_decay_curve": {},
            "edge_hazard_curve": {},
            "expected_edge_after_latency": {},
            "market_profitability_ranking": [],
            "structural_edge_candidates": [],
            "structural_edge_new_markets": [],
            "sim_summary": None,
            "cross_market_arbitrage_opportunities": [],
            "cross_market_trade_candidates": [],
            "market_regime": {"regime_score": 0.0, "state": "hostile", "components": {}},
            "shadow_trading_summary": {},
            "risk_engine_state": {},
            "opportunity_ranking": [],
            "aggregates": {},
        }


def _structural_analytics(tracker) -> dict:
    try:
        return tracker.compute_structural_edge_discovery()
    except Exception:
        return {"structural_edge_candidates": [], "structural_edge_new_markets": []}


def _cross_market_analytics() -> dict:
    """Scan current market universe for cross-market arbitrage (non-fatal)."""
    try:
        from src.strategy.arbitrage_scanner import ArbitrageScanner
        from src.strategy.market_graph import MarketGraph

        universe = _system.get("universe")
        if universe is None:
            return {"cross_market_arbitrage_opportunities": []}

        snapshot = []
        try:
            infos = universe.get_market_infos() if hasattr(universe, "get_market_infos") else {}
            for mid, info in infos.items():
                prob = info.get("prob") or info.get("oracle_price") or 0
                snapshot.append({
                    "market_id": mid,
                    "prob": float(prob),
                    "category": info.get("category", info.get("question", "unknown")),
                    "best_bid": info.get("best_bid"),
                    "best_ask": info.get("best_ask"),
                })
        except Exception:
            pass

        if not snapshot:
            return {"cross_market_arbitrage_opportunities": []}

        graph = _system.get("market_graph") or MarketGraph()
        scanner = ArbitrageScanner(graph=graph)
        result = scanner.scan(snapshot)
        return {"cross_market_arbitrage_opportunities": result.get("cross_market_arbitrage_opportunities", [])}
    except Exception:
        return {"cross_market_arbitrage_opportunities": []}


def _shadow_trading_summary() -> dict:
    """Return shadow trader summary if available (non-fatal)."""
    try:
        trader = _system.get("shadow_trader")
        if trader is None:
            return {}
        return trader.summary()
    except Exception:
        return {}


def _risk_engine_state() -> dict:
    """Return risk engine state if available (non-fatal)."""
    try:
        engine = _system.get("risk_engine")
        if engine is None:
            return {}
        return engine.state()
    except Exception:
        return {}


def _opportunity_ranking(analytics_snapshot: dict) -> list:
    """Rank market profitability entries by regime-adjusted score (non-fatal)."""
    try:
        from src.trading.opportunity_prioritizer import OpportunityPrioritizer

        ranking = analytics_snapshot.get("market_profitability_ranking", [])
        if not ranking:
            return []

        regime = analytics_snapshot.get("market_regime", {})
        regime_score = regime.get("regime_score", 0.5) if isinstance(regime, dict) else 0.5

        opportunities = []
        for entry in ranking:
            opportunities.append({
                "market_id": entry.get("market_slug", entry.get("token_id", "")),
                "expected_pnl_per_hour": entry.get("episodes_per_hour", 0) * entry.get("mean_edge_after_latency", 0),
                "liquidity_score": entry.get("latency_survival", 0.5),
                "fill_rate": entry.get("latency_survival", 0.5),
            })

        prioritizer = OpportunityPrioritizer()
        return prioritizer.rank(opportunities, regime_score=regime_score, top_n=10)
    except Exception:
        return []


def _compute_market_regime(analytics_snapshot: dict) -> dict:
    """Compute market regime from the analytics snapshot (non-fatal)."""
    try:
        from src.strategy.market_regime_detector import MarketRegimeDetector
        return MarketRegimeDetector().compute(analytics_snapshot)
    except Exception:
        return {"regime_score": 0.0, "state": "hostile", "components": {}}


def _build_trade_candidates(opportunities: list) -> list:
    """Convert arbitrage opportunities into simulated trade candidates (non-fatal)."""
    try:
        from src.strategy.arbitrage_basket_builder import ArbitrageBasketBuilder
        from src.sim.basket_simulator import simulate_baskets, rank_baskets

        if not opportunities:
            return []

        builder = ArbitrageBasketBuilder()
        baskets = builder.build_many(opportunities, capital=100.0, max_baskets=100)
        if not baskets:
            return []

        cfg_dict = {
            "fee_taker_bps": float(os.environ.get("FEE_TAKER_BPS", "0")),
            "slippage_from_spread_mult": float(os.environ.get("SLIPPAGE_FROM_SPREAD_MULT", "0.5")),
            "slippage_bps_floor": float(os.environ.get("SLIPPAGE_BPS_FLOOR", "2.0")),
            "lat_signal_to_order_ms": float(os.environ.get("LAT_SIGNAL_TO_ORDER_MS", "50")),
            "lat_order_to_exchange_ms": float(os.environ.get("LAT_ORDER_TO_EXCHANGE_MS", "150")),
            "lat_exchange_to_ack_ms": float(os.environ.get("LAT_EXCHANGE_TO_ACK_MS", "50")),
        }

        sim_result = simulate_baskets(baskets, cfg_dict, seed=42)
        return rank_baskets(sim_result.get("basket_simulations", []), top_n=10)
    except Exception:
        return []


# -----------------------------------------------------------------------
# Simulate endpoint
# -----------------------------------------------------------------------

_last_sim_summary: Optional[Dict[str, Any]] = None

@app.get("/simulate")
def simulate_endpoint(
    from_utc: Optional[str] = None,
    to_utc: Optional[str] = None,
    market_key: Optional[str] = None,
    style: str = "both",
    seed: Optional[int] = None,
):
    """Run execution simulation over recorded data + episodes."""
    try:
        from src.sim.execution_simulator import SimConfig, simulate_v2, save_sim_trades_jsonl
        from src.sim.marketdata_recorder import load_recorded_snapshots

        state_dir = _system.get("resolved_data_dir") or "/tmp/polymarket_state"

        recorder = _system.get("marketdata_recorder")
        snap_path = recorder.path if recorder else os.path.join(state_dir, "marketdata_snapshots.jsonl")
        snapshots = load_recorded_snapshots(snap_path)

        tracker = _system.get("edge_episode_tracker")
        episodes: list = []
        if tracker is not None:
            with tracker._lock:
                episodes = [tracker._episode_to_dict(e) for e in tracker._closed]

        overrides: Dict[str, Any] = {}
        if seed is not None:
            overrides["seed"] = seed

        cfg = SimConfig.from_env(**overrides)

        from_ts = None
        to_ts = None
        if from_utc:
            try:
                from_ts = datetime.fromisoformat(from_utc.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass
        if to_utc:
            try:
                to_ts = datetime.fromisoformat(to_utc.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass

        result = simulate_v2(
            episodes=episodes,
            snapshots=snapshots,
            cfg=cfg,
            style=style,
            from_ts=from_ts,
            to_ts=to_ts,
            market_key_filter=market_key,
        )

        save_sim_trades_jsonl(result.get("trades", []), state_dir=state_dir)

        global _last_sim_summary
        _last_sim_summary = result.get("summary")

        return result
    except Exception as ex:
        return {
            "error": str(ex),
            "trades": [],
            "summary": {},
        }


# -----------------------------------------------------------------------
# Startup — create system + start orchestrator
# -----------------------------------------------------------------------

def _resolve_git_sha() -> str:
    """
    Tenta obter git SHA: env primeiro, depois subprocess.
    Nunca levanta. Nunca chamar no import.
    """
    for key in ("RENDER_GIT_COMMIT", "GIT_SHA", "COMMIT_SHA", "SOURCE_VERSION"):
        val = os.environ.get(key, "").strip()
        if val:
            return val[:12] if len(val) > 12 else val
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _log_git_commit() -> None:
    """Log do hash do commit para auditoria. Chamar apenas no startup."""
    logger.info("GIT_COMMIT: %s", _git_sha)


def _log_boot_info() -> None:
    """Log de versão Python, seed e variáveis críticas no boot."""
    v = sys.version_info
    logger.info(
        "PYTHON_VERSION: %s.%s.%s",
        v.major, v.minor, v.micro,
    )
    logger.info("GLOBAL_SEED_SET: %s", _seed)
    env_mode = os.environ.get("ENV", os.environ.get("ENVIRONMENT", "development"))
    logger.info("ENV: %s", env_mode)
    for k in ("RUN_MODE", "PORT", "TRADING_SERVER", "TOKEN_SET"):
        val = os.environ.get(k, "<unset>")
        logger.info("[BOOT] [ENV] %s=%s", k, val)
    tokens = os.environ.get("POLYMARKET_TOKENS", "")
    n = len([t for t in tokens.split(",") if t.strip()]) if tokens else 0
    logger.info("[BOOT] [ENV] POLYMARKET_TOKENS=<%d tokens>", n)
    _log_git_commit()


@app.on_event("startup")
async def startup() -> None:
    global _boot_error, _git_sha, _build_time_utc
    os.environ["TRADING_SERVER"] = "1"
    try:
        _log_python_version()
    except Exception:
        pass
    _git_sha = _resolve_git_sha()
    from datetime import datetime, timezone
    _build_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _log_boot_info()

    logger.info("[BOOT] startup event fired")

    # --- Step 1: Create system (sem os.makedirs; create_system usa resolve_storage_paths) ---
    try:
        from src.orchestrator import create_system

        initial_capital = float(os.environ.get("INITIAL_CAPITAL", "1000.0"))

        logger.info("[BOOT] calling create_system(tokens=%d)", len(_manual_tokens))
        system = create_system(
            initial_capital=initial_capital,
            token_ids=_manual_tokens,
        )

        if _auto_universe and not _manual_tokens:
            system["auto_universe"] = True

        _system.update(system)
        logger.info("[BOOT] system created — keys=%d", len(_system))

    except Exception as e:
        _boot_error = f"create_system failed: {type(e).__name__}: {e}"
        logger.critical("[BOOT] FAILED: %s", _boot_error)
        logger.critical(traceback.format_exc())
        return

    # --- Step 2: Start orchestrator in background thread ---
    def _run() -> None:
        global _boot_error
        try:
            from src.orchestrator import run_system
            logger.info("[BOOT] run_system starting")
            run_system(_system)
        except Exception as e:
            _boot_error = f"run_system crashed: {type(e).__name__}: {e}"
            logger.critical("[BOOT] Orchestrator crashed: %s", e)
            logger.critical(traceback.format_exc())

    t = threading.Thread(target=_run, name="OrchestratorThread", daemon=True)
    t.start()

    if _auto_universe and not _manual_tokens:
        logger.info("[BOOT] AUTO_UNIVERSE enabled — markets will be discovered in background")
    else:
        logger.info("[BOOT] Manual mode — %d tokens", len(_manual_tokens))


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
