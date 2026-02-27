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

import logging
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional

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


def _resolve_tokens_fast() -> list:
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
# Health check — ALWAYS works, exposes boot errors
# -----------------------------------------------------------------------

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

        return {
            "status": "ok",
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
            "universe_source": universe.get("universe_source", "none"),
            "universe_error": universe.get("universe_error"),
            "degraded_components": _system.get("degraded_components", []),
        }
    except Exception as ex:
        return {
            "status": "ok",
            "boot_error": _boot_error,
            "health_error": str(ex),
            "git_sha": _git_sha,
            "build_time_utc": _build_time_utc,
            "resolved_state_dir": _system.get("resolved_state_dir"),
            "resolved_data_dir": _system.get("resolved_data_dir"),
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
    for k in ("RUN_MODE", "PORT", "TRADING_SERVER"):
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
