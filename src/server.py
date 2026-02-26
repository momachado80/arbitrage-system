"""
Servidor unificado — FastAPI + Dashboard + /health para Render
=============================================================

Um único processo: uvicorn serve FastAPI com dashboard e /health.
Orchestrator roda em thread em background.

Uso:
    uvicorn src.server:app --host 0.0.0.0 --port $PORT

Render:
    Procfile: web: uvicorn src.server:app --host 0.0.0.0 --port $PORT
    Health check path: /health
"""

import logging
import os
import sys
import threading
from typing import Any, Dict

# -----------------------------------------------------------------------------
# Fail fast: verificar versão Python antes de qualquer import pesado
# -----------------------------------------------------------------------------

_REQUIRED_PYTHON = (3, 11)
_REQUIRED_MAX = (3, 12)  # Rejeitar 3.12+ para garantia 3.11.x


def _check_python_version() -> None:
    """Garante Python 3.11.x exclusivamente. Rejeita 3.12+."""
    v = sys.version_info
    if (v.major, v.minor) < _REQUIRED_PYTHON or (v.major, v.minor) >= _REQUIRED_MAX:
        raise RuntimeError(
            "Unsupported Python version. Required: 3.11.x. "
            f"Current: {v.major}.{v.minor}.{v.micro}"
        )


_check_python_version()


from src.seed import set_global_seed

_seed = int(os.environ.get("RANDOM_SEED", "42"))
set_global_seed(_seed)

from fastapi import FastAPI

from src.api.dashboard_server import create_dashboard_app
from src.config.config import load_app_config_from_env
from src.metrics import get_ingestion_metrics, get_decision_metrics
from src.orchestrator import create_system, run_system, _resolve_token_ids

logger = logging.getLogger(__name__)

# Estado global (preenchido antes de criar app)
_system: Dict[str, Any] = {}


def _health_handler():
    """GET /health para Render."""
    cfg = load_app_config_from_env()
    metrics = get_ingestion_metrics()
    out = {
        "status": "ok",
        "markets_subscribed": metrics.get("markets_subscribed", 0),
        "snapshots_received": metrics.get("snapshots_received", 0),
        "engine_events": metrics.get("engine_events", 0),
        "run_mode": cfg.run_mode,
    }
    if _system:
        out["resolved_state_dir"] = _system.get("resolved_state_dir")
        out["resolved_data_dir"] = _system.get("resolved_data_dir")
        out["boot_error"] = _system.get("boot_error")
    else:
        out["resolved_state_dir"] = None
        out["resolved_data_dir"] = None
        out["boot_error"] = None
    return out


# Criar app (com system se tokens disponíveis)
token_ids = _resolve_token_ids()
if not token_ids:
    logger.critical("[SERVER] [NO_MARKETS] Usando modo demo")
    from src.api.dashboard_server import _create_mock_system
    _system = _create_mock_system()
else:
    initial_capital = float(os.environ.get("INITIAL_CAPITAL", "1000.0"))
    _system = create_system(
        initial_capital=initial_capital,
        token_ids=token_ids,
    )

app = create_dashboard_app(_system)


@app.get("/health")
def health():
    """Health check para Render. Path: /health"""
    try:
        return _health_handler()
    except Exception:
        return {"status": "ok"}


def _log_git_commit() -> None:
    """Log do hash do commit para auditoria."""
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        logger.info("GIT_COMMIT: %s", commit)
    except Exception:
        logger.warning("Could not retrieve git commit hash")


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
    """Inicia orchestrator em background (run_system com loop)."""
    _log_boot_info()
    os.environ["TRADING_SERVER"] = "1"
    token_ids = _system.get("token_ids", []) if _system else []
    n_tokens = len(token_ids) if token_ids else 0
    logger.info(
        "[SERVER] Iniciando orchestrator (token_ids=%d, auto-discovery habilitado)",
        n_tokens,
    )

    def _run() -> None:
        run_system(_system)

    t = threading.Thread(target=_run, name="OrchestratorThread", daemon=True)
    t.start()
    logger.info("[SERVER] [ORCHESTRATOR_STARTED]")


# -----------------------------------------------------------------------------
# Ponto de entrada para execução direta (python -m src.server)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
