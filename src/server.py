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
import threading
from typing import Any, Dict

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
    return {
        "status": "ok",
        "markets_subscribed": metrics.get("markets_subscribed", 0),
        "snapshots_received": metrics.get("snapshots_received", 0),
        "engine_events": metrics.get("engine_events", 0),
        "run_mode": cfg.run_mode,
    }


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


@app.on_event("startup")
async def startup() -> None:
    """Inicia orchestrator em background (run_system com loop)."""
    os.environ["TRADING_SERVER"] = "1"
    if not _system or not _system.get("token_ids"):
        logger.warning("[SERVER] Sem tokens — orchestrator não iniciado")
        return

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
