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

IMPORTANTE:
- App DEVE subir rápido (< 10s) para health check do Render
- Auto-discovery de mercados roda DEPOIS do startup, não durante import
- Se POLYMARKET_TOKENS definido, usa imediatamente sem HTTP
- Se vazio/não definido E AUTO_UNIVERSE=true (default), cria sistema real
  e descobre mercados em background via Gamma API
"""

import logging
import os
import threading
import time
import traceback
from typing import Any, Dict

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def _resolve_tokens_fast() -> list:
    """
    Resolve tokens SEM bloquear startup.

    Se POLYMARKET_TOKENS definido → usa direto (0ms).
    Se vazio → retorna [] (auto-discovery ou modo demo).
    """
    raw = os.environ.get("POLYMARKET_TOKENS", "").strip()
    if not raw:
        return []
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if tokens:
        logger.info("[SERVER] POLYMARKET_TOKENS: %d tokens", len(tokens))
    return tokens


def _is_auto_universe() -> bool:
    """Check if AUTO_UNIVERSE is enabled (default: true)."""
    return os.environ.get("AUTO_UNIVERSE", "true").strip().lower() in ("true", "1", "yes")


def _create_system_safe(token_ids: list, auto_universe: bool = False) -> Dict[str, Any]:
    """Cria system com tratamento de erro para deploy."""
    try:
        from src.orchestrator import create_system
        state_dir = os.environ.get(
            "STATE_DIR",
            os.path.join(os.path.expanduser("~"), ".polymarket_bot"),
        )
        os.makedirs(state_dir, exist_ok=True)

        data_dir = os.environ.get("DATA_DIR", "data")
        os.makedirs(data_dir, exist_ok=True)

        initial_capital = float(os.environ.get("INITIAL_CAPITAL", "1000.0"))
        system = create_system(
            initial_capital=initial_capital,
            token_ids=token_ids,
            state_dir=state_dir,
        )
        if auto_universe:
            system["auto_universe"] = True
        return system
    except Exception as e:
        logger.error("[SERVER] create_system FAILED: %s", e)
        logger.error(traceback.format_exc())
        return {}


def _create_mock_safe() -> Dict[str, Any]:
    """Mock system para modo demo."""
    try:
        from src.api.dashboard_server import _create_mock_system
        return _create_mock_system()
    except Exception as e:
        logger.error("[SERVER] mock system FAILED: %s", e)
        return {}


# -----------------------------------------------------------------------
# Criar app — RÁPIDO, sem HTTP calls
# -----------------------------------------------------------------------

token_ids = _resolve_tokens_fast()
auto_universe = _is_auto_universe()

if token_ids:
    # Manual mode: user provided specific tokens
    logger.info("[SERVER] Criando system com %d tokens (manual)", len(token_ids))
    _system = _create_system_safe(token_ids)
    _universe_source = "manual"
elif auto_universe:
    # Auto-universe mode: create real system, discover markets in background
    logger.info("[SERVER] AUTO_UNIVERSE=true — criando sistema real (discovery em background)")
    _system = _create_system_safe([], auto_universe=True)
    _universe_source = "auto"
else:
    # Demo mode: no tokens, no discovery
    logger.info("[SERVER] Sem POLYMARKET_TOKENS e AUTO_UNIVERSE=false — modo demo")
    _system = _create_mock_safe()
    _universe_source = "none"

# Set universe source in metrics
try:
    from src.metrics import set_universe_source
    set_universe_source(_universe_source)
except Exception:
    pass

try:
    from src.api.dashboard_server import create_dashboard_app
    app = create_dashboard_app(_system)
except Exception as e:
    logger.error("[SERVER] Dashboard creation FAILED: %s", e)
    app = FastAPI(title="Trading System - Fallback")

    @app.get("/")
    async def fallback_root():
        return {"status": "error", "message": "Dashboard failed to initialize", "error": str(e)}


# -----------------------------------------------------------------------
# Health check — SEMPRE deve funcionar, com observabilidade completa
# -----------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check para Render com observabilidade completa."""
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
            "token_count": metrics.get("markets_subscribed", 0),
            "markets_subscribed": metrics.get("markets_subscribed", 0),
            "snapshots_received": metrics.get("snapshots_received", 0),
            "last_snapshot_age_seconds": last_age,
            "ws_disconnects_total": universe.get("ws_disconnects_total", 0),
            "gaps_total": universe.get("gaps_total", 0),
            "universe_last_refresh_timestamp": universe.get("universe_last_refresh_timestamp"),
            "universe_source": universe.get("universe_source", "none"),
            "universe_error": universe.get("universe_error"),
            "has_system": bool(_system),
        }
    except Exception:
        return {"status": "ok"}


# -----------------------------------------------------------------------
# Startup — inicia orchestrator em background
# -----------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    """Inicia orchestrator em background thread."""
    os.environ["TRADING_SERVER"] = "1"

    has_tokens = bool(_system and _system.get("token_ids"))
    has_auto = bool(_system and _system.get("auto_universe"))

    if not _system or (not has_tokens and not has_auto):
        logger.warning("[SERVER] Sem tokens e sem auto_universe — orchestrator não iniciado")
        return

    def _run() -> None:
        try:
            from src.orchestrator import run_system
            run_system(_system)
        except Exception as e:
            logger.error("[SERVER] Orchestrator crashed: %s", e)
            logger.error(traceback.format_exc())

    t = threading.Thread(target=_run, name="OrchestratorThread", daemon=True)
    t.start()
    mode = "auto_universe" if has_auto and not has_tokens else f"tokens={len(_system.get('token_ids', []))}"
    logger.info("[SERVER] [ORCHESTRATOR_STARTED] mode=%s", mode)


# -----------------------------------------------------------------------
# Ponto de entrada direto
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
