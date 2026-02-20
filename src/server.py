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
- Se vazio/não definido, sobe em modo demo e descobre mercados em background
"""

import logging
import os
import threading
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
    Se vazio → retorna [] (modo demo). Discovery roda depois.
    """
    raw = os.environ.get("POLYMARKET_TOKENS", "").strip()
    if not raw:
        return []
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if tokens:
        logger.info("[SERVER] POLYMARKET_TOKENS: %d tokens", len(tokens))
    return tokens


def _create_system_safe(token_ids: list) -> Dict[str, Any]:
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

if token_ids:
    logger.info("[SERVER] Criando system com %d tokens", len(token_ids))
    _system = _create_system_safe(token_ids)
else:
    logger.info("[SERVER] Sem POLYMARKET_TOKENS — modo demo")
    _system = _create_mock_safe()

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
# Health check — SEMPRE deve funcionar
# -----------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check para Render. Path: /health"""
    try:
        from src.config.config import load_app_config_from_env
        from src.metrics import get_ingestion_metrics
        cfg = load_app_config_from_env()
        metrics = get_ingestion_metrics()
        return {
            "status": "ok",
            "run_mode": cfg.run_mode,
            "markets_subscribed": metrics.get("markets_subscribed", 0),
            "snapshots_received": metrics.get("snapshots_received", 0),
            "has_system": bool(_system),
            "token_count": len(token_ids),
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

    if not _system or not _system.get("token_ids"):
        logger.warning("[SERVER] Sem tokens — orchestrator não iniciado")
        # Tentar auto-discovery em background para próximo restart
        _try_background_discovery()
        return

    def _run() -> None:
        try:
            from src.orchestrator import run_system
            run_system(_system)
        except Exception as e:
            logger.error("[SERVER] Orchestrator crashed: %s", e)

    t = threading.Thread(target=_run, name="OrchestratorThread", daemon=True)
    t.start()
    logger.info("[SERVER] [ORCHESTRATOR_STARTED] tokens=%d", len(_system.get("token_ids", [])))


def _try_background_discovery() -> None:
    """Tenta descobrir mercados em background (não bloqueia startup)."""
    def _discover():
        try:
            from src.orchestrator import _resolve_token_ids
            tokens = _resolve_token_ids()
            if tokens:
                logger.info("[SERVER] [BG_DISCOVERY] Encontrou %d mercados (usar em próximo restart)", len(tokens))
            else:
                logger.info("[SERVER] [BG_DISCOVERY] Nenhum mercado encontrado")
        except Exception as e:
            logger.warning("[SERVER] [BG_DISCOVERY] Failed: %s", e)

    t = threading.Thread(target=_discover, name="BGDiscovery", daemon=True)
    t.start()


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
