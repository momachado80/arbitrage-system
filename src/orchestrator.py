"""
Orquestrador Principal — Sistema de Trading
=============================================

Instancia e coordena todos os componentes do sistema:

COMPONENTES CORE (sync):
- CapitalManager        → contabilidade de capital
- ProbabilityCalibrator → calibragem de sizing
- TradeLedger           → registro de trades
- OrderManager          → gestão de ordens (SPOT_ONLY)
- RiskManager           → limites de exposição
- PositionManager       → posições spot
- OrderReconciler       → reconciliação com exchange

THREADS:
- Watcher           → daemon thread (asyncio event loop)
- MarketDataEngine  → thread normal (consumidor)
- ReconcilerLoop    → thread normal (reconciliação periódica)

GRACEFUL SHUTDOWN (ordem):
1. Parar Watcher (shutdown async)
2. Parar MarketDataEngine (stop + join)
3. Parar ReconcilerLoop (stop + join)
4. save_state() final
5. Log "[SYSTEM] Shutdown completo"

REGRAS:
- Core nunca importa asyncio
- Watcher nunca chama pipeline diretamente
- Nenhuma thread daemon exceto Watcher
- Nenhum estado global
- Logs prefixo [SYSTEM] ou [MARKET_ENGINE]
"""

import asyncio
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

# --- Core (sem asyncio) ---
from src.risk.capital_manager import CapitalManager
from src.risk.position_manager import PositionManager
from src.risk.risk_manager import RiskManager
from src.engine.probability_calibrator import ProbabilityCalibrator
from src.engine.trade_ledger import TradeLedger
from src.execution.order_manager import OrderManager
from src.execution.reconciler import OrderReconciler, ExchangeAPI

# --- Config ---
from src.config.config import AppConfig, load_app_config_from_env
from src.config.config_snapshot import ConfigSnapshotManager

# --- Safety ---
from src.safety.safety_state import (
    SafetyState,
    SAFE_MODE, THREAD_DEAD, EXCHANGE_UNREACHABLE,
    INVARIANT_FAIL, CAPITAL_MISMATCH, CONFIG_CHANGED,
)
from src.safety.invariants import RuntimeInvariantsEngine
from src.safety.safety_ack import SafetyAckStore
from src.safety.audit_report import build_audit_report
from src.safety.health_checks import HealthChecker

# --- Engine ---
from src.engine.market_data import MarketDataEngine

# --- Execution stubs ---
from src.execution.mode_stubs import ObserveOrderManager, PaperExchangeAPI
from src.execution.audit_logger import SimulationAuditLogger
from src.execution.realistic_simulator import (
    RealisticSimulationExchangeAPI,
    RealisticSimulationOrderManager,
    SimulationStats,
)

# --- Watcher (async, isolado) ---
from src.watcher.bridge import WatcherBridge, MarketSnapshot
from src.watcher.client import PolymarketClient

# --- Strategy ---
from src.strategy.strategy_engine import StrategyEngine
from src.strategy.decision_pipeline import DecisionPipeline

# --- Oracle ---
from src.oracle.probability_oracle import (
    ProbabilityOracle,
    MockOracleProvider,
    OraclePerformanceTracker,
    EdgeDecayTracker,
)
from src.oracle.heuristic_oracle import HeuristicMicroOracle
from src.oracle.metrics import OracleMetricsStore

# --- Universe ---
from src.universe.universe_builder import UniverseBuilder

# --- Research ---
from src.research.edge_validation import EdgeValidator

# --- Market Universe (auto-discovery) ---
from src.market.universe_manager import MarketUniverseManager
from src.metrics import set_markets_subscribed, run_sanity_checks

logger = logging.getLogger(__name__)

# ======================================================================
# Configuração (defaults — AppConfig override)
# ======================================================================

DEFAULT_INITIAL_CAPITAL: float = 1000.0
DEFAULT_STATE_DIR: str = os.environ.get(
    "STATE_DIR",
    os.path.join(os.path.expanduser("~"), ".polymarket_bot"),
)
DEFAULT_DATA_DIR: str = os.environ.get("DATA_DIR", "data")
HEARTBEAT_INTERVAL_SECONDS: int = 60
# Fallback quando config não disponível
RECONCILER_INTERVAL_SECONDS: int = 30


# ======================================================================
# ExchangeAPI Stub (substituir por implementação real)
# ======================================================================

class ExchangeAPIStub(ExchangeAPI):
    """
    Stub da API de exchange para desenvolvimento.

    Em produção, substituir por implementação real que chama
    a API REST do Polymarket CLOB.
    """

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        return {"order_id": order_id, "status": "PENDING", "filled_size": 0.0}

    def get_fills(self, order_id: str) -> List[Dict[str, Any]]:
        return []

    def get_position(self, market_id: str) -> Dict[str, Any]:
        return {"market_id": market_id, "size": 0.0, "avg_entry": 0.0}


# ======================================================================
# PipelineHandler — adapta MarketSnapshot para o core
# ======================================================================

class PipelineHandler:
    """
    Handler que recebe MarketSnapshot do MarketDataEngine.

    Armazena o último snapshot por token_id (thread-safe).
    Integra com StrategyEngine e DecisionPipeline para gerar trades.
    """

    def __init__(
        self,
        strategy_engine: Optional[StrategyEngine] = None,
        decision_pipeline: Optional[DecisionPipeline] = None,
    ) -> None:
        self._latest: Dict[str, MarketSnapshot] = {}
        self._lock = threading.Lock()
        self._update_count: int = 0
        self._strategy_engine = strategy_engine
        self._decision_pipeline = decision_pipeline

    def process_market_update(self, event: MarketSnapshot) -> None:
        """Processa snapshot de mercado. Thread-safe."""
        with self._lock:
            self._latest[event.token_id] = event
            self._update_count += 1

        logger.debug(
            f"[MARKET_ENGINE] Snapshot token={event.token_id} "
            f"bid={event.best_bid_price} ask={event.best_ask_price}"
        )

        # Alimentar strategy engine
        if self._strategy_engine is not None:
            try:
                signal = self._strategy_engine.process_snapshot(event)
                if signal is not None and self._decision_pipeline is not None:
                    self._decision_pipeline.process_signal(signal)
            except Exception as e:
                logger.error(
                    "[MARKET_ENGINE] Strategy error: %s: %s",
                    type(e).__name__, e,
                )

    def get_latest(self, token_id: str) -> Optional[MarketSnapshot]:
        """Retorna último snapshot para um token."""
        with self._lock:
            return self._latest.get(token_id)

    @property
    def update_count(self) -> int:
        """Total de snapshots processados."""
        with self._lock:
            return self._update_count


# ======================================================================
# ReconcilerLoop — thread de reconciliação periódica
# ======================================================================

class ReconcilerLoop(threading.Thread):
    """
    Thread que executa reconciliação periódica com a exchange
    e recebe notificações de trade do MarketDataEngine.

    Não é daemon. Deve ser encerrado via stop() + join().
    """

    def __init__(
        self,
        reconciler: OrderReconciler,
        order_manager: OrderManager,
        trade_ledger: TradeLedger,
        exchange_api: ExchangeAPI,
        capital_manager: Optional[CapitalManager] = None,
        position_manager: Optional[PositionManager] = None,
        safety_state: Optional[SafetyState] = None,
        invariants_engine: Optional[RuntimeInvariantsEngine] = None,
        interval: int = 30,
    ) -> None:
        super().__init__(name="ReconcilerLoop", daemon=False)
        self._reconciler = reconciler
        self._order_manager = order_manager
        self._trade_ledger = trade_ledger
        self._exchange_api = exchange_api
        self._capital_manager = capital_manager
        self._position_manager = position_manager
        self._safety_state = safety_state
        self._invariants_engine = invariants_engine
        self._interval = interval
        self.running: bool = True
        self._trade_events: List[Any] = []
        self._lock = threading.Lock()
        self._cycles: int = 0

    def notify_trade_event(self, event: Any) -> None:
        """
        Recebe notificação de trade do MarketDataEngine.
        Thread-safe (chamado de outra thread).
        """
        with self._lock:
            self._trade_events.append(event)
        logger.debug(
            f"[SYSTEM:RECONCILER] Trade event: "
            f"token={getattr(event, 'token_id', '?')} "
            f"side={getattr(event, 'side', '?')}"
        )

    def run(self) -> None:
        """Loop de reconciliação periódica."""
        logger.info("[RECONCILER] [STARTED] Loop iniciado")

        while self.running:
            try:
                self._reconciler.reconcile_open_trades(
                    order_manager=self._order_manager,
                    trade_ledger=self._trade_ledger,
                    exchange_api=self._exchange_api,
                    capital_manager=self._capital_manager,
                    position_manager=self._position_manager,
                    safety_state=self._safety_state,
                    invariants_engine=self._invariants_engine,
                )
                self._cycles += 1
            except Exception as e:
                logger.error(
                    f"[SYSTEM:RECONCILER:ERROR] {type(e).__name__}: {e}"
                )

            # Mitigação ativa: se SAFE_MODE, cancelar BUY open
            if (
                self._safety_state is not None
                and self._safety_state.global_safe_mode
            ):
                cancelled = self._order_manager.cancel_all_buy_open_orders_best_effort()
                if cancelled:
                    logger.warning(
                        f"[SYSTEM:RECONCILER] SAFE_MODE mitigation: "
                        f"cancelled {len(cancelled)} BUY orders"
                    )

            # Sleep interruptível (1s intervals)
            for _ in range(self._interval):
                if not self.running:
                    break
                time.sleep(1)

        logger.info(
            f"[SYSTEM:RECONCILER] Loop encerrado — {self._cycles} ciclos"
        )

    def stop(self) -> None:
        """Sinaliza encerramento do loop."""
        self.running = False


# ======================================================================
# Watcher thread helper (asyncio isolado)
# ======================================================================

def _run_watcher_loop(
    client: PolymarketClient,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Executa o event loop asyncio do watcher em thread daemon."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(client.run_forever())
    except Exception as e:
        logger.error(f"[SYSTEM:WATCHER] Erro fatal: {e}")
    finally:
        loop.close()


# ======================================================================
# Orquestrador
# ======================================================================

def _can_enter_live(config: AppConfig, safety_state: SafetyState) -> bool:
    """Verifica pré-condições para LIVE mode."""
    if safety_state.buy_blocked:
        return False
    if safety_state.global_safe_mode:
        return False
    if safety_state.invariants_ok_streak < config.invariant_strikes_to_clear:
        return False
    if safety_state.capital_mismatch_ok_streak < config.trade_drop_strikes_to_clear:
        return False
    if safety_state.reasons:
        return False
    if CONFIG_CHANGED in safety_state.reasons:
        return False
    return True


def create_system(
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    state_dir: str = DEFAULT_STATE_DIR,
    exchange_api: Optional[ExchangeAPI] = None,
    token_ids: Optional[List[str]] = None,
    config: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    """
    Instancia todos os componentes na ordem correta.

    Se config=None, carrega de ENV via load_app_config_from_env().
    RUN_MODE=LIVE só efetivo se pré-condições atendidas; senão downgrade para OBSERVE.
    """
    cfg = config or load_app_config_from_env()
    logger.info(f"[SYSTEM] Inicializando componentes... RUN_MODE={cfg.run_mode}")

    # 1. Core managers
    capital_manager = CapitalManager(initial_capital=initial_capital)
    calibrator = ProbabilityCalibrator(
        state_path=os.path.join(state_dir, "calibrator_state.json"),
    )
    position_manager = PositionManager(
        state_path=os.path.join(state_dir, "position_state.json"),
    )
    trade_ledger = TradeLedger(
        state_path=os.path.join(state_dir, "trade_ledger.json"),
        calibrator=calibrator,
        capital_manager=capital_manager,
    )
    risk_manager = RiskManager(config={
        "max_exposure_total": initial_capital,
        "max_exposure_per_category": initial_capital * 0.5,
        "max_exposure_per_market": initial_capital * 0.2,
        "max_open_trades": 50,
        "total_capital": initial_capital,
        "max_percent_capital_deadzone": 0.5,
    })
    reconciler = OrderReconciler()

    # 2. Safety layer (antes de config snapshot)
    safety_state = SafetyState()
    ack_store = SafetyAckStore(
        state_path=os.path.join(state_dir, "safety_ack.json"),
    )
    config_snapshot = ConfigSnapshotManager(
        state_path=os.path.join(state_dir, "config_snapshot.json"),
    )
    config_snapshot.initialize_or_validate(cfg, safety_state)

    invariants_engine = RuntimeInvariantsEngine(
        capital_manager=capital_manager,
        position_manager=position_manager,
        trade_ledger=trade_ledger,
        safety_state=safety_state,
    )

    # 3. Queue + Watcher (precisa antes de REALISTIC_SIMULATION)
    update_queue: queue.Queue = queue.Queue(maxsize=cfg.queue_maxsize)  # type: ignore[type-arg]
    bridge = WatcherBridge(update_queue)
    ws_client = PolymarketClient(bridge)

    # 4. RUN_MODE: effective mode e stubs
    effective_mode = cfg.run_mode
    if cfg.run_mode == "LIVE":
        if not _can_enter_live(cfg, safety_state):
            logger.warning(
                "[SYSTEM] LIVE_MODE_DENIED — pré-condições não atendidas, "
                "downgrade para OBSERVE"
            )
            effective_mode = "OBSERVE"

    sim_stats = None
    if effective_mode == "OBSERVE":
        order_manager: Any = ObserveOrderManager()
        api: ExchangeAPI = ExchangeAPIStub()
    elif effective_mode == "PAPER":
        order_manager = OrderManager(account_mode="SPOT_ONLY")
        api = PaperExchangeAPI(order_manager)
    elif effective_mode == "REALISTIC_SIMULATION":
        audit_log = SimulationAuditLogger()
        sim_stats = SimulationStats()
        strict = os.environ.get("STRICT_REALISM", "").strip().upper() in ("1", "TRUE", "YES")
        book_provider = lambda tid: ws_client.get_book_snapshot(tid)
        order_manager = RealisticSimulationOrderManager(
            book_provider=book_provider,
            audit_logger=audit_log,
            strict_realism=strict,
            stats=sim_stats,
        )
        api = RealisticSimulationExchangeAPI(order_manager)
    else:
        order_manager = OrderManager(account_mode="SPOT_ONLY")
        api = exchange_api or ExchangeAPIStub()

    logger.info(f"[SYSTEM] RUN_MODE efetivo: {effective_mode}")

    # 5. Oracle + Strategy Engine + Decision Pipeline
    heuristic_oracle = HeuristicMicroOracle()
    oracle_provider = heuristic_oracle  # V1: microstructure-based

    probability_oracle = ProbabilityOracle(
        provider=oracle_provider,
        performance_tracker=OraclePerformanceTracker(),
        edge_decay_tracker=EdgeDecayTracker(),
    )
    universe_builder = UniverseBuilder(
        manual_tokens=token_ids or [],
        oracle_provider=oracle_provider,
    )

    # Metrics + Validation (persistência leve)
    data_dir = DEFAULT_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    oracle_metrics_store = OracleMetricsStore(
        metrics_path=os.path.join(data_dir, "oracle_metrics.jsonl"),
    )
    edge_validator = EdgeValidator(
        log_path=os.path.join(data_dir, "edge_validation.jsonl"),
    )

    strategy_engine = StrategyEngine(
        safety_state=safety_state,
        oracle=probability_oracle,
    )
    decision_pipeline = DecisionPipeline(
        run_mode=effective_mode,
        safety_state=safety_state,
        risk_manager=risk_manager,
        calibrator=calibrator,
        capital_manager=capital_manager,
        position_manager=position_manager,
        order_manager=order_manager,
        trade_ledger=trade_ledger,
    )

    # 6. Pipeline handler + Reconciler loop
    pipeline_handler = PipelineHandler(
        strategy_engine=strategy_engine,
        decision_pipeline=decision_pipeline,
    )
    reconciler_loop = ReconcilerLoop(
        reconciler=reconciler,
        order_manager=order_manager,
        trade_ledger=trade_ledger,
        exchange_api=api,
        capital_manager=capital_manager,
        position_manager=position_manager,
        safety_state=safety_state,
        invariants_engine=invariants_engine,
        interval=cfg.reconcile_loop_interval_seconds,
    )

    # 6. MarketDataEngine
    market_engine = MarketDataEngine(
        update_queue=update_queue,
        pipeline=pipeline_handler,
        reconciler=reconciler_loop,
    )

    logger.info("[SYSTEM] Componentes inicializados")

    return {
        "capital_manager": capital_manager,
        "calibrator": calibrator,
        "position_manager": position_manager,
        "trade_ledger": trade_ledger,
        "order_manager": order_manager,
        "risk_manager": risk_manager,
        "reconciler": reconciler,
        "safety_state": safety_state,
        "ack_store": ack_store,
        "config": cfg,
        "config_snapshot": config_snapshot,
        "invariants_engine": invariants_engine,
        "update_queue": update_queue,
        "bridge": bridge,
        "ws_client": ws_client,
        "pipeline_handler": pipeline_handler,
        "reconciler_loop": reconciler_loop,
        "market_engine": market_engine,
        "token_ids": token_ids or [],
        "exchange_api": api,
        "simulation_stats": sim_stats if effective_mode == "REALISTIC_SIMULATION" else None,
        "strategy_engine": strategy_engine,
        "decision_pipeline": decision_pipeline,
        "reversion_metrics_collector": strategy_engine.metrics_collector,
        "probability_oracle": probability_oracle,
        "universe_builder": universe_builder,
        "oracle_provider": oracle_provider,
        "heuristic_oracle": heuristic_oracle,
        "oracle_metrics_store": oracle_metrics_store,
        "edge_validator": edge_validator,
    }


def _start_dashboard_thread(system: Dict[str, Any], config: AppConfig) -> None:
    """Inicia o dashboard em thread separada. Não bloqueia o main loop."""
    import uvicorn
    from src.api.dashboard_server import create_dashboard_app

    port = getattr(config, "dashboard_port", 8765)
    app = create_dashboard_app(system)

    def _run() -> None:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", str(port))),
            log_level="warning",
        )

    t = threading.Thread(target=_run, name="DashboardThread", daemon=True)
    t.start()
    logger.info(f"[SYSTEM] Dashboard iniciado em http://127.0.0.1:{port}")


def run_system(system: Dict[str, Any]) -> None:
    """
    Inicia todas as threads e entra no main loop.

    Shutdown via KeyboardInterrupt (Ctrl+C).
    Health check a cada health_check_interval_seconds.
    """
    system["system_start_time"] = time.time()
    ws_client: PolymarketClient = system["ws_client"]
    market_engine: MarketDataEngine = system["market_engine"]
    reconciler_loop: ReconcilerLoop = system["reconciler_loop"]
    token_ids: List[str] = system["token_ids"]
    cfg: AppConfig = system.get("config") or load_app_config_from_env()
    health_interval = cfg.health_check_interval_seconds

    if not token_ids:
        logger.critical("[SYSTEM] Nenhum market ativo — engine não iniciada")
        return

    set_markets_subscribed(len(token_ids))

    # --- Start threads ---

    # 1. Watcher (daemon thread com event loop asyncio)
    watcher_loop = asyncio.new_event_loop()

    # Subscribe tokens antes de iniciar
    for token_id in token_ids:
        watcher_loop.run_until_complete(ws_client.subscribe_market(token_id))

    watcher_thread = threading.Thread(
        target=_run_watcher_loop,
        args=(ws_client, watcher_loop),
        name="WatcherThread",
        daemon=True,
    )
    watcher_thread.start()
    logger.info("[WATCHER] [STARTED] Thread iniciada (daemon)")
    system["watcher_thread"] = watcher_thread
    system["build_audit_report"] = build_audit_report

    # 2. ReconcilerLoop (thread normal)
    reconciler_loop.start()
    logger.info("[RECONCILER] [THREAD_STARTED] ReconcilerLoop")

    # 3. MarketDataEngine (thread normal)
    market_engine.start()
    logger.info("[ENGINE] [THREAD_STARTED] MarketDataEngine")

    # 4. Health checker
    health_checker = HealthChecker(
        exchange_api=system.get("exchange_api"),
        safety_state=system["safety_state"],
        config=cfg,
        market_engine=market_engine,
        reconciler_loop=reconciler_loop,
        watcher_thread=watcher_thread,
    )

    # 5. Dashboard (thread separada, se habilitado — skip se rodando via server)
    if getattr(cfg, "enable_dashboard", False) and not os.getenv("TRADING_SERVER"):
        _start_dashboard_thread(system, cfg)

    # --- Main loop: 1s sleep + counters ---
    logger.info("[SYSTEM] Sistema operacional — Ctrl+C para encerrar")
    counter = 0

    try:
        while True:
            time.sleep(1)
            counter += 1

            if counter % health_interval == 0:
                health_checker.run_health_check()
                run_sanity_checks(system_start_time=system.get("system_start_time"))

            if counter % HEARTBEAT_INTERVAL_SECONDS == 0:
                _log_heartbeat(system, watcher_thread)

    except KeyboardInterrupt:
        logger.info("[SYSTEM] KeyboardInterrupt — iniciando shutdown...")

    # --- Graceful shutdown ---
    _graceful_shutdown(system, watcher_loop, watcher_thread)


def _log_heartbeat(
    system: Dict[str, Any],
    watcher_thread: threading.Thread,
) -> None:
    """Log heartbeat estruturado com telemetria financeira."""
    import json as _json

    market_engine: MarketDataEngine = system["market_engine"]
    reconciler_loop: ReconcilerLoop = system["reconciler_loop"]
    pipeline_handler: PipelineHandler = system["pipeline_handler"]
    update_queue: queue.Queue = system["update_queue"]  # type: ignore[type-arg]
    safety_state: Optional[SafetyState] = system.get("safety_state")

    threads_alive = {
        "watcher": watcher_thread.is_alive(),
        "market_engine": market_engine.is_alive(),
        "reconciler": reconciler_loop.is_alive(),
    }

    dead = [name for name, alive in threads_alive.items() if not alive]

    # Marcar threads mortas no safety_state
    if dead and safety_state is not None:
        from src.safety.safety_state import THREAD_DEAD
        safety_state.block_buy(THREAD_DEAD)

    # Construir telemetria
    safety_snap = safety_state.snapshot() if safety_state else {}
    last_rec_ts = safety_snap.get("last_reconcile_ok_ts")
    rec_age = (time.time() - last_rec_ts) if last_rec_ts else None

    telemetry = {
        "queue_size": update_queue.qsize(),
        "snapshots_processed": pipeline_handler.update_count,
        "engine_events": market_engine.events_processed,
        "engine_consecutive_errors": market_engine.consecutive_errors,
        "reconciler_cycles": reconciler_loop._cycles,
        "last_reconcile_ok_age_seconds": round(rec_age, 1) if rec_age else None,
        "buy_blocked": safety_snap.get("buy_blocked", False),
        "buy_block_reasons": safety_snap.get("buy_block_reasons", []),
        "invariant_fail_count": safety_snap.get("invariant_fail_count", 0),
        "threads_dead": dead,
    }

    if dead:
        logger.warning(
            f"[SYSTEM:HEARTBEAT] {_json.dumps(telemetry)}"
        )
    else:
        logger.info(
            f"[SYSTEM:HEARTBEAT] {_json.dumps(telemetry)}"
        )

    # Audit report quando safe_mode ativo
    if safety_state is not None and safety_state.global_safe_mode:
        report = build_audit_report(
            capital_manager=system.get("capital_manager"),
            position_manager=system.get("position_manager"),
            trade_ledger=system.get("trade_ledger"),
            safety_state=safety_state,
        )
        logger.critical(f"[SYSTEM:SAFE_MODE_REPORT]\n{report}")


def _graceful_shutdown(
    system: Dict[str, Any],
    watcher_loop: asyncio.AbstractEventLoop,
    watcher_thread: threading.Thread,
) -> None:
    """Encerra tudo na ordem correta."""
    ws_client: PolymarketClient = system["ws_client"]
    market_engine: MarketDataEngine = system["market_engine"]
    reconciler_loop: ReconcilerLoop = system["reconciler_loop"]
    position_manager: PositionManager = system["position_manager"]
    calibrator: ProbabilityCalibrator = system["calibrator"]

    # 1. Parar Watcher
    logger.info("[SYSTEM] Parando Watcher...")
    try:
        future = asyncio.run_coroutine_threadsafe(
            ws_client.shutdown(), watcher_loop
        )
        future.result(timeout=5)
    except Exception:
        pass  # Daemon — será encerrado com o processo
    logger.info("[SYSTEM] Watcher parado")

    # 2. Parar MarketDataEngine
    logger.info("[SYSTEM] Parando MarketDataEngine...")
    market_engine.stop()
    market_engine.join(timeout=5)
    if market_engine.is_alive():
        logger.warning("[SYSTEM] MarketDataEngine não encerrou a tempo")
    else:
        logger.info("[SYSTEM] MarketDataEngine parado")

    # 3. Parar ReconcilerLoop
    logger.info("[SYSTEM] Parando ReconcilerLoop...")
    reconciler_loop.stop()
    reconciler_loop.join(timeout=5)
    if reconciler_loop.is_alive():
        logger.warning("[SYSTEM] ReconcilerLoop não encerrou a tempo")
    else:
        logger.info("[SYSTEM] ReconcilerLoop parado")

    # 4. Save state final
    logger.info("[SYSTEM] Salvando estado final...")
    try:
        position_manager.save_state()
    except Exception as e:
        logger.error(f"[SYSTEM] Erro ao salvar PositionManager: {e}")
    try:
        calibrator.save_state()
    except Exception as e:
        logger.error(f"[SYSTEM] Erro ao salvar Calibrator: {e}")

    # 5. Audit report final
    safety_state: Optional[SafetyState] = system.get("safety_state")
    report = build_audit_report(
        capital_manager=system.get("capital_manager"),
        position_manager=system.get("position_manager"),
        trade_ledger=system.get("trade_ledger"),
        safety_state=safety_state,
    )
    logger.info(f"[SYSTEM:SHUTDOWN_REPORT]\n{report}")

    # 6. Confirmar shutdown
    logger.info("[SYSTEM] Shutdown completo")


# ======================================================================
# Safety ACK — limpar reasons manuais com precondições
# ======================================================================

def ack_reason(
    reason: str,
    operator_note: str,
    system: Dict[str, Any],
    watcher_thread: Optional[threading.Thread] = None,
) -> bool:
    """
    Reconhece e limpa uma razão de bloqueio com verificação de precondições.

    Args:
        reason: a razão a limpar
        operator_note: nota do operador para auditoria
        system: dict do sistema (de create_system)
        watcher_thread: referência à thread do watcher (para THREAD_DEAD)

    Returns:
        True se limpou com sucesso, False se precondições não atendidas.
    """
    ss: SafetyState = system["safety_state"]
    ack_store: SafetyAckStore = system["ack_store"]
    reconciler_loop: ReconcilerLoop = system["reconciler_loop"]
    market_engine: MarketDataEngine = system["market_engine"]
    exchange_api = system.get("exchange_api")

    preconditions: Dict[str, Any] = {}

    if reason == THREAD_DEAD:
        # Precondição: todas threads críticas is_alive()
        threads_ok = {
            "market_engine": market_engine.is_alive(),
            "reconciler": reconciler_loop.is_alive(),
        }
        if watcher_thread is not None:
            threads_ok["watcher"] = watcher_thread.is_alive()
        all_alive = all(threads_ok.values())
        preconditions = {"threads_alive": threads_ok, "all_alive": all_alive}
        if not all_alive:
            logger.warning(
                f"[SAFETY_ACK:REJECTED] {reason}: threads not all alive "
                f"{threads_ok}"
            )
            return False
        ack_store.record_ack(reason, operator_note, preconditions)
        ss.clear_reason(THREAD_DEAD)
        return True

    if reason == EXCHANGE_UNREACHABLE:
        # Precondição: exchange respondeu (tentativa de get_account_summary)
        api = exchange_api or ExchangeAPIStub()
        try:
            result = api.get_account_summary()
            reachable = result is not None
        except Exception:
            reachable = False
        preconditions = {"exchange_reachable": reachable}
        if not reachable:
            logger.warning(
                f"[SAFETY_ACK:REJECTED] {reason}: exchange not reachable"
            )
            return False
        ack_store.record_ack(reason, operator_note, preconditions)
        ss.clear_reason(EXCHANGE_UNREACHABLE)
        return True

    if reason == INVARIANT_FAIL:
        # Precondição: invariants_ok_streak >= 3
        streak = ss.invariants_ok_streak
        preconditions = {"invariants_ok_streak": streak}
        if streak < 3:
            logger.warning(
                f"[SAFETY_ACK:REJECTED] {reason}: streak={streak} < 3"
            )
            return False
        ack_store.record_ack(reason, operator_note, preconditions)
        ss.clear_reason(INVARIANT_FAIL)
        return True

    if reason == CAPITAL_MISMATCH:
        # Precondição: capital_mismatch_ok_streak >= 3
        streak = ss.capital_mismatch_ok_streak
        preconditions = {"capital_mismatch_ok_streak": streak}
        if streak < 3:
            logger.warning(
                f"[SAFETY_ACK:REJECTED] {reason}: streak={streak} < 3"
            )
            return False
        ack_store.record_ack(reason, operator_note, preconditions)
        ss.clear_reason(CAPITAL_MISMATCH)
        return True

    if reason == SAFE_MODE:
        # Precondição: nenhuma reason crítica ativa além de SAFE_MODE
        current = ss.reasons
        other = current - {SAFE_MODE}
        preconditions = {"other_reasons": sorted(other)}
        if other:
            logger.warning(
                f"[SAFETY_ACK:REJECTED] {reason}: other reasons active: "
                f"{sorted(other)}"
            )
            return False
        ack_store.record_ack(reason, operator_note, preconditions)
        ss.disable_safe_mode()
        return True

    if reason == CONFIG_CHANGED:
        # Precondição: operador validou nova config — force update snapshot
        config_snap = system.get("config_snapshot")
        config = system.get("config")
        if config_snap is None or config is None:
            logger.warning(
                f"[SAFETY_ACK:REJECTED] {reason}: config_snapshot/config ausente"
            )
            return False
        preconditions = {"config_validated": True}
        ack_store.record_ack(reason, operator_note, preconditions)
        config_snap.force_update(config)
        ss.clear_reason(CONFIG_CHANGED)
        return True

    logger.warning(f"[SAFETY_ACK:UNKNOWN_REASON] {reason}")
    return False


# ======================================================================
# Entry point
# ======================================================================

def _resolve_token_ids() -> List[str]:
    """Resolve token_ids: POLYMARKET_TOKENS ou auto-discovery."""
    manual = os.environ.get("POLYMARKET_TOKENS", "").split(",")
    manual = [t.strip() for t in manual if t.strip()]
    if manual:
        logger.info("[UNIVERSE] [MANUAL_OVERRIDE] tokens=%d", len(manual))
        return manual
    logger.info("[UNIVERSE] [AUTO_DISCOVERY] POLYMARKET_TOKENS vazio, buscando mercados...")
    universe = MarketUniverseManager()
    token_ids = universe.run_once()
    if not token_ids:
        logger.critical("[UNIVERSE] [NO_MARKETS] Nenhum mercado ativo encontrado")
        return []
    logger.info("[UNIVERSE] [DISCOVERED] markets=%d", len(token_ids))
    return token_ids


def main() -> None:
    """Entry point do sistema de trading."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    token_ids = _resolve_token_ids()
    if not token_ids:
        logger.critical("[SYSTEM] Abortando: nenhum mercado para monitorar")
        return

    initial_capital = float(
        os.environ.get("INITIAL_CAPITAL", str(DEFAULT_INITIAL_CAPITAL))
    )

    system = create_system(
        initial_capital=initial_capital,
        token_ids=token_ids,
    )
    run_system(system)


if __name__ == "__main__":
    main()
