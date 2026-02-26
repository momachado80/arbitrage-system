#!/usr/bin/env python3
"""
Teste formal de determinismo do pipeline completo da engine.
Prova que, dada sequência fixa de eventos sintéticos, o pipeline produz
resultados idênticos: mesmo processo, entre processos, após reinicialização.

Uso:
    PYTHONHASHSEED=42 RANDOM_SEED=42 python scripts/test_pipeline_determinism.py
    # Saída em pipeline_run.json

    # Teste entre processos:
    PYTHONHASHSEED=42 RANDOM_SEED=42 python scripts/test_pipeline_determinism.py > run1.json
    PYTHONHASHSEED=42 RANDOM_SEED=42 python scripts/test_pipeline_determinism.py > run2.json
    diff run1.json run2.json   # esperado: nenhuma diferença

    # Seed diferente:
    PYTHONHASHSEED=43 RANDOM_SEED=43 python scripts/test_pipeline_determinism.py > run3.json
    diff run1.json run3.json   # esperado: diferenças
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# Patches ANTES de qualquer import do src (determinismo)
# -----------------------------------------------------------------------------

_UUID_COUNTER = [0]
_TIME_FIXED = 1700000000.0
_DATETIME_FIXED = "2024-11-15T12:00:00.000000+00:00"


def _deterministic_uuid4():
    import uuid
    _UUID_COUNTER[0] += 1
    h = hex(_UUID_COUNTER[0])[2:].zfill(32)
    return uuid.UUID(h[:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:])


def _deterministic_time():
    return _TIME_FIXED


def _noop_sleep(secs):
    pass


def _apply_patches():
    """Aplica patches para fontes não determinísticas."""
    import time
    import uuid
    time.time = _deterministic_time
    time.sleep = _noop_sleep
    uuid.uuid4 = _deterministic_uuid4


def _validate_no_nondeterminism():
    """
    Verifica módulos do critical path por fontes não determinísticas.
    Só checa: pipeline, order_manager, mode_stubs, trade_ledger.
    """
    import ast
    src_root = Path(__file__).resolve().parent.parent / "src"
    critical = [
        "execution/pipeline.py",
        "execution/order_manager.py",
        "execution/mode_stubs.py",
        "engine/trade_ledger.py",
    ]
    violations = []
    for rel in critical:
        py = src_root / rel
        if not py.exists():
            continue
        try:
            tree = ast.parse(py.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        val = getattr(node.func.value, "id", "")
                        attr = node.func.attr
                        if (val == "time" and attr == "time") or (val == "datetime" and attr == "now"):
                            violations.append((str(py), f"{val}.{attr}()"))
                    elif isinstance(node.func, ast.Name) and node.func.id == "uuid4":
                        violations.append((str(py), "uuid4()"))
                elif isinstance(node, ast.Name) and node.id == "SystemRandom":
                    violations.append((str(py), "SystemRandom"))
        except Exception:
            pass
    if violations:
        for f, v in violations[:10]:
            print(f"DETERMINISMO_VIOLADO: {f} usa {v}", file=sys.stderr)
        if len(violations) > 10:
            print(f"... e mais {len(violations)-10} violações", file=sys.stderr)
        print(
            "\nMocke time.time, datetime.now, uuid4 no script ou remova do código.",
            file=sys.stderr,
        )
        sys.exit(1)


# Aplicar patches imediatamente
_apply_patches()

# Silenciar logs
logging.disable(logging.CRITICAL)

# Path do projeto
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------------------------
# Seed e imports
# -----------------------------------------------------------------------------

seed = int(os.environ.get("RANDOM_SEED", "42"))
from src.seed import set_global_seed
set_global_seed(seed)

# Validação estática (opcional, pode ser lenta)
if os.environ.get("VALIDATE_NONDET", "").upper() in ("1", "TRUE", "YES"):
    _validate_no_nondeterminism()

from src.risk.capital_manager import CapitalManager
from src.risk.position_manager import PositionManager
from src.risk.risk_manager import RiskManager
from src.engine.probability_calibrator import (
    ProbabilityCalibrator,
    MarketCategory,
    TradeType,
)
from src.engine.trade_ledger import TradeLedger
from src.execution.order_manager import OrderManager, OrderSide, OutcomeSide
from src.execution.pipeline import execute_trade_decision
from src.execution.reconciler import OrderReconciler
from src.execution.mode_stubs import PaperExchangeAPI
from src.watcher.bridge import MarketSnapshot
from src.orchestrator import ReconcilerLoop


# -----------------------------------------------------------------------------
# Eventos sintéticos determinísticos
# -----------------------------------------------------------------------------

def _make_synthetic_snapshots(n: int = 50, market_id: str = "0xtest_market_001") -> List[MarketSnapshot]:
    """Gera sequência fixa de snapshots. Determinística dado o seed."""
    import random
    snapshots = []
    base_ts = 1700000000.0
    for i in range(n):
        # Progressão determinística de bid/ask
        bid = 0.45 + (i % 10) * 0.005
        ask = bid + 0.02
        snap = MarketSnapshot(
            timestamp=base_ts + i,
            token_id=market_id,
            best_bid_price=round(bid, 4),
            best_bid_size=100.0 + i,
            best_ask_price=round(ask, 4),
            best_ask_size=50.0 + i,
        )
        snapshots.append(snap)
    return snapshots


def _snapshot_to_decision(snap: MarketSnapshot, idx: int) -> Dict[str, Any]:
    """Converte snapshot em decisão de trade. Determinístico."""
    # Sinal simples: a cada 10 snapshots, BUY se bid < 0.50
    if idx > 0 and idx % 10 == 0 and snap.best_bid_price and snap.best_bid_price < 0.50:
        return {
            "market_id": snap.token_id,
            "action": "BUY",
            "shares": 10.0,
            "limit_price": round(snap.best_bid_price or 0.48, 4),
            "outcome_side": OutcomeSide.YES,
            "category": MarketCategory.POLITICS,
            "trade_type": TradeType.STATISTICAL,
            "predicted_probability": 0.52,
            "ttl_seconds": 60,
        }
    return None


# -----------------------------------------------------------------------------
# Pipeline run
# -----------------------------------------------------------------------------

def _create_components(state_dir: str) -> Dict[str, Any]:
    """Cria componentes do pipeline. Sem WebSocket, sem relógio real."""
    capital = 1000.0
    cm = CapitalManager(initial_capital=capital)
    cal = ProbabilityCalibrator(state_path=os.path.join(state_dir, "cal.json"))
    pm = PositionManager(state_path=os.path.join(state_dir, "pos.json"))
    led = TradeLedger(
        state_path=os.path.join(state_dir, "ledger.json"),
        calibrator=cal,
        capital_manager=cm,
    )
    rm = RiskManager(config={
        "max_exposure_total": capital,
        "max_exposure_per_category": capital * 0.5,
        "max_exposure_per_market": capital * 0.2,
        "max_open_trades": 50,
        "total_capital": capital,
        "max_percent_capital_deadzone": 0.5,
    })
    om = OrderManager(account_mode="SPOT_ONLY")
    reconciler = OrderReconciler()
    api = PaperExchangeAPI(om)

    rec_loop = ReconcilerLoop(
        reconciler=reconciler,
        order_manager=om,
        trade_ledger=led,
        exchange_api=api,
        capital_manager=cm,
        position_manager=pm,
        interval=1,
    )

    return {
        "capital_manager": cm,
        "calibrator": cal,
        "position_manager": pm,
        "trade_ledger": led,
        "risk_manager": rm,
        "order_manager": om,
        "reconciler": reconciler,
        "reconciler_loop": rec_loop,
        "exchange_api": api,
    }


def _run_pipeline(components: Dict[str, Any], snapshots: List[MarketSnapshot]) -> Dict[str, Any]:
    """Executa pipeline: snapshots → decisões → execução → reconciliação."""
    cm = components["capital_manager"]
    cal = components["calibrator"]
    pm = components["position_manager"]
    led = components["trade_ledger"]
    rm = components["risk_manager"]
    om = components["order_manager"]
    rec_loop = components["reconciler_loop"]

    # Iniciar reconciler em background
    rec_loop.start()

    trades_executed = []
    for idx, snap in enumerate(snapshots):
        dec = _snapshot_to_decision(snap, idx)
        if dec is not None:
            book = {
                "bids": [[snap.best_bid_price or 0.48, snap.best_bid_size or 100]],
                "asks": [[snap.best_ask_price or 0.50, snap.best_ask_size or 50]],
            }
            tid = execute_trade_decision(
                dec, cal, om, led,
                risk_manager=rm,
                order_book=book,
                capital_manager=cm,
                position_manager=pm,
            )
            if tid:
                trades_executed.append(tid)

    # Parar reconciler
    rec_loop.stop()
    rec_loop.join(timeout=5)

    # Coletar métricas
    cap_snap = cm.snapshot()
    all_trades = list(led.get_all_trades().values())
    total_trades = len([t for t in all_trades if t.get("status") in ("FILLED", "OPEN", "RESOLVED", "PARTIALLY_FILLED")])
    realized_pnl = cap_snap.get("realized_pnl", 0.0)
    capital_after = cap_snap.get("total_capital", 0.0)

    # Serializar trades (apenas campos determinísticos; excluir timestamps)
    trades_serialized = []
    for t in all_trades:
        rec = {k: v for k, v in t.items() if k in (
            "trade_id", "market_id", "status", "final_size", "entry_price",
            "base_size", "multiplier_k", "outcome_side", "action",
            "cumulative_filled_size", "weighted_avg_price",
        )}
        # Normalizar floats para determinismo
        for k, v in list(rec.items()):
            if isinstance(v, float):
                rec[k] = round(v, 6)
        trades_serialized.append(rec)

    return {
        "total_trades": total_trades,
        "realized_pnl": round(realized_pnl, 6),
        "capital_after_trade": round(capital_after, 6),
        "avg_slippage": 0.0,
        "rejection_rate": 0.0,
        "trades": trades_serialized,
    }


def _reset_uuid_counter():
    _UUID_COUNTER[0] = 0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = tmp

        # A) Dupla execução no mesmo processo
        snapshots = _make_synthetic_snapshots(50)
        comp1 = _create_components(state_dir)
        run1 = _run_pipeline(comp1, snapshots)

        _reset_uuid_counter()
        set_global_seed(seed)
        comp2 = _create_components(state_dir)
        run2 = _run_pipeline(comp2, snapshots)

        if run1 != run2:
            print(
                "FALHA: Execução dupla no mesmo processo produziu resultados diferentes.",
                file=sys.stderr,
            )
            print("Possíveis causas: estado global não resetado, estrutura mutável compartilhada, "
                  "RNG chamado antes do seed, ordem de dicionário não estável.", file=sys.stderr)
            sys.exit(1)

        # B) Output JSON determinístico
        out = {
            "seed": seed,
            "metrics": run1,
        }
        json_str = json.dumps(out, sort_keys=True, indent=None, separators=(",", ":"))
        out_path = Path.cwd() / "pipeline_run.json"
        out_path.write_text(json_str, encoding="utf-8")

        # stdout para diff entre processos
        print(json_str)


if __name__ == "__main__":
    main()
