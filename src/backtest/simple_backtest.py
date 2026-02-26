"""
Backtest — Strategy Backtesting
===================================

Fase 1: JSON scenarios (determinístico, validação lógica)
Fase 2: Snapshot replay (temporal, slippage + latência)

SAÍDA:
- PnL total
- Sharpe ratio simples
- Max drawdown
- Win rate
- Brier score
- Relatório JSON

Thread-safe. Sem infra pesada.
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_FEE: float = 0.02
DEFAULT_SLIPPAGE: float = 0.005
DEFAULT_LATENCY_MS: float = 100.0  # 100ms latência simulada


@dataclass
class BacktestScenario:
    """Um cenário de backtest (Fase 1)."""
    token_id: str
    p_market: float          # probabilidade implícita do mercado
    p_oracle: float          # probabilidade estimada pelo oracle
    outcome: float           # 1.0 = YES, 0.0 = NO
    oracle_confidence: float = 0.7
    spread: float = 0.02
    timestamp: float = 0.0


@dataclass
class BacktestTrade:
    """Trade executado no backtest."""
    token_id: str
    side: str              # "BUY" ou "SELL"
    entry_price: float
    exit_value: float      # outcome-based value
    size: float
    pnl: float
    net_edge: float
    p_oracle: float
    p_market: float
    fee: float
    slippage: float


@dataclass
class BacktestResult:
    """Resultado completo do backtest."""
    total_trades: int = 0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    brier_score: float = 0.0
    total_fees: float = 0.0
    avg_net_edge: float = 0.0
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class SimpleBacktest:
    """
    Backtester simples para validação de estratégia.

    Fase 1: Recebe lista de cenários JSON, simula decisões, calcula métricas.
    Fase 2: Replay de snapshots com latência e slippage temporal.
    """

    def __init__(
        self,
        edge_threshold: float = 0.03,
        fee: float = DEFAULT_FEE,
        slippage: float = DEFAULT_SLIPPAGE,
        base_size: float = 10.0,
        initial_capital: float = 1000.0,
    ) -> None:
        self._edge_threshold = edge_threshold
        self._fee = fee
        self._slippage = slippage
        self._base_size = base_size
        self._initial_capital = initial_capital

    def run_scenarios(self, scenarios: List[BacktestScenario]) -> BacktestResult:
        """
        Fase 1: Executa backtest em lista de cenários.

        Cada cenário é avaliado independentemente.
        Edge = p_oracle - p_market - fee - slippage - spread_impact.
        Trade executado se |net_edge| >= threshold.
        PnL = (outcome - entry_price) * size para BUY,
              (entry_price - outcome) * size para SELL.
        """
        trades: List[BacktestTrade] = []
        brier_scores: List[float] = []
        capital = self._initial_capital
        peak = capital
        max_dd = 0.0
        equity = [capital]

        for s in scenarios:
            # Brier score: sempre calculado
            brier = (s.p_oracle - s.outcome) ** 2
            brier_scores.append(brier)

            # Calcular net edge
            raw_edge = s.p_oracle - s.p_market
            spread_impact = s.spread * 0.5
            cost = self._fee + self._slippage + spread_impact

            if raw_edge > 0:
                net_edge = raw_edge - cost
            elif raw_edge < 0:
                net_edge = raw_edge + cost
            else:
                net_edge = 0.0

            # Decisão de trade
            side: Optional[str] = None
            if net_edge >= self._edge_threshold:
                side = "BUY"
            elif net_edge <= -self._edge_threshold:
                side = "SELL"

            if side is None:
                continue

            # Executar trade
            if side == "BUY":
                entry_price = s.p_market + s.spread / 2 + self._slippage
                exit_value = s.outcome
                pnl = (exit_value - entry_price) * self._base_size
            else:  # SELL
                entry_price = s.p_market - s.spread / 2 - self._slippage
                exit_value = s.outcome
                pnl = (entry_price - exit_value) * self._base_size

            fee_cost = self._fee * self._base_size
            pnl -= fee_cost

            capital += pnl
            peak = max(peak, capital)
            dd = peak - capital
            max_dd = max(max_dd, dd)
            equity.append(capital)

            trade = BacktestTrade(
                token_id=s.token_id,
                side=side,
                entry_price=entry_price,
                exit_value=exit_value,
                size=self._base_size,
                pnl=pnl,
                net_edge=net_edge,
                p_oracle=s.p_oracle,
                p_market=s.p_market,
                fee=fee_cost,
                slippage=self._slippage * self._base_size,
            )
            trades.append(trade)

        # Calcular métricas
        n = len(trades)
        if n == 0:
            return BacktestResult(
                brier_score=sum(brier_scores) / len(brier_scores) if brier_scores else 0.0,
                equity_curve=equity,
            )

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / n
        total_fees = sum(t.fee for t in trades)
        avg_edge = sum(abs(t.net_edge) for t in trades) / n

        # Sharpe ratio simples (daily returns não disponível, usar trade returns)
        returns = [t.pnl / self._base_size for t in trades]
        mean_ret = sum(returns) / len(returns)
        if len(returns) > 1:
            var_ret = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            std_ret = math.sqrt(var_ret) if var_ret > 0 else 0.001
        else:
            std_ret = 0.001
        sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

        # Max drawdown %
        max_dd_pct = (max_dd / (self._initial_capital + peak)) * 100 if peak > 0 else 0.0

        return BacktestResult(
            total_trades=n,
            total_pnl=round(total_pnl, 2),
            win_count=len(wins),
            loss_count=len(losses),
            win_rate=round(len(wins) / n, 3) if n > 0 else 0.0,
            avg_pnl_per_trade=round(avg_pnl, 4),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            sharpe_ratio=round(sharpe, 3),
            brier_score=round(sum(brier_scores) / len(brier_scores), 4) if brier_scores else 0.0,
            total_fees=round(total_fees, 2),
            avg_net_edge=round(avg_edge, 4),
            trades=[
                {
                    "token_id": t.token_id,
                    "side": t.side,
                    "entry": round(t.entry_price, 4),
                    "exit": round(t.exit_value, 4),
                    "pnl": round(t.pnl, 4),
                    "net_edge": round(t.net_edge, 4),
                }
                for t in trades
            ],
            equity_curve=[round(e, 2) for e in equity],
        )

    def run_snapshot_replay(
        self,
        snapshots: List[Dict[str, Any]],
        oracle_estimates: Dict[str, Dict[str, float]],
        outcomes: Dict[str, float],
        latency_ms: float = DEFAULT_LATENCY_MS,
    ) -> BacktestResult:
        """
        Fase 2: Replay de snapshots históricos reais.

        Args:
            snapshots: lista de {token_id, timestamp, best_bid, best_ask, ...}
            oracle_estimates: token_id → {probability, confidence}
            outcomes: token_id → outcome (0.0 ou 1.0)
            latency_ms: latência simulada em ms
        """
        scenarios: List[BacktestScenario] = []

        # Agrupar snapshots por token, usar último snapshot
        latest_by_token: Dict[str, Dict[str, Any]] = {}
        for snap in sorted(snapshots, key=lambda s: s.get("timestamp", 0)):
            tid = snap.get("token_id", "")
            if tid:
                latest_by_token[tid] = snap

        for token_id, snap in latest_by_token.items():
            if token_id not in oracle_estimates:
                continue
            if token_id not in outcomes:
                continue

            bid = snap.get("best_bid", 0)
            ask = snap.get("best_ask", 0)
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue

            mid = (bid + ask) / 2.0
            spread = ask - bid
            est = oracle_estimates[token_id]

            # Adicionar latência como slippage extra
            latency_slippage = (latency_ms / 1000.0) * 0.01  # ~1% per second drift

            scenarios.append(BacktestScenario(
                token_id=token_id,
                p_market=mid,
                p_oracle=est.get("probability", 0.5),
                outcome=outcomes[token_id],
                oracle_confidence=est.get("confidence", 0.5),
                spread=spread + latency_slippage,
                timestamp=snap.get("timestamp", 0),
            ))

        # Usar Fase 1 com slippage ajustado
        original_slippage = self._slippage
        self._slippage = self._slippage + (latency_ms / 1000.0) * 0.005
        result = self.run_scenarios(scenarios)
        self._slippage = original_slippage

        return result

    @staticmethod
    def result_to_json(result: BacktestResult) -> str:
        """Serializa resultado para JSON."""
        return json.dumps({
            "total_trades": result.total_trades,
            "total_pnl": result.total_pnl,
            "win_count": result.win_count,
            "loss_count": result.loss_count,
            "win_rate": result.win_rate,
            "avg_pnl_per_trade": result.avg_pnl_per_trade,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "brier_score": result.brier_score,
            "total_fees": result.total_fees,
            "avg_net_edge": result.avg_net_edge,
            "equity_curve": result.equity_curve,
            "trades": result.trades,
        }, indent=2)
