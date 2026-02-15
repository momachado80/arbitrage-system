"""
Risk Management Module
======================
Implementa gestão de risco matemática rigorosa para trading.

Inclui:
- Kelly Criterion para position sizing
- Drawdown tracking
- Métricas de risco (Sharpe, Sortino, Profit Factor)
- Risk of Ruin calculation
- Correlation matrix
"""
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class TradeResult:
    """Resultado de um trade individual"""
    id: str
    timestamp: datetime
    categoria: str
    entry_price: float
    exit_price: float
    position_size: float  # em USD
    pnl: float  # Profit/Loss em USD
    pnl_percent: float  # Profit/Loss em %
    is_win: bool
    holding_time_hours: float
    signal_confidence: float
    fees_paid: float = 0.0


@dataclass
class RiskMetrics:
    """Métricas de risco calculadas"""
    # Básicas
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    
    # PnL
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Ratios
    profit_factor: float = 0.0  # gross_profit / gross_loss
    risk_reward_ratio: float = 0.0  # avg_win / avg_loss
    expectancy: float = 0.0  # Expected value per trade
    
    # Volatilidade
    returns_std: float = 0.0
    sharpe_ratio: float = 0.0  # (avg_return - risk_free) / std
    sortino_ratio: float = 0.0  # (avg_return - risk_free) / downside_std
    
    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    current_drawdown: float = 0.0
    avg_drawdown_duration_days: float = 0.0
    
    # Risk
    risk_of_ruin: float = 0.0  # Probabilidade de perder todo capital
    kelly_fraction: float = 0.0  # Fração ótima do capital por trade
    
    # Por categoria
    metrics_by_category: Dict[str, dict] = field(default_factory=dict)


@dataclass
class PositionSizing:
    """Resultado do cálculo de position sizing"""
    recommended_size_usd: float
    recommended_size_percent: float
    kelly_full: float
    kelly_half: float  # Mais conservador
    kelly_quarter: float  # Ultra conservador
    max_risk_usd: float
    risk_level: RiskLevel
    reasoning: str


class RiskManager:
    """
    Gerenciador de Risco Principal
    
    Implementa:
    - Kelly Criterion
    - Drawdown tracking
    - Risk metrics calculation
    - Position sizing
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_risk_per_trade: float = 0.02,  # 2% max por trade
        max_portfolio_risk: float = 0.20,  # 20% max exposição total
        risk_free_rate: float = 0.05,  # 5% ao ano
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.risk_free_rate = risk_free_rate
        
        # Histórico completo
        self.trade_history: List[TradeResult] = []
        self.equity_curve: List[Tuple[datetime, float]] = [(datetime.utcnow(), initial_capital)]
        
        # Posições abertas
        self.open_positions: Dict[str, dict] = {}
        
        # Peak para drawdown
        self.peak_capital = initial_capital
        
        # Métricas por categoria
        self.category_stats: Dict[str, dict] = {}
    
    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win_percent: float,
        avg_loss_percent: float
    ) -> float:
        """
        Calcula a fração ótima do capital usando Kelly Criterion.
        
        f* = (p × b - q) / b
        
        onde:
        - p = probabilidade de ganhar
        - q = probabilidade de perder (1 - p)
        - b = win/loss ratio (avg_win / avg_loss)
        
        Returns:
            Fração do capital (0.0 a 1.0)
        """
        if avg_loss_percent == 0:
            return 0.0
        
        p = win_rate
        q = 1 - win_rate
        b = abs(avg_win_percent / avg_loss_percent) if avg_loss_percent != 0 else 0
        
        if b == 0:
            return 0.0
        
        kelly = (p * b - q) / b
        
        # Kelly pode ser negativo (significa não apostar)
        return max(0.0, min(kelly, 1.0))
    
    def calculate_position_size(
        self,
        signal_confidence: float,
        categoria: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float = None
    ) -> PositionSizing:
        """
        Calcula o tamanho ideal da posição baseado em:
        1. Kelly Criterion ajustado pela confiança
        2. Máximo risco por trade
        3. Histórico da categoria
        4. Capital disponível
        """
        # Busca estatísticas da categoria
        cat_stats = self.category_stats.get(categoria, {})
        win_rate = cat_stats.get('win_rate', 0.5)  # Default 50%
        avg_win = cat_stats.get('avg_win_percent', 0.30)  # Default 30%
        avg_loss = cat_stats.get('avg_loss_percent', -0.15)  # Default -15%
        
        # Calcula Kelly
        kelly_full = self.calculate_kelly_criterion(win_rate, avg_win, abs(avg_loss))
        kelly_half = kelly_full * 0.5  # Mais conservador
        kelly_quarter = kelly_full * 0.25  # Ultra conservador
        
        # Ajusta pela confiança do sinal (normalizada 0-1)
        confidence_factor = signal_confidence / 100.0
        adjusted_kelly = kelly_half * confidence_factor
        
        # Calcula risco no stop loss
        risk_per_unit = abs(entry_price - stop_loss_price) / entry_price
        
        # Máximo risco em USD
        max_risk_usd = self.current_capital * self.max_risk_per_trade
        
        # Tamanho da posição baseado no risco
        if risk_per_unit > 0:
            position_from_risk = max_risk_usd / risk_per_unit
        else:
            position_from_risk = 0
        
        # Tamanho da posição baseado em Kelly
        position_from_kelly = self.current_capital * adjusted_kelly
        
        # Usa o menor dos dois (mais conservador)
        recommended_size = min(position_from_risk, position_from_kelly)
        
        # Limita pela exposição máxima do portfólio
        current_exposure = sum(p['size'] for p in self.open_positions.values())
        max_additional = (self.current_capital * self.max_portfolio_risk) - current_exposure
        recommended_size = min(recommended_size, max(0, max_additional))
        
        # Determina nível de risco
        exposure_percent = (current_exposure + recommended_size) / self.current_capital
        if exposure_percent > 0.5:
            risk_level = RiskLevel.CRITICAL
        elif exposure_percent > 0.3:
            risk_level = RiskLevel.HIGH
        elif exposure_percent > 0.15:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Reasoning
        reasoning = f"""
Position Sizing Analysis:
- Capital: ${self.current_capital:,.2f}
- Kelly Full: {kelly_full*100:.1f}% | Half: {kelly_half*100:.1f}% | Quarter: {kelly_quarter*100:.1f}%
- Signal Confidence: {signal_confidence:.0f}%
- Adjusted Kelly: {adjusted_kelly*100:.2f}%
- Risk per unit: {risk_per_unit*100:.2f}%
- Max risk USD: ${max_risk_usd:,.2f}
- Current exposure: ${current_exposure:,.2f} ({current_exposure/self.current_capital*100:.1f}%)
- Category win rate: {win_rate*100:.1f}%
""".strip()
        
        return PositionSizing(
            recommended_size_usd=recommended_size,
            recommended_size_percent=(recommended_size / self.current_capital * 100) if self.current_capital > 0 else 0,
            kelly_full=kelly_full,
            kelly_half=kelly_half,
            kelly_quarter=kelly_quarter,
            max_risk_usd=max_risk_usd,
            risk_level=risk_level,
            reasoning=reasoning
        )
    
    def record_trade(self, trade: TradeResult) -> None:
        """Registra um trade completo no histórico"""
        self.trade_history.append(trade)
        
        # Atualiza capital
        self.current_capital += trade.pnl
        
        # Atualiza equity curve
        self.equity_curve.append((trade.timestamp, self.current_capital))
        
        # Atualiza peak para drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Atualiza stats da categoria
        self._update_category_stats(trade)
    
    def _update_category_stats(self, trade: TradeResult) -> None:
        """Atualiza estatísticas por categoria"""
        cat = trade.categoria
        if cat not in self.category_stats:
            self.category_stats[cat] = {
                'trades': [],
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'gross_profit': 0,
                'gross_loss': 0
            }
        
        stats = self.category_stats[cat]
        stats['trades'].append(trade.pnl_percent)
        stats['total_pnl'] += trade.pnl
        
        if trade.is_win:
            stats['wins'] += 1
            stats['gross_profit'] += trade.pnl
        else:
            stats['losses'] += 1
            stats['gross_loss'] += abs(trade.pnl)
        
        total = stats['wins'] + stats['losses']
        stats['win_rate'] = stats['wins'] / total if total > 0 else 0
        
        # Avg win/loss
        wins = [t for t in stats['trades'] if t > 0]
        losses = [t for t in stats['trades'] if t < 0]
        stats['avg_win_percent'] = statistics.mean(wins) if wins else 0
        stats['avg_loss_percent'] = statistics.mean(losses) if losses else 0
    
    def calculate_metrics(self) -> RiskMetrics:
        """Calcula todas as métricas de risco"""
        metrics = RiskMetrics()
        
        if not self.trade_history:
            return metrics
        
        # Básicas
        metrics.total_trades = len(self.trade_history)
        metrics.wins = sum(1 for t in self.trade_history if t.is_win)
        metrics.losses = metrics.total_trades - metrics.wins
        metrics.win_rate = metrics.wins / metrics.total_trades
        
        # PnL
        pnls = [t.pnl for t in self.trade_history]
        pnl_percents = [t.pnl_percent for t in self.trade_history]
        
        metrics.total_pnl = sum(pnls)
        metrics.gross_profit = sum(p for p in pnls if p > 0)
        metrics.gross_loss = abs(sum(p for p in pnls if p < 0))
        
        wins = [t.pnl_percent for t in self.trade_history if t.is_win]
        losses = [t.pnl_percent for t in self.trade_history if not t.is_win]
        
        metrics.avg_win = statistics.mean(wins) if wins else 0
        metrics.avg_loss = statistics.mean(losses) if losses else 0
        metrics.largest_win = max(pnl_percents) if pnl_percents else 0
        metrics.largest_loss = min(pnl_percents) if pnl_percents else 0
        
        # Ratios
        metrics.profit_factor = (
            metrics.gross_profit / metrics.gross_loss 
            if metrics.gross_loss > 0 else 999.0
        )
        metrics.risk_reward_ratio = (
            abs(metrics.avg_win / metrics.avg_loss)
            if metrics.avg_loss != 0 else 0
        )
        
        # Expectancy: E = (P_win × Avg_win) - (P_loss × Avg_loss)
        metrics.expectancy = (
            (metrics.win_rate * metrics.avg_win) - 
            ((1 - metrics.win_rate) * abs(metrics.avg_loss))
        )
        
        # Volatilidade
        if len(pnl_percents) > 1:
            metrics.returns_std = statistics.stdev(pnl_percents)
            
            # Sharpe Ratio (anualizado)
            avg_return = statistics.mean(pnl_percents)
            daily_rf = self.risk_free_rate / 365
            if metrics.returns_std > 0:
                metrics.sharpe_ratio = (avg_return - daily_rf) / metrics.returns_std * math.sqrt(365)
            
            # Sortino Ratio (usa apenas downside deviation)
            # Calcula downside deviation corretamente (desvios abaixo do MAR - Minimum Acceptable Return)
            mar = 0  # MAR = 0 (break-even)
            downside_deviations = [(r - mar)**2 for r in pnl_percents if r < mar]
            if len(downside_deviations) >= 1:
                downside_variance = sum(downside_deviations) / len(pnl_percents)  # Usa N total, não apenas negativos
                downside_std = math.sqrt(downside_variance)
                if downside_std > 0:
                    metrics.sortino_ratio = (avg_return - daily_rf) / downside_std * math.sqrt(252)  # 252 trading days
        
        # Drawdown
        metrics.max_drawdown, metrics.max_drawdown_percent = self._calculate_max_drawdown()
        metrics.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        
        # Kelly
        metrics.kelly_fraction = self.calculate_kelly_criterion(
            metrics.win_rate,
            metrics.avg_win,
            abs(metrics.avg_loss)
        )
        
        # Risk of Ruin (aproximação)
        if metrics.win_rate > 0 and metrics.win_rate < 1:
            # Fórmula simplificada: ((1-p)/p)^n onde n = número de unidades até ruína
            q_over_p = (1 - metrics.win_rate) / metrics.win_rate
            units_to_ruin = int(self.current_capital / (self.current_capital * self.max_risk_per_trade))
            metrics.risk_of_ruin = min(1.0, q_over_p ** units_to_ruin) if q_over_p < 1 else 1.0
        
        # Por categoria
        metrics.metrics_by_category = {
            cat: {
                'win_rate': stats['win_rate'],
                'trades': len(stats['trades']),
                'total_pnl': stats['total_pnl'],
                'profit_factor': (
                    stats['gross_profit'] / stats['gross_loss']
                    if stats['gross_loss'] > 0 else 999.0
                )
            }
            for cat, stats in self.category_stats.items()
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> Tuple[float, float]:
        """Calcula o máximo drawdown histórico"""
        if len(self.equity_curve) < 2:
            return 0.0, 0.0
        
        peak = self.equity_curve[0][1]
        max_dd = 0.0
        max_dd_percent = 0.0
        
        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_percent = dd / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
            if dd_percent > max_dd_percent:
                max_dd_percent = dd_percent
        
        return max_dd, max_dd_percent
    
    def get_risk_dashboard(self) -> dict:
        """Retorna dados para o dashboard de risco"""
        metrics = self.calculate_metrics()
        
        # Exposição atual
        total_exposure = sum(p['size'] for p in self.open_positions.values())
        exposure_percent = total_exposure / self.current_capital if self.current_capital > 0 else 0
        
        # Risk level geral
        if metrics.current_drawdown > 0.15 or exposure_percent > 0.5:
            overall_risk = RiskLevel.CRITICAL
        elif metrics.current_drawdown > 0.10 or exposure_percent > 0.3:
            overall_risk = RiskLevel.HIGH
        elif metrics.current_drawdown > 0.05 or exposure_percent > 0.15:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW
        
        return {
            "capital": {
                "initial": self.initial_capital,
                "current": self.current_capital,
                "pnl": self.current_capital - self.initial_capital,
                "pnl_percent": ((self.current_capital / self.initial_capital) - 1) * 100
            },
            "exposure": {
                "total_usd": total_exposure,
                "percent": exposure_percent * 100,
                "positions_open": len(self.open_positions),
                "max_allowed": self.max_portfolio_risk * 100
            },
            "performance": {
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate * 100,
                "profit_factor": metrics.profit_factor,
                "expectancy": metrics.expectancy * 100,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio
            },
            "drawdown": {
                "current": metrics.current_drawdown * 100,
                "max_historical": metrics.max_drawdown_percent * 100,
                "peak_capital": self.peak_capital
            },
            "risk": {
                "overall_level": overall_risk.value,
                "risk_of_ruin": metrics.risk_of_ruin * 100,
                "kelly_optimal": metrics.kelly_fraction * 100,
                "max_risk_per_trade": self.max_risk_per_trade * 100
            },
            "by_category": metrics.metrics_by_category
        }
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        categoria: str,
        volatility_multiplier: float = 2.0
    ) -> dict:
        """
        Calcula stop loss dinâmico baseado em:
        1. ATR (Average True Range) simulado
        2. Histórico da categoria
        3. Máximo risco aceitável
        """
        # Busca volatilidade histórica da categoria
        cat_stats = self.category_stats.get(categoria, {})
        trades = cat_stats.get('trades', [])
        
        if trades:
            # Usa desvio padrão dos retornos como proxy de volatilidade
            volatility = statistics.stdev(trades) if len(trades) > 1 else 0.15
        else:
            volatility = 0.15  # Default 15%
        
        # Stop loss baseado em volatilidade
        sl_distance = volatility * volatility_multiplier
        stop_loss_price = entry_price * (1 - sl_distance)
        
        # Stop loss baseado em risco máximo
        max_loss_percent = self.max_risk_per_trade / 0.5  # Assume 50% position
        sl_from_risk = entry_price * (1 - max_loss_percent)
        
        # Usa o mais conservador (stop mais apertado)
        final_sl = max(stop_loss_price, sl_from_risk)
        
        # Take profit (2:1 risk/reward mínimo)
        risk_amount = entry_price - final_sl
        take_profit_1 = entry_price + (risk_amount * 2)  # 2:1 R/R
        take_profit_2 = entry_price + (risk_amount * 3)  # 3:1 R/R
        take_profit_3 = entry_price + (risk_amount * 5)  # 5:1 R/R
        
        return {
            "entry_price": entry_price,
            "stop_loss": round(final_sl, 4),
            "stop_loss_percent": round((1 - final_sl/entry_price) * 100, 2),
            "take_profit_targets": [
                {"target": 1, "price": round(take_profit_1, 4), "rr_ratio": "2:1"},
                {"target": 2, "price": round(take_profit_2, 4), "rr_ratio": "3:1"},
                {"target": 3, "price": round(take_profit_3, 4), "rr_ratio": "5:1"}
            ],
            "risk_reward_minimum": "2:1",
            "volatility_used": round(volatility * 100, 2),
            "max_loss_usd": round(self.current_capital * self.max_risk_per_trade, 2)
        }


# Singleton para uso global
_risk_manager: Optional[RiskManager] = None


def get_risk_manager(initial_capital: float = 10000.0) -> RiskManager:
    """Retorna instância singleton do RiskManager"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager(initial_capital=initial_capital)
    return _risk_manager
