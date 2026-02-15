"""
Configurações para o motor de arbitragem profissional.
Todos os parâmetros são configuráveis e validados.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Optional


@dataclass
class FeeConfig:
    """Configuração de taxas por plataforma."""
    
    # Polymarket
    poly_trading_fee: Decimal = Decimal("0.02")      # 2% taxa de trading
    poly_withdrawal_fee: Decimal = Decimal("0.00")   # Sem taxa de saque (USDC on Polygon)
    poly_deposit_fee: Decimal = Decimal("0.00")      # Sem taxa de depósito
    
    # Kalshi
    kalshi_trading_fee: Decimal = Decimal("0.07")    # 7% sobre lucro
    kalshi_maker_fee: Decimal = Decimal("0.00")      # Maker: 0%
    kalshi_taker_fee: Decimal = Decimal("0.01")      # Taker: 1% (algumas ordens)
    kalshi_withdrawal_fee: Decimal = Decimal("0.00") # Sem taxa USD
    kalshi_deposit_fee: Decimal = Decimal("0.00")    # Sem taxa USD
    
    # Conversão de moeda (se aplicável)
    fx_spread: Decimal = Decimal("0.005")            # 0.5% spread câmbio
    iof_rate: Decimal = Decimal("0.0038")            # IOF 0.38% (transferência internacional)
    
    def get_total_poly_fee(self) -> Decimal:
        """Retorna taxa total Polymarket por operação."""
        return self.poly_trading_fee + self.poly_withdrawal_fee + self.poly_deposit_fee
    
    def get_total_kalshi_fee(self, is_winner: bool, profit: Decimal) -> Decimal:
        """
        Retorna taxa Kalshi considerando se ganhou ou não.
        Kalshi cobra 7% apenas sobre o lucro em contratos vencedores.
        """
        if is_winner and profit > 0:
            return profit * self.kalshi_trading_fee
        return Decimal("0")


@dataclass
class ProfitConfig:
    """Configuração de colchão de lucro mínimo."""
    
    min_profit_percentage: Decimal = Decimal("2.0")    # Lucro mínimo 2%
    min_profit_absolute: Decimal = Decimal("1.0")      # Lucro mínimo $1
    
    # Penalidades
    rule_mismatch_penalty: Decimal = Decimal("0.50")   # 50% penalidade se regras diferentes
    old_data_penalty: Decimal = Decimal("0.25")        # 25% penalidade dados velhos
    low_liquidity_penalty: Decimal = Decimal("0.20")   # 20% penalidade baixa liquidez


@dataclass
class LiquidityConfig:
    """Configuração de liquidez e profundidade."""
    
    min_volume_usd: Decimal = Decimal("100.0")        # Volume mínimo $100
    target_sizes: list = field(default_factory=lambda: [100, 500, 1000])  # Tamanhos para calcular
    max_slippage_percentage: Decimal = Decimal("2.0") # Slippage máximo 2%
    min_book_depth_levels: int = 3                     # Mínimo 3 níveis no book


@dataclass
class TimingConfig:
    """Configuração de sincronização e idade dos dados."""
    
    max_data_age_seconds: int = 10         # Dados válidos por 10 segundos
    warning_age_seconds: int = 5           # Aviso após 5 segundos
    stale_age_seconds: int = 15            # Dados "velhos" após 15 segundos
    refresh_interval_ms: int = 2000        # Atualizar a cada 2 segundos


@dataclass
class RiskConfig:
    """Configuração de gestão de risco e exposição."""
    
    max_size_per_trade: Decimal = Decimal("1000.0")      # Máximo $1000 por trade
    max_size_per_market: Decimal = Decimal("5000.0")     # Máximo $5000 por mercado
    max_exposure_per_exchange: Decimal = Decimal("10000.0")  # Máximo $10000 por exchange
    max_daily_loss: Decimal = Decimal("500.0")           # Perda máxima diária $500
    max_open_positions: int = 10                          # Máximo 10 posições abertas


@dataclass
class MatchingConfig:
    """Configuração de equivalência de mercados."""
    
    min_title_similarity: float = 0.92       # 92% similaridade mínima
    require_same_expiration: bool = True     # Exigir mesma data de expiração
    max_expiration_diff_hours: int = 24      # Máxima diferença de 24h na expiração
    require_same_resolution_source: bool = False  # Verificar fonte de resolução
    
    # Lista de termos que indicam diferenças importantes
    disqualifying_terms: list = field(default_factory=lambda: [
        "including", "excluding", "only", "exactly", 
        "at least", "at most", "more than", "less than",
        "before", "after", "by", "until"
    ])


@dataclass
class ArbitrageConfig:
    """Configuração completa do motor de arbitragem."""
    
    fees: FeeConfig = field(default_factory=FeeConfig)
    profit: ProfitConfig = field(default_factory=ProfitConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    
    def to_dict(self) -> Dict:
        """Converte configuração para dicionário."""
        return {
            "fees": {
                "poly_trading_fee": str(self.fees.poly_trading_fee),
                "kalshi_trading_fee": str(self.fees.kalshi_trading_fee),
                "kalshi_taker_fee": str(self.fees.kalshi_taker_fee),
            },
            "profit": {
                "min_profit_percentage": str(self.profit.min_profit_percentage),
                "min_profit_absolute": str(self.profit.min_profit_absolute),
            },
            "liquidity": {
                "min_volume_usd": str(self.liquidity.min_volume_usd),
                "target_sizes": self.liquidity.target_sizes,
            },
            "timing": {
                "max_data_age_seconds": self.timing.max_data_age_seconds,
            },
            "risk": {
                "max_size_per_trade": str(self.risk.max_size_per_trade),
                "max_size_per_market": str(self.risk.max_size_per_market),
            },
            "matching": {
                "min_title_similarity": self.matching.min_title_similarity,
            }
        }


# Configuração padrão
DEFAULT_CONFIG = ArbitrageConfig()



