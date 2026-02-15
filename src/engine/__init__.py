"""Engine module - Core calculation and simulation components."""

from .execution_simulator import ExecutionSimulator, analyze_arbitrage_potential
from .arbitrage_calculator import ArbitrageCalculator, RejectionReason
from .contract_equivalence import (
    ContractEquivalenceAnalyzer,
    EquivalenceResult,
    EquivalenceLevel,
    equivalence_to_penalty,
)
from .market_matcher import MarketMatcher, MarketPair, similarity_to_equivalence_score
from .auto_scanner import AutoScanner, ScanResult, OpportunityDetail, run_continuous_scanner
from .market_indexer import MarketIndexer, MarketIndex
from .large_scale_scanner import LargeScaleScanner, FullScanResult
from .probability_calibrator import (
    ProbabilityCalibrator,
    CategoryTypeMetrics,
    MarketCategory,
    TradeType,
    calculate_brier_score,
    calculate_lambda,
    # Constantes
    SCHEMA_VERSION,
    HALF_LIFE_FAST,
    HALF_LIFE_SLOW,
    E_MAX,
    ALPHA,
    K_MIN,
    K_MIN_GLOBAL,
    N_MIN,
    THETA,
    BASELINE_BRIER_BY_CATEGORY,
)

__all__ = [
    "ExecutionSimulator",
    "analyze_arbitrage_potential",
    "ArbitrageCalculator",
    "RejectionReason",
    "ContractEquivalenceAnalyzer",
    "EquivalenceResult",
    "EquivalenceLevel",
    "equivalence_to_penalty",
    "MarketMatcher",
    "MarketPair",
    "similarity_to_equivalence_score",
    "AutoScanner",
    "ScanResult",
    "OpportunityDetail",
    "run_continuous_scanner",
    "MarketIndexer",
    "MarketIndex",
    "LargeScaleScanner",
    "FullScanResult",
    # Probability Calibration
    "ProbabilityCalibrator",
    "CategoryTypeMetrics",
    "MarketCategory",
    "TradeType",
    "calculate_brier_score",
    "calculate_lambda",
    "SCHEMA_VERSION",
    "HALF_LIFE_FAST",
    "HALF_LIFE_SLOW",
    "E_MAX",
    "ALPHA",
    "K_MIN",
    "K_MIN_GLOBAL",
    "N_MIN",
    "THETA",
    "BASELINE_BRIER_BY_CATEGORY",
]
