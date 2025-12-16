"""Engine module - Core calculation and simulation components."""

from .execution_simulator import ExecutionSimulator, analyze_arbitrage_potential
from .arbitrage_calculator import ArbitrageCalculator, RejectionReason
from .market_matcher import MarketMatcher, MarketPair, similarity_to_equivalence_score
from .auto_scanner import AutoScanner, ScanResult, OpportunityDetail, run_continuous_scanner

__all__ = [
    "ExecutionSimulator",
    "analyze_arbitrage_potential",
    "ArbitrageCalculator",
    "RejectionReason",
    "MarketMatcher",
    "MarketPair",
    "similarity_to_equivalence_score",
    "AutoScanner",
    "ScanResult",
    "OpportunityDetail",
    "run_continuous_scanner",
]
