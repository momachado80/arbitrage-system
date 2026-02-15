"""
Whale Tracker - Sistema de Rastreamento de Carteiras
=====================================================

Monitora movimentações anômalas em mercados de previsão.
Detecta potenciais insiders e whales.
"""

from .polymarket_tracker import PolymarketWhaleTracker
from .anomaly_detector import AnomalyDetector
from .alert_system import AlertSystem, Alert, AlertType

__all__ = [
    'PolymarketWhaleTracker',
    'AnomalyDetector', 
    'AlertSystem',
    'Alert',
    'AlertType'
]



