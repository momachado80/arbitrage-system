"""
Anomaly Detector
================

Detecta padrões anômalos que podem indicar insider trading:
- Trades grandes antes de eventos importantes
- Carteiras novas com comportamento de whale
- Concentração incomum em mercados específicos
- Timing suspeito (trades pouco antes de resolução)
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Tipos de anomalia detectáveis."""
    WHALE_TRADE = "whale_trade"
    NEW_WALLET_LARGE_TRADE = "new_wallet_large_trade"
    VOLUME_SPIKE = "volume_spike"
    PRICE_MANIPULATION = "price_manipulation"
    CONCENTRATED_POSITION = "concentrated_position"
    TIMING_SUSPICIOUS = "timing_suspicious"
    COORDINATED_TRADING = "coordinated_trading"
    WASH_TRADING = "wash_trading"


class Severity(Enum):
    """Severidade da anomalia."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Representa uma anomalia detectada."""
    type: AnomalyType
    severity: Severity
    market_id: str
    market_title: str
    description: str
    details: Dict
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    addresses_involved: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0-1
    
    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "market_id": self.market_id,
            "market_title": self.market_title,
            "description": self.description,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "addresses": self.addresses_involved,
            "confidence": self.confidence
        }


class AnomalyDetector:
    """
    Detector de anomalias em mercados de previsão.
    """
    
    # Configurações de detecção
    CONFIG = {
        # Thresholds de volume
        "whale_threshold_usd": Decimal("10000"),
        "large_trade_threshold_usd": Decimal("5000"),
        "volume_spike_multiplier": 3.0,
        
        # Thresholds de tempo
        "new_wallet_days": 7,
        "suspicious_timing_hours": 24,  # Trades muito perto da resolução
        
        # Thresholds de comportamento
        "concentration_max_markets": 2,
        "concentration_min_trades": 5,
        "coordinated_time_window_minutes": 10,
        "coordinated_min_wallets": 3,
        
        # Wash trading
        "wash_trade_time_window_seconds": 60,
        "wash_trade_price_tolerance": Decimal("0.01"),
    }
    
    def __init__(self, config: Optional[Dict] = None):
        if config:
            self.CONFIG.update(config)
        
        # Histórico para comparações
        self.volume_history: Dict[str, List[Decimal]] = {}
        self.price_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        
    def detect_whale_trade(
        self,
        trade_value: Decimal,
        wallet_total_volume: Decimal,
        market_id: str,
        market_title: str,
        address: str
    ) -> Optional[Anomaly]:
        """Detecta trades de whales."""
        
        if trade_value >= self.CONFIG["whale_threshold_usd"]:
            severity = Severity.MEDIUM
            confidence = 0.7
            
            # Aumentar severidade se for um whale conhecido fazendo trade muito grande
            if trade_value >= self.CONFIG["whale_threshold_usd"] * 5:
                severity = Severity.HIGH
                confidence = 0.85
            
            if trade_value >= self.CONFIG["whale_threshold_usd"] * 10:
                severity = Severity.CRITICAL
                confidence = 0.95
            
            return Anomaly(
                type=AnomalyType.WHALE_TRADE,
                severity=severity,
                market_id=market_id,
                market_title=market_title,
                description=f"Trade de whale detectado: ${trade_value:,.2f}",
                details={
                    "trade_value_usd": float(trade_value),
                    "wallet_total_volume": float(wallet_total_volume),
                    "threshold": float(self.CONFIG["whale_threshold_usd"])
                },
                addresses_involved=[address],
                confidence=confidence
            )
        
        return None
    
    def detect_new_wallet_large_trade(
        self,
        trade_value: Decimal,
        wallet_first_seen: Optional[datetime],
        market_id: str,
        market_title: str,
        address: str
    ) -> Optional[Anomaly]:
        """Detecta carteiras novas fazendo trades grandes."""
        
        if not wallet_first_seen:
            return None
        
        wallet_age_days = (datetime.now(timezone.utc) - wallet_first_seen).days
        
        if (wallet_age_days <= self.CONFIG["new_wallet_days"] and 
            trade_value >= self.CONFIG["large_trade_threshold_usd"]):
            
            # Quanto mais novo e maior o trade, mais suspeito
            severity = Severity.MEDIUM
            confidence = 0.6
            
            if wallet_age_days <= 1:
                severity = Severity.HIGH
                confidence = 0.8
                
            if wallet_age_days <= 1 and trade_value >= self.CONFIG["whale_threshold_usd"]:
                severity = Severity.CRITICAL
                confidence = 0.9
            
            return Anomaly(
                type=AnomalyType.NEW_WALLET_LARGE_TRADE,
                severity=severity,
                market_id=market_id,
                market_title=market_title,
                description=f"Carteira nova ({wallet_age_days} dias) fez trade de ${trade_value:,.2f}",
                details={
                    "trade_value_usd": float(trade_value),
                    "wallet_age_days": wallet_age_days,
                    "threshold_days": self.CONFIG["new_wallet_days"],
                    "threshold_usd": float(self.CONFIG["large_trade_threshold_usd"])
                },
                addresses_involved=[address],
                confidence=confidence
            )
        
        return None
    
    def detect_volume_spike(
        self,
        current_volume: Decimal,
        historical_volumes: List[Decimal],
        market_id: str,
        market_title: str
    ) -> Optional[Anomaly]:
        """Detecta picos anormais de volume."""
        
        if len(historical_volumes) < 3:
            return None
        
        avg_volume = sum(historical_volumes) / len(historical_volumes)
        
        if avg_volume > 0:
            multiplier = float(current_volume / avg_volume)
            
            if multiplier >= self.CONFIG["volume_spike_multiplier"]:
                severity = Severity.MEDIUM
                confidence = min(0.5 + (multiplier - 3) * 0.1, 0.9)
                
                if multiplier >= 5:
                    severity = Severity.HIGH
                if multiplier >= 10:
                    severity = Severity.CRITICAL
                
                return Anomaly(
                    type=AnomalyType.VOLUME_SPIKE,
                    severity=severity,
                    market_id=market_id,
                    market_title=market_title,
                    description=f"Volume {multiplier:.1f}x acima da média",
                    details={
                        "current_volume": float(current_volume),
                        "average_volume": float(avg_volume),
                        "multiplier": multiplier,
                        "threshold_multiplier": self.CONFIG["volume_spike_multiplier"]
                    },
                    confidence=confidence
                )
        
        return None
    
    def detect_concentrated_position(
        self,
        wallet_markets_count: int,
        wallet_total_trades: int,
        wallet_total_volume: Decimal,
        market_id: str,
        market_title: str,
        address: str
    ) -> Optional[Anomaly]:
        """Detecta posições muito concentradas em poucos mercados."""
        
        if (wallet_total_trades >= self.CONFIG["concentration_min_trades"] and
            wallet_markets_count <= self.CONFIG["concentration_max_markets"] and
            wallet_total_volume >= self.CONFIG["large_trade_threshold_usd"]):
            
            concentration = wallet_total_trades / max(wallet_markets_count, 1)
            
            severity = Severity.LOW
            confidence = 0.5
            
            if concentration >= 10:
                severity = Severity.MEDIUM
                confidence = 0.7
            
            if concentration >= 20 and wallet_total_volume >= self.CONFIG["whale_threshold_usd"]:
                severity = Severity.HIGH
                confidence = 0.8
            
            return Anomaly(
                type=AnomalyType.CONCENTRATED_POSITION,
                severity=severity,
                market_id=market_id,
                market_title=market_title,
                description=f"Posição concentrada: {wallet_total_trades} trades em {wallet_markets_count} mercados",
                details={
                    "total_trades": wallet_total_trades,
                    "markets_count": wallet_markets_count,
                    "total_volume_usd": float(wallet_total_volume),
                    "concentration_ratio": concentration
                },
                addresses_involved=[address],
                confidence=confidence
            )
        
        return None
    
    def detect_suspicious_timing(
        self,
        trade_timestamp: datetime,
        market_resolution_time: Optional[datetime],
        trade_value: Decimal,
        market_id: str,
        market_title: str,
        address: str
    ) -> Optional[Anomaly]:
        """Detecta trades feitos muito perto da resolução do mercado."""
        
        if not market_resolution_time:
            return None
        
        time_until_resolution = market_resolution_time - trade_timestamp
        hours_until = time_until_resolution.total_seconds() / 3600
        
        if (0 < hours_until <= self.CONFIG["suspicious_timing_hours"] and
            trade_value >= self.CONFIG["large_trade_threshold_usd"]):
            
            severity = Severity.MEDIUM
            confidence = 0.6
            
            if hours_until <= 6:
                severity = Severity.HIGH
                confidence = 0.75
            
            if hours_until <= 1:
                severity = Severity.CRITICAL
                confidence = 0.9
            
            return Anomaly(
                type=AnomalyType.TIMING_SUSPICIOUS,
                severity=severity,
                market_id=market_id,
                market_title=market_title,
                description=f"Trade de ${trade_value:,.2f} apenas {hours_until:.1f}h antes da resolução",
                details={
                    "trade_value_usd": float(trade_value),
                    "hours_until_resolution": hours_until,
                    "resolution_time": market_resolution_time.isoformat()
                },
                addresses_involved=[address],
                confidence=confidence
            )
        
        return None
    
    def detect_coordinated_trading(
        self,
        trades: List[Dict],  # Lista de trades com timestamp, address, value
        market_id: str,
        market_title: str
    ) -> Optional[Anomaly]:
        """Detecta possível trading coordenado entre múltiplas carteiras."""
        
        if len(trades) < self.CONFIG["coordinated_min_wallets"]:
            return None
        
        # Agrupar trades por janela de tempo
        window_minutes = self.CONFIG["coordinated_time_window_minutes"]
        
        # Ordenar por timestamp
        sorted_trades = sorted(trades, key=lambda t: t.get("timestamp", datetime.min))
        
        # Procurar clusters de trades diferentes em janelas curtas
        for i, trade in enumerate(sorted_trades):
            cluster = [trade]
            addresses_in_cluster = {trade.get("address", "").lower()}
            
            for j in range(i + 1, len(sorted_trades)):
                other = sorted_trades[j]
                time_diff = (other.get("timestamp") - trade.get("timestamp")).total_seconds() / 60
                
                if time_diff <= window_minutes:
                    other_addr = other.get("address", "").lower()
                    if other_addr not in addresses_in_cluster:
                        cluster.append(other)
                        addresses_in_cluster.add(other_addr)
                else:
                    break
            
            # Verificar se há coordenação suspeita
            if len(addresses_in_cluster) >= self.CONFIG["coordinated_min_wallets"]:
                total_volume = sum(Decimal(str(t.get("value", 0))) for t in cluster)
                
                if total_volume >= self.CONFIG["whale_threshold_usd"]:
                    return Anomaly(
                        type=AnomalyType.COORDINATED_TRADING,
                        severity=Severity.HIGH,
                        market_id=market_id,
                        market_title=market_title,
                        description=f"{len(addresses_in_cluster)} carteiras diferentes fizeram trades em {window_minutes} minutos",
                        details={
                            "wallets_count": len(addresses_in_cluster),
                            "total_volume": float(total_volume),
                            "time_window_minutes": window_minutes,
                            "trades": [
                                {
                                    "address": t.get("address"),
                                    "value": float(t.get("value", 0)),
                                    "time": t.get("timestamp").isoformat() if t.get("timestamp") else None
                                } for t in cluster
                            ]
                        },
                        addresses_involved=list(addresses_in_cluster),
                        confidence=0.7
                    )
        
        return None
    
    def analyze_market(
        self,
        market_id: str,
        market_title: str,
        trades: List[Dict],
        market_resolution_time: Optional[datetime] = None,
        historical_volumes: Optional[List[Decimal]] = None
    ) -> List[Anomaly]:
        """
        Análise completa de um mercado para detectar todas as anomalias.
        """
        anomalies = []
        
        # Calcular volume atual
        current_volume = sum(Decimal(str(t.get("value", 0))) for t in trades)
        
        # 1. Volume spike
        if historical_volumes:
            anomaly = self.detect_volume_spike(
                current_volume, historical_volumes, market_id, market_title
            )
            if anomaly:
                anomalies.append(anomaly)
        
        # 2. Trading coordenado
        anomaly = self.detect_coordinated_trading(trades, market_id, market_title)
        if anomaly:
            anomalies.append(anomaly)
        
        # 3. Análise individual de cada trade
        for trade in trades:
            trade_value = Decimal(str(trade.get("value", 0)))
            address = trade.get("address", "")
            wallet_data = trade.get("wallet_data", {})
            
            # Whale trade
            anomaly = self.detect_whale_trade(
                trade_value,
                Decimal(str(wallet_data.get("total_volume", 0))),
                market_id, market_title, address
            )
            if anomaly:
                anomalies.append(anomaly)
            
            # New wallet large trade
            first_seen = wallet_data.get("first_seen")
            if first_seen and isinstance(first_seen, str):
                first_seen = datetime.fromisoformat(first_seen.replace("Z", "+00:00"))
            
            anomaly = self.detect_new_wallet_large_trade(
                trade_value, first_seen, market_id, market_title, address
            )
            if anomaly:
                anomalies.append(anomaly)
            
            # Concentrated position
            anomaly = self.detect_concentrated_position(
                wallet_data.get("markets_count", 0),
                wallet_data.get("total_trades", 0),
                Decimal(str(wallet_data.get("total_volume", 0))),
                market_id, market_title, address
            )
            if anomaly:
                anomalies.append(anomaly)
            
            # Suspicious timing
            trade_time = trade.get("timestamp")
            if trade_time and isinstance(trade_time, str):
                trade_time = datetime.fromisoformat(trade_time.replace("Z", "+00:00"))
            
            if trade_time and market_resolution_time:
                anomaly = self.detect_suspicious_timing(
                    trade_time, market_resolution_time,
                    trade_value, market_id, market_title, address
                )
                if anomaly:
                    anomalies.append(anomaly)
        
        # Ordenar por severidade
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3
        }
        anomalies.sort(key=lambda a: severity_order[a.severity])
        
        return anomalies



