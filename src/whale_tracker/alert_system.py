"""
Alert System
============

Sistema de alertas para notificar sobre atividades anômalas.
Suporta múltiplos canais: console, arquivo, webhook, Telegram.
"""

import asyncio
import aiohttp
import ssl
import certifi
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Tipos de alerta."""
    WHALE_DETECTED = "whale_detected"
    NEW_WALLET_ACTIVITY = "new_wallet_activity"
    VOLUME_SPIKE = "volume_spike"
    SUSPICIOUS_TIMING = "suspicious_timing"
    COORDINATED_TRADING = "coordinated_trading"
    WATCHED_WALLET_ACTIVE = "watched_wallet_active"
    MARKET_ANOMALY = "market_anomaly"
    POTENTIAL_INSIDER = "potential_insider"


class AlertPriority(Enum):
    """Prioridade do alerta."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Alert:
    """Representa um alerta."""
    id: str
    type: AlertType
    priority: AlertPriority
    title: str
    message: str
    market_id: Optional[str] = None
    market_title: Optional[str] = None
    addresses: List[str] = field(default_factory=list)
    data: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "priority_name": self.priority.name,
            "title": self.title,
            "message": self.message,
            "market_id": self.market_id,
            "market_title": self.market_title,
            "addresses": self.addresses,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }
    
    def format_console(self) -> str:
        """Formata alerta para exibição no console."""
        priority_emoji = {
            AlertPriority.LOW: "ℹ️",
            AlertPriority.MEDIUM: "⚠️",
            AlertPriority.HIGH: "🚨",
            AlertPriority.CRITICAL: "🔴"
        }
        
        emoji = priority_emoji.get(self.priority, "📢")
        time_str = self.timestamp.strftime("%H:%M:%S")
        
        lines = [
            f"\n{'='*60}",
            f"{emoji} [{self.priority.name}] {self.title}",
            f"🕐 {time_str}",
            f"📝 {self.message}"
        ]
        
        if self.market_title:
            lines.append(f"📊 Mercado: {self.market_title}")
        
        if self.addresses:
            lines.append(f"👛 Carteiras: {', '.join(a[:10] + '...' for a in self.addresses[:3])}")
        
        if self.data:
            for key, value in list(self.data.items())[:5]:
                if isinstance(value, float):
                    lines.append(f"   • {key}: ${value:,.2f}" if 'usd' in key.lower() or 'volume' in key.lower() else f"   • {key}: {value:.2f}")
                else:
                    lines.append(f"   • {key}: {value}")
        
        lines.append(f"{'='*60}")
        
        return "\n".join(lines)
    
    def format_telegram(self) -> str:
        """Formata alerta para Telegram (Markdown)."""
        priority_emoji = {
            AlertPriority.LOW: "ℹ️",
            AlertPriority.MEDIUM: "⚠️",
            AlertPriority.HIGH: "🚨",
            AlertPriority.CRITICAL: "🔴"
        }
        
        emoji = priority_emoji.get(self.priority, "📢")
        
        lines = [
            f"{emoji} *{self.priority.name}: {self.title}*",
            f"_{self.message}_"
        ]
        
        if self.market_title:
            lines.append(f"📊 `{self.market_title}`")
        
        if self.addresses:
            for addr in self.addresses[:2]:
                lines.append(f"👛 `{addr}`")
        
        if self.data:
            lines.append("")
            for key, value in list(self.data.items())[:5]:
                if isinstance(value, (int, float)):
                    lines.append(f"• {key}: `{value:,.2f}`")
                else:
                    lines.append(f"• {key}: `{value}`")
        
        return "\n".join(lines)


class AlertSystem:
    """
    Sistema central de gerenciamento de alertas.
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        webhook_url: Optional[str] = None,
        max_history: int = 1000
    ):
        self.log_file = Path(log_file) if log_file else None
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.webhook_url = webhook_url
        
        # Histórico de alertas
        self.alerts: deque = deque(maxlen=max_history)
        
        # Callbacks customizados
        self.callbacks: List[Callable[[Alert], Any]] = []
        
        # Contadores
        self.total_alerts = 0
        self.alerts_by_type: Dict[str, int] = {}
        self.alerts_by_priority: Dict[str, int] = {}
        
        # Filtros
        self.min_priority = AlertPriority.LOW
        self.muted_types: set = set()
        
        # Sessão HTTP
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        self.session = aiohttp.ClientSession(connector=connector)
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    def _generate_id(self) -> str:
        """Gera ID único para alerta."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    async def emit(self, alert: Alert) -> bool:
        """
        Emite um alerta para todos os canais configurados.
        """
        # Verificar filtros
        if alert.priority.value < self.min_priority.value:
            return False
        
        if alert.type.value in self.muted_types:
            return False
        
        # Atribuir ID se não tiver
        if not alert.id:
            alert.id = self._generate_id()
        
        # Adicionar ao histórico
        self.alerts.append(alert)
        self.total_alerts += 1
        
        # Contadores
        self.alerts_by_type[alert.type.value] = self.alerts_by_type.get(alert.type.value, 0) + 1
        self.alerts_by_priority[alert.priority.name] = self.alerts_by_priority.get(alert.priority.name, 0) + 1
        
        # Enviar para cada canal
        tasks = []
        
        # Console
        tasks.append(self._send_console(alert))
        
        # Arquivo
        if self.log_file:
            tasks.append(self._send_file(alert))
        
        # Telegram
        if self.telegram_bot_token and self.telegram_chat_id:
            tasks.append(self._send_telegram(alert))
        
        # Webhook
        if self.webhook_url:
            tasks.append(self._send_webhook(alert))
        
        # Callbacks customizados
        for callback in self.callbacks:
            try:
                result = callback(alert)
                if asyncio.iscoroutine(result):
                    tasks.append(result)
            except Exception as e:
                logger.error(f"Erro em callback: {e}")
        
        # Executar todos
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return True
    
    async def _send_console(self, alert: Alert):
        """Envia alerta para o console."""
        print(alert.format_console())
    
    async def _send_file(self, alert: Alert):
        """Salva alerta em arquivo."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.log_file, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Erro ao salvar alerta em arquivo: {e}")
    
    async def _send_telegram(self, alert: Alert):
        """Envia alerta via Telegram."""
        if not self.session:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": alert.format_telegram(),
                "parse_mode": "Markdown"
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Erro ao enviar Telegram: {await response.text()}")
        except Exception as e:
            logger.error(f"Erro ao enviar Telegram: {e}")
    
    async def _send_webhook(self, alert: Alert):
        """Envia alerta via webhook."""
        if not self.session:
            return
        
        try:
            async with self.session.post(
                self.webhook_url,
                json=alert.to_dict()
            ) as response:
                if response.status not in (200, 201, 204):
                    logger.error(f"Erro ao enviar webhook: {await response.text()}")
        except Exception as e:
            logger.error(f"Erro ao enviar webhook: {e}")
    
    def add_callback(self, callback: Callable[[Alert], Any]):
        """Adiciona callback customizado."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Alert], Any]):
        """Remove callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def set_min_priority(self, priority: AlertPriority):
        """Define prioridade mínima para alertas."""
        self.min_priority = priority
    
    def mute_type(self, alert_type: AlertType):
        """Silencia um tipo de alerta."""
        self.muted_types.add(alert_type.value)
    
    def unmute_type(self, alert_type: AlertType):
        """Reativa um tipo de alerta."""
        self.muted_types.discard(alert_type.value)
    
    def get_recent_alerts(
        self,
        limit: int = 50,
        alert_type: Optional[AlertType] = None,
        min_priority: Optional[AlertPriority] = None
    ) -> List[Alert]:
        """Retorna alertas recentes."""
        alerts = list(self.alerts)
        
        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]
        
        if min_priority:
            alerts = [a for a in alerts if a.priority.value >= min_priority.value]
        
        # Mais recentes primeiro
        alerts.reverse()
        
        return alerts[:limit]
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas dos alertas."""
        return {
            "total_alerts": self.total_alerts,
            "by_type": self.alerts_by_type,
            "by_priority": self.alerts_by_priority,
            "recent_count": len(self.alerts)
        }
    
    def acknowledge(self, alert_id: str) -> bool:
        """Marca um alerta como reconhecido."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def clear_acknowledged(self):
        """Remove alertas reconhecidos do histórico."""
        self.alerts = deque(
            (a for a in self.alerts if not a.acknowledged),
            maxlen=self.alerts.maxlen
        )


# Funções de conveniência para criar alertas
def create_whale_alert(
    trade_value: float,
    address: str,
    market_id: str,
    market_title: str,
    side: str = "YES"
) -> Alert:
    """Cria alerta de whale."""
    priority = AlertPriority.MEDIUM
    if trade_value >= 50000:
        priority = AlertPriority.HIGH
    if trade_value >= 100000:
        priority = AlertPriority.CRITICAL
    
    return Alert(
        id="",
        type=AlertType.WHALE_DETECTED,
        priority=priority,
        title=f"🐋 Whale Trade Detectado",
        message=f"Trade de ${trade_value:,.2f} em {side}",
        market_id=market_id,
        market_title=market_title,
        addresses=[address],
        data={
            "trade_value_usd": trade_value,
            "side": side
        }
    )


def create_new_wallet_alert(
    trade_value: float,
    address: str,
    wallet_age_days: int,
    market_id: str,
    market_title: str
) -> Alert:
    """Cria alerta de carteira nova."""
    priority = AlertPriority.MEDIUM
    if wallet_age_days <= 1:
        priority = AlertPriority.HIGH
    if wallet_age_days <= 1 and trade_value >= 10000:
        priority = AlertPriority.CRITICAL
    
    return Alert(
        id="",
        type=AlertType.NEW_WALLET_ACTIVITY,
        priority=priority,
        title=f"🆕 Carteira Nova com Trade Grande",
        message=f"Carteira de {wallet_age_days} dia(s) fez trade de ${trade_value:,.2f}",
        market_id=market_id,
        market_title=market_title,
        addresses=[address],
        data={
            "trade_value_usd": trade_value,
            "wallet_age_days": wallet_age_days
        }
    )


def create_suspicious_timing_alert(
    trade_value: float,
    address: str,
    hours_until_resolution: float,
    market_id: str,
    market_title: str,
    side: str
) -> Alert:
    """Cria alerta de timing suspeito."""
    priority = AlertPriority.HIGH
    if hours_until_resolution <= 6:
        priority = AlertPriority.CRITICAL
    
    return Alert(
        id="",
        type=AlertType.SUSPICIOUS_TIMING,
        priority=priority,
        title=f"⏰ Trade Suspeito Perto da Resolução",
        message=f"Trade de ${trade_value:,.2f} apenas {hours_until_resolution:.1f}h antes da resolução",
        market_id=market_id,
        market_title=market_title,
        addresses=[address],
        data={
            "trade_value_usd": trade_value,
            "hours_until_resolution": hours_until_resolution,
            "side": side
        }
    )


def create_potential_insider_alert(
    address: str,
    evidence: List[str],
    market_id: str,
    market_title: str,
    total_volume: float
) -> Alert:
    """Cria alerta de potencial insider."""
    return Alert(
        id="",
        type=AlertType.POTENTIAL_INSIDER,
        priority=AlertPriority.CRITICAL,
        title=f"🎯 Potencial Insider Detectado",
        message=f"Múltiplos indicadores de possível insider trading",
        market_id=market_id,
        market_title=market_title,
        addresses=[address],
        data={
            "total_volume_usd": total_volume,
            "evidence": evidence
        }
    )



