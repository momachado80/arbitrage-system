"""
Sistema de Alertas Multi-Canal
==============================
Envia alertas via Telegram, Push Notifications e Email.
"""
import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuração de alertas"""
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    email_smtp_host: Optional[str] = None
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = None
    webhook_url: Optional[str] = None  # Para push notifications
    
    def __post_init__(self):
        if self.email_to is None:
            self.email_to = []


class AlertSystem:
    """
    Sistema de alertas multi-canal.
    
    Canais suportados:
    - Telegram Bot
    - Email (SMTP)
    - Webhook (para push notifications)
    """
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._alerts_sent: List[Dict[str, Any]] = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    # =========================================================================
    # TELEGRAM
    # =========================================================================
    
    async def send_telegram(
        self,
        message: str,
        parse_mode: str = "HTML"
    ) -> bool:
        """
        Envia mensagem via Telegram Bot.
        """
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            logger.warning("Telegram não configurado")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            
            payload = {
                "chat_id": self.config.telegram_chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": False
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("✅ Alerta Telegram enviado")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Erro Telegram: {error}")
                    return False
        
        except Exception as e:
            logger.error(f"Erro ao enviar Telegram: {e}")
            return False
    
    def format_telegram_signal(self, signal: Dict[str, Any]) -> str:
        """
        Formata sinal para Telegram.
        """
        urgencia_emoji = {
            "CRÍTICA": "🔴",
            "NORMAL": "🟡",
            "MONITORAR": "⚪"
        }
        
        emoji = urgencia_emoji.get(signal.get("urgencia", ""), "⚪")
        
        message = f"""
{emoji} <b>SINAL {signal.get('urgencia', 'NOVO')}</b> - {signal.get('categoria', '')} ({signal.get('confianca', 0):.0f}% conf)

<b>{signal.get('evento', {}).get('titulo', 'Evento')}</b>

🐋 <b>Whale Activity:</b>
• Volume: ${signal.get('whale_activity', {}).get('volume_total_usd', 0):,.0f}
• Timing: {signal.get('whale_activity', {}).get('tempo_antes_noticia', 'N/A')}
• Carteiras: {signal.get('whale_activity', {}).get('whales_detectados', 0)}

📊 <b>Mercado:</b>
• {signal.get('mercados_polymarket', [{}])[0].get('titulo', 'N/A')[:50]}...
• Entrada: {signal.get('recomendacao', {}).get('preco_entrada', 0):.2f}
• Alvo: {signal.get('recomendacao', {}).get('alvo_principal', 0):.2f}
• ROI: {signal.get('recomendacao', {}).get('roi_esperado', '+0%')}

🎯 <b>Recomendação:</b> {signal.get('recomendacao', {}).get('acao', 'COMPRAR')}

<a href="{signal.get('botao_acao', {}).get('link_direto', '')}">🚀 ENTRAR AGORA</a>
"""
        return message.strip()
    
    # =========================================================================
    # WEBHOOK (Push Notifications)
    # =========================================================================
    
    async def send_webhook(
        self,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Envia via webhook (para integração com serviços de push).
        """
        if not self.config.webhook_url:
            logger.warning("Webhook não configurado")
            return False
        
        try:
            async with self.session.post(
                self.config.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status in [200, 201, 202]:
                    logger.info("✅ Webhook enviado")
                    return True
                else:
                    logger.error(f"Erro webhook: {response.status}")
                    return False
        
        except Exception as e:
            logger.error(f"Erro ao enviar webhook: {e}")
            return False
    
    def format_push_notification(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formata sinal para push notification.
        """
        return {
            "title": f"⚡ Sinal {signal.get('categoria', '')} ({signal.get('confianca', 0):.0f}%)",
            "body": f"{signal.get('evento', {}).get('titulo', '')[:50]}... | ROI: {signal.get('recomendacao', {}).get('roi_esperado', '+0%')}",
            "url": signal.get("botao_acao", {}).get("link_direto", ""),
            "data": {
                "signal_id": signal.get("id"),
                "categoria": signal.get("categoria"),
                "confianca": signal.get("confianca")
            }
        }
    
    # =========================================================================
    # EMAIL
    # =========================================================================
    
    async def send_email(
        self,
        subject: str,
        body: str,
        html: bool = True
    ) -> bool:
        """
        Envia email via SMTP.
        """
        if not all([
            self.config.email_smtp_host,
            self.config.email_username,
            self.config.email_password,
            self.config.email_to
        ]):
            logger.warning("Email não configurado")
            return False
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.email_from or self.config.email_username
            msg["To"] = ", ".join(self.config.email_to)
            
            part = MIMEText(body, "html" if html else "plain")
            msg.attach(part)
            
            with smtplib.SMTP(self.config.email_smtp_host, self.config.email_smtp_port) as server:
                server.starttls()
                server.login(self.config.email_username, self.config.email_password)
                server.sendmail(
                    self.config.email_from or self.config.email_username,
                    self.config.email_to,
                    msg.as_string()
                )
            
            logger.info("✅ Email enviado")
            return True
        
        except Exception as e:
            logger.error(f"Erro ao enviar email: {e}")
            return False
    
    def format_email_signal(self, signal: Dict[str, Any]) -> tuple:
        """
        Formata sinal para email.
        Returns: (subject, body_html)
        """
        urgencia_emoji = {
            "CRÍTICA": "🔴",
            "NORMAL": "🟡",
            "MONITORAR": "⚪"
        }
        
        emoji = urgencia_emoji.get(signal.get("urgencia", ""), "⚪")
        
        subject = f"{emoji} SINAL {signal.get('urgencia', '')}: {signal.get('evento', {}).get('titulo', '')[:40]}... - {signal.get('confianca', 0):.0f}% confiança"
        
        body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
        .section {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 8px; }}
        .btn {{ display: inline-block; background: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; margin: 20px 0; }}
        .highlight {{ color: #2196F3; font-weight: bold; }}
        .warning {{ color: #ff9800; }}
        .stats {{ display: flex; justify-content: space-between; }}
        .stat {{ text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{emoji} SINAL {signal.get('urgencia', '')} - {signal.get('confianca', 0):.0f}% Confiança</h1>
        <p>Categoria: {signal.get('categoria', '')}</p>
    </div>
    
    <h2>{signal.get('evento', {}).get('titulo', 'Evento')}</h2>
    <p>{signal.get('evento', {}).get('descricao', '')}</p>
    <p><small>Fonte: {signal.get('evento', {}).get('fonte', '')} | {signal.get('evento', {}).get('mentions', 0)} menções</small></p>
    
    <div class="section">
        <h3>🐋 Atividade de Whale</h3>
        <p>Seu sistema detectou movimentação de <span class="highlight">${signal.get('whale_activity', {}).get('volume_total_usd', 0):,.0f}</span> em capital</p>
        <p>Timing: <span class="highlight">{signal.get('whale_activity', {}).get('tempo_antes_noticia', 'N/A')}</span> antes da notícia</p>
        <p>{signal.get('whale_activity', {}).get('whales_detectados', 0)} carteiras whale detectadas</p>
    </div>
    
    <div class="section">
        <h3>📊 Mercado Recomendado</h3>
        <p><strong>{signal.get('mercados_polymarket', [{}])[0].get('titulo', 'N/A')}</strong></p>
        <div class="stats">
            <div class="stat">
                <div class="highlight">Entrada</div>
                <div>{signal.get('recomendacao', {}).get('preco_entrada', 0):.2f}</div>
            </div>
            <div class="stat">
                <div class="highlight">Alvo</div>
                <div>{signal.get('recomendacao', {}).get('alvo_principal', 0):.2f}</div>
            </div>
            <div class="stat">
                <div class="highlight">ROI Esperado</div>
                <div>{signal.get('recomendacao', {}).get('roi_esperado', '+0%')}</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h3>🎯 Recomendação</h3>
        <p><strong>{signal.get('recomendacao', {}).get('acao', 'COMPRAR')}</strong></p>
        <p>Posição sugerida: ${signal.get('recomendacao', {}).get('tamanho_posicao_sugerido', 0):,.0f}</p>
        <p>Stop Loss: {signal.get('recomendacao', {}).get('stop_loss', 0):.2f}</p>
        <p>Timing: {signal.get('recomendacao', {}).get('timing_esperado', '72h')}</p>
    </div>
    
    <div class="section">
        <h3>📈 Histórico da Categoria</h3>
        <p>Taxa de acerto: <span class="highlight">{signal.get('historico_categoria', {}).get(f"{signal.get('categoria', '').lower()}_taxa_acerto", '0%')}</span></p>
        <p>ROI médio: <span class="highlight">{signal.get('historico_categoria', {}).get(f"{signal.get('categoria', '').lower()}_roi_medio", '+0%')}</span></p>
    </div>
    
    <center>
        <a href="{signal.get('botao_acao', {}).get('link_direto', '')}" class="btn">🚀 ABRIR POLYMARKET AGORA</a>
    </center>
    
    <p class="warning">⚠️ Atenção: Trading envolve riscos. Esta é uma recomendação baseada em análise automatizada. Faça sua própria pesquisa.</p>
    
    <hr>
    <p><small>Smart Event Trader | Alertas Automatizados</small></p>
</body>
</html>
"""
        return subject, body
    
    # =========================================================================
    # ENVIO UNIFICADO
    # =========================================================================
    
    async def send_signal_alert(
        self,
        signal: Dict[str, Any],
        channels: List[str] = None
    ) -> Dict[str, bool]:
        """
        Envia alerta de sinal para todos os canais configurados.
        
        Args:
            signal: Dicionário do sinal (signal.to_dict())
            channels: Lista de canais ["telegram", "email", "push"]
                     None = todos configurados
        
        Returns:
            Dict com status de cada canal
        """
        if channels is None:
            channels = ["telegram", "email", "push"]
        
        results = {}
        
        # Telegram
        if "telegram" in channels:
            message = self.format_telegram_signal(signal)
            results["telegram"] = await self.send_telegram(message)
        
        # Email
        if "email" in channels:
            subject, body = self.format_email_signal(signal)
            results["email"] = await self.send_email(subject, body)
        
        # Push (webhook)
        if "push" in channels:
            payload = self.format_push_notification(signal)
            results["push"] = await self.send_webhook(payload)
        
        # Registra alerta
        self._alerts_sent.append({
            "signal_id": signal.get("id"),
            "timestamp": datetime.utcnow().isoformat(),
            "channels": results
        })
        
        return results
    
    def get_alerts_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna histórico de alertas enviados"""
        return self._alerts_sent[-limit:]


# Factory function
def create_alert_system(
    telegram_token: str = None,
    telegram_chat: str = None,
    webhook_url: str = None,
    **email_config
) -> AlertSystem:
    """
    Cria sistema de alertas com configuração.
    """
    config = AlertConfig(
        telegram_bot_token=telegram_token,
        telegram_chat_id=telegram_chat,
        webhook_url=webhook_url,
        **email_config
    )
    return AlertSystem(config)

