try:
    from alerts.alert_system import AlertSystem, AlertConfig, create_alert_system
except ImportError:
    from .alert_system import AlertSystem, AlertConfig, create_alert_system

__all__ = ["AlertSystem", "AlertConfig", "create_alert_system"]

