try:
    from engine.signal_engine import SignalEngine, SignalComponents
except ImportError:
    from .signal_engine import SignalEngine, SignalComponents

__all__ = ["SignalEngine", "SignalComponents"]

