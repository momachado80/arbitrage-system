try:
    from monitors.news_monitor import NewsMonitor, TrendingEvent
    from monitors.whale_monitor import WhaleMonitor
    from monitors.polymarket_monitor import PolymarketMonitor, MarketMomentum
except ImportError:
    from .news_monitor import NewsMonitor, TrendingEvent
    from .whale_monitor import WhaleMonitor
    from .polymarket_monitor import PolymarketMonitor, MarketMomentum

__all__ = [
    "NewsMonitor", "TrendingEvent",
    "WhaleMonitor",
    "PolymarketMonitor", "MarketMomentum"
]

