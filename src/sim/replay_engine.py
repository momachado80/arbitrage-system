"""
Replay Engine — Sequential event-driven replay of recorded snapshots.
Deterministic iteration. Handles 100k+ events with constant memory overhead.
"""

from typing import Any, Callable, Dict, List, Optional


class ReplayEngine:
    """Iterate through time-sorted market events and dispatch to a simulator."""

    def __init__(self, events: List[Dict[str, Any]]) -> None:
        self.events = events
        self._processed = 0

    @property
    def processed_count(self) -> int:
        return self._processed

    def run(self, on_event: Callable[[Dict[str, Any]], None]) -> int:
        """Replay all events in order.  Returns count of events processed."""
        self._processed = 0
        for event in self.events:
            on_event(event)
            self._processed += 1
        return self._processed

    def run_filtered(
        self,
        on_event: Callable[[Dict[str, Any]], None],
        market_key: Optional[str] = None,
        from_ts: Optional[float] = None,
        to_ts: Optional[float] = None,
    ) -> int:
        """Replay with optional time/market filters."""
        self._processed = 0
        for event in self.events:
            ts = event.get("ts_utc", 0)
            if from_ts is not None and ts < from_ts:
                continue
            if to_ts is not None and ts > to_ts:
                continue
            if market_key is not None and event.get("token_id") != market_key:
                continue
            on_event(event)
            self._processed += 1
        return self._processed
