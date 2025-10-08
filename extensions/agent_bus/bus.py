"""Simple event bus for agent communication."""
import threading
from typing import Callable, Dict, List, Any


class EventBus:
    """Simple event bus for publishing and subscribing to events."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass
    
    def publish(self, event_type: str, data: Any = None):
        """Publish an event to all subscribers."""
        with self._lock:
            subscribers = self._subscribers.get(event_type, []).copy()
        
        for callback in subscribers:
            try:
                callback(data)
            except Exception as e:
                print(f"[EventBus] Error in callback for {event_type}: {e}")


# Global event bus instance
bus = EventBus()
