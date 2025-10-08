"""
Agent Learning Extension - Tracks and exposes learning metrics for local agent.

This extension listens to task_completed events from the agent_bus, accumulates
metrics (success_rate, reaction_time, logic, memory, creativity), calculates
overall progress, and exposes the data via a Flask API endpoint.
"""

from .learning_metrics import metrics
from .learning_api import learning_bp

params = {
    "display_name": "Agent Learning",
    "is_tab": False,
}


def setup():
    """
    Gets executed only once, when the extension is imported.
    Initializes the learning metrics system.
    """
    print("[Agent Learning] Extension loaded. Listening for task_completed events.")
    print(f"[Agent Learning] Current progress: {metrics.get_metrics()['progress']:.2f}%")


def ui():
    """
    Gets executed when the UI is drawn.
    This extension does not add UI components.
    """
    pass
