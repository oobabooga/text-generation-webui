import json
import os
import time
import threading
from statistics import mean
from pathlib import Path
from typing import Dict, Any, Optional
from extensions.agent_bus.bus import bus  # Expected local EventBus

DATA_FILE = Path(__file__).parent / "data" / "progress.json"
MAX_HISTORY = 500  # limit arrays length to prevent unlimited growth

class LearningMetrics:
    """Maintain and update local agent learning metrics."""
    def __init__(self):
        self._lock = threading.RLock()
        self.metrics: Dict[str, Any] = {
            "success_rate": [],
            "reaction_time": [],
            "logic": [],
            "memory": [],
            "creativity": [],
            "progress": 0.0,
            "last_update_ts": None,
            "version": 1
        }
        self.load_data()

    def load_data(self):
        if DATA_FILE.exists():
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in self.metrics.items():
                        if k not in data:
                            data[k] = v
                    self.metrics = data
            except Exception as e:
                print(f"[LearningMetrics] Failed to load progress.json: {e}")

    def save_data(self):
        try:
            os.makedirs(DATA_FILE.parent, exist_ok=True)
            tmp_path = DATA_FILE.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, DATA_FILE)
        except Exception as e:
            print(f"[LearningMetrics] Error saving metrics: {e}")

    def _trim(self):
        for key in ["success_rate", "reaction_time", "logic", "memory", "creativity"]:
            arr = self.metrics[key]
            if len(arr) > MAX_HISTORY:
                self.metrics[key] = arr[-MAX_HISTORY:]

    def update(self, success: bool, reaction_time: float, weights: Optional[Dict[str, float]] = None):
        with self._lock:
            weights = weights or {"logic": 1.0, "memory": 1.0, "creativity": 1.0}
            try:
                reaction_time = float(reaction_time)
                if reaction_time < 0:
                    reaction_time = 0.0
                
                # Update success rate
                self.metrics["success_rate"].append(1.0 if success else 0.0)
                
                # Update reaction time (normalized, lower is better)
                # Cap at reasonable maximum (e.g., 60 seconds)
                normalized_rt = min(reaction_time, 60.0) / 60.0
                self.metrics["reaction_time"].append(1.0 - normalized_rt)
                
                # Update weighted metrics
                for key in ["logic", "memory", "creativity"]:
                    weight = weights.get(key, 1.0)
                    if success:
                        # On success, use the weight directly
                        self.metrics[key].append(float(weight))
                    else:
                        # On failure, penalize by reducing the weight
                        self.metrics[key].append(float(weight) * 0.5)
                
                # Trim arrays to prevent unlimited growth
                self._trim()
                
                # Calculate overall progress
                self._calculate_progress()
                
                # Update timestamp
                self.metrics["last_update_ts"] = time.time()
                
                # Save to disk
                self.save_data()
                
            except Exception as e:
                print(f"[LearningMetrics] Error updating metrics: {e}")

    def _calculate_progress(self):
        """Calculate overall progress percentage based on all metrics."""
        try:
            progress_sum = 0.0
            progress_count = 0
            
            for key in ["success_rate", "reaction_time", "logic", "memory", "creativity"]:
                arr = self.metrics[key]
                if arr:
                    progress_sum += mean(arr)
                    progress_count += 1
            
            if progress_count > 0:
                # Progress is the average of all metrics (0.0 to 1.0)
                self.metrics["progress"] = (progress_sum / progress_count) * 100.0
            else:
                self.metrics["progress"] = 0.0
                
        except Exception as e:
            print(f"[LearningMetrics] Error calculating progress: {e}")
            self.metrics["progress"] = 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            return dict(self.metrics)

    def on_task_completed(self, data: Dict[str, Any]):
        """Callback for task_completed events from the event bus."""
        try:
            success = data.get("success", False)
            reaction_time = data.get("reaction_time", 0.0)
            weights = data.get("weights")
            self.update(success, reaction_time, weights)
        except Exception as e:
            print(f"[LearningMetrics] Error handling task_completed event: {e}")


# Global metrics instance
metrics = LearningMetrics()

# Subscribe to task_completed events
bus.subscribe("task_completed", metrics.on_task_completed)
