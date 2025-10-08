# Agent Learning Extension

The Agent Learning extension provides metrics tracking, persistence, and API access for monitoring local agent learning progress.

## Features

- **Metrics Tracking**: Tracks success rate, reaction time, logic, memory, and creativity metrics
- **Progress Calculation**: Calculates overall progress percentage (0-100%)
- **Event Bus Integration**: Automatically listens to `task_completed` events
- **Data Persistence**: Stores metrics in JSON format with atomic writes
- **Bounded History**: Limits each metric to 500 entries to prevent unlimited growth
- **Thread-Safe**: Uses locks to ensure thread-safe metric updates
- **API Endpoint**: Exposes metrics via Flask Blueprint

## Installation

1. The extension is automatically available in the `extensions/` directory
2. Enable it by adding `agent_learning` to your extensions list:
   ```bash
   python server.py --extensions agent_learning
   ```

## Dependencies

- **agent_bus**: Event bus extension (included)
- **Flask**: For API endpoint (optional, included in requirements)

## Usage

### Via Event Bus

Publish `task_completed` events to automatically update metrics:

```python
from extensions.agent_bus.bus import bus

# Publish a task completion event
bus.publish("task_completed", {
    "success": True,
    "reaction_time": 2.5,  # in seconds
    "weights": {
        "logic": 1.0,
        "memory": 0.9,
        "creativity": 0.8
    }
})
```

### Via API Endpoint

Access metrics via the Flask API endpoint:

```bash
# Get current learning progress
GET /learning/get_learning_progress
```

Response format:
```json
{
    "success_rate": [1.0, 0.0, 1.0],
    "reaction_time": [0.95, 0.85, 0.92],
    "logic": [1.0, 0.5, 1.0],
    "memory": [0.9, 0.45, 0.9],
    "creativity": [0.8, 0.4, 0.8],
    "progress": 75.5,
    "last_update_ts": 1234567890.123,
    "version": 1
}
```

### Programmatic Access

Access metrics directly in Python:

```python
from extensions.agent_learning.learning_metrics import metrics

# Get current metrics
current = metrics.get_metrics()
print(f"Progress: {current['progress']:.2f}%")

# Manually update metrics
metrics.update(
    success=True,
    reaction_time=3.0,
    weights={"logic": 1.0, "memory": 1.0, "creativity": 0.9}
)
```

## Data Storage

Metrics are stored in `extensions/agent_learning/data/progress.json`:

- **Location**: Auto-created on first update
- **Format**: JSON with pretty printing
- **Atomic Writes**: Uses temporary file to prevent corruption
- **Bounded**: Each metric array limited to 500 entries

## Metrics Explanation

### success_rate
- Range: 0.0 (failure) to 1.0 (success)
- Tracks whether tasks completed successfully

### reaction_time
- Range: 0.0 (slow) to 1.0 (fast)
- Normalized reaction time (capped at 60 seconds)
- Formula: `1.0 - (min(time, 60) / 60)`

### logic, memory, creativity
- Range: 0.0 to weight value
- On success: Uses full weight value
- On failure: Uses 50% of weight value
- Default weights: 1.0 for all

### progress
- Range: 0.0% to 100.0%
- Calculated as: Average of all metrics × 100
- Updated automatically on each metric update

## Configuration

### MAX_HISTORY
Default: 500 entries per metric

To change, edit `learning_metrics.py`:
```python
MAX_HISTORY = 1000  # Store more history
```

## Extension Structure

```
extensions/agent_learning/
├── __init__.py              # Module initialization
├── script.py                # Extension lifecycle
├── learning_metrics.py      # Core metrics logic
├── learning_api.py          # Flask Blueprint API
├── data/
│   └── progress.json        # Persisted metrics
└── README.md               # This file
```

## Thread Safety

All metric operations are protected by a reentrant lock (`threading.RLock`):
- Safe to call from multiple threads
- Event callbacks are thread-safe
- File writes are atomic

## Error Handling

- Import errors are caught and logged
- Invalid data is rejected with error messages
- File I/O errors don't crash the extension
- Metrics calculation errors set progress to 0.0

## Integration with text-generation-webui

The extension integrates seamlessly:
1. Auto-loads when specified in `--extensions`
2. Prints startup message with current progress
3. No UI components (background service)
4. No interference with core functionality

## Example: Complete Workflow

```python
# 1. Import required modules
from extensions.agent_bus.bus import bus
from extensions.agent_learning.learning_metrics import metrics

# 2. Check initial progress
initial = metrics.get_metrics()
print(f"Starting progress: {initial['progress']:.2f}%")

# 3. Simulate task completion
for i in range(10):
    success = i % 3 != 0  # Simulate some failures
    reaction_time = 2.0 + i * 0.1
    
    bus.publish("task_completed", {
        "success": success,
        "reaction_time": reaction_time,
        "weights": {
            "logic": 1.0,
            "memory": 0.9,
            "creativity": 0.8
        }
    })

# 4. Check final progress
final = metrics.get_metrics()
print(f"Final progress: {final['progress']:.2f}%")
print(f"Tasks tracked: {len(final['success_rate'])}")
```

## Troubleshooting

### Extension not loading
- Check that `agent_bus` extension is available
- Verify Python path includes extensions directory
- Check for import errors in terminal output

### Metrics not updating
- Verify events are being published to `task_completed`
- Check that weights are numeric values
- Look for error messages in console

### API endpoint not accessible
- Ensure Flask is installed
- Check that the Blueprint is registered with your Flask app
- Verify the route: `/learning/get_learning_progress`

### Data not persisting
- Check file permissions for `data/` directory
- Verify disk space is available
- Look for file I/O errors in logs

## Contributing

See `.github/ISSUE_TEMPLATE/agent_learning_task.md` for issue template.
See `.github/PULL_REQUEST_TEMPLATE/add_agent_learning_module.md` for PR template.

## License

Same as text-generation-webui project.
