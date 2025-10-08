# Agent Bus Extension

A simple event bus for agent-to-agent and extension-to-extension communication within text-generation-webui.

## Overview

The Agent Bus provides a lightweight publish-subscribe pattern for decoupled communication between extensions and components.

## Features

- **Publish-Subscribe Pattern**: Decouple event publishers from subscribers
- **Thread-Safe**: All operations protected by reentrant locks
- **Multiple Subscribers**: Multiple callbacks can subscribe to the same event
- **Error Isolation**: Errors in one callback don't affect others
- **Simple API**: Easy to use publish/subscribe methods

## Installation

The extension is automatically available in the `extensions/` directory. Enable it if needed:

```bash
python server.py --extensions agent_bus
```

## Usage

### Basic Usage

```python
from extensions.agent_bus.bus import bus

# Define a callback function
def on_event(data):
    print(f"Received event: {data}")

# Subscribe to an event type
bus.subscribe("my_event", on_event)

# Publish an event
bus.publish("my_event", {"message": "Hello, World!"})

# Unsubscribe when done
bus.unsubscribe("my_event", on_event)
```

### Multiple Subscribers

```python
from extensions.agent_bus.bus import bus

def handler1(data):
    print(f"Handler 1: {data}")

def handler2(data):
    print(f"Handler 2: {data}")

# Both handlers will receive the event
bus.subscribe("task_completed", handler1)
bus.subscribe("task_completed", handler2)

bus.publish("task_completed", {"task": "done"})
# Output:
# Handler 1: {'task': 'done'}
# Handler 2: {'task': 'done'}
```

### Event Types

You can use any string as an event type. Common patterns:

```python
# Task events
bus.publish("task_started", {"task_id": 123})
bus.publish("task_completed", {"task_id": 123, "success": True})
bus.publish("task_failed", {"task_id": 123, "error": "..."})

# Model events
bus.publish("model_loaded", {"model_name": "llama-7b"})
bus.publish("model_unloaded", {})

# Chat events
bus.publish("chat_message", {"user": "John", "text": "Hello"})
bus.publish("chat_response", {"bot": "Assistant", "text": "Hi!"})
```

### Extension Integration

Extensions can subscribe in their `setup()` function:

```python
# In your extension's script.py
from extensions.agent_bus.bus import bus

def on_task_completed(data):
    print(f"Task completed: {data}")

def setup():
    """Called once when extension is loaded."""
    bus.subscribe("task_completed", on_task_completed)
```

## API Reference

### EventBus Class

#### `subscribe(event_type: str, callback: Callable)`
Subscribe a callback function to an event type.

**Parameters:**
- `event_type`: String identifying the event type
- `callback`: Function to call when event is published

**Example:**
```python
def my_handler(data):
    print(data)
    
bus.subscribe("my_event", my_handler)
```

#### `unsubscribe(event_type: str, callback: Callable)`
Remove a callback from an event type.

**Parameters:**
- `event_type`: String identifying the event type
- `callback`: The callback function to remove

**Example:**
```python
bus.unsubscribe("my_event", my_handler)
```

#### `publish(event_type: str, data: Any = None)`
Publish an event to all subscribers.

**Parameters:**
- `event_type`: String identifying the event type
- `data`: Optional data to pass to callbacks (can be any type)

**Example:**
```python
bus.publish("my_event", {"key": "value"})
bus.publish("simple_event")  # No data
```

## Thread Safety

All EventBus operations are thread-safe:
- Uses `threading.RLock` for reentrant locking
- Safe to call from multiple threads simultaneously
- Callbacks are executed synchronously but safely

## Error Handling

Errors in callbacks are caught and logged:
```python
def buggy_callback(data):
    raise ValueError("Oops!")

bus.subscribe("test", buggy_callback)
bus.publish("test", {})
# Output: [EventBus] Error in callback for test: Oops!
# Other callbacks still execute normally
```

## Examples

### Example 1: Task Monitoring

```python
from extensions.agent_bus.bus import bus

class TaskMonitor:
    def __init__(self):
        self.tasks = {}
        bus.subscribe("task_started", self.on_start)
        bus.subscribe("task_completed", self.on_complete)
    
    def on_start(self, data):
        task_id = data["task_id"]
        self.tasks[task_id] = "running"
    
    def on_complete(self, data):
        task_id = data["task_id"]
        self.tasks[task_id] = "completed"

monitor = TaskMonitor()

# Somewhere else in your code
bus.publish("task_started", {"task_id": 1})
bus.publish("task_completed", {"task_id": 1})
```

### Example 2: Logging System

```python
from extensions.agent_bus.bus import bus
import json

def log_all_events(data):
    """Universal event logger."""
    with open("events.log", "a") as f:
        f.write(json.dumps(data) + "\n")

# Log all types of events
for event_type in ["task_started", "task_completed", "model_loaded"]:
    bus.subscribe(event_type, log_all_events)
```

### Example 3: Metrics Aggregation

```python
from extensions.agent_bus.bus import bus
from collections import defaultdict

class MetricsAggregator:
    def __init__(self):
        self.counters = defaultdict(int)
        bus.subscribe("metric", self.on_metric)
    
    def on_metric(self, data):
        metric_name = data.get("name")
        metric_value = data.get("value", 1)
        self.counters[metric_name] += metric_value
    
    def report(self):
        for name, value in self.counters.items():
            print(f"{name}: {value}")

metrics = MetricsAggregator()

# Publish metrics from anywhere
bus.publish("metric", {"name": "requests", "value": 1})
bus.publish("metric", {"name": "errors", "value": 1})
bus.publish("metric", {"name": "requests", "value": 1})

metrics.report()
# Output:
# requests: 2
# errors: 1
```

## Design Philosophy

The Agent Bus is intentionally simple:
- No complex routing or filtering
- No async/await complexity
- No external dependencies
- Minimal overhead

For more complex needs, consider using a full message broker or event system.

## Integration with agent_learning

The `agent_learning` extension uses the agent bus to listen for task completion events:

```python
# In agent_learning extension
from extensions.agent_bus.bus import bus

def on_task_completed(data):
    success = data.get("success", False)
    reaction_time = data.get("reaction_time", 0.0)
    # Update learning metrics...

bus.subscribe("task_completed", on_task_completed)
```

## Best Practices

1. **Use descriptive event names**: `task_completed` instead of `event1`
2. **Document expected data format**: What keys are in the data dict?
3. **Keep callbacks fast**: Don't block the event bus
4. **Handle missing data**: Use `.get()` with defaults
5. **Unsubscribe when done**: Prevent memory leaks

## Troubleshooting

### Events not received
- Verify the event type string matches exactly (case-sensitive)
- Check that subscription happens before publication
- Ensure callback function is defined correctly

### Callback errors
- Check console for error messages
- Use try-except in your callbacks for better control
- Test callbacks independently before subscribing

## Extension Structure

```
extensions/agent_bus/
├── __init__.py    # Module initialization
├── bus.py         # EventBus implementation
├── script.py      # Extension lifecycle
└── README.md      # This file
```

## Contributing

This is a core infrastructure extension. Changes should be minimal and backward-compatible.

## License

Same as text-generation-webui project.
