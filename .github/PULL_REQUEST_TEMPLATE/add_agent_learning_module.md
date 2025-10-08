## Add Agent Learning Module

**Description**

This PR adds the agent_learning extension module to the text-generation-webui, providing metrics tracking, persistence, and API endpoint functionality for local agent learning.

**Related Issue**

Closes #9

**Changes Made**

### New Extensions
- [ ] Created `extensions/agent_bus/` - Event bus for agent communication
  - `__init__.py` - Module initialization
  - `bus.py` - EventBus implementation
  - `script.py` - Extension setup
  
- [ ] Created `extensions/agent_learning/` - Agent learning metrics module
  - `__init__.py` - Module initialization
  - `script.py` - Extension setup and lifecycle
  - `learning_metrics.py` - Metrics tracking and calculation
  - `learning_api.py` - Flask Blueprint with `/learning/get_learning_progress` endpoint
  - `data/progress.json` - Initial metrics data structure

### GitHub Templates
- [ ] Created `.github/ISSUE_TEMPLATE/agent_learning_task.md` - Issue template for agent learning tasks
- [ ] Created `.github/PULL_REQUEST_TEMPLATE/add_agent_learning_module.md` - This PR template

**Features Implemented**

1. **Metrics Tracking**
   - Success rate tracking
   - Reaction time measurement
   - Logic, memory, and creativity weighted metrics
   - Overall progress calculation (0-100%)

2. **Data Persistence**
   - JSON file storage with atomic writes
   - Bounded history (max 500 entries per metric)
   - Automatic trimming to prevent unlimited growth

3. **Event Bus Integration**
   - Subscribe to `task_completed` events
   - Automatic metric updates on task completion

4. **API Endpoint**
   - Flask Blueprint: `/learning/get_learning_progress`
   - Returns current metrics and progress as JSON
   - Error handling with appropriate HTTP status codes

**Testing**

- [ ] Extension loads without errors
- [ ] Metrics can be updated via event bus
- [ ] Data persists correctly to progress.json
- [ ] API endpoint returns valid JSON
- [ ] Array trimming works correctly at MAX_HISTORY limit
- [ ] Thread safety is maintained with locks

**Checklist**

- [ ] I have read the [Contributing guidelines](https://github.com/oobabooga/text-generation-webui/wiki/Contributing-guidelines)
- [ ] Code follows the project's coding style
- [ ] All new files have appropriate headers/documentation
- [ ] No breaking changes to existing functionality
- [ ] Extension can be enabled/disabled independently

**Additional Notes**

The agent_learning module is designed to be modular and non-intrusive. It can be safely enabled or disabled without affecting the core functionality of the text-generation-webui.

To use this module:
1. Enable the `agent_learning` extension
2. Publish `task_completed` events to the agent_bus with format:
   ```python
   {
       "success": bool,
       "reaction_time": float,
       "weights": {"logic": float, "memory": float, "creativity": float}
   }
   ```
3. Access metrics via GET `/learning/get_learning_progress`
