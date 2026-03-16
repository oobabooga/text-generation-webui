## Supported models

The following models are supported:

- Qwen 3.5
- GPT-OSS
- Mistral Small / Devstral
- DeepSeek V3
- Kimi-K2
- MiniMax-M2.5
- GLM-5
- Llama 4

Other models that output tool calls as JSON (inside XML tags, code blocks, or plain JSON) are also supported through a generic fallback parser.

## Tool calling in the UI

### 1. Load a model with tool-calling support

Load a model with tool-calling support from the Model tab.

### 2. Select tools

In the chat sidebar, check the tools you want the model to use:

- **web_search** -- Search the web using DuckDuckGo.
- **fetch_webpage** -- Fetch the content of a URL.
- **calculate** -- Evaluate math expressions.
- **get_datetime** -- Get the current date and time.
- **roll_dice** -- Roll dice.

### 3. Chat

Send a message as usual. When the model decides it needs a tool, it will call it automatically. You will see each tool call and its result in a collapsible accordion inside the chat message.

The model may call multiple tools in sequence before giving its final answer.

## Writing custom tools

Each tool is a single `.py` file in `user_data/tools/`. It needs two things:

1. A `tool` dictionary that describes the function (name, description, parameters).
2. An `execute(arguments)` function that runs it and returns the result.

Here is a minimal example (`user_data/tools/get_datetime.py`):

```python
from datetime import datetime

tool = {
    "type": "function",
    "function": {
        "name": "get_datetime",
        "description": "Get the current date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    }
}


def execute(arguments):
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%I:%M %p")}
```

An example with parameters (`user_data/tools/roll_dice.py`):

```python
import random

tool = {
    "type": "function",
    "function": {
        "name": "roll_dice",
        "description": "Roll one or more dice with the specified number of sides.",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Number of dice to roll.", "default": 1},
                "sides": {"type": "integer", "description": "Number of sides per die.", "default": 20},
            },
        }
    }
}


def execute(arguments):
    count = max(1, min(arguments.get("count", 1), 1000))
    sides = max(2, min(arguments.get("sides", 20), 1000))
    rolls = [random.randint(1, sides) for _ in range(count)]
    return {"rolls": rolls, "total": sum(rolls)}
```

You can open the built-in tools in `user_data/tools/` for more examples.

## Tool calling over the API

Tool calling over the API follows the [OpenAI API](https://platform.openai.com/docs/guides/function-calling) convention. Define your tools, send them with your messages, and handle tool calls in a loop until the model gives a final answer.

```python
import json
import requests

url = "http://127.0.0.1:5000/v1/chat/completions"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"]
            }
        }
    }
]


def execute_tool(name, arguments):
    if name == "get_weather":
        return {"temperature": "14°C", "condition": "partly cloudy"}
    return {"error": f"Unknown tool: {name}"}


messages = [{"role": "user", "content": "What's the weather like in Paris?"}]

for _ in range(10):
    response = requests.post(url, json={"messages": messages, "tools": tools}).json()
    choice = response["choices"][0]

    if choice["finish_reason"] == "tool_calls":
        messages.append({
            "role": "assistant",
            "content": choice["message"]["content"],
            "tool_calls": choice["message"]["tool_calls"],
        })

        for tool_call in choice["message"]["tool_calls"]:
            name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            result = execute_tool(name, arguments)
            print(f"Tool call: {name}({arguments}) => {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result),
            })
    else:
        print(f"\nAssistant: {choice['message']['content']}")
        break
```
