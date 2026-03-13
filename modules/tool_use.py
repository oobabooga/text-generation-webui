import importlib.util
import json
import random

from modules import shared
from modules.logging_colors import logger
from modules.utils import natural_keys


def get_available_tools():
    """Return sorted list of tool script names from user_data/tools/*.py."""
    tools_dir = shared.user_data_dir / 'tools'
    tools_dir.mkdir(parents=True, exist_ok=True)
    return sorted((p.stem for p in tools_dir.glob('*.py')), key=natural_keys)


def load_tools(selected_names):
    """
    Import selected tool scripts and return their definitions and executors.
    Returns (tool_defs, executors) where:
      - tool_defs: list of OpenAI-format tool dicts
      - executors: dict mapping function_name -> execute callable
    """
    tool_defs = []
    executors = {}
    for name in selected_names:
        path = shared.user_data_dir / 'tools' / f'{name}.py'
        if not path.exists():
            continue

        try:
            spec = importlib.util.spec_from_file_location(f"tool_{name}", str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            logger.exception(f'Failed to load tool script "{name}"')
            continue

        tool_def = getattr(module, 'tool', None)
        execute_fn = getattr(module, 'execute', None)
        if tool_def is None or execute_fn is None:
            logger.warning(f'Tool "{name}" is missing a "tool" dict or "execute" function.')
            continue

        func_name = tool_def.get('function', {}).get('name', name)
        tool_defs.append(tool_def)
        executors[func_name] = execute_fn

    return tool_defs, executors


def generate_tool_call_id():
    """Generate a unique tool call ID (e.g. 'call_a1b2c3d4')."""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "call_" + "".join(random.choice(chars) for _ in range(8))


def execute_tool(func_name, arguments, executors):
    """Execute a tool by function name. Returns result as a JSON string."""
    fn = executors.get(func_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {func_name}"})

    try:
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        result = fn(arguments)
        return json.dumps(result) if not isinstance(result, str) else result
    except Exception as e:
        logger.exception(f'Tool "{func_name}" execution failed')
        return json.dumps({"error": str(e)})
