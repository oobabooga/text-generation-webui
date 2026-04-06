import asyncio
import importlib.util
import json

from modules import shared
from modules.logging_colors import logger
from modules.utils import natural_keys, sanitize_filename


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
        name = sanitize_filename(name)
        if not name:
            continue

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
        if func_name in executors:
            logger.warning(f'Tool "{name}" declares function name "{func_name}" which conflicts with an already loaded tool. Skipping.')
            continue
        tool_defs.append(tool_def)
        executors[func_name] = execute_fn

    return tool_defs, executors


def _parse_mcp_servers(servers_str):
    """Parse MCP servers textbox: one server per line, format 'url' or 'url,Header: value,Header2: value2'."""
    servers = []
    for line in servers_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        url = parts[0].strip()
        headers = {}
        for part in parts[1:]:
            part = part.strip()
            if ':' in part:
                key, val = part.split(':', 1)
                headers[key.strip()] = val.strip()
        servers.append((url, headers))
    return servers


def _mcp_tool_to_openai(tool):
    """Convert an MCP Tool object to OpenAI-format tool dict."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}}
        }
    }


async def _mcp_session(url, headers, callback):
    """Open an MCP session and pass it to the callback."""
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    async with streamablehttp_client(url, headers=headers or None) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await callback(session)


def _make_mcp_executor(name, url, headers):
    def executor(arguments):
        return asyncio.run(_call_mcp_tool(name, arguments, url, headers))
    return executor


async def _connect_mcp_server(url, headers):
    """Connect to one MCP server and return (tool_defs, executors)."""

    async def _discover(session):
        result = await session.list_tools()
        tool_defs = []
        executors = {}
        for tool in result.tools:
            tool_defs.append(_mcp_tool_to_openai(tool))
            executors[tool.name] = _make_mcp_executor(tool.name, url, headers)
        return tool_defs, executors

    return await _mcp_session(url, headers, _discover)


async def _call_mcp_tool(name, arguments, url, headers):
    """Connect to an MCP server and call a single tool."""

    async def _invoke(session):
        result = await session.call_tool(name, arguments)
        parts = []
        for content in result.content:
            if hasattr(content, 'text'):
                parts.append(content.text)
            else:
                parts.append(str(content))
        return '\n'.join(parts) if parts else ''

    return await _mcp_session(url, headers, _invoke)


async def _connect_all_mcp_servers(servers):
    """Connect to all MCP servers concurrently."""
    results = await asyncio.gather(
        *(_connect_mcp_server(url, headers) for url, headers in servers),
        return_exceptions=True
    )
    all_defs = []
    all_executors = {}
    for (url, _), result in zip(servers, results):
        if isinstance(result, Exception):
            logger.exception(f'Failed to connect to MCP server "{url}"', exc_info=result)
            continue
        defs, execs = result
        for td, (fn, ex) in zip(defs, execs.items()):
            if fn in all_executors:
                logger.warning(f'MCP tool "{fn}" from {url} conflicts with an already loaded tool. Skipping.')
                continue
            all_defs.append(td)
            all_executors[fn] = ex
    return all_defs, all_executors


def load_mcp_tools(servers_str):
    """
    Parse MCP servers string and discover tools from each server.
    Returns (tool_defs, executors) in the same format as load_tools.
    """
    servers = _parse_mcp_servers(servers_str)
    if not servers:
        return [], {}

    return asyncio.run(_connect_all_mcp_servers(servers))


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
