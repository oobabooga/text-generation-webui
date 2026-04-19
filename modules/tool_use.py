import asyncio
import importlib.util
import json

from modules import shared
from modules.logging_colors import logger
from modules.utils import natural_keys, sanitize_filename

_MCP_JSON_PATH = shared.user_data_dir / 'mcp.json'


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
    """Parse MCP servers textbox: one HTTP server per line, format 'url' or 'url,Header: value,Header2: value2'."""
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
        servers.append({"type": "http", "url": url, "headers": headers})
    return servers


def has_mcp_config():
    """Check if user_data/mcp.json exists."""
    return _MCP_JSON_PATH.exists()


def _load_mcp_json():
    """Load stdio MCP servers from user_data/mcp.json (Claude Desktop / Cursor format).

    Expected format:
    {
        "mcpServers": {
            "server-name": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
                "env": {"KEY": "value"}
            }
        }
    }
    """
    if not _MCP_JSON_PATH.exists():
        return []

    try:
        with open(_MCP_JSON_PATH) as f:
            config = json.load(f)
    except Exception:
        logger.exception(f'Failed to parse {_MCP_JSON_PATH}')
        return []

    servers = []
    for name, entry in config.get('mcpServers', {}).items():
        command = entry.get('command')
        if not command:
            logger.warning(f'MCP server "{name}" in mcp.json is missing "command". Skipping.')
            continue

        servers.append({
            "type": "stdio",
            "command": command,
            "args": entry.get("args", []),
            "env": entry.get("env"),
        })

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


def _mcp_server_id(server):
    """Return a human-readable identifier for a server config."""
    if server["type"] == "http":
        return server["url"]
    elif server["type"] == "stdio":
        return f'{server["command"]} {" ".join(server["args"])}'
    else:
        raise ValueError(f"Unknown MCP server type: {server['type']}")


async def _mcp_session(server, callback):
    """Open an MCP session and pass it to the callback."""
    from mcp import ClientSession

    if server["type"] == "http":
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(server["url"], headers=server["headers"] or None) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await callback(session)
    elif server["type"] == "stdio":
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(command=server["command"], args=server["args"], env=server.get("env"))
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await callback(session)
    else:
        raise ValueError(f"Unknown MCP server type: {server['type']}")


def _make_mcp_executor(name, server):
    def executor(arguments):
        return asyncio.run(_call_mcp_tool(name, arguments, server))
    return executor


async def _connect_mcp_server(server):
    """Connect to one MCP server and return (tool_defs, executors)."""

    async def _discover(session):
        result = await session.list_tools()
        tool_defs = []
        executors = {}
        for tool in result.tools:
            tool_defs.append(_mcp_tool_to_openai(tool))
            executors[tool.name] = _make_mcp_executor(tool.name, server)
        return tool_defs, executors

    return await _mcp_session(server, _discover)


async def _call_mcp_tool(name, arguments, server):
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

    return await _mcp_session(server, _invoke)


_mcp_server_cache = {}


def load_mcp_tools(servers_str):
    """
    Discover tools from MCP servers (HTTP from UI textbox + stdio from mcp.json).
    Returns (tool_defs, executors) in the same format as load_tools.
    Tool discovery is cached per server so each server is only queried once.
    """
    servers = _parse_mcp_servers(servers_str) if servers_str else []
    servers += _load_mcp_json()
    if not servers:
        return [], {}

    uncached = [s for s in servers if _mcp_server_id(s) not in _mcp_server_cache]
    if uncached:
        async def _discover_uncached():
            return await asyncio.gather(
                *(_connect_mcp_server(s) for s in uncached),
                return_exceptions=True
            )

        results = asyncio.run(_discover_uncached())
        for server, result in zip(uncached, results):
            sid = _mcp_server_id(server)
            if isinstance(result, Exception):
                logger.exception(f'Failed to connect to MCP server "{sid}"', exc_info=result)
                _mcp_server_cache[sid] = ([], {})
            else:
                _mcp_server_cache[sid] = result

    all_defs = []
    all_executors = {}
    for server in servers:
        sid = _mcp_server_id(server)
        defs, execs = _mcp_server_cache[sid]
        for td, (fn, ex) in zip(defs, execs.items()):
            if fn in all_executors:
                logger.warning(f'MCP tool "{fn}" from {sid} conflicts with an already loaded tool. Skipping.')
                continue
            all_defs.append(td)
            all_executors[fn] = ex

    return all_defs, all_executors


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
