#!/usr/bin/env python3
"""AgentHub MCP Server — Impact analysis tools for large codebases.

Exposes dependency graph analysis as MCP tools for Claude Code.
No LLM calls, no API key needed, zero token cost.

Usage:
    Configure in ~/.claude.json:

    {
      "mcpServers": {
        "agenthub": {
          "command": "python",
          "args": ["-m", "agenthub.mcp_server", "--project", "/path/to/project"],
          "env": {
            "PYTHONPATH": "/path/to/AgentHub/src"
          }
        }
      }
    }

Tools:
    - impact_check: Check blast radius before editing a file
    - affected_tests: Find tests to run after editing files
    - codebase_overview: Get project structure overview

Resources:
    - repo://map: Auto-injected repo map with key modules, dependency chains,
      and editing guidance. Loaded into every agent's context automatically.
"""
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional

_graph = None
_repo_map_cache: Optional[str] = None
_project_root: Optional[str] = None


def _resolve_project_root() -> Optional[str]:
    """Resolve project root from --project flag or env var."""
    if _project_root:
        return str(Path(_project_root).resolve())

    env_project = os.environ.get("AGENTHUB_PROJECT")
    if env_project:
        return str(Path(env_project).resolve())

    return None


def get_graph():
    """Get or build the import graph (lazy, cached)."""
    global _graph

    if _graph is not None:
        return _graph

    root = _resolve_project_root()
    if not root:
        print("ERROR: No project path. Use --project or AGENTHUB_PROJECT env.", file=sys.stderr)
        return None

    if not Path(root).is_dir():
        print(f"ERROR: Project path does not exist: {root}", file=sys.stderr)
        return None

    try:
        from agenthub.auto.import_graph import ImportGraph

        _graph = ImportGraph(root)
        _graph.build()
        stats = _graph.get_stats()
        print(f"Import graph built: {stats['total_modules']} modules, {stats['total_edges']} edges", file=sys.stderr)
        return _graph
    except Exception as e:
        print(f"Error building import graph: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


def get_repo_map() -> Optional[str]:
    """Get the repo map (cached after first generation)."""
    global _repo_map_cache

    if _repo_map_cache is not None:
        return _repo_map_cache

    graph = get_graph()
    if graph is None:
        return None

    try:
        from agenthub.repo_map import generate_repo_map
        _repo_map_cache = generate_repo_map(graph)
        return _repo_map_cache
    except Exception as e:
        print(f"Error generating repo map: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------

RESOURCES = [
    {
        "uri": "repo://map",
        "name": "Repo Map",
        "description": (
            "Auto-generated map of the codebase showing key modules, dependency chains, "
            "directory layout, and editing guidance. Use this to understand the project "
            "structure before making changes."
        ),
        "mimeType": "text/markdown",
    },
]


def _normalize_path(file_path: str) -> str:
    """Normalize a file path to be relative to project root."""
    root = _resolve_project_root()
    if not root:
        return file_path

    p = Path(file_path)
    # If absolute, make relative
    if p.is_absolute():
        try:
            return str(p.relative_to(root))
        except ValueError:
            return file_path

    # Strip leading ./
    return str(p).lstrip("./")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "impact_check",
        "description": (
            "Check the blast radius of editing a file. Returns what the file exports, "
            "what depends on it (directly and transitively), affected tests, and risk level. "
            "Call this BEFORE editing a file to understand what could break."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file from project root (e.g. 'src/utils.py')",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "affected_tests",
        "description": (
            "Find test files affected by changes to the given files. Returns test file paths "
            "and a suggested test command. Call this AFTER editing files to know what to run."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of relative file paths that were changed",
                },
            },
            "required": ["file_paths"],
        },
    },
    {
        "name": "codebase_overview",
        "description": (
            "Get a quick overview of the codebase: module count, language breakdown, "
            "central hub modules, and dependency stats. Use for orientation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_impact_check(arguments: dict) -> str:
    graph = get_graph()
    if graph is None:
        return "Error: Import graph not available. Check --project path."

    raw_path = arguments.get("file_path", "")
    path = _normalize_path(raw_path)

    if path not in graph.nodes:
        # Try common variations
        candidates = [p for p in graph.nodes if p.endswith(path) or path.endswith(p)]
        if len(candidates) == 1:
            path = candidates[0]
        elif candidates:
            return f"Ambiguous path '{raw_path}'. Did you mean one of:\n" + "\n".join(f"  - {c}" for c in candidates[:10])
        else:
            return f"File '{raw_path}' not found in import graph. Use codebase_overview to see indexed modules."

    # Gather data
    interface = graph.get_exported_interface(path)
    neighbors = graph.get_module_neighbors(path)
    transitive = graph.get_transitive_importers(path)
    tests = graph.get_affected_tests([path])
    role = graph.get_module_role(path)

    # Risk assessment
    n_transitive = len(transitive)
    if n_transitive > 10 or role == "hub":
        risk = "HIGH"
        risk_note = f"This is a {role} module with {n_transitive} transitive dependents. Changes here ripple widely."
    elif n_transitive > 3:
        risk = "MEDIUM"
        risk_note = f"{n_transitive} files transitively depend on this."
    else:
        risk = "LOW"
        risk_note = f"Only {n_transitive} transitive dependent(s)."

    # Format output
    lines = [
        f"Impact Analysis: {path}",
        f"Role: {role} | Risk: {risk}",
        f"{risk_note}",
        "",
    ]

    # Exported interface
    lines.append("EXPORTED INTERFACE:")
    if interface["classes"]:
        for cls in interface["classes"]:
            bases = f"({', '.join(cls['bases'])})" if cls.get("bases") else ""
            lines.append(f"  class {cls['name']}{bases}")
            for method in cls.get("methods", []):
                lines.append(f"    .{method}()")
    if interface["functions"]:
        for func in interface["functions"]:
            args = ", ".join(func.get("args", []))
            async_prefix = "async " if func.get("is_async") else ""
            lines.append(f"  {async_prefix}def {func['name']}({args})")
    if interface["constants"]:
        for const in interface["constants"]:
            lines.append(f"  {const['name']}")
    if not interface["classes"] and not interface["functions"] and not interface["constants"]:
        lines.append("  (no public exports detected)")
    lines.append("")

    # Direct dependents
    direct = neighbors.get("imported_by", [])
    lines.append(f"DIRECT DEPENDENTS ({len(direct)}):")
    for dep in direct[:20]:
        lines.append(f"  {dep}")
    if len(direct) > 20:
        lines.append(f"  ... and {len(direct) - 20} more")
    lines.append("")

    # Transitive dependents (exclude direct to avoid duplication)
    indirect = [t for t in transitive if t not in direct]
    if indirect:
        lines.append(f"TRANSITIVE DEPENDENTS ({len(indirect)} additional):")
        for dep in indirect[:15]:
            lines.append(f"  {dep}")
        if len(indirect) > 15:
            lines.append(f"  ... and {len(indirect) - 15} more")
        lines.append("")

    # Affected tests
    lines.append(f"AFFECTED TESTS ({len(tests)}):")
    if tests:
        for t in tests:
            lines.append(f"  {t}")
    else:
        lines.append("  (no test files found in dependency chain)")

    return "\n".join(lines)


def handle_affected_tests(arguments: dict) -> str:
    graph = get_graph()
    if graph is None:
        return "Error: Import graph not available. Check --project path."

    raw_paths = arguments.get("file_paths", [])
    paths = [_normalize_path(p) for p in raw_paths]

    tests = graph.get_affected_tests(paths)

    if not tests:
        return f"No test files found in the dependency chain of: {', '.join(paths)}"

    # Detect language for command suggestion
    py_tests = [t for t in tests if t.endswith(".py")]
    ts_js_tests = [t for t in tests if any(t.endswith(ext) for ext in (".ts", ".tsx", ".js", ".jsx"))]

    lines = [f"Affected tests ({len(tests)}):", ""]
    for t in tests:
        lines.append(f"  {t}")

    lines.append("")
    lines.append("SUGGESTED COMMANDS:")
    if py_tests:
        lines.append(f"  pytest {' '.join(py_tests)} -x")
    if ts_js_tests:
        # Convert file paths to patterns for jest
        patterns = [Path(t).stem.replace(".test", "").replace(".spec", "") for t in ts_js_tests]
        lines.append(f"  npx jest {' '.join(patterns)}")

    return "\n".join(lines)


def handle_codebase_overview(arguments: dict) -> str:
    graph = get_graph()
    if graph is None:
        return "Error: Import graph not available. Check --project path."

    stats = graph.get_stats()
    hubs = graph.get_central_modules(top_n=10)

    lines = [
        "Codebase Overview",
        "=" * 40,
        "",
        f"Modules: {stats['total_modules']}",
        f"Dependencies: {stats['total_edges']}",
        f"Clusters: {stats['num_clusters']}",
        f"Size: {stats['total_size_kb']:.0f} KB",
        "",
        "Languages:",
    ]

    if stats.get("python_modules"):
        lines.append(f"  Python: {stats['python_modules']} modules")
    if stats.get("typescript_modules"):
        lines.append(f"  TypeScript: {stats['typescript_modules']} modules")
    if stats.get("javascript_modules"):
        lines.append(f"  JavaScript: {stats['javascript_modules']} modules")

    lines.extend(["", "Structure:"])
    lines.append(f"  Hub modules (high fan-in): {stats.get('hub_modules', 0)}")
    lines.append(f"  Leaf modules (no dependents): {stats.get('leaf_modules', 0)}")
    lines.append(f"  Isolated modules: {stats.get('isolated_modules', 0)}")

    if hubs:
        lines.extend(["", "CENTRAL MODULES (most depended on):"])
        for hub_path in hubs:
            node = graph.nodes.get(hub_path)
            n_importers = len(node.imported_by) if node else 0
            role = graph.get_module_role(hub_path)
            lines.append(f"  {hub_path} ({role}, {n_importers} importers)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def handle_tool_call(name: str, arguments: dict) -> str:
    handlers = {
        "impact_check": handle_impact_check,
        "affected_tests": handle_affected_tests,
        "codebase_overview": handle_codebase_overview,
    }
    handler = handlers.get(name)
    if handler:
        return handler(arguments)
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# JSON-RPC server
# ---------------------------------------------------------------------------

def send_response(response: dict):
    """Send a JSON-RPC response to stdout."""
    print(json.dumps(response), flush=True)


def main():
    """MCP server loop — JSON-RPC over stdin/stdout."""
    global _project_root

    print("AgentHub Impact Server starting...", file=sys.stderr)

    # Parse --project arg
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--project" and i + 1 < len(args):
            _project_root = args[i + 1]
            break

    if _project_root:
        print(f"  Project: {_project_root}", file=sys.stderr)
    elif os.environ.get("AGENTHUB_PROJECT"):
        print(f"  Project (env): {os.environ['AGENTHUB_PROJECT']}", file=sys.stderr)
    else:
        print("  WARNING: No --project arg or AGENTHUB_PROJECT env.", file=sys.stderr)

    def _shutdown(signum, frame):
        print(f"Shutting down (signal {signum}).", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            send_response({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
            continue

        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "initialize":
            send_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                    },
                    "serverInfo": {"name": "agenthub-impact", "version": "2.0.0"},
                },
            })

        elif method == "notifications/initialized":
            pass

        elif method == "tools/list":
            send_response({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}})

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            result = handle_tool_call(tool_name, tool_args)
            send_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": str(result)}]},
            })

        elif method == "resources/list":
            send_response({"jsonrpc": "2.0", "id": request_id, "result": {"resources": RESOURCES}})

        elif method == "resources/read":
            uri = params.get("uri", "")
            if uri == "repo://map":
                repo_map = get_repo_map()
                if repo_map:
                    send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "contents": [{"uri": uri, "mimeType": "text/markdown", "text": repo_map}],
                        },
                    })
                else:
                    send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32603, "message": "Failed to generate repo map"},
                    })
            else:
                send_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32602, "message": f"Unknown resource: {uri}"},
                })

        elif method == "ping":
            send_response({"jsonrpc": "2.0", "id": request_id, "result": {}})

        else:
            send_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


if __name__ == "__main__":
    main()
