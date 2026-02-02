#!/usr/bin/env python3
"""AgentHub MCP Server - Expose AgentHub agents as tools for Claude Code.

This MCP server creates its own AgentHub instance and exposes agents
as tools for Claude Code through the Model Context Protocol.

Usage:
    Configure Claude Code to use this MCP server by adding to ~/.claude.json:

    {
      "mcpServers": {
        "agenthub": {
          "command": "python",
          "args": ["-m", "agenthub.mcp_server", "--project", "/path/to/your/project"],
          "env": {
            "PYTHONPATH": "/path/to/AgentHub/src"
          }
        }
      }
    }

    IMPORTANT: Always specify --project to ensure the MCP server analyzes
    the correct project. This should match the project you run 'agenthub up' on.

    Alternative: Set AGENTHUB_PROJECT environment variable instead of --project.

Available Tools:
    - agenthub_query: Ask a question routed to the best agent
    - agenthub_list_agents: See all available agents
    - agenthub_routing_rules: See how queries are routed
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# AgentHub will be initialized lazily
_hub = None
_project_root: Optional[str] = None
_env_loaded = False

# Dashboard URL for broadcasting events (optional)
DASHBOARD_URL = os.environ.get("AGENTHUB_DASHBOARD_URL", "http://localhost:3001")


def broadcast_to_dashboard(event_type: str, description: str, details: dict = None):
    """Send an event to the dashboard for display.

    This is fire-and-forget - errors are silently ignored since
    the dashboard may not be running.
    """
    try:
        data = {
            "event_type": event_type,
            "description": description,
            "details": details or {},
        }
        req = urllib.request.Request(
            f"{DASHBOARD_URL}/api/claude-code/log",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=1)
    except Exception:
        # Dashboard not running or unreachable - that's fine
        pass


def load_project_env(project_path: str) -> None:
    """Load .env from project directory and home directory.

    Priority (later overrides earlier):
    1. Home directory ~/.env
    2. Project directory .env
    """
    global _env_loaded
    if _env_loaded:
        return

    # Load from home directory first
    home_env = Path.home() / ".env"
    if home_env.exists():
        load_dotenv(home_env)
        print(f"Loaded .env from: {home_env}", file=sys.stderr)

    # Load from project directory - this overrides home
    project_env = Path(project_path) / ".env"
    if project_env.exists():
        load_dotenv(project_env, override=True)
        print(f"Loaded .env from: {project_env}", file=sys.stderr)

    _env_loaded = True


def get_hub():
    """Get or create the AgentHub instance."""
    global _hub, _project_root

    if _hub is not None:
        return _hub

    # Get project root: command arg > env var > saved config > error
    if _project_root is None:
        _project_root = os.environ.get("AGENTHUB_PROJECT")

    if not _project_root:
        # Try to load from saved config (from 'agenthub build')
        try:
            from agenthub.setup import get_current_project
            _project_root = get_current_project()
            if _project_root:
                print(f"Using saved project path: {_project_root}", file=sys.stderr)
        except ImportError:
            pass

    if not _project_root:
        print("ERROR: No project path specified.", file=sys.stderr)
        print("Options:", file=sys.stderr)
        print("  1. Run 'agenthub build /path/to/project' first", file=sys.stderr)
        print("  2. Use --project /path/to/project in MCP config", file=sys.stderr)
        print("  3. Set AGENTHUB_PROJECT environment variable", file=sys.stderr)
        return None

    # Resolve to absolute path
    _project_root = str(Path(_project_root).resolve())

    if not Path(_project_root).is_dir():
        print(f"ERROR: Project path does not exist: {_project_root}", file=sys.stderr)
        return None

    # Load .env from the actual project directory
    load_project_env(_project_root)

    try:
        from agenthub.auto import discover_all_agents
        _hub, summary = discover_all_agents(_project_root)
        # Log to stderr (won't interfere with MCP protocol on stdout)
        print(f"AgentHub initialized for: {_project_root}", file=sys.stderr)
        print(summary, file=sys.stderr)
        return _hub
    except Exception as e:
        print(f"Error initializing AgentHub: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


# Tool definitions for MCP
TOOLS = [
    {
        "name": "agenthub_query",
        "description": """Ask a question about the codebase using AgentHub's specialized agents.

The query will be automatically routed to the best agent based on keywords:
- API/endpoint questions -> API Expert
- Database/model questions -> Model Expert
- Business logic questions -> Service Expert
- Configuration questions -> Config Expert
- Test questions -> Test Expert

Use this for codebase-specific questions that benefit from specialized knowledge.""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask about the codebase"
                },
                "agent_id": {
                    "type": "string",
                    "description": "Optional: specific agent ID to use (bypasses auto-routing)"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "agenthub_list_agents",
        "description": "List all available AgentHub agents with their specializations and keywords.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "agenthub_routing_rules",
        "description": "Show the routing rules that determine which agent handles which queries.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


def handle_tool_call(name: str, arguments: dict) -> Any:
    """Handle a tool call and return the result."""

    hub = get_hub()
    if hub is None:
        return """Error: AgentHub not initialized.

Please ensure:
1. You specified --project /path/to/project in your MCP config
2. Or set AGENTHUB_PROJECT environment variable
3. ANTHROPIC_API_KEY is set (in .env or environment)

Example MCP config:
{
  "mcpServers": {
    "agenthub": {
      "command": "python",
      "args": ["-m", "agenthub.mcp_server", "--project", "/path/to/your/project"],
      "env": {
        "PYTHONPATH": "/path/to/AgentHub/src"
      }
    }
  }
}"""

    if name == "agenthub_query":
        question = arguments.get("question", "")
        agent_id = arguments.get("agent_id")

        # Broadcast query to dashboard
        broadcast_to_dashboard(
            "task",
            f"Claude Code query: {question[:100]}{'...' if len(question) > 100 else ''}",
            {"query": question, "requested_agent": agent_id},
        )

        try:
            response = hub.run(question, agent_id=agent_id)

            # Broadcast result to dashboard (full content)
            broadcast_to_dashboard(
                "result",
                f"Agent {response.agent_id} responded ({response.tokens_used} tokens)",
                {
                    "agent_id": response.agent_id,
                    "tokens_used": response.tokens_used,
                    "query": question,
                    "response": response.content,
                },
            )

            return f"""Agent: {response.agent_id}
Tokens used: {response.tokens_used}

Response:
{response.content}"""
        except Exception as e:
            broadcast_to_dashboard("error", f"Query failed: {str(e)}")
            return f"Error querying agent: {e}"

    elif name == "agenthub_list_agents":
        agents = hub.list_agents()
        if not agents:
            return "No agents available."

        lines = ["Available AgentHub Agents:", ""]
        for spec in agents:
            tier = "A" if not spec.metadata.get("auto_generated") else "B"
            lines.append(f"* {spec.name} [{spec.agent_id}] (Tier {tier})")
            lines.append(f"  {spec.description}")
            keywords = spec.context_keywords[:5]
            if keywords:
                lines.append(f"  Keywords: {', '.join(keywords)}")
            lines.append("")

        return "\n".join(lines)

    elif name == "agenthub_routing_rules":
        agents = hub.list_agents()

        lines = [
            "AgentHub Routing Rules",
            "=" * 40,
            "",
            "Queries are matched against agent keywords.",
            "Tier A agents are checked first, then Tier B.",
            "First match wins.",
            ""
        ]

        # Sort by tier
        tier_a = [s for s in agents if not s.metadata.get("auto_generated")]
        tier_b = [s for s in agents if s.metadata.get("auto_generated")]

        for spec in tier_a + tier_b:
            tier = "A" if not spec.metadata.get("auto_generated") else "B"
            lines.append(f"-> {spec.name} (Tier {tier})")
            lines.append(f"   {spec.description}")
            keywords = spec.context_keywords[:10]
            if keywords:
                lines.append(f"   Triggers: {', '.join(keywords)}")
            lines.append("")

        return "\n".join(lines)

    else:
        return f"Unknown tool: {name}"


def send_response(response: dict):
    """Send a JSON-RPC response to stdout."""
    print(json.dumps(response), flush=True)


def main():
    """Main MCP server loop - reads JSON-RPC from stdin, writes to stdout."""
    global _project_root

    # Parse command line args for project path
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--project" and i + 1 < len(args):
            _project_root = args[i + 1]
            break

    # Log startup info
    print(f"AgentHub MCP Server starting...", file=sys.stderr)
    if _project_root:
        print(f"Project: {_project_root}", file=sys.stderr)
    else:
        env_project = os.environ.get("AGENTHUB_PROJECT")
        if env_project:
            print(f"Project (from env): {env_project}", file=sys.stderr)
        else:
            print("WARNING: No project specified. Use --project or AGENTHUB_PROJECT", file=sys.stderr)

    # Read requests from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            send_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            })
            continue

        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        # Handle MCP protocol methods
        if method == "initialize":
            send_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "agenthub",
                        "version": "1.0.0"
                    }
                }
            })

        elif method == "notifications/initialized":
            # Client acknowledged initialization, no response needed
            pass

        elif method == "tools/list":
            send_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": TOOLS
                }
            })

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})

            result = handle_tool_call(tool_name, tool_args)

            send_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                }
            })

        elif method == "ping":
            send_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {}
            })

        else:
            send_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            })


if __name__ == "__main__":
    main()
