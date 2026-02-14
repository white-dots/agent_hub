from __future__ import annotations
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

    Automatically includes the current repo name from _project_root
    so the dashboard can show which project is being worked on.
    """
    try:
        enriched_details = details.copy() if details else {}

        # Include repo context from the current project root
        if _project_root and "repo" not in enriched_details:
            enriched_details["repo"] = Path(_project_root).name

        data = {
            "event_type": event_type,
            "description": description,
            "details": enriched_details,
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


def _resolve_project_root() -> Optional[str]:
    """Resolve the project root from all available sources.

    Priority order (highest to lowest):
        1. --project CLI argument (set in main() before this is called)
        2. AGENTHUB_PROJECT environment variable
        3. Saved config from ~/.agenthub/config.json (set by 'agenthub build')

    Returns:
        Resolved absolute project path, or None if not found.
    """
    global _project_root

    # Source 1: CLI argument (already set by main())
    if _project_root:
        print(f"Project root (from --project): {_project_root}", file=sys.stderr)
        return str(Path(_project_root).resolve())

    # Source 2: Environment variable
    env_project = os.environ.get("AGENTHUB_PROJECT")
    if env_project:
        print(f"Project root (from AGENTHUB_PROJECT env): {env_project}", file=sys.stderr)
        return str(Path(env_project).resolve())

    # Source 3: Saved config (from 'agenthub build')
    try:
        from agenthub.setup import get_current_project
        saved_project = get_current_project()
        if saved_project:
            print(f"Project root (from ~/.agenthub/config.json): {saved_project}", file=sys.stderr)
            return str(Path(saved_project).resolve())
    except ImportError:
        pass

    return None


def get_hub(force_refresh: bool = False):
    """Get or create the AgentHub instance.

    Args:
        force_refresh: If True, recreate the hub even if it exists.
    """
    global _hub, _project_root, _env_loaded

    if _hub is not None and not force_refresh:
        return _hub

    # Clear existing hub if refreshing
    if force_refresh:
        print("Refreshing AgentHub...", file=sys.stderr)
        _hub = None
        _env_loaded = False  # Allow .env to be reloaded

    # Resolve project root from all sources
    resolved_root = _resolve_project_root()

    if not resolved_root:
        print("ERROR: No project path specified.", file=sys.stderr)
        print("Options:", file=sys.stderr)
        print("  1. Use --project /path/to/project in MCP config", file=sys.stderr)
        print("  2. Set AGENTHUB_PROJECT environment variable", file=sys.stderr)
        print("  3. Run 'agenthub build /path/to/project' first", file=sys.stderr)
        return None

    _project_root = resolved_root

    if not Path(_project_root).is_dir():
        print(f"ERROR: Project path does not exist: {_project_root}", file=sys.stderr)
        return None

    # Load .env from the actual project directory
    load_project_env(_project_root)

    try:
        from agenthub.auto import discover_all_agents
        _hub, summary = discover_all_agents(_project_root)

        # Enable team execution if import graph is available
        if _hub._import_graph is not None:
            try:
                _hub.enable_teams()
                print("DAG team execution enabled", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Could not enable teams: {e}", file=sys.stderr)

        # Enable parallel sessions if project is a git repo
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=_project_root,
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                _hub.enable_parallel_sessions(_project_root)
                print("Parallel sessions enabled", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not enable parallel sessions: {e}", file=sys.stderr)

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

For complex cross-cutting queries (e.g., "How does data flow from API to database?"),
AgentHub can use team execution with multiple agents working together.

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
                },
                "team_mode": {
                    "type": "string",
                    "enum": ["auto", "always", "never"],
                    "description": "Team execution mode: 'auto' (let AgentHub decide), 'always' (force team), 'never' (force single agent). Default: 'auto'. Prefer 'auto' for most queries - only use 'always' for explicitly cross-cutting questions that span multiple domains."
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
        "name": "agenthub_refresh",
        "description": """Refresh the AgentHub to reload all agents.

Use this when:
- You've modified agent configurations
- You've added new Tier A agents
- The agent list seems stale or outdated
- After running 'agenthub clean' or 'agenthub up'

This will re-discover all Tier A agents and regenerate Tier B agents.""",
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
    },
    {
        "name": "agenthub_parallel_preview",
        "description": """Preview how a multi-part request would be decomposed into parallel tasks.

Use this to understand what parallel execution would do before actually running it.
Shows task breakdown, risk analysis, and execution plan.""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The multi-part request to analyze (e.g., 'Add save button and chart component')"
                }
            },
            "required": ["request"]
        }
    },
    {
        "name": "agenthub_parallel_execute",
        "description": """Execute a multi-part request using parallel Claude Code sessions.

This decomposes the request into tasks and runs them in parallel on separate git branches,
then merges the results. Use for requests that involve multiple independent changes.

Prerequisites:
- Project must be a git repository
- Working tree must be clean (no uncommitted changes)
- Auto-agents must be enabled""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The multi-part request to execute in parallel"
                },
                "base_branch": {
                    "type": "string",
                    "description": "Git branch to work from (default: 'main')"
                }
            },
            "required": ["request"]
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
        team_mode = arguments.get("team_mode", "auto")

        # Broadcast query to dashboard
        broadcast_to_dashboard(
            "task",
            f"Claude Code query: {question[:100]}{'...' if len(question) > 100 else ''}",
            {"query": question, "requested_agent": agent_id, "team_mode": team_mode},
        )

        try:
            response = hub.run(question, agent_id=agent_id, team_mode=team_mode)

            # Build response info
            team_info = ""
            if response.metadata.get("team_execution"):
                agents_used = response.metadata.get("agents_used", [])
                team_info = f"\nTeam execution: {len(agents_used)} agents used ({', '.join(agents_used)})"

            # Broadcast result to dashboard (full content)
            broadcast_to_dashboard(
                "result",
                f"Agent {response.agent_id} responded ({response.tokens_used} tokens)",
                {
                    "agent_id": response.agent_id,
                    "tokens_used": response.tokens_used,
                    "query": question,
                    "response": response.content,
                    "team_execution": response.metadata.get("team_execution", False),
                },
            )

            return f"""Agent: {response.agent_id}
Tokens used: {response.tokens_used}{team_info}

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

        # Count by tier
        tier_a = [s for s in agents if s.metadata.get("tier") == "A" or not s.metadata.get("auto_generated")]
        tier_b = [s for s in agents if s.metadata.get("tier") == "B" and s.metadata.get("auto_generated") and not s.metadata.get("is_sub_agent")]
        sub_agents = [s for s in agents if s.metadata.get("is_sub_agent")]

        lines.append(f"Total: {len(agents)} agents ({len(tier_a)} Tier A, {len(tier_b)} Tier B, {len(sub_agents)} Sub-agents)")
        lines.append("")

        for spec in agents:
            tier = spec.metadata.get("tier", "A" if not spec.metadata.get("auto_generated") else "B")
            is_sub = spec.metadata.get("is_sub_agent", False)
            tier_label = f"Tier {tier}" + (" Sub" if is_sub else "")
            lines.append(f"* {spec.name} [{spec.agent_id}] ({tier_label})")
            lines.append(f"  {spec.description}")
            keywords = spec.context_keywords[:5]
            if keywords:
                lines.append(f"  Keywords: {', '.join(keywords)}")
            lines.append("")

        return "\n".join(lines)

    elif name == "agenthub_refresh":
        # Force refresh the hub
        refreshed_hub = get_hub(force_refresh=True)
        if refreshed_hub is None:
            return "Error: Failed to refresh AgentHub."

        agents = refreshed_hub.list_agents()
        tier_a = [s for s in agents if s.metadata.get("tier") == "A" or not s.metadata.get("auto_generated")]
        tier_b = [s for s in agents if s.metadata.get("tier") == "B" and s.metadata.get("auto_generated") and not s.metadata.get("is_sub_agent")]
        sub_agents = [s for s in agents if s.metadata.get("is_sub_agent")]

        return f"""AgentHub refreshed successfully!

Discovered agents:
- Tier A (Business): {len(tier_a)} agents
- Tier B (Code): {len(tier_b)} agents
- Sub-agents: {len(sub_agents)} agents
- Total: {len(agents)} agents

Use 'agenthub_list_agents' to see the full list."""

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

    elif name == "agenthub_parallel_preview":
        request = arguments.get("request", "")

        try:
            # Ensure parallel sessions is enabled
            if not hub.is_parallel_enabled:
                if _project_root:
                    hub.enable_parallel_sessions(_project_root)
                else:
                    return "Error: Cannot enable parallel sessions - no project root specified."

            decomp, plan = hub.preview_parallel(request)

            lines = [
                "Parallel Execution Preview",
                "=" * 40,
                "",
                f"Request: {request}",
                "",
                f"Tasks ({len(decomp.tasks)}):",
            ]

            for task in decomp.tasks:
                lines.append(f"  - [{task.task_id}] {task.description}")
                lines.append(f"    Complexity: {task.complexity}")
                if task.estimated_files:
                    lines.append(f"    Files: {', '.join(task.estimated_files[:3])}")
                if task.depends_on:
                    lines.append(f"    Depends on: {', '.join(task.depends_on)}")

            lines.extend([
                "",
                "Risk Analysis:",
                f"  Overall Risk: {plan.overall_risk.value.upper()}",
                f"  PM Recommendation: {plan.pm_recommendation}",
                f"  Confidence: {plan.confidence:.0%}",
            ])

            if plan.file_overlaps:
                lines.append(f"  File Overlaps: {len(plan.file_overlaps)}")

            lines.extend([
                "",
                f"Parallel Groups: {len(plan.parallel_groups)}",
                f"Estimated Speedup: {plan.estimated_speedup:.1f}x",
            ])

            if plan.reasoning:
                lines.extend(["", "Reasoning:", plan.reasoning])

            return "\n".join(lines)

        except Exception as e:
            return f"Error previewing parallel execution: {e}"

    elif name == "agenthub_parallel_execute":
        request = arguments.get("request", "")
        base_branch = arguments.get("base_branch", "main")

        # Broadcast to dashboard
        broadcast_to_dashboard(
            "parallel_execute",
            f"Parallel execution: {request[:100]}{'...' if len(request) > 100 else ''}",
            {"request": request, "base_branch": base_branch},
        )

        try:
            # Ensure parallel sessions is enabled
            if not hub.is_parallel_enabled:
                if _project_root:
                    hub.enable_parallel_sessions(_project_root)
                else:
                    return "Error: Cannot enable parallel sessions - no project root specified."

            result = hub.execute_parallel(request, base_branch)

            lines = [
                "Parallel Execution Result",
                "=" * 40,
                "",
                f"Success: {result.success}",
                f"Tasks: {len(result.tasks)}",
                f"Time: {result.total_time_seconds:.1f}s",
                f"Speedup: {result.speedup:.1f}x",
                f"Total Tokens: {result.total_tokens:,}",
                "",
            ]

            # Include error details if execution failed
            if not result.success and result.error_message:
                lines.extend([
                    "ERROR DETAILS:",
                    result.error_message,
                    "",
                ])
                if result.trace and result.trace.error_traceback:
                    lines.extend([
                        "Traceback:",
                        result.trace.error_traceback,
                        "",
                    ])

            if result.session_results:
                lines.append("Session Results:")
                for sr in result.session_results:
                    status = "Success" if sr.success else "Failed"
                    lines.append(f"  - {sr.task_id}: {status} ({sr.tokens_used} tokens)")
                    if sr.files_changed:
                        lines.append(f"    Changed: {', '.join(sr.files_changed[:3])}")
                    if sr.boundary_crossings:
                        lines.append(f"    Boundary crossings: {len(sr.boundary_crossings)}")
                lines.append("")

            if result.merge_result:
                lines.append("Merge Result:")
                lines.append(f"  Success: {result.merge_result.success}")
                lines.append(f"  Branch: {result.merge_result.merged_branch}")
                if result.merge_result.conflicts:
                    lines.append(f"  Conflicts: {len(result.merge_result.conflicts)}")
                if result.merge_result.needs_user_input:
                    lines.append(f"  Needs user input: {result.merge_result.escalation_reason}")

            # Broadcast result to dashboard
            broadcast_to_dashboard(
                "parallel_result",
                f"Parallel execution {'succeeded' if result.success else 'failed'}: {result.speedup:.1f}x speedup",
                {
                    "success": result.success,
                    "tasks": len(result.tasks),
                    "speedup": result.speedup,
                    "tokens": result.total_tokens,
                },
            )

            return "\n".join(lines)

        except Exception as e:
            broadcast_to_dashboard("error", f"Parallel execution failed: {str(e)}")
            return f"Error executing parallel sessions: {e}"

    else:
        return f"Unknown tool: {name}"


def send_response(response: dict):
    """Send a JSON-RPC response to stdout."""
    print(json.dumps(response), flush=True)


def main():
    """Main MCP server loop - reads JSON-RPC from stdin, writes to stdout."""
    global _project_root

    # Log all args for debugging
    print(f"AgentHub MCP Server starting...", file=sys.stderr)
    print(f"  sys.argv: {sys.argv}", file=sys.stderr)
    print(f"  AGENTHUB_PROJECT env: {os.environ.get('AGENTHUB_PROJECT', '(not set)')}", file=sys.stderr)

    # Parse command line args for project path
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--project" and i + 1 < len(args):
            _project_root = args[i + 1]
            break

    # Log resolved project source
    if _project_root:
        print(f"  Project (from --project arg): {_project_root}", file=sys.stderr)
    else:
        env_project = os.environ.get("AGENTHUB_PROJECT")
        if env_project:
            print(f"  Project will use AGENTHUB_PROJECT env: {env_project}", file=sys.stderr)
        else:
            print("  WARNING: No --project arg or AGENTHUB_PROJECT env. Will fall back to ~/.agenthub/config.json", file=sys.stderr)

    # Handle graceful shutdown so processes don't linger
    import signal

    def _shutdown(signum, frame):
        print(f"AgentHub MCP Server shutting down (signal {signum}).", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Read requests from stdin — exit cleanly when stdin closes (parent died)
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
