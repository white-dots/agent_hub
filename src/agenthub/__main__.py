"""Entry point for running AgentHub as a module.

Usage:
    python -m agenthub build [path]   # Discover Tier A, generate Tier B agents
    python -m agenthub up [--port N]  # Start dashboard
    python -m agenthub watch [path]   # Watch for file changes
    python -m agenthub status         # Show agent status

    python -m agenthub mcp            # Start MCP server
    python -m agenthub mcp --setup    # Configure MCP server for Claude Code
"""

import sys


def main():
    args = sys.argv[1:]

    # MCP server commands (legacy compatibility)
    if len(args) > 0 and args[0] == "mcp":
        if "--setup" in args:
            from agenthub.setup_mcp import main as setup_main
            setup_main()
        else:
            from agenthub.mcp_server import main as mcp_main
            mcp_main()
        return

    # Legacy flags for MCP (backwards compatible)
    if "--setup" in args:
        from agenthub.setup_mcp import main as setup_main
        setup_main()
        return

    if "--server" in args:
        from agenthub.mcp_server import main as mcp_main
        mcp_main()
        return

    # Default: Use CLI (build, up, watch, status)
    from agenthub.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
