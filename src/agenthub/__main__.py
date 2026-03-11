"""Entry point for running AgentHub as a module.

Usage:
    python -m agenthub mcp --project /path   # Start MCP server
    python -m agenthub map [path]            # Generate CLAUDE.md
    python -m agenthub map --stdout          # Print repo map to stdout
"""

import sys


def main():
    args = sys.argv[1:]

    # MCP server
    if len(args) > 0 and args[0] == "mcp":
        from agenthub.mcp_server import main as mcp_main
        mcp_main()
        return

    if "--server" in args:
        from agenthub.mcp_server import main as mcp_main
        mcp_main()
        return

    # CLI (map command)
    from agenthub.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
