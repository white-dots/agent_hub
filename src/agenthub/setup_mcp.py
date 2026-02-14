from __future__ import annotations
#!/usr/bin/env python3
"""Setup script to configure AgentHub MCP server for Claude Code.

This script:
1. Detects your Claude Code settings location
2. Adds the AgentHub MCP server configuration
3. Shows you how to use it

Usage:
    python -m agenthub.setup_mcp
    python -m agenthub --setup
"""

import json
import os
import sys
from pathlib import Path


def get_claude_config_path() -> Path:
    """Get the Claude Code configuration file path.

    MCP servers are configured in ~/.claude.json (NOT ~/.claude/settings.json).
    This file is shared between the CLI and VSCode extension.
    """
    # MCP servers go in ~/.claude.json (home directory)
    return Path.home() / ".claude.json"


def get_agenthub_path() -> str:
    """Get the path to the agenthub package (src directory)."""
    # This file is at src/agenthub/setup_mcp.py
    # We need src/ directory
    return str(Path(__file__).parent.parent)


def main():
    print("=" * 60)
    print("  AgentHub MCP Server Setup")
    print("=" * 60)
    print()

    agenthub_path = get_agenthub_path()

    # Generate the MCP server configuration - no project path needed!
    mcp_config = {
        "agenthub": {
            "command": "python",
            "args": ["-m", "agenthub.mcp_server"],
            "env": {
                "PYTHONPATH": agenthub_path
            }
        }
    }

    print("MCP Server Configuration")
    print("-" * 40)
    print()
    print("Add this to your Claude Code settings (~/.claude.json):")
    print()

    settings_example = {
        "mcpServers": mcp_config
    }
    print(json.dumps(settings_example, indent=2))

    print()
    print("-" * 40)
    print()
    print("HOW IT WORKS:")
    print()
    print("1. The MCP server uses AgentHub library DIRECTLY")
    print("   - No dashboard required")
    print("   - No project path needed in config")
    print()
    print("2. Project is detected AUTOMATICALLY")
    print("   - Uses Claude Code's current working directory")
    print("   - Works with ANY project you open")
    print()
    print("3. Claude Code will have these tools:")
    print("   - agenthub_query: Ask questions using specialized agents")
    print("   - agenthub_list_agents: See available agents")
    print("   - agenthub_routing_rules: See routing logic")
    print()
    print("4. ANTHROPIC_API_KEY must be set in your environment")
    print()
    print("-" * 40)
    print()

    # Try to find existing Claude settings
    config_path = get_claude_config_path()
    print(f"Claude Code settings location: {config_path}")

    if config_path.exists():
        print("  [OK] Settings file exists")

        # Read and show current config
        try:
            with open(config_path) as f:
                current = json.load(f)

            if "mcpServers" in current:
                print("  [OK] mcpServers section exists")
                if "agenthub" in current["mcpServers"]:
                    print("  [OK] AgentHub already configured!")
                else:
                    print("  [TODO] Add 'agenthub' to existing mcpServers")
            else:
                print("  [TODO] Add mcpServers section with agenthub config")

        except Exception as e:
            print(f"  Could not read settings: {e}")
    else:
        print("  [TODO] Settings file will be created")

    print()
    print("=" * 60)

    # Ask if user wants auto-setup
    response = input("\nWould you like to automatically add/update the config? [y/N]: ")

    if response.lower() == 'y':
        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Read existing or create new
            if config_path.exists():
                with open(config_path) as f:
                    settings = json.load(f)
            else:
                settings = {}

            # Add/update mcpServers
            if "mcpServers" not in settings:
                settings["mcpServers"] = {}

            settings["mcpServers"]["agenthub"] = mcp_config["agenthub"]

            # Write back
            with open(config_path, 'w') as f:
                json.dump(settings, indent=2, fp=f)

            print(f"\n[OK] Configuration saved to {config_path}")
            print("\nRestart Claude Code to use AgentHub tools!")
            print("\nAgentHub will work with ANY project you open.")

        except Exception as e:
            print(f"\nError saving config: {e}")
            print("Please add the configuration manually.")
    else:
        print("\nNo changes made. Add the configuration manually when ready.")


if __name__ == "__main__":
    main()
