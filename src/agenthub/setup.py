"""AgentHub setup utilities for API keys, MCP config, and project configuration.

This module handles:
- Checking and prompting for API keys
- Storing project path in global config
- Generating MCP configuration for Claude Code
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Global config location
AGENTHUB_CONFIG_DIR = Path.home() / ".agenthub"
AGENTHUB_CONFIG_FILE = AGENTHUB_CONFIG_DIR / "config.json"

# Claude Code MCP config location
CLAUDE_CONFIG_FILE = Path.home() / ".claude.json"


def get_config() -> dict:
    """Load AgentHub global config."""
    if AGENTHUB_CONFIG_FILE.exists():
        try:
            with open(AGENTHUB_CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(config: dict) -> None:
    """Save AgentHub global config."""
    AGENTHUB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(AGENTHUB_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_current_project() -> Optional[str]:
    """Get the currently configured project path."""
    config = get_config()
    return config.get("project")


def set_current_project(project_path: str) -> None:
    """Set the current project path."""
    config = get_config()
    config["project"] = str(Path(project_path).resolve())
    save_config(config)


def check_api_keys(project_path: str) -> dict:
    """Check for API keys in environment and .env files.

    Returns:
        Dict with 'anthropic' and 'openai' keys, each True/False.
    """
    from agenthub.config import load_env_files

    # Load .env files
    load_env_files(verbose=False)

    # Also try project-specific .env
    project_env = Path(project_path) / ".env"
    if project_env.exists():
        from dotenv import load_dotenv
        load_dotenv(project_env, override=True)

    return {
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
    }


def prompt_for_api_key(project_path: str) -> bool:
    """Prompt user to enter API key if missing.

    Returns:
        True if at least one API key is now available.
    """
    keys = check_api_keys(project_path)

    if keys["anthropic"] or keys["openai"]:
        return True

    print()
    print("No API keys found!")
    print("-" * 40)
    print("AgentHub needs at least one API key to work.")
    print()
    print("Options:")
    print("  1. Enter ANTHROPIC_API_KEY now")
    print("  2. Enter OPENAI_API_KEY now")
    print("  3. Skip (I'll set it up manually)")
    print()

    choice = input("Choose [1/2/3]: ").strip()

    if choice == "1":
        api_key = input("Enter ANTHROPIC_API_KEY: ").strip()
        if api_key:
            _save_api_key_to_env(project_path, "ANTHROPIC_API_KEY", api_key)
            os.environ["ANTHROPIC_API_KEY"] = api_key
            print("Saved to .env file")
            return True
    elif choice == "2":
        api_key = input("Enter OPENAI_API_KEY: ").strip()
        if api_key:
            _save_api_key_to_env(project_path, "OPENAI_API_KEY", api_key)
            os.environ["OPENAI_API_KEY"] = api_key
            print("Saved to .env file")
            return True
    elif choice == "3":
        print()
        print("Manual setup instructions:")
        print(f"  1. Create {project_path}/.env")
        print("  2. Add: ANTHROPIC_API_KEY=your-key-here")
        print("  3. Run 'agenthub build' again")
        return False

    return False


def _save_api_key_to_env(project_path: str, key_name: str, key_value: str) -> None:
    """Save an API key to the project's .env file."""
    env_file = Path(project_path) / ".env"

    # Read existing content
    existing_lines = []
    if env_file.exists():
        with open(env_file) as f:
            existing_lines = f.readlines()

    # Check if key already exists
    key_found = False
    new_lines = []
    for line in existing_lines:
        if line.strip().startswith(f"{key_name}="):
            new_lines.append(f"{key_name}={key_value}\n")
            key_found = True
        else:
            new_lines.append(line)

    # Add key if not found
    if not key_found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"{key_name}={key_value}\n")

    # Write back
    with open(env_file, "w") as f:
        f.writelines(new_lines)


def get_agenthub_src_path() -> str:
    """Get the path to AgentHub's src directory."""
    # This file is in src/agenthub/setup.py
    return str(Path(__file__).parent.parent.resolve())


def generate_mcp_config(project_path: str) -> dict:
    """Generate MCP server configuration for Claude Code.

    Args:
        project_path: Path to the project to configure.

    Returns:
        MCP server config dict.
    """
    src_path = get_agenthub_src_path()
    project_path = str(Path(project_path).resolve())

    # Determine python command
    python_cmd = sys.executable

    return {
        "command": python_cmd,
        "args": ["-m", "agenthub.mcp_server", "--project", project_path],
        "env": {
            "PYTHONPATH": src_path,
        }
    }


def update_claude_config(project_path: str) -> bool:
    """Update Claude Code's MCP configuration.

    Args:
        project_path: Path to the project.

    Returns:
        True if config was updated successfully.
    """
    mcp_config = generate_mcp_config(project_path)

    # Load existing Claude config
    claude_config = {}
    if CLAUDE_CONFIG_FILE.exists():
        try:
            with open(CLAUDE_CONFIG_FILE) as f:
                claude_config = json.load(f)
        except Exception:
            pass

    # Ensure mcpServers exists
    if "mcpServers" not in claude_config:
        claude_config["mcpServers"] = {}

    # Update agenthub config
    claude_config["mcpServers"]["agenthub"] = mcp_config

    # Write back
    try:
        with open(CLAUDE_CONFIG_FILE, "w") as f:
            json.dump(claude_config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error updating Claude config: {e}")
        return False


def print_mcp_config(project_path: str) -> None:
    """Print the MCP configuration for manual setup."""
    config = generate_mcp_config(project_path)

    print()
    print("MCP Configuration for Claude Code:")
    print("-" * 40)
    print(f"Add this to {CLAUDE_CONFIG_FILE}:")
    print()
    print(json.dumps({"mcpServers": {"agenthub": config}}, indent=2))
    print()


def setup_mcp_config(project_path: str, auto: bool = False) -> bool:
    """Set up MCP configuration, optionally prompting user.

    Args:
        project_path: Path to the project.
        auto: If True, automatically update without prompting.

    Returns:
        True if MCP was configured.
    """
    if auto:
        return update_claude_config(project_path)

    print()
    print("MCP Configuration")
    print("-" * 40)
    print("Would you like to configure Claude Code to use AgentHub?")
    print()
    print("Options:")
    print(f"  1. Auto-configure (update {CLAUDE_CONFIG_FILE})")
    print("  2. Show config (I'll set it up manually)")
    print("  3. Skip")
    print()

    choice = input("Choose [1/2/3]: ").strip()

    if choice == "1":
        if update_claude_config(project_path):
            print(f"Updated {CLAUDE_CONFIG_FILE}")
            print("Restart Claude Code to use the new MCP server.")
            return True
        else:
            print("Failed to update config. Please set up manually.")
            print_mcp_config(project_path)
            return False
    elif choice == "2":
        print_mcp_config(project_path)
        return True

    return False
