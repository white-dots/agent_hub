from __future__ import annotations
"""AgentHub CLI - Docker-like commands for agent management.

Commands:
    agenthub init        - Initialize AgentHub in a project (one-command setup)
    agenthub build       - Discover Tier A agents, generate Tier B agents
    agenthub up          - Start dashboard and verify agents
    agenthub watch       - Enable file watching for automatic refresh
    agenthub status      - Show current agent status
    agenthub clean       - Remove Tier B agents and sub-agents
    agenthub restructure - Delete and regenerate Tier B agents

Example:
    $ agenthub init              # Initialize current directory
    $ agenthub init /path/to/dir # Initialize specific directory
    $ agenthub build ./my-project
    $ agenthub up --port 3001
    $ agenthub watch ./my-project
    $ agenthub clean             # Remove Tier B agents
    $ agenthub restructure ./my-project --force
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _setup_path():
    """Ensure agenthub is importable."""
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _load_env():
    """Load environment variables from .env files."""
    from agenthub.config import load_env_files
    load_env_files()


def cmd_init(args):
    """Init command - one-command project setup.

    This is the recommended starting point for new projects:
    1. Detects project type (Python, Node.js, etc.)
    2. Creates {project}_agents.py with auto-discovery
    3. Creates .agenthubignore for ignoring build artifacts
    4. Configures MCP for Claude Code integration
    5. Checks for API keys

    Usage:
        $ agenthub init              # Initialize current directory
        $ agenthub init /path/to/dir # Initialize specific directory
    """
    _setup_path()

    from agenthub.setup import setup_project

    project_path = args.path or os.getcwd()
    project_path = os.path.abspath(project_path)

    if not os.path.isdir(project_path):
        print(f"Error: '{project_path}' is not a valid directory")
        return 1

    results = setup_project(
        project_path=project_path,
        auto_mcp=not args.manual_mcp,
        skip_api_key_check=args.skip_api_check,
        verbose=not args.quiet,
    )

    if results["errors"]:
        for err in results["errors"]:
            print(f"Error: {err}")
        return 1

    if results["warnings"] and not args.quiet:
        print("Warnings:")
        for warn in results["warnings"]:
            print(f"  - {warn}")

    return 0 if results["success"] else 1


def cmd_build(args):
    """Build command - discovers Tier A and generates Tier B agents.

    This is the setup phase that:
    1. Checks for API keys (prompts if missing)
    2. Analyzes the codebase and discovers agents
    3. Saves project configuration
    4. Offers to configure MCP for Claude Code
    """
    _setup_path()

    from agenthub.auto import discover_all_agents
    from agenthub.setup import (
        check_api_keys,
        prompt_for_api_key,
        set_current_project,
        setup_mcp_config,
    )

    project_root = args.path or os.getcwd()
    project_root = os.path.abspath(project_root)

    if not os.path.isdir(project_root):
        print(f"Error: '{project_root}' is not a valid directory")
        return 1

    project_name = Path(project_root).name

    print("=" * 60)
    print(f"  AgentHub Build: {project_name}")
    print("=" * 60)
    print()
    print(f"Project root: {project_root}")
    print()

    # Step 1: Check for API keys
    print("Step 1: Checking API keys...")
    keys = check_api_keys(project_root)

    if not keys["anthropic"] and not keys["openai"]:
        if not prompt_for_api_key(project_root):
            print()
            print("Warning: No API keys configured.")
            print("Agents will be discovered but won't be able to run queries.")
            if not args.force:
                response = input("Continue anyway? [y/N] ").strip().lower()
                if response != 'y':
                    return 1
        # Re-check after prompt
        keys = check_api_keys(project_root)

    if keys["anthropic"] or keys["openai"]:
        providers = []
        if keys["anthropic"]:
            providers.append("Claude")
        if keys["openai"]:
            providers.append("ChatGPT")
        print(f"Available LLM providers: {', '.join(providers)}")
    print()

    # Step 2: Discover agents
    print("Step 2: Discovering agents...")
    print("-" * 40)

    try:
        hub, summary = discover_all_agents(
            project_root,
            include_tier_a=not args.no_tier_a,
            include_tier_b=not args.no_tier_b,
        )

        print()
        print(summary)
        print()

        # Step 3: Save project configuration
        print("Step 3: Saving project configuration...")
        set_current_project(project_root)
        print(f"Project saved to ~/.agenthub/config.json")
        print()

        # Step 4: Configure MCP (unless --skip-mcp)
        if not args.skip_mcp:
            print("Step 4: MCP Configuration")
            setup_mcp_config(project_root, auto=args.auto_mcp)

        print()
        print("=" * 60)
        print("  Build complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  agenthub up        - Start the dashboard")
        print("  agenthub watch     - Enable file watching")
        print()
        print("With Claude Code:")
        print("  1. Restart Claude Code to load the MCP server")
        print("  2. Use agenthub_query tool to ask questions")
        print()

        return 0

    except Exception as e:
        print(f"Error during build: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_up(args):
    """Up command - starts dashboard and verifies agents are working."""
    _setup_path()
    _load_env()

    from agenthub.auto import discover_all_agents
    from agenthub.setup import get_current_project

    # Use provided path, or saved project, or current directory
    project_root = args.path
    if not project_root:
        project_root = get_current_project()
    if not project_root:
        project_root = os.getcwd()
    project_root = os.path.abspath(project_root)

    if not os.path.isdir(project_root):
        print(f"Error: '{project_root}' is not a valid directory")
        return 1

    # Load .env from project directory
    project_env = Path(project_root) / ".env"
    if project_env.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(project_env, override=True)
        except ImportError:
            pass

    project_name = Path(project_root).name
    port = args.port or 3001

    print("=" * 60)
    print(f"  AgentHub Up: {project_name}")
    print("=" * 60)
    print()

    # Check dependencies
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("Dashboard requires fastapi and uvicorn.")
        print("Install with: pip install fastapi uvicorn")
        return 1

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set. Queries will fail.")
        print("Set it with: export ANTHROPIC_API_KEY=your-key-here")
        print()

    print("Discovering and building agents...")

    try:
        hub, summary = discover_all_agents(
            project_root,
            include_tier_a=True,
            include_tier_b=True,
        )

        # Print agent summary
        tier_a = [a for a in hub.list_agents() if not a.metadata.get("auto_generated")]
        tier_b = [a for a in hub.list_agents() if a.metadata.get("auto_generated")]

        print()
        print(f"Tier A (Business): {len(tier_a)} agents")
        for a in tier_a:
            provider = a.metadata.get("llm_provider", "claude")
            print(f"  - {a.name} [{provider}]")

        print(f"\nTier B (Code): {len(tier_b)} agents")
        for a in tier_b:
            module_type = a.metadata.get("module_type", "code")
            print(f"  - {a.name} ({module_type})")

        print()
        print("-" * 40)

        # Test agent availability (optional health check)
        if args.test:
            print("\nTesting agent connectivity...")
            test_passed = True
            for spec in hub.list_agents()[:3]:  # Test first 3 agents
                agent = hub.get_agent(spec.agent_id)
                if agent:
                    try:
                        context = agent.get_context()
                        if context:
                            print(f"  [OK] {spec.name}: context loaded ({len(context)} chars)")
                        else:
                            print(f"  [!] {spec.name}: empty context")
                    except Exception as e:
                        print(f"  [X] {spec.name}: {e}")
                        test_passed = False

            if not test_passed:
                print("\nSome agents failed health check. Continue anyway? (y/n)")
                if input().strip().lower() != 'y':
                    return 1
            print()

        # Enable QC analysis if requested
        if args.qc:
            print("\nEnabling QC analysis...")
            hub.enable_qc_analysis(
                auto_analyze=True,
                on_concern=lambda c: print(f"  [QC] {c.severity.value.upper()}: {c.title}"),
                on_report=lambda r: print(f"  [QC Report] {r.recommendation}: {r.total_concerns} concerns"),
            )
            print("  QC Analysis: ENABLED (auto-analyze on file changes)")

        # Start dashboard
        print("=" * 60)
        print(f"  Starting dashboard on http://localhost:{port}")
        print("=" * 60)
        print()
        if not args.qc:
            print("Tip: Use --qc to enable QC analysis")
        print("Press Ctrl+C to stop")
        print()

        from agenthub.dashboard import run_dashboard
        run_dashboard(hub, port=port)

        return 0

    except KeyboardInterrupt:
        print("\nShutdown requested...")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_watch(args):
    """Watch command - enables file watching for automatic Tier B refresh."""
    _setup_path()
    _load_env()

    from agenthub.auto import discover_all_agents
    from agenthub.cache.watcher import FileWatcher
    from agenthub.setup import get_current_project

    # Use provided path, or saved project, or current directory
    project_root = args.path
    if not project_root:
        project_root = get_current_project()
    if not project_root:
        project_root = os.getcwd()
    project_root = os.path.abspath(project_root)

    if not os.path.isdir(project_root):
        print(f"Error: '{project_root}' is not a valid directory")
        return 1

    project_name = Path(project_root).name

    print("=" * 60)
    print(f"  AgentHub Watch: {project_name}")
    print("=" * 60)
    print()

    print("Building initial agent set...")

    try:
        hub, summary = discover_all_agents(
            project_root,
            include_tier_a=True,
            include_tier_b=True,
        )

        tier_b = [a for a in hub.list_agents() if a.metadata.get("auto_generated")]
        print(f"\nWatching {len(tier_b)} Tier B agents for changes...")
        print()

        # Enable git cache
        hub.enable_git_cache(project_root)
        print("[Cache] Git-aware caching enabled")

        # Define refresh callback
        def on_change(changed_paths):
            print()
            print("-" * 40)
            print(f"[Watch] Detected changes in {len(changed_paths)} file(s):")
            for p in list(changed_paths)[:5]:
                print(f"  - {p}")
            if len(changed_paths) > 5:
                print(f"  ... and {len(changed_paths) - 5} more")

            # Invalidate cache for affected agents
            cache = hub._cache
            if cache:
                invalidated = 0
                for spec in tier_b:
                    agent = hub.get_agent(spec.agent_id)
                    if agent and hasattr(agent, 'spec'):
                        # Check if any watched paths overlap
                        agent_paths = set(agent.spec.context_paths)
                        if agent_paths & changed_paths:
                            cache.invalidate(f"agent_context_{spec.agent_id}")
                            invalidated += 1
                            print(f"[Cache] Invalidated: {spec.agent_id}")

                if invalidated > 0:
                    print(f"\n[Watch] {invalidated} agent context(s) will refresh on next query")
            print("-" * 40)

        # Create and start watcher
        watcher = FileWatcher(
            project_root,
            on_refresh=on_change,
            debounce_seconds=args.debounce or 1.0,
        )

        # Add patterns to watch
        watch_patterns = ["**/*.py", "**/*.json", "**/*.yaml", "**/*.yml"]
        for pattern in watch_patterns:
            print(f"[Watch] Pattern: {pattern}")

        print()
        print("Watching for file changes... (Press Ctrl+C to stop)")
        print()

        watcher.start()

        try:
            # Keep running until interrupted
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping watcher...")
            watcher.stop()

        return 0

    except ImportError as e:
        if "watchdog" in str(e):
            print("File watching requires the watchdog library.")
            print("Install with: pip install watchdog")
            return 1
        raise
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_restructure(args):
    """Restructure command - delete and regenerate Tier B agents.

    Useful when:
    - Codebase has changed dramatically
    - You want to use different domain detection settings
    - Tier B agents have become stale or inaccurate
    """
    _setup_path()
    _load_env()

    from agenthub.auto import discover_all_agents, print_agent_tree
    from agenthub.setup import get_current_project

    # Use provided path, or saved project, or current directory
    project_root = args.path
    if not project_root:
        project_root = get_current_project()
    if not project_root:
        project_root = os.getcwd()
    project_root = os.path.abspath(project_root)

    if not os.path.isdir(project_root):
        print(f"Error: '{project_root}' is not a valid directory")
        return 1

    project_name = Path(project_root).name

    print("=" * 60)
    print(f"  AgentHub Restructure: {project_name}")
    print("=" * 60)
    print()

    # First, discover current agents
    print("Analyzing current agent structure...")
    try:
        hub, _ = discover_all_agents(
            project_root,
            include_tier_a=True,
            include_tier_b=True,
        )

        # Get current Tier B agents
        current_tier_b = hub.list_agents(tier="B")
        current_tier_a = hub.list_agents(tier="A")

        print()
        print("Current Structure:")
        print("-" * 40)
        print(f"  Tier A (Business): {len(current_tier_a)} agents")
        for a in current_tier_a:
            print(f"    - {a.name}")
        print(f"  Tier B (Code): {len(current_tier_b)} agents")
        for a in current_tier_b:
            module_type = a.metadata.get("module_type", "code")
            files_count = len(a.context_paths)
            print(f"    - {a.name} ({module_type}, {files_count} files)")

        print()

        # Confirm unless --force
        if not args.force:
            print(f"This will delete {len(current_tier_b)} Tier B agent(s) and regenerate them.")
            print("Tier A agents will be preserved.")
            print()
            response = input("Continue? [y/N] ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return 0

        print()
        print("Removing Tier B agents...")

        # Remove Tier B agents
        removed = hub.unregister_tier_b()
        print(f"  Removed {len(removed)} agent(s)")

        # Regenerate Tier B agents
        print()
        print("Regenerating Tier B agents...")
        print("-" * 40)

        # Re-run discovery with fresh Tier B generation
        hub_new, summary = discover_all_agents(
            project_root,
            include_tier_a=True,
            include_tier_b=True,
        )

        new_tier_b = hub_new.list_agents(tier="B")

        print()
        print(summary)

        # Show comparison
        print()
        print("=" * 60)
        print("  Restructure Complete!")
        print("=" * 60)
        print()
        print("Comparison:")
        print(f"  Before: {len(current_tier_b)} Tier B agents")
        print(f"  After:  {len(new_tier_b)} Tier B agents")

        # Show new tree
        print()
        print("New Agent Tree:")
        print(print_agent_tree(hub_new, project_name))

        if args.verbose:
            print()
            print("New Tier B agents:")
            for a in new_tier_b:
                module_type = a.metadata.get("module_type", "code")
                files_count = len(a.context_paths)
                keywords = a.context_keywords[:5]
                print(f"  - {a.name} ({module_type})")
                print(f"      Files: {files_count}")
                print(f"      Keywords: {', '.join(keywords)}")

        print()
        print("Next steps:")
        print("  agenthub up        - Start the dashboard with new agents")
        print("  agenthub status    - View current agent status")
        print()

        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_status(args):
    """Status command - show current agent status and cache info."""
    _setup_path()
    _load_env()

    from agenthub.auto import discover_all_agents, print_agent_tree
    from agenthub.setup import get_current_project

    # Use provided path, or saved project, or current directory
    project_root = args.path
    if not project_root:
        project_root = get_current_project()
    if not project_root:
        project_root = os.getcwd()
    project_root = os.path.abspath(project_root)

    if not os.path.isdir(project_root):
        print(f"Error: '{project_root}' is not a valid directory")
        return 1

    project_name = Path(project_root).name

    print("=" * 60)
    print(f"  AgentHub Status: {project_name}")
    print("=" * 60)
    print()

    try:
        hub, _ = discover_all_agents(project_root)

        # Print tree
        print(print_agent_tree(hub, project_name))
        print()

        # Print summary stats
        all_agents = hub.list_agents()
        tier_a = [a for a in all_agents if not a.metadata.get("auto_generated")]
        tier_b = [a for a in all_agents if a.metadata.get("auto_generated")]

        print("-" * 40)
        print(f"Total agents: {len(all_agents)}")
        print(f"  Tier A (Business): {len(tier_a)}")
        print(f"  Tier B (Code): {len(tier_b)}")

        # Cache info
        if hasattr(hub, '_cache') and hub._cache:
            stats = hub.get_cache_stats()
            print()
            print("Cache Status:")
            print(f"  Entries: {stats.get('entries', 0)}")
            print(f"  Git commit: {stats.get('current_commit', 'N/A')[:8] if stats.get('current_commit') else 'N/A'}")

        print()
        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_map(args):
    """Map command - generate a repo-aware CLAUDE.md from the dependency graph.

    Analyzes the codebase's import structure and generates a CLAUDE.md file
    that gives Claude Code (and all its subagents) repo-specific context:
    key modules, dependency chains, and editing guidance.

    Usage:
        $ agenthub map               # Generate CLAUDE.md in current directory
        $ agenthub map /path/to/dir  # Generate for specific project
        $ agenthub map --stdout      # Print to stdout instead of writing file
    """
    _setup_path()

    project_root = args.path or os.getcwd()
    project_root = os.path.abspath(project_root)

    if not os.path.isdir(project_root):
        print(f"Error: '{project_root}' is not a valid directory")
        return 1

    project_name = Path(project_root).name

    print(f"Analyzing {project_name}...", file=sys.stderr)

    try:
        from agenthub.auto.import_graph import ImportGraph
        from agenthub.repo_map import generate_claude_md, generate_repo_map

        graph = ImportGraph(project_root)
        graph.build()
        stats = graph.get_stats()
        print(f"  {stats['total_modules']} modules, {stats['total_edges']} dependencies", file=sys.stderr)

        if args.stdout:
            print(generate_claude_md(graph, project_root))
            return 0

        # Write CLAUDE.md
        output_path = Path(project_root) / "CLAUDE.md"
        existing_content = ""

        if output_path.exists():
            existing_content = output_path.read_text(encoding="utf-8")

            # Check if it already has an auto-generated section
            marker = "## Repo Map (auto-generated by agenthub)"
            if marker in existing_content:
                # Replace just the repo map section
                before = existing_content.split(marker)[0].rstrip()
                # Find the next ## heading after the repo map
                after_map = existing_content.split(marker)[1]
                next_heading = after_map.find("\n## ")
                if next_heading != -1:
                    after = after_map[next_heading:]
                else:
                    after = ""

                repo_map = generate_repo_map(graph)
                new_content = before + "\n\n" + repo_map + after
                output_path.write_text(new_content, encoding="utf-8")
                print(f"Updated repo map in: {output_path}")
                return 0

            # Append to existing CLAUDE.md
            repo_map = generate_repo_map(graph)
            new_content = existing_content.rstrip() + "\n\n" + repo_map + "\n"
            output_path.write_text(new_content, encoding="utf-8")
            print(f"Appended repo map to: {output_path}")
        else:
            # Create new CLAUDE.md
            content = generate_claude_md(graph, project_root)
            output_path.write_text(content, encoding="utf-8")
            print(f"Created: {output_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_clean(args):
    """Clean command - remove Tier B agents and sub-agents.

    This removes all auto-generated code agents while preserving
    Tier A (business/domain) agents. Useful when you want to:
    - Regenerate Tier B agents with new settings
    - Clear cached agent configurations
    - Start fresh with only Tier A agents

    Usage:
        $ agenthub clean              # Clean current project
        $ agenthub clean /path/to/dir # Clean specific project
        $ agenthub clean --all        # Also clear sub-agent policy config
    """
    _setup_path()
    _load_env()

    from pathlib import Path
    import json

    from agenthub.setup import get_current_project

    # Get project path
    project_root = args.path
    if not project_root:
        project_root = get_current_project()
    if not project_root:
        project_root = os.getcwd()
    project_root = os.path.abspath(project_root)

    if not os.path.isdir(project_root):
        print(f"Error: '{project_root}' is not a valid directory")
        return 1

    project_name = Path(project_root).name

    print("=" * 60)
    print(f"  AgentHub Clean: {project_name}")
    print("=" * 60)
    print()

    # Show what will be cleaned
    print("This will remove:")
    print("  - All Tier B (auto-generated code) agents")
    print("  - All sub-agents")
    if args.all:
        print("  - Sub-agent policy configuration")
    print()
    print("Tier A (business/domain) agents will be preserved.")
    print()

    # Confirm unless --force
    if not args.force:
        response = input("Continue? [y/N] ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return 0

    cleaned_items = []

    # Clean sub-agent policy config if --all
    if args.all:
        config_file = Path.home() / ".agenthub" / "sub_agent_policy.json"
        if config_file.exists():
            config_file.unlink()
            cleaned_items.append("Sub-agent policy config")
            print("  Removed: ~/.agenthub/sub_agent_policy.json")

    # Note: Tier B agents are generated dynamically by discover_all_agents()
    # They are not persisted to disk, so "cleaning" means they simply won't
    # be generated on the next run if we clear the relevant caches.

    # Clear any cached agent data
    cache_dir = Path.home() / ".agenthub" / "cache"
    if cache_dir.exists():
        import shutil
        # Only remove project-specific cache
        project_cache = cache_dir / project_name
        if project_cache.exists():
            shutil.rmtree(project_cache)
            cleaned_items.append(f"Project cache: {project_name}")
            print(f"  Removed: ~/.agenthub/cache/{project_name}/")

    # Clear import graph cache if exists
    import_graph_cache = Path.home() / ".agenthub" / "import_graphs" / f"{project_name}.json"
    if import_graph_cache.exists():
        import_graph_cache.unlink()
        cleaned_items.append("Import graph cache")
        print(f"  Removed: ~/.agenthub/import_graphs/{project_name}.json")

    print()
    if cleaned_items:
        print(f"Cleaned {len(cleaned_items)} item(s).")
    else:
        print("Nothing to clean (Tier B agents are generated dynamically).")

    print()
    print("Next steps:")
    print("  - Run 'agenthub up' to regenerate Tier B agents")
    print("  - Adjust settings in dashboard if needed")
    print()

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="agenthub",
        description="AgentHub - AI Agent Orchestration Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  init         Initialize AgentHub in a project (recommended for new users)
  build        Discover Tier A agents, generate Tier B agents
  map          Generate repo-aware CLAUDE.md from dependency graph
  up           Start dashboard and verify agents are working
  watch        Enable file watching for automatic context refresh
  status       Show current agent status
  clean        Remove Tier B agents and sub-agents
  restructure  Delete and regenerate all Tier B agents

Examples:
  agenthub init              # One-command setup
  agenthub map               # Generate CLAUDE.md with repo map
  agenthub map --stdout      # Print repo map without writing file
  agenthub build ./my-project
  agenthub up --port 3001
  agenthub status
        """
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Init command (recommended for new users)
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize AgentHub in a project (one-command setup)"
    )
    init_parser.add_argument(
        "path",
        nargs="?",
        help="Project root path (default: current directory)"
    )
    init_parser.add_argument(
        "--manual-mcp",
        action="store_true",
        help="Don't auto-configure MCP, show config for manual setup"
    )
    init_parser.add_argument(
        "--skip-api-check",
        action="store_true",
        help="Skip API key verification"
    )
    init_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output"
    )
    init_parser.set_defaults(func=cmd_init)

    # Build command
    build_parser = subparsers.add_parser(
        "build",
        help="Discover and build agents for a project"
    )
    build_parser.add_argument(
        "path",
        nargs="?",
        help="Project root path (default: current directory)"
    )
    build_parser.add_argument(
        "--no-tier-a",
        action="store_true",
        help="Skip Tier A agent discovery"
    )
    build_parser.add_argument(
        "--no-tier-b",
        action="store_true",
        help="Skip Tier B agent generation"
    )
    build_parser.add_argument(
        "--skip-mcp",
        action="store_true",
        help="Skip MCP configuration for Claude Code"
    )
    build_parser.add_argument(
        "--auto-mcp",
        action="store_true",
        help="Automatically configure MCP without prompting"
    )
    build_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Continue even without API keys configured"
    )
    build_parser.set_defaults(func=cmd_build)

    # Up command
    up_parser = subparsers.add_parser(
        "up",
        help="Start the dashboard"
    )
    up_parser.add_argument(
        "path",
        nargs="?",
        help="Project root path (default: current directory)"
    )
    up_parser.add_argument(
        "-p", "--port",
        type=int,
        default=3001,
        help="Dashboard port (default: 3001)"
    )
    up_parser.add_argument(
        "--test",
        action="store_true",
        help="Run health check before starting"
    )
    up_parser.add_argument(
        "--qc",
        action="store_true",
        help="Enable QC analysis (Tier B concerns + Tier C synthesis)"
    )
    up_parser.set_defaults(func=cmd_up)

    # Watch command
    watch_parser = subparsers.add_parser(
        "watch",
        help="Watch for file changes and refresh agents"
    )
    watch_parser.add_argument(
        "path",
        nargs="?",
        help="Project root path (default: current directory)"
    )
    watch_parser.add_argument(
        "--debounce",
        type=float,
        default=1.0,
        help="Debounce time in seconds (default: 1.0)"
    )
    watch_parser.set_defaults(func=cmd_watch)

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show agent status and cache info"
    )
    status_parser.add_argument(
        "path",
        nargs="?",
        help="Project root path (default: current directory)"
    )
    status_parser.set_defaults(func=cmd_status)

    # Map command
    map_parser = subparsers.add_parser(
        "map",
        help="Generate repo-aware CLAUDE.md from dependency graph"
    )
    map_parser.add_argument(
        "path",
        nargs="?",
        help="Project root path (default: current directory)"
    )
    map_parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of writing CLAUDE.md"
    )
    map_parser.set_defaults(func=cmd_map)

    # Clean command
    clean_parser = subparsers.add_parser(
        "clean",
        help="Remove Tier B agents and sub-agents"
    )
    clean_parser.add_argument(
        "path",
        nargs="?",
        help="Project root path (default: current directory)"
    )
    clean_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    clean_parser.add_argument(
        "--all",
        action="store_true",
        help="Also clear sub-agent policy configuration"
    )
    clean_parser.set_defaults(func=cmd_clean)

    # Restructure command
    restructure_parser = subparsers.add_parser(
        "restructure",
        help="Delete and regenerate all Tier B agents"
    )
    restructure_parser.add_argument(
        "path",
        nargs="?",
        help="Project root path (default: current directory)"
    )
    restructure_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    restructure_parser.set_defaults(func=cmd_restructure)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
