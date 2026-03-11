"""AgentHub: Impact analysis for Claude Code.

Prevent breaking changes in large repos by exposing static import graph
analysis as MCP tools. Zero LLM calls, zero token cost.

MCP tools:
    impact_check    - Blast radius for a file edit
    affected_tests  - Tests to run after edits
    codebase_overview - Module count, language breakdown, central modules

CLI:
    $ agenthub map /path/to/project   # Generate CLAUDE.md with repo map
"""

from agenthub.auto.import_graph import ImportGraph

__version__ = "2.0.0"

__all__ = [
    "ImportGraph",
    "__version__",
]
