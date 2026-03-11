# AgentHub

**Impact analysis for Claude Code — prevent breaking changes in large repos.**

When AI edits a file in a large codebase, it doesn't know what depends on that file. A renamed function, a changed signature, or a deleted class silently breaks downstream code. You find out later when tests fail — or worse, in production.

AgentHub fixes this. It builds a static import graph of your project and exposes it as MCP tools for Claude Code. Before editing a file, Claude checks its blast radius. After editing, it knows exactly which tests to run. **No LLM calls, no API key, zero token cost.**

## What you get

| Tool | When Claude calls it | What it returns |
|------|---------------------|-----------------|
| `impact_check` | Before editing a file | Exported interface, dependents (direct + transitive), affected tests, risk level |
| `affected_tests` | After editing files | Test files to run + suggested pytest/jest command |
| `codebase_overview` | Orientation | Module count, language breakdown, central modules, project stats |

Plus:
- **Repo Map** — auto-generated MCP resource injected into every agent's context (including parallel subagents), showing key modules, dependency chains, and editing guidance
- **Post-edit hooks** — template for running ruff/mypy/tsc after every file edit to catch breakage immediately

## Quick start

### 1. Install

```bash
git clone https://github.com/white-dots/agent_hub
cd agent_hub
pip install -e ".[dev]"
```

### 2. Configure Claude Code

Add to your `~/.claude.json`:

```json
{
  "mcpServers": {
    "agenthub": {
      "command": "python",
      "args": ["-m", "agenthub.mcp_server", "--project", "/path/to/your/project"],
      "env": {
        "PYTHONPATH": "/path/to/agent_hub/src"
      }
    }
  }
}
```

### 3. Restart Claude Code

The tools are now available. Claude will see `impact_check`, `affected_tests`, and `codebase_overview` in its tool list.

## Example: impact_check

```
Impact Analysis: src/models/user.py
Role: hub | Risk: HIGH
This is a hub module with 14 transitive dependents. Changes here ripple widely.

EXPORTED INTERFACE:
  class User
    .to_dict()
    .validate()
  class UserRole
  def format_user(user)
  MAX_USERS

DIRECT DEPENDENTS (5):
  src/auth/service.py
  src/api/routes.py
  src/api/admin.py
  src/utils/helpers.py
  src/workers/sync.py

TRANSITIVE DEPENDENTS (9 additional):
  src/auth/middleware.py
  src/api/v2/routes.py
  ...

AFFECTED TESTS (4):
  tests/test_auth.py
  tests/test_api.py
  tests/test_routes.py
  tests/test_sync.py
```

## Auto-generated repo map

Run `agenthub map /path/to/project` to generate a CLAUDE.md with:
- Key modules ranked by impact (number of dependents)
- Dependency chains between hub modules
- Directory layout with module counts
- Editing guidance for high-risk files

This content is also served as an MCP resource (`repo://map`) — automatically injected into every Claude Code session, including parallel subagents.

```bash
# Generate CLAUDE.md in the project root
agenthub map /path/to/project

# Print to stdout instead
agenthub map /path/to/project --stdout
```

## Post-edit hooks

Copy `templates/hooks.json` to your project's `.claude/hooks.json` to run type checkers after every file edit:

- Python files: runs `ruff check` and `mypy`
- TypeScript files: runs `tsc --noEmit`

This catches interface breakage immediately, before Claude moves on to the next file.

## Benchmark: with vs without impact analysis

We simulated 64 realistic code edits (function renames, signature changes, class renames) across synthetic codebases and measured how many downstream files break silently.

### Without impact analysis

| Codebase | Edits | Edits causing breakage | Files silently broken | Avg broken per edit |
|----------|-------|----------------------|----------------------|-------------------|
| Small (10 modules) | 6 | 4 (67%) | 16 | 2.7 |
| Medium (50 modules) | 15 | 7 (47%) | 87 | 5.8 |
| Large (200 modules) | 43 | 25 (58%) | 1,199 | 27.9 |

### With impact analysis

| Codebase | Breakage surfaced | Latency (avg) | Latency (p95) |
|----------|------------------|--------------|--------------|
| Small (10 modules) | 100% | 0.1ms | 0.2ms |
| Medium (50 modules) | 100% | 0.2ms | 0.3ms |
| Large (200 modules) | 100% | 0.3ms | 0.3ms |

**56% of hub file edits silently break downstream code. Impact analysis catches 100% of it — in under 1ms, with zero LLM cost.**

### Performance

| Codebase | Modules | Graph build | Blast radius (avg) | Detection |
|----------|---------|------------|-------------------|-----------|
| Small | 10 | 8ms | 7 files | 100% |
| Medium | 50 | 23ms | 35 files | 100% |
| Large | 200 | 84ms | 59 files | 100% |

Run the benchmarks yourself:

```bash
python benchmarks/bench_comparative.py  # with vs without comparison
python benchmarks/bench_impact.py       # detection accuracy & performance
```

## How it works

AgentHub builds a static import graph by parsing your source files:
- **Python**: AST parsing (`ast.parse`) for accurate import resolution
- **TypeScript/JavaScript**: regex-based import/export detection

The graph maps every module's imports and importers. When you ask "what breaks if I edit this file?", it walks the `imported_by` edges transitively to find every affected file — including test files.

No heuristics, no LLM calls, no network requests. Pure static analysis.

## Supported languages

- Python (`.py`) — full AST parsing, `__all__` support
- TypeScript (`.ts`, `.tsx`) — export detection, interface/type extraction
- JavaScript (`.js`, `.jsx`) — export detection

## Development

```bash
pip install -e ".[dev]"

# Run all tests
pytest

# Run impact analysis tests only
pytest tests/test_impact_graph.py tests/test_mcp_server.py tests/test_repo_map.py -v

# Run benchmark
python benchmarks/bench_impact.py

# Format
ruff format .
ruff check .
```

## License

MIT
