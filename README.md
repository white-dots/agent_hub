# AgentHub

**Lightweight agent orchestration for context-efficient LLM applications.**

Stop hitting context limits. Build specialized agents that know their domain.

## Quick Start with Claude Code

AgentHub integrates with Claude Code (terminal or VSCode) to give Claude specialized knowledge about your codebase.

### 1. Install

```bash
# Clone the repository
git clone https://github.com/white-dots/agent_hub
cd agenthub

# Install with all features
pip install -e ".[all]"
```

### 2. Build (one-time setup)

```bash
# Navigate to your project
cd /path/to/your/project

# Run build - this will:
# - Check for API keys (prompts if missing)
# - Analyze your codebase and create agents
# - Configure Claude Code's MCP server
agenthub build .
```

The build command will prompt you for:
1. **API Key**: Enter your `ANTHROPIC_API_KEY` if not already set
2. **MCP Setup**: Choose "1" to auto-configure Claude Code

### 3. Restart Claude Code

After running `agenthub build`, restart Claude Code to load the MCP server.

### 4. Use It

Now in Claude Code, you have access to AgentHub tools:

```
# Ask questions about your codebase
Use agenthub_query to ask: "How does the authentication system work?"

# List available agents
Use agenthub_list_agents to see all specialized agents

# See routing rules
Use agenthub_routing_rules to understand how queries are routed
```

## CLI Commands

```bash
# Build and configure agents (run once per project)
agenthub build /path/to/project

# Start the dashboard
agenthub up

# Start with QC analysis enabled
agenthub up --qc

# Watch for file changes
agenthub watch

# Check status
agenthub status

# Regenerate Tier B agents
agenthub restructure
```

### Build Options

```bash
agenthub build /path/to/project [options]

  --auto-mcp      Auto-configure MCP without prompting
  --skip-mcp      Skip MCP configuration
  --force         Continue even without API keys
  --no-tier-a     Skip Tier A (business) agent discovery
  --no-tier-b     Skip Tier B (code) agent generation
```

## Dashboard

Start the web dashboard to see agents, run queries, and monitor activity:

```bash
agenthub up --port 3001
```

Then open http://localhost:3001 in your browser.

## Two-Tier Agent System

| Tier | Created By | Purpose |
|------|-----------|---------|
| **A: Business** | You (manually) | Domain expertise, APIs, business logic |
| **B: Code** | AgentHub (auto) | Codebase navigation, file explanations |

Tier A agents are defined in `agents/*.py` files in your project.
Tier B agents are automatically generated from your codebase structure.

## Configuration

### API Keys

AgentHub needs at least one API key. You can:

1. Let `agenthub build` prompt you (saves to `.env`)
2. Create a `.env` file manually:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Set environment variable:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

### Config Files

- `~/.agenthub/config.json` - Stores current project path
- `~/.claude.json` - Claude Code MCP configuration (auto-updated by `agenthub build`)

## Programmatic Usage

### Auto-Agents for Existing Codebases

```python
from agenthub import AgentHub
from agenthub.auto import discover_all_agents

# Discover and create all agents
hub, summary = discover_all_agents("./my-project")
print(summary)

# Query your codebase
response = hub.run("How does user authentication work?")
print(response.content)
```

### Custom Agents (Tier A)

Create `agents/pricing.py` in your project:

```python
from agenthub import AgentSpec, BaseAgent

class PricingAgent(BaseAgent):
    def __init__(self, client):
        spec = AgentSpec(
            agent_id="pricing",
            name="Pricing Expert",
            description="Knows pricing strategy and margins",
            context_keywords=["price", "margin", "discount"],
        )
        super().__init__(spec, client)

    def build_context(self) -> str:
        return "... your pricing documentation ..."
```

## Requirements

- Python 3.11+
- Anthropic API key (or OpenAI API key)

### Optional Dependencies

```bash
pip install agenthub[dashboard]  # Dashboard (fastapi, uvicorn)
pip install agenthub[watch]      # File watching (watchdog)
pip install agenthub[openai]     # OpenAI support
pip install agenthub[all]        # Everything
```

## Troubleshooting

### MCP server not working

1. Make sure you ran `agenthub build` for your project
2. Restart Claude Code after build
3. Check `~/.claude.json` has the `agenthub` entry in `mcpServers`

### Agents not found

```bash
# Check what agents were discovered
agenthub status

# Rebuild if needed
agenthub restructure --force
```

### API key issues

```bash
# Build will prompt for missing keys
agenthub build .

# Or set manually
export ANTHROPIC_API_KEY=your-key-here
```

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/white-dots/agent_hub
cd agenthub
pip install -e ".[dev,all]"

# Run tests
pytest

# Format code
ruff format .
ruff check .
```

## License

MIT
