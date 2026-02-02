# AgentHub Auto: Auto-Agent Generation for Existing Codebases

> **Status:** Concept Spec  
> **Location:** `agenthub.auto` submodule  
> **Goal:** Point at any existing codebase → get intelligent code agents instantly

---

## 0. The Core Workflow

```python
# Developer has an existing Django/FastAPI/Flask/etc project
# They want AI assistance without manually setting up agents

from agenthub import AgentHub
from agenthub.auto import enable_auto_agents

hub = AgentHub()

# ONE LINE to get code agents for your entire project
agents = enable_auto_agents(hub, "./my-existing-project")

# Now they can query their codebase intelligently
hub.run("How does the payment flow work?")
hub.run("What does the UserSerializer class do?")
hub.run("Where is database connection configured?")
```

**Target users:** Developers with existing codebases who want AI coding assistance without setup overhead.

---

## 1. The Problem

Developers using AgentHub face a chicken-and-egg problem:

1. **Small projects** → Don't need agents, just use Claude directly
2. **Growing projects** → When should I create agents? What boundaries?
3. **Large projects** → Manual agent creation is tedious and error-prone

### The Insight

Code structure already tells us where natural boundaries exist:
- Folders = domains
- Files = responsibilities  
- Size = complexity

**Why not auto-generate agents from this structure?**

---

## 2. Two-Tier Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AgentHub                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TIER A: Business/Product Agents (Human-Defined)               │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│   │ Smartstore  │ │   Pricing   │ │  Analytics  │               │
│   │   Agent     │ │    Agent    │ │    Agent    │               │
│   │             │ │             │ │             │               │
│   │ • Naver API │ │ • Margins   │ │ • Metrics   │               │
│   │ • Orders    │ │ • Discounts │ │ • Reports   │               │
│   └─────────────┘ └─────────────┘ └─────────────┘               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TIER B: Code Agents (Auto-Generated)                          │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│   │  src/api/   │ │ src/models/ │ │  src/utils/ │               │
│   │   Agent     │ │   Agent     │ │    Agent    │               │
│   │             │ │             │ │             │               │
│   │ • 12 files  │ │ • 8 files   │ │ • 15 files  │               │
│   │ • 45KB      │ │ • 22KB      │ │ • 18KB      │               │
│   └─────────────┘ └─────────────┘ └─────────────┘               │
│         ▲                 ▲               ▲                     │
│         └─────────────────┴───────────────┘                     │
│                    Auto-generated when                          │
│                  thresholds exceeded                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tier A: Business/Product Agents

- **Created by:** Developers (you)
- **Purpose:** Domain expertise, business logic, external integrations
- **Context:** Curated - API docs, schemas, business rules
- **Lifetime:** Persistent, versioned with code
- **Examples:** SmartstoreAgent, PricingAgent, NaverAdAgent

### Tier B: Code Agents (Auto-Generated)

- **Created by:** AgentHub automatically
- **Purpose:** Navigate and explain codebase structure
- **Context:** Auto-extracted from file contents
- **Lifetime:** Ephemeral, regenerated on changes
- **Examples:** `src/api/` agent, `src/models/` agent

---

## 3. Auto-Generation Triggers

### 3.1 Threshold-Based Triggers

```python
class AutoAgentConfig(BaseModel):
    """Configuration for auto-agent generation"""
    
    # Size thresholds
    min_folder_size_kb: int = 50          # Folder must be > 50KB
    max_agent_context_kb: int = 100       # Split if agent context > 100KB
    
    # Count thresholds  
    min_files_per_folder: int = 5         # Folder must have > 5 files
    max_files_per_agent: int = 20         # Split if agent has > 20 files
    
    # Depth settings
    max_depth: int = 3                    # Don't go deeper than 3 levels
    
    # Ignore patterns
    ignore_patterns: list[str] = [
        "__pycache__", ".git", "node_modules",
        "*.pyc", ".venv", "dist", "build"
    ]
```

### 3.2 When Agents Get Created

```
Project: smartstore_ai/
├── src/                    (150KB total)
│   ├── api/                (45KB, 12 files) → Creates: src_api_agent
│   ├── models/             (22KB, 8 files)  → Creates: src_models_agent
│   ├── pipelines/          (38KB, 15 files) → Creates: src_pipelines_agent
│   └── utils/              (18KB, 6 files)  → Creates: src_utils_agent
├── tests/                  (30KB, 10 files) → Creates: tests_agent
└── docs/                   (5KB, 2 files)   → Too small, no agent
```

### 3.3 Hierarchical Splitting

When a folder exceeds thresholds, split recursively:

```
src/pipelines/             (120KB total - exceeds 100KB threshold!)
├── naver/                 (45KB) → src_pipelines_naver_agent
├── smartstore/            (50KB) → src_pipelines_smartstore_agent
└── analytics/             (25KB) → src_pipelines_analytics_agent
```

---

## 4. Core Components

### 4.1 CodebaseAnalyzer

```python
from pathlib import Path
from dataclasses import dataclass

@dataclass
class FolderStats:
    path: Path
    total_size_kb: float
    file_count: int
    subfolder_count: int
    languages: dict[str, int]  # {"python": 15, "sql": 3}
    
@dataclass  
class AgentBoundary:
    """Proposed agent boundary"""
    agent_id: str
    root_path: Path
    include_patterns: list[str]
    estimated_context_kb: float
    file_count: int


class CodebaseAnalyzer:
    """Analyzes codebase structure to determine agent boundaries"""
    
    def __init__(self, root_path: str, config: AutoAgentConfig):
        self.root = Path(root_path)
        self.config = config
    
    def analyze(self) -> list[FolderStats]:
        """Walk the codebase and collect statistics"""
        stats = []
        for folder in self._walk_folders():
            stats.append(self._analyze_folder(folder))
        return stats
    
    def propose_boundaries(self) -> list[AgentBoundary]:
        """Propose agent boundaries based on thresholds"""
        stats = self.analyze()
        boundaries = []
        
        for folder_stat in stats:
            if self._should_create_agent(folder_stat):
                # Check if needs splitting
                if self._should_split(folder_stat):
                    boundaries.extend(self._split_folder(folder_stat))
                else:
                    boundaries.append(self._create_boundary(folder_stat))
        
        return boundaries
    
    def _should_create_agent(self, stats: FolderStats) -> bool:
        """Determine if folder warrants its own agent"""
        return (
            stats.total_size_kb >= self.config.min_folder_size_kb and
            stats.file_count >= self.config.min_files_per_folder
        )
    
    def _should_split(self, stats: FolderStats) -> bool:
        """Determine if folder should be split into multiple agents"""
        return (
            stats.total_size_kb > self.config.max_agent_context_kb or
            stats.file_count > self.config.max_files_per_agent
        )
    
    def _walk_folders(self):
        """Generator yielding folders up to max_depth"""
        # Implementation
        pass
    
    def _analyze_folder(self, path: Path) -> FolderStats:
        """Collect statistics for a single folder"""
        # Implementation  
        pass
    
    def _split_folder(self, stats: FolderStats) -> list[AgentBoundary]:
        """Split large folder into multiple agent boundaries"""
        # Implementation
        pass
    
    def _create_boundary(self, stats: FolderStats) -> AgentBoundary:
        """Create agent boundary from folder stats"""
        return AgentBoundary(
            agent_id=self._generate_agent_id(stats.path),
            root_path=stats.path,
            include_patterns=["**/*.py", "**/*.sql"],
            estimated_context_kb=stats.total_size_kb,
            file_count=stats.file_count
        )
    
    def _generate_agent_id(self, path: Path) -> str:
        """Generate agent ID from path: src/api → src_api_agent"""
        parts = path.relative_to(self.root).parts
        return "_".join(parts) + "_agent"
```

### 4.2 AutoAgentFactory

```python
class AutoAgentFactory:
    """Creates agents from boundaries"""
    
    def __init__(self, client, config: AutoAgentConfig):
        self.client = client
        self.config = config
        self.context_builder = ContextBuilder()
    
    def create_agent(self, boundary: AgentBoundary) -> Agent:
        """Create an agent from a boundary definition"""
        
        spec = AgentSpec(
            agent_id=boundary.agent_id,
            name=self._generate_name(boundary),
            description=self._generate_description(boundary),
            context_paths=[str(boundary.root_path / p) for p in boundary.include_patterns],
            context_keywords=self._extract_keywords(boundary),
            estimated_tokens=int(boundary.estimated_context_kb * 400),  # ~400 tokens/KB
            system_prompt=self._generate_system_prompt(boundary),
            metadata={
                "auto_generated": True,
                "tier": "B",
                "root_path": str(boundary.root_path),
                "generated_at": datetime.now().isoformat()
            }
        )
        
        return CodeAgent(spec, self.client, str(boundary.root_path))
    
    def _generate_name(self, boundary: AgentBoundary) -> str:
        """Generate human-readable name: src/api → 'API Module Expert'"""
        folder_name = boundary.root_path.name
        return f"{folder_name.title()} Module Expert"
    
    def _generate_description(self, boundary: AgentBoundary) -> str:
        """Generate description from folder contents"""
        return f"Expert on {boundary.root_path} ({boundary.file_count} files, {boundary.estimated_context_kb:.0f}KB)"
    
    def _extract_keywords(self, boundary: AgentBoundary) -> list[str]:
        """Extract keywords from filenames and common terms"""
        keywords = []
        
        # Add folder names as keywords
        for part in boundary.root_path.parts:
            keywords.append(part.lower())
        
        # Scan files for common imports/class names
        # (simplified - real implementation would parse AST)
        for file in boundary.root_path.glob("**/*.py"):
            keywords.append(file.stem.lower())
        
        return list(set(keywords))[:20]  # Limit to 20
    
    def _generate_system_prompt(self, boundary: AgentBoundary) -> str:
        """Generate system prompt for code agent"""
        return f"""You are an expert on the {boundary.root_path} module.

You know:
- All {boundary.file_count} files in this module
- The patterns and conventions used
- How this module interacts with others

When answering:
- Reference specific files and line numbers
- Explain the "why" behind code decisions
- Suggest improvements when relevant

Your scope is LIMITED to {boundary.root_path}. If asked about code outside 
your module, say so and suggest which module might handle it."""
```

### 4.3 AutoAgentManager

```python
class AutoAgentManager:
    """Manages lifecycle of auto-generated agents"""
    
    def __init__(
        self, 
        hub: AgentHub,
        project_root: str,
        config: AutoAgentConfig = None
    ):
        self.hub = hub
        self.project_root = Path(project_root)
        self.config = config or AutoAgentConfig()
        self.analyzer = CodebaseAnalyzer(project_root, self.config)
        self.factory = AutoAgentFactory(hub.client, self.config)
        
        # Track auto-generated agents
        self._auto_agents: dict[str, Agent] = {}
        self._last_scan: datetime = None
    
    def scan_and_register(self) -> list[str]:
        """Scan codebase and register auto-generated agents.
        
        Returns:
            List of newly registered agent IDs
        """
        boundaries = self.analyzer.propose_boundaries()
        new_agents = []
        
        for boundary in boundaries:
            if boundary.agent_id not in self._auto_agents:
                agent = self.factory.create_agent(boundary)
                self.hub.register(agent)
                self._auto_agents[boundary.agent_id] = agent
                new_agents.append(boundary.agent_id)
        
        self._last_scan = datetime.now()
        return new_agents
    
    def refresh(self) -> tuple[list[str], list[str]]:
        """Re-scan and update agents. Returns (added, removed)."""
        current_boundaries = {
            b.agent_id: b for b in self.analyzer.propose_boundaries()
        }
        
        # Find new agents
        added = []
        for agent_id, boundary in current_boundaries.items():
            if agent_id not in self._auto_agents:
                agent = self.factory.create_agent(boundary)
                self.hub.register(agent)
                self._auto_agents[agent_id] = agent
                added.append(agent_id)
        
        # Find removed agents
        removed = []
        for agent_id in list(self._auto_agents.keys()):
            if agent_id not in current_boundaries:
                self.hub.unregister(agent_id)
                del self._auto_agents[agent_id]
                removed.append(agent_id)
        
        # Refresh context for existing agents
        for agent_id, agent in self._auto_agents.items():
            if agent_id in current_boundaries:
                agent.get_context(force_refresh=True)
        
        return added, removed
    
    def list_auto_agents(self) -> list[AgentSpec]:
        """List all auto-generated agents"""
        return [a.spec for a in self._auto_agents.values()]
    
    def get_coverage_report(self) -> dict:
        """Report on codebase coverage by auto-agents"""
        stats = self.analyzer.analyze()
        total_kb = sum(s.total_size_kb for s in stats)
        covered_kb = sum(
            s.total_size_kb for s in stats 
            if self._generate_agent_id(s.path) in self._auto_agents
        )
        
        return {
            "total_folders": len(stats),
            "covered_folders": len(self._auto_agents),
            "total_kb": total_kb,
            "covered_kb": covered_kb,
            "coverage_percent": (covered_kb / total_kb * 100) if total_kb > 0 else 0
        }
```

---

## 5. Integration with AgentHub

### 5.1 Extended Hub Interface

```python
class AgentHub:
    """Extended with auto-agent support"""
    
    def __init__(self, client=None, auto_agent_config=None):
        # ... existing init ...
        self._auto_manager: AutoAgentManager = None
    
    def enable_auto_agents(
        self, 
        project_root: str,
        config: AutoAgentConfig = None
    ) -> list[str]:
        """Enable automatic agent generation for a project.
        
        Args:
            project_root: Path to the project to analyze
            config: Auto-generation configuration
            
        Returns:
            List of auto-generated agent IDs
        """
        self._auto_manager = AutoAgentManager(
            hub=self,
            project_root=project_root,
            config=config
        )
        return self._auto_manager.scan_and_register()
    
    def refresh_auto_agents(self) -> tuple[list[str], list[str]]:
        """Refresh auto-generated agents after code changes."""
        if not self._auto_manager:
            raise ValueError("Auto-agents not enabled. Call enable_auto_agents first.")
        return self._auto_manager.refresh()
    
    def list_agents(self, tier: str = None) -> list[AgentSpec]:
        """List agents, optionally filtered by tier.
        
        Args:
            tier: "A" for business agents, "B" for auto-generated, None for all
        """
        all_agents = [a.spec for a in self._agents.values()]
        
        if tier is None:
            return all_agents
        elif tier == "A":
            return [a for a in all_agents if not a.metadata.get("auto_generated")]
        elif tier == "B":
            return [a for a in all_agents if a.metadata.get("auto_generated")]
        else:
            raise ValueError(f"Unknown tier: {tier}. Use 'A', 'B', or None.")
```

### 5.2 Usage Example

```python
import anthropic
from agenthub import AgentHub, AutoAgentConfig
from agenthub.agents import SmartstoreAgent  # Your Tier A agent

# Initialize
client = anthropic.Anthropic()
hub = AgentHub(client)

# Register Tier A (business) agents manually
hub.register(SmartstoreAgent(client))
hub.register(PricingAgent(client))

# Enable Tier B (code) agents automatically
auto_config = AutoAgentConfig(
    min_folder_size_kb=30,
    max_agent_context_kb=80,
    ignore_patterns=["__pycache__", ".git", "tests"]
)

auto_agents = hub.enable_auto_agents(
    project_root="./smartstore_ai",
    config=auto_config
)

print(f"Auto-generated {len(auto_agents)} code agents:")
for agent_id in auto_agents:
    print(f"  - {agent_id}")

# Use - routing works across both tiers
response = hub.run("How does the Naver API authentication work?")
# → Routes to SmartstoreAgent (Tier A)

response = hub.run("What does the data_loader.py file do?")
# → Routes to src_pipelines_agent (Tier B)

# After code changes, refresh
added, removed = hub.refresh_auto_agents()
print(f"Added: {added}, Removed: {removed}")
```

---

## 6. Smart Routing Across Tiers

### 6.1 Tier-Aware Router

```python
class TierAwareRouter:
    """Routes queries considering both tier A and B agents"""
    
    def __init__(self, client, prefer_tier_a: bool = True):
        self.client = client
        self.prefer_tier_a = prefer_tier_a
    
    def route(self, query: str, agents: list[AgentSpec]) -> str:
        """Route with tier preference.
        
        Strategy:
        1. If query is clearly business/product → Tier A
        2. If query mentions specific files/folders → Tier B
        3. If ambiguous and prefer_tier_a → Try Tier A first
        """
        tier_a = [a for a in agents if not a.metadata.get("auto_generated")]
        tier_b = [a for a in agents if a.metadata.get("auto_generated")]
        
        # Check for file/path references → Tier B
        if self._mentions_code_paths(query):
            return self._route_to_tier(query, tier_b) or self._route_to_tier(query, tier_a)
        
        # Check for business terms → Tier A
        if self._mentions_business_terms(query):
            return self._route_to_tier(query, tier_a) or self._route_to_tier(query, tier_b)
        
        # Ambiguous - use preference
        if self.prefer_tier_a:
            return self._route_to_tier(query, tier_a) or self._route_to_tier(query, tier_b)
        else:
            return self._route_to_tier(query, tier_b) or self._route_to_tier(query, tier_a)
    
    def _mentions_code_paths(self, query: str) -> bool:
        """Check if query mentions file paths, extensions, etc."""
        code_indicators = [
            ".py", ".sql", ".js", "file", "folder", "module",
            "function", "class", "import", "src/", "tests/"
        ]
        query_lower = query.lower()
        return any(ind in query_lower for ind in code_indicators)
    
    def _mentions_business_terms(self, query: str) -> bool:
        """Check if query is business/product focused"""
        business_indicators = [
            "pricing", "customer", "order", "revenue", "campaign",
            "ad", "광고", "스마트스토어", "naver", "strategy"
        ]
        query_lower = query.lower()
        return any(ind in query_lower for ind in business_indicators)
    
    def _route_to_tier(self, query: str, agents: list[AgentSpec]) -> str | None:
        """Route within a specific tier using keyword matching"""
        if not agents:
            return None
        
        query_lower = query.lower()
        for agent in agents:
            for keyword in agent.context_keywords:
                if keyword.lower() in query_lower:
                    return agent.agent_id
        
        return agents[0].agent_id if agents else None
```

---

## 7. File Watcher Integration (Optional)

For development environments, auto-refresh when files change:

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

class CodebaseWatcher(FileSystemEventHandler):
    """Watches for file changes and triggers agent refresh"""
    
    def __init__(self, auto_manager: AutoAgentManager, debounce_seconds: float = 2.0):
        self.auto_manager = auto_manager
        self.debounce_seconds = debounce_seconds
        self._timer: threading.Timer = None
    
    def on_any_event(self, event):
        # Ignore non-code files
        if not event.src_path.endswith(('.py', '.sql', '.js', '.ts')):
            return
        
        # Debounce - wait for changes to settle
        if self._timer:
            self._timer.cancel()
        
        self._timer = threading.Timer(
            self.debounce_seconds, 
            self._trigger_refresh
        )
        self._timer.start()
    
    def _trigger_refresh(self):
        added, removed = self.auto_manager.refresh()
        if added or removed:
            print(f"[AutoAgent] Refreshed: +{len(added)} -{len(removed)}")


def watch_and_refresh(hub: AgentHub, project_root: str):
    """Start watching project for changes"""
    if not hub._auto_manager:
        raise ValueError("Enable auto-agents first")
    
    handler = CodebaseWatcher(hub._auto_manager)
    observer = Observer()
    observer.schedule(handler, project_root, recursive=True)
    observer.start()
    
    return observer  # Call observer.stop() to stop watching
```

---

## 8. Visualization / Debug Tools

### 8.1 Coverage Map

```python
def print_coverage_map(hub: AgentHub, project_root: str):
    """Print visual map of agent coverage"""
    
    if not hub._auto_manager:
        print("Auto-agents not enabled")
        return
    
    report = hub._auto_manager.get_coverage_report()
    
    print(f"\n📊 Agent Coverage Report")
    print(f"{'='*50}")
    print(f"Total: {report['total_folders']} folders, {report['total_kb']:.0f}KB")
    print(f"Covered: {report['covered_folders']} folders, {report['covered_kb']:.0f}KB")
    print(f"Coverage: {report['coverage_percent']:.1f}%")
    print()
    
    # List agents by tier
    tier_a = hub.list_agents(tier="A")
    tier_b = hub.list_agents(tier="B")
    
    print(f"🅰️  Tier A (Business): {len(tier_a)} agents")
    for agent in tier_a:
        print(f"   • {agent.agent_id}: {agent.description}")
    
    print()
    print(f"🅱️  Tier B (Auto-Code): {len(tier_b)} agents")
    for agent in tier_b:
        print(f"   • {agent.agent_id}: {agent.description}")


# Usage:
# print_coverage_map(hub, "./smartstore_ai")
```

### 8.2 Example Output

```
📊 Agent Coverage Report
==================================================
Total: 12 folders, 245KB
Covered: 8 folders, 198KB
Coverage: 80.8%

🅰️  Tier A (Business): 2 agents
   • smartstore_agent: Naver Smartstore API and e-commerce expert
   • pricing_agent: Pricing optimization and margin analysis

🅱️  Tier B (Auto-Code): 6 agents
   • src_api_agent: Expert on src/api (12 files, 45KB)
   • src_models_agent: Expert on src/models (8 files, 22KB)
   • src_pipelines_naver_agent: Expert on src/pipelines/naver (6 files, 28KB)
   • src_pipelines_smartstore_agent: Expert on src/pipelines/smartstore (5 files, 32KB)
   • src_utils_agent: Expert on src/utils (15 files, 18KB)
   • tests_agent: Expert on tests (10 files, 30KB)
```

---

## 9. Configuration Presets

```python
class AutoAgentPresets:
    """Pre-configured settings for common scenarios"""
    
    @staticmethod
    def small_project() -> AutoAgentConfig:
        """For projects < 500KB - minimal splitting"""
        return AutoAgentConfig(
            min_folder_size_kb=100,
            max_agent_context_kb=200,
            min_files_per_folder=10,
            max_depth=2
        )
    
    @staticmethod
    def medium_project() -> AutoAgentConfig:
        """For projects 500KB - 5MB - balanced"""
        return AutoAgentConfig(
            min_folder_size_kb=50,
            max_agent_context_kb=100,
            min_files_per_folder=5,
            max_depth=3
        )
    
    @staticmethod
    def large_project() -> AutoAgentConfig:
        """For projects > 5MB - aggressive splitting"""
        return AutoAgentConfig(
            min_folder_size_kb=30,
            max_agent_context_kb=60,
            min_files_per_folder=3,
            max_depth=4
        )
    
    @staticmethod
    def monorepo() -> AutoAgentConfig:
        """For monorepos - treat each package as boundary"""
        return AutoAgentConfig(
            min_folder_size_kb=20,
            max_agent_context_kb=80,
            min_files_per_folder=3,
            max_depth=5,
            # Monorepo-specific ignores
            ignore_patterns=[
                "__pycache__", ".git", "node_modules",
                "dist", "build", ".venv", "coverage"
            ]
        )


# Usage:
hub.enable_auto_agents(
    project_root="./my-monorepo",
    config=AutoAgentPresets.monorepo()
)
```

---

## 10. Open Questions

### To Decide Before Implementation

1. **Granularity heuristics** - Are KB/file count the right metrics? Consider:
   - Cyclomatic complexity
   - Import graph density
   - Git change frequency

2. **Agent naming** - `src_api_agent` is functional but ugly. Alternatives:
   - Use folder's `__doc__` if exists
   - LLM-generated names from file contents
   - User-provided aliases

3. **Cross-agent references** - When `src_api_agent` needs to reference code in `src_models_agent`:
   - Include import stubs in context?
   - Agent handoff suggestions?
   - Shared "interface" context?

4. **Caching strategy** - Auto-agents could have stale context:
   - Hash-based invalidation?
   - Time-based refresh?
   - Git-hook integration?

5. **User overrides** - Let users:
   - Force certain folders to be one agent
   - Exclude folders from auto-generation
   - Customize auto-agent system prompts

---

## 11. Implementation Priority

### Phase 1: Basic Auto-Generation
- [ ] CodebaseAnalyzer with size/count thresholds
- [ ] AutoAgentFactory creating basic agents
- [ ] hub.enable_auto_agents() API

### Phase 2: Smart Routing
- [ ] TierAwareRouter
- [ ] Keyword extraction from code
- [ ] Path-based routing hints

### Phase 3: Lifecycle Management
- [ ] Refresh mechanism
- [ ] File watcher integration
- [ ] Coverage reporting

### Phase 4: Polish
- [ ] Configuration presets
- [ ] User overrides
- [ ] Documentation

---

*This is the `agenthub.auto` submodule. It ships with the core library - no separate install needed. Users get this automatically with `pip install agenthub`.*
