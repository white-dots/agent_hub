# Parallel Sessions Implementation Plan

This document translates the business spec (`agenthub_parallel_sessions_spec_v2.md`) into a concrete code plan, mapping each feature to the existing AgentHub architecture.

---

## Overview: What We're Building

A system that enables **multiple Claude Code sessions to work simultaneously** on separate git branches when a user submits a multi-part request. The system:

1. **Decomposes** requests into discrete implementation tasks
2. **Analyzes** conflict risk using ImportGraph + domain agents
3. **Orchestrates** parallel Claude Code sessions on separate git branches
4. **Merges** results with domain-agent-assisted conflict resolution
5. **Escalates** to the CEO (human) when agents can't resolve conflicts

---

## Integration Points with Existing Architecture

| New Component | Integrates With | How |
|---------------|-----------------|-----|
| `SubAgentManager` | `AutoAgentManager` | Extends to create hierarchical sub-agents |
| `TaskDecomposer` | `QueryDecomposer` | Similar pattern but outputs `ImplementationTask` not sub-questions |
| `ConflictRiskAnalyzer` | `ImportGraph` | Uses graph for static file/import overlap analysis |
| `ConflictRiskAnalyzer` | `BaseAgent.run()` | Consults domain agents for semantic conflict detection |
| `BranchOrchestrator` | `DAGTeamExecutor` | Similar parallel execution pattern but with git worktrees |
| `BranchOrchestrator` | Claude Code Agent Teams | Preferred execution backend for parallel sessions |
| `AgentTeamsAdapter` | Agent Teams inbox API | Real-time boundary crossing negotiation between teammates |
| `MergeCoordinator` | `BaseAgent.run()` | Uses domain agents for conflict resolution proposals |
| `ParallelSessionManager` | `AgentHub` | Top-level orchestrator, registered with hub |
| MCP Tools | `mcp_server.py` | New tools added to existing MCP server |

---

## Phase 0: Sub-Tier B Agents

**Goal**: Enable Tier B agents to subdivide into focused sub-agents when their domain is too large.

### 0.1 Data Model Extensions

**File**: `src/agenthub/models.py`

```python
# Add to AgentSpec (existing class)
@dataclass
class AgentSpec:
    # ... existing fields ...

    # NEW: Hierarchy fields
    parent_agent_id: Optional[str] = None       # None for top-level Tier B
    children_ids: list[str] = field(default_factory=list)
    hierarchy_level: int = 0                     # 0 = Tier B (team lead), 1+ = sub-agent
    is_team_lead: bool = False                   # True when agent has children


# NEW: Add to models.py
@dataclass
class SubAgentBoundary:
    """Proposed subdivision of a Tier B agent."""
    parent_agent_id: str
    sub_agent_id: str
    root_path: Path
    include_patterns: list[str]         # e.g., ["api/**/*.py"]
    estimated_context_kb: float
    file_count: int
    role_description: str               # Auto-generated
    key_modules: list[str]              # Central files within sub-domain
    interfaces_with: list[str]          # Other sub-agent IDs it imports from
```

### 0.2 SubAgentPolicy

**File**: `src/agenthub/auto/sub_agent_policy.py` (NEW)

```python
@dataclass
class SubAgentPolicy:
    """Determines when a Tier B agent should be subdivided."""

    # Thresholds — only subdivide when BOTH conditions met
    min_files_to_split: int = 60
    min_subdirs_to_split: int = 3

    # Stop conditions
    min_files_per_sub: int = 10
    max_sub_agents: int = 6

    # Context efficiency
    context_utilization_threshold: float = 0.8

    def should_subdivide(self, agent: BaseAgent, graph: ImportGraph) -> bool:
        """
        Check if agent warrants subdivision.

        Conditions (ALL must be true):
        1. Agent covers > min_files_to_split files
        2. Agent's context_paths span 3+ distinct subdirectories
        3. Import graph shows distinct sub-clusters within domain
        4. Agent's estimated context exceeds utilization threshold
        """

    def propose_subdivisions(
        self, agent: BaseAgent, graph: ImportGraph
    ) -> list[SubAgentBoundary]:
        """
        Propose how to split a Tier B agent.

        Strategy:
        1. Get all files in agent's context_paths
        2. Build sub-graph restricted to these files
        3. Cluster sub-graph (reuse ImportGraph.get_clusters())
        4. Each sub-cluster → SubAgentBoundary
        5. Identify inter-sub-agent interfaces
        6. Generate role descriptions

        Falls back to directory-based splitting if import graph too sparse.
        """
```

### 0.3 SubAgentManager

**File**: `src/agenthub/auto/sub_agent_manager.py` (NEW)

```python
class SubAgentManager:
    """Manages sub-Tier B agent lifecycle."""

    def __init__(
        self,
        auto_manager: AutoAgentManager,
        import_graph: ImportGraph,
        policy: SubAgentPolicy = None,
    ):
        self._auto_manager = auto_manager
        self._graph = import_graph
        self._policy = policy or SubAgentPolicy()
        self._sub_agents: dict[str, dict[str, BaseAgent]] = {}
        # parent_agent_id → {sub_agent_id: agent}

    def evaluate_and_subdivide(self) -> dict[str, list[SubAgentBoundary]]:
        """
        Scan all Tier B agents and subdivide where needed.

        Flow:
        1. For each Tier B agent:
           a. Check policy.should_subdivide()
           b. If yes: propose_subdivisions()
           c. Create sub-agents from boundaries
           d. Update parent's children_ids and is_team_lead
           e. Register sub-agents with AutoAgentManager
        2. Return report
        """

    def get_team(self, parent_agent_id: str) -> list[BaseAgent]:
        """Get all sub-agents for a team lead."""

    def get_team_lead(self, sub_agent_id: str) -> Optional[BaseAgent]:
        """Get the team lead for a sub-agent."""

    def route_to_sub_agent(
        self, parent_agent_id: str, file_path: str
    ) -> Optional[BaseAgent]:
        """
        Route a file to the most specific owning sub-agent.
        Used by MergeCoordinator._get_owning_agent().
        """

    def team_query(
        self,
        parent_agent_id: str,
        query: str,
        delegate: bool = True,
    ) -> AgentResponse:
        """
        Query the team.

        delegate=True: Team lead delegates to relevant sub-agent(s)
        delegate=False: Only team lead answers (faster, cheaper)
        """
```

### 0.4 Integration with AutoAgentManager

**File**: `src/agenthub/auto/manager.py` (MODIFY)

```python
class AutoAgentManager:
    # ... existing code ...

    # NEW: Add sub-agent support
    def __init__(self, ...):
        # ... existing ...
        self._sub_agent_manager: Optional[SubAgentManager] = None

    def enable_sub_agents(
        self,
        import_graph: ImportGraph,
        policy: SubAgentPolicy = None
    ) -> SubAgentManager:
        """Enable sub-agent subdivision for large Tier B agents."""
        self._sub_agent_manager = SubAgentManager(self, import_graph, policy)
        self._sub_agent_manager.evaluate_and_subdivide()
        return self._sub_agent_manager

    def get_most_specific_agent(self, file_path: str) -> Optional[BaseAgent]:
        """
        Get the most specific agent owning a file.
        Checks sub-agents first, then Tier B agents.
        """
        if self._sub_agent_manager:
            # Check if any Tier B agent owns this file
            for agent_id, agent in self._agents.items():
                if self._file_in_agent_scope(file_path, agent):
                    # Try to route to sub-agent
                    sub = self._sub_agent_manager.route_to_sub_agent(agent_id, file_path)
                    if sub:
                        return sub
                    return agent
        # Fall back to regular routing
        return self._get_agent_for_file(file_path)
```

### 0.5 Tests for Phase 0

**File**: `tests/test_sub_agents.py` (NEW)

```python
def test_should_subdivide_threshold():
    """Agent with 60+ files and 3+ subdirs should subdivide."""

def test_should_not_subdivide_small_agent():
    """Agent with <60 files should not subdivide."""

def test_propose_subdivisions_uses_import_graph():
    """Subdivisions should follow import graph clusters."""

def test_propose_subdivisions_fallback_to_directory():
    """Falls back to directory-based when import graph sparse."""

def test_route_to_sub_agent_precision():
    """Most specific sub-agent should be returned."""

def test_team_query_delegation():
    """Team lead should delegate to correct sub-agent."""

def test_team_query_no_delegation():
    """delegate=False should use team lead only."""
```

---

## Phase 1: Foundation — Task Decomposition & Conflict Analysis

**Goal**: Decompose requests into tasks and analyze parallelization safety.

### 1.1 Data Models

**File**: `src/agenthub/parallel/models.py` (NEW)

```python
from dataclasses import dataclass, field
from typing import Literal, Optional
from enum import Enum

class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class OverlapType(str, Enum):
    DIRECT = "direct"              # Same file modified
    SHARED_IMPORT = "shared_import"  # Transitive import overlap
    SHARED_TYPE = "shared_type"    # Both import same hub module
    SHARED_CONFIG = "shared_config"  # Shared config/env files

@dataclass
class ImplementationTask:
    """A discrete unit of work that produces code changes."""
    task_id: str
    description: str
    estimated_files: list[str]
    estimated_new_files: list[str]
    domain_agents: list[str]
    complexity: Literal["trivial", "moderate", "complex"]
    estimated_tokens: int
    depends_on: list[str] = field(default_factory=list)

@dataclass
class DecompositionResult:
    """Result of decomposing a user request into tasks."""
    tasks: list[ImplementationTask]
    original_request: str
    appears_simple: bool
    actual_complexity: Literal["single", "multi_independent", "multi_dependent", "multi_mixed"]
    decomposition_reasoning: str
    ceo_briefing: Optional[str]  # Message when hidden complexity detected
    tokens_used: int

@dataclass
class DomainClaim:
    """A domain agent's claim that a request touches its domain."""
    agent_id: str
    agent_name: str
    is_involved: bool
    description: str              # What the agent needs to do
    estimated_files: list[str]
    confidence: float             # "How sure am I that this request touches my domain?"
                                  # Threshold: claims with confidence < 0.3 are filtered out
                                  # in _survey_domains(). NOT used for risk decisions.

@dataclass
class FileOverlap:
    """Detected overlap between tasks."""
    file_path: str
    tasks_touching: list[str]
    overlap_type: OverlapType
    risk_level: RiskLevel

@dataclass
class AgentConflictAssessment:
    """A domain agent's assessment of conflict between two tasks."""
    agent_id: str
    agent_name: str
    task_pair: tuple[str, str]
    has_concern: bool
    concern_description: str
    severity: RiskLevel
    affected_files: list[str]
    tokens_used: int

@dataclass
class ParallelizationPlan:
    """Plan for executing tasks."""
    parallel_groups: list[list[str]]  # Groups of task_ids safe to run together
    sequential_order: list[str]       # Order for sequential tasks (when HIGH risk)
    file_overlaps: list[FileOverlap]
    agent_assessments: list[AgentConflictAssessment]
    overall_risk: RiskLevel
    confidence: float              # PM's overall confidence in the plan's safety.
                                   # NOT used for auto-fallback (no silent fallbacks).
                                   # Displayed to CEO as context for their shot-call.
    reasoning: str
    pm_recommendation: Literal["parallel", "sequential"]
    estimated_speedup: float
    estimated_total_tokens: int
```

### 1.2 TaskDecomposer

**File**: `src/agenthub/parallel/decomposer.py` (NEW)

```python
class TaskDecomposer:
    """
    The Project Manager — breaks requests into implementation tasks.

    Uses two-pass decomposition:
    1. Domain Survey: Ask each agent if the request touches their domain
    2. Structured Decomposition: LLM decomposes with full domain context
    """

    def __init__(
        self,
        client: Anthropic,
        hub: AgentHub,
        import_graph: ImportGraph,
    ):
        self._client = client
        self._hub = hub
        self._graph = import_graph

    def decompose(self, request: str) -> DecompositionResult:
        """
        Decompose a user request into implementation tasks.

        Flow:
        1. Survey all domain agents (Tier A + Tier B)
        2. Collect domain claims
        3. Build structured prompt with claims
        4. LLM generates task breakdown
        5. Generate CEO briefing if hidden complexity detected
        """

    def _survey_domains(self, request: str) -> list[DomainClaim]:
        """
        Ask each domain agent: "Does this request touch your domain?"

        Uses Haiku for speed. Returns claims from all agents with score > 0.3.
        """

    def _build_decomposition_prompt(
        self,
        request: str,
        claims: list[DomainClaim]
    ) -> str:
        """
        Build the LLM prompt for structured decomposition.

        Includes:
        - Original request
        - Domain survey results
        - Codebase domains (from AgentHub)
        - Module dependency summary (from ImportGraph)
        - Business context (from Tier A agents)
        """

    def _generate_ceo_briefing(
        self,
        request: str,
        result: DecompositionResult,
    ) -> Optional[str]:
        """
        Generate a briefing when complexity is hidden.

        Only generated when:
        - appears_simple=True but actual_complexity != "single"
        - Multiple domains are involved
        """
```

### 1.3 ConflictRiskAnalyzer

**File**: `src/agenthub/parallel/analyzer.py` (NEW)

```python
class ConflictRiskAnalyzer:
    """
    Analyzes conflict risk between tasks.

    Four-phase analysis:
    A. Static file overlap (fast, free)
    B. Import dependency overlap (uses ImportGraph)
    C. Tier B agent consultation (semantic)
    D. Tier A business review (business impact)
    """

    def __init__(
        self,
        import_graph: ImportGraph,
        hub: AgentHub,
        sub_agent_manager: Optional[SubAgentManager] = None,
    ):
        self._graph = import_graph
        self._hub = hub
        self._sub_manager = sub_agent_manager

    def analyze(
        self,
        tasks: list[ImplementationTask],
        consult_agents: bool = True,
    ) -> ParallelizationPlan:
        """
        Full analysis pipeline.

        Returns ParallelizationPlan with:
        - parallel_groups: Tasks safe to run together
        - sequential_order: Recommended order if sequential
        - overall_risk: NONE | LOW | MEDIUM | HIGH
        - pm_recommendation: "parallel" or "sequential"
        """

    # === Phase A: Static Analysis ===

    def _check_direct_overlap(
        self, tasks: list[ImplementationTask]
    ) -> list[FileOverlap]:
        """Check if any two tasks modify the same file."""

    def _check_import_overlap(
        self, tasks: list[ImplementationTask]
    ) -> list[FileOverlap]:
        """Check for transitive import overlaps (depth=2)."""

    def _check_hub_overlap(
        self, tasks: list[ImplementationTask]
    ) -> list[FileOverlap]:
        """Check if tasks share hub modules (high in-degree)."""

    def _check_model_overlap(
        self, tasks: list[ImplementationTask]
    ) -> list[FileOverlap]:
        """Check for shared database models, types, schemas."""

    # === Phase B: Tier B Agent Consultation ===

    def _consult_tier_b_agents(
        self,
        tasks: list[ImplementationTask],
        static_overlaps: list[FileOverlap],
    ) -> list[AgentConflictAssessment]:
        """
        Ask domain agents about conflicts.

        Routes to sub-agents when available for more precise assessment.
        Uses Haiku for speed.
        """

    def _get_owning_agent(self, file_path: str) -> Optional[BaseAgent]:
        """
        Get the most specific agent owning a file.
        Sub-agent > team lead > Tier B.
        """

    # === Phase C: Tier A Business Review ===

    def _consult_tier_a_agents(
        self,
        tasks: list[ImplementationTask],
        static_risk: RiskLevel,
        tier_b_assessments: list[AgentConflictAssessment],
    ) -> list[AgentConflictAssessment]:
        """
        Ask Tier A agents for business impact assessment.

        Only runs if Tier A agents are registered.
        Can upgrade risk even if static analysis shows NONE.
        """

    # === Phase D: Plan Building ===

    def _build_plan(
        self,
        tasks: list[ImplementationTask],
        overlaps: list[FileOverlap],
        tier_b: list[AgentConflictAssessment],
        tier_a: list[AgentConflictAssessment],
    ) -> ParallelizationPlan:
        """
        Build the parallelization plan.

        Risk aggregation:
        - Take max(static_risk, tier_b_risk, tier_a_risk)
        - Agents can only UPGRADE risk, never downgrade

        Plan building:
        - HIGH risk → sequential_order + pm_recommendation="sequential"
        - MEDIUM risk → parallel_groups but ask CEO
        - LOW/NONE → parallel_groups, auto-proceed
        """

    def _compute_sequential_order(
        self, tasks: list[ImplementationTask]
    ) -> list[str]:
        """
        Compute optimal sequential order respecting dependencies.
        Uses topological sort on task dependencies.
        """
```

### 1.4 MCP Tool: agenthub_analyze_parallelism

**File**: `src/agenthub/mcp_server.py` (MODIFY)

```python
# Add to existing MCP tools

@mcp.tool()
async def agenthub_analyze_parallelism(request: str) -> str:
    """
    Analyze whether a multi-part request can be safely parallelized.

    Returns task decomposition, file overlap analysis, and estimated speedup
    WITHOUT executing anything.
    """
    hub = get_hub()

    # Decompose
    decomposer = TaskDecomposer(hub._client, hub, hub._import_graph)
    decomposition = decomposer.decompose(request)

    # Analyze
    analyzer = ConflictRiskAnalyzer(
        hub._import_graph,
        hub,
        hub._sub_agent_manager
    )
    plan = analyzer.analyze(decomposition.tasks)

    # Format response
    return format_analysis_report(decomposition, plan)
```

### 1.5 Tests for Phase 1

**File**: `tests/test_decomposer.py` (NEW)

```python
def test_decompose_simple_request():
    """Single-task request should return single task."""

def test_decompose_multi_part_request():
    """Multi-part request should return multiple tasks."""

def test_domain_survey_catches_hidden_complexity():
    """'Add Excel upload button' should trigger 4+ domain claims."""

def test_ceo_briefing_generated_for_hidden_complexity():
    """CEO briefing should be generated when appears_simple but complex."""

def test_task_dependencies_detected():
    """depends_on should be populated for dependent tasks."""
```

**File**: `tests/test_analyzer.py` (NEW)

```python
def test_direct_overlap_high_risk():
    """Tasks modifying same file should be HIGH risk."""

def test_import_overlap_medium_risk():
    """Tasks with transitive import overlap should be MEDIUM risk."""

def test_no_overlap_safe():
    """Tasks with no overlap should be SAFE."""

def test_agent_can_upgrade_risk():
    """Agent concern should upgrade NONE → MEDIUM."""

def test_agent_cannot_downgrade_risk():
    """Agent 'no concern' should not downgrade HIGH → MEDIUM."""

def test_tier_a_business_review():
    """Tier A agent should flag business-level conflicts."""

def test_sequential_order_respects_dependencies():
    """Sequential order should respect task dependencies."""
```

---

## Phase 2: Execution — Branch Orchestration

**Goal**: Create git branches and spawn parallel Claude Code sessions.

**Execution Backend**: Supports two backends — CLI mode (`claude --print`) and Agent Teams
mode (full Claude Code teammates with inter-agent messaging). Agent Teams is the preferred
backend when available, with CLI as the stable fallback.

### 2.1 Data Models

**File**: `src/agenthub/parallel/models.py` (ADD)

```python
@dataclass
class SessionSpec:
    """Specification for a Claude Code session."""
    task: ImplementationTask
    branch_name: str
    worktree_path: str            # git worktree path (NOT checkout — see below)
    scoped_files: list[str]
    scoped_dirs: list[str]
    context_from_agents: list[str]
    prompt: str
    timeout_seconds: int

@dataclass
class SessionResult:
    """Result from a Claude Code session."""
    task_id: str
    branch_name: str
    success: bool
    files_changed: list[str]
    files_created: list[str]
    stdout: str
    tokens_used: int
    time_seconds: float
    test_results: Optional[dict]
    error: Optional[str]
    boundary_crossings: list[str]  # Parsed [BOUNDARY_CROSSING] tags
    execution_backend: Literal["cli", "agent_teams"]
```

### 2.2 BranchOrchestrator

**File**: `src/agenthub/parallel/orchestrator.py` (NEW)

**IMPORTANT — git worktree, not git checkout:**
`git checkout` mutates the working directory globally. If two parallel sessions
both `git checkout` different branches, they race on the same `.git` directory and
corrupt each other. `git worktree add` creates separate working directories that
share the same `.git` object store but have independent checkouts. This is the
only correct way to run truly parallel git-based sessions.

```python
import asyncio
import subprocess
from pathlib import Path
from typing import Literal

class BranchOrchestrator:
    """
    Manages git branches and spawns parallel Claude Code sessions.

    Uses git worktrees for filesystem isolation between parallel sessions.
    Supports two execution backends:
    - "cli": claude --print (one-shot, non-interactive, stable)
    - "agent_teams": Claude Code Agent Teams (full instances, inter-agent
      messaging, real-time boundary crossing negotiation)
    """

    def __init__(
        self,
        project_root: str,
        hub: AgentHub,
        max_parallel: int = 3,
        claude_model: str = "claude-sonnet-4-20250514",
        session_timeout: int = 300,
        execution_backend: Literal["cli", "agent_teams"] = "cli",
    ):
        self._root = Path(project_root)
        self._hub = hub
        self._max_parallel = max_parallel
        self._model = claude_model
        self._timeout = session_timeout
        self._backend = execution_backend
        self._base_branch: str = ""
        self._worktrees: list[Path] = []  # Track for cleanup

    def execute_plan(
        self,
        plan: ParallelizationPlan,
        tasks: list[ImplementationTask],
    ) -> list[SessionResult]:
        """
        Execute the parallelization plan.

        Flow:
        1. Capture base state (current branch/commit)
        2. Ensure working tree is clean
        3. For each parallel group:
           a. Create worktrees with branches from base
           b. Spawn sessions in parallel (via selected backend)
           c. Collect results
        4. For sequential tasks:
           a. Run one at a time in temporary worktree
           b. Merge into base before next
        5. Cleanup worktrees
        6. Return all results
        """

    def _capture_base_state(self) -> str:
        """Record current branch and commit SHA."""

    def _ensure_clean_tree(self) -> None:
        """Fail if working tree has uncommitted changes."""

    def _create_worktree(self, task_id: str) -> tuple[str, str]:
        """
        Create isolated working directory for a parallel session.

            git worktree add .worktrees/parallel/{task_id} -b parallel/{task_id}

        Returns (branch_name, worktree_path).
        Each worktree is a full working directory with its own checkout,
        sharing the .git object store with the main repo.
        """

    def _cleanup_worktrees(self) -> None:
        """
        Remove all parallel worktrees after execution.

            git worktree remove .worktrees/parallel/{task_id}
            git branch -D parallel/{task_id}  # only on rollback
        """

    def _build_scoped_prompt(
        self,
        task: ImplementationTask,
        other_tasks: list[ImplementationTask],
    ) -> str:
        """
        Build a Claude Code prompt with:
        - Task description
        - Scoped files (YOUR TEAM'S FILES)
        - Other teams' files (do NOT modify)
        - Tier B context from team lead + sub-agents
        - Tier A business context
        - Boundary crossing instructions
        """

    # === CLI Backend ===

    def _spawn_session_cli(self, spec: SessionSpec) -> SessionResult:
        """
        Spawn a Claude Code session via CLI.

            cd {worktree_path}
            claude --print "{prompt}" --output-format json --model {model}

        Uses asyncio.create_subprocess_exec (I/O-bound, not CPU-bound —
        each session is waiting on Claude API, so asyncio is more efficient
        than ProcessPoolExecutor).

        Returns SessionResult with files changed, tokens, timing.
        """

    async def _spawn_parallel_group_cli(
        self,
        specs: list[SessionSpec]
    ) -> list[SessionResult]:
        """
        Spawn multiple CLI sessions concurrently.
        Uses asyncio.gather for concurrent I/O-bound subprocess management.
        """
        tasks = [self._spawn_session_cli_async(spec) for spec in specs]
        return await asyncio.gather(*tasks)

    # === Agent Teams Backend ===

    def _spawn_session_agent_teams(self, spec: SessionSpec) -> SessionResult:
        """
        Spawn a session as a Claude Code Agent Teams teammate.

        Uses the Agent Teams API to create a teammate with:
        - Scoped prompt as teammate instructions
        - Working directory set to worktree_path
        - Inter-agent messaging enabled for boundary crossing negotiation

        Key advantage over CLI: teammates can communicate with each other
        during execution, enabling real-time boundary crossing resolution
        instead of post-execution tag parsing.
        """

    async def _spawn_parallel_group_agent_teams(
        self,
        specs: list[SessionSpec]
    ) -> list[SessionResult]:
        """
        Spawn teammates and coordinate via Agent Teams.

        Flow:
        1. Create teammates from specs
        2. Start all teammates
        3. Monitor inbox for boundary crossing messages
        4. Route boundary crossings to owning agents (real-time)
        5. Collect results when all teammates complete
        """

    # === Common ===

    def _get_files_changed(self, worktree_path: str, base_branch: str) -> list[str]:
        """git diff --name-only {base_branch}...HEAD (run inside worktree)"""

    def _parse_boundary_crossings(self, stdout: str) -> list[str]:
        """Parse [BOUNDARY_CROSSING: ...] tags from session output."""

    def rollback(self) -> None:
        """
        Rollback all changes.

        git worktree remove .worktrees/parallel/{task_id} --force
        git branch -D parallel/task_1 parallel/task_2 ...
        """
```

### 2.3 Agent Teams Adapter

**File**: `src/agenthub/parallel/teams_adapter.py` (NEW)

```python
class AgentTeamsAdapter:
    """
    Bridges AgentHub's orchestration → Claude Code Agent Teams execution.

    Translates between AgentHub's SessionSpec and Agent Teams' teammate API.
    Handles:
    - Teammate creation with scoped prompts
    - Inbox message routing for boundary crossing negotiation
    - Result collection and translation back to SessionResult
    """

    def __init__(
        self,
        hub: AgentHub,
        escalation_handler: MidExecutionEscalationHandler,
    ):
        self._hub = hub
        self._escalation = escalation_handler

    def is_available(self) -> bool:
        """
        Check if Agent Teams is enabled.
        Checks CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS env var.
        Falls back to CLI if not available.
        """

    def create_teammate(
        self,
        spec: SessionSpec,
        other_specs: list[SessionSpec],
    ) -> str:
        """
        Create an Agent Teams teammate from a SessionSpec.

        Injects:
        - AgentHub's scoped prompt as teammate instructions
        - Working directory as worktree_path
        - Teammate awareness: names/roles of other active teammates
          (so they can message each other for boundary crossings)

        Returns teammate_id.
        """

    def monitor_inbox(
        self,
        teammate_ids: list[str],
        timeout: int,
    ) -> list[BoundaryCrossing]:
        """
        Monitor Agent Teams inbox for boundary crossing messages.

        When a teammate sends a message like:
          "I need to modify src/api/routes.py but it's outside my scope"

        This method:
        1. Parses the message into a BoundaryCrossing
        2. Routes to the owning agent's teammate via _escalation
        3. Sends the resolution back to the requesting teammate
        4. Returns all crossings for the execution trace

        This replaces the post-execution [BOUNDARY_CROSSING] tag parsing
        with real-time inter-agent negotiation.
        """

    def collect_results(
        self,
        teammate_ids: list[str],
    ) -> list[SessionResult]:
        """Collect results from completed teammates → SessionResult."""
```

### 2.4 Scoped Prompt Template

**File**: `src/agenthub/parallel/prompts.py` (NEW)

```python
SCOPED_SESSION_PROMPT = """
You are working on a specific task within a larger codebase.
Other teams are working on other tasks simultaneously on separate branches.

TASK: {task_description}

YOUR TEAM'S FILES (you should modify these):
{your_files}

OTHER TEAMS' FILES (do NOT modify these):
{other_files}

RELEVANT TECHNICAL CONTEXT (from your Tier B team):
{tier_b_context}

BUSINESS CONTEXT (from Tier A agents):
{tier_a_context}

IMPORTANT CONSTRAINTS:
- Focus ONLY on the task described above
- Only modify files within YOUR TEAM'S FILES listed above
- Do NOT modify any file listed under OTHER TEAMS' FILES
- If you discover you NEED to modify a file outside your scope:
  1. STOP modifying that file
  2. Note it clearly with the tag: [BOUNDARY_CROSSING: {{file_path}} — {{reason}}]
  3. Continue implementing what you can without that modification
  4. The orchestrator will coordinate with the other team
- Run relevant tests when done (tests may fail if you noted boundary crossings)

Please implement this task and run tests to verify.
"""
```

### 2.5 Tests for Phase 2

**File**: `tests/test_orchestrator.py` (NEW)

```python
def test_create_worktree():
    """Should create .worktrees/parallel/{task_id} with its own checkout."""

def test_worktrees_are_independent():
    """Two worktrees should have independent file states."""

def test_ensure_clean_tree_fails_on_dirty():
    """Should fail if uncommitted changes exist."""

def test_spawn_parallel_sessions_cli():
    """Should spawn CLI sessions concurrently via asyncio."""

def test_spawn_parallel_sessions_agent_teams():
    """Should spawn Agent Teams teammates when backend='agent_teams'."""

def test_fallback_to_cli_when_teams_unavailable():
    """Should fall back to CLI if Agent Teams env var not set."""

def test_session_timeout():
    """Should timeout sessions that run too long."""

def test_parse_boundary_crossings():
    """Should extract [BOUNDARY_CROSSING] tags from output."""

def test_rollback_cleans_worktrees():
    """Should remove all worktrees and delete parallel branches."""

def test_files_changed_detection():
    """Should correctly detect files changed in worktree vs base."""
```

---

## Phase 3: Merge + Conflict Resolution

**Goal**: Merge branches with domain-agent-assisted conflict resolution.

### 3.1 Data Models

**File**: `src/agenthub/parallel/models.py` (ADD)

```python
class ConflictType(str, Enum):
    TEXTUAL = "textual"
    SEMANTIC = "semantic"
    SCOPE_VIOLATION = "scope_violation"

@dataclass
class MergeConflict:
    """A detected merge conflict."""
    file_path: str
    conflict_type: ConflictType
    description: str
    branch_a: str
    branch_b: str
    diff_a: str
    diff_b: str
    owning_agent: Optional[str]
    auto_resolvable: bool
    suggested_resolution: Optional[str]

@dataclass
class DomainResolutionProposal:
    """A domain agent's proposed resolution."""
    agent_id: str
    agent_name: str
    conflict_file: str
    proposed_resolution: str
    reasoning: str
    confidence: float             # "How confident am I in THIS resolution?"
                                  # < 0.6 → triggers CEO escalation via _should_escalate_to_ceo()
                                  # Different from DomainClaim.confidence (domain relevance).
    side_effects: list[str]
    needs_ceo: bool

@dataclass
class MergeResult:
    """Result of merging parallel sessions."""
    success: bool
    merged_branch: str
    conflicts: list[MergeConflict]
    resolutions: list[DomainResolutionProposal]
    files_merged: list[str]
    test_results: Optional[dict]
    needs_user_input: bool
    escalation_reason: Optional[str]
    summary: str
```

### 3.2 MergeCoordinator

**File**: `src/agenthub/parallel/merge.py` (NEW)

```python
class MergeCoordinator:
    """
    Handles merging parallel session branches.

    Uses domain agents for conflict resolution proposals.
    Escalates to CEO when agents can't resolve.
    """

    def __init__(
        self,
        project_root: str,
        hub: AgentHub,
        client: Anthropic,
        sub_agent_manager: Optional[SubAgentManager] = None,
    ):
        self._root = Path(project_root)
        self._hub = hub
        self._client = client
        self._sub_manager = sub_agent_manager

    def merge_results(
        self,
        base_branch: str,
        session_results: list[SessionResult],
        tasks: list[ImplementationTask],
        run_tests: bool = True,
    ) -> MergeResult:
        """
        Merge strategy:

        1. SCOPE VALIDATION
           - Verify files_changed matches expected scope
           - Flag scope violations

        2. SEQUENTIAL MERGE
           - Create parallel/merged branch from base
           - Merge each branch (smallest first)
           - On conflict: consult owning agent

        3. POST-MERGE VERIFICATION
           - Run test suite
           - If tests fail: identify semantic conflicts

        4. TIERED CONFLICT RESOLUTION
           Level 1: Agent self-resolution (simple conflicts)
           Level 2: Inter-agent negotiation (overlapping domains)
           Level 3: CEO escalation (agents disagree or low confidence)
        """

    def _validate_scope(
        self,
        result: SessionResult,
        task: ImplementationTask,
    ) -> list[MergeConflict]:
        """Check if session stayed within scope."""

    def _get_owning_agent(self, file_path: str) -> Optional[BaseAgent]:
        """
        Get most specific agent owning a file.
        Sub-agent > team lead > Tier B.
        """

    def _request_agent_resolution(
        self,
        agent: BaseAgent,
        conflict: MergeConflict,
        task_a: ImplementationTask,
        task_b: ImplementationTask,
    ) -> DomainResolutionProposal:
        """Ask domain agent to propose resolution."""

    def _negotiate_between_agents(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        conflict: MergeConflict,
    ) -> tuple[DomainResolutionProposal, Optional[DomainResolutionProposal]]:
        """
        When two agents claim the conflict file:
        - Both propose resolutions
        - Compare: equivalent, compatible, or contradictory
        - Contradictory → return both for CEO escalation
        """

    def _should_escalate_to_ceo(
        self,
        conflict: MergeConflict,
        proposals: list[DomainResolutionProposal],
    ) -> tuple[bool, str]:
        """
        Escalate when:
        - No owning agent found
        - Agent confidence < 0.6
        - Agents disagree
        - Agent sets needs_ceo=True
        - Scope violation
        """

    def _build_ceo_escalation_message(
        self,
        conflict: MergeConflict,
        proposals: list[DomainResolutionProposal],
    ) -> str:
        """Build human-friendly escalation message."""

    def _apply_resolution(
        self,
        conflict: MergeConflict,
        resolution: DomainResolutionProposal,
    ) -> None:
        """Apply resolved content to file and stage."""

    def _run_tests(self) -> dict:
        """
        Run test suite and return results.
        Auto-detects: pytest, jest, make test, npm test.
        """

    def _analyze_semantic_conflict(
        self,
        test_output: str,
        session_results: list[SessionResult],
    ) -> list[MergeConflict]:
        """
        When tests fail but no textual conflicts:
        - Parse test failures
        - Consult domain agents for diagnosis
        """
```

### 3.3 Tests for Phase 3

**File**: `tests/test_merge.py` (NEW)

```python
def test_clean_merge():
    """Non-overlapping changes should merge cleanly."""

def test_textual_conflict_detection():
    """Should detect textual merge conflicts."""

def test_agent_resolution_proposal():
    """Agent should propose valid resolution."""

def test_inter_agent_negotiation():
    """Two agents should negotiate on shared file."""

def test_escalate_on_low_confidence():
    """Should escalate when agent confidence < 0.6."""

def test_escalate_on_disagreement():
    """Should escalate when agents disagree."""

def test_scope_violation_detection():
    """Should detect files modified outside scope."""

def test_post_merge_test_run():
    """Should run tests after merge."""

def test_semantic_conflict_from_test_failure():
    """Should detect semantic conflict when tests fail."""
```

---

## Phase 4: Mid-Execution Boundary Crossing Protocol

**Goal**: Handle cases where a session needs files outside its scope.

**Design note — two modes depending on execution backend:**
- **CLI backend**: Boundary crossings are detected *post-execution* by parsing
  `[BOUNDARY_CROSSING]` tags from session stdout. Resolution happens between
  execution and merge. This is simpler but batch-only.
- **Agent Teams backend**: Boundary crossings are negotiated *in real-time* via
  the Agent Teams inbox system. The AgentTeamsAdapter monitors messages and routes
  crossing requests to owning agents during execution. This is the preferred mode.

In practice, this means Phase 4 is lighter than it looks — for CLI mode, the
detection is just tag parsing (a few helper methods), and the resolution logic
is shared with MergeCoordinator's agent consultation. For Agent Teams mode,
the heavy lifting is in `teams_adapter.py` (Phase 2).

### 4.1 Data Models

**File**: `src/agenthub/parallel/models.py` (ADD)

```python
class CrossingResolutionType(str, Enum):
    APPROVED_AS_IS = "approved_as_is"
    APPROVED_WITH_MODIFICATION = "approved_with_modification"
    DEFERRED_TO_MERGE = "deferred_to_merge"
    ESCALATED_TO_CEO = "escalated_to_ceo"
    REJECTED = "rejected"

@dataclass
class BoundaryCrossing:
    """Detected when a session needs files outside its scope."""
    session_task_id: str
    requesting_agent: str
    target_file: str
    owning_agent: Optional[str]
    reason: str
    proposed_change: Optional[str]
    blocking: bool
    detected_at: Literal["post_execution", "real_time"]  # Which backend caught it

@dataclass
class BoundaryCrossingResolution:
    """Resolution for a boundary crossing."""
    crossing: BoundaryCrossing
    approved: bool
    resolution_type: CrossingResolutionType
    modified_change: Optional[str]
    reasoning: str
    confidence: float             # "How confident is the owning agent in this resolution?"
                                  # < 0.6 → escalate to CEO (same threshold as merge resolution)
```

### 4.2 MidExecutionEscalationHandler

**File**: `src/agenthub/parallel/escalation.py` (NEW)

```python
class MidExecutionEscalationHandler:
    """
    Handles boundary crossings detected during/after session execution.

    Two modes:
    - CLI backend: Post-execution detection from [BOUNDARY_CROSSING] tags.
      Crossings are resolved between execution and merge.
    - Agent Teams backend: Real-time detection via inbox monitoring.
      Crossings are resolved during execution by AgentTeamsAdapter.

    The resolution logic (_resolve_crossing, _ask_owning_agent) is shared
    between both modes and also reused by MergeCoordinator for conflict
    resolution — they're the same pattern (ask owning agent, escalate if
    low confidence).
    """

    def __init__(
        self,
        hub: AgentHub,
        sub_agent_manager: Optional[SubAgentManager] = None,
    ):
        self._hub = hub
        self._sub_manager = sub_agent_manager

    def detect_boundary_crossings(
        self,
        session_result: SessionResult,
        task: ImplementationTask,
    ) -> list[BoundaryCrossing]:
        """
        Parse session output for [BOUNDARY_CROSSING: file — reason] tags.
        Only used in CLI mode. Agent Teams mode detects in real-time via
        AgentTeamsAdapter.monitor_inbox().
        """

    def resolve_crossing(
        self,
        crossing: BoundaryCrossing,
        task: ImplementationTask,
    ) -> BoundaryCrossingResolution:
        """
        Inter-team negotiation:

        1. Find owning agent for target file
        2. Ask agent: approve, modify, defer, or reject?
        3. If confidence < 0.6 → escalate to CEO
        4. Return resolution

        Used by both CLI (post-execution) and Agent Teams (real-time) modes.
        """

    def _ask_owning_agent(
        self,
        crossing: BoundaryCrossing,
        owning_agent: BaseAgent,
        task: ImplementationTask,
    ) -> dict:
        """
        Ask the owning agent about the boundary crossing.

        Returns: {decision, reasoning, modified_change, confidence}
        """
```

### 4.3 Tests for Phase 4

**File**: `tests/test_escalation.py` (NEW)

```python
def test_detect_boundary_crossing_tags():
    """Should parse [BOUNDARY_CROSSING] tags from output (CLI mode)."""

def test_resolve_crossing_approved():
    """Owning agent approval should allow crossing."""

def test_resolve_crossing_modified():
    """Owning agent should be able to suggest modification."""

def test_resolve_crossing_rejected():
    """Owning agent rejection should block crossing."""

def test_escalate_on_low_confidence():
    """Should escalate when agent confidence < 0.6."""

def test_real_time_detection_sets_detected_at():
    """Agent Teams crossings should have detected_at='real_time'."""
```

---

## Phase 5: Integration — ParallelSessionManager

**Goal**: Top-level orchestrator that ties everything together.

### 5.1 Data Models

**File**: `src/agenthub/parallel/models.py` (ADD)

```python
@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel execution."""
    max_parallel_sessions: int = 3
    session_timeout_seconds: int = 300
    claude_model: str = "claude-sonnet-4-20250514"
    auto_resolve_conflicts: bool = True
    run_tests_after_merge: bool = True
    test_command: Optional[str] = None  # Auto-detect if None
    auto_proceed_threshold: Literal["safe", "caution"] = "caution"
    token_budget: int = 100000
    execution_backend: Literal["cli", "agent_teams"] = "cli"
    # "cli": claude --print (stable, non-interactive)
    # "agent_teams": Claude Code Agent Teams (real-time inter-agent messaging)
    # Falls back to "cli" if Agent Teams is unavailable

@dataclass
class ParallelExecutionTrace:
    """Comprehensive trace for debugging and benchmarking."""
    decomposition_time_ms: int
    analysis_time_ms: int
    session_times: dict[str, int]
    merge_time_ms: int
    total_time_ms: int
    decomposition_tokens: int
    analysis_tokens: int
    session_tokens: dict[str, int]
    merge_tokens: int
    total_tokens: int
    parallel_groups: list[list[str]]
    conflicts_found: int
    conflicts_auto_resolved: int
    scope_violations: int
    test_pass_rate: Optional[float]

@dataclass
class ParallelExecutionResult:
    """Result of parallel execution."""
    success: bool
    tasks: list[ImplementationTask]
    plan: ParallelizationPlan
    session_results: list[SessionResult]
    merge_result: MergeResult
    total_time_seconds: float
    sequential_estimate_seconds: float
    speedup: float
    total_tokens: int
    trace: ParallelExecutionTrace
```

### 5.2 ParallelSessionManager

**File**: `src/agenthub/parallel/manager.py` (NEW)

```python
class ParallelSessionManager:
    """
    Top-level orchestrator for parallel sessions.

    Implements the company model:
    - Decompose (PM breaks down work)
    - Analyze (Risk assessment)
    - CEO Confirmation (when MEDIUM/HIGH risk)
    - Execute (Teams work on branches)
    - Merge (Merge committee integrates)
    - Verify (QA test gate)
    """

    def __init__(
        self,
        hub: AgentHub,
        project_root: str,
        config: ParallelExecutionConfig = None,
    ):
        self._hub = hub
        self._root = project_root
        self._config = config or ParallelExecutionConfig()

        # Initialize components
        self._decomposer = TaskDecomposer(hub._client, hub, hub._import_graph)
        self._analyzer = ConflictRiskAnalyzer(
            hub._import_graph, hub, hub._sub_agent_manager
        )
        self._orchestrator = BranchOrchestrator(
            project_root, hub,
            max_parallel=self._config.max_parallel_sessions,
            claude_model=self._config.claude_model,
            session_timeout=self._config.session_timeout_seconds,
            execution_backend=self._config.execution_backend,
        )
        self._merger = MergeCoordinator(
            project_root, hub, hub._client, hub._sub_agent_manager
        )
        self._escalation = MidExecutionEscalationHandler(
            hub, hub._sub_agent_manager
        )

    def analyze(self, request: str) -> tuple[DecompositionResult, ParallelizationPlan]:
        """
        Dry-run analysis. Returns decomposition and plan without executing.
        Used by agenthub_analyze_parallelism MCP tool.
        """

    def execute(
        self,
        request: str,
        confirm_callback: Optional[Callable[[str], bool]] = None,
    ) -> ParallelExecutionResult:
        """
        Full execution pipeline:

        1. DECOMPOSE
           - Break request into tasks
           - If single task → run directly without parallelization

        2. ANALYZE
           - Compute risk and parallelization plan

        3. CEO CONFIRMATION (risk-tiered)
           - SAFE/CAUTION: Auto-proceed (based on config.auto_proceed_threshold)
           - MEDIUM: Ask CEO, PM recommends parallel with caution
           - HIGH: Ask CEO, PM recommends sequential + provides order
           - Never silent fallback — CEO always gets a shot-call for MEDIUM/HIGH

        4. EXECUTE
           - Spawn sessions per plan (parallel groups or sequential)

        5. BOUNDARY CROSSING CHECK
           - Detect and resolve crossings

        6. MERGE
           - Merge branches with agent-assisted resolution

        7. VERIFY
           - Run tests, consult agents on failures

        8. RETURN
           - Return result with full trace
        """

    def _should_auto_proceed(self, risk: RiskLevel) -> bool:
        """Check if risk level allows auto-proceed."""
        threshold = self._config.auto_proceed_threshold
        if threshold == "safe":
            return risk == RiskLevel.NONE
        elif threshold == "caution":
            return risk in [RiskLevel.NONE, RiskLevel.LOW]
        return False

    def _build_ceo_prompt(
        self,
        decomposition: DecompositionResult,
        plan: ParallelizationPlan,
    ) -> str:
        """
        Build the CEO confirmation prompt.

        Includes:
        - Task breakdown
        - Risk assessment with agent concerns
        - PM recommendation (parallel or sequential)
        - If sequential: execution order plan
        - Options: parallel, sequential, modify
        """

    def _format_pm_briefing(
        self,
        decomposition: DecompositionResult,
        plan: ParallelizationPlan,
    ) -> str:
        """Format the PM briefing for CEO."""
```

### 5.3 MCP Tools

**File**: `src/agenthub/mcp_server.py` (MODIFY)

```python
@mcp.tool()
async def agenthub_parallel_execute(
    request: str,
    confirm: bool = True,
    max_parallel: int = 3,
) -> str:
    """
    Analyze and execute a multi-part request in parallel.

    Uses import graph analysis to determine safe parallelization.
    Returns merged result.
    """
    hub = get_hub()
    config = ParallelExecutionConfig(max_parallel_sessions=max_parallel)
    manager = ParallelSessionManager(hub, PROJECT_ROOT, config)

    if confirm:
        # Return plan for confirmation
        decomp, plan = manager.analyze(request)
        return format_confirmation_prompt(decomp, plan)
    else:
        # Execute directly
        result = manager.execute(request)
        return format_execution_result(result)


@mcp.tool()
async def agenthub_resolve_conflict(
    conflict_id: str,
    resolution: Literal["accept_a", "accept_b", "auto_merge", "custom"],
    custom_content: Optional[str] = None,
) -> str:
    """
    Resolve a merge conflict from parallel execution.
    """
    # Implementation depends on conflict tracking mechanism
```

### 5.4 Hub Integration

**File**: `src/agenthub/hub.py` (MODIFY)

```python
class AgentHub:
    # ... existing code ...

    def enable_parallel_sessions(
        self,
        project_root: str,
        config: ParallelExecutionConfig = None,
    ) -> ParallelSessionManager:
        """
        Enable parallel session execution.

        Requires:
        - Auto-agents enabled (for domain agent consultation)
        - Import graph built (for conflict analysis)
        """
        if not self._import_graph:
            raise RuntimeError("Import graph required. Call enable_auto_agents first.")

        self._parallel_manager = ParallelSessionManager(self, project_root, config)
        return self._parallel_manager

    def parallel_execute(
        self,
        request: str,
        confirm: bool = True,
    ) -> ParallelExecutionResult:
        """Execute a request using parallel sessions."""
        if not self._parallel_manager:
            raise RuntimeError("Parallel sessions not enabled.")
        return self._parallel_manager.execute(request)
```

### 5.5 Tests for Phase 5

**File**: `tests/test_parallel_manager.py` (NEW)

```python
def test_single_task_no_parallelization():
    """Single-task request should run directly."""

def test_parallel_execution_safe_risk():
    """SAFE risk should auto-proceed."""

def test_parallel_execution_medium_risk_asks_ceo():
    """MEDIUM risk should ask CEO for confirmation."""

def test_parallel_execution_high_risk_recommends_sequential():
    """HIGH risk should recommend sequential with order plan."""

def test_ceo_can_override_sequential():
    """CEO should be able to override and run parallel."""

def test_full_pipeline_happy_path():
    """Full pipeline with no conflicts."""

def test_full_pipeline_with_conflict():
    """Full pipeline with merge conflict resolved by agent."""

def test_full_pipeline_with_escalation():
    """Full pipeline with CEO escalation."""

def test_token_budget_enforcement():
    """Should fall back to sequential when budget exceeded."""

def test_rollback_on_failure():
    """Should rollback all branches on failure."""
```

---

## Phase 6: Polish

### 6.1 Progressive Output

**File**: `src/agenthub/parallel/output.py` (NEW)

```python
class ProgressReporter:
    """Reports progress during parallel execution."""

    def __init__(self, broadcast_callback: Optional[Callable] = None):
        self._callback = broadcast_callback

    def report_decomposition(self, result: DecompositionResult) -> None:
        """Report task decomposition complete."""

    def report_analysis(self, plan: ParallelizationPlan) -> None:
        """Report conflict analysis complete."""

    def report_session_start(self, task_id: str, branch: str) -> None:
        """Report session started."""

    def report_session_progress(self, task_id: str, progress: float) -> None:
        """Report session progress (0.0 - 1.0)."""

    def report_session_complete(self, result: SessionResult) -> None:
        """Report session completed."""

    def report_boundary_crossing(self, crossing: BoundaryCrossing) -> None:
        """Report boundary crossing detected."""

    def report_merge_progress(self, files_merged: int, total: int) -> None:
        """Report merge progress."""

    def report_conflict(self, conflict: MergeConflict) -> None:
        """Report conflict detected."""

    def report_resolution(self, proposal: DomainResolutionProposal) -> None:
        """Report agent resolution proposed."""
```

### 6.2 Dashboard Integration

**File**: `src/agenthub/parallel/dashboard.py` (NEW)

```python
def broadcast_parallel_event(event_type: str, data: dict) -> None:
    """
    Broadcast parallel execution events to dashboard.

    Event types:
    - parallel_started
    - task_decomposed
    - risk_analyzed
    - session_started
    - session_progress
    - session_completed
    - boundary_crossing
    - merge_started
    - conflict_detected
    - conflict_resolved
    - merge_completed
    - parallel_completed
    """
```

### 6.3 Sub-Agent Tree Visualization

**File**: `src/agenthub/auto/tree.py` (MODIFY)

```python
def format_agent_tree(
    hub: AgentHub,
    include_sub_agents: bool = True,
) -> str:
    """
    Format agent hierarchy as tree.

    Example:
    AgentHub (5 agents)
    ├── Tier A (1 agent)
    │   └── business_agent
    └── Tier B (4 agents)
        ├── backend_agent (Team Lead)
        │   ├── backend_api_agent
        │   ├── backend_models_agent
        │   └── backend_services_agent
        └── frontend_agent
    """
```

### 6.4 Confidence Calibration (POST-MVP — skip for initial release)

**File**: `src/agenthub/parallel/calibration.py` (NEW — post-MVP)

**Why post-MVP**: Calibration requires hundreds of execution records to be
statistically meaningful. Building this before you have real usage data is
premature optimization. For MVP, hardcode the confidence thresholds (0.3 for
domain claims, 0.6 for resolution escalation) and revisit calibration once
you have 100+ parallel execution traces to learn from.

```python
class ConfidenceCalibrator:
    """
    Calibrates agent confidence based on historical execution.

    Tracks:
    - Predictions vs actual conflicts
    - Agent resolution success rate
    - False positive/negative rates

    POST-MVP: Only build this after collecting sufficient execution history.
    Target: 100+ executions before calibration is meaningful.
    """

    def __init__(self, history_file: str = ".agenthub/parallel_history.json"):
        self._history_file = history_file

    def record_execution(self, result: ParallelExecutionResult) -> None:
        """Record execution for calibration."""

    def get_calibrated_confidence(
        self,
        agent_id: str,
        raw_confidence: float,
    ) -> float:
        """Apply calibration to agent's confidence score."""

    def get_agent_track_record(self, agent_id: str) -> dict:
        """Get agent's historical accuracy."""
```

---

## Directory Structure

After all phases, the new files will be:

```
src/agenthub/
├── parallel/                          # NEW directory
│   ├── __init__.py
│   ├── models.py                      # All data models
│   ├── decomposer.py                  # TaskDecomposer
│   ├── analyzer.py                    # ConflictRiskAnalyzer
│   ├── orchestrator.py                # BranchOrchestrator (git worktree + dual backend)
│   ├── teams_adapter.py              # NEW — AgentHub → Agent Teams bridge
│   ├── merge.py                       # MergeCoordinator
│   ├── escalation.py                  # MidExecutionEscalationHandler
│   ├── manager.py                     # ParallelSessionManager
│   ├── prompts.py                     # Prompt templates
│   ├── output.py                      # ProgressReporter
│   ├── dashboard.py                   # Dashboard integration
│   └── calibration.py                 # Confidence calibration (POST-MVP)
│
├── auto/
│   ├── sub_agent_policy.py            # NEW
│   ├── sub_agent_manager.py           # NEW
│   ├── manager.py                     # MODIFIED (add sub-agent support)
│   └── tree.py                        # MODIFIED (add sub-agent visualization)
│
├── models.py                          # MODIFIED (add hierarchy fields to AgentSpec)
├── hub.py                             # MODIFIED (add enable_parallel_sessions)
└── mcp_server.py                      # MODIFIED (add new MCP tools)

tests/
├── test_sub_agents.py                 # NEW
├── test_decomposer.py                 # NEW
├── test_analyzer.py                   # NEW
├── test_orchestrator.py               # NEW (covers both CLI + Agent Teams backends)
├── test_teams_adapter.py             # NEW — Agent Teams integration tests
├── test_merge.py                      # NEW
├── test_escalation.py                 # NEW
└── test_parallel_manager.py           # NEW
```

---

## Implementation Order

| Phase | Duration | Priority | Dependencies | Notes |
|-------|----------|----------|--------------|-------|
| Phase 0: Sub-Agents | 1 week | HIGH | None | |
| Phase 1: Foundation | 1 week | HIGH | Phase 0 | |
| Phase 2: Execution (CLI) | 3-4 days | HIGH | Phase 1 | git worktree + asyncio. CLI backend first. |
| Phase 2b: Execution (Agent Teams) | 2-3 days | MEDIUM | Phase 2 | teams_adapter.py. Can be deferred if Agent Teams API unstable. |
| Phase 3: Merge | 2 weeks | HIGH | Phase 2 | Hardest phase — semantic conflicts, inter-agent negotiation, git merge edge cases. Budget extra time here. |
| Phase 4: Escalation | 2-3 days | MEDIUM | Phase 2, 3 | Lighter than originally scoped — CLI mode is just tag parsing + shared resolution logic. Agent Teams mode handled in Phase 2b. |
| Phase 5: Integration | 1 week | HIGH | Phase 0-4 | |
| Phase 6: Polish (MVP) | 3-4 days | LOW | Phase 5 | ProgressReporter, dashboard, tree viz. Skip ConfidenceCalibrator. |
| Phase 6b: Calibration | post-MVP | LOW | 100+ executions | Only after sufficient execution history. |

**Total**: ~7 weeks (same total, redistributed)

**MVP (Phases 0-3, 5, 6-MVP)**: ~5.5 weeks — functional parallel execution with CLI backend, merge resolution, and basic progress reporting

**Full (with Agent Teams)**: ~6.5 weeks — adds real-time boundary crossing negotiation via Agent Teams

---

## Key Design Decisions Preserved from Business Spec

1. **Git branches** for isolation (not in-memory)
2. **Git worktrees** for true parallel filesystem isolation (NOT git checkout)
3. **Conservative by default** (`auto_proceed_threshold: "caution"`)
4. **No silent fallbacks** — CEO always gets a shot-call for MEDIUM/HIGH risk
5. **Agents can only upgrade risk**, never downgrade
6. **Sub-agent routing** for precise conflict analysis
7. **Tiered escalation** (agent self-resolve → inter-agent negotiation → CEO)
8. **Post-merge test gate** as final safety check
9. **Full rollback** on any failure
10. **Dual execution backend** — CLI (stable fallback) + Agent Teams (preferred when available)
11. **AgentHub = brain, Agent Teams = muscle** — Agent Teams provides parallel execution infrastructure; AgentHub provides domain intelligence, risk analysis, scoped context, and structured merge

---

## Open Questions for Implementation

1. **~~Session spawning in v1~~**: ~~Use `claude --print` CLI or subprocess with API?~~
   - **RESOLVED**: CLI (`claude --print`) as v1 backend, Agent Teams as v2 backend.
     CLI is stable and non-interactive. Agent Teams adds inter-agent communication
     but is still experimental (research preview, Feb 2026).

2. **Conflict resolution UI**: How does CEO provide custom resolution?
   - MCP tool with custom_content parameter?
   - Interactive prompt?

3. **Test command detection**: How robust should auto-detection be?
   - Check for pytest.ini, jest.config.js, Makefile, package.json scripts?

4. **Token budget tracking**: Per-session or aggregate?
   - Aggregate makes more sense for budget enforcement

5. **~~Parallel execution limit~~**: ~~ProcessPoolExecutor or asyncio?~~
   - **RESOLVED**: `asyncio.create_subprocess_exec` with `asyncio.gather`.
     Claude sessions are I/O-bound (waiting on API), not CPU-bound.
     asyncio gives concurrency without process pool overhead.
     Each session runs in its own git worktree for filesystem isolation.

6. **Agent Teams API stability**: The Agent Teams feature shipped as research
   preview (Feb 5, 2026). The API may change. Design the `AgentTeamsAdapter`
   as a thin translation layer so changes to the Agent Teams API only require
   updating one file (`teams_adapter.py`), not the entire orchestration pipeline.

7. **Agent Teams + git worktree interaction**: Does Agent Teams respect
   working directory when spawning teammates? Need to verify that each
   teammate can be pointed at a specific worktree path. If not, may need
   to `cd` into the worktree before teammate creation.

8. **Agent Teams inbox message format**: What's the actual message format
   for inter-agent communication? Need to define a structured format for
   boundary crossing requests so `AgentTeamsAdapter.monitor_inbox()` can
   parse them reliably (not just free-text messages).
