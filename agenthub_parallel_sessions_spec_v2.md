# AgentHub Parallel Sessions Spec

## The Problem

When working on a large project with Claude Code, a developer often gives multi-part requests like:

> "Add a save button to the toolbar and also build a time series chart component"

Today, Claude Code handles these sequentially — it finishes the button, then starts the chart. But these two tasks touch completely different files and could safely run in parallel. The developer has no way to know this upfront, so they just wait.

**This feature lets AgentHub analyze task independence using its import graph and domain knowledge, then orchestrate multiple Claude Code sessions working simultaneously on separate git branches, merging the results when done.**

---

## Architecture Overview

```
User Request
     │
     ▼
┌─────────────────────┐
│   Task Decomposer   │ ← Breaks request into discrete tasks
│   (Haiku / Sonnet)  │
└─────────┬───────────┘
          │ list[Task]
          ▼
┌─────────────────────┐
│ Conflict Risk        │ ← Static: ImportGraph file/dependency overlap
│ Analyzer             │ ← Semantic: domain agents flag hidden conflicts
│  (static + agents)   │
└─────────┬───────────┘
          │ ParallelizationPlan
          ▼
┌─────────────────────┐     ┌───────────────┐
│ Branch Orchestrator  │────▶│ git branch A  │──▶ Claude Session A (scoped)
│                      │────▶│ git branch B  │──▶ Claude Session B (scoped)
│                      │────▶│ git branch C  │──▶ Claude Session C (scoped)
└─────────┬───────────┘     └───────────────┘
          │ all sessions done
          ▼
┌─────────────────────┐
│  Merge Coordinator   │ ← git merge + semantic conflict detection
└─────────┬───────────┘
          │
          ├─ Clean merge → run tests → done
          ├─ Textual conflict → ask user OR auto-resolve
          └─ Semantic conflict (tests fail) → Resolution Agent
```

---

## Design Philosophy: The Company Model

AgentHub's parallel sessions are modeled after **how a real company operates**, not just how git branches work. The full analogy tells the story of *why each layer exists*:

### The Growth Story

When a **CEO (human)** and a **CTO (Claude Code)** start a **company (repository)** as a startup, they know everything that's going on. Every file, every function, every decision — it's all in their heads.

But as the company gets bigger, the CTO starts losing track. Even if the CTO gets smarter (larger context windows), having a smarter CTO doesn't mean the CTO personally reviewing every line of code is *economical*. That's when you **recruit** — Tier B agents are auto-generated from the codebase's import graph, each owning a domain.

When the company gets even bigger, a single recruit can't handle their entire domain alone. A "backend agent" covering 200 files is like one employee running the entire engineering department. That's when you need **teams** — sub-Tier B agents split a domain into focused sub-domains, each with a team lead (the parent Tier B agent) coordinating their work.

And when there's a company-wide mission (a multi-part user request), these teams need to **work simultaneously**, communicating at boundaries, and only escalating to the CEO when they genuinely can't resolve something themselves.

```
                         ┌──────────────┐
                         │     CEO      │ ← Human developer
                         │  (shot-calls)│    Only escalated when teams can't resolve
                         └──────┬───────┘
                                │
                  ┌─────────────┼─────────────┐
                  ▼                            ▼
           ┌──────┴───────┐          ┌─────────────────┐
           │     CTO      │          │  Business Leads  │ ← Tier A agents
           │ (orchestrate)│          │ (domain experts) │    Human-curated, know WHY
           └──────┬───────┘          │                  │    the code exists
                  │                  │ • Product agent  │
                  │                  │ • BizContext agent│
                  │                  │ • Analytics agent │
                  │                  └────────┬─────────┘
                  │                           │
                  │    ┌──────────────────────┘
                  │    │  Tier B consults Tier A for
                  │    │  business context when needed
                  │    ▼
     ┌────────────┼────────────────┐
     ▼            ▼                ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Team Lead A│ │ Team Lead B │ │ Team Lead C │ ← Tier B agents
│ (UI domain) │ │(Data domain)│ │(API domain) │    Auto-generated from
│             │ │             │ │             │    import graph clusters
│  ┌───┬───┐  │ │  ┌───┬───┐  │ │  ┌───┬───┐  │
│  │W1 │W2 │  │ │  │W1 │W2 │  │ │  │W1 │W2 │  │ ← Sub-Tier B agents
│  │W3 │   │  │ │  │W3 │W4 │  │ │  │   │   │  │    Team members under
│  └───┴───┘  │ │  └───┴───┘  │ │  └───┴───┘  │    each team lead
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       └───────────────┼───────────────┘
                       ▼
              Inter-Team Protocol
              (boundary negotiation)
```

### Tier A = The Business Side of the Company

Tier B agents know *how the code works*. Tier A agents know *why the code exists*.

In a real company, the engineering team doesn't operate in a vacuum. When the backend team has a resource conflict, they don't just resolve it based on code structure — they ask product: "Which feature matters more to the business?" When the data team is designing a pipeline, they consult the business context team: "What metrics does the client actually care about?"

Tier A agents are the **business leads** — human-curated agents that carry domain knowledge no import graph can derive:

| Tier A Agent (example) | What It Knows | When Tier B Consults It |
|---|---|---|
| Product Strategy agent | What the product does, who it serves, what the roadmap is | Task decomposition: "which of these tasks is higher priority?" |
| Business Context agent | Industry context, client requirements, compliance rules | Conflict resolution: "the analytics feature is client-facing, handle with care" |
| Data/Analytics agent | What metrics matter, how data flows, what KPIs mean | Merge decisions: "ROAS calculation is the core BM — don't break it" |
| text2sql agent | How business questions map to data queries | Scoped prompts: inject query patterns into sessions touching the data layer |

**Why this matters for parallel sessions:**

Without Tier A context, Tier B agents make purely structural decisions: "these files don't overlap, so it's safe." But some conflicts are *business* conflicts, not code conflicts. Two tasks might touch completely different files but both affect the same customer-facing metric. A Tier B agent can't see that. A Tier A agent can.

**How Tier A agents participate:**

1. **Task Decomposition**: The `TaskDecomposer` consults Tier A agents to understand business priority. If two tasks are both important but one is customer-facing and the other is internal, that affects sequencing decisions when parallelization isn't safe.

2. **Conflict Risk Analysis**: After static analysis and Tier B agent consultation, Tier A agents get a final review: "These tasks affect the ad analytics pipeline. From a business perspective, is running them in parallel risky?" A Tier A agent might flag: "Both tasks affect ROAS calculation. That's our core value prop — run sequentially to be safe."

3. **Merge Conflict Resolution**: When Tier B agents propose a code resolution, Tier A agents validate the business impact: "The proposed merge preserves both code changes, but the combined behavior changes how campaign budgets are calculated. The client expects the old behavior."

4. **CEO Escalation Context**: When conflicts escalate to the human, the escalation message includes both the Tier B technical assessment AND the Tier A business assessment. The CEO gets the full picture.

**The key insight**: Tier A agents don't touch code. They don't propose merge resolutions or analyze import graphs. They provide *context* that Tier B agents can't derive from code structure alone. They're the company's institutional knowledge — the "why" behind the "how."

### When Each Layer Activates

| Codebase Size | What Happens | Analogy |
|---|---|---|
| **< 50 files** | CTO (Claude Code) handles everything directly. No agents needed. | Startup: CEO and CTO do everything. |
| **50–200 files** | Tier B agents are generated from import graph clusters. CTO delegates domain knowledge to them. | Growing company: hire specialists. |
| **200–1000 files** | Some Tier B agents cover too much ground. They split into sub-agents. Parent becomes team lead, children become team workers. | Departments form: leads + team members. |
| **1000+ files** | Full company model: multiple teams, sub-agents in each, parallel sessions for company-wide missions, inter-team negotiation, CEO escalation. | Enterprise: parallel teams, R&R boundaries, executive escalation. |

### Core Principles

**1. Teams work in parallel by default.**
Just like in a company, teams don't wait for each other unless their work overlaps. The default state is parallel execution. Sequential is the fallback, not the norm.

**2. Teams own their R&R (roles & responsibilities).**
Each domain agent knows its boundaries — which files, modules, and concerns belong to it. When a session touches files outside its team's domain, that's a boundary crossing that needs negotiation.

**3. Boundary crossings trigger inter-team communication, not failure.**
If Team A discovers it needs to modify a file that belongs to Team B's domain, it doesn't silently do it and it doesn't halt. It communicates: "I need to touch your code. Here's why and what I'm changing." Team B's domain agent evaluates and either approves, proposes an alternative, or escalates.

**4. Domain agents propose resolutions, not generic LLMs.**
When a merge conflict happens, the domain agent that *owns the conflicting file* is the one proposing the resolution — not a generic Sonnet call with no context. The agent knows the codebase, the patterns, and the intent behind the code. Its resolution proposals are grounded in domain knowledge.

**5. The CEO (human) only gets shot-calls.**
The developer is only bothered for decisions that the teams genuinely can't resolve themselves:
- High-risk parallel execution approval
- Merge conflicts where domain agents disagree
- Architectural decisions that span multiple domains
- Scope violations that change the nature of a task

Everything else — routine merges, low-risk parallel execution, non-overlapping changes — the teams handle autonomously.

**6. Risk transparency, not risk hiding.**
When the system detects medium risk, it doesn't silently proceed or silently fall back. It tells the developer: "Merge conflict risk is medium. Team A's agent says: [specific concern]. Proceed in parallel or run sequentially?" The developer makes an informed choice.

### How This Maps to Components

| Company Concept | AgentHub Component | Role |
|---|---|---|
| CEO | Human developer | Final authority, shot-calls only |
| CTO | Claude Code (single session) | Handles everything at startup scale, delegates as codebase grows |
| Business Leads (CPO, CDO, etc.) | Tier A agents (human-curated) | Know *why* the code exists — business context, product strategy, domain expertise |
| VP of Engineering | `ParallelSessionManager` | Orchestrates the whole process |
| Project Manager | `TaskDecomposer` | Breaks work into team assignments (consults Tier A for priority) |
| Risk Assessment | `ConflictRiskAnalyzer` | Evaluates inter-team dependencies (technical from Tier B, business from Tier A) |
| Team Lead | Tier B domain agent | Owns a code domain, coordinates sub-agents, proposes conflict resolutions |
| Team Member | Sub-Tier B agent | Focused on a sub-domain within the team lead's scope |
| Team doing the work | Claude Code session (scoped) | Does the actual implementation, informed by team lead + members + Tier A context |
| HR / Merge Committee | `MergeCoordinator` | Facilitates inter-team integration |
| Post-launch QA | Test suite | Verifies the merged work |

---

## Sub-Tier B Agents: Teams Within Teams

This section specifies how Tier B agents split into sub-agents when their domain gets too large. This is the prerequisite that makes the company model work at scale — you can't have meaningful "teams" in parallel sessions if each "team" is just one agent covering 200 files.

### The Problem Sub-Agents Solve

Today, `CodebaseAnalyzer.propose_boundaries()` creates `AgentBoundary` objects — one per cluster in the import graph. Each boundary becomes a Tier B agent. But import graph clustering is coarse. A cluster like `backend/` might contain:

```
backend/
├── api/          ← REST endpoints (40 files)
├── models/       ← Database models (25 files)
├── services/     ← Business logic (35 files)
├── middleware/   ← Auth, logging, etc. (10 files)
└── utils/        ← Shared helpers (15 files)
```

That's 125 files under one agent. This agent would need massive context to answer any question about "its" domain, and during parallel sessions it can't meaningfully tell you whether `api/` and `services/` changes would conflict — it treats the whole backend as one blob.

**Sub-Tier B agents** split this into focused sub-domains, each with specific file ownership:

```
backend_agent (Team Lead — Tier B)
├── backend_api_agent (Team Member — Sub-Tier B)
│   └── owns: backend/api/**
├── backend_models_agent (Team Member — Sub-Tier B)
│   └── owns: backend/models/**
├── backend_services_agent (Team Member — Sub-Tier B)
│   └── owns: backend/services/**
└── backend_infra_agent (Team Member — Sub-Tier B)
    └── owns: backend/middleware/**, backend/utils/**
```

The parent `backend_agent` becomes the **team lead**: it coordinates its sub-agents, handles cross-sub-domain queries, and represents the team in inter-team negotiations. The sub-agents are the **team members**: each deeply knows a subset of files.

### Data Model Changes

```python
# Extension to existing AgentSpec
@dataclass
class AgentSpec:
    agent_id: str
    name: str
    description: str
    capabilities: list[AgentCapability]
    context_paths: list[str]
    context_keywords: list[str]
    estimated_tokens: int = 2000
    max_context_size: int = 50000
    system_prompt: str = ""
    temperature: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)

    # === NEW: Hierarchy fields ===
    parent_agent_id: Optional[str] = None       # None for top-level Tier B
    children_ids: list[str] = field(default_factory=list)  # Empty for leaf agents
    hierarchy_level: int = 0                     # 0 = Tier B (team lead), 1+ = sub-agent
    is_team_lead: bool = False                   # True when agent has children

# New model for sub-agent boundary proposals
@dataclass
class SubAgentBoundary:
    parent_agent_id: str                # Which Tier B agent this subdivides
    sub_agent_id: str                   # Proposed ID (e.g., "backend_api")
    root_path: Path                     # Sub-directory root
    include_patterns: list[str]         # File patterns (e.g., "api/**/*.py")
    estimated_context_kb: float
    file_count: int
    role_description: str               # Auto-generated: "REST API endpoints for..."
    key_modules: list[str]              # Central files within this sub-domain
    interfaces_with: list[str]          # Other sub-agent IDs it imports from
```

### When to Create Sub-Agents

Sub-agents are NOT always needed. They're created when a Tier B agent's domain exceeds manageable thresholds:

```python
class SubAgentPolicy:
    """Determines when a Tier B agent should be subdivided."""

    # Thresholds — only subdivide when BOTH conditions met
    min_files_to_split: int = 60        # Agent must cover 60+ files
    min_subdirs_to_split: int = 3       # With 3+ distinct subdirectories

    # Stop conditions — never create sub-agents smaller than these
    min_files_per_sub: int = 10         # Each sub-agent must have 10+ files
    max_sub_agents: int = 6             # No more than 6 sub-agents per team

    # Context efficiency threshold
    # If a single agent's context is < 80% of max_context_size,
    # it can handle the domain alone — no split needed
    context_utilization_threshold: float = 0.8

    def should_subdivide(self, agent: BaseAgent, graph: ImportGraph) -> bool:
        """
        Check if this Tier B agent's domain warrants sub-agents.

        Conditions (ALL must be true):
        1. Agent covers more than min_files_to_split files
        2. Agent's context_paths span 3+ distinct subdirectories
        3. The import graph shows distinct sub-clusters within this agent's domain
        4. The agent's estimated context exceeds context_utilization_threshold
        """

    def propose_subdivisions(
        self, agent: BaseAgent, graph: ImportGraph
    ) -> list[SubAgentBoundary]:
        """
        Propose how to split a Tier B agent into sub-agents.

        Strategy:
        1. Take the agent's context_paths (e.g., ["backend/"])
        2. Get all files in those paths
        3. Build a sub-graph: import graph restricted to this agent's files
        4. Cluster the sub-graph (same algorithm as top-level clustering)
        5. Each sub-cluster becomes a SubAgentBoundary
        6. Identify inter-sub-agent interfaces (imports across sub-clusters)
        7. Assign role descriptions based on file analysis

        Falls back to directory-based splitting if import graph is too sparse
        for meaningful sub-clusters.
        """
```

### SubAgentManager (extends existing `AutoAgentManager`)

```python
class SubAgentManager:
    """
    Manages the lifecycle of sub-Tier B agents.

    Responsibilities:
    - Evaluates which Tier B agents need subdivision
    - Creates sub-agents and establishes parent-child relationships
    - Re-evaluates subdivision as the codebase changes
    - Provides team-level querying (ask the team lead, it delegates to members)
    """

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

        Returns mapping of parent_agent_id → created sub-agent boundaries.

        Flow:
        1. For each Tier B agent:
           a. Check policy.should_subdivide()
           b. If yes: propose_subdivisions()
           c. Create sub-agents from boundaries
           d. Update parent's children_ids and is_team_lead
           e. Register sub-agents with AutoAgentManager
        2. Return report of what was created
        """

    def get_team(self, parent_agent_id: str) -> list[BaseAgent]:
        """Get all sub-agents for a team lead."""

    def get_team_lead(self, sub_agent_id: str) -> Optional[BaseAgent]:
        """Get the team lead for a sub-agent."""

    def route_to_sub_agent(
        self, parent_agent_id: str, file_path: str
    ) -> Optional[BaseAgent]:
        """
        Given a file path within a team lead's domain,
        route to the specific sub-agent that owns that file.

        Used by MergeCoordinator._get_owning_agent() to get the most
        specific owner: sub-agent > team lead > generic.
        """

    def team_query(
        self,
        parent_agent_id: str,
        query: str,
        delegate: bool = True,
    ) -> AgentResponse:
        """
        Query the team. Two modes:

        delegate=True (default):
          Team lead analyzes the query, delegates to the relevant
          sub-agent(s), and synthesizes their responses. This is how
          the ConflictRiskAnalyzer consults agents — it asks the team
          lead, who knows which team member to ask.

        delegate=False:
          Only the team lead answers, using its high-level domain knowledge.
          Faster, cheaper, used for quick assessments.
        """
```

### How Sub-Agents Integrate with Existing Features

**Routing**: When `KeywordRouter` or `TierAwareRouter` routes to a Tier B agent that has sub-agents, the response can either come from the team lead (quick, high-level) or be delegated to the relevant sub-agent (detailed, specific). The `team_query()` method handles this transparently.

**Import Graph**: The sub-agent's `context_paths` are a strict subset of the parent's. The sub-graph within those paths determines inter-sub-agent interfaces — which sub-agents import from each other. This is the "intra-team dependency map."

**Cross-Agent Context**: `CrossAgentContextManager` already injects context from related agents. With sub-agents, context injection becomes more precise: instead of injecting the entire backend domain, inject only the `backend_models` sub-agent's context when the query is about database schemas.

**Conflict Risk Analyzer**: When analyzing conflicts, the analyzer now asks the *sub-agent* that owns the specific files, not the broad team lead. This gives much more precise conflict assessments. If `backend/api/users.py` and `backend/services/auth.py` are touched by two tasks, the `backend_api_agent` and `backend_services_agent` can independently assess whether the changes conflict — they know their own files deeply.

---

## Component Design

### 1. TaskDecomposer — The Project Manager

The existing `QueryDecomposer` breaks queries into sub-questions per agent. This new decomposer breaks queries into **implementation tasks** — discrete units of work that produce code changes.

**The key insight**: A request doesn't have to *sound* complex to *be* complex. "Add an Excel upload button" sounds like one task, but in a real codebase it might span frontend (UI + file picker), backend (API endpoint + file parsing), Redis (job queue), and a Celery worker (background processing). A good PM catches this.

```python
@dataclass
class ImplementationTask:
    task_id: str
    description: str                    # "Add save button to toolbar"
    estimated_files: list[str]          # Predicted files to touch
    estimated_new_files: list[str]      # Files that may be created
    domain_agents: list[str]            # AgentHub agents with relevant knowledge
    complexity: Literal["trivial", "moderate", "complex"]
    estimated_tokens: int               # Expected Claude Code cost
    depends_on: list[str]              # task_ids that must complete first

@dataclass
class DecompositionResult:
    tasks: list[ImplementationTask]
    original_request: str
    appears_simple: bool                # True if request is one sentence / sounds simple
    actual_complexity: Literal["single", "multi_independent", "multi_dependent", "multi_mixed"]
    decomposition_reasoning: str        # Why tasks were split this way
    ceo_briefing: Optional[str]         # Message to CEO when complexity is hidden
    tokens_used: int
```

**How it works — two-pass decomposition:**

**Pass 1: Domain Survey (ask the team leads)**

Before the LLM even tries to decompose, the PM asks each domain agent: "Does this request touch your domain?" This catches complexity that a generic LLM might miss.

```python
def _survey_domains(
    self, request: str, agents: list[BaseAgent]
) -> list[DomainClaim]:
    """
    Quick check: which domains does this request involve?

    For each Tier B agent (and Tier A agent):
    - Route the request through KeywordRouter
    - If score > threshold: this domain is involved
    - Ask the agent (Haiku, quick): "Does this request require
      changes in your domain? If so, what files?"

    Example:
      Request: "Add an Excel upload button"

      UI agent (score 0.8): "Yes — need a file upload component
        and an upload button in the toolbar"
      API agent (score 0.6): "Yes — need a POST /api/upload endpoint
        to receive the file"
      Data agent (score 0.7): "Yes — need to parse the Excel file
        and validate/transform the data"
      Worker agent (score 0.5): "Yes — Excel parsing should be async,
        need a Celery task"
      Redis agent (score 0.4): "Yes — the async job needs a Redis
        queue entry and status tracking"

    Returns: 5 domain claims for a "simple" one-sentence request.
    """
```

**Pass 2: Structured Decomposition (the PM plans the work)**

With domain claims in hand, the LLM has the full picture and can decompose accurately.

```
You are a Project Manager analyzing a developer request. Your job is to
break it into discrete implementation tasks and assess the TRUE complexity.

REQUEST: {user_request}

DOMAIN SURVEY RESULTS (which teams are involved):
{for each claiming agent: agent_name, what they need to do, estimated files}

CODEBASE DOMAINS (from AgentHub):
{for each agent: agent_id, name, description, context_paths, keywords}

MODULE DEPENDENCY SUMMARY:
{import_graph.get_stats() + central_modules + cluster info}

BUSINESS CONTEXT (from Tier A agents, if any):
{relevant Tier A agent assessments of this request}

INSTRUCTIONS:
1. The domain survey tells you which teams are involved. Use this to
   determine the TRUE number of tasks — even if the request sounds simple.
2. A "simple" request that touches 4 domains is NOT simple. It's a
   multi-team project that the CEO needs to know about.
3. Break into the MINIMUM number of discrete tasks, respecting domain
   boundaries. Each task should be assignable to one team.
4. Identify dependencies: which tasks must complete before others can start?
5. If the request appears simple but is actually complex, include a
   ceo_briefing explaining WHY it's more work than it sounds.

Respond in JSON:
{
  "appears_simple": true/false,
  "actual_complexity": "single" | "multi_independent" | "multi_dependent" | "multi_mixed",
  "ceo_briefing": "This looks like a simple button, but it actually requires
    changes across 4 domains: frontend (upload UI), backend (API endpoint),
    data processing (Excel parser), and async infrastructure (Celery task +
    Redis queue). Estimated time: ~4 min parallel vs ~12 min sequential.",
  "tasks": [
    {
      "task_id": "task_1",
      "description": "Add Excel upload button and file picker to toolbar",
      "estimated_files": ["src/components/Toolbar.tsx", "src/components/FileUpload.tsx"],
      "estimated_new_files": ["src/components/FileUpload.tsx"],
      "domain_agents": ["ui_agent"],
      "complexity": "moderate",
      "depends_on": []
    },
    {
      "task_id": "task_2",
      "description": "Create POST /api/upload endpoint for receiving Excel files",
      "estimated_files": ["src/api/routes/upload.py", "src/api/schemas/upload.py"],
      "estimated_new_files": ["src/api/routes/upload.py", "src/api/schemas/upload.py"],
      "domain_agents": ["api_agent"],
      "complexity": "moderate",
      "depends_on": []
    },
    {
      "task_id": "task_3",
      "description": "Implement Excel parsing and data validation logic",
      "estimated_files": ["src/services/excel_parser.py", "src/models/upload_job.py"],
      "estimated_new_files": ["src/services/excel_parser.py"],
      "domain_agents": ["data_agent"],
      "complexity": "complex",
      "depends_on": []
    },
    {
      "task_id": "task_4",
      "description": "Set up Celery task for async Excel processing with Redis queue",
      "estimated_files": ["src/workers/tasks.py", "src/workers/excel_task.py", "config/celery.py"],
      "estimated_new_files": ["src/workers/excel_task.py"],
      "domain_agents": ["worker_agent", "redis_agent"],
      "complexity": "moderate",
      "depends_on": ["task_3"]
    }
  ],
  "reasoning": "Tasks 1-3 are independent (UI, API, and parsing logic can be
    built in parallel). Task 4 depends on Task 3 because the Celery task wraps
    the parsing logic. Total: 3 parallel + 1 sequential."
}
```

**Why two passes matter:**

A single-pass LLM decomposition often misses domains. It sees "upload button" and thinks "frontend task." The domain survey catches the backend, data, and infrastructure implications *before* decomposition starts. This is exactly what a good PM does — they don't just read the ticket, they walk the floor and ask each team: "Does this affect you?"

**CEO Briefing — managing expectations:**

When a seemingly simple request turns out to be complex, the PM doesn't just silently plan a 4-team project. It tells the CEO:

```
"You asked for an Excel upload button. This is actually a 4-domain project:

 1. Frontend: Upload UI + file picker          (~2 min)
 2. Backend: API endpoint for file receiving   (~2 min)
 3. Data: Excel parsing + validation           (~3 min)
 4. Infrastructure: Celery task + Redis queue   (~2 min, depends on #3)

 Tasks 1-3 can run in parallel. Task 4 runs after Task 3.
 Estimated: ~5 min parallel vs ~9 min sequential.
 Risk: LOW (no file overlaps between domains).

 Proceed?"
```

This transparency is the whole point of the company model. The CEO shouldn't be surprised when a "quick feature" takes longer than expected. The PM's job is to surface the real scope early.

---

### 2. ConflictRiskAnalyzer

This is the **critical safety component**. It takes the decomposed tasks and determines which can safely run in parallel.

```python
@dataclass
class FileOverlap:
    file_path: str
    tasks_touching: list[str]           # task_ids
    overlap_type: Literal["direct", "shared_import", "shared_type", "shared_config"]
    risk_level: Literal["none", "low", "medium", "high"]

@dataclass
class ParallelizationPlan:
    parallel_groups: list[list[str]]    # Groups of task_ids safe to run together
    sequential_tasks: list[str]         # task_ids that must run alone
    file_overlaps: list[FileOverlap]    # Detected overlaps
    overall_risk: Literal["safe", "caution", "unsafe"]
    confidence: float                   # 0.0 - 1.0
    reasoning: str
    estimated_speedup: float            # e.g., 1.8x
    estimated_total_tokens: int

class ConflictRiskAnalyzer:
    def __init__(self, import_graph: ImportGraph, hub: AgentHub):
        self._graph = import_graph
        self._hub = hub

    def analyze(
        self,
        tasks: list[ImplementationTask],
        consult_agents: bool = True,
    ) -> ParallelizationPlan:
        """
        Analysis pipeline — two phases: static analysis, then agent consultation.

        PHASE A: STATIC ANALYSIS (fast, free, no LLM calls)

        1. FILE OVERLAP CHECK
           - For each pair of tasks, check if estimated_files overlap
           - Direct overlap → HIGH risk

        2. IMPORT DEPENDENCY CHECK
           - For each file in task A, get all imports (transitive, depth=2)
           - Check if any import target is in task B's file set
           - Shared import → MEDIUM risk

        3. SHARED TYPE/MODEL CHECK
           - Identify "hub" modules (high in-degree in import graph)
           - If two tasks both touch files that import the same hub → LOW risk
           - Special attention to: models/, types/, schemas/, config/

        4. SHARED STATE CHECK
           - Database models, global config, environment files
           - If both tasks might modify shared state → HIGH risk

        PHASE B: DOMAIN AGENT CONSULTATION (semantic, LLM-powered)

        5. AGENT CONFLICT REVIEW
           - For each task, identify which domain agents own the affected files
           - Ask each relevant agent: "Task X plans to modify [files].
             Task Y plans to modify [files]. From your domain knowledge,
             would these changes conflict?"
           - Agents can flag hidden conflicts the import graph misses:
             * Shared database tables/schemas
             * Event bus / message queue side effects
             * Global state or singletons
             * API contract changes that affect callers
             * Shared configuration or environment variables
           - Agent responses upgrade/downgrade risk levels from Phase A

        PHASE C: TIER A BUSINESS REVIEW (semantic, LLM-powered)

        6. BUSINESS IMPACT CHECK
           - Ask Tier A agents (if any are registered): "These tasks
             affect your domain. From a business perspective, is running
             them in parallel risky?"
           - Tier A agents flag business-level conflicts the code can't see:
             * Both tasks affect a customer-facing metric (e.g., ROAS)
             * One task changes behavior that a client contract depends on
             * Both tasks touch a compliance-sensitive workflow
             * A task changes the data model that the business logic relies on
           - Tier A concerns upgrade risk (same asymmetry as Tier B agents)
           - If no Tier A agents exist → skip this phase

        PHASE D: PLAN BUILDING

        7. BUILD PARALLELIZATION PLAN
           - Combine static risk + agent risk assessments
           - Agent concerns override static "safe" → bump to MEDIUM/HIGH
           - Agent "no concern" does NOT override static HIGH → stays HIGH
           - Group tasks with no/low overlap into parallel groups
           - Tasks with medium+ overlap → sequential or same group with caution
           - Tasks with explicit depends_on → respect ordering
        """

    def _consult_domain_agents(
        self,
        tasks: list[ImplementationTask],
        static_overlaps: list[FileOverlap],
    ) -> list[AgentConflictAssessment]:
        """
        Ask domain agents about potential conflicts between tasks.

        WITH SUB-AGENTS: When a Tier B agent has sub-agents, the analyzer
        routes to the specific sub-agent that owns the affected files.
        This gives much more precise assessments than asking the broad
        team lead.

        Example: If Task A touches backend/api/users.py and Task B
        touches backend/services/auth.py, the analyzer asks:
        - backend_api_agent about Task A's files (not the generic backend_agent)
        - backend_services_agent about Task B's files
        Then asks the backend_agent (team lead) if the cross-sub-domain
        interaction is safe.

        For each pair of tasks that share a domain agent:
        1. Route to the most specific agent (sub-agent > team lead)
        2. Ask it: "Would implementing {task_A} and {task_B}
           simultaneously cause conflicts in your domain?"
        3. If tasks span sub-agents within the same team:
           Also ask the team lead about cross-sub-domain conflicts
        4. Parse responses into structured assessments

        Uses Haiku for speed/cost — this is a quick judgment call,
        not deep analysis.

        Returns list of AgentConflictAssessment with:
        - agent_id: which agent raised the concern
        - task_pair: (task_id_a, task_id_b)
        - has_concern: bool
        - concern_description: str (e.g., "Both tasks write to the users table")
        - risk_upgrade: Optional risk level to upgrade to
        """

    def _build_agent_conflict_prompt(
        self,
        agent: BaseAgent,
        task_a: ImplementationTask,
        task_b: ImplementationTask,
    ) -> str:
        """
        Build prompt for domain agent conflict review.

        Template:
        '''
        You are the domain expert for {agent.name} ({agent.description}).
        Your domain covers these files: {agent.context_paths}

        Two implementation tasks are being considered for PARALLEL execution
        (simultaneous, on separate git branches, then merged):

        TASK A: {task_a.description}
          Will modify: {task_a.estimated_files}

        TASK B: {task_b.description}
          Will modify: {task_b.estimated_files}

        From your knowledge of this domain, would these two tasks
        conflict if implemented simultaneously? Consider:
        - Shared database tables or schemas
        - Shared state, singletons, or global variables
        - Event handlers, message queues, or side effects
        - API contracts or interface changes
        - Shared configuration files
        - Test fixtures or test data

        Respond in JSON:
        {
          "has_concern": true/false,
          "concern": "description of the conflict risk (or empty)",
          "severity": "none" | "low" | "medium" | "high",
          "affected_files": ["files that would cause the conflict"]
        }
        '''
        """

    def _get_transitive_imports(
        self, file_path: str, depth: int = 2
    ) -> set[str]:
        """Walk import graph to find all files reachable within N hops."""

    def _identify_shared_hubs(
        self, task_a_files: list[str], task_b_files: list[str]
    ) -> list[str]:
        """Find hub modules imported by both task file sets."""

    def _check_model_overlap(
        self, task_a_files: list[str], task_b_files: list[str]
    ) -> list[str]:
        """Check for shared database models, type definitions, schemas."""
```

**New model for agent assessments:**

```python
@dataclass
class AgentConflictAssessment:
    agent_id: str
    agent_name: str
    task_pair: tuple[str, str]          # (task_id_a, task_id_b)
    has_concern: bool
    concern_description: str
    severity: Literal["none", "low", "medium", "high"]
    affected_files: list[str]
    tokens_used: int
```

**Risk classification logic (static + Tier B agents + Tier A agents):**

```
For each pair of tasks (A, B):

  # Phase A: Static analysis (fast, free, deterministic)
  direct_overlap = A.estimated_files ∩ B.estimated_files
  import_overlap = transitive_imports(A.files, depth=2) ∩ B.files
  hub_overlap = shared_hub_modules(A.files, B.files)
  model_overlap = shared_models(A.files, B.files)

  if direct_overlap:
      static_risk = HIGH
  elif model_overlap:
      static_risk = HIGH
  elif import_overlap:
      static_risk = MEDIUM
  elif hub_overlap:
      static_risk = LOW
  else:
      static_risk = NONE

  # Phase B: Tier B agent consultation (can only UPGRADE risk)
  tierB_assessments = consult_tierB_agents_for_pair(A, B)
  tierB_risk = max(a.severity for a in tierB_assessments) or NONE

  # Phase C: Tier A business review (can only UPGRADE risk)
  tierA_assessments = consult_tierA_agents_for_pair(A, B)
  tierA_risk = max(a.severity for a in tierA_assessments) or NONE

  # Combined risk: take the highest across all three layers
  final_risk = max(static_risk, tierB_risk, tierA_risk)

  # Plan assignment
  if final_risk == HIGH:
      → must be sequential
  elif final_risk == MEDIUM:
      → parallel with caution (strict file scoping, post-merge verification)
      → OR ask CEO if Tier A flagged a business concern
  elif final_risk == LOW:
      → parallel (shared read-only dependencies, normal merge)
  else:
      → fully parallel

  # Key principle: ALL agents (Tier A and Tier B) can only UPGRADE risk.
  # Static HIGH stays HIGH regardless of what any agent says.
  # But Tier A can upgrade NONE → MEDIUM if both tasks affect a
  # business-critical workflow, even if the files don't overlap at all.
```

**Why agents can only upgrade risk:**

Static analysis is deterministic — if two tasks modify the same file, that's a hard fact. An agent saying "I think it's fine" doesn't change that. But if static analysis says NONE (no file overlap) and an agent says "actually, both tasks write to the same database table via different ORM models," that's critical information the graph can't see. The asymmetry is intentional: false negatives (missing a conflict) are far worse than false positives (unnecessary sequential execution).

**No silent fallbacks — always a CEO shot-call.**

The PM never silently decides to run sequentially. Even when risk is HIGH, the PM presents its assessment and *recommends* sequential, but the CEO makes the call. And when the call is sequential, the PM still plans the execution order — which team goes first, what dependencies exist, how each team's output feeds into the next.

```
HIGH risk example:

PM: "⚠ Risk is HIGH. Both tasks modify the auth middleware.
     I strongly recommend running sequentially.

     Recommended order:
       1. Team A: Refactor auth middleware (~3 min)
       2. Team B: Update API endpoints for new auth (~4 min)
          (depends on Team A's new interface)

     Total: ~7 min sequential.

     Override to parallel? (not recommended)"
```

The PM always has a plan, whether it's parallel or sequential.

---

### 3. BranchOrchestrator

Manages git branches and spawns parallel Claude Code sessions.

```python
@dataclass
class SessionSpec:
    task: ImplementationTask
    branch_name: str
    scoped_files: list[str]             # Files this session is allowed to modify
    scoped_dirs: list[str]              # Directories this session should focus on
    context_from_agents: list[str]      # Agent IDs to inject context from
    prompt: str                         # The scoped prompt for this session
    timeout_seconds: int

@dataclass
class SessionResult:
    task_id: str
    branch_name: str
    success: bool
    files_changed: list[str]            # Actual files modified (from git diff)
    files_created: list[str]            # New files created
    stdout: str                         # Claude Code output
    tokens_used: int
    time_seconds: float
    test_results: Optional[dict]        # If tests were run
    error: Optional[str]

class BranchOrchestrator:
    def __init__(
        self,
        project_root: str,
        hub: AgentHub,
        max_parallel: int = 3,
        claude_model: str = "claude-opus-4-6",
        session_timeout: int = 300,     # 5 minutes per session
    ):
        self._root = project_root
        self._hub = hub
        self._base_branch: str = ""     # Captured at start

    def execute_plan(
        self, plan: ParallelizationPlan, tasks: list[ImplementationTask]
    ) -> list[SessionResult]:
        """
        Execution flow:

        1. Capture base state
           - Record current branch/commit as base
           - Ensure working tree is clean (fail if dirty)

        2. For each parallel group:
           a. Create branches: parallel/<task_id> from base
           b. Build scoped prompts for each task
           c. Spawn Claude Code sessions in parallel
           d. Wait for all to complete (or timeout)
           e. Collect results

        3. For sequential tasks:
           - Run one at a time on their own branches
           - After each, merge into base before next

        4. Return all SessionResults
        """

    def _create_branch(self, task_id: str) -> str:
        """Create and checkout a new branch: parallel/{task_id}"""

    def _build_scoped_prompt(
        self, task: ImplementationTask, scoped_files: list[str]
    ) -> str:
        """Build a Claude Code prompt that:
        - Describes the specific task
        - Lists the files the session should focus on
        - Injects relevant context from AgentHub domain agents
        - Instructs to run tests when done
        - Instructs NOT to modify files outside the scope
        """

    def _spawn_session(self, spec: SessionSpec) -> SessionResult:
        """
        Spawn a Claude Code session via CLI:

        cd {project_root}
        git checkout {spec.branch_name}
        claude --print "{spec.prompt}" \
               --output-format json \
               --model {claude_model}

        OR for interactive (API-based):

        Use Anthropic API directly with tool_use enabled,
        providing file read/write/bash tools scoped to spec.scoped_dirs.

        Returns SessionResult with files changed, tokens, timing.
        """

    def _get_files_changed(self, branch_name: str) -> list[str]:
        """git diff --name-only {base_branch}...{branch_name}"""
```

**Scoped prompt template:**

```
You are working on a specific task within a larger codebase.
Other teams are working on other tasks simultaneously on separate branches.

TASK: {task.description}

YOUR TEAM'S FILES (you should modify these):
{task.estimated_files}

OTHER TEAMS' FILES (do NOT modify these):
{other_tasks_files}

RELEVANT TECHNICAL CONTEXT (from your Tier B team):
{injected context from team lead + relevant sub-agents}

BUSINESS CONTEXT (from Tier A agents):
{injected context from relevant Tier A agents — e.g., "This component
 is part of the ad analytics pipeline. ROAS calculation accuracy is
 critical. The client expects campaign budget totals to match the
 dashboard within 0.01%."}

IMPORTANT CONSTRAINTS:
- Focus ONLY on the task described above
- Only modify files within YOUR TEAM'S FILES listed above
- Do NOT modify any file listed under OTHER TEAMS' FILES
- If you discover you NEED to modify a file outside your scope:
  1. STOP modifying that file
  2. Note it clearly with the tag: [BOUNDARY_CROSSING: {file_path} — {reason}]
  3. Continue implementing what you can without that modification
  4. The orchestrator will coordinate with the other team
- Run relevant tests when done (tests may fail if you noted boundary crossings)

Please implement this task and run tests to verify.
```

---

### 4a. Mid-Execution Escalation Protocol

In a real company, a team doesn't just silently modify another team's deliverable. If Team A discovers mid-implementation that it needs to change Team B's code, it communicates. This protocol mirrors that behavior.

```python
@dataclass
class BoundaryCrossing:
    """Detected when a session needs files outside its scope."""
    session_task_id: str                # Which session hit the boundary
    requesting_agent: str               # Domain agent making the request
    target_file: str                    # The out-of-scope file it needs
    owning_agent: Optional[str]         # Domain agent that owns that file
    reason: str                         # Why it needs the file
    proposed_change: Optional[str]      # What it wants to change (if known)
    blocking: bool                      # True if session can't proceed without it

@dataclass
class BoundaryCrossingResolution:
    crossing: BoundaryCrossing
    approved: bool
    resolution_type: Literal[
        "approved_as_is",               # Owning agent says go ahead
        "approved_with_modification",   # Owning agent says change it this way instead
        "deferred_to_merge",            # Handle it during merge phase
        "escalated_to_ceo",             # Agents can't agree, ask human
        "rejected"                      # Owning agent says don't touch it
    ]
    modified_change: Optional[str]      # If approved_with_modification
    reasoning: str

class MidExecutionEscalationHandler:
    """
    Monitors running sessions for boundary crossing signals.

    In the company model, this is the "inter-team communication channel."
    When Team A says "I need to touch Team B's file," this handler:
    1. Identifies Team B's domain agent
    2. Asks Team B: "Team A wants to modify your file. Here's why. OK?"
    3. Team B responds: approve, modify, defer, or reject
    4. The resolution is fed back to Team A's session (or the merge phase)
    """

    def __init__(self, hub: AgentHub, orchestrator: BranchOrchestrator):
        self._hub = hub
        self._orchestrator = orchestrator

    def detect_boundary_crossings(
        self, session_output: str, task: ImplementationTask
    ) -> list[BoundaryCrossing]:
        """
        Parse session output for [BOUNDARY_CROSSING: ...] tags.

        The scoped prompt instructs Claude Code to emit these tags
        instead of silently modifying out-of-scope files.
        """

    def resolve_crossing(
        self, crossing: BoundaryCrossing
    ) -> BoundaryCrossingResolution:
        """
        Inter-team negotiation for a boundary crossing.

        Flow:
        1. Find the owning agent for the target file
        2. Ask the owning agent:
           '''
           You are the domain expert for {owning_agent.name}.
           You own the file: {crossing.target_file}

           Another team ({crossing.requesting_agent}) is working on:
             {task.description}

           They've discovered they need to modify YOUR file because:
             {crossing.reason}

           Their proposed change:
             {crossing.proposed_change}

           As the file owner, do you:
           1. APPROVE — the change is safe and consistent with your domain
           2. MODIFY — approve with a different approach (explain)
           3. DEFER — let it be handled during the merge phase
           4. REJECT — the change would break your domain (explain why)

           Respond in JSON:
           {
             "decision": "approve" | "modify" | "defer" | "reject",
             "reasoning": "...",
             "modified_change": "..." (only if decision=modify),
             "confidence": 0.0-1.0
           }
           '''
        3. If owning agent confidence < 0.6 → escalate to CEO
        4. Return resolution
        """

    def handle_blocking_crossing(
        self, crossing: BoundaryCrossing, resolution: BoundaryCrossingResolution
    ) -> str:
        """
        When a crossing is blocking (session can't proceed without it):

        If approved/modified:
          → Apply the change to the other team's branch
          → Let the session continue (or note it for merge)
        If deferred:
          → Session continues with a workaround (if possible)
          → Full resolution happens at merge time
        If rejected:
          → Session must find an alternative approach
          → If no alternative: escalate to CEO
        If escalated:
          → Pause session, present to human, resume after decision
        """
```

**Why this matters:**

Without mid-execution escalation, the only conflict detection happens at merge time — after all sessions have finished. This is late. In a real company, a team that discovers mid-project they need another team's help raises it immediately, not at the final integration meeting. Early detection = cheaper resolution.

**Practical limitation (v1):**

In v1 (CLI mode), we can't pause a running Claude Code session. So boundary crossings are detected post-execution by parsing the session output. The `[BOUNDARY_CROSSING]` tags in the output let us identify what needs negotiation before merge.

In v2 (API mode), we can implement true real-time escalation: the session pauses when it encounters an out-of-scope file, the handler negotiates, and the session resumes with the resolution.

---

### 4b. MergeCoordinator (The "Merge Committee")

Handles combining the work from parallel sessions. In the company model, this is the **inter-team integration process** — domain agents own the resolution, not a generic mediator.

```python
@dataclass
class MergeConflict:
    file_path: str
    conflict_type: Literal["textual", "semantic", "scope_violation"]
    description: str
    branch_a: str
    branch_b: str
    diff_a: str                         # Changes from branch A
    diff_b: str                         # Changes from branch B
    owning_agent: Optional[str]         # Domain agent that owns this file
    auto_resolvable: bool
    suggested_resolution: Optional[str]

@dataclass
class DomainResolutionProposal:
    """A domain agent's proposed resolution for a merge conflict."""
    agent_id: str
    agent_name: str
    conflict_file: str
    proposed_resolution: str            # The actual resolved code
    reasoning: str                      # Why this resolution is correct
    confidence: float                   # 0.0-1.0, how sure the agent is
    side_effects: list[str]             # Other files that might need updating
    needs_ceo: bool                     # Agent itself says "I'm not sure, ask the human"

@dataclass
class MergeResult:
    success: bool
    merged_branch: str
    conflicts: list[MergeConflict]
    resolutions: list[DomainResolutionProposal]  # How conflicts were resolved
    files_merged: list[str]
    test_results: Optional[dict]        # Post-merge test run
    needs_user_input: bool
    escalation_reason: Optional[str]    # Why the CEO was needed (if applicable)
    summary: str

class MergeCoordinator:
    def __init__(
        self,
        project_root: str,
        hub: AgentHub,
        client: anthropic.Anthropic,
    ):
        self._root = project_root
        self._hub = hub
        self._client = client

    def merge_results(
        self,
        base_branch: str,
        session_results: list[SessionResult],
        task_agent_map: dict[str, list[str]],   # task_id → owning agent_ids
        run_tests: bool = True,
    ) -> MergeResult:
        """
        Merge strategy (company model):

        1. SCOPE VALIDATION (HR check)
           - For each session result, verify files_changed matches expected scope
           - Flag any scope violations (files modified outside task scope)
           - If a session touched another team's files → boundary crossing protocol

        2. SEQUENTIAL MERGE
           - Create merge branch: parallel/merged from base
           - For each session (ordered by file count, smallest first):
             a. git merge parallel/{task_id} --no-edit
             b. If textual conflict: identify owning domain agent
             c. Send conflict to owning agent for resolution proposal
             d. If agent proposes with high confidence → apply resolution
             e. If agent is unsure or agents disagree → escalate to CEO

        3. POST-MERGE VERIFICATION (QA)
           - git diff base..parallel/merged (full diff review)
           - If run_tests: execute test suite on merged branch
           - If tests fail: identify semantic conflicts, consult domain agents

        4. TIERED CONFLICT RESOLUTION
           Level 1 — Agent Self-Resolution:
             Simple conflicts (import ordering, non-overlapping additions)
             The owning domain agent resolves unilaterally.

           Level 2 — Inter-Agent Negotiation:
             Two agents' domains overlap on the conflict file.
             Both propose resolutions. If they agree → apply.
             If they disagree → escalate.

           Level 3 — CEO Escalation:
             Agents disagree, or agent confidence < 0.6,
             or the conflict involves architectural decisions.
             Present to user with agent proposals + reasoning.

        5. RESULT
           - If clean: merge into base, return success
           - If needs CEO: return MergeResult with proposals for user decision
        """

    def _get_owning_agent(self, file_path: str) -> Optional[BaseAgent]:
        """
        Determine which domain agent 'owns' a file.
        Returns the MOST SPECIFIC owner (sub-agent over team lead).

        Uses the AgentHub's routing system + sub-agent hierarchy:
        1. Check if any sub-agent's context_paths directly contain this file
           → If yes, return the sub-agent (most specific owner)
        2. Check if any Tier B agent's context_paths contain this file
           → If yes, and it has sub-agents, use SubAgentManager.route_to_sub_agent()
           → If yes, and no sub-agents, return the Tier B agent directly
        3. Fall back to KeywordRouter.get_all_scores() on the file path
        4. If multiple agents claim ownership → the most specific one

        Example:
          file_path = "backend/api/users.py"
          → Tier B backend_agent owns "backend/"
          → Sub-agent backend_api_agent owns "backend/api/"
          → Returns backend_api_agent (more specific)

        Returns the domain agent or None (for files no agent claims).
        """

    def _request_agent_resolution(
        self,
        agent: BaseAgent,
        conflict: MergeConflict,
    ) -> DomainResolutionProposal:
        """
        Ask a domain agent to propose a merge conflict resolution.

        The agent gets:
        - Its full domain context (context_paths, injected knowledge)
        - The conflicting file's full content
        - Both branches' diffs
        - What each branch was trying to accomplish

        Prompt template:
        '''
        You are the domain expert for {agent.name} ({agent.description}).
        You own the file: {conflict.file_path}

        Two parallel implementation tasks modified this file simultaneously,
        causing a merge conflict:

        BRANCH A ({conflict.branch_a}) — intended to:
          {task_a.description}
          Changes:
          {conflict.diff_a}

        BRANCH B ({conflict.branch_b}) — intended to:
          {task_b.description}
          Changes:
          {conflict.diff_b}

        As the domain expert, propose a resolution that correctly
        integrates BOTH sets of changes. Consider:
        - Both tasks' intents should be preserved
        - The resolved code should be consistent with the codebase style
        - Flag any side effects that might need updating in other files

        Respond in JSON:
        {
          "proposed_resolution": "<the resolved code for this file>",
          "reasoning": "<why this resolution correctly integrates both changes>",
          "confidence": <0.0-1.0>,
          "side_effects": ["<other files that might need updating>"],
          "needs_human_review": <true/false — set true if you're unsure>
        }
        '''
        """

    def _negotiate_between_agents(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        conflict: MergeConflict,
    ) -> tuple[DomainResolutionProposal, Optional[DomainResolutionProposal]]:
        """
        When a conflict file is claimed by two domain agents:

        1. Both agents independently propose resolutions
        2. Compare proposals:
           - If equivalent (same code, different words) → use either
           - If compatible (both can be combined) → merge proposals
           - If contradictory → return both for CEO escalation

        This mirrors real inter-team negotiation:
        both teams present their case, and if they can't agree,
        it goes up to the CEO.
        """

    def _should_escalate_to_ceo(
        self,
        conflict: MergeConflict,
        proposals: list[DomainResolutionProposal],
    ) -> tuple[bool, str]:
        """
        Determine if a conflict requires human (CEO) intervention.

        Escalate when:
        - No owning agent found (orphan file, no domain expertise)
        - Agent confidence < 0.6
        - Multiple agents disagree on resolution
        - Conflict involves architectural patterns (new abstractions, API changes)
        - Agent explicitly sets needs_ceo=True
        - Scope violation (a session modified files outside its team's domain)

        Returns (should_escalate, reason).
        """

    def _build_ceo_escalation_message(
        self,
        conflict: MergeConflict,
        proposals: list[DomainResolutionProposal],
    ) -> str:
        """
        Build a clear, concise message for the human developer.

        Format:
        '''
        ⚠ Merge conflict needs your decision.

        FILE: {conflict.file_path}
        CONFLICT TYPE: {conflict.conflict_type}

        TECHNICAL ASSESSMENT (Tier B):
        Team A ({agent_a.name}) proposes:
          {proposal_a.reasoning}
          Confidence: {proposal_a.confidence}

        Team B ({agent_b.name}) proposes:
          {proposal_b.reasoning}
          Confidence: {proposal_b.confidence}

        BUSINESS CONTEXT (Tier A):
        {tierA_agent.name} says:
          "{business_impact_assessment}"
          (e.g., "This file handles ROAS calculation — our core metric.
           Team A's resolution preserves the existing formula.")

        Options:
        1. Accept Team A's resolution
        2. Accept Team B's resolution
        3. Provide your own resolution
        4. Abort and run tasks sequentially
        '''
        """

    def _analyze_semantic_conflict(
        self,
        test_output: str,
        diffs: dict[str, str],
        task_agent_map: dict[str, list[str]],
    ) -> list[MergeConflict]:
        """
        When tests fail after merge but there were no textual conflicts,
        the combined changes are semantically incompatible.

        Instead of a generic LLM call, consults the domain agents that
        own the failing test files and the modified source files:

        1. Parse test failures to identify affected test files
        2. Find domain agents that own those test files + source files
        3. Ask each agent: "These tests fail after merging. From your
           domain knowledge, what's the likely cause and fix?"
        4. Compile agent assessments into structured conflicts

        Returns list of semantic conflicts with agent-proposed resolutions.
        """

    def _detect_scope_violations(
        self,
        result: SessionResult,
        expected_scope: list[str],
        task_agent_map: dict[str, list[str]],
    ) -> list[MergeConflict]:
        """
        Check if a session modified files outside its designated scope.

        In the company model, this is a boundary crossing:
        - Identify which domain agent owns the out-of-scope file
        - Notify that agent: "Team X modified your file. Is this acceptable?"
        - Agent can approve (minor, non-breaking change) or flag (escalate)

        Always escalates to CEO if:
        - The out-of-scope file has no owning agent
        - The owning agent says the change is problematic
        """
```

---

### 5. ParallelSessionManager (Top-Level Orchestrator)

The main entry point that ties everything together.

```python
@dataclass
class ParallelExecutionConfig:
    max_parallel_sessions: int = 3
    session_timeout_seconds: int = 300
    claude_model: str = "claude-opus-4-6"
    auto_resolve_conflicts: bool = True
    run_tests_after_merge: bool = True
    test_command: Optional[str] = None  # Auto-detect if None
    auto_proceed_threshold: Literal["safe", "caution"] = "caution"  # Auto-proceed up to this risk level; above → ask CEO

@dataclass
class ParallelExecutionResult:
    success: bool
    tasks: list[ImplementationTask]
    plan: ParallelizationPlan
    session_results: list[SessionResult]
    merge_result: MergeResult
    total_time_seconds: float
    sequential_estimate_seconds: float  # What it would have taken sequentially
    speedup: float                      # sequential / actual
    total_tokens: int
    trace: ParallelExecutionTrace       # Full observability

@dataclass
class ParallelExecutionTrace:
    """Comprehensive trace for debugging and benchmarking."""
    decomposition_time_ms: int
    analysis_time_ms: int
    session_times: dict[str, int]       # task_id -> ms
    merge_time_ms: int
    total_time_ms: int
    decomposition_tokens: int
    analysis_tokens: int
    session_tokens: dict[str, int]      # task_id -> tokens
    merge_tokens: int
    total_tokens: int
    parallel_groups: list[list[str]]
    conflicts_found: int
    conflicts_auto_resolved: int
    scope_violations: int
    test_pass_rate: Optional[float]

class ParallelSessionManager:
    def __init__(
        self,
        hub: AgentHub,
        config: ParallelExecutionConfig = None,
    ):
        self._hub = hub
        self._config = config or ParallelExecutionConfig()
        self._import_graph = hub._import_graph
        self._decomposer = TaskDecomposer(hub._client)
        self._analyzer = ConflictRiskAnalyzer(self._import_graph, hub)
        self._orchestrator = BranchOrchestrator(
            project_root=...,
            hub=hub,
            max_parallel=config.max_parallel_sessions,
            claude_model=config.claude_model,
        )
        self._merger = MergeCoordinator(
            project_root=...,
            hub=hub,
            client=hub._client,
        )

    def execute(self, request: str) -> ParallelExecutionResult:
        """
        Full pipeline (company model):

        1. DECOMPOSE (Project Manager breaks down the work)
           tasks = decomposer.decompose(request)
           If single task → skip parallelization, run directly

        2. ANALYZE (Risk Assessment evaluates inter-team dependencies)
           plan = analyzer.analyze(tasks)  # static + domain agent consultation

           Risk-based routing:
           ┌─────────────────────────────────────────────────────────────┐
           │ Risk     │ Action                                          │
           │──────────│─────────────────────────────────────────────────│
           │ SAFE     │ Proceed automatically. Teams work independently.│
           │ CAUTION  │ Proceed with merge verification. Teams work     │
           │          │ independently but with boundary monitoring.     │
           │ MEDIUM   │ ASK THE CEO. Present risk + agent concerns.     │
           │          │ PM recommends parallel with caution.            │
           │ HIGH     │ ASK THE CEO. Present risk + agent concerns.     │
           │          │ PM recommends sequential + provides order plan. │
           └─────────────────────────────────────────────────────────────┘

           The PM NEVER silently falls back to sequential. Even at HIGH
           risk, the CEO gets a shot-call with the PM's recommendation.
           And when the call is sequential, the PM still plans the order
           — which team goes first, what dependencies exist, how each
           team's output feeds into the next.

        3. CEO CONFIRMATION (when risk is MEDIUM or user requested confirm=True)
           Present plan to user with domain agent assessments:

           "I've broken this into {N} tasks:

            Team A (UI agent): {description}
              → Files: {files}

            Team B (Data agent): {description}
              → Files: {files}

            ⚠ Risk Assessment: MEDIUM
            UI agent says: 'Both tasks import from shared UserContext.
              If Task A changes the context shape, Task B's chart
              component would break.'

            Options:
            1. Run in parallel anyway (will verify at merge)
            2. Run sequentially (safer, ~{X}s slower)
            3. Modify task breakdown"

        4. EXECUTE (Teams work in parallel on their branches)
           results = orchestrator.execute_plan(plan, tasks)

        5. BOUNDARY CROSSING CHECK (Inter-team communication)
           For each session result:
             crossings = escalation_handler.detect_boundary_crossings(result)
             For each crossing:
               resolution = escalation_handler.resolve_crossing(crossing)
               Apply resolution or queue for merge phase

        6. MERGE (Merge Committee integrates the work)
           merge = merger.merge_results(base_branch, results, task_agent_map)

        7. HANDLE CONFLICTS (Tiered escalation)
           For each conflict:
             Level 1: Owning domain agent proposes resolution
             Level 2: If two agents disagree, negotiate
             Level 3: If still unresolved, escalate to CEO with proposals

           If merge.needs_user_input:
               Present conflicts + agent proposals to user
               Wait for user decisions
               Re-merge with resolutions

        8. VERIFY (QA — post-merge test gate)
           If merge.test_results and not all passing:
               Consult domain agents about test failures
               If agents can fix it → spawn resolution session
               If agents can't → escalate to CEO

        9. RETURN
           Return ParallelExecutionResult with full trace
        """
```

---

## MCP Server Integration

### New MCP Tool: `agenthub_parallel_execute`

```python
{
    "name": "agenthub_parallel_execute",
    "description": "Analyze a multi-part request and execute tasks in parallel on separate git branches. Uses import graph analysis to determine safe parallelization. Returns merged result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The multi-part development request"
            },
            "confirm": {
                "type": "boolean",
                "default": true,
                "description": "If true, shows plan and waits for confirmation before executing"
            },
            "max_parallel": {
                "type": "integer",
                "default": 3,
                "description": "Maximum concurrent sessions"
            }
        },
        "required": ["request"]
    }
}
```

### New MCP Tool: `agenthub_analyze_parallelism`

Dry-run tool that shows the plan without executing.

```python
{
    "name": "agenthub_analyze_parallelism",
    "description": "Analyze whether a multi-part request can be safely parallelized. Returns task decomposition, file overlap analysis, and estimated speedup WITHOUT executing anything.",
    "input_schema": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The development request to analyze"
            }
        },
        "required": ["request"]
    }
}
```

### Updated MCP Tool: `agenthub_resolve_conflict`

For interactive conflict resolution during merge.

```python
{
    "name": "agenthub_resolve_conflict",
    "description": "Resolve a merge conflict from parallel execution. Provide the conflict ID and either a resolution choice or custom resolution.",
    "input_schema": {
        "type": "object",
        "properties": {
            "conflict_id": {"type": "string"},
            "resolution": {
                "type": "string",
                "enum": ["accept_a", "accept_b", "auto_merge", "custom"]
            },
            "custom_content": {
                "type": "string",
                "description": "Custom resolution content (when resolution='custom')"
            }
        },
        "required": ["conflict_id", "resolution"]
    }
}
```

---

## User Experience Flow

### Hidden Complexity Path (PM catches it)

```
User: "Add an Excel upload button"

AgentHub: Analyzing request...

  PM (TaskDecomposer) surveying domains...
    UI agent: "Yes — need upload button + file picker component"
    API agent: "Yes — need POST /api/upload endpoint"
    Data agent: "Yes — need Excel parser + data validation"
    Worker agent: "Yes — parsing should be async via Celery"

  📋 PM Briefing:
    "This looks like a simple button, but it's actually a 4-domain project:

     Team A (UI): Upload button + file picker       → 2 files
     Team B (API): POST /api/upload endpoint         → 2 files (new)
     Team C (Data): Excel parser + validation        → 2 files (new)
     Team D (Worker): Celery task + Redis queue       → 2 files, depends on Team C

     Tasks A-C can run in parallel. Task D runs after C.
     Estimated: ~5 min parallel vs ~12 min sequential.
     Risk: LOW (no file overlaps between domains)."

  ⚠ Your call:
    1. Run in parallel (3 teams + 1 sequential follow-up) — est. 5 min
    2. Run fully sequential — est. 12 min
    3. Modify breakdown

User: "1"

AgentHub: Running 3 teams in parallel...

  [Team A — UI]     ████████████ Done (1m 45s)
  [Team B — API]    ████████████ Done (2m 10s)
  [Team C — Data]   ████████████ Done (2m 35s)

  Sequential follow-up: Team D depends on Team C's parser...
  [Team D — Worker] ████████████ Done (1m 50s)

  ✓ Merging all 4 branches...
  ✓ No conflicts (each team worked in its own domain)
  ✓ Tests passing (31/31)

  Done! "Simple button" implemented in 4m 30s (vs ~12 min sequential).
  Speedup: 2.7x | 4 tasks across 4 domains
```

### Happy Path (No Conflicts)

```
User: "Add a save button to the toolbar and build a time series chart component"

AgentHub: Analyzing request...
  ✓ Decomposed into 2 tasks:
    Task 1: Add save button to toolbar
      → Files: src/components/Toolbar.tsx, src/hooks/useSave.ts
      → Domain: UI agent, State agent
    Task 2: Build time series chart component
      → Files: src/components/TimeSeriesChart.tsx, src/utils/chartHelpers.ts
      → Domain: Charts agent

  ✓ Conflict analysis: NO overlap detected
    Risk: SAFE (confidence: 0.95)
    Estimated speedup: 1.8x

  Proceeding with parallel execution...

  [Branch parallel/task_1] ████████████ Done (45s, 12K tokens)
  [Branch parallel/task_2] ██████████░░ Done (52s, 15K tokens)

  ✓ Merging results...
  ✓ No conflicts
  ✓ Tests passing (14/14)

  Done! Both features implemented in 55s (vs ~97s sequential).
  Speedup: 1.76x | Total tokens: 29K
```

### Medium-Risk Path (CEO Confirmation)

```
User: "Add user preferences to settings and add a theme toggle to the header"

AgentHub: Analyzing request...
  ✓ Decomposed into 2 tasks:

    Team A (Settings agent): Add user preferences to settings page
      → Files: src/pages/Settings.tsx, src/store/userPrefs.ts, src/types/user.ts

    Team B (UI agent): Add theme toggle to header
      → Files: src/components/Header.tsx, src/store/theme.ts, src/types/user.ts

  ⚠ Risk Assessment: MEDIUM (confidence: 0.72)
    SHARED FILE: src/types/user.ts is modified by both teams
    Settings agent says: "Both tasks extend the User interface. If the type
      changes are additive (new fields), merge is straightforward. But if
      either task restructures existing fields, the other will break."

  ⚠ Your call:
    1. Run in parallel (will verify at merge) — est. 55s
    2. Run sequentially (safer) — est. 100s
    3. Modify task breakdown

User: "1"

AgentHub: Running in parallel...

  [Team A — Settings] ████████████ Done (47s)
  [Team B — UI]       ████████████ Done (42s)

  ⚠ Merge conflict in src/types/user.ts:
    Team A added: preferences: UserPreferences
    Team B added: theme: 'light' | 'dark'

  Settings agent (file owner) proposes:
    "Both changes are additive — new interface fields.
     Combine them into the same interface extension.
     Confidence: 0.92"

  ✓ Applied agent resolution
  ✓ Tests passing (18/18)
  Done! Both features implemented in 52s (vs ~100s sequential).
```

### Boundary Crossing Path (Inter-Team Communication)

```
User: "Add a notification bell to the header and add push notification settings to the user profile"

AgentHub: Analyzing request...
  ✓ Decomposed into 2 tasks:

    Team A (UI agent): Add notification bell to header
      → Files: src/components/Header.tsx, src/components/NotificationBell.tsx

    Team B (Settings agent): Add push notification settings to profile
      → Files: src/pages/Profile.tsx, src/store/notifications.ts

  ✓ Risk Assessment: SAFE (confidence: 0.89)
    No file overlap detected.

  Running in parallel...

  [Team A — UI]       ████████░░░░ Working...
    Team A signals: [BOUNDARY_CROSSING: src/store/notifications.ts]
    "I need to import the notification count from the store that
     Team B is building. Can't show the bell badge without it."

  → Asking Team B's agent...
    Settings agent says: "APPROVE. Team A can import notificationCount
    from my store. I'll export it as part of my public API.
    Confidence: 0.95"

  → Noted: Team A will import from Team B's store at merge time.

  [Team A — UI]       ████████████ Done (52s)
  [Team B — Settings] ████████████ Done (58s)

  ✓ Merging... clean merge with deferred import resolution
  ✓ Tests passing (22/22)
  Done!
```

### CEO Escalation Path (Agents Disagree)

```
User: "Add caching to the API layer and add real-time data refresh to the dashboard"

AgentHub: Analyzing request...
  ⚠ Risk: MEDIUM — API agent and Dashboard agent have concerns

  API agent says: "Caching introduces stale data. If the dashboard
    expects real-time data, my cache will serve stale responses."

  Dashboard agent says: "My real-time refresh hits the API directly.
    If the API layer adds caching, I need a cache-bust header."

  ⚠ Your call:
    1. Run in parallel — agents have flagged a semantic conflict,
       but both say it's resolvable at merge time
    2. Run sequentially — API caching first, then dashboard adapts
    3. Modify: have both teams coordinate on a cache-busting strategy first

User: "3 — have them agree on cache headers first"

AgentHub: Consulting agents...
  API agent proposes: "I'll add X-Cache-Bypass: true header support"
  Dashboard agent confirms: "I'll use that header for real-time queries"
  ✓ Shared contract established.

  Running in parallel with shared contract...
```

### High-Risk Path (PM recommends sequential, CEO decides)

```
User: "Refactor the authentication system and update all API endpoints to use the new auth"

AgentHub: Analyzing request...
  ✓ Decomposed into 2 tasks:

    Team A (Auth agent): Refactor authentication system
      → Files: src/auth/*, src/middleware/auth.ts, src/types/auth.ts

    Team B (API agent): Update API endpoints for new auth
      → Files: src/api/*, src/middleware/auth.ts

  ⚠ Risk: HIGH — HARD DEPENDENCY
    Team B depends on Team A's output (src/middleware/auth.ts, src/types/auth.ts)
    Auth agent says: "Task 2 literally can't start until Task 1's new
    auth interface exists. This is a hard dependency."

  📋 PM Recommendation: Run sequentially.

    Planned order:
      1. Team A: Refactor auth system (~4 min)
      2. Team B: Update API endpoints using new auth (~5 min)
         (starts after Team A finishes, uses Team A's new interface)

    Total: ~9 min sequential.

  ⚠ Your call:
    1. Run sequentially as recommended — est. 9 min
    2. Override: run in parallel anyway (high conflict risk)
    3. Modify task breakdown

User: "1"

AgentHub: Running sequentially per plan...

  [Team A — Auth]  ████████████ Done (3m 52s)
  [Team B — API]   ████████████ Done (4m 41s)

  ✓ No merge needed (sequential execution)
  ✓ Tests passing (47/47)
  Done! Auth refactor complete in 8m 33s.
```

---

## Leveraging Existing AgentHub Infrastructure

| Existing Component | How Parallel Sessions Uses It | Company Model Role |
|---|---|---|
| `ImportGraph` | Core of `ConflictRiskAnalyzer` — module dependency analysis | Org chart of code dependencies |
| `ImportGraph.get_clusters()` | Natural task boundaries often align with clusters | Team boundaries |
| `ImportGraph.get_central_modules()` | Identifies "hub" files likely to cause conflicts | Shared resources across teams |
| `ImportGraph.get_module_role()` | "bridge" modules are high-risk for parallel work | Inter-team interfaces |
| `KeywordRouter.get_all_scores()` | Maps tasks → agents, maps files → owning agents | R&R assignment, file ownership |
| `BaseAgent.run()` | Tier B agents for conflict detection + resolution, Tier A for business review | Team lead + business lead decision-making |
| `BaseAgent.context_paths` | Determines file ownership (Tier B) and business domain (Tier A) for merge resolution | Team's R&R scope |
| `ComplexityClassifier` | Extends to also classify parallelizability | Project Manager's scoping |
| `DAGTeamExecutor` pattern | Same topological execution, but with git branches | Parallel team execution |
| `QueryDecomposer` | Extended from sub-questions to implementation tasks | PM breaking down the project |
| `CrossAgentContextManager` | Provides scoped context for each parallel session; more precise with sub-agents | Team knowledge handoff |
| `AutoAgentManager` | Extended by `SubAgentManager` for hierarchical agent creation | Org chart builder |
| `CodebaseAnalyzer` | Reused for sub-clustering within a Tier B agent's domain | Department org design |
| `TeamExecutionTrace` | Extended to `ParallelExecutionTrace` | Project status dashboard |
| `AgentResponse.metadata` | Stores parallel execution trace + agent proposals | Team meeting minutes |
| MCP server tools | New tools for parallel execute/analyze/resolve | External API for the CEO |

---

## Key Design Decisions

### 1. Git Branches vs In-Memory Isolation

**Decision: Git branches.**

Rationale:
- Real conflict detection via `git merge`
- User can inspect each branch independently
- Rollback is trivial (`git branch -D`)
- Works with any editor, not just Claude Code
- Branches are the mental model developers already have

### 2. Claude CLI vs API for Parallel Sessions

**Decision: Support both, prefer API.**

- **CLI mode** (`claude --print`): Simpler, works today, but less control. Each session is a separate process. No real-time scoping enforcement.
- **API mode** (Anthropic Messages API + tool_use): Full control over which tools are available. Can restrict file access to scoped directories. More expensive but more reliable.

For v1, use CLI mode. For v2, use API mode with custom tool definitions that enforce file scoping.

### 3. Aggressive vs Conservative Parallelization

**Decision: Conservative by default.**

- Default `min_confidence_for_parallel: 0.7`
- If ANY doubt, run sequentially
- User can opt into aggressive mode: `conservative_mode: False`
- Track actual conflict rates over time to improve confidence calibration

### 4. Scoping Enforcement

**Decision: Advisory in v1, enforced in v2.**

- v1 (CLI): Prompt instructs "don't modify files outside scope" — Claude Code usually respects this but isn't guaranteed. Post-execution scope validation catches violations.
- v2 (API): Custom tool definitions restrict which files can be read/written. Hard enforcement.

### 5. Test Verification

**Decision: Always run tests after merge.**

- If tests pass: merge is clean
- If tests fail: semantic conflict detected, spawn resolution session
- If no test command: skip (but warn user)
- Auto-detect test command: check for pytest, jest, make test, etc.

---

## Safety Mechanisms

### Guardrail 1: Risk-Tiered Confirmation (CEO Protocol)

The confirmation behavior depends on risk level:

| Risk | Confirmation Required? | What the User Sees |
|------|----------------------|-------------------|
| SAFE | No (auto-proceed) | Progress bar only |
| CAUTION | No (auto-proceed with monitoring) | Progress bar + "monitoring boundary crossings" |
| MEDIUM | **Yes — CEO shot-call** | Risk report + agent concerns. PM recommends: parallel with caution |
| HIGH | **Yes — CEO shot-call** | Risk report + agent concerns. PM recommends: sequential + provides execution order plan |

For MEDIUM risk, the user sees specific agent concerns, not vague warnings:
```
⚠ Merge conflict risk is MEDIUM.
Settings agent says: "Both tasks modify the User type definition.
  If changes are additive, merge is safe. If either restructures
  existing fields, the other will break."
Options: [Parallel] [Sequential] [Modify tasks]
```

### Guardrail 2: Scope Violation Detection + Boundary Crossing Protocol

After each session completes:
1. Compare actual files changed vs expected scope
2. If a session modified unexpected files:
   a. Ask the owning domain agent if the change is acceptable
   b. If the agent approves → note it, proceed
   c. If the agent rejects → flag for CEO

### Guardrail 3: Post-Merge Test Gate

No merge into the base branch without passing tests (if tests exist). Failed tests trigger the domain-agent resolution flow:
1. Find agents that own the failing tests
2. Ask them to diagnose the issue with their domain knowledge
3. If agents can propose a fix → spawn resolution session
4. If agents can't → escalate to CEO

### Guardrail 4: Rollback on Failure

If anything goes wrong:
```bash
git checkout {base_branch}
git branch -D parallel/task_1 parallel/task_2 parallel/merged
```
The user's codebase is restored to exactly where it was. No partial states.

### Guardrail 5: Token Budget

Total token budget across all sessions is capped. Default: 100K tokens total. If exceeded, remaining tasks fall back to sequential.

### Guardrail 6: Agent Confidence Floor

If any domain agent's confidence drops below 0.6 during conflict resolution, the system automatically escalates to the CEO. Agents that are "unsure" don't get to make decisions — that's what the human is for.

---

## Implementation Phases

### Phase 0: Sub-Tier B Agents (1-2 weeks) ⭐ NEW — prerequisite for the rest
- [ ] Add hierarchy fields to `AgentSpec`: `parent_agent_id`, `children_ids`, `hierarchy_level`, `is_team_lead`
- [ ] `SubAgentBoundary` model
- [ ] `SubAgentPolicy` — thresholds for when to subdivide (60+ files, 3+ subdirs)
- [ ] `SubAgentManager` — evaluate, subdivide, and manage hierarchical agents
- [ ] `propose_subdivisions()` — reuse import graph sub-clustering within a domain
- [ ] `route_to_sub_agent()` — file-level routing to most specific agent
- [ ] `team_query()` — team lead delegates to relevant sub-agent(s)
- [ ] Integration with `AutoAgentManager` — sub-agents registered alongside Tier B
- [ ] Unit tests: verify subdivision thresholds, routing precision, team delegation

### Phase 1: Foundation (1-2 weeks)
- [ ] `ImplementationTask` and `DecompositionResult` models
- [ ] `TaskDecomposer` — LLM-based request decomposition
- [ ] `ConflictRiskAnalyzer` — file overlap + import graph (static analysis only)
- [ ] `ParallelizationPlan` model and risk classification logic
- [ ] `AgentConflictAssessment` model + domain agent consultation in analyzer
- [ ] Sub-agent-aware conflict analysis (route to sub-agent, not just team lead)
- [ ] Unit tests for analyzer with mock import graphs
- [ ] MCP tool: `agenthub_analyze_parallelism` (dry-run only)

### Phase 2: Execution (1-2 weeks)
- [ ] `BranchOrchestrator` — git branch management
- [ ] `SessionSpec` and scoped prompt generation (with boundary crossing tags)
- [ ] Context injection from team lead + relevant sub-agents
- [ ] `_spawn_session()` via Claude CLI (`claude --print`)
- [ ] `SessionResult` collection and validation
- [ ] Parallel execution via `concurrent.futures.ProcessPoolExecutor`
- [ ] Timeout and error handling per session

### Phase 3: Merge + Company Model (2-3 weeks)
- [ ] `MergeCoordinator` — sequential git merge
- [ ] `_get_owning_agent()` — file ownership via routing (sub-agent > team lead)
- [ ] `_request_agent_resolution()` — domain agent conflict proposals
- [ ] `_negotiate_between_agents()` — inter-agent negotiation
- [ ] `_should_escalate_to_ceo()` — escalation decision logic
- [ ] `_build_ceo_escalation_message()` — human-friendly conflict presentation
- [ ] `DomainResolutionProposal` model
- [ ] `_detect_scope_violations()` — post-session validation with agent approval
- [ ] `_analyze_semantic_conflict()` — test failure analysis via domain agents
- [ ] Post-merge test runner (auto-detect test command)
- [ ] Rollback mechanism

### Phase 4: Mid-Execution Protocol (1-2 weeks)
- [ ] `BoundaryCrossing` and `BoundaryCrossingResolution` models
- [ ] `MidExecutionEscalationHandler` — boundary crossing detection
- [ ] `detect_boundary_crossings()` — parse session output for tags
- [ ] `resolve_crossing()` — inter-team negotiation via domain agents (sub-agent level)
- [ ] `handle_blocking_crossing()` — resolution application
- [ ] Integration with `BranchOrchestrator` execution loop

### Phase 5: Integration (1 week)
- [ ] `ParallelSessionManager` — top-level orchestrator with company model
- [ ] Risk-tiered confirmation flow (SAFE → auto, MEDIUM → ask CEO, UNSAFE → sequential)
- [ ] MCP tool: `agenthub_parallel_execute`
- [ ] MCP tool: `agenthub_resolve_conflict`
- [ ] Dashboard integration (broadcast parallel execution events)
- [ ] `ParallelExecutionTrace` for observability

### Phase 6: Polish (1 week)
- [ ] Progressive output (show branch progress + inter-team communication in real-time)
- [ ] Confidence calibration from execution history
- [ ] Agent confidence floor enforcement (< 0.6 → escalate)
- [ ] Sub-agent tree visualization in dashboard (extend existing `auto/tree.py`)
- [ ] Documentation and examples
- [ ] Integration tests with real repos (dogfood on AgentHub itself)

---

## Cost Analysis

| Component | Model | Tokens | Cost (est.) | Company Role |
|-----------|-------|--------|-------------|---|
| Task decomposition | Sonnet | ~3K | ~$0.02 | PM scoping |
| Conflict analysis (static) | Code only | 0 | $0.00 | Risk assessment |
| Conflict analysis (agent consultation) | Haiku | ~2K per pair | ~$0.01–$0.03 | Team leads reviewing plan |
| Sub-agent precision routing | Code only | 0 | $0.00 | Routing to specific team member |
| Sub-agent delegated query | Haiku | ~1K per query | ~$0.005 | Team member answering team lead |
| Tier A business review | Haiku | ~1.5K per pair | ~$0.008 | Business lead reviewing parallel plan |
| Session A (typical) | Opus | ~15K | ~$0.75 | Team A working |
| Session B (typical) | Opus | ~15K | ~$0.75 | Team B working |
| Session C (typical) | Opus | ~15K | ~$0.75 | Team C working |
| Boundary crossing resolution | Haiku | ~1K per crossing | ~$0.005 | Inter-team negotiation |
| Domain agent merge resolution | Sonnet | ~4K per conflict | ~$0.03 | Team lead resolving conflict |
| Inter-agent negotiation | Sonnet | ~6K (2 agents) | ~$0.05 | Teams negotiating |
| Post-merge test diagnosis | Haiku | ~2K | ~$0.01 | QA + team lead triage |
| **Total (3 parallel, 1 conflict)** | | **~62K** | **~$2.43** | |
| **Sequential equivalent** | Opus | ~45K | **~$2.25** | |

**Key insight:** Parallel execution costs slightly MORE in tokens (~8% overhead for decomposition + agent consultation + merge coordination) but saves significant wall-clock time. For 3 tasks at ~50s each:
- Sequential: ~150s
- Parallel: ~55s (slowest task + overhead)
- **Speedup: ~2.7x**

The value proposition is **time, not tokens**. And the agent-based resolution means conflicts are resolved with domain knowledge, not generic pattern matching — higher quality resolutions, fewer CEO escalations.

---

## Relationship to DAG Teams

| Aspect | DAG Teams | Parallel Sessions |
|--------|-----------|-------------------|
| **Unit of work** | Answer a sub-question | Implement a code change |
| **Execution** | Agent.run() calls in parallel | Claude Code sessions on git branches |
| **Output** | Text responses synthesized | Code changes merged via git |
| **Isolation** | Context injection | Git branch isolation |
| **Conflict detection** | N/A (read-only) | Import graph + git merge |
| **Merge** | LLM synthesis | Git merge + test verification |
| **Typical use** | Complex cross-domain questions | Multi-feature implementation requests |
| **Cost** | ~$0.10-0.30 | ~$1.50-3.00 |

DAG Teams = parallel thinking. Parallel Sessions = parallel doing.

They can coexist: a parallel session could use DAG Teams internally to understand its task before implementing.

---

## Example: Dogfooding on AgentHub Itself

```
User: "Add a /health MCP tool that returns agent status, and also add
       request logging middleware to the MCP server"

Project Manager (TaskDecomposer):
  Team A (MCP agent): Add /health MCP tool
    → Files: mcp_server.py (add tool), models.py (HealthStatus model)
  Team B (MCP agent): Add request logging middleware
    → Files: mcp_server.py (add middleware), utils/logging.py (new)

Risk Assessment (ConflictRiskAnalyzer):
  ⚠ Static: Both tasks modify mcp_server.py → MEDIUM
  Agent consultation:
    MCP agent says: "Task 1 adds a new tool handler function.
      Task 2 wraps the handle_tool_call dispatcher. These are
      different sections of the file — additive, not conflicting."
    Agent risk: LOW
  Combined: MEDIUM (agent can't downgrade static)

CEO Confirmation:
  "⚠ Risk is MEDIUM. MCP agent says changes are additive
   (different sections of mcp_server.py). Proceed in parallel?"
User: "Yes"

Execution:
  [Team A] ████████████ Adds health tool + handler
  [Team B] ████████████ Adds logging wrapper around handle_tool_call

Merge Committee (MergeCoordinator):
  Git merge clean (different lines in mcp_server.py)
  MCP agent confirms: "Merged code looks correct — logging wraps
  all tool handlers including the new /health one."
  Tests pass
  ✓ Done
```
