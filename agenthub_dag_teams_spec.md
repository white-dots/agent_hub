# AgentHub DAG Teams Spec — Tier B Agent Collaboration

## Overview

This spec introduces **DAG-based team execution** for Tier B agents. Currently, each query is routed to a single agent. This works for scoped questions ("what does `build_context` do?") but breaks down for cross-cutting queries ("how does the checkout flow work end to end?") where the answer lives across multiple modules.

DAG Teams let multiple Tier B agents collaborate on a single query. The collaboration structure is derived from the **import graph** that AgentHub already builds — the same data that drives agent creation now also drives agent coordination.

### Design Principle

Simple queries → single agent (fast, cheap).
Cross-cutting queries → DAG team (thorough, structured).
The router decides which path to take. No user configuration required.

---

## Architecture

### Current Flow (Single Agent)

```
Query → Router → Agent → Response
```

### New Flow (DAG Team)

```
Query → Router → ComplexityClassifier
                      │
              ┌───────┴────────┐
              │ simple         │ complex
              ▼                ▼
         Single Agent    DAGTeamExecutor
                              │
                         Decomposer (LLM)
                              │
                    ┌─────┬───┴───┬─────┐
                    ▼     ▼       ▼     ▼
                 Agent₁ Agent₂ Agent₃ Agent₄   ← parallel leaf execution
                    │     │       │     │
                    └─────┴───┬───┴─────┘
                              ▼
                     Synthesizer (LLM)          ← merge results
                              │
                              ▼
                      Final Response
```

### DAG Structure

The DAG is **not hardcoded**. It is built per-query from the import graph:

1. The **Decomposer** identifies which agents are relevant to the query
2. The import graph provides the **dependency edges** between those agents
3. Agents that depend on other agents' output execute after their dependencies
4. Independent agents execute in **parallel**

```
Example: "How does user authentication flow from request to database?"

         api_agent (handles routes, middleware)
              │
              ▼
        service_agent (handles auth logic)
         │              │
         ▼              ▼
  model_agent     config_agent (handles token settings)
  (handles User
   schema)
```

The DAG is a subgraph of the full import graph, filtered to agents relevant to the query.

---

## New Components

### 1. ComplexityClassifier

**File:** `src/agenthub/routing.py` (extend existing module)

Determines whether a query needs single-agent or multi-agent handling.

```python
class ComplexityClassifier:
    """Classifies query complexity to decide single-agent vs DAG team."""

    # Indicators that a query is cross-cutting
    CROSS_CUTTING_SIGNALS = [
        "end to end", "flow", "how does .* work",
        "trace", "lifecycle", "from .* to .*",
        "across", "between", "relationship",
        "architecture", "overview", "full picture",
    ]

    def __init__(self, hub: "AgentHub", threshold: float = 0.4):
        """
        Args:
            hub: AgentHub instance for access to agents and router.
            threshold: Minimum score to trigger DAG execution.
                       Lower = more queries go to DAG.
        """
        ...

    def classify(self, query: str) -> ComplexityResult:
        """Classify a query as simple or complex.

        Returns ComplexityResult with:
            - is_complex: bool
            - matched_agents: list[str]  (agent_ids that scored > 0)
            - confidence: float
        """
        ...
```

**Classification logic:**

1. Run the existing `KeywordRouter.route()` against ALL agents and collect scores (not just the top one)
2. Count how many agents scored above zero → `matched_count`
3. Check for cross-cutting signal words in the query
4. Decision:
   - `matched_count <= 1` → **simple** (single agent handles it)
   - `matched_count >= 2` AND cross-cutting signal present → **complex** (DAG team)
   - `matched_count >= 3` regardless of signals → **complex**

This keeps the fast path fast. A query like "what does `parse_config` do?" matches one agent and goes straight there. A query like "how does data flow from the API to the database?" matches 3+ agents and triggers the DAG.

```python
@dataclass
class ComplexityResult:
    is_complex: bool
    matched_agents: list[str]       # agent_ids that scored > 0
    agent_scores: dict[str, int]    # agent_id -> keyword match score
    confidence: float               # 0.0 to 1.0
    trigger_reason: str             # "multi_agent_match" | "cross_cutting_signal" | "simple"
```

---

### 2. DAGTeamExecutor

**File:** `src/agenthub/teams/executor.py` (new module)

Orchestrates multi-agent query execution.

```python
class DAGTeamExecutor:
    """Executes a query across multiple agents in DAG order."""

    def __init__(
        self,
        hub: "AgentHub",
        import_graph: ImportGraph,
        max_parallel: int = 4,
        synthesizer_model: str = "claude-sonnet-4-20250514",
    ):
        """
        Args:
            hub: AgentHub instance.
            import_graph: Built ImportGraph from the project.
            max_parallel: Max agents to run in parallel.
            synthesizer_model: Model for the final synthesis step.
        """
        ...

    def execute(
        self,
        query: str,
        matched_agents: list[str],
        session: Session,
    ) -> AgentResponse:
        """Execute a query across multiple agents.

        Steps:
            1. Build execution DAG from matched agents + import graph
            2. Decompose query into sub-questions per agent
            3. Execute agents in topological order (parallelize independent nodes)
            4. Synthesize final response from all agent outputs

        Returns:
            Single AgentResponse with synthesized content.
        """
        ...
```

#### Step 1: Build Execution DAG

```python
def _build_execution_dag(
    self,
    matched_agents: list[str],
) -> ExecutionDAG:
    """Build a DAG from the import graph, filtered to matched agents.

    Uses the ImportGraph edges to determine which agents depend on which.
    An agent A depends on agent B if modules in A import modules in B.

    Returns:
        ExecutionDAG with nodes (agents) and directed edges (dependencies).
    """
    ...
```

**Data model:**

```python
@dataclass
class DAGNode:
    agent_id: str
    dependencies: list[str]     # agent_ids this node depends on
    sub_question: str = ""      # filled by decomposer
    result: str = ""            # filled after execution
    status: str = "pending"     # "pending" | "running" | "done" | "failed"

@dataclass
class ExecutionDAG:
    nodes: dict[str, DAGNode]   # agent_id -> DAGNode
    execution_order: list[list[str]]  # topological layers for parallel execution

    def get_ready_nodes(self) -> list[str]:
        """Get nodes whose dependencies are all 'done'."""
        ...

    def topological_layers(self) -> list[list[str]]:
        """Return agents grouped by execution layer.

        Layer 0: agents with no dependencies (run first, in parallel)
        Layer 1: agents that depend only on layer 0 (run next, in parallel)
        ...
        """
        ...
```

**How edges are determined:**

```python
# Pseudocode for building edges between agents
for agent_a in matched_agents:
    for agent_b in matched_agents:
        if agent_a == agent_b:
            continue
        # Check if any module in agent_a imports any module in agent_b
        modules_a = hub.get_agent(agent_a).spec.context_paths
        modules_b = hub.get_agent(agent_b).spec.context_paths
        for mod_a in modules_a:
            if mod_a in import_graph.nodes:
                for imported in import_graph.nodes[mod_a].imports:
                    if imported in modules_b:
                        # agent_a depends on agent_b
                        dag.add_edge(agent_a, agent_b)
```

#### Step 2: Decompose Query

```python
def _decompose_query(
    self,
    query: str,
    dag: ExecutionDAG,
) -> dict[str, str]:
    """Decompose a query into sub-questions for each agent.

    Uses a lightweight LLM call to generate focused sub-questions.

    Args:
        query: Original user query.
        dag: Execution DAG with agent info.

    Returns:
        Dict mapping agent_id -> sub-question string.
    """
    ...
```

**Decomposer prompt template:**

```
You are a query decomposer for a multi-agent system.

The user asked: "{query}"

The following specialized agents are available:
{for each agent: agent_id, name, description, keywords}

Break the original query into focused sub-questions, one per agent.
Each sub-question should:
- Be answerable using ONLY that agent's domain knowledge
- Contribute a distinct piece of the overall answer
- Reference what specific aspect this agent should address

If an agent has dependencies on another agent, note what context
it needs from the dependency.

Respond as JSON:
{
    "sub_questions": {
        "agent_id_1": "focused question for agent 1",
        "agent_id_2": "focused question for agent 2, considering output from agent_id_1"
    }
}
```

**Model choice:** Use `claude-haiku-4-5-20251001` for decomposition — it's fast and cheap. Decomposition is a simple classification/rephrasing task that doesn't need a large model.

#### Step 3: Execute in Topological Order

```python
def _execute_dag(
    self,
    dag: ExecutionDAG,
    session: Session,
) -> dict[str, AgentResponse]:
    """Execute agents in topological order.

    Agents in the same layer run in parallel (up to max_parallel).
    Each agent receives:
      - Its sub-question
      - Results from its dependency agents (injected into the prompt)

    Returns:
        Dict mapping agent_id -> AgentResponse.
    """
    ...
```

**Execution strategy:**

```python
for layer in dag.topological_layers():
    # All agents in this layer can run in parallel
    # Use ThreadPoolExecutor with max_parallel workers
    with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
        futures = {}
        for agent_id in layer:
            node = dag.nodes[agent_id]
            # Inject dependency results into the sub-question
            augmented_query = self._augment_with_dependencies(node, dag)
            future = pool.submit(
                self._run_single_agent,
                agent_id,
                augmented_query,
                session,
            )
            futures[agent_id] = future

        # Collect results
        for agent_id, future in futures.items():
            dag.nodes[agent_id].result = future.result().content
            dag.nodes[agent_id].status = "done"
```

**Dependency injection format:**

When an agent has dependencies, prepend the dependency outputs:

```
## Context from related agents

### From api_agent (API & Endpoints Expert):
{api_agent's response}

### From model_agent (Data Model Expert):
{model_agent's response}

---

## Your Question
{sub_question for this agent}
```

#### Step 4: Synthesize

```python
def _synthesize(
    self,
    query: str,
    dag: ExecutionDAG,
    agent_responses: dict[str, AgentResponse],
) -> AgentResponse:
    """Synthesize agent outputs into a single coherent response.

    Args:
        query: Original user query.
        dag: Execution DAG (for structure context).
        agent_responses: All agent outputs.

    Returns:
        Final synthesized AgentResponse.
    """
    ...
```

**Synthesizer prompt template:**

```
You are a synthesis agent for a multi-agent code analysis system.

The user asked: "{query}"

Multiple specialized agents analyzed different aspects of the codebase.
Here are their findings, in execution order:

{for each agent in topological order:}
### {agent_name} ({agent_id})
**Sub-question:** {sub_question}
**Response:**
{agent_response}
---

Synthesize these into a single, coherent response that:
1. Directly answers the user's original question
2. Traces the flow/connection between different parts of the codebase
3. References specific files, functions, and classes mentioned by the agents
4. Resolves any contradictions between agent responses
5. Identifies gaps — if no agent covered an aspect, note it

Do NOT just concatenate the agent responses. Produce a unified narrative.
```

**Model choice:** Use the same model as the hub's default (typically `claude-sonnet-4-20250514`) for synthesis, since it requires reasoning across domains.

---

### 3. TeamResponse (Extended AgentResponse)

**File:** `src/agenthub/models.py` (extend existing)

```python
@dataclass
class TeamExecutionTrace:
    """Trace of a DAG team execution for debugging/observability."""
    dag_structure: dict[str, list[str]]    # agent_id -> dependency agent_ids
    execution_layers: list[list[str]]      # topological execution order
    sub_questions: dict[str, str]          # agent_id -> sub-question
    agent_responses: dict[str, str]        # agent_id -> response content
    agent_tokens: dict[str, int]           # agent_id -> tokens used
    decomposition_tokens: int              # tokens for decomposer call
    synthesis_tokens: int                  # tokens for synthesizer call
    total_tokens: int                      # sum of all tokens
    total_time_ms: int                     # wall-clock time
    parallel_speedup: float                # sequential_time / actual_time
```

The `AgentResponse.metadata` dict should include the `TeamExecutionTrace` when the response came from a DAG team, so it's available for the dashboard and debugging.

```python
# In AgentResponse returned by DAGTeamExecutor:
response.metadata["team_execution"] = True
response.metadata["trace"] = trace.__dict__
response.metadata["agents_used"] = list(agent_responses.keys())
```

---

## Integration Points

### Hub.run() — Modify to Support Teams

**File:** `src/agenthub/hub.py`

The change to `hub.run()` should be minimal. Add a `team_mode` parameter:

```python
def run(
    self,
    query: str,
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    model: Optional[str] = None,
    team_mode: str = "auto",  # NEW: "auto" | "always" | "never"
) -> AgentResponse:
    """Execute a query through the hub.

    Args:
        ...existing args...
        team_mode: Controls DAG team execution.
            "auto" — use ComplexityClassifier to decide (default)
            "always" — always use DAG team (useful for testing)
            "never" — never use DAG team (original behavior)
    """
    # ... existing session setup ...

    # If agent_id is forced, skip team execution
    if agent_id:
        # ... existing forced-agent path ...
        pass

    # NEW: Check if team execution is appropriate
    elif team_mode != "never" and self._team_executor:
        classifier = ComplexityClassifier(self)
        result = classifier.classify(query)

        if result.is_complex or team_mode == "always":
            return self._team_executor.execute(
                query=query,
                matched_agents=result.matched_agents,
                session=session,
            )

    # ... existing single-agent path ...
```

### Hub — Enable Teams

```python
# In hub.py, add a new method:

def enable_teams(
    self,
    import_graph: Optional[ImportGraph] = None,
    max_parallel: int = 4,
    complexity_threshold: float = 0.4,
) -> None:
    """Enable DAG team execution for complex queries.

    Args:
        import_graph: ImportGraph instance. If None, will be built
                     from the auto-agent manager's project root.
        max_parallel: Max agents to run in parallel.
        complexity_threshold: Score threshold for triggering teams.
    """
    if import_graph is None and self._auto_manager:
        import_graph = self._auto_manager.import_graph
    elif import_graph is None:
        raise ValueError("No import graph available. "
                        "Either provide one or enable auto-agents first.")

    self._team_executor = DAGTeamExecutor(
        hub=self,
        import_graph=import_graph,
        max_parallel=max_parallel,
    )
    self._complexity_threshold = complexity_threshold
```

### Auto-Agent Manager — Expose Import Graph

The `AutoAgentManager` (or `SmartAgentFactory`) already builds an `ImportGraph`. Expose it so `enable_teams()` can reuse it instead of rebuilding:

```python
# In auto/manager.py or wherever the graph is built:
class AutoAgentManager:
    ...
    @property
    def import_graph(self) -> ImportGraph:
        """Get the import graph used for agent generation."""
        return self._import_graph
```

### MCP Server — Expose Team Queries

**File:** `src/agenthub/mcp_server.py`

Add a new tool or extend `agenthub_query` to support team mode:

```python
# Option A: Add parameter to existing tool
@server.tool()
async def agenthub_query(query: str, team_mode: str = "auto"):
    """Query AgentHub with optional team execution.

    Args:
        query: Your question about the codebase.
        team_mode: "auto" (let AgentHub decide), "team" (force team), "single" (force single agent)
    """
    response = hub.run(query, team_mode=team_mode)
    ...
```

### Dashboard — Team Visualization

**File:** `src/agenthub/dashboard/server.py`

Add an endpoint to visualize DAG team executions:

```python
@app.get("/api/team-trace/{session_id}")
async def get_team_trace(session_id: str):
    """Get the DAG execution trace for a session."""
    session = hub.get_session(session_id)
    # Find the latest team execution in session messages
    for msg in reversed(session.messages):
        if msg.metadata.get("team_execution"):
            return msg.metadata["trace"]
    return {"error": "No team execution found"}
```

The dashboard frontend can render the DAG as a visual graph with:
- Nodes colored by status (green=done, yellow=running, red=failed)
- Edges showing dependency flow
- Timing info on each node
- Expandable response content per agent

---

## New Directory Structure

```
src/agenthub/
├── teams/                    # NEW module
│   ├── __init__.py
│   ├── classifier.py         # ComplexityClassifier
│   ├── executor.py           # DAGTeamExecutor
│   ├── dag.py                # ExecutionDAG, DAGNode data models
│   ├── decomposer.py         # Query decomposition logic
│   └── synthesizer.py        # Response synthesis logic
├── routing.py                # Modified: ComplexityClassifier may live here or in teams/
├── hub.py                    # Modified: enable_teams(), team_mode param
├── models.py                 # Modified: TeamExecutionTrace
└── ... (existing files unchanged)
```

---

## Configuration

```python
@dataclass
class TeamConfig:
    """Configuration for DAG team execution."""

    # Complexity classification
    complexity_threshold: float = 0.4       # Min score to trigger teams
    min_agents_for_team: int = 2            # Minimum matched agents
    max_agents_per_team: int = 6            # Cap on agents in a single team

    # Execution
    max_parallel: int = 4                   # Max concurrent agent calls
    agent_timeout_seconds: float = 30.0     # Per-agent timeout
    total_timeout_seconds: float = 120.0    # Total team execution timeout

    # Models
    decomposer_model: str = "claude-haiku-4-5-20251001"     # Fast/cheap for decomposition
    synthesizer_model: str = "claude-sonnet-4-20250514"     # Strong model for synthesis

    # Cost control
    max_total_tokens: int = 50000           # Kill switch if tokens exceed this
    skip_synthesis_if_single: bool = True   # If only 1 agent ended up running, skip synthesis
```

---

## Example Usage

### Automatic (Recommended)

```python
from agenthub import AgentHub

hub = AgentHub()
hub.enable_auto_agents("./my-project")
hub.enable_teams()  # That's it

# Simple queries → single agent (fast)
response = hub.run("What does parse_config do?")

# Cross-cutting queries → DAG team (thorough)
response = hub.run("How does data flow from the API request to the database?")

# Force team mode for testing
response = hub.run("Explain the auth system", team_mode="always")
```

### CLI Integration

```bash
# Build with team support enabled
agenthub build . --enable-teams

# Query with explicit team mode
agenthub query "How does auth work end to end?" --team
```

---

## Cost & Latency Analysis

### Single Agent (Current)
- 1 LLM call
- ~2-4K input tokens (context + query)
- ~1-2K output tokens
- ~2-5 seconds
- **Total: ~5K tokens, ~3s**

### DAG Team (3 agents, 2 layers)
- 1 decomposer call (Haiku): ~500 input, ~200 output
- 3 agent calls (parallel within layers): ~4K input × 3, ~1.5K output × 3
- 1 synthesis call: ~6K input (all responses), ~2K output
- **Total: ~25K tokens, ~8-12s**

### Mitigation Strategies
1. **Haiku for decomposition** — cheap and fast
2. **Parallel execution** — independent agents don't add latency
3. **Skip synthesis for single-agent results** — if classifier triggers but only one agent produces useful output
4. **Token budget** — hard cap via `max_total_tokens` in TeamConfig
5. **Cache agent contexts** — already implemented via GitAwareCache

---

## Edge Cases

### Circular Dependencies in Import Graph
The import graph may have cycles (A imports B, B imports C, C imports A). For DAG construction, break cycles by removing the weakest edge (fewest imported names). Log a warning.

### Agent Failure
If one agent in the DAG fails (timeout, API error), the synthesizer should work with available results and note the gap. Don't fail the entire team.

### Empty Decomposition
If the decomposer can't generate meaningful sub-questions (e.g., query is too vague), fall back to single-agent routing.

### All Agents Match
If the classifier matches ALL agents, cap at `max_agents_per_team` (default 6) and pick the top-scoring ones.

### No Import Graph Available
If `enable_teams()` is called without an import graph (and auto-agents aren't enabled), treat all matched agents as independent (flat parallel execution with no dependency edges). Still works, just doesn't benefit from structured ordering.

---

## Testing Strategy

### Unit Tests

```
tests/
├── test_teams/
│   ├── test_classifier.py      # ComplexityClassifier logic
│   ├── test_dag.py             # DAG construction, topological sort, cycle breaking
│   ├── test_decomposer.py      # Query decomposition (mock LLM)
│   ├── test_executor.py        # Full execution flow (mock agents)
│   └── test_synthesizer.py     # Response synthesis (mock LLM)
```

**Key test cases:**

1. **Classifier:** single-keyword query → simple; multi-agent + "flow" → complex
2. **DAG construction:** import graph with 3 agents → correct edges and layers
3. **Cycle breaking:** circular imports → DAG still valid
4. **Parallel execution:** independent agents run concurrently (measure wall-clock)
5. **Dependency injection:** agent B receives agent A's output when B depends on A
6. **Failure handling:** one agent times out → synthesis still works with partial results
7. **Token budget:** execution stops if budget exceeded

### Integration Tests

Use the existing `tmp_project` fixture from `conftest.py` to create a multi-module project, build auto-agents, enable teams, and run cross-cutting queries. Verify the response references files from multiple modules.

---

## Implementation Order

### Phase 1: Data Models & DAG Construction
1. Create `teams/dag.py` — `DAGNode`, `ExecutionDAG` with topological sort
2. Add `TeamExecutionTrace` to `models.py`
3. Write `tests/test_teams/test_dag.py`

### Phase 2: Complexity Classifier
1. Create `teams/classifier.py` — `ComplexityClassifier`, `ComplexityResult`
2. Write `tests/test_teams/test_classifier.py`

### Phase 3: Decomposer & Synthesizer
1. Create `teams/decomposer.py`
2. Create `teams/synthesizer.py`
3. Write tests with mocked LLM calls

### Phase 4: Executor & Hub Integration
1. Create `teams/executor.py` — `DAGTeamExecutor`
2. Modify `hub.py` — `enable_teams()`, `team_mode` parameter
3. Expose `import_graph` from auto-agent manager
4. Write `tests/test_teams/test_executor.py`

### Phase 5: MCP & Dashboard
1. Update MCP server with team_mode parameter
2. Add dashboard endpoint for team traces
3. Add CLI `--team` flag

### Phase 6: Polish
1. Add `TeamConfig` to `config.py`
2. Add team example to `examples/`
3. Update README
