"""DAG Team Executor for multi-agent query execution.

This module orchestrates the execution of complex queries across
multiple agents in topological order based on their dependencies.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional

from agenthub.models import AgentResponse, Message, TeamExecutionTrace
from agenthub.teams.dag import ExecutionDAG
from agenthub.teams.decomposer import QueryDecomposer
from agenthub.teams.synthesizer import ResponseSynthesizer

if TYPE_CHECKING:
    import anthropic

    from agenthub.auto.import_graph import ImportGraph
    from agenthub.hub import AgentHub
    from agenthub.models import Session

logger = logging.getLogger(__name__)


class DAGTeamExecutor:
    """Executes a query across multiple agents in DAG order.

    Orchestrates multi-agent query execution by:
    1. Building an execution DAG from matched agents
    2. Decomposing the query into agent-specific sub-questions
    3. Executing agents in topological order (parallel where possible)
    4. Synthesizing results into a unified response

    Example:
        >>> executor = DAGTeamExecutor(hub, import_graph)
        >>> response = executor.execute(
        ...     "How does auth work end to end?",
        ...     ["api_agent", "service_agent", "model_agent"],
        ...     session
        ... )
    """

    def __init__(
        self,
        hub: "AgentHub",
        import_graph: "ImportGraph",
        max_parallel: int = 4,
        decomposer_model: str = "claude-haiku-4-5-20251001",
        synthesizer_model: str = "claude-sonnet-4-20250514",
        agent_timeout_seconds: float = 30.0,
        total_timeout_seconds: float = 120.0,
        max_total_tokens: int = 50000,
        skip_synthesis_if_single: bool = True,
    ):
        """Initialize the executor.

        Args:
            hub: AgentHub instance.
            import_graph: Built ImportGraph from the project.
            max_parallel: Max agents to run in parallel per layer.
            decomposer_model: Model for query decomposition.
            synthesizer_model: Model for response synthesis.
            agent_timeout_seconds: Timeout per agent execution.
            total_timeout_seconds: Total timeout for team execution.
            max_total_tokens: Kill switch if tokens exceed this.
            skip_synthesis_if_single: Skip synthesis if only 1 agent ran.
        """
        self.hub = hub
        self.import_graph = import_graph
        self.max_parallel = max_parallel
        self.agent_timeout_seconds = agent_timeout_seconds
        self.total_timeout_seconds = total_timeout_seconds
        self.max_total_tokens = max_total_tokens
        self.skip_synthesis_if_single = skip_synthesis_if_single

        # Initialize decomposer and synthesizer
        self._decomposer = QueryDecomposer(hub.client, model=decomposer_model)
        self._synthesizer = ResponseSynthesizer(hub.client, model=synthesizer_model)

    def execute(
        self,
        query: str,
        matched_agents: list[str],
        session: "Session",
    ) -> AgentResponse:
        """Execute a query across multiple agents.

        Args:
            query: Original user query.
            matched_agents: List of agent IDs to involve.
            session: Current session for context.

        Returns:
            Single AgentResponse with synthesized content.
        """
        start_time = time.time()

        # Step 1: Build execution DAG
        dag = self._build_execution_dag(matched_agents)
        if not dag.nodes:
            # No agents to execute - fall back to empty response
            return AgentResponse(
                content="No agents available to answer this query.",
                agent_id="team",
                session_id=session.session_id,
                tokens_used=0,
                metadata={"team_execution": True, "error": "no_agents"},
            )

        # Step 2: Decompose query into sub-questions
        sub_questions, decomposition_tokens = self._decomposer.decompose(query, dag, self.hub)

        # Store sub-questions in DAG nodes
        for agent_id, sub_q in sub_questions.items():
            if agent_id in dag.nodes:
                dag.nodes[agent_id].sub_question = sub_q

        # Step 3: Execute agents in topological order
        agent_responses, agent_tokens, agent_times = self._execute_dag(dag, session)

        # Calculate timing
        total_time_ms = int((time.time() - start_time) * 1000)
        sequential_time = sum(agent_times.values())
        parallel_speedup = sequential_time / total_time_ms if total_time_ms > 0 else 1.0

        # Step 4: Synthesize results
        successful_count = sum(1 for r in agent_responses.values() if r and r.content)

        if successful_count == 0:
            # No successful responses
            synthesized_content = "All agents failed to produce responses."
            synthesis_tokens = 0
        elif successful_count == 1 and self.skip_synthesis_if_single:
            # Only one agent succeeded - return its output directly
            for resp in agent_responses.values():
                if resp and resp.content:
                    synthesized_content = resp.content
                    synthesis_tokens = 0
                    break
        else:
            # Multiple agents - synthesize
            synthesized_content, synthesis_tokens = self._synthesizer.synthesize(
                query, dag, agent_responses, self.hub
            )

        # Build execution trace
        total_tokens = (
            decomposition_tokens
            + synthesis_tokens
            + sum(agent_tokens.values())
        )

        trace = TeamExecutionTrace(
            dag_structure={
                agent_id: node.dependencies
                for agent_id, node in dag.nodes.items()
            },
            execution_layers=dag.topological_layers(),
            sub_questions=sub_questions,
            agent_responses={
                agent_id: resp.content if resp else ""
                for agent_id, resp in agent_responses.items()
            },
            agent_tokens=agent_tokens,
            agent_times=agent_times,
            decomposition_tokens=decomposition_tokens,
            synthesis_tokens=synthesis_tokens,
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
            parallel_speedup=parallel_speedup,
        )

        # Add message to session
        session.messages.append(
            Message(
                role="assistant",
                content=synthesized_content,
                metadata={
                    "team_execution": True,
                    "trace": trace.model_dump(),
                    "agents_used": list(agent_responses.keys()),
                },
            )
        )

        return AgentResponse(
            content=synthesized_content,
            agent_id="team",
            session_id=session.session_id,
            tokens_used=total_tokens,
            metadata={
                "team_execution": True,
                "trace": trace.model_dump(),
                "agents_used": list(agent_responses.keys()),
            },
        )

    def _build_execution_dag(self, matched_agents: list[str]) -> ExecutionDAG:
        """Build an execution DAG from matched agents.

        Args:
            matched_agents: List of agent IDs.

        Returns:
            ExecutionDAG with nodes and dependency edges.
        """
        return ExecutionDAG.from_import_graph(
            matched_agents,
            self.import_graph,
            self.hub,
        )

    def _execute_dag(
        self,
        dag: ExecutionDAG,
        session: "Session",
    ) -> tuple[dict[str, AgentResponse], dict[str, int], dict[str, int]]:
        """Execute agents in topological order.

        Args:
            dag: Execution DAG with nodes and sub-questions.
            session: Current session.

        Returns:
            Tuple of (agent_responses, agent_tokens, agent_times).
        """
        agent_responses: dict[str, AgentResponse] = {}
        agent_tokens: dict[str, int] = {}
        agent_times: dict[str, int] = {}
        total_tokens = 0

        for layer in dag.topological_layers():
            # Check token budget
            if total_tokens >= self.max_total_tokens:
                logger.warning(f"Token budget exceeded ({total_tokens}), stopping execution")
                break

            # Execute all agents in this layer in parallel
            with ThreadPoolExecutor(max_workers=min(len(layer), self.max_parallel)) as pool:
                futures = {}

                for agent_id in layer:
                    node = dag.nodes[agent_id]
                    dag.mark_running(agent_id)

                    # Build augmented query with dependency results
                    augmented_query = self._augment_with_dependencies(node, dag)

                    future = pool.submit(
                        self._run_single_agent,
                        agent_id,
                        augmented_query,
                        session,
                    )
                    futures[future] = agent_id

                # Collect results
                for future in as_completed(futures, timeout=self.agent_timeout_seconds):
                    agent_id = futures[future]
                    try:
                        response, exec_time = future.result(timeout=self.agent_timeout_seconds)
                        if response:
                            agent_responses[agent_id] = response
                            agent_tokens[agent_id] = response.tokens_used
                            agent_times[agent_id] = exec_time
                            total_tokens += response.tokens_used
                            dag.mark_done(
                                agent_id,
                                response.content,
                                response.tokens_used,
                                exec_time,
                            )
                        else:
                            dag.mark_failed(agent_id, "No response returned")
                    except Exception as e:
                        logger.error(f"Agent {agent_id} failed: {e}")
                        dag.mark_failed(agent_id, str(e))

        return agent_responses, agent_tokens, agent_times

    def _augment_with_dependencies(
        self,
        node,
        dag: ExecutionDAG,
    ) -> str:
        """Build an augmented query with dependency results.

        Args:
            node: The DAGNode for the agent.
            dag: Execution DAG with results.

        Returns:
            Augmented query string.
        """
        sub_question = node.sub_question or "Answer the query based on your expertise."

        # Get dependency results
        dep_results = dag.get_dependency_results(node.agent_id)

        if not dep_results:
            return sub_question

        # Format dependency context
        context_parts = ["## Context from related agents\n"]

        for dep_id, result in dep_results.items():
            agent = self.hub.get_agent(dep_id)
            agent_name = agent.spec.name if agent else dep_id
            context_parts.append(f"### From {agent_name} ({dep_id}):\n{result}\n")

        context_parts.append("---\n")
        context_parts.append(f"## Your Question\n{sub_question}")

        return "\n".join(context_parts)

    def _run_single_agent(
        self,
        agent_id: str,
        query: str,
        session: "Session",
    ) -> tuple[Optional[AgentResponse], int]:
        """Run a single agent and return its response.

        Args:
            agent_id: The agent to run.
            query: The query (possibly augmented).
            session: Current session.

        Returns:
            Tuple of (AgentResponse or None, execution_time_ms).
        """
        start_time = time.time()

        try:
            agent = self.hub.get_agent(agent_id)
            if not agent:
                logger.warning(f"Agent {agent_id} not found")
                return None, 0

            response = agent.run(query, session)
            exec_time = int((time.time() - start_time) * 1000)

            return response, exec_time

        except Exception as e:
            logger.error(f"Error running agent {agent_id}: {e}")
            exec_time = int((time.time() - start_time) * 1000)
            return None, exec_time
