"""Query decomposition for multi-agent execution.

This module breaks down a user query into focused sub-questions,
one per agent, so each agent can address their specific domain.
"""

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic

    from agenthub.hub import AgentHub
    from agenthub.teams.dag import ExecutionDAG

logger = logging.getLogger(__name__)


DECOMPOSER_PROMPT = """You are a query decomposer for a multi-agent code analysis system.

The user asked: "{query}"

The following specialized agents are available to answer this query:

{agent_descriptions}

Break the original query into focused sub-questions, one per agent.
Each sub-question should:
- Be answerable using ONLY that agent's domain knowledge
- Contribute a distinct piece of the overall answer
- Reference what specific aspect this agent should address

Agent dependencies (if any):
{dependency_info}

If an agent has dependencies on another agent, note what context it might need from the dependency in the sub-question.

Respond with ONLY valid JSON in this exact format:
{{
    "sub_questions": {{
        "agent_id_1": "focused question for agent 1",
        "agent_id_2": "focused question for agent 2"
    }}
}}

Important:
- Include ALL agent IDs provided above
- Each sub-question should be self-contained and focused
- Keep sub-questions concise but specific
"""


class QueryDecomposer:
    """Decomposes a complex query into agent-specific sub-questions.

    Uses a fast/cheap LLM (Haiku) to generate focused sub-questions
    for each agent involved in a multi-agent query.

    Example:
        >>> decomposer = QueryDecomposer(client)
        >>> sub_questions, tokens = decomposer.decompose(
        ...     "How does auth work end to end?",
        ...     dag,
        ...     hub
        ... )
        >>> print(sub_questions)
        {'api_agent': 'How is authentication handled in the API routes?', ...}
    """

    def __init__(
        self,
        client: "anthropic.Anthropic",
        model: str = "claude-haiku-4-5-20251001",
    ):
        """Initialize the decomposer.

        Args:
            client: Anthropic client for API calls.
            model: Model to use (default: Haiku for speed/cost).
        """
        self.client = client
        self.model = model

    def decompose(
        self,
        query: str,
        dag: "ExecutionDAG",
        hub: "AgentHub",
    ) -> tuple[dict[str, str], int]:
        """Decompose a query into sub-questions for each agent.

        Args:
            query: Original user query.
            dag: Execution DAG with agent nodes.
            hub: AgentHub instance for agent info.

        Returns:
            Tuple of (agent_id -> sub-question dict, tokens_used).
        """
        if not dag.nodes:
            return {}, 0

        # Build agent descriptions
        agent_descriptions = self._build_agent_descriptions(dag, hub)

        # Build dependency info
        dependency_info = self._build_dependency_info(dag)

        # Format prompt
        prompt = DECOMPOSER_PROMPT.format(
            query=query,
            agent_descriptions=agent_descriptions,
            dependency_info=dependency_info if dependency_info else "No dependencies between agents.",
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            content = response.content[0].text.strip()

            # Parse JSON response
            sub_questions = self._parse_response(content, dag)

            return sub_questions, tokens_used

        except Exception as e:
            logger.warning(f"Decomposition failed: {e}. Using original query for all agents.")
            # Fallback: use original query for all agents
            return {agent_id: query for agent_id in dag.nodes}, 0

    def _build_agent_descriptions(
        self,
        dag: "ExecutionDAG",
        hub: "AgentHub",
    ) -> str:
        """Build formatted agent descriptions for the prompt."""
        descriptions = []
        for agent_id in dag.nodes:
            agent = hub.get_agent(agent_id)
            if agent:
                spec = agent.spec
                keywords = ", ".join(spec.context_keywords[:5])  # Limit keywords
                descriptions.append(
                    f"- {agent_id} ({spec.name}): {spec.description}\n"
                    f"  Keywords: {keywords}"
                )
            else:
                descriptions.append(f"- {agent_id}: (Unknown agent)")
        return "\n".join(descriptions)

    def _build_dependency_info(self, dag: "ExecutionDAG") -> str:
        """Build dependency information for the prompt."""
        deps = []
        for agent_id, node in dag.nodes.items():
            if node.dependencies:
                dep_list = ", ".join(node.dependencies)
                deps.append(f"- {agent_id} depends on: {dep_list}")
        return "\n".join(deps) if deps else ""

    def _parse_response(
        self,
        content: str,
        dag: "ExecutionDAG",
    ) -> dict[str, str]:
        """Parse the LLM response into sub-questions.

        Args:
            content: Raw LLM response text.
            dag: Execution DAG to validate agent IDs.

        Returns:
            Dict mapping agent_id to sub-question.
        """
        # Try to extract JSON from response
        try:
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)
            sub_questions = data.get("sub_questions", {})

            # Validate that all agents are covered
            result = {}
            for agent_id in dag.nodes:
                if agent_id in sub_questions:
                    result[agent_id] = sub_questions[agent_id]
                else:
                    # Generate a generic question for missing agents
                    result[agent_id] = f"What is your perspective on this query?"

            return result

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse decomposition response: {e}")
            # Return empty sub-questions (executor will use original query)
            return {}
