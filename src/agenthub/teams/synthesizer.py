from __future__ import annotations
"""Response synthesis for multi-agent execution.

This module combines outputs from multiple agents into a single,
coherent response that answers the user's original query.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic

    from agenthub.hub import AgentHub
    from agenthub.models import AgentResponse
    from agenthub.teams.dag import ExecutionDAG

logger = logging.getLogger(__name__)


SYNTHESIZER_PROMPT = """You are a synthesis agent for a multi-agent code analysis system.

The user asked: "{query}"

Multiple specialized agents analyzed different aspects of the codebase.
Here are their findings, in execution order:

{agent_responses}

Synthesize these into a single, coherent response that:
1. Directly answers the user's original question
2. Traces the flow/connection between different parts of the codebase
3. References specific files, functions, and classes mentioned by the agents
4. Resolves any contradictions between agent responses
5. Identifies gaps - if no agent covered an aspect, note it

Do NOT just concatenate the agent responses. Produce a unified narrative that tells the complete story.

Important:
- Use clear section headers if the answer has multiple parts
- Include code paths and file references where relevant
- Keep the response focused and actionable
"""


class ResponseSynthesizer:
    """Synthesizes multiple agent responses into a coherent answer.

    Uses a capable LLM (Sonnet) to merge agent outputs into a unified
    response that directly addresses the user's query.

    Example:
        >>> synthesizer = ResponseSynthesizer(client)
        >>> content, tokens = synthesizer.synthesize(
        ...     "How does auth work?",
        ...     dag,
        ...     agent_responses,
        ...     hub
        ... )
    """

    def __init__(
        self,
        client: "anthropic.Anthropic",
        model: str = "claude-opus-4-20250514",
    ):
        """Initialize the synthesizer.

        Args:
            client: Anthropic client for API calls.
            model: Model to use (default: Sonnet for quality).
        """
        self.client = client
        self.model = model

    def synthesize(
        self,
        query: str,
        dag: "ExecutionDAG",
        agent_responses: dict[str, "AgentResponse"],
        hub: "AgentHub",
    ) -> tuple[str, int]:
        """Synthesize agent outputs into a coherent response.

        Args:
            query: Original user query.
            dag: Execution DAG for structure context.
            agent_responses: Dict mapping agent_id to AgentResponse.
            hub: AgentHub instance for agent info.

        Returns:
            Tuple of (synthesized_content, tokens_used).
        """
        if not agent_responses:
            return "No agent responses available to synthesize.", 0

        # If only one agent responded successfully, return its output directly
        successful_responses = {
            agent_id: resp
            for agent_id, resp in agent_responses.items()
            if resp and resp.content
        }

        if len(successful_responses) == 1:
            agent_id, resp = next(iter(successful_responses.items()))
            return resp.content, 0

        # Format agent responses for the prompt
        formatted_responses = self._format_agent_responses(dag, agent_responses, hub)

        prompt = SYNTHESIZER_PROMPT.format(
            query=query,
            agent_responses=formatted_responses,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            content = response.content[0].text.strip()

            return content, tokens_used

        except Exception as e:
            logger.error(f"Synthesis failed: {e}. Concatenating responses.")
            # Fallback: concatenate responses
            return self._fallback_concatenate(dag, agent_responses, hub), 0

    def _format_agent_responses(
        self,
        dag: "ExecutionDAG",
        agent_responses: dict[str, "AgentResponse"],
        hub: "AgentHub",
    ) -> str:
        """Format agent responses for the synthesis prompt.

        Formats responses in topological order to show the flow.
        """
        formatted = []
        layers = dag.topological_layers()

        for layer_idx, layer in enumerate(layers):
            for agent_id in layer:
                node = dag.nodes.get(agent_id)
                agent = hub.get_agent(agent_id)
                response = agent_responses.get(agent_id)

                agent_name = agent.spec.name if agent else agent_id
                sub_question = node.sub_question if node else ""

                if response and response.content:
                    formatted.append(
                        f"### {agent_name} ({agent_id}) [Layer {layer_idx}]\n"
                        f"**Sub-question:** {sub_question}\n"
                        f"**Response:**\n{response.content}\n"
                        f"---"
                    )
                elif node and node.status == "failed":
                    formatted.append(
                        f"### {agent_name} ({agent_id}) [Layer {layer_idx}]\n"
                        f"**Status:** FAILED - {node.error}\n"
                        f"---"
                    )
                else:
                    formatted.append(
                        f"### {agent_name} ({agent_id}) [Layer {layer_idx}]\n"
                        f"**Status:** No response available\n"
                        f"---"
                    )

        return "\n\n".join(formatted)

    def _fallback_concatenate(
        self,
        dag: "ExecutionDAG",
        agent_responses: dict[str, "AgentResponse"],
        hub: "AgentHub",
    ) -> str:
        """Fallback: simple concatenation of responses.

        Used when synthesis fails.
        """
        parts = ["# Combined Agent Responses\n"]
        layers = dag.topological_layers()

        for layer in layers:
            for agent_id in layer:
                agent = hub.get_agent(agent_id)
                response = agent_responses.get(agent_id)
                agent_name = agent.spec.name if agent else agent_id

                if response and response.content:
                    parts.append(f"## {agent_name}\n{response.content}\n")

        return "\n".join(parts)
