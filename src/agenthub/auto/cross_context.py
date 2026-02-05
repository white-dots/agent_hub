"""Cross-agent context injection for enhanced query responses.

When an agent's code imports from another agent's domain, this module
enables automatic injection of relevant context from the imported domain.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from agenthub.auto.import_graph import ImportGraph
    from agenthub.hub import AgentHub


@dataclass
class CrossContextConfig:
    """Configuration for cross-agent context injection."""

    enabled: bool = True
    max_injected_chars: int = 10000  # Max chars to inject from other agents
    max_agents_to_inject: int = 3  # Max number of related agents
    injection_ratio: float = 0.2  # Max 20% of total context can be injected
    include_summary_only: bool = False  # If True, only inject agent descriptions


@dataclass
class InjectedContext:
    """Context injected from another agent."""

    source_agent_id: str
    source_domain: str
    context_snippet: str
    relevance_reason: str  # Why this was included
    char_count: int
    imported_modules: list[str] = field(default_factory=list)


class CrossAgentContextManager:
    """Manages cross-agent context injection based on import relationships.

    When Agent A is queried about code that imports from Agent B's domain,
    this manager can automatically inject relevant context from Agent B
    to help Agent A provide more accurate responses.

    Example:
        >>> manager = CrossAgentContextManager(hub, import_graph)
        >>> related = manager.get_related_context("auth_agent", "How does login work?")
        >>> for ctx in related:
        ...     print(f"From {ctx.source_domain}: {ctx.char_count} chars")
    """

    def __init__(
        self,
        hub: "AgentHub",
        import_graph: Optional["ImportGraph"] = None,
        config: Optional[CrossContextConfig] = None,
    ):
        """Initialize the cross-agent context manager.

        Args:
            hub: AgentHub instance for accessing agents.
            import_graph: ImportGraph for dependency analysis.
            config: Configuration for context injection.
        """
        self.hub = hub
        self.import_graph = import_graph
        self.config = config or CrossContextConfig()

    def get_related_context(
        self,
        agent_id: str,
        query: str,
        max_chars: Optional[int] = None,
    ) -> list[InjectedContext]:
        """Get context from related agents that might help answer the query.

        Analyzes the import relationships of the queried agent's modules
        and fetches relevant context from agents that own imported code.

        Args:
            agent_id: The agent handling the query.
            query: The user's query (for potential future relevance scoring).
            max_chars: Override for max injected characters.

        Returns:
            List of context snippets from related agents.
        """
        if not self.config.enabled:
            return []

        if not self.import_graph:
            return []

        max_chars = max_chars or self.config.max_injected_chars

        # 1. Get the agent's modules
        agent = self.hub.get_agent(agent_id)
        if not agent:
            return []

        # Get module paths from agent spec
        module_paths = agent.spec.context_paths or []
        if not module_paths:
            # Try to get from metadata for SmartCodeAgent
            module_paths = agent.spec.metadata.get("module_paths", [])

        if not module_paths:
            return []

        # 2. Find imports from this agent's modules to other agents
        related_agents = self._find_imported_agents(module_paths, agent_id)

        # 3. Fetch and format context from related agents
        injected: list[InjectedContext] = []
        remaining_chars = max_chars

        for other_agent_id, imported_modules in related_agents[
            : self.config.max_agents_to_inject
        ]:
            if remaining_chars <= 0:
                break

            context = self._get_agent_context_snippet(
                other_agent_id,
                imported_modules,
                max_chars=remaining_chars,
            )

            if context:
                injected.append(context)
                remaining_chars -= context.char_count

        return injected

    def _find_imported_agents(
        self,
        module_paths: list[str],
        exclude_agent_id: str,
    ) -> list[tuple[str, list[str]]]:
        """Find agents whose code is imported by the given modules.

        Args:
            module_paths: List of module paths belonging to the querying agent.
            exclude_agent_id: Agent ID to exclude (the querying agent itself).

        Returns:
            List of (agent_id, imported_modules) tuples, sorted by relevance.
        """
        if not self.import_graph:
            return []

        # Collect all imports from the agent's modules
        imported_modules: dict[str, set[str]] = {}  # imported -> set of importers

        for module_path in module_paths:
            neighbors = self.import_graph.get_module_neighbors(module_path)
            for imported in neighbors.get("imports", []):
                # Skip if the imported module is part of this agent's own modules
                if imported in module_paths:
                    continue
                if imported not in imported_modules:
                    imported_modules[imported] = set()
                imported_modules[imported].add(module_path)

        # Map imported modules to agents
        agent_imports: dict[str, list[str]] = {}  # agent_id -> list of imported modules

        for imported_module in imported_modules:
            # Find which agent owns this module
            for spec in self.hub.list_agents():
                if spec.agent_id == exclude_agent_id:
                    continue

                # Check if this module is in the agent's context paths
                context_paths = spec.context_paths or []
                module_paths_meta = spec.metadata.get("module_paths", [])
                all_paths = set(context_paths) | set(module_paths_meta)

                if imported_module in all_paths:
                    if spec.agent_id not in agent_imports:
                        agent_imports[spec.agent_id] = []
                    agent_imports[spec.agent_id].append(imported_module)
                    break

        # Sort by number of imports (most relevant first)
        sorted_agents = sorted(
            agent_imports.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )

        return sorted_agents

    def _get_agent_context_snippet(
        self,
        agent_id: str,
        relevant_modules: list[str],
        max_chars: int,
    ) -> Optional[InjectedContext]:
        """Get a context snippet from another agent.

        Args:
            agent_id: ID of the agent to get context from.
            relevant_modules: Modules from this agent that were imported.
            max_chars: Maximum characters to include.

        Returns:
            InjectedContext or None if unable to get context.
        """
        agent = self.hub.get_agent(agent_id)
        if not agent:
            return None

        if self.config.include_summary_only:
            # Just provide a summary
            snippet = f"[{agent.spec.name}]: {agent.spec.description}"
        else:
            # Get the agent's full context and extract relevant portions
            full_context = agent.get_context()
            snippet = self._extract_relevant_context(
                full_context,
                relevant_modules,
                max_chars,
            )

        if not snippet:
            return None

        return InjectedContext(
            source_agent_id=agent_id,
            source_domain=agent.spec.name,
            context_snippet=snippet,
            relevance_reason=f"Imported by: {', '.join(relevant_modules[:3])}",
            char_count=len(snippet),
            imported_modules=relevant_modules,
        )

    def _extract_relevant_context(
        self,
        full_context: str,
        relevant_modules: list[str],
        max_chars: int,
    ) -> str:
        """Extract relevant portions of context based on imported modules.

        The context format is typically:
        ### module/path.py
        ```python
        ...code...
        ```

        Args:
            full_context: The agent's full context string.
            relevant_modules: List of module paths that were imported.
            max_chars: Maximum characters to return.

        Returns:
            Extracted context snippet.
        """
        # Split by section headers (### path)
        sections = re.split(r"(?=###\s+)", full_context)

        relevant_sections: list[str] = []
        total_chars = 0

        for section in sections:
            if not section.strip():
                continue

            # Check if this section is about a relevant module
            for module in relevant_modules:
                module_name = module.split("/")[-1].replace(".py", "")
                module_path_parts = module.replace(".py", "").split("/")

                # Check if any part of the module path appears in the section header
                section_lower = section.lower()
                is_relevant = (
                    module_name.lower() in section_lower
                    or module.lower() in section_lower
                    or any(part.lower() in section_lower for part in module_path_parts)
                )

                if is_relevant:
                    if total_chars + len(section) <= max_chars:
                        relevant_sections.append(section)
                        total_chars += len(section)
                    break

        if relevant_sections:
            return "\n".join(relevant_sections)

        # Fallback: return first part of context if no specific sections matched
        if len(full_context) > max_chars:
            return full_context[:max_chars] + "\n... [truncated]"
        return full_context


def format_injected_context(injected: list[InjectedContext]) -> str:
    """Format injected context for inclusion in system prompt.

    Args:
        injected: List of InjectedContext objects.

    Returns:
        Formatted string ready to include in system prompt.
    """
    if not injected:
        return ""

    parts = ["## Related Context (from imported modules)\n"]
    parts.append(
        "*The following context is from other agents whose code is imported by your domain.*\n"
    )

    for ctx in injected:
        parts.append(f"### From {ctx.source_domain} ({ctx.source_agent_id})")
        parts.append(f"*{ctx.relevance_reason}*\n")
        parts.append(ctx.context_snippet)
        parts.append("")

    return "\n".join(parts)
