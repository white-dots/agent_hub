from __future__ import annotations
"""Base agent implementation."""

import fnmatch
import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from agenthub.models import AgentResponse, AgentSpec, Artifact, Message, Session

if TYPE_CHECKING:
    import anthropic

    from agenthub.agents.domain_tools import DomainToolExecutor
    from agenthub.agents.project_tools import ProjectToolExecutor
    from agenthub.cache import GitAwareCache
    from agenthub.qc.models import AgentAnalysisResult, ChangeSet, Concern

logger = logging.getLogger(__name__)


def heuristic_scope_check(spec: "AgentSpec", query: str) -> dict:
    """Shared heuristic scope check for auto-generated agents.

    Catches obviously out-of-scope queries without any LLM call:
    1. Zero keyword overlap between query words and agent keywords/description.
    2. Multiple exclusion term matches from the agent's RoutingConfig.

    The check is intentionally lenient — it only rejects queries that have
    absolutely zero signal of relevance. False negatives (letting through
    out-of-scope queries) are cheaper than false positives (rejecting in-scope
    queries that then have to burn retries).

    Args:
        spec: The agent's AgentSpec.
        query: The user's query.

    Returns:
        Dict with 'in_scope' bool, 'message' str, and optional 'suggested_agent'.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())

    # Check 1: Zero keyword overlap — strongest free signal.
    # If the query has enough words to be meaningful but none match
    # any of the agent's routing keywords, it's very likely out of scope.
    # Uses both exact-word AND substring matching (to handle plurals like
    # "tenants" matching keyword "tenant").
    agent_keywords = set(kw.lower() for kw in spec.context_keywords)
    has_overlap = bool(query_words & agent_keywords)
    if not has_overlap:
        # Fallback: check if any keyword appears as a substring in any query word
        # (handles plurals, suffixes, compound words)
        has_overlap = any(
            kw in qw or qw in kw
            for kw in agent_keywords
            for qw in query_words
            if len(kw) >= 3 and len(qw) >= 3
        )

    if not has_overlap:
        # Fallback 2: check overlap with agent description words
        # (catches conceptual matches like "permission" in description
        #  even if not an explicit keyword)
        desc_words = set(spec.description.lower().split())
        desc_overlap = bool(query_words & desc_words)
        if not desc_overlap:
            desc_overlap = any(
                dw in qw or qw in dw
                for dw in desc_words
                for qw in query_words
                if len(dw) >= 4 and len(qw) >= 4
            )
        if desc_overlap:
            has_overlap = True

    if not has_overlap:
        # Fallback 3: check if query words overlap with context_paths
        # (catches file/folder name matches like "insight" in "insight.py")
        path_words: set[str] = set()
        for cp in spec.context_paths:
            # Extract meaningful words from path segments
            for segment in cp.replace("/", " ").replace(".", " ").replace("_", " ").replace("-", " ").split():
                if len(segment) >= 3 and segment != "py" and segment != "app":
                    path_words.add(segment.lower())
        if path_words:
            path_overlap = any(
                pw in qw or qw in pw
                for pw in path_words
                for qw in query_words
                if len(pw) >= 3 and len(qw) >= 3
            )
            if path_overlap:
                has_overlap = True

    # Only reject if we have a long enough query (5+ words) and zero overlap
    # across keywords, description, and context paths. Short queries are more
    # ambiguous and should be let through to the LLM.
    if not has_overlap and len(query_words) >= 5:
        top_keywords = ", ".join(list(spec.context_keywords)[:5])
        return {
            "in_scope": False,
            "message": (
                f"This question doesn't appear to be about my domain. "
                f"I handle: {top_keywords}."
            ),
            "suggested_agent": None,
        }

    # Check 2: Exclusion list — if 2+ exclusion terms match, reject.
    routing = getattr(spec, "routing", None)
    if routing and routing.exclusions:
        excl_matches = sum(1 for excl in routing.exclusions if excl.lower() in query_lower)
        if excl_matches >= 2:
            return {
                "in_scope": False,
                "message": "This question is outside my scope based on exclusion rules.",
                "suggested_agent": None,
            }

    return {"in_scope": True, "message": ""}


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Agents are specialized components that handle specific types of queries.
    Each agent maintains its own context and can respond to queries within
    its domain of expertise.

    Subclasses must implement the `build_context` method to provide
    domain-specific context for the agent.

    Features:
        - Simple in-memory caching (default)
        - Git-aware caching with automatic invalidation (optional)
        - File-based cache invalidation (optional)

    Example:
        >>> class MyAgent(BaseAgent):
        ...     def build_context(self) -> str:
        ...         return "My specialized context..."
        ...
        >>> agent = MyAgent(spec, client)
        >>> response = agent.run("How does X work?", session)

    With git-aware caching:
        >>> from agenthub.cache import GitAwareCache
        >>> cache = GitAwareCache("/path/to/repo")
        >>> agent = MyAgent(spec, client, cache=cache)
        >>> # Context auto-refreshes when git commit changes
    """

    def __init__(
        self,
        spec: AgentSpec,
        client: "anthropic.Anthropic",
        cache: Optional["GitAwareCache"] = None,
    ):
        """Initialize the agent.

        Args:
            spec: Agent specification defining capabilities and behavior.
            client: Anthropic client for API calls.
            cache: Optional GitAwareCache for intelligent cache invalidation.
                   If provided, context will auto-refresh on git changes.
        """
        self.spec = spec
        self.client = client
        self._context_cache: Optional[str] = None
        self._git_cache = cache

    @abstractmethod
    def build_context(self) -> str:
        """Build the agent's specialized context.

        Override this method to load relevant files, schemas, documentation,
        or other context that the agent needs to respond effectively.

        Returns:
            String containing the agent's context.
        """
        pass

    def get_context(self, force_refresh: bool = False) -> str:
        """Get cached context or rebuild.

        If a GitAwareCache was provided, uses git-aware invalidation:
        - Automatically refreshes when git commit changes
        - Automatically refreshes when watched files change
        - Respects TTL if configured

        Args:
            force_refresh: If True, rebuild context even if cached.

        Returns:
            The agent's context string.
        """
        # Force refresh clears all caches
        if force_refresh:
            self.clear_context_cache()

        # Use git-aware cache if available
        if self._git_cache is not None:
            return self._git_cache.get_or_compute(
                key=self.spec.agent_id,
                compute_fn=self.build_context,
                watch_paths=self.spec.context_paths,
                track_git=True,
            )

        # Fall back to simple in-memory cache
        if self._context_cache is None:
            self._context_cache = self.build_context()
        return self._context_cache

    def clear_context_cache(self) -> None:
        """Clear the cached context.

        Clears both in-memory cache and git-aware cache if present.
        """
        self._context_cache = None
        if self._git_cache is not None:
            self._git_cache.invalidate(self.spec.agent_id)

    def set_cache(self, cache: "GitAwareCache") -> None:
        """Set or replace the git-aware cache.

        Args:
            cache: GitAwareCache instance to use.
        """
        self._git_cache = cache
        self._context_cache = None  # Clear simple cache

    def run(
        self,
        query: str,
        session: Session,
        model: Optional[str] = None,
        injected_context: str = "",
        max_tool_tokens: int = 30000,
    ) -> AgentResponse:
        """Execute the agent on a query.

        Three-stage scope filtering before the expensive full-context LLM call:
        1. Free heuristic check (_quick_scope_check) — keyword overlap, exclusions
        2. Cheap LLM pre-screen (_llm_scope_prescreen) — Haiku with ~500 tokens
        3. Full context load + main LLM call — only if both checks pass

        Args:
            query: User's query to process.
            session: Current conversation session.
            model: Optional model override.
            injected_context: Optional context from related agents (cross-agent sharing).
            max_tool_tokens: Token budget for domain tool usage (default 30K,
                reduced in team mode to prevent token explosion).

        Returns:
            AgentResponse with content and metadata.
        """
        # Stage 1: Free heuristic check — avoid any LLM call
        scope_check = self._quick_scope_check(query)
        if not scope_check["in_scope"]:
            return AgentResponse(
                content=scope_check["message"],
                agent_id=self.spec.agent_id,
                session_id=session.session_id,
                tokens_used=0,
                artifacts=[],
                needs_followup=True,
                suggested_agent=scope_check.get("suggested_agent"),
                metadata={
                    "scope_rejected": True,
                    "rejection_method": "heuristic",
                    "suggested_agent": scope_check.get("suggested_agent"),
                },
            )

        # Stage 2: Cheap LLM pre-screen (Haiku, ~500 tokens) — only for Tier B agents.
        # Avoids loading full context (up to 50K chars) just to get a rejection.
        prescreen = self._llm_scope_prescreen(query)
        if not prescreen["in_scope"]:
            return AgentResponse(
                content=prescreen["message"],
                agent_id=self.spec.agent_id,
                session_id=session.session_id,
                tokens_used=prescreen.get("tokens_used", 0),
                artifacts=[],
                needs_followup=True,
                metadata={
                    "scope_rejected": True,
                    "rejection_method": "llm_prescreen",
                    "suggested_agent": None,
                },
            )

        # Stage 3: Full context load + main LLM call
        # Check for enhanced tools first (exploration-first agents)
        enhanced = self._get_enhanced_tools()
        if enhanced is not None:
            tool_defs, executor = enhanced
            messages = self._build_messages(query, session)
            system_prompt = self._build_system_prompt(
                injected_context, tools_active=False, enhanced_tools_active=True
            )
            content, tokens_used = self._run_with_tools_enhanced(
                model=model or "claude-opus-4-20250514",
                system_prompt=system_prompt,
                messages=messages,
                tool_defs=tool_defs,
                executor=executor,
            )
            # Include pending changes in metadata
            pending = executor.get_pending_changes()
            metadata: dict[str, Any] = {
                "used_tools": True,
                "enhanced_tools": True,
                "files_explored": list(executor._files_read.keys()),
            }
            if pending:
                metadata["pending_changes"] = [
                    {"path": c.path, "description": c.description, "diff": c.unified_diff}
                    for c in pending
                ]
            return AgentResponse(
                content=content,
                agent_id=self.spec.agent_id,
                session_id=session.session_id,
                tokens_used=tokens_used,
                artifacts=self._extract_artifacts(content),
                metadata=metadata,
            )

        # Fallback: existing domain tools or no-tools path
        tools_and_executor = self._get_domain_tools()
        has_tools = tools_and_executor is not None

        messages = self._build_messages(query, session)
        system_prompt = self._build_system_prompt(
            injected_context, tools_active=has_tools
        )

        if tools_and_executor is None:
            # Original single-shot path (no tools available)
            response = self.client.messages.create(
                model=model or "claude-opus-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                temperature=self.spec.temperature,
            )
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
        else:
            # Agentic tool-use path (domain-scoped)
            tool_defs, executor = tools_and_executor
            content, tokens_used = self._run_with_tools(
                model=model or "claude-opus-4-20250514",
                system_prompt=system_prompt,
                messages=messages,
                tool_defs=tool_defs,
                executor=executor,
                max_tool_tokens=max_tool_tokens,
            )

        return AgentResponse(
            content=content,
            agent_id=self.spec.agent_id,
            session_id=session.session_id,
            tokens_used=tokens_used,
            artifacts=self._extract_artifacts(content),
            metadata={"used_tools": has_tools},
        )

    def _quick_scope_check(self, query: str) -> dict:
        """Quick check if query is in this agent's scope without LLM call.

        This is a cheap heuristic check to avoid expensive LLM calls for
        queries that are clearly out of scope.

        Args:
            query: User's query to check.

        Returns:
            Dict with 'in_scope' bool, 'message' for rejection, and optional 'suggested_agent'.
        """
        # Default: assume in scope (let LLM decide)
        # Subclasses can override for more specific checks
        return {"in_scope": True, "message": ""}

    def _llm_scope_prescreen(self, query: str) -> dict:
        """Lightweight LLM scope check using Haiku with minimal context.

        Uses only agent metadata (name, description, keywords, R&R) — roughly
        500 input tokens — to decide if the query is in scope. This avoids
        loading full context (up to 50K chars) for obviously out-of-scope queries.

        Only runs for Tier B (auto-generated) agents. Tier A agents skip this
        because they have better-targeted routing and manually curated scope.

        Args:
            query: User's query to check.

        Returns:
            Dict with 'in_scope' bool, 'message' str, and optional 'tokens_used' int.
        """
        # Only pre-screen auto-generated (Tier B) agents
        if not self.spec.metadata.get("auto_generated"):
            return {"in_scope": True, "message": ""}

        # Build minimal prompt from metadata (no full context)
        rnr = self.spec.metadata.get("rnr", {})
        in_scope_items = rnr.get("in_scope", [])
        out_of_scope_items = rnr.get("out_of_scope", [])

        system = (
            f"You are a routing classifier. Decide if a query belongs to agent '{self.spec.name}'.\n"
            f"IN SCOPE: {', '.join(in_scope_items[:5]) if in_scope_items else self.spec.description}\n"
            f"OUT OF SCOPE: {', '.join(out_of_scope_items[:5]) if out_of_scope_items else 'N/A'}\n"
            f"Keywords: {', '.join(self.spec.context_keywords[:10])}\n\n"
            f"Reply ONLY 'yes' or 'no'. Is this query in scope for this agent?"
        )

        try:
            response = self.client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=10,
                system=system,
                messages=[{"role": "user", "content": query}],
                temperature=0.0,
            )
            answer = response.content[0].text.strip().lower()
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            if answer.startswith("no"):
                return {
                    "in_scope": False,
                    "message": (
                        f"This question appears outside my area of expertise. "
                        f"I handle {self.spec.name.lower()}."
                    ),
                    "tokens_used": tokens_used,
                }
            return {"in_scope": True, "message": "", "tokens_used": tokens_used}
        except Exception:
            # Fail open — on any error, assume in scope
            return {"in_scope": True, "message": ""}

    def _build_system_prompt(
        self,
        injected_context: str = "",
        tools_active: bool = False,
        enhanced_tools_active: bool = False,
    ) -> str:
        """Combine spec prompt with context.

        Args:
            injected_context: Optional context from related agents.
            tools_active: If True, reduce static context and add domain tool docs.
            enhanced_tools_active: If True, use exploration-first prompt with
                project-wide tools (no pre-loaded context dump).

        Returns:
            Complete system prompt for the agent.
        """
        base_prompt = self.spec.system_prompt or f"You are {self.spec.name}. {self.spec.description}"

        if enhanced_tools_active:
            # Exploration-first mode: no context dump, rely on tools
            return self._build_enhanced_system_prompt(base_prompt, injected_context)

        # When domain tools are active, reduce static context to leave room for tool results
        if tools_active:
            orig_max = self.spec.max_context_size
            self.spec.max_context_size = min(orig_max, 20000)
            context = self.get_context(force_refresh=True)
            self.spec.max_context_size = orig_max
        else:
            context = self.get_context()

        prompt = f"""{base_prompt}

## Your Specialized Context

{context}
"""
        # Add cross-agent context if provided
        if injected_context:
            prompt += f"""
{injected_context}
"""

        if tools_active:
            prompt += """## Available Tools

You have tools to search and read files within your domain.
Use them when the answer isn't fully covered in the pre-loaded context above.
- **grep_domain**: search for regex patterns across your domain files
- **read_file**: read the contents of a specific file
- **list_files**: list files in a directory

Strategy: First check if the pre-loaded context answers the question.
If not, use grep_domain to find relevant files, then read_file to get details.
Always note which files you found information in — your final answer MUST cite
the specific file paths (e.g. `backend/app/services/insight.py`).
"""

        prompt += """## Instructions

- Focus only on your domain of expertise
- If a query is outside your scope, say so clearly
- **ALWAYS cite specific file paths** when referencing code (e.g. "in `backend/app/services/insight.py`", "the `PricingEngine` class in `backend/app/services/best_price.py`")
- When describing implementations, name the exact source file, class, and method
- Be concise but thorough
"""
        return prompt

    def _build_enhanced_system_prompt(
        self, base_prompt: str, injected_context: str = ""
    ) -> str:
        """Build exploration-first system prompt for enhanced Tier B agents.

        Instead of pre-loading file content, this prompt tells the agent to
        use its tools to explore the codebase autonomously — matching the
        FDA Local Worker pattern.
        """
        from agenthub.agents.project_tools import TechStackDetector

        root_path = self.spec.metadata.get("root_path", "")
        tech_stack: dict[str, list[str]] = {}
        if root_path:
            tech_stack = TechStackDetector.detect(root_path)

        tech_info = ""
        if tech_stack.get("languages"):
            tech_info += f"Languages: {', '.join(tech_stack['languages'])}\n"
        if tech_stack.get("frameworks"):
            tech_info += f"Frameworks: {', '.join(tech_stack['frameworks'])}\n"
        if tech_stack.get("build_tools"):
            tech_info += f"Build tools: {', '.join(tech_stack['build_tools'])}\n"
        if tech_stack.get("test_frameworks"):
            tech_info += f"Test frameworks: {', '.join(tech_stack['test_frameworks'])}\n"

        # Include agent's known context paths as hints
        context_hints = ""
        if self.spec.context_paths:
            paths = self.spec.context_paths[:20]
            context_hints = "Key files/directories in your domain:\n"
            context_hints += "\n".join(f"  - {p}" for p in paths)

        prompt = f"""{base_prompt}

## Project Info

{tech_info if tech_info else "Tech stack: unknown (use list_directory and read_file to discover)"}

{context_hints}

## Available Tools

You have powerful tools to explore and work with the entire project:

- **list_directory**: List files/dirs in a directory (start here to understand structure)
- **read_file**: Read file contents with line numbers (use to understand implementations)
- **search_files**: Regex search across the project (find functions, classes, patterns)
- **write_file**: Record proposed file changes (NOT applied to disk — stored for review)
- **run_command**: Execute shell commands (run tests, check versions, lint, etc.)

## Exploration Strategy

Do NOT assume you know the codebase. Use your tools to explore:
1. Start with `list_directory` to understand the project structure
2. Use `search_files` to find relevant code (functions, classes, imports)
3. Use `read_file` to understand specific implementations
4. If changes are needed, use `write_file` to record them (requires approval)
5. Use `run_command` for tasks like running tests or checking build status

Always cite the specific file paths you found information in.

## write_file Guidelines

- Always `read_file` before `write_file` — understand the current state first
- Provide COMPLETE file content, not just the changed parts
- Changes are recorded but NOT applied — they require separate approval
- Include a clear description of what the change does

## run_command Guidelines

- Commands have a 30-second timeout and 10KB output limit
- Dangerous commands (rm -rf /, dd, mkfs, etc.) are blocked
- Use for: running tests, checking versions, linting, grepping, building
"""

        if injected_context:
            prompt += f"""
## Context from Related Agents

{injected_context}
"""

        prompt += """## Instructions

- Focus on your domain of expertise but explore freely within the project
- If a query is outside your scope, say so clearly and suggest which agent to ask
- **ALWAYS cite specific file paths** when referencing code
- When describing implementations, name the exact source file, class, and method
- Be thorough — explore until you have a confident answer
"""
        return prompt

    def _build_messages(self, query: str, session: Session) -> list[dict[str, str]]:
        """Build message list from session history + new query.

        Args:
            query: The new user query.
            session: Current session with message history.

        Returns:
            List of message dicts for the API.
        """
        messages: list[dict[str, str]] = []

        # Add history (limit to avoid context overflow)
        for msg in session.messages[-10:]:  # Last 10 messages
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})

        # Add new query
        messages.append({"role": "user", "content": query})

        return messages

    def _extract_artifacts(self, content: str) -> list[Artifact]:
        """Extract code blocks and other artifacts from response.

        Args:
            content: Response content to parse.

        Returns:
            List of extracted artifacts.
        """
        artifacts: list[Artifact] = []

        # Extract code blocks
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)

        for lang, code in matches:
            artifacts.append(
                Artifact(
                    artifact_type="code",
                    content=code.strip(),
                    language=lang or "text",
                )
            )

        return artifacts

    # =========================================================================
    # Domain-scoped tool support
    # =========================================================================

    def _get_domain_tools(
        self,
    ) -> Optional[tuple[list[dict[str, Any]], "DomainToolExecutor"]]:
        """Get domain tools if available for this agent.

        Returns tool definitions + executor, or None if tools aren't
        available (agent has no root_path or context_paths).
        """
        try:
            from agenthub.agents.domain_tools import create_domain_tools

            return create_domain_tools(self)
        except Exception:
            return None

    def _get_enhanced_tools(
        self,
    ) -> Optional[tuple[list[dict[str, Any]], "ProjectToolExecutor"]]:
        """Get project-scoped tools for enhanced (exploration-first) agents.

        Returns tool definitions + executor, or None if the agent doesn't
        have ``enhanced_tools`` enabled in its metadata.
        """
        if not self.spec.metadata.get("enhanced_tools"):
            return None
        try:
            from agenthub.agents.project_tools import create_project_tools

            return create_project_tools(self)
        except Exception:
            return None

    def _run_with_tools(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        tool_defs: list[dict[str, Any]],
        executor: "DomainToolExecutor",
        max_iterations: int = 5,
        max_tool_tokens: int = 30000,
    ) -> tuple[str, int]:
        """Run the agentic tool-use loop.

        Calls the LLM with tools, executes any requested tools, sends
        results back, and repeats until the model produces a final answer
        or budget/iteration limits are hit.

        Returns:
            Tuple of (final_text_content, total_tokens_used).
        """
        # Adaptive iteration cap: tighter budget → fewer iterations
        if max_tool_tokens <= 15000:
            max_iterations = min(max_iterations, 3)

        total_tokens = 0

        for iteration in range(max_iterations):
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                tools=tool_defs,
                temperature=self.spec.temperature,
            )
            total_tokens += (
                response.usage.input_tokens + response.usage.output_tokens
            )

            # If the model is done, extract text and return
            if response.stop_reason == "end_turn":
                text = self._extract_text_from_content(response.content)
                return text, total_tokens

            # If the model wants to use tools
            if response.stop_reason == "tool_use":
                # Add assistant message with tool_use blocks.
                # Convert Pydantic models to dicts to avoid serialization
                # errors (by_alias) on subsequent API calls.
                serialized_content = [
                    block.model_dump() for block in response.content
                ]
                messages.append(
                    {"role": "assistant", "content": serialized_content}
                )

                # Execute each tool and collect results
                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.debug(
                            f"Agent {self.spec.agent_id} calling "
                            f"{block.name}({block.input})"
                        )
                        result = executor.execute(block.name, block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )

                # Send tool results back
                messages.append({"role": "user", "content": tool_results})

            # Check token budget
            if total_tokens >= max_tool_tokens:
                logger.warning(
                    f"Agent {self.spec.agent_id} hit tool token budget "
                    f"({total_tokens}/{max_tool_tokens}) at iteration "
                    f"{iteration + 1}"
                )
                break

        # Loop exhausted or budget exceeded — force a final answer without tools.
        # Add a nudge so the model summarises what it found instead of trying
        # to call more tools.
        logger.info(
            f"Agent {self.spec.agent_id}: forcing final answer "
            f"(tokens={total_tokens}, messages={len(messages)})"
        )
        messages.append({
            "role": "user",
            "content": (
                "Please provide your final answer now based on what you've "
                "found so far. Summarize the key findings concisely. "
                "IMPORTANT: Always cite specific file paths (e.g. `backend/app/services/foo.py`) "
                "when referencing code implementations."
            ),
        })
        response = self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            temperature=self.spec.temperature,
        )
        total_tokens += (
            response.usage.input_tokens + response.usage.output_tokens
        )
        text = self._extract_text_from_content(response.content)
        if not text:
            # Fallback: try to extract text from earlier tool-use responses
            logger.warning(
                f"Agent {self.spec.agent_id}: forced final answer was empty "
                f"(stop_reason={response.stop_reason}, "
                f"content_types={[b.type for b in response.content]})"
            )
            # Walk backwards through messages to find the last assistant text
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, str):
                        text = content
                        break
                    elif isinstance(content, list):
                        extracted = self._extract_text_from_content(content)
                        if extracted:
                            text = extracted
                            break
        return text, total_tokens

    def _run_with_tools_enhanced(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        tool_defs: list[dict[str, Any]],
        executor: "ProjectToolExecutor",
        max_iterations: int = 15,
        timeout_seconds: float = 300.0,
    ) -> tuple[str, int]:
        """Run the exploration-first agentic tool-use loop.

        Unlike ``_run_with_tools`` (token-budget, 5 iterations, domain-scoped),
        this uses wall-clock timeout (300s), up to 15 iterations, and
        project-wide tools. On timeout, forces a graceful summary instead
        of losing work.

        Returns:
            Tuple of (final_text_content, total_tokens_used).
        """
        start = time.monotonic()
        total_tokens = 0
        TOKEN_WARNING_THRESHOLD = 100_000

        for iteration in range(max_iterations):
            # Check timeout at start of iteration
            elapsed = time.monotonic() - start
            if elapsed >= timeout_seconds:
                logger.warning(
                    f"Agent {self.spec.agent_id} hit timeout "
                    f"({elapsed:.0f}s/{timeout_seconds:.0f}s) at iteration "
                    f"{iteration}"
                )
                break

            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                tools=tool_defs,
                temperature=self.spec.temperature,
            )
            total_tokens += (
                response.usage.input_tokens + response.usage.output_tokens
            )

            # If the model is done, extract text and return
            if response.stop_reason == "end_turn":
                text = self._extract_text_from_content(response.content)
                return text, total_tokens

            # If the model wants to use tools
            if response.stop_reason == "tool_use":
                serialized_content = [
                    block.model_dump() for block in response.content
                ]
                messages.append(
                    {"role": "assistant", "content": serialized_content}
                )

                # Execute each tool and collect results
                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.debug(
                            f"Agent {self.spec.agent_id} calling "
                            f"{block.name}({block.input})"
                        )
                        result = executor.execute(block.name, block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )

                messages.append({"role": "user", "content": tool_results})

            # Check timeout after tool execution
            elapsed = time.monotonic() - start
            if elapsed >= timeout_seconds:
                logger.warning(
                    f"Agent {self.spec.agent_id} hit timeout after tools "
                    f"({elapsed:.0f}s/{timeout_seconds:.0f}s) at iteration "
                    f"{iteration + 1}"
                )
                break

            # Soft token warning — nudge agent to wrap up
            if total_tokens >= TOKEN_WARNING_THRESHOLD:
                logger.info(
                    f"Agent {self.spec.agent_id} at {total_tokens} tokens, "
                    f"adding wrap-up nudge"
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "[System note: You are approaching the token limit. "
                        "Please wrap up your exploration and provide your answer soon.]"
                    ),
                })

        # Loop exhausted or timeout — force a final answer without tools.
        logger.info(
            f"Agent {self.spec.agent_id}: forcing final answer "
            f"(tokens={total_tokens}, elapsed={time.monotonic() - start:.0f}s, "
            f"messages={len(messages)})"
        )

        # Include executor summary so the model knows what it found
        summary = executor.get_summary()
        messages.append({
            "role": "user",
            "content": (
                "Please provide your final answer now based on what you've "
                "found so far. Summarize the key findings concisely.\n\n"
                f"Your exploration summary:\n{summary}\n\n"
                "IMPORTANT: Always cite specific file paths "
                "(e.g. `backend/app/services/foo.py`) when referencing code."
            ),
        })
        response = self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            temperature=self.spec.temperature,
        )
        total_tokens += (
            response.usage.input_tokens + response.usage.output_tokens
        )
        text = self._extract_text_from_content(response.content)
        if not text:
            # Fallback: walk backwards to find last assistant text
            logger.warning(
                f"Agent {self.spec.agent_id}: forced final answer was empty"
            )
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, str):
                        text = content
                        break
                    elif isinstance(content, list):
                        extracted = self._extract_text_from_content(content)
                        if extracted:
                            text = extracted
                            break
        return text, total_tokens

    @staticmethod
    def _extract_text_from_content(content: list) -> str:
        """Extract text from a list of content blocks.

        Handles both Pydantic model objects (from API responses) and
        plain dicts (from serialized messages).
        """
        parts = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
        return "\n".join(parts) if parts else ""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.spec.agent_id!r}>"

    # =========================================================================
    # QC Analysis Methods (for Tier B "raise concerns" feature)
    # =========================================================================

    def can_analyze_changes(self) -> bool:
        """Check if this agent supports change analysis.

        Returns:
            True if the agent can analyze code changes for concerns.
            Tier B (auto-generated) agents return True by default.
        """
        return self.spec.metadata.get("auto_generated", False)

    def get_affected_files(self, change_set: "ChangeSet") -> list[str]:
        """Get files from the change set that fall within this agent's domain.

        Args:
            change_set: Set of file changes to check.

        Returns:
            List of file paths that this agent is responsible for.
        """
        affected = []
        for file_change in change_set.files:
            for context_path in self.spec.context_paths:
                if self._path_matches(file_change.path, context_path):
                    affected.append(file_change.path)
                    break
        return affected

    def _path_matches(self, file_path: str, pattern: str) -> bool:
        """Check if a file path matches a context pattern."""
        # Normalize paths
        file_path = file_path.replace("\\", "/")
        pattern = pattern.replace("\\", "/")

        # Direct match or glob pattern match
        if fnmatch.fnmatch(file_path, pattern):
            return True

        # Check if file is within the pattern directory
        if pattern.endswith("/*") or pattern.endswith("/**"):
            dir_pattern = pattern.rstrip("/*")
            if file_path.startswith(dir_pattern):
                return True

        # Check if pattern is a prefix
        if file_path.startswith(pattern.rstrip("/")):
            return True

        return False

    def analyze_changes(
        self,
        change_set: "ChangeSet",
        affected_files: list[str] | None = None,
    ) -> "AgentAnalysisResult":
        """Analyze code changes and raise concerns.

        This is the main method for the "raise concerns" feature.
        Subclasses can override this for custom analysis logic.

        Args:
            change_set: Set of file changes to analyze.
            affected_files: Pre-filtered list of files in this agent's domain.
                           If None, will be computed from change_set.

        Returns:
            AgentAnalysisResult with any concerns found.
        """
        from agenthub.qc.models import AgentAnalysisResult

        start_time = time.time()

        # Get affected files if not provided
        if affected_files is None:
            affected_files = self.get_affected_files(change_set)

        # Skip if no files affect this agent
        if not affected_files:
            return AgentAnalysisResult(
                agent_id=self.spec.agent_id,
                domain=self.spec.name,
                analyzed_files=[],
                concerns=[],
                analysis_time_ms=0,
                tokens_used=0,
                skipped_reason="No files in domain",
            )

        # Build the analysis prompt
        prompt = self._build_analysis_prompt(change_set, affected_files)

        # Call the LLM
        response = self.client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=4096,
            system=self._build_analysis_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more consistent analysis
        )

        # Parse concerns from response
        concerns = self._parse_concerns_from_response(
            response.content[0].text,
            affected_files,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        return AgentAnalysisResult(
            agent_id=self.spec.agent_id,
            domain=self.spec.name,
            analyzed_files=affected_files,
            concerns=concerns,
            analysis_time_ms=elapsed_ms,
            tokens_used=tokens_used,
        )

    def _build_analysis_system_prompt(self) -> str:
        """Build the system prompt for change analysis."""
        return f"""You are {self.spec.name}, a code quality analyst for the {self.spec.name} domain.

Your job is to analyze code changes and identify potential concerns:
- Breaking changes (API changes, signature changes, behavior changes)
- Security issues (injection, auth bypass, data exposure)
- Performance concerns (N+1 queries, missing indexes, inefficient algorithms)
- Missing tests (new code without tests, broken test coverage)
- Code quality issues (complexity, duplication, naming)
- Documentation gaps (missing docstrings, outdated comments)
- Error handling issues (missing error handling, swallowed exceptions)
- Type safety issues (missing types, type mismatches)

For each concern, provide:
1. Category: breaking_change, security, performance, missing_tests, code_quality, documentation, error_handling, type_safety, other
2. Severity: critical, high, medium, low, info
3. Clear title
4. Detailed description
5. Affected files/functions
6. Suggestion for how to fix

Format your response as structured JSON for each concern found.
If no concerns are found, return an empty concerns array.

{self.spec.system_prompt or ''}"""

    def _build_analysis_prompt(
        self,
        change_set: "ChangeSet",
        affected_files: list[str],
    ) -> str:
        """Build the user prompt for change analysis."""
        # Get file changes for affected files
        changes_text = []
        for file_change in change_set.files:
            if file_change.path in affected_files:
                change_info = f"### {file_change.path} ({file_change.change_type})\n"
                if file_change.diff:
                    change_info += f"```diff\n{file_change.diff[:3000]}\n```\n"
                elif file_change.new_content:
                    change_info += f"```python\n{file_change.new_content[:3000]}\n```\n"
                changes_text.append(change_info)

        # Get current context for comparison
        context = self.get_context()

        return f"""Analyze these code changes for concerns in my domain:

## Changes
{chr(10).join(changes_text) if changes_text else "No change content available."}

## My Current Context
{context[:8000]}

## Instructions
1. Review each change carefully
2. Identify any concerns based on the categories above
3. For each concern, provide structured analysis
4. Be specific about file names, function names, and line numbers when possible
5. Suggest fixes where applicable

Respond in JSON format:
```json
{{
    "concerns": [
        {{
            "category": "...",
            "severity": "...",
            "title": "...",
            "description": "...",
            "affected_files": ["..."],
            "affected_functions": ["..."],
            "suggestion": "...",
            "code_snippet": "...",
            "confidence": 0.8
        }}
    ],
    "summary": "Brief summary of analysis"
}}
```"""

    def _parse_concerns_from_response(
        self,
        response_text: str,
        affected_files: list[str],
    ) -> list["Concern"]:
        """Parse concerns from LLM response."""
        from agenthub.qc.models import Concern, ConcernCategory, ConcernSeverity

        concerns = []

        # Try to extract JSON from response
        json_match = re.search(r"```json\n?(.*?)```", response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                for item in data.get("concerns", []):
                    try:
                        # Map category string to enum
                        category_str = item.get("category", "other").lower()
                        try:
                            category = ConcernCategory(category_str)
                        except ValueError:
                            category = ConcernCategory.OTHER

                        # Map severity string to enum
                        severity_str = item.get("severity", "medium").lower()
                        try:
                            severity = ConcernSeverity(severity_str)
                        except ValueError:
                            severity = ConcernSeverity.MEDIUM

                        concern = Concern(
                            concern_id=str(uuid.uuid4())[:8],
                            agent_id=self.spec.agent_id,
                            domain=self.spec.name,
                            category=category,
                            severity=severity,
                            title=item.get("title", "Unnamed concern"),
                            description=item.get("description", ""),
                            affected_files=item.get("affected_files", affected_files[:3]),
                            affected_functions=item.get("affected_functions", []),
                            suggestion=item.get("suggestion"),
                            code_snippet=item.get("code_snippet"),
                            confidence=float(item.get("confidence", 0.8)),
                        )
                        concerns.append(concern)
                    except Exception:
                        pass  # Skip malformed concerns
            except json.JSONDecodeError:
                pass  # Could not parse JSON

        return concerns
