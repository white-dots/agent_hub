"""Base agent implementation."""

import fnmatch
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from agenthub.models import AgentResponse, AgentSpec, Artifact, Message, Session

if TYPE_CHECKING:
    import anthropic

    from agenthub.cache import GitAwareCache
    from agenthub.qc.models import AgentAnalysisResult, ChangeSet, Concern


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
    ) -> AgentResponse:
        """Execute the agent on a query.

        Args:
            query: User's query to process.
            session: Current conversation session.
            model: Optional model override.
            injected_context: Optional context from related agents (cross-agent sharing).

        Returns:
            AgentResponse with content and metadata.
        """
        # Build messages
        messages = self._build_messages(query, session)

        # Call API
        response = self.client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            max_tokens=4096,
            system=self._build_system_prompt(injected_context),
            messages=messages,
            temperature=self.spec.temperature,
        )

        # Parse response
        content = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        return AgentResponse(
            content=content,
            agent_id=self.spec.agent_id,
            session_id=session.session_id,
            tokens_used=tokens_used,
            artifacts=self._extract_artifacts(content),
        )

    def _build_system_prompt(self, injected_context: str = "") -> str:
        """Combine spec prompt with context.

        Args:
            injected_context: Optional context from related agents.

        Returns:
            Complete system prompt for the agent.
        """
        context = self.get_context()
        base_prompt = self.spec.system_prompt or f"You are {self.spec.name}. {self.spec.description}"

        prompt = f"""{base_prompt}

## Your Specialized Context

{context}
"""
        # Add cross-agent context if provided
        if injected_context:
            prompt += f"""
{injected_context}
"""

        prompt += """## Instructions

- Focus only on your domain of expertise
- If a query is outside your scope, say so clearly
- Reference the context above when answering
- Be concise but thorough
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
            model="claude-sonnet-4-20250514",
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
