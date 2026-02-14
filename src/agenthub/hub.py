from __future__ import annotations
"""AgentHub - Central orchestrator for agent routing and management."""

import hashlib
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

from agenthub.models import AgentResponse, AgentSpec, Message, Session
from agenthub.routing import KeywordRouter, Router, TierAwareRouter

if TYPE_CHECKING:
    import anthropic

    from agenthub.agents.base import BaseAgent
    from agenthub.auto.cross_context import CrossAgentContextManager, CrossContextConfig
    from agenthub.auto.import_graph import ImportGraph
    from agenthub.auto.manager import AutoAgentManager
    from agenthub.auto.routing_index import RoutingIndex
    from agenthub.cache import FileWatcher, GitAwareCache, WatchConfig
    from agenthub.models import AgentContextStatus
    from agenthub.parallel import (
        Escalation,
        ParallelExecutionConfig,
        ParallelExecutionResult,
        ParallelSessionManager,
    )
    from agenthub.qc.models import ChangeSet, Concern, ConcernReport
    from agenthub.qc.pipeline import ChangeAnalysisPipeline
    from agenthub.qc.qc_agent import QCAgent
    from agenthub.teams.executor import DAGTeamExecutor


class AgentHub:
    """Central orchestrator for agent routing and management.

    AgentHub manages a collection of specialized agents and routes
    queries to the appropriate agent based on content analysis.

    Example:
        >>> from agenthub import AgentHub
        >>> hub = AgentHub()
        >>> hub.register(my_agent)
        >>> response = hub.run("How does authentication work?")
        >>> print(response.content)

    Attributes:
        client: Anthropic client for API calls.
    """

    def __init__(
        self,
        client: Optional["anthropic.Anthropic"] = None,
        router: Optional[Router] = None,
    ):
        """Initialize AgentHub.

        Args:
            client: Anthropic client. If None, creates one from environment.
            router: Custom router implementation. Defaults to TierAwareRouter.
        """
        if client is None:
            import anthropic

            client = anthropic.Anthropic()

        self.client = client
        self._agents: dict[str, "BaseAgent"] = {}
        self._sessions: dict[str, Session] = {}
        self._router: Router = router or TierAwareRouter()

        # For auto-agent management
        self._auto_manager: Optional["AutoAgentManager"] = None

        # For file watching and caching
        self._file_watcher: Optional["FileWatcher"] = None
        self._git_cache: Optional["GitAwareCache"] = None
        self._on_context_refresh: Optional[Callable[[str, list[str]], None]] = None

        # For QC analysis (Tier C)
        self._qc_agent: Optional["QCAgent"] = None
        self._analysis_pipeline: Optional["ChangeAnalysisPipeline"] = None
        self._on_concern_raised: Optional[Callable[["Concern"], None]] = None
        self._on_qc_report: Optional[Callable[["ConcernReport"], None]] = None

        # For cross-agent context sharing
        self._cross_context_manager: Optional["CrossAgentContextManager"] = None
        self._cross_context_enabled: bool = False

        # For DAG team execution
        self._team_executor: Optional["DAGTeamExecutor"] = None
        self._complexity_threshold: float = 0.4
        self._import_graph: Optional["ImportGraph"] = None  # Set by discover_all_agents

        # Team execution traces storage (session_id -> trace)
        self._team_traces: dict[str, dict] = {}
        self._max_team_traces: int = 100  # Keep last 100 traces
        self._traces_file: Optional[Path] = None  # File path for persisting traces

        # For parallel sessions
        self._parallel_manager: Optional["ParallelSessionManager"] = None
        self._on_parallel_escalation: Optional[Callable[["Escalation"], None]] = None

        # Response cache for deduplication (saves tokens on repeated queries)
        self._response_cache: dict[str, tuple[AgentResponse, float]] = {}
        self._response_cache_ttl: float = 300.0  # 5 minutes default
        self._response_cache_max_size: int = 100  # Max cached responses

        # Pre-computed routing index (built at setup time by discover_all_agents)
        self._routing_index: Optional["RoutingIndex"] = None
        self._project_root: Optional[str] = None  # Set by discover_all_agents

    # ==================== Agent Registration ====================

    def register(self, agent: "BaseAgent") -> None:
        """Register an agent with the hub.

        Args:
            agent: Agent instance to register.

        Raises:
            ValueError: If agent with same ID already registered.
        """
        if agent.spec.agent_id in self._agents:
            raise ValueError(f"Agent '{agent.spec.agent_id}' already registered")

        self._agents[agent.spec.agent_id] = agent

    def unregister(self, agent_id: str) -> None:
        """Remove an agent from the hub.

        Args:
            agent_id: ID of agent to remove.
        """
        if agent_id in self._agents:
            del self._agents[agent_id]

    def unregister_tier_b(self) -> list[str]:
        """Remove all Tier B (auto-generated) agents from the hub.

        Returns:
            List of agent IDs that were removed.
        """
        tier_b_ids = [
            agent_id for agent_id, agent in self._agents.items()
            if agent.spec.metadata.get("auto_generated", False)
            and agent.spec.metadata.get("tier") != "C"  # Keep QC agent
        ]

        for agent_id in tier_b_ids:
            del self._agents[agent_id]

        return tier_b_ids

    def get_agent(self, agent_id: str) -> Optional["BaseAgent"]:
        """Get a registered agent by ID.

        Args:
            agent_id: ID of agent to retrieve.

        Returns:
            Agent instance or None if not found.
        """
        return self._agents.get(agent_id)

    def list_agents(self, tier: Optional[str] = None) -> list[AgentSpec]:
        """List all registered agent specs.

        Args:
            tier: Filter by tier - "A" for business, "B" for auto-generated,
                  "C" for meta-agents, None for all.

        Returns:
            List of agent specifications.
        """
        all_agents = [a.spec for a in self._agents.values()]

        if tier is None:
            return all_agents

        # Helper to classify agent tier
        def get_tier(spec: AgentSpec) -> str:
            explicit_tier = spec.metadata.get("tier")
            if explicit_tier in ("A", "B", "C"):
                return explicit_tier
            return "B" if spec.metadata.get("auto_generated") else "A"

        if tier in ("A", "B", "C"):
            return [a for a in all_agents if get_tier(a) == tier]
        else:
            raise ValueError(f"Unknown tier: {tier}. Use 'A', 'B', 'C', or None.")

    # ==================== Session Management ====================

    def create_session(self, agent_id: Optional[str] = None) -> Session:
        """Create a new conversation session.

        Args:
            agent_id: Optional agent to associate with session.

        Returns:
            New session instance.
        """
        session = Session(
            session_id=str(uuid.uuid4()),
            agent_id=agent_id or "router",
        )
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve an existing session.

        Args:
            session_id: ID of session to retrieve.

        Returns:
            Session instance or None if not found.
        """
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete a session.

        Args:
            session_id: ID of session to delete.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> list[Session]:
        """List all active sessions.

        Returns:
            List of session instances.
        """
        return list(self._sessions.values())

    # ==================== Routing ====================

    def set_router(self, router: Union[Router, callable]) -> None:
        """Set custom routing strategy.

        Args:
            router: Router instance or callable(query, agents) -> agent_id.
        """
        if callable(router) and not isinstance(router, Router):
            # Wrap callable in a simple router
            class CallableRouter:
                def __init__(self, fn: callable):
                    self._fn = fn

                def route(self, query: str, agents: list[AgentSpec]) -> Optional[str]:
                    return self._fn(query, agents)

            self._router = CallableRouter(router)
        else:
            self._router = router

    def route(self, query: str) -> str:
        """Determine which agent should handle a query.

        Uses pre-computed routing index if available (faster O(keywords) lookup),
        otherwise falls back to standard router (O(agents × keywords)).

        Supports per-agent routing settings: confidence thresholds and fallback
        chains. When the best-scoring agent's confidence threshold isn't met,
        follows the fallback chain (up to 3 hops).

        Args:
            query: User's query to route.

        Returns:
            Agent ID to handle the query.

        Raises:
            ValueError: If no agents registered or no match found.
        """
        if not self._agents:
            raise ValueError("No agents registered")

        # Try using pre-computed routing index first (built at setup time)
        # This is O(keywords) instead of O(agents × keywords)
        # The IndexedKeywordRouter handles confidence thresholds and
        # fallback chains internally via per-agent RoutingConfig.
        routing_index = getattr(self, "_routing_index", None)
        if routing_index is not None:
            from agenthub.auto.routing_index import IndexedKeywordRouter

            indexed_router = IndexedKeywordRouter(routing_index)
            agent_id = indexed_router.route(query)

            if agent_id and agent_id in self._agents:
                return agent_id

        # Fallback to standard router (also respects RoutingConfig)
        agents = self.list_agents()
        agent_id = self._router.route(query, agents)

        if agent_id and agent_id in self._agents:
            # Check confidence via fallback chain for standard router path
            resolved = self._resolve_with_fallback(agent_id, query)
            if resolved and resolved in self._agents:
                return resolved
            return agent_id

        # Fallback to first agent
        return list(self._agents.keys())[0]

    def _resolve_with_fallback(self, agent_id: str, query: str, max_depth: int = 3) -> Optional[str]:
        """Follow fallback chain if agent's confidence threshold isn't met.

        Args:
            agent_id: Starting agent ID.
            query: The user's query (for score re-checking).
            max_depth: Maximum fallback hops.

        Returns:
            Agent ID that accepts, or the original agent_id.
        """
        visited: set[str] = set()
        current = agent_id

        for _ in range(max_depth):
            if not current or current in visited or current not in self._agents:
                break
            visited.add(current)

            agent = self._agents[current]
            routing = getattr(agent.spec, "routing", None)

            # If no min_confidence set, agent accepts everything
            if not routing or routing.min_confidence <= 0.0:
                return current

            # Agent has a threshold — check if we should fall back
            if routing.fallback_agent_id and routing.fallback_agent_id in self._agents:
                current = routing.fallback_agent_id
            else:
                return current  # No fallback available, use this agent

        return agent_id  # Exhausted chain, return original

    def _find_retry_agent(
        self,
        suggested_agent: Optional[str],
        query: str,
        tried_agents: set[str],
    ) -> Optional[str]:
        """Find the next agent to try after a scope rejection.

        Strategy cascade:
        1. If the rejected agent suggested a specific agent ID, try that.
        2. Get scores from the router and pick the next-best untried agent.
        3. Use IndexedKeywordRouter if available (different scoring).
        4. Cross-tier: if only Tier A tried, try best Tier B, and vice versa.

        Args:
            suggested_agent: Agent ID or display name suggested by the rejecting agent.
            query: Original user query.
            tried_agents: Set of agent IDs already tried.

        Returns:
            Agent ID to try next, or None if no candidates.
        """
        # Strategy 1: Try the suggested agent (may be an ID or a display name)
        if suggested_agent:
            # Direct ID match
            if suggested_agent in self._agents and suggested_agent not in tried_agents:
                return suggested_agent
            # Fuzzy match on agent name/id (the suggested_agent might be
            # "Backend Expert" rather than "backend_agent")
            suggested_lower = suggested_agent.lower().replace(" ", "_")
            for aid, agent in self._agents.items():
                if aid in tried_agents:
                    continue
                if (suggested_lower in aid.lower() or
                        suggested_lower in agent.spec.name.lower().replace(" ", "_")):
                    return aid

        # Strategy 2: Get ranked scores and pick next-best untried agent
        agents = self.list_agents()
        scores: dict[str, float] = {}
        if isinstance(self._router, KeywordRouter):
            scores = self._router.get_all_scores(query, agents)
        elif isinstance(self._router, TierAwareRouter):
            scores = self._router._keyword_router.get_all_scores(query, agents)

        if scores:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for aid, score in ranked:
                if aid not in tried_agents and score > 0 and aid in self._agents:
                    return aid

        # Strategy 3: Use IndexedKeywordRouter if routing index exists
        if self._routing_index is not None:
            try:
                from agenthub.auto.routing_index import IndexedKeywordRouter

                indexed_router = IndexedKeywordRouter(self._routing_index)
                idx_scores = indexed_router.get_all_scores(query)
                idx_ranked = sorted(
                    idx_scores.items(), key=lambda x: x[1], reverse=True
                )
                for aid, score in idx_ranked:
                    if aid not in tried_agents and score > 0 and aid in self._agents:
                        return aid
            except Exception:
                pass  # Fail gracefully

        # Strategy 4: Cross-tier — try the other tier
        tried_tiers: set[str] = set()
        for aid in tried_agents:
            agent = self._agents.get(aid)
            if agent:
                is_auto = agent.spec.metadata.get("auto_generated", False)
                tried_tiers.add("B" if is_auto else "A")

        # If we haven't tried any Tier B agents, pick the best one
        if "B" not in tried_tiers:
            tier_b = [a for a in agents if a.metadata.get("auto_generated")]
            if tier_b:
                b_router = KeywordRouter()
                b_scores = b_router.get_all_scores(query, tier_b)
                for aid, score in sorted(
                    b_scores.items(), key=lambda x: x[1], reverse=True
                ):
                    if aid not in tried_agents and score > 0 and aid in self._agents:
                        return aid

        # If we haven't tried any Tier A agents, pick the best one
        if "A" not in tried_tiers:
            tier_a = [a for a in agents if not a.metadata.get("auto_generated")]
            if tier_a:
                a_router = KeywordRouter()
                a_scores = a_router.get_all_scores(query, tier_a)
                for aid, score in sorted(
                    a_scores.items(), key=lambda x: x[1], reverse=True
                ):
                    if aid not in tried_agents and score > 0 and aid in self._agents:
                        return aid

        # Last resort: try any untried agent (even with score 0)
        for aid in self._agents:
            if aid not in tried_agents:
                return aid

        return None

    # ==================== Execution ====================

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        model: Optional[str] = None,
        team_mode: str = "auto",
    ) -> AgentResponse:
        """Execute a query through the hub.

        This is the main entry point for running queries. It handles
        session management, routing, and agent execution.

        Args:
            query: User's query to process.
            session_id: Existing session ID (creates new if None).
            agent_id: Force specific agent (auto-routes if None).
            model: Override model for this call.
            team_mode: Controls DAG team execution.
                "auto" - use ComplexityClassifier to decide (default)
                "always" - always use DAG team (useful for testing)
                "never" - never use DAG team (original behavior)

        Returns:
            AgentResponse with content and metadata.

        Raises:
            ValueError: If session not found or agent not registered.
        """
        # Check response cache first (saves tokens on repeated queries)
        cache_key = self._get_cache_key(query, agent_id, team_mode)
        cached = self._get_cached_response(cache_key)
        if cached:
            # Return cached response with updated metadata
            cached.metadata["cache_hit"] = True
            return cached

        # Get or create session
        if session_id:
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
        else:
            session = self.create_session()

        # Add user message to session
        session.messages.append(Message(role="user", content=query))
        session.updated_at = datetime.now()

        # Check for team execution if not forcing specific agent
        if agent_id is None and team_mode != "never" and self._team_executor:
            from agenthub.teams.classifier import ComplexityClassifier

            classifier = ComplexityClassifier(self, threshold=self._complexity_threshold)
            result = classifier.classify(query)

            if result.is_complex or team_mode == "always":
                response = self._team_executor.execute(
                    query=query,
                    matched_agents=result.matched_agents,
                    session=session,
                )
                session.total_tokens_used += response.tokens_used

                # Store team trace for dashboard visualization
                if "trace" in response.metadata:
                    self._store_team_trace(session.session_id, query, response.metadata["trace"])

                # Cache the response for future identical queries
                self._cache_response(cache_key, response)
                return response

        # Route to agent (single agent path) with retry on scope rejection.
        # If the chosen agent rejects the query, try the suggested agent or
        # next-best-scoring agent. Cap at 5 retries to try both tiers.
        target_agent_id = agent_id or self.route(query)
        max_retries = 5
        tried_agents: set[str] = set()
        tried_order: list[str] = []  # Preserve insertion order for metadata
        response: Optional[AgentResponse] = None

        for attempt in range(max_retries + 1):
            if target_agent_id not in self._agents:
                raise ValueError(f"Agent {target_agent_id} not registered")

            if target_agent_id in tried_agents:
                break  # Already tried this agent, avoid loops
            tried_agents.add(target_agent_id)
            tried_order.append(target_agent_id)

            agent = self._agents[target_agent_id]

            # Get cross-agent context if enabled
            injected_context = ""
            if self._cross_context_enabled and self._cross_context_manager:
                from agenthub.auto.cross_context import format_injected_context

                related = self._cross_context_manager.get_related_context(
                    agent_id=target_agent_id,
                    query=query,
                )
                if related:
                    injected_context = format_injected_context(related)

            # Update session
            session.agent_id = target_agent_id

            # Execute with optional cross-agent context
            response = agent.run(query, session, model=model, injected_context=injected_context)

            # Check for scope rejection and retry with another agent
            if response.metadata.get("scope_rejected") and attempt < max_retries:
                next_agent_id = self._find_retry_agent(
                    suggested_agent=response.metadata.get("suggested_agent"),
                    query=query,
                    tried_agents=tried_agents,
                )
                if next_agent_id:
                    target_agent_id = next_agent_id
                    continue  # Retry with the new agent

            break  # Accept this response

        # LLM-based last resort: if 3+ agents rejected, use LLM routing
        # to semantically match the query to an untried agent. This is
        # expensive (one Haiku call) but only fires on clear routing failures.
        if (
            response
            and response.metadata.get("scope_rejected")
            and len(tried_agents) >= 3
        ):
            untried_specs = [
                a for a in self.list_agents()
                if a.agent_id not in tried_agents
            ]
            if untried_specs and self.client:
                try:
                    from agenthub.routing import LLMRouter

                    llm_router = LLMRouter(
                        self.client, model="claude-haiku-4-20250414"
                    )
                    llm_choice = llm_router.route(query, untried_specs)
                    if llm_choice and llm_choice in self._agents:
                        agent = self._agents[llm_choice]
                        session.agent_id = llm_choice
                        response = agent.run(
                            query, session, model=model,
                            injected_context=injected_context,
                        )
                        tried_order.append(llm_choice)
                        target_agent_id = llm_choice
                except Exception:
                    pass  # Fail gracefully — keep the last rejection response

        # Add retry metadata if retries were used
        if len(tried_order) > 1:
            response.metadata["retry_history"] = tried_order
            response.metadata["retries_used"] = len(tried_order) - 1

        # Mark agent as queried (for staleness tracking)
        self.mark_agent_queried(target_agent_id)

        # Store response in session
        session.messages.append(
            Message(
                role="assistant",
                content=response.content,
                metadata={"tokens": response.tokens_used, "agent": target_agent_id},
            )
        )
        session.total_tokens_used += response.tokens_used

        # Cache the response for future identical queries
        self._cache_response(cache_key, response)
        return response

    # ==================== Response Cache ====================

    def _get_cache_key(self, query: str, agent_id: Optional[str], team_mode: str) -> str:
        """Generate a cache key for a query.

        Args:
            query: The query string.
            agent_id: Optional specific agent ID.
            team_mode: Team execution mode.

        Returns:
            Cache key string (MD5 hash).
        """
        # Normalize query for caching (lowercase, strip whitespace)
        normalized_query = query.lower().strip()
        key_parts = f"{normalized_query}|{agent_id or 'auto'}|{team_mode}"
        return hashlib.md5(key_parts.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[AgentResponse]:
        """Get a cached response if it exists and is still valid.

        Args:
            cache_key: The cache key to look up.

        Returns:
            Cached AgentResponse if valid, None otherwise.
        """
        if cache_key not in self._response_cache:
            return None

        response, cached_at = self._response_cache[cache_key]
        elapsed = time.time() - cached_at

        if elapsed > self._response_cache_ttl:
            # Cache expired
            del self._response_cache[cache_key]
            return None

        return response

    def _cache_response(self, cache_key: str, response: AgentResponse) -> None:
        """Cache a response for future use.

        Args:
            cache_key: The cache key.
            response: The response to cache.
        """
        # Don't cache scope-rejected responses (they're already cheap)
        if response.metadata.get("scope_rejected"):
            return

        # Evict oldest entries if at capacity
        if len(self._response_cache) >= self._response_cache_max_size:
            # Remove oldest entry
            oldest_key = min(self._response_cache, key=lambda k: self._response_cache[k][1])
            del self._response_cache[oldest_key]

        self._response_cache[cache_key] = (response, time.time())

    def clear_response_cache(self) -> int:
        """Clear the response cache.

        Returns:
            Number of entries cleared.
        """
        count = len(self._response_cache)
        self._response_cache.clear()
        return count

    def set_response_cache_ttl(self, ttl_seconds: float) -> None:
        """Set the TTL for cached responses.

        Args:
            ttl_seconds: Time-to-live in seconds.
        """
        self._response_cache_ttl = ttl_seconds

    # ==================== Auto-Agents ====================

    def enable_auto_agents(
        self,
        project_root: str,
        config: Optional["AutoAgentConfig"] = None,
    ) -> list[str]:
        """Enable automatic agent generation for a project.

        This analyzes the project structure and creates specialized
        agents for each significant code module.

        Args:
            project_root: Path to the project to analyze.
            config: Auto-generation configuration.

        Returns:
            List of auto-generated agent IDs.
        """
        from agenthub.auto.manager import AutoAgentManager

        self._auto_manager = AutoAgentManager(
            hub=self,
            project_root=project_root,
            config=config,
        )
        return self._auto_manager.scan_and_register()

    def refresh_auto_agents(self) -> tuple[list[str], list[str]]:
        """Refresh auto-generated agents after code changes.

        Returns:
            Tuple of (added_agent_ids, removed_agent_ids).

        Raises:
            ValueError: If auto-agents not enabled.
        """
        if not self._auto_manager:
            raise ValueError("Auto-agents not enabled. Call enable_auto_agents first.")
        return self._auto_manager.refresh()

    def get_coverage_report(self) -> Optional[dict]:
        """Get coverage report for auto-generated agents.

        Returns:
            Dict with coverage statistics, or None if auto-agents not enabled.
        """
        if not self._auto_manager:
            return None
        return self._auto_manager.get_coverage_report()

    # ==================== Utilities ====================

    def refresh_all_contexts(self) -> None:
        """Force all agents to rebuild their context caches."""
        for agent in self._agents.values():
            agent.get_context(force_refresh=True)

    def rebuild_routing_index(self) -> Optional["RoutingIndex"]:
        """Rebuild the pre-computed routing index.

        Call this after registering/unregistering agents to update the index.
        The index is used for fast O(keywords) routing at query time.

        Returns:
            The rebuilt RoutingIndex, or None if no project root available.

        Example:
            >>> hub.register(new_agent)
            >>> hub.rebuild_routing_index()  # Update index to include new agent
        """
        if not self._project_root:
            return None

        from agenthub.auto.routing_index import RoutingIndexBuilder

        all_agent_specs = self.list_agents()
        if not all_agent_specs:
            return None

        builder = RoutingIndexBuilder(self._project_root)
        self._routing_index = builder.build(all_agent_specs)

        # Update cache
        builder.save(self._routing_index)

        return self._routing_index

    def get_routing_index_stats(self) -> Optional[dict]:
        """Get statistics about the pre-computed routing index.

        Returns:
            Dict with index statistics, or None if no index available.

        Example:
            >>> stats = hub.get_routing_index_stats()
            >>> print(f"Keywords: {stats['total_keywords']}")
            >>> print(f"Agents: {stats['agent_count']}")
        """
        if not self._routing_index:
            return None

        return {
            "agent_count": self._routing_index.agent_count,
            "total_keywords": len(self._routing_index.keyword_to_agents),
            "domains": list(self._routing_index.domain_to_agents.keys()),
            "generated_at": self._routing_index.generated_at,
            "git_commit": self._routing_index.git_commit,
        }

    def refresh_agent_keywords(self, agent_id: Optional[str] = None) -> dict[str, list[str]]:
        """Refresh routing keywords for agents based on current file content.

        Keywords are extracted from:
        - File/folder names in context_paths
        - Class and function names in the code
        - Important domain terms

        Args:
            agent_id: Specific agent to refresh, or None for all Tier B agents.

        Returns:
            Dict mapping agent_id to new keywords list.

        Example:
            >>> # Refresh all Tier B agents' keywords
            >>> updated = hub.refresh_agent_keywords()
            >>> print(f"Updated {len(updated)} agents")
            >>>
            >>> # Refresh specific agent
            >>> hub.refresh_agent_keywords("get_backend_agent")
        """
        from pathlib import Path
        import re
        from collections import Counter

        updated: dict[str, list[str]] = {}

        # Get project root for resolving paths
        project_root = getattr(self, '_project_root', None)
        if not project_root:
            return updated

        def extract_keywords_from_paths(paths: list[str], max_keywords: int = 12) -> list[str]:
            """Extract keywords from file paths and their content."""
            all_words: Counter[str] = Counter()

            for path_pattern in paths:
                # Handle glob patterns
                if "**" in path_pattern or "*" in path_pattern:
                    from glob import glob
                    full_pattern = str(Path(project_root) / path_pattern)
                    matching_files = glob(full_pattern, recursive=True)
                else:
                    matching_files = [str(Path(project_root) / path_pattern)]

                for file_path in matching_files[:50]:  # Limit files to scan
                    path = Path(file_path)
                    if not path.exists() or not path.is_file():
                        continue

                    # Extract from file/folder names
                    name_words = re.findall(r"[a-z]+", path.stem.lower())
                    for word in name_words:
                        if len(word) > 2 and word not in {"init", "test", "tests", "main", "index"}:
                            all_words[word] += 3  # High weight for file names

                    # Extract from folder path
                    for part in path.parts[-4:-1]:  # Last 3 folders
                        part_words = re.findall(r"[a-z]+", part.lower())
                        for word in part_words:
                            if len(word) > 2 and word not in {"src", "app", "lib"}:
                                all_words[word] += 2

                    # Extract from file content (class/function names)
                    try:
                        if path.suffix in {".py", ".ts", ".tsx", ".js", ".jsx"}:
                            content = path.read_text(errors='ignore')[:5000]
                            # Python class/function names
                            if path.suffix == ".py":
                                classes = re.findall(r"class\s+([A-Z][a-zA-Z0-9]+)", content)
                                functions = re.findall(r"def\s+([a-z_][a-z0-9_]+)", content)
                            else:
                                # JS/TS
                                classes = re.findall(r"class\s+([A-Z][a-zA-Z0-9]+)", content)
                                functions = re.findall(r"function\s+([a-z_][a-zA-Z0-9_]+)", content)
                                functions += re.findall(r"const\s+([a-z][a-zA-Z0-9]+)\s*=", content)

                            for cls in classes:
                                cls_words = re.findall(r"[A-Z][a-z]+", cls)
                                for word in cls_words:
                                    if len(word) > 2:
                                        all_words[word.lower()] += 2

                            for func in functions:
                                func_words = re.findall(r"[a-z]+", func.lower())
                                for word in func_words:
                                    if len(word) > 3 and word not in {"self", "init", "none", "true", "false"}:
                                        all_words[word] += 1
                    except Exception:
                        pass  # Skip files that can't be read

            # Return top keywords, filtering out very common words
            common_words = {"get", "set", "the", "and", "for", "from", "with", "this", "that", "new"}
            keywords = [
                word for word, _ in all_words.most_common(max_keywords * 2)
                if word not in common_words
            ][:max_keywords]

            return keywords

        # Determine which agents to refresh
        agents_to_refresh = []
        for spec in self.list_agents():
            # Only refresh Tier B agents (auto-generated code agents)
            if not spec.metadata.get("auto_generated"):
                continue
            if agent_id and spec.agent_id != agent_id:
                continue
            agents_to_refresh.append(spec)

        # Refresh keywords for each agent
        for spec in agents_to_refresh:
            if spec.context_paths:
                new_keywords = extract_keywords_from_paths(spec.context_paths)
                if new_keywords:
                    # Update the spec's keywords
                    spec.context_keywords = new_keywords
                    updated[spec.agent_id] = new_keywords

        return updated

    def get_token_usage(self) -> dict[str, int]:
        """Get token usage across all sessions.

        Returns:
            Dict mapping session_id to total tokens used.
        """
        return {
            session.session_id: session.total_tokens_used
            for session in self._sessions.values()
        }

    # ==================== File Watching & Caching ====================

    def enable_git_cache(self, project_root: str) -> "GitAwareCache":
        """Enable git-aware caching for all agents.

        This makes agent contexts automatically refresh when:
        - Git commit changes (pull, checkout, commit)
        - Watched files are modified

        Args:
            project_root: Path to the git repository.

        Returns:
            The GitAwareCache instance.

        Example:
            >>> hub.enable_git_cache("/path/to/repo")
            >>> # Contexts now auto-refresh on git changes
        """
        from agenthub.cache import GitAwareCache

        self._git_cache = GitAwareCache(project_root)

        # Apply to all existing agents
        for agent in self._agents.values():
            agent.set_cache(self._git_cache)

        return self._git_cache

    def enable_file_watching(
        self,
        project_root: str,
        config: Optional["WatchConfig"] = None,
        on_refresh: Optional[Callable[[str, list[str]], None]] = None,
    ) -> "FileWatcher":
        """Enable automatic context refresh on file changes.

        This watches the file system and refreshes agent contexts
        when relevant files are modified.

        Args:
            project_root: Directory to watch.
            config: Optional watch configuration.
            on_refresh: Optional callback when contexts are refreshed.
                        Called with (event_type, list_of_affected_agent_ids).

        Returns:
            The FileWatcher instance.

        Example:
            >>> def on_update(event, agents):
            ...     print(f"Refreshed: {agents}")
            >>>
            >>> watcher = hub.enable_file_watching(
            ...     "/path/to/project",
            ...     on_refresh=on_update,
            ... )
            >>> # Files are now being watched
            >>> # Edit a .py file and see agents refresh

        Note:
            Requires watchdog package: pip install watchdog
        """
        from agenthub.cache import FileWatcher

        self._on_context_refresh = on_refresh

        self._file_watcher = FileWatcher(
            hub=self,
            project_root=project_root,
            config=config,
        )
        self._file_watcher.on_refresh = self._handle_file_refresh
        self._file_watcher.start()

        return self._file_watcher

    def _handle_file_refresh(self, event_type: str, affected_agents: list[str]) -> None:
        """Internal handler for file watcher refresh events."""
        if self._on_context_refresh:
            self._on_context_refresh(event_type, affected_agents)

    def disable_file_watching(self) -> None:
        """Stop watching for file changes."""
        if self._file_watcher:
            self._file_watcher.stop()
            self._file_watcher = None

    def get_cache_stats(self) -> Optional[dict]:
        """Get statistics about the git-aware cache.

        Returns:
            Dict with cache stats, or None if git cache not enabled.
        """
        if self._git_cache:
            return self._git_cache.get_stats()
        return None

    @property
    def is_watching(self) -> bool:
        """Check if file watching is active."""
        return self._file_watcher is not None and self._file_watcher.is_running

    def get_agent_context_status(self, agent_id: str) -> Optional["AgentContextStatus"]:
        """Get the context awareness status for an agent.

        Args:
            agent_id: ID of the agent to check.

        Returns:
            AgentContextStatus with staleness information, or None if no watcher.
        """
        if not self._file_watcher:
            return None
        return self._file_watcher.stale_tracker.get_agent_status(agent_id)

    def get_all_agent_context_statuses(self) -> list["AgentContextStatus"]:
        """Get context awareness status for all agents.

        Returns:
            List of AgentContextStatus for all registered agents.
        """
        if not self._file_watcher:
            # Return default statuses for all agents
            from agenthub.models import AgentContextStatus

            return [
                AgentContextStatus(
                    agent_id=spec.agent_id,
                    is_stale=False,
                    changed_files=[],
                    last_query_time=None,
                    last_change_time=None,
                    status="never_queried",
                )
                for spec in self.list_agents()
            ]
        return self._file_watcher.stale_tracker.get_all_statuses()

    def mark_agent_queried(self, agent_id: str) -> None:
        """Mark that an agent was queried (for tracking freshness).

        Args:
            agent_id: ID of the agent that was queried.
        """
        if self._file_watcher:
            self._file_watcher.mark_agent_queried(agent_id)

    # ==================== QC Analysis (Tier C) ====================

    def enable_qc_analysis(
        self,
        auto_analyze: bool = True,
        on_concern: Optional[Callable[["Concern"], None]] = None,
        on_report: Optional[Callable[["ConcernReport"], None]] = None,
    ) -> "ChangeAnalysisPipeline":
        """Enable QC analysis with Tier B concerns and Tier C synthesis.

        This sets up:
        - QC Agent (Tier C) for synthesizing concerns
        - Change Analysis Pipeline for orchestrating Tier B analysis
        - Optional auto-trigger on file changes

        Args:
            auto_analyze: If True and file watching is enabled, automatically
                         analyze changes when files are modified.
            on_concern: Callback when a Tier B agent raises a concern.
            on_report: Callback when QC report is complete.

        Returns:
            ChangeAnalysisPipeline instance for manual triggering.

        Example:
            >>> hub.enable_file_watching("./project")
            >>> pipeline = hub.enable_qc_analysis(
            ...     auto_analyze=True,
            ...     on_report=lambda r: print(f"QC: {r.recommendation}")
            ... )
            >>>
            >>> # Changes are now automatically analyzed
            >>> # Or trigger manually:
            >>> report = hub.analyze_changes(["src/api/routes.py"])
        """
        from agenthub.qc.pipeline import ChangeAnalysisPipeline
        from agenthub.qc.qc_agent import QCAgent

        # Store callbacks
        self._on_concern_raised = on_concern
        self._on_qc_report = on_report

        # Create and register QC Agent
        self._qc_agent = QCAgent(self.client)
        try:
            self.register(self._qc_agent)
        except ValueError:
            pass  # Already registered

        # Create pipeline
        self._analysis_pipeline = ChangeAnalysisPipeline(
            hub=self,
            qc_agent=self._qc_agent,
            on_concern_raised=on_concern,
            on_report_complete=on_report,
        )

        # Set up auto-analysis if requested and file watching is enabled
        if auto_analyze and self._file_watcher:
            # Store the original callback
            original_callback = self._on_context_refresh

            def combined_callback(event_type: str, affected_agents: list[str]) -> None:
                # Call original callback
                if original_callback:
                    original_callback(event_type, affected_agents)

                # Trigger QC analysis for changed files
                # Note: We need to get the changed files from the file watcher
                # For now, this will be handled separately

            self._on_context_refresh = combined_callback

        return self._analysis_pipeline

    def analyze_changes(
        self,
        files: list[str],
        source: str = "manual",
    ) -> "ConcernReport":
        """Manually trigger change analysis on specific files.

        Args:
            files: List of file paths to analyze.
            source: Source of the analysis request ("manual", "file_watcher", "git").

        Returns:
            ConcernReport with concerns and recommendations.

        Raises:
            ValueError: If QC analysis not enabled.

        Example:
            >>> report = hub.analyze_changes([
            ...     "src/api/routes.py",
            ...     "src/services/auth.py"
            ... ])
            >>> print(f"Risk: {report.risk_level}")
            >>> print(f"Recommendation: {report.recommendation}")
            >>> for concern in report.all_concerns:
            ...     print(f"  - [{concern.severity}] {concern.title}")
        """
        if not self._analysis_pipeline:
            raise ValueError("QC analysis not enabled. Call enable_qc_analysis() first.")

        from agenthub.qc.models import ChangeSet, FileChange

        # Build change set
        change_set = ChangeSet(
            change_id=str(uuid.uuid4())[:8],
            files=[FileChange(path=f, change_type="modified") for f in files],
            source=source,
        )

        return self._analysis_pipeline.analyze(change_set)

    def get_qc_reports(self, limit: int = 20) -> list["ConcernReport"]:
        """Get recent QC reports.

        Args:
            limit: Maximum number of reports to return.

        Returns:
            List of recent ConcernReports, newest first.
        """
        if not self._analysis_pipeline:
            return []
        return self._analysis_pipeline.get_history(limit)

    def get_qc_report(self, report_id: str) -> Optional["ConcernReport"]:
        """Get a specific QC report by ID.

        Args:
            report_id: Report ID to find.

        Returns:
            ConcernReport if found, None otherwise.
        """
        if not self._analysis_pipeline:
            return None
        return self._analysis_pipeline.get_report(report_id)

    @property
    def is_qc_enabled(self) -> bool:
        """Check if QC analysis is enabled."""
        return self._analysis_pipeline is not None

    # ==================== Cross-Agent Context Sharing ====================

    def enable_cross_context(
        self,
        config: Optional["CrossContextConfig"] = None,
    ) -> "CrossAgentContextManager":
        """Enable cross-agent context injection.

        When enabled, queries will automatically receive relevant context
        from other agents whose code is imported by the queried agent's domain.

        This helps agents provide more accurate responses when dealing with
        code that depends on modules from other agents' domains.

        Args:
            config: Optional configuration for context injection.
                    Defaults to CrossContextConfig() with sensible defaults.

        Returns:
            CrossAgentContextManager instance for manual control.

        Example:
            >>> hub.enable_auto_agents("./project")
            >>> hub.enable_cross_context()
            >>> # Now queries automatically get related context
            >>> response = hub.run("How does auth work?")

            # With custom config:
            >>> from agenthub.auto.cross_context import CrossContextConfig
            >>> hub.enable_cross_context(CrossContextConfig(
            ...     max_injected_chars=5000,
            ...     max_agents_to_inject=2,
            ... ))
        """
        from agenthub.auto.cross_context import CrossAgentContextManager, CrossContextConfig

        # Get import graph from auto manager if available
        import_graph = None
        if self._auto_manager:
            import_graph = getattr(self._auto_manager, "_import_graph", None)

        self._cross_context_manager = CrossAgentContextManager(
            hub=self,
            import_graph=import_graph,
            config=config or CrossContextConfig(),
        )
        self._cross_context_enabled = True

        return self._cross_context_manager

    def disable_cross_context(self) -> None:
        """Disable cross-agent context injection."""
        self._cross_context_enabled = False

    @property
    def is_cross_context_enabled(self) -> bool:
        """Check if cross-agent context sharing is enabled."""
        return self._cross_context_enabled

    # ==================== DAG Team Execution ====================

    def enable_teams(
        self,
        import_graph: Optional["ImportGraph"] = None,
        max_parallel: int = 4,
        complexity_threshold: float = 0.4,
    ) -> "DAGTeamExecutor":
        """Enable DAG team execution for complex queries.

        When enabled, complex queries that span multiple agent domains
        will be handled by a team of agents working together in
        dependency order derived from the import graph.

        Args:
            import_graph: ImportGraph instance. If None, will be obtained
                         from the auto-agent manager's project analysis.
            max_parallel: Max agents to run in parallel per layer.
            complexity_threshold: Score threshold for triggering teams.
                                 Lower values = more queries go to DAG teams.

        Returns:
            DAGTeamExecutor instance for direct use if needed.

        Raises:
            ValueError: If no import graph available (auto-agents not enabled
                       and no graph provided).

        Example:
            >>> hub = AgentHub()
            >>> hub.enable_auto_agents("./my-project")
            >>> hub.enable_teams()
            >>>
            >>> # Simple queries go to single agent
            >>> hub.run("What does parse_config do?")
            >>>
            >>> # Complex queries use team execution
            >>> hub.run("How does data flow from API to database?")
            >>>
            >>> # Force team mode for testing
            >>> hub.run("Explain auth", team_mode="always")
        """
        from agenthub.teams.executor import DAGTeamExecutor

        # Try to get import graph from multiple sources if not provided
        if import_graph is None:
            # First try the hub's own _import_graph (set by discover_all_agents)
            if self._import_graph is not None:
                import_graph = self._import_graph
            # Then try auto manager
            elif self._auto_manager:
                import_graph = getattr(self._auto_manager, "_import_graph", None)

        if import_graph is None:
            raise ValueError(
                "No import graph available. Either provide one or "
                "enable auto-agents first with enable_auto_agents() or discover_all_agents()."
            )

        self._team_executor = DAGTeamExecutor(
            hub=self,
            import_graph=import_graph,
            max_parallel=max_parallel,
        )
        self._complexity_threshold = complexity_threshold

        return self._team_executor

    def disable_teams(self) -> None:
        """Disable DAG team execution."""
        self._team_executor = None

    @property
    def is_teams_enabled(self) -> bool:
        """Check if DAG team execution is enabled."""
        return self._team_executor is not None

    def _store_team_trace(self, session_id: str, query: str, trace: dict) -> None:
        """Store a team execution trace for dashboard visualization.

        Traces are stored both in-memory and persisted to disk so they
        can be shared between MCP server and dashboard instances.

        Args:
            session_id: Session ID for the trace.
            query: Original query that triggered team execution.
            trace: TeamExecutionTrace as dict.
        """
        import json

        # Store with timestamp
        trace_entry = {
            "session_id": session_id,
            "query": query,
            "trace": trace,
            "timestamp": datetime.now().isoformat(),
        }
        self._team_traces[session_id] = trace_entry

        # Prune old traces if limit exceeded
        if len(self._team_traces) > self._max_team_traces:
            sorted_keys = sorted(
                self._team_traces.keys(),
                key=lambda k: self._team_traces[k].get("timestamp", ""),
            )
            for key in sorted_keys[: len(sorted_keys) - self._max_team_traces]:
                del self._team_traces[key]

        # Persist to file for cross-process sharing (MCP <-> Dashboard)
        self._persist_traces_to_file()

    def _persist_traces_to_file(self) -> None:
        """Persist traces to disk for sharing between processes."""
        import json

        traces_file = Path.home() / ".agenthub" / "team_traces.json"
        try:
            traces_file.parent.mkdir(exist_ok=True)
            traces_file.write_text(json.dumps(list(self._team_traces.values()), indent=2))
        except Exception:
            pass  # Non-critical if persistence fails

    def _load_traces_from_file(self) -> None:
        """Load persisted traces from disk."""
        import json

        traces_file = Path.home() / ".agenthub" / "team_traces.json"
        try:
            if traces_file.exists():
                data = json.loads(traces_file.read_text())
                for trace in data:
                    if trace.get("session_id"):
                        self._team_traces[trace["session_id"]] = trace
        except Exception:
            pass  # Non-critical if loading fails

    def get_team_traces(self, limit: int = 20) -> list[dict]:
        """Get recent team execution traces.

        Args:
            limit: Maximum number of traces to return.

        Returns:
            List of traces sorted by timestamp (newest first).
        """
        # Load from file first to get traces from other processes (e.g., MCP server)
        self._load_traces_from_file()

        traces = list(self._team_traces.values())
        traces.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
        return traces[:limit]

    def get_team_trace(self, session_id: str) -> Optional[dict]:
        """Get a specific team execution trace.

        Args:
            session_id: Session ID of the trace.

        Returns:
            Trace dict or None if not found.
        """
        # Load from file first
        self._load_traces_from_file()
        return self._team_traces.get(session_id)

    # ==================== Parallel Sessions ====================

    def enable_parallel_sessions(
        self,
        project_root: str,
        config: Optional["ParallelExecutionConfig"] = None,
        on_escalation: Optional[Callable[["Escalation"], None]] = None,
    ) -> "ParallelSessionManager":
        """Enable parallel session execution for multi-part requests.

        When enabled, multi-part requests can be decomposed into tasks
        and executed in parallel using separate git branches.

        Args:
            project_root: Path to the git repository.
            config: Optional configuration for parallel execution.
            on_escalation: Optional callback for handling escalations.
                          If None, escalations will use default behavior.

        Returns:
            ParallelSessionManager instance.

        Example:
            >>> hub.enable_auto_agents("./project")
            >>> hub.enable_parallel_sessions("./project")
            >>>
            >>> # Execute multi-part request
            >>> result = hub.execute_parallel(
            ...     "Add a save button and build a chart component"
            ... )
            >>> print(f"Speedup: {result.speedup:.1f}x")

        Note:
            - Requires auto-agents to be enabled first
            - Project must be a git repository with clean working tree
        """
        from agenthub.parallel import ParallelExecutionConfig, ParallelSessionManager

        # Get import graph if available
        import_graph = self._import_graph
        if import_graph is None and self._auto_manager:
            import_graph = getattr(self._auto_manager, "_import_graph", None)

        self._on_parallel_escalation = on_escalation

        self._parallel_manager = ParallelSessionManager(
            hub=self,
            project_root=project_root,
            config=config or ParallelExecutionConfig(),
            import_graph=import_graph,
            on_escalation=self._handle_parallel_escalation if on_escalation else None,
        )

        return self._parallel_manager

    def _handle_parallel_escalation(self, escalation: "Escalation"):
        """Handle escalation from parallel session manager.

        Args:
            escalation: The escalation to handle.

        Returns:
            EscalationResult with user decision.
        """
        from agenthub.parallel import EscalationResult

        # Call user callback if provided
        if self._on_parallel_escalation:
            self._on_parallel_escalation(escalation)

        # Default: use recommended option
        recommended = next(
            (o for o in escalation.options if o.get("recommended")),
            escalation.options[0] if escalation.options else {"id": "abort"},
        )

        return EscalationResult(
            escalation_id=escalation.escalation_id,
            chosen_option=recommended["id"],
        )

    def execute_parallel(
        self,
        request: str,
        base_branch: str = "main",
        precomputed: Optional[tuple] = None,
    ) -> "ParallelExecutionResult":
        """Execute a multi-part request using parallel sessions.

        This is a convenience method that uses the ParallelSessionManager.
        For more control, use the manager directly via enable_parallel_sessions().

        Args:
            request: Multi-part request to execute.
            base_branch: Git branch to work from.
            precomputed: Optional (DecompositionResult, ParallelizationPlan) from
                        a prior preview_parallel() call.  Avoids re-running the
                        expensive decomposition + analysis phases.

        Returns:
            ParallelExecutionResult with full execution details.

        Raises:
            ValueError: If parallel sessions not enabled.

        Example:
            >>> hub.enable_auto_agents("./project")
            >>> hub.enable_parallel_sessions("./project")
            >>>
            >>> result = hub.execute_parallel(
            ...     "Add a save button and implement chart component"
            ... )
            >>> if result.success:
            ...     print(f"Completed {len(result.tasks)} tasks")
            ...     print(f"Speedup: {result.speedup:.1f}x")
        """
        if not self._parallel_manager:
            raise ValueError(
                "Parallel sessions not enabled. Call enable_parallel_sessions() first."
            )

        return self._parallel_manager.execute(request, base_branch, precomputed=precomputed)

    def preview_parallel(
        self,
        request: str,
    ) -> tuple:
        """Preview how a request would be decomposed and analyzed.

        Useful for understanding what parallel execution would do
        without actually running sessions.

        Args:
            request: The request to preview.

        Returns:
            Tuple of (DecompositionResult, ParallelizationPlan).

        Raises:
            ValueError: If parallel sessions not enabled.

        Example:
            >>> decomp, plan = hub.preview_parallel(
            ...     "Add save button and chart component"
            ... )
            >>> print(f"Would create {len(decomp.tasks)} tasks")
            >>> print(f"Risk level: {plan.overall_risk.value}")
        """
        if not self._parallel_manager:
            raise ValueError(
                "Parallel sessions not enabled. Call enable_parallel_sessions() first."
            )

        return self._parallel_manager.analyze(request)

    def disable_parallel_sessions(self) -> None:
        """Disable parallel session execution."""
        self._parallel_manager = None

    @property
    def is_parallel_enabled(self) -> bool:
        """Check if parallel sessions are enabled."""
        return self._parallel_manager is not None

    def __repr__(self) -> str:
        watching = " watching" if self.is_watching else ""
        qc = " qc" if self.is_qc_enabled else ""
        cross = " cross-context" if self.is_cross_context_enabled else ""
        teams = " teams" if self.is_teams_enabled else ""
        parallel = " parallel" if self.is_parallel_enabled else ""
        return f"<AgentHub agents={len(self._agents)} sessions={len(self._sessions)}{watching}{qc}{cross}{teams}{parallel}>"


# Type hint for auto-agent config (avoid circular import)
if TYPE_CHECKING:
    from agenthub.auto.config import AutoAgentConfig
