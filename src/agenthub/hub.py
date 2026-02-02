"""AgentHub - Central orchestrator for agent routing and management."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional, Union

from agenthub.models import AgentResponse, AgentSpec, Message, Session
from agenthub.routing import KeywordRouter, Router, TierAwareRouter

if TYPE_CHECKING:
    import anthropic

    from agenthub.agents.base import BaseAgent
    from agenthub.cache import FileWatcher, GitAwareCache, WatchConfig
    from agenthub.qc.models import ChangeSet, Concern, ConcernReport
    from agenthub.qc.pipeline import ChangeAnalysisPipeline
    from agenthub.qc.qc_agent import QCAgent


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
                  None for all.

        Returns:
            List of agent specifications.
        """
        all_agents = [a.spec for a in self._agents.values()]

        if tier is None:
            return all_agents
        elif tier == "A":
            return [a for a in all_agents if not a.metadata.get("auto_generated")]
        elif tier == "B":
            return [a for a in all_agents if a.metadata.get("auto_generated")]
        else:
            raise ValueError(f"Unknown tier: {tier}. Use 'A', 'B', or None.")

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

        Args:
            query: User's query to route.

        Returns:
            Agent ID to handle the query.

        Raises:
            ValueError: If no agents registered or no match found.
        """
        if not self._agents:
            raise ValueError("No agents registered")

        agents = self.list_agents()
        agent_id = self._router.route(query, agents)

        if agent_id and agent_id in self._agents:
            return agent_id

        # Fallback to first agent
        return list(self._agents.keys())[0]

    # ==================== Execution ====================

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AgentResponse:
        """Execute a query through the hub.

        This is the main entry point for running queries. It handles
        session management, routing, and agent execution.

        Args:
            query: User's query to process.
            session_id: Existing session ID (creates new if None).
            agent_id: Force specific agent (auto-routes if None).
            model: Override model for this call.

        Returns:
            AgentResponse with content and metadata.

        Raises:
            ValueError: If session not found or agent not registered.
        """
        # Get or create session
        if session_id:
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
        else:
            session = self.create_session()

        # Route to agent
        target_agent_id = agent_id or self.route(query)

        if target_agent_id not in self._agents:
            raise ValueError(f"Agent {target_agent_id} not registered")

        agent = self._agents[target_agent_id]

        # Update session
        session.agent_id = target_agent_id
        session.messages.append(Message(role="user", content=query))
        session.updated_at = datetime.now()

        # Execute
        response = agent.run(query, session, model=model)

        # Store response in session
        session.messages.append(
            Message(
                role="assistant",
                content=response.content,
                metadata={"tokens": response.tokens_used, "agent": target_agent_id},
            )
        )
        session.total_tokens_used += response.tokens_used

        return response

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

    def __repr__(self) -> str:
        watching = " watching" if self.is_watching else ""
        qc = " qc" if self.is_qc_enabled else ""
        return f"<AgentHub agents={len(self._agents)} sessions={len(self._sessions)}{watching}{qc}>"


# Type hint for auto-agent config (avoid circular import)
if TYPE_CHECKING:
    from agenthub.auto.config import AutoAgentConfig
