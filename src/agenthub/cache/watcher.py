from __future__ import annotations
"""File system watcher for automatic context refresh and QC analysis.

This module monitors file changes and triggers:
1. Agent context refresh when relevant files are modified
2. QC analysis pipeline when changes are detected (if enabled)
3. Per-agent stale status tracking for dashboard awareness indicators

Requires: pip install watchdog

Example:
    >>> from agenthub.cache import FileWatcher
    >>>
    >>> watcher = FileWatcher(hub, project_root="/path/to/project")
    >>> watcher.start()
    >>>
    >>> # Files are now being watched
    >>> # When src/api/routes.py changes, the api_agent's context refreshes
    >>> # If QC analysis is enabled, it also triggers concern analysis
    >>>
    >>> watcher.stop()
"""

import fnmatch
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from agenthub.hub import AgentHub
    from agenthub.models import AgentContextStatus

# Try to import watchdog
_watchdog_available = False
try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    _watchdog_available = True
except ImportError:
    # Create stub classes for type checking
    class FileSystemEventHandler:  # type: ignore
        pass

    class FileSystemEvent:  # type: ignore
        pass

    class Observer:  # type: ignore
        pass


@dataclass
class AgentStaleInfo:
    """Tracks staleness info for a single agent."""

    agent_id: str
    last_query_time: Optional[datetime] = None
    last_change_time: Optional[datetime] = None
    changed_files: list[str] = field(default_factory=list)

    @property
    def is_stale(self) -> bool:
        """Check if agent has unprocessed changes."""
        if self.last_change_time is None:
            return False
        if self.last_query_time is None:
            return True  # Never queried but has changes
        return self.last_change_time > self.last_query_time

    @property
    def status(self) -> str:
        """Get human-readable status."""
        if self.last_query_time is None:
            return "never_queried"
        if self.is_stale:
            return "stale"
        return "fresh"


@dataclass
class WatchConfig:
    """Configuration for file watching."""

    # Patterns to watch (glob patterns)
    include_patterns: list[str] = field(
        default_factory=lambda: ["**/*.py", "**/*.json", "**/*.yaml", "**/*.yml"]
    )

    # Patterns to ignore
    ignore_patterns: list[str] = field(
        default_factory=lambda: [
            "**/__pycache__/**",
            "**/.git/**",
            "**/.venv/**",
            "**/venv/**",
            "**/node_modules/**",
            "**/*.pyc",
            "**/.pytest_cache/**",
        ]
    )

    # Debounce time in seconds (ignore rapid successive changes)
    debounce_seconds: float = 1.0

    # Debounce time for QC analysis (longer to batch multiple changes)
    analysis_debounce_seconds: float = 3.0

    # Whether to refresh all agents or just affected ones
    refresh_all: bool = False

    # Whether to refresh agent keywords on file changes (for routing accuracy)
    refresh_keywords: bool = True

    # Debounce time for keyword refresh (longer since it's more expensive)
    keyword_refresh_debounce_seconds: float = 10.0


class AgentContextHandler(FileSystemEventHandler):
    """Handles file system events and refreshes agent contexts."""

    def __init__(
        self,
        hub: "AgentHub",
        project_root: str,
        config: WatchConfig,
        on_refresh: Optional[Callable[[str, list[str]], None]] = None,
        on_changes_for_analysis: Optional[Callable[[list[str]], None]] = None,
    ):
        """Initialize the handler.

        Args:
            hub: AgentHub instance to manage.
            project_root: Root directory being watched.
            config: Watch configuration.
            on_refresh: Optional callback when refresh happens.
                        Called with (event_type, affected_agent_ids).
            on_changes_for_analysis: Optional callback for QC analysis.
                        Called with list of changed file paths.
        """
        super().__init__()
        self.hub = hub
        self.project_root = Path(project_root).resolve()
        self.config = config
        self.on_refresh = on_refresh
        self.on_changes_for_analysis = on_changes_for_analysis

        # Debounce tracking for context refresh
        self._last_event_time: float = 0
        self._pending_paths: set[str] = set()
        self._debounce_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

        # Debounce tracking for keyword refresh (separate, longer debounce)
        self._keyword_refresh_timer: Optional[threading.Timer] = None
        self._pending_keyword_refresh_paths: set[str] = set()

        # Debounce tracking for QC analysis (separate, longer debounce)
        self._analysis_pending_paths: set[str] = set()
        self._analysis_debounce_timer: Optional[threading.Timer] = None
        self._analysis_lock = threading.Lock()

    def _matches_pattern(self, path: str, patterns: list[str]) -> bool:
        """Check if path matches any of the patterns."""
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    def _should_process(self, path: str) -> bool:
        """Check if this path should trigger a refresh."""
        # Get relative path
        try:
            rel_path = Path(path).relative_to(self.project_root)
            rel_str = str(rel_path).replace("\\", "/")
        except ValueError:
            return False

        # Check ignore patterns
        if self._matches_pattern(rel_str, self.config.ignore_patterns):
            return False

        # Check include patterns
        if self._matches_pattern(rel_str, self.config.include_patterns):
            return True

        return False

    def _find_affected_agents(self, changed_paths: set[str]) -> list[str]:
        """Find agents whose context includes the changed files."""
        affected = []

        for agent_spec in self.hub.list_agents():
            agent = self.hub.get_agent(agent_spec.agent_id)
            if not agent:
                continue

            # Check if any changed path matches agent's context_paths
            for context_path in agent_spec.context_paths:
                for changed in changed_paths:
                    try:
                        changed_rel = str(Path(changed).relative_to(self.project_root))
                        if fnmatch.fnmatch(changed_rel, context_path):
                            affected.append(agent_spec.agent_id)
                            break
                    except ValueError:
                        pass

        return list(set(affected))

    def _do_refresh(self) -> None:
        """Perform the actual refresh after debounce."""
        with self._lock:
            paths = self._pending_paths.copy()
            self._pending_paths.clear()
            self._debounce_timer = None

        if not paths:
            return

        if self.config.refresh_all:
            # Refresh all agents
            self.hub.refresh_all_contexts()
            affected = [a.agent_id for a in self.hub.list_agents()]
        else:
            # Refresh only affected agents
            affected = self._find_affected_agents(paths)
            for agent_id in affected:
                agent = self.hub.get_agent(agent_id)
                if agent:
                    agent.clear_context_cache()

        # Notify callback
        if self.on_refresh and affected:
            self.on_refresh("file_change", affected)

        # Schedule keyword refresh for affected agents (with longer debounce)
        if self.config.refresh_keywords and affected:
            self._schedule_keyword_refresh(affected)

    def _do_keyword_refresh(self) -> None:
        """Perform keyword refresh for agents after longer debounce."""
        with self._lock:
            agents_to_refresh = self._pending_keyword_refresh_paths.copy()
            self._pending_keyword_refresh_paths.clear()
            self._keyword_refresh_timer = None

        if not agents_to_refresh:
            return

        # Refresh keywords for each affected agent
        try:
            for agent_id in agents_to_refresh:
                self.hub.refresh_agent_keywords(agent_id)

            if self.on_refresh:
                self.on_refresh("keyword_refresh", list(agents_to_refresh))
        except Exception:
            pass  # Non-critical if keyword refresh fails

    def _schedule_keyword_refresh(self, agent_ids: list[str]) -> None:
        """Schedule a debounced keyword refresh for agents."""
        with self._lock:
            self._pending_keyword_refresh_paths.update(agent_ids)

            # Cancel existing timer
            if self._keyword_refresh_timer:
                self._keyword_refresh_timer.cancel()

            # Schedule new timer with longer debounce
            self._keyword_refresh_timer = threading.Timer(
                self.config.keyword_refresh_debounce_seconds, self._do_keyword_refresh
            )
            self._keyword_refresh_timer.start()

    def _schedule_refresh(self, path: str) -> None:
        """Schedule a debounced refresh."""
        with self._lock:
            self._pending_paths.add(path)

            # Cancel existing timer
            if self._debounce_timer:
                self._debounce_timer.cancel()

            # Schedule new timer
            self._debounce_timer = threading.Timer(
                self.config.debounce_seconds, self._do_refresh
            )
            self._debounce_timer.start()

    def _schedule_analysis(self, path: str) -> None:
        """Schedule a debounced QC analysis.

        Uses a longer debounce than refresh to batch multiple file changes.
        """
        if not self.on_changes_for_analysis:
            return

        with self._analysis_lock:
            self._analysis_pending_paths.add(path)

            # Cancel existing timer
            if self._analysis_debounce_timer:
                self._analysis_debounce_timer.cancel()

            # Schedule new timer with longer debounce
            self._analysis_debounce_timer = threading.Timer(
                self.config.analysis_debounce_seconds, self._do_analysis
            )
            self._analysis_debounce_timer.start()

    def _do_analysis(self) -> None:
        """Perform the actual QC analysis after debounce."""
        with self._analysis_lock:
            paths = list(self._analysis_pending_paths)
            self._analysis_pending_paths.clear()
            self._analysis_debounce_timer = None

        if not paths or not self.on_changes_for_analysis:
            return

        # Convert to relative paths
        relative_paths = []
        for path in paths:
            try:
                rel_path = str(Path(path).relative_to(self.project_root))
                relative_paths.append(rel_path)
            except ValueError:
                relative_paths.append(path)

        # Trigger analysis callback
        try:
            self.on_changes_for_analysis(relative_paths)
        except Exception:
            pass  # Don't let analysis errors break the watcher

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification."""
        if event.is_directory:
            return

        if self._should_process(event.src_path):
            self._schedule_refresh(event.src_path)
            self._schedule_analysis(event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation."""
        if event.is_directory:
            return

        if self._should_process(event.src_path):
            self._schedule_refresh(event.src_path)
            self._schedule_analysis(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion."""
        if event.is_directory:
            return

        if self._should_process(event.src_path):
            self._schedule_refresh(event.src_path)
            self._schedule_analysis(event.src_path)


class AgentStaleTracker:
    """Tracks staleness status for all agents.

    This class monitors when agents are queried and when their domain
    files change, allowing the dashboard to display awareness indicators.

    Example:
        >>> tracker = AgentStaleTracker(hub, project_root)
        >>> tracker.mark_agent_queried("api_agent")
        >>> tracker.mark_files_changed(["src/api/routes.py"])
        >>> status = tracker.get_agent_status("api_agent")
        >>> print(status.is_stale)  # True if files changed after query
    """

    def __init__(self, hub: "AgentHub", project_root: str):
        """Initialize the stale tracker.

        Args:
            hub: AgentHub instance.
            project_root: Root directory of the project.
        """
        self.hub = hub
        self.project_root = Path(project_root).resolve()
        self._agent_info: dict[str, AgentStaleInfo] = {}
        self._lock = threading.Lock()

        # Callback for stale events (agent_id, is_stale)
        self.on_stale_change: Optional[Callable[[str, bool], None]] = None

    def _get_or_create_info(self, agent_id: str) -> AgentStaleInfo:
        """Get or create stale info for an agent."""
        if agent_id not in self._agent_info:
            self._agent_info[agent_id] = AgentStaleInfo(agent_id=agent_id)
        return self._agent_info[agent_id]

    def mark_agent_queried(self, agent_id: str) -> None:
        """Mark that an agent was just queried.

        Args:
            agent_id: ID of the agent that was queried.
        """
        with self._lock:
            info = self._get_or_create_info(agent_id)
            was_stale = info.is_stale
            info.last_query_time = datetime.now()
            info.changed_files = []  # Clear pending changes

            # Notify if status changed
            if was_stale and self.on_stale_change:
                self.on_stale_change(agent_id, False)

    def mark_files_changed(self, file_paths: list[str]) -> list[str]:
        """Mark that files have changed and update affected agents.

        Args:
            file_paths: List of relative file paths that changed.

        Returns:
            List of agent IDs that became stale.
        """
        affected_agents = self._find_affected_agents(file_paths)
        newly_stale = []

        with self._lock:
            for agent_id in affected_agents:
                info = self._get_or_create_info(agent_id)
                was_stale = info.is_stale
                info.last_change_time = datetime.now()

                # Add changed files that are relevant to this agent
                agent_spec = self.hub.get_agent(agent_id)
                if agent_spec:
                    for path in file_paths:
                        for context_path in agent_spec.spec.context_paths:
                            if fnmatch.fnmatch(path, context_path):
                                if path not in info.changed_files:
                                    info.changed_files.append(path)
                                break

                # Track newly stale agents
                if not was_stale and info.is_stale:
                    newly_stale.append(agent_id)
                    if self.on_stale_change:
                        self.on_stale_change(agent_id, True)

        return newly_stale

    def _find_affected_agents(self, file_paths: list[str]) -> list[str]:
        """Find which agents are affected by the given file changes."""
        affected = set()

        for agent_spec in self.hub.list_agents():
            for context_path in agent_spec.context_paths:
                for path in file_paths:
                    if fnmatch.fnmatch(path, context_path):
                        affected.add(agent_spec.agent_id)
                        break

        return list(affected)

    def get_agent_status(self, agent_id: str) -> "AgentContextStatus":
        """Get the current context status for an agent.

        Args:
            agent_id: ID of the agent.

        Returns:
            AgentContextStatus with staleness information.
        """
        from agenthub.models import AgentContextStatus

        with self._lock:
            info = self._get_or_create_info(agent_id)
            return AgentContextStatus(
                agent_id=agent_id,
                is_stale=info.is_stale,
                changed_files=info.changed_files.copy(),
                last_query_time=info.last_query_time,
                last_change_time=info.last_change_time,
                status=info.status,
            )

    def get_all_statuses(self) -> list["AgentContextStatus"]:
        """Get context status for all registered agents.

        Returns:
            List of AgentContextStatus for all agents.
        """
        statuses = []
        for agent_spec in self.hub.list_agents():
            statuses.append(self.get_agent_status(agent_spec.agent_id))
        return statuses

    def get_stale_agents(self) -> list[str]:
        """Get list of agents with pending changes.

        Returns:
            List of agent IDs that have stale contexts.
        """
        stale = []
        with self._lock:
            for agent_id, info in self._agent_info.items():
                if info.is_stale:
                    stale.append(agent_id)
        return stale

    def reset_agent(self, agent_id: str) -> None:
        """Reset tracking info for an agent.

        Args:
            agent_id: ID of the agent to reset.
        """
        with self._lock:
            if agent_id in self._agent_info:
                del self._agent_info[agent_id]


class FileWatcher:
    """Watches files and refreshes agent contexts on changes.

    This class provides automatic context refresh when monitored files
    change, ensuring agents always have up-to-date knowledge.

    Additionally, when QC analysis is enabled, it can trigger the
    analysis pipeline to detect potential issues in changed code.

    Also provides per-agent stale tracking via the stale_tracker property.

    Example:
        >>> watcher = FileWatcher(hub, "/path/to/project")
        >>>
        >>> # Optional: Add callback for refresh events
        >>> watcher.on_refresh = lambda event, agents: print(f"Refreshed: {agents}")
        >>>
        >>> # Optional: Add callback for QC analysis
        >>> watcher.on_changes_for_analysis = lambda files: hub.analyze_changes(files)
        >>>
        >>> # Optional: Add callback for stale events
        >>> watcher.on_stale_change = lambda agent_id, is_stale: print(f"{agent_id}: stale={is_stale}")
        >>>
        >>> watcher.start()
        >>> # ... application runs ...
        >>> watcher.stop()

    Note:
        Requires watchdog package: pip install watchdog
    """

    def __init__(
        self,
        hub: "AgentHub",
        project_root: str,
        config: Optional[WatchConfig] = None,
    ):
        """Initialize the file watcher.

        Args:
            hub: AgentHub instance to manage.
            project_root: Directory to watch for changes.
            config: Optional watch configuration.

        Raises:
            ImportError: If watchdog is not installed.
        """
        if not _watchdog_available:
            raise ImportError(
                "watchdog package required for file watching. "
                "Install with: pip install watchdog"
            )

        self.hub = hub
        self.project_root = Path(project_root).resolve()
        self.config = config or WatchConfig()

        self._observer: Optional[Observer] = None
        self._handler: Optional[AgentContextHandler] = None
        self._running = False

        # Stale tracker for per-agent awareness
        self._stale_tracker = AgentStaleTracker(hub, str(self.project_root))

        # Callback for refresh events
        self.on_refresh: Optional[Callable[[str, list[str]], None]] = None

        # Callback for QC analysis (triggered with list of changed file paths)
        self.on_changes_for_analysis: Optional[Callable[[list[str]], None]] = None

        # Callback for stale events (agent_id, is_stale)
        self.on_stale_change: Optional[Callable[[str, bool], None]] = None

    @property
    def stale_tracker(self) -> "AgentStaleTracker":
        """Get the stale tracker for querying agent staleness."""
        return self._stale_tracker

    def start(self) -> None:
        """Start watching for file changes.

        This spawns a background thread that monitors the file system.
        """
        if self._running:
            return

        self._handler = AgentContextHandler(
            hub=self.hub,
            project_root=str(self.project_root),
            config=self.config,
            on_refresh=self._handle_refresh,
            on_changes_for_analysis=self._handle_changes_for_analysis,
        )

        self._observer = Observer()
        self._observer.schedule(
            self._handler, str(self.project_root), recursive=True
        )
        self._observer.start()
        self._running = True

    def stop(self) -> None:
        """Stop watching for file changes."""
        if not self._running:
            return

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        self._handler = None
        self._running = False

    def _handle_refresh(self, event_type: str, affected_agents: list[str]) -> None:
        """Internal handler for refresh events."""
        # Forward stale callback if set
        if self.on_stale_change:
            self._stale_tracker.on_stale_change = self.on_stale_change

        if self.on_refresh:
            self.on_refresh(event_type, affected_agents)

    def _handle_changes_for_analysis(self, changed_files: list[str]) -> None:
        """Internal handler for QC analysis trigger."""
        # Update stale tracker with changed files
        self._stale_tracker.mark_files_changed(changed_files)

        if self.on_changes_for_analysis:
            self.on_changes_for_analysis(changed_files)

    def mark_agent_queried(self, agent_id: str) -> None:
        """Mark that an agent was queried (used to track freshness).

        Args:
            agent_id: ID of the agent that was queried.
        """
        self._stale_tracker.mark_agent_queried(agent_id)

    @property
    def is_running(self) -> bool:
        """Check if watcher is currently running."""
        return self._running

    def __enter__(self) -> "FileWatcher":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.stop()

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return f"<FileWatcher path={self.project_root} status={status}>"


def is_watchdog_available() -> bool:
    """Check if watchdog is installed."""
    return _watchdog_available
