"""File system watcher for automatic context refresh and QC analysis.

This module monitors file changes and triggers:
1. Agent context refresh when relevant files are modified
2. QC analysis pipeline when changes are detected (if enabled)

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
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from agenthub.hub import AgentHub

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


class FileWatcher:
    """Watches files and refreshes agent contexts on changes.

    This class provides automatic context refresh when monitored files
    change, ensuring agents always have up-to-date knowledge.

    Additionally, when QC analysis is enabled, it can trigger the
    analysis pipeline to detect potential issues in changed code.

    Example:
        >>> watcher = FileWatcher(hub, "/path/to/project")
        >>>
        >>> # Optional: Add callback for refresh events
        >>> watcher.on_refresh = lambda event, agents: print(f"Refreshed: {agents}")
        >>>
        >>> # Optional: Add callback for QC analysis
        >>> watcher.on_changes_for_analysis = lambda files: hub.analyze_changes(files)
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

        # Callback for refresh events
        self.on_refresh: Optional[Callable[[str, list[str]], None]] = None

        # Callback for QC analysis (triggered with list of changed file paths)
        self.on_changes_for_analysis: Optional[Callable[[list[str]], None]] = None

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
        if self.on_refresh:
            self.on_refresh(event_type, affected_agents)

    def _handle_changes_for_analysis(self, changed_files: list[str]) -> None:
        """Internal handler for QC analysis trigger."""
        if self.on_changes_for_analysis:
            self.on_changes_for_analysis(changed_files)

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
