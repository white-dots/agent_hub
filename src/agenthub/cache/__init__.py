"""Context caching with automatic invalidation.

This module provides intelligent caching for agent contexts with:
- Git-aware invalidation (refresh on new commits)
- File watching (refresh on file changes)
- TTL-based expiry

Example:
    >>> from agenthub.cache import GitAwareCache, FileWatcher
    >>>
    >>> # Git-aware caching
    >>> cache = GitAwareCache("/path/to/repo")
    >>> context = cache.get_or_compute(
    ...     "api_agent",
    ...     compute_fn=build_context,
    ...     watch_paths=["src/api/**/*.py"],
    ... )
    >>>
    >>> # File watching for auto-refresh
    >>> watcher = FileWatcher(hub, "/path/to/repo")
    >>> watcher.start()
"""

from agenthub.cache.git_cache import CacheEntry, GitAwareCache
from agenthub.cache.watcher import (
    FileWatcher,
    WatchConfig,
    is_watchdog_available,
)

__all__ = [
    # Git-aware cache
    "GitAwareCache",
    "CacheEntry",
    # File watcher
    "FileWatcher",
    "WatchConfig",
    "is_watchdog_available",
]
