from __future__ import annotations
"""Git-aware context caching.

This module provides caching that automatically invalidates when:
1. Git commit hash changes (code was updated)
2. Specific files are modified
3. TTL expires

Example:
    >>> cache = GitAwareCache("./my-project")
    >>>
    >>> # Cache with automatic git invalidation
    >>> context = cache.get("api_agent")
    >>> if context is None:
    ...     context = build_expensive_context()
    ...     cache.set("api_agent", context, watch_paths=["src/api/*.py"])
"""

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class CacheEntry:
    """A single cache entry with metadata for invalidation."""

    content: str
    created_at: float
    git_commit: Optional[str] = None
    file_hashes: dict[str, str] = field(default_factory=dict)
    watch_paths: list[str] = field(default_factory=list)
    ttl_seconds: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds


class GitAwareCache:
    """Context cache with git-aware invalidation.

    This cache automatically invalidates entries when:
    - Git HEAD changes (new commits, pulls, checkouts)
    - Watched files are modified
    - TTL expires

    Example:
        >>> cache = GitAwareCache("/path/to/repo")
        >>>
        >>> # Get or compute with automatic invalidation
        >>> context = cache.get_or_compute(
        ...     "api_agent",
        ...     compute_fn=lambda: expensive_context_build(),
        ...     watch_paths=["src/api/**/*.py"],
        ...     ttl_seconds=300,  # 5 minutes
        ... )
    """

    def __init__(self, project_root: str):
        """Initialize the cache.

        Args:
            project_root: Path to the git repository root.
        """
        self.project_root = Path(project_root).resolve()
        self._cache: dict[str, CacheEntry] = {}
        self._git_available = self._check_git()

    def _check_git(self) -> bool:
        """Check if git is available and this is a git repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_current_commit(self) -> Optional[str]:
        """Get the current git HEAD commit hash."""
        if not self._git_available:
            return None

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get MD5 hash of a file's contents."""
        full_path = self.project_root / file_path
        if not full_path.exists():
            return None

        try:
            content = full_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return None

    def get_path_hashes(self, patterns: list[str]) -> dict[str, str]:
        """Get hashes for all files matching patterns.

        Args:
            patterns: Glob patterns relative to project root.

        Returns:
            Dict mapping file path to content hash.
        """
        hashes: dict[str, str] = {}

        for pattern in patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(self.project_root))
                    file_hash = self.get_file_hash(rel_path)
                    if file_hash:
                        hashes[rel_path] = file_hash

        return hashes

    def is_valid(self, key: str) -> bool:
        """Check if a cache entry is still valid.

        Checks:
        1. Entry exists
        2. TTL not expired
        3. Git commit hasn't changed (if git-aware)
        4. Watched files haven't changed
        """
        if key not in self._cache:
            return False

        entry = self._cache[key]

        # Check TTL
        if entry.is_expired():
            return False

        # Check git commit
        if entry.git_commit is not None:
            current_commit = self.get_current_commit()
            if current_commit != entry.git_commit:
                return False

        # Check file hashes
        if entry.watch_paths:
            current_hashes = self.get_path_hashes(entry.watch_paths)
            if current_hashes != entry.file_hashes:
                return False

        return True

    def get(self, key: str) -> Optional[str]:
        """Get a cached value if valid.

        Args:
            key: Cache key.

        Returns:
            Cached content or None if invalid/missing.
        """
        if self.is_valid(key):
            return self._cache[key].content
        return None

    def set(
        self,
        key: str,
        content: str,
        watch_paths: Optional[list[str]] = None,
        ttl_seconds: Optional[float] = None,
        track_git: bool = True,
    ) -> None:
        """Set a cache value.

        Args:
            key: Cache key.
            content: Content to cache.
            watch_paths: Glob patterns for files to watch for changes.
            ttl_seconds: Time-to-live in seconds.
            track_git: Whether to track git commit for invalidation.
        """
        watch_paths = watch_paths or []

        entry = CacheEntry(
            content=content,
            created_at=time.time(),
            git_commit=self.get_current_commit() if track_git else None,
            file_hashes=self.get_path_hashes(watch_paths),
            watch_paths=watch_paths,
            ttl_seconds=ttl_seconds,
        )

        self._cache[key] = entry

    def invalidate(self, key: str) -> bool:
        """Manually invalidate a cache entry.

        Args:
            key: Cache key to invalidate.

        Returns:
            True if entry existed and was removed.
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def invalidate_all(self) -> int:
        """Invalidate all cache entries.

        Returns:
            Number of entries invalidated.
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], str],
        watch_paths: Optional[list[str]] = None,
        ttl_seconds: Optional[float] = None,
        track_git: bool = True,
    ) -> str:
        """Get cached value or compute and cache it.

        This is the recommended way to use the cache.

        Args:
            key: Cache key.
            compute_fn: Function to call if cache miss.
            watch_paths: Glob patterns for files to watch.
            ttl_seconds: Time-to-live in seconds.
            track_git: Whether to track git commit.

        Returns:
            Cached or freshly computed content.

        Example:
            >>> context = cache.get_or_compute(
            ...     "api_agent",
            ...     compute_fn=lambda: build_api_context(),
            ...     watch_paths=["src/api/**/*.py"],
            ...     ttl_seconds=300,
            ... )
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        # Compute fresh content
        content = compute_fn()

        # Cache it
        self.set(
            key=key,
            content=content,
            watch_paths=watch_paths,
            ttl_seconds=ttl_seconds,
            track_git=track_git,
        )

        return content

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats.
        """
        valid_count = sum(1 for key in self._cache if self.is_valid(key))
        invalid_count = len(self._cache) - valid_count

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_count,
            "invalid_entries": invalid_count,
            "current_git_commit": self.get_current_commit(),
            "git_available": self._git_available,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"<GitAwareCache entries={stats['total_entries']} valid={stats['valid_entries']}>"
