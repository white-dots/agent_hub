"""Support for .agenthubignore files.

This module provides gitignore-style pattern matching to exclude files
and directories from Tier B agent auto-discovery.

The .agenthubignore file supports:
- Glob patterns (*.test.ts, **/__tests__/*)
- Directory patterns (node_modules/, dist/)
- Negation patterns (!important.py)
- Comments (# this is a comment)
"""

import fnmatch
import re
from pathlib import Path


class IgnorePatterns:
    """Handles .agenthubignore pattern matching.

    Supports gitignore-style patterns for excluding files and directories
    from auto-agent discovery.

    Example:
        >>> ignore = IgnorePatterns("/path/to/project")
        >>> ignore.is_ignored("node_modules/foo.js")
        True
        >>> ignore.is_ignored("src/main.py")
        False
    """

    # Default patterns always ignored (in addition to .agenthubignore)
    DEFAULT_IGNORE = [
        # Python
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "*.egg-info/",
        ".eggs/",
        "*.egg",
        ".tox/",
        "htmlcov/",
        ".coverage",

        # Virtual environments
        ".venv/",
        "venv/",
        "env/",
        ".env/",

        # Node
        "node_modules/",

        # Build outputs
        "dist/",
        "build/",
        "out/",
        ".next/",
        ".nuxt/",

        # IDE
        ".idea/",
        ".vscode/",
        "*.swp",
        "*.swo",

        # Version control
        ".git/",
        ".svn/",
        ".hg/",

        # OS
        ".DS_Store",
        "Thumbs.db",

        # Lock files (low value for agents)
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "Pipfile.lock",

        # Compiled/minified (not readable)
        "*.min.js",
        "*.min.css",
        "*.bundle.js",
        "*.chunk.js",

        # Generated types
        "*.d.ts",

        # Test fixtures and mocks (optional - can be un-ignored)
        "__mocks__/",
        "__fixtures__/",

        # Documentation builds
        "_build/",
        "site/",
    ]

    def __init__(self, project_root: str | Path):
        """Initialize with project root path.

        Args:
            project_root: Path to the project root directory.
        """
        self.root = Path(project_root).resolve()
        self.patterns: list[tuple[str, bool]] = []  # (pattern, is_negation)

        # Load default patterns
        for pattern in self.DEFAULT_IGNORE:
            self.patterns.append((pattern, False))

        # Load .agenthubignore if exists
        self._load_ignore_file()

    def _load_ignore_file(self) -> None:
        """Load patterns from .agenthubignore file."""
        ignore_file = self.root / ".agenthubignore"
        if not ignore_file.exists():
            return

        try:
            content = ignore_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Handle negation
                if line.startswith("!"):
                    pattern = line[1:].strip()
                    self.patterns.append((pattern, True))  # is_negation=True
                else:
                    self.patterns.append((line, False))
        except Exception:
            pass

    def is_ignored(self, rel_path: str | Path) -> bool:
        """Check if a relative path should be ignored.

        Args:
            rel_path: Path relative to project root.

        Returns:
            True if the path should be ignored.
        """
        rel_path = str(rel_path).replace("\\", "/")

        # Track if currently ignored (can be toggled by negation)
        ignored = False

        for pattern, is_negation in self.patterns:
            if self._matches(rel_path, pattern):
                ignored = not is_negation

        return ignored

    def _matches(self, path: str, pattern: str) -> bool:
        """Check if path matches a single pattern.

        Args:
            path: Relative file path.
            pattern: Gitignore-style pattern.

        Returns:
            True if the path matches the pattern.
        """
        # Normalize pattern
        pattern = pattern.replace("\\", "/")

        # Directory pattern (ends with /)
        if pattern.endswith("/"):
            dir_pattern = pattern[:-1]
            # Match if any path component matches
            parts = path.split("/")
            for i, part in enumerate(parts):
                partial = "/".join(parts[:i+1])
                if fnmatch.fnmatch(part, dir_pattern):
                    return True
                if fnmatch.fnmatch(partial, dir_pattern):
                    return True
            return False

        # Check if pattern contains /
        if "/" in pattern:
            # Pattern with path separator - match from root or as suffix
            if pattern.startswith("/"):
                # Anchored to root
                return fnmatch.fnmatch(path, pattern[1:])
            # Can match anywhere in path
            if fnmatch.fnmatch(path, pattern):
                return True
            if fnmatch.fnmatch(path, "**/" + pattern):
                return True
            # Try matching suffix
            pattern_parts = pattern.split("/")
            path_parts = path.split("/")
            if len(path_parts) >= len(pattern_parts):
                suffix = "/".join(path_parts[-len(pattern_parts):])
                if fnmatch.fnmatch(suffix, pattern):
                    return True
        else:
            # Simple pattern - match against filename or any path component
            filename = path.split("/")[-1]
            if fnmatch.fnmatch(filename, pattern):
                return True
            # Also check against each path component
            for part in path.split("/"):
                if fnmatch.fnmatch(part, pattern):
                    return True

        return False

    def filter_paths(self, paths: list[str | Path]) -> list[str]:
        """Filter a list of paths, removing ignored ones.

        Args:
            paths: List of paths to filter.

        Returns:
            List of paths that should not be ignored.
        """
        result = []
        for path in paths:
            rel_path = str(path)
            if not self.is_ignored(rel_path):
                result.append(rel_path)
        return result

    def get_active_patterns(self) -> list[str]:
        """Get list of active ignore patterns for debugging.

        Returns:
            List of pattern strings (with ! prefix for negations).
        """
        return [
            f"!{pattern}" if is_neg else pattern
            for pattern, is_neg in self.patterns
        ]


def load_ignore_patterns(project_root: str | Path) -> IgnorePatterns:
    """Convenience function to load ignore patterns.

    Args:
        project_root: Path to project root.

    Returns:
        IgnorePatterns instance.
    """
    return IgnorePatterns(project_root)
