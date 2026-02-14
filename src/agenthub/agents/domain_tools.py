from __future__ import annotations
"""Domain-scoped tools for agents.

Gives agents the ability to dynamically search and read files within their
domain directories, rather than relying only on pre-loaded static context.
"""

import fnmatch
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agenthub.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (Anthropic API format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "grep_domain",
        "description": (
            "Search for a text pattern (regex) within files in your domain. "
            "Returns matching lines with file paths and line numbers. "
            "Use this to find specific functions, classes, variables, or patterns "
            "when the answer isn't in your pre-loaded context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "file_glob": {
                    "type": "string",
                    "description": (
                        "Optional glob to filter files (e.g. '*.py'). "
                        "Default: all files."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file within your domain. "
            "Use this when you need to see the complete implementation "
            "of a specific file you found via grep_domain or list_files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path from project root "
                        "(e.g. 'src/pricing/engine.py')"
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Start line (1-indexed). Default: 1",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to read. Default: 200",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files in a directory within your domain. "
            "Use this to discover what files exist before reading them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": (
                        "Relative directory path (e.g. 'src/pricing/')"
                    ),
                },
                "pattern": {
                    "type": "string",
                    "description": (
                        "Optional glob pattern (e.g. '*.py'). Default: all files."
                    ),
                },
            },
            "required": ["directory"],
        },
    },
]


# ---------------------------------------------------------------------------
# Domain scoping
# ---------------------------------------------------------------------------

class DomainScope:
    """Computes and validates allowed directories for an agent.

    Derives allowed directories from an agent's ``context_paths``.  For
    example, if an agent owns ``["src/pricing/engine.py",
    "src/pricing/models.py"]``, its allowed dirs are ``{"src/pricing"}``.

    All path operations resolve to absolute paths under ``root_path`` and
    reject path traversal (``..``).
    """

    def __init__(self, root_path: str, context_paths: list[str]) -> None:
        self.root = Path(root_path).resolve()
        self.context_paths = context_paths
        self.allowed_dirs: list[Path] = self._compute_allowed_dirs()

    # -- public API ---------------------------------------------------------

    def is_allowed(self, relative_path: str) -> bool:
        """Check if *relative_path* falls within any allowed directory."""
        try:
            resolved = self.resolve_path(relative_path)
        except (PermissionError, ValueError):
            return False
        return True

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path to absolute, validating it is allowed.

        Raises ``PermissionError`` if the path is outside the allowed dirs.
        Raises ``ValueError`` for invalid paths (e.g. containing ``..``).
        """
        # Block path traversal
        if ".." in Path(relative_path).parts:
            raise ValueError(f"Path traversal not allowed: {relative_path}")

        resolved = (self.root / relative_path).resolve()

        # Must be under root
        if not str(resolved).startswith(str(self.root)):
            raise PermissionError(f"Path outside project root: {relative_path}")

        # Must be under an allowed directory
        for allowed in self.allowed_dirs:
            if str(resolved).startswith(str(allowed)):
                return resolved

        raise PermissionError(
            f"Path outside agent domain: {relative_path}. "
            f"Allowed dirs: {[str(d.relative_to(self.root)) for d in self.allowed_dirs]}"
        )

    # -- internal -----------------------------------------------------------

    def _compute_allowed_dirs(self) -> list[Path]:
        """Derive allowed directories from context_paths."""
        raw_dirs: set[Path] = set()

        for cp in self.context_paths:
            # Strip glob characters to get the directory portion
            clean = cp.split("*")[0].rstrip("/")
            if not clean:
                # Pattern like "**/*.py" → allow entire root
                raw_dirs.add(self.root)
                continue

            p = self.root / clean
            if p.is_file():
                raw_dirs.add(p.parent.resolve())
            elif p.is_dir():
                raw_dirs.add(p.resolve())
            else:
                # Path doesn't exist yet; use parent
                raw_dirs.add(p.parent.resolve())

        if not raw_dirs:
            return []

        # Deduplicate to shortest unique prefixes.
        # Sort by string length so shorter paths come first.
        sorted_dirs = sorted(raw_dirs, key=lambda d: len(str(d)))
        result: list[Path] = []
        for d in sorted_dirs:
            # Skip if already covered by a shorter prefix
            if any(str(d).startswith(str(existing)) for existing in result):
                continue
            result.append(d)

        return result


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

MAX_GREP_MATCHES = 50
MAX_LIST_FILES = 100
MAX_READ_LINES = 500


class DomainToolExecutor:
    """Executes domain-scoped tools within a ``DomainScope``."""

    def __init__(self, scope: DomainScope) -> None:
        self.scope = scope

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch and execute a tool, returning a string result."""
        try:
            if tool_name == "grep_domain":
                return self._grep_domain(
                    pattern=tool_input["pattern"],
                    file_glob=tool_input.get("file_glob"),
                )
            elif tool_name == "read_file":
                return self._read_file(
                    path=tool_input["path"],
                    offset=tool_input.get("offset", 1),
                    limit=tool_input.get("limit", 200),
                )
            elif tool_name == "list_files":
                return self._list_files(
                    directory=tool_input["directory"],
                    pattern=tool_input.get("pattern"),
                )
            else:
                return f"Unknown tool: {tool_name}"
        except PermissionError as e:
            return f"Access denied: {e}"
        except ValueError as e:
            return f"Invalid input: {e}"
        except Exception as e:
            logger.warning(f"Tool {tool_name} error: {e}")
            return f"Error: {type(e).__name__}: {e}"

    # -- tool implementations -----------------------------------------------

    def _grep_domain(
        self, pattern: str, file_glob: Optional[str] = None
    ) -> str:
        """Search for *pattern* across all files in allowed dirs."""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex: {e}"

        matches: list[str] = []

        for allowed_dir in self.scope.allowed_dirs:
            if not allowed_dir.is_dir():
                continue
            for fpath in sorted(allowed_dir.rglob("*")):
                if not fpath.is_file():
                    continue
                # Apply file glob filter
                if file_glob and not fnmatch.fnmatch(fpath.name, file_glob):
                    continue
                # Skip binary files
                if _is_binary(fpath):
                    continue

                try:
                    rel = fpath.relative_to(self.scope.root)
                    lines = fpath.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                    for i, line in enumerate(lines, 1):
                        if regex.search(line):
                            matches.append(f"{rel}:{i}: {line.rstrip()}")
                            if len(matches) >= MAX_GREP_MATCHES:
                                matches.append(
                                    f"... (truncated at {MAX_GREP_MATCHES} matches)"
                                )
                                return "\n".join(matches)
                except Exception:
                    continue

        if not matches:
            return f"No matches found for pattern: {pattern}"

        return "\n".join(matches)

    def _read_file(
        self, path: str, offset: int = 1, limit: int = 200
    ) -> str:
        """Read a file with line numbers."""
        resolved = self.scope.resolve_path(path)

        if not resolved.is_file():
            return f"File not found: {path}"

        if _is_binary(resolved):
            return f"Binary file, cannot read: {path}"

        limit = min(limit, MAX_READ_LINES)
        offset = max(1, offset)

        try:
            lines = resolved.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
        except Exception as e:
            return f"Error reading file: {e}"

        total_lines = len(lines)
        selected = lines[offset - 1 : offset - 1 + limit]

        parts: list[str] = []
        for i, line in enumerate(selected, offset):
            parts.append(f"{i:>5}\t{line}")

        header = f"# {path} (lines {offset}-{offset + len(selected) - 1} of {total_lines})"
        return header + "\n" + "\n".join(parts)

    def _list_files(
        self, directory: str, pattern: Optional[str] = None
    ) -> str:
        """List files in a directory."""
        resolved = self.scope.resolve_path(directory)

        if not resolved.is_dir():
            return f"Not a directory: {directory}"

        files: list[str] = []
        for fpath in sorted(resolved.iterdir()):
            if pattern and not fnmatch.fnmatch(fpath.name, pattern):
                continue
            rel = fpath.relative_to(self.scope.root)
            suffix = "/" if fpath.is_dir() else ""
            files.append(f"{rel}{suffix}")
            if len(files) >= MAX_LIST_FILES:
                files.append(f"... (truncated at {MAX_LIST_FILES} entries)")
                break

        if not files:
            return f"No files found in: {directory}"

        return "\n".join(files)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_tool_definitions() -> list[dict[str, Any]]:
    """Return the tool definitions for the Anthropic API."""
    return TOOL_DEFINITIONS


def create_domain_tools(
    agent: "BaseAgent",
) -> Optional[tuple[list[dict[str, Any]], DomainToolExecutor]]:
    """Create domain tools for an agent if possible.

    Returns ``None`` if the agent doesn't have enough information
    (root_path + context_paths) to create scoped tools — in that case
    the agent falls back to the original single-shot behaviour.
    """
    # Get root_path from agent attribute or metadata
    root_path: Optional[str] = getattr(agent, "root_path", None)
    if root_path is None:
        root_path = agent.spec.metadata.get("root_path")
    if root_path is None:
        # Fallback: Tier A agents store self.project_root
        root_path = getattr(agent, "project_root", None)
    if root_path is None:
        return None

    context_paths = agent.spec.context_paths
    if not context_paths:
        return None

    scope = DomainScope(root_path, context_paths)
    if not scope.allowed_dirs:
        return None

    executor = DomainToolExecutor(scope)
    return get_tool_definitions(), executor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BINARY_EXTENSIONS = frozenset({
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll", ".exe",
    ".class", ".jar", ".war", ".ear",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".db", ".sqlite", ".sqlite3",
    ".lock",
})


def _is_binary(path: Path) -> bool:
    """Quick check if a file is likely binary."""
    return path.suffix.lower() in _BINARY_EXTENSIONS
