from __future__ import annotations
"""Project-scoped tools for enhanced Tier B agents.

Gives agents exploration-first capabilities across the entire project,
mirroring the FDA Local Worker pattern. Unlike domain_tools.py (which
restricts agents to their assigned directories), project tools allow
full project access with path-traversal validation.

Tools:
    list_directory  — tree listing of project structure
    read_file       — read file contents (tracks reads for diffing)
    search_files    — regex search across entire project
    write_file      — record proposed file changes (NOT applied to disk)
    run_command     — execute shell commands with safety checks
"""

import difflib
import fnmatch
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from agenthub.agents.domain_tools import _is_binary

if TYPE_CHECKING:
    from agenthub.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LIST_ENTRIES = 200
MAX_READ_LINES = 1000
MAX_SEARCH_MATCHES = 100
MAX_COMMAND_OUTPUT = 10 * 1024  # 10KB
COMMAND_TIMEOUT = 30  # seconds

EXCLUDED_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".env", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", ".eggs", "*.egg-info", ".next", ".nuxt",
    ".svelte-kit", "target", ".cargo",
})

BLOCKED_COMMAND_PATTERNS = [
    r"rm\s+(-\w*\s+)*-\w*r\w*\s+/\s*$",  # rm -rf /
    r"rm\s+(-\w*\s+)*-\w*r\w*\s+/[a-z]",  # rm -rf /etc, /usr, etc.
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\binit\s+0",
    r">\s*/dev/sd",
    r">\s*/dev/nvm",
    r":\(\)\s*\{\s*:\|\:\s*&\s*\}",  # fork bomb
    r"\bchmod\s+(-\w+\s+)*777\s+/",
    r"\bchown\s+.*\s+/\s*$",
    r"\bcurl\b.*\|\s*(ba)?sh",  # curl | sh
    r"\bwget\b.*\|\s*(ba)?sh",
]

# ---------------------------------------------------------------------------
# Tool definitions (Anthropic API format)
# ---------------------------------------------------------------------------

ENHANCED_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_directory",
        "description": (
            "List files and directories in the project. Shows a tree view "
            "of the structure. Use this first to understand the project layout "
            "before searching or reading files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": (
                        "Relative directory path from project root "
                        "(e.g. 'src/', '.'). Default: project root."
                    ),
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Max depth to traverse. Default: 3",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file with line numbers. "
            "Use this to understand specific implementations after "
            "finding relevant files via search_files or list_directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path from project root "
                        "(e.g. 'src/auth/handler.py')"
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Start line (1-indexed). Default: 1",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Max lines to read. Default: 200, Max: 1000"
                    ),
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search for a regex pattern across files in the project. "
            "Returns matching lines with file paths and line numbers. "
            "Use this to find functions, classes, imports, or any pattern."
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
                        "Glob to filter files (e.g. '*.py', '*.ts'). "
                        "Default: all text files."
                    ),
                },
                "directory": {
                    "type": "string",
                    "description": (
                        "Directory to search within (relative). "
                        "Default: entire project."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Record a proposed file change. The change is NOT applied to disk — "
            "it is stored for review and approval. You MUST provide the complete "
            "file content (not just a diff). Always read_file first before writing "
            "to understand the current state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path from project root "
                        "(e.g. 'src/auth/handler.py')"
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "The complete new file content",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Brief description of the change "
                        "(e.g. 'Add input validation to login handler')"
                    ),
                },
            },
            "required": ["path", "content", "description"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Execute a shell command in the project directory. "
            "Use for running tests, checking versions, linting, or "
            "other analysis commands. Has a 30-second timeout and "
            "10KB output limit. Dangerous commands are blocked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "working_directory": {
                    "type": "string",
                    "description": (
                        "Relative working directory. Default: project root."
                    ),
                },
            },
            "required": ["command"],
        },
    },
]


# ---------------------------------------------------------------------------
# Project scoping (full-project access with traversal protection)
# ---------------------------------------------------------------------------

class ProjectScope:
    """Validates paths are within the project root.

    Unlike ``DomainScope``, this allows access to ANY file under the
    project root — not just the agent's assigned directories.
    """

    def __init__(self, root_path: str) -> None:
        self.root = Path(root_path).resolve()

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path, blocking traversal outside root.

        Raises ``ValueError`` for paths with ``..``.
        Raises ``PermissionError`` for paths outside root.
        """
        if ".." in Path(relative_path).parts:
            raise ValueError(f"Path traversal not allowed: {relative_path}")

        resolved = (self.root / relative_path).resolve()

        if not str(resolved).startswith(str(self.root)):
            raise PermissionError(f"Path outside project root: {relative_path}")

        return resolved

    def is_allowed(self, relative_path: str) -> bool:
        try:
            self.resolve_path(relative_path)
            return True
        except (PermissionError, ValueError):
            return False


# ---------------------------------------------------------------------------
# Pending changes
# ---------------------------------------------------------------------------

@dataclass
class PendingChange:
    """A proposed file change recorded by write_file."""
    path: str
    new_content: str
    original_content: Optional[str]
    description: str
    unified_diff: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Tech stack detection
# ---------------------------------------------------------------------------

class TechStackDetector:
    """Detect project tech stack from marker files."""

    # (marker_file, language, framework, build_tool)
    MARKERS = [
        ("pyproject.toml", "python", None, "uv/pip"),
        ("setup.py", "python", None, "pip"),
        ("requirements.txt", "python", None, "pip"),
        ("Pipfile", "python", None, "pipenv"),
        ("package.json", "javascript", None, "npm"),
        ("yarn.lock", "javascript", None, "yarn"),
        ("pnpm-lock.yaml", "javascript", None, "pnpm"),
        ("bun.lockb", "javascript", None, "bun"),
        ("Cargo.toml", "rust", None, "cargo"),
        ("go.mod", "go", None, "go"),
        ("Gemfile", "ruby", None, "bundler"),
        ("pom.xml", "java", None, "maven"),
        ("build.gradle", "java", None, "gradle"),
        ("build.gradle.kts", "kotlin", None, "gradle"),
        ("composer.json", "php", None, "composer"),
        ("mix.exs", "elixir", None, "mix"),
        ("CMakeLists.txt", "c/c++", None, "cmake"),
        ("Makefile", None, None, "make"),
    ]

    FRAMEWORK_MARKERS = [
        # Python
        ("manage.py", "django"),
        ("app.py", "flask"),
        # JS/TS
        ("next.config.js", "nextjs"),
        ("next.config.mjs", "nextjs"),
        ("next.config.ts", "nextjs"),
        ("nuxt.config.ts", "nuxt"),
        ("svelte.config.js", "svelte"),
        ("angular.json", "angular"),
        ("vite.config.ts", "vite"),
        ("vite.config.js", "vite"),
        ("tsconfig.json", "typescript"),
    ]

    TEST_MARKERS = [
        ("pytest.ini", "pytest"),
        ("setup.cfg", "pytest"),  # often contains [tool:pytest]
        ("jest.config.js", "jest"),
        ("jest.config.ts", "jest"),
        ("vitest.config.ts", "vitest"),
        (".mocharc.yml", "mocha"),
    ]

    @classmethod
    def detect(cls, root_path: str) -> dict[str, list[str]]:
        root = Path(root_path)
        languages: set[str] = set()
        frameworks: set[str] = set()
        build_tools: set[str] = set()
        test_frameworks: set[str] = set()

        for marker, lang, framework, tool in cls.MARKERS:
            if (root / marker).exists():
                if lang:
                    languages.add(lang)
                if framework:
                    frameworks.add(framework)
                if tool:
                    build_tools.add(tool)

        for marker, framework in cls.FRAMEWORK_MARKERS:
            if (root / marker).exists():
                frameworks.add(framework)

        for marker, test_fw in cls.TEST_MARKERS:
            if (root / marker).exists():
                test_frameworks.add(test_fw)

        # Detect frameworks from pyproject.toml dependencies
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            try:
                text = pyproject.read_text(encoding="utf-8", errors="ignore")
                if "fastapi" in text.lower():
                    frameworks.add("fastapi")
                if "django" in text.lower():
                    frameworks.add("django")
                if "flask" in text.lower():
                    frameworks.add("flask")
                if "pytest" in text.lower():
                    test_frameworks.add("pytest")
            except Exception:
                pass

        # Detect from package.json
        pkg = root / "package.json"
        if pkg.exists():
            try:
                text = pkg.read_text(encoding="utf-8", errors="ignore")
                if '"react"' in text:
                    frameworks.add("react")
                if '"vue"' in text:
                    frameworks.add("vue")
                if '"express"' in text:
                    frameworks.add("express")
            except Exception:
                pass

        return {
            "languages": sorted(languages),
            "frameworks": sorted(frameworks),
            "build_tools": sorted(build_tools),
            "test_frameworks": sorted(test_frameworks),
        }


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ProjectToolExecutor:
    """Executes project-scoped tools with state tracking.

    Tracks files read (for diff generation) and pending changes
    (for approval-gated writes).
    """

    def __init__(self, scope: ProjectScope) -> None:
        self.scope = scope
        self._files_read: dict[str, str] = {}  # path -> content at read time
        self._pending_changes: dict[str, PendingChange] = {}

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch and execute a tool, returning a string result."""
        try:
            if tool_name == "list_directory":
                return self._list_directory(
                    directory=tool_input.get("directory", "."),
                    max_depth=tool_input.get("max_depth", 3),
                )
            elif tool_name == "read_file":
                return self._read_file(
                    path=tool_input["path"],
                    offset=tool_input.get("offset", 1),
                    limit=tool_input.get("limit", 200),
                )
            elif tool_name == "search_files":
                return self._search_files(
                    pattern=tool_input["pattern"],
                    file_glob=tool_input.get("file_glob"),
                    directory=tool_input.get("directory"),
                )
            elif tool_name == "write_file":
                return self._write_file(
                    path=tool_input["path"],
                    content=tool_input["content"],
                    description=tool_input.get("description", ""),
                )
            elif tool_name == "run_command":
                return self._run_command(
                    command=tool_input["command"],
                    working_directory=tool_input.get("working_directory"),
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

    def get_pending_changes(self) -> list[PendingChange]:
        return list(self._pending_changes.values())

    def get_summary(self) -> str:
        """Summary for graceful timeout — show what was explored and changed."""
        parts: list[str] = []

        if self._files_read:
            parts.append(f"Files explored ({len(self._files_read)}):")
            for path in sorted(self._files_read):
                parts.append(f"  - {path}")

        if self._pending_changes:
            parts.append(f"\nPending changes ({len(self._pending_changes)}):")
            for path, change in sorted(self._pending_changes.items()):
                parts.append(f"  - {path}: {change.description}")

        if not parts:
            parts.append("No files explored or changes recorded.")

        return "\n".join(parts)

    # -- tool implementations -----------------------------------------------

    def _list_directory(
        self, directory: str = ".", max_depth: int = 3
    ) -> str:
        resolved = self.scope.resolve_path(directory)

        if not resolved.is_dir():
            return f"Not a directory: {directory}"

        entries: list[str] = []
        self._walk_tree(resolved, resolved, entries, 0, max_depth)

        if not entries:
            return f"Empty directory: {directory}"

        rel_dir = resolved.relative_to(self.scope.root)
        header = f"# {rel_dir}/ ({len(entries)} entries)"
        return header + "\n" + "\n".join(entries)

    def _walk_tree(
        self,
        current: Path,
        base: Path,
        entries: list[str],
        depth: int,
        max_depth: int,
    ) -> None:
        if depth > max_depth or len(entries) >= MAX_LIST_ENTRIES:
            return

        try:
            children = sorted(current.iterdir())
        except PermissionError:
            return

        for child in children:
            if len(entries) >= MAX_LIST_ENTRIES:
                entries.append(f"... (truncated at {MAX_LIST_ENTRIES} entries)")
                return

            name = child.name
            if name in EXCLUDED_DIRS:
                continue
            # Also match patterns like *.egg-info
            if any(fnmatch.fnmatch(name, pat) for pat in EXCLUDED_DIRS if "*" in pat):
                continue

            indent = "  " * depth
            rel = child.relative_to(self.scope.root)

            if child.is_dir():
                entries.append(f"{indent}{rel}/")
                self._walk_tree(child, base, entries, depth + 1, max_depth)
            else:
                entries.append(f"{indent}{rel}")

    def _read_file(
        self, path: str, offset: int = 1, limit: int = 200
    ) -> str:
        resolved = self.scope.resolve_path(path)

        if not resolved.is_file():
            return f"File not found: {path}"

        if _is_binary(resolved):
            return f"Binary file, cannot read: {path}"

        limit = min(limit, MAX_READ_LINES)
        offset = max(1, offset)

        try:
            content = resolved.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"Error reading file: {e}"

        # Track read for diff generation
        self._files_read[path] = content

        lines = content.splitlines()
        total_lines = len(lines)
        selected = lines[offset - 1 : offset - 1 + limit]

        parts: list[str] = []
        for i, line in enumerate(selected, offset):
            parts.append(f"{i:>5}\t{line}")

        end_line = offset + len(selected) - 1
        header = f"# {path} (lines {offset}-{end_line} of {total_lines})"
        return header + "\n" + "\n".join(parts)

    def _search_files(
        self,
        pattern: str,
        file_glob: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex: {e}"

        search_root = self.scope.root
        if directory:
            search_root = self.scope.resolve_path(directory)
            if not search_root.is_dir():
                return f"Not a directory: {directory}"

        matches: list[str] = []

        for fpath in sorted(search_root.rglob("*")):
            if not fpath.is_file():
                continue

            # Skip excluded directories
            if any(part in EXCLUDED_DIRS for part in fpath.relative_to(self.scope.root).parts):
                continue

            if file_glob and not fnmatch.fnmatch(fpath.name, file_glob):
                continue

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
                        if len(matches) >= MAX_SEARCH_MATCHES:
                            matches.append(
                                f"... (truncated at {MAX_SEARCH_MATCHES} matches)"
                            )
                            return "\n".join(matches)
            except Exception:
                continue

        if not matches:
            return f"No matches found for pattern: {pattern}"

        return "\n".join(matches)

    def _write_file(
        self, path: str, content: str, description: str = ""
    ) -> str:
        # Validate path
        self.scope.resolve_path(path)

        # Auto-read if not already read (for diff generation)
        original: Optional[str] = None
        if path in self._files_read:
            original = self._files_read[path]
        else:
            resolved = self.scope.resolve_path(path)
            if resolved.is_file():
                try:
                    original = resolved.read_text(
                        encoding="utf-8", errors="ignore"
                    )
                    self._files_read[path] = original
                except Exception:
                    pass

        # Generate unified diff
        if original is not None:
            diff_lines = list(difflib.unified_diff(
                original.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            ))
            diff_str = "".join(diff_lines)
        else:
            diff_str = f"(new file: {len(content.splitlines())} lines)"

        change = PendingChange(
            path=path,
            new_content=content,
            original_content=original,
            description=description,
            unified_diff=diff_str,
        )
        self._pending_changes[path] = change

        if diff_str.strip():
            return (
                f"Change recorded for {path}: {description}\n"
                f"Diff preview:\n{diff_str[:3000]}"
            )
        return f"Change recorded for {path}: {description} (no diff — content unchanged)"

    def _run_command(
        self,
        command: str,
        working_directory: Optional[str] = None,
    ) -> str:
        # Safety check
        for pattern in BLOCKED_COMMAND_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return f"Blocked: potentially dangerous command: {command}"

        cwd = self.scope.root
        if working_directory:
            cwd = self.scope.resolve_path(working_directory)
            if not cwd.is_dir():
                return f"Not a directory: {working_directory}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                timeout=COMMAND_TIMEOUT,
                capture_output=True,
                text=True,
            )

            output_parts: list[str] = []

            if result.stdout:
                stdout = result.stdout[:MAX_COMMAND_OUTPUT]
                if len(result.stdout) > MAX_COMMAND_OUTPUT:
                    stdout += "\n... (stdout truncated at 10KB)"
                output_parts.append(stdout)

            if result.stderr:
                stderr = result.stderr[:MAX_COMMAND_OUTPUT]
                if len(result.stderr) > MAX_COMMAND_OUTPUT:
                    stderr += "\n... (stderr truncated at 10KB)"
                output_parts.append(f"STDERR:\n{stderr}")

            if result.returncode != 0:
                output_parts.append(f"\nExit code: {result.returncode}")

            return "\n".join(output_parts) if output_parts else "(no output)"

        except subprocess.TimeoutExpired:
            return f"Command timed out after {COMMAND_TIMEOUT}s: {command}"
        except Exception as e:
            return f"Command failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_project_tools(
    agent: "BaseAgent",
) -> Optional[tuple[list[dict[str, Any]], ProjectToolExecutor]]:
    """Create project-scoped tools for an enhanced agent.

    Returns ``None`` if the agent doesn't have ``enhanced_tools`` enabled
    or lacks a ``root_path``.
    """
    if not agent.spec.metadata.get("enhanced_tools"):
        return None

    root_path: Optional[str] = getattr(agent, "root_path", None)
    if root_path is None:
        root_path = agent.spec.metadata.get("root_path")
    if root_path is None:
        return None

    scope = ProjectScope(root_path)
    executor = ProjectToolExecutor(scope)
    return ENHANCED_TOOL_DEFINITIONS, executor
