"""Context building utilities for agents."""

import fnmatch
from pathlib import Path
from typing import Optional


class ContextBuilder:
    """Utility for building agent context from files."""

    def __init__(self, base_path: str = "."):
        """Initialize ContextBuilder.

        Args:
            base_path: Base directory for file operations.
        """
        self.base_path = Path(base_path).resolve()

    def read_files(
        self,
        patterns: list[str],
        max_size: int = 50000,
        encoding: str = "utf-8",
    ) -> str:
        """Read files matching patterns into context string.

        Args:
            patterns: Glob patterns like ["src/**/*.py", "*.md"]
            max_size: Maximum total characters
            encoding: File encoding to use

        Returns:
            Formatted context string with file contents
        """
        content_parts: list[str] = []
        total_size = 0

        for pattern in patterns:
            for file_path in sorted(self.base_path.glob(pattern)):
                if file_path.is_file():
                    try:
                        text = file_path.read_text(encoding=encoding)

                        # Check size limit
                        if total_size + len(text) > max_size:
                            remaining = max_size - total_size
                            if remaining > 1000:  # Worth including partial
                                text = text[:remaining] + "\n... [truncated]"
                            else:
                                continue

                        relative_path = file_path.relative_to(self.base_path)
                        content_parts.append(f"### {relative_path}\n```\n{text}\n```")
                        total_size += len(text)

                        if total_size >= max_size:
                            break

                    except Exception as e:
                        print(f"Warning: Could not read {file_path}: {e}")

            if total_size >= max_size:
                break

        return "\n\n".join(content_parts)

    def read_directory_structure(
        self,
        path: str = ".",
        max_depth: int = 3,
        ignore: Optional[list[str]] = None,
    ) -> str:
        """Generate directory tree for context.

        Args:
            path: Starting directory relative to base_path
            max_depth: How deep to traverse
            ignore: Patterns to ignore (default: common dev files)

        Returns:
            Tree-formatted string
        """
        if ignore is None:
            ignore = [
                "__pycache__",
                ".git",
                "node_modules",
                ".venv",
                "venv",
                ".pytest_cache",
                "*.pyc",
                ".mypy_cache",
                ".ruff_cache",
                "dist",
                "build",
                "*.egg-info",
            ]

        def should_ignore(name: str) -> bool:
            return any(fnmatch.fnmatch(name, pattern) for pattern in ignore)

        def build_tree(dir_path: Path, prefix: str = "", depth: int = 0) -> list[str]:
            if depth > max_depth:
                return []

            lines: list[str] = []
            try:
                entries = sorted(
                    dir_path.iterdir(),
                    key=lambda e: (e.is_file(), e.name.lower()),
                )
                entries = [e for e in entries if not should_ignore(e.name)]

                for i, entry in enumerate(entries):
                    is_last = i == len(entries) - 1
                    connector = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{connector}{entry.name}")

                    if entry.is_dir():
                        extension = "    " if is_last else "│   "
                        lines.extend(build_tree(entry, prefix + extension, depth + 1))
            except PermissionError:
                pass

            return lines

        start_path = self.base_path / path
        tree_lines = build_tree(start_path)
        return "\n".join(tree_lines)

    def get_file_summary(
        self,
        patterns: list[str],
        ignore: Optional[list[str]] = None,
    ) -> dict[str, int]:
        """Get summary of files matching patterns.

        Args:
            patterns: Glob patterns to match
            ignore: Patterns to ignore

        Returns:
            Dict with file counts by extension
        """
        if ignore is None:
            ignore = ["__pycache__", ".git", "node_modules", ".venv"]

        def should_ignore(path: Path) -> bool:
            return any(part in ignore for part in path.parts)

        summary: dict[str, int] = {}

        for pattern in patterns:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file() and not should_ignore(file_path):
                    ext = file_path.suffix or "no_extension"
                    summary[ext] = summary.get(ext, 0) + 1

        return summary

    def calculate_size(
        self,
        patterns: list[str],
        ignore: Optional[list[str]] = None,
    ) -> int:
        """Calculate total size of files matching patterns.

        Args:
            patterns: Glob patterns to match
            ignore: Patterns to ignore

        Returns:
            Total size in bytes
        """
        if ignore is None:
            ignore = ["__pycache__", ".git", "node_modules", ".venv"]

        def should_ignore(path: Path) -> bool:
            return any(part in ignore for part in path.parts)

        total = 0
        for pattern in patterns:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file() and not should_ignore(file_path):
                    try:
                        total += file_path.stat().st_size
                    except OSError:
                        pass

        return total


class FileContext:
    """Helper for building context from specific files."""

    def __init__(self, files: list[str], base_path: str = "."):
        """Initialize FileContext.

        Args:
            files: List of file paths to include
            base_path: Base directory for relative paths
        """
        self.files = files
        self.base_path = Path(base_path).resolve()

    def build(self, max_size: int = 50000) -> str:
        """Build context from specified files.

        Args:
            max_size: Maximum total characters

        Returns:
            Formatted context string
        """
        parts: list[str] = []
        total_size = 0

        for file_path_str in self.files:
            file_path = self.base_path / file_path_str
            if file_path.exists() and file_path.is_file():
                try:
                    text = file_path.read_text(encoding="utf-8")
                    if total_size + len(text) > max_size:
                        remaining = max_size - total_size
                        if remaining > 500:
                            text = text[:remaining] + "\n... [truncated]"
                        else:
                            continue

                    parts.append(f"### {file_path_str}\n```\n{text}\n```")
                    total_size += len(text)
                except Exception:
                    pass

        return "\n\n".join(parts)


class SQLContext:
    """Helper for building context from database schemas."""

    def __init__(self, connection_string: str):
        """Initialize SQLContext.

        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string

    def build(self, tables: Optional[list[str]] = None) -> str:
        """Build context from database schema.

        Args:
            tables: Specific tables to include (None for all)

        Returns:
            Formatted schema string
        """
        try:
            import psycopg2

            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            # Get table schemas
            if tables:
                placeholders = ",".join(["%s"] * len(tables))
                table_filter = f"AND table_name IN ({placeholders})"
                params = tables
            else:
                table_filter = ""
                params = []

            cursor.execute(
                f"""
                SELECT table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' {table_filter}
                ORDER BY table_name, ordinal_position
                """,
                params,
            )

            schemas: dict[str, list[str]] = {}
            for table, column, dtype, nullable in cursor.fetchall():
                if table not in schemas:
                    schemas[table] = []
                null_str = "NULL" if nullable == "YES" else "NOT NULL"
                schemas[table].append(f"  {column}: {dtype} {null_str}")

            result = []
            for table, columns in schemas.items():
                result.append(f"Table: {table}\n" + "\n".join(columns))

            conn.close()
            return "\n\n".join(result)

        except ImportError:
            return "# Database schema extraction requires psycopg2"
        except Exception as e:
            return f"# Error extracting schema: {e}"
