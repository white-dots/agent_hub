"""Codebase analyzer for auto-agent generation."""

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agenthub.auto.config import AutoAgentConfig


@dataclass
class FolderStats:
    """Statistics for a single folder."""

    path: Path
    total_size_kb: float
    file_count: int
    subfolder_count: int
    languages: dict[str, int] = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        """Check if folder has significant content."""
        return self.file_count > 0 and self.total_size_kb > 0


@dataclass
class AgentBoundary:
    """Proposed agent boundary."""

    agent_id: str
    root_path: Path
    include_patterns: list[str]
    estimated_context_kb: float
    file_count: int


class CodebaseAnalyzer:
    """Analyzes codebase structure to determine agent boundaries.

    Walks the directory tree, collects statistics, and proposes
    agent boundaries based on configurable thresholds.

    Example:
        >>> analyzer = CodebaseAnalyzer("./my-project", config)
        >>> boundaries = analyzer.propose_boundaries()
        >>> for b in boundaries:
        ...     print(f"{b.agent_id}: {b.file_count} files")
    """

    # Map file extensions to language names
    EXTENSION_MAP = {
        ".py": "python",
        ".sql": "sql",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
    }

    def __init__(self, root_path: str, config: Optional[AutoAgentConfig] = None):
        """Initialize CodebaseAnalyzer.

        Args:
            root_path: Root directory of the project to analyze.
            config: Configuration for analysis. Uses defaults if None.
        """
        self.root = Path(root_path).resolve()
        self.config = config or AutoAgentConfig()

    def analyze(self) -> list[FolderStats]:
        """Walk the codebase and collect statistics.

        Returns:
            List of FolderStats for each significant folder.
        """
        stats: list[FolderStats] = []

        for folder in self._walk_folders():
            folder_stats = self._analyze_folder(folder)
            if folder_stats.is_significant:
                stats.append(folder_stats)

        return stats

    def propose_boundaries(self) -> list[AgentBoundary]:
        """Propose agent boundaries based on thresholds.

        Returns:
            List of AgentBoundary defining proposed agents.
        """
        stats = self.analyze()
        boundaries: list[AgentBoundary] = []
        processed_paths: set[Path] = set()

        # Sort by path depth (process shallow folders first)
        stats.sort(key=lambda s: len(s.path.parts))

        for folder_stat in stats:
            # Skip if already covered by a parent boundary
            if self._is_covered(folder_stat.path, processed_paths):
                continue

            if self._should_create_agent(folder_stat):
                # Check if needs splitting
                if self._should_split(folder_stat):
                    sub_boundaries = self._split_folder(folder_stat)
                    boundaries.extend(sub_boundaries)
                    for b in sub_boundaries:
                        processed_paths.add(b.root_path)
                else:
                    boundary = self._create_boundary(folder_stat)
                    boundaries.append(boundary)
                    processed_paths.add(folder_stat.path)

        return boundaries

    def _walk_folders(self):
        """Generator yielding folders up to max_depth.

        Yields:
            Path objects for each folder within depth limit.
        """

        def should_ignore(name: str) -> bool:
            return any(
                fnmatch.fnmatch(name, pattern)
                for pattern in self.config.ignore_patterns
            )

        def walk(path: Path, depth: int):
            if depth > self.config.max_depth:
                return

            if should_ignore(path.name):
                return

            yield path

            try:
                for child in path.iterdir():
                    if child.is_dir() and not should_ignore(child.name):
                        yield from walk(child, depth + 1)
            except PermissionError:
                pass

        yield from walk(self.root, 0)

    def _analyze_folder(self, path: Path) -> FolderStats:
        """Collect statistics for a single folder.

        Args:
            path: Folder path to analyze.

        Returns:
            FolderStats for the folder.
        """
        total_size = 0
        file_count = 0
        subfolder_count = 0
        languages: dict[str, int] = {}

        def should_ignore(name: str) -> bool:
            return any(
                fnmatch.fnmatch(name, pattern)
                for pattern in self.config.ignore_patterns
            )

        try:
            for child in path.iterdir():
                if should_ignore(child.name):
                    continue

                if child.is_file():
                    ext = child.suffix.lower()
                    if ext in self.config.include_extensions:
                        try:
                            size = child.stat().st_size
                            total_size += size
                            file_count += 1

                            lang = self.EXTENSION_MAP.get(ext, ext)
                            languages[lang] = languages.get(lang, 0) + 1
                        except OSError:
                            pass
                elif child.is_dir():
                    subfolder_count += 1

        except PermissionError:
            pass

        return FolderStats(
            path=path,
            total_size_kb=total_size / 1024,
            file_count=file_count,
            subfolder_count=subfolder_count,
            languages=languages,
        )

    def _should_create_agent(self, stats: FolderStats) -> bool:
        """Determine if folder warrants its own agent.

        Args:
            stats: Folder statistics.

        Returns:
            True if folder should have an agent.
        """
        return (
            stats.total_size_kb >= self.config.min_folder_size_kb
            and stats.file_count >= self.config.min_files_per_folder
        )

    def _should_split(self, stats: FolderStats) -> bool:
        """Determine if folder should be split into multiple agents.

        Args:
            stats: Folder statistics.

        Returns:
            True if folder should be split.
        """
        return (
            stats.total_size_kb > self.config.max_agent_context_kb
            or stats.file_count > self.config.max_files_per_agent
        )

    def _is_covered(self, path: Path, processed: set[Path]) -> bool:
        """Check if path is already covered by a processed boundary.

        Args:
            path: Path to check.
            processed: Set of already processed paths.

        Returns:
            True if path is covered.
        """
        for parent in path.parents:
            if parent in processed:
                return True
        return False

    def _split_folder(self, stats: FolderStats) -> list[AgentBoundary]:
        """Split large folder into multiple agent boundaries.

        Args:
            stats: Folder statistics.

        Returns:
            List of boundaries for subfolders.
        """
        boundaries: list[AgentBoundary] = []

        try:
            for child in stats.path.iterdir():
                if child.is_dir():
                    child_stats = self._analyze_folder(child)
                    if self._should_create_agent(child_stats):
                        boundaries.append(self._create_boundary(child_stats))
        except PermissionError:
            pass

        # If no significant subfolders, create one for the whole folder
        if not boundaries:
            boundaries.append(self._create_boundary(stats))

        return boundaries

    def _create_boundary(self, stats: FolderStats) -> AgentBoundary:
        """Create agent boundary from folder stats.

        Args:
            stats: Folder statistics.

        Returns:
            AgentBoundary for the folder.
        """
        # Determine include patterns based on languages
        patterns: list[str] = []
        for ext in self.config.include_extensions:
            patterns.append(f"**/*{ext}")

        return AgentBoundary(
            agent_id=self._generate_agent_id(stats.path),
            root_path=stats.path,
            include_patterns=patterns,
            estimated_context_kb=stats.total_size_kb,
            file_count=stats.file_count,
        )

    def _generate_agent_id(self, path: Path) -> str:
        """Generate agent ID from path: src/api -> src_api_agent.

        Args:
            path: Folder path.

        Returns:
            Generated agent ID.
        """
        try:
            relative = path.relative_to(self.root)
            parts = [p for p in relative.parts if p]
            if parts:
                return "_".join(parts) + "_agent"
            return "root_agent"
        except ValueError:
            return path.name + "_agent"
