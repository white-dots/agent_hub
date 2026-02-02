"""Tests for codebase analyzer."""

import pytest

from agenthub.auto.analyzer import CodebaseAnalyzer, FolderStats
from agenthub.auto.config import AutoAgentConfig, Presets


class TestCodebaseAnalyzer:
    """Tests for CodebaseAnalyzer."""

    def test_analyze_project(self, tmp_project):
        """Test analyzing a project structure."""
        config = AutoAgentConfig(
            min_folder_size_kb=0,  # Allow any size for testing
            min_files_per_folder=1,
        )
        analyzer = CodebaseAnalyzer(str(tmp_project), config)

        stats = analyzer.analyze()

        assert len(stats) > 0
        # Should find src, src/api, src/models, tests folders
        folder_names = [s.path.name for s in stats]
        assert "api" in folder_names
        assert "models" in folder_names

    def test_analyze_folder_stats(self, tmp_project):
        """Test folder statistics are calculated correctly."""
        config = AutoAgentConfig(min_folder_size_kb=0, min_files_per_folder=1)
        analyzer = CodebaseAnalyzer(str(tmp_project), config)

        stats = analyzer.analyze()

        # Find the api folder stats
        api_stats = next((s for s in stats if s.path.name == "api"), None)
        assert api_stats is not None
        assert api_stats.file_count >= 2  # __init__.py, routes.py, auth.py
        assert api_stats.total_size_kb > 0
        assert "python" in api_stats.languages

    def test_propose_boundaries(self, tmp_project):
        """Test proposing agent boundaries."""
        config = AutoAgentConfig(
            min_folder_size_kb=0,
            min_files_per_folder=1,
            max_agent_context_kb=1000,  # Don't split
        )
        analyzer = CodebaseAnalyzer(str(tmp_project), config)

        boundaries = analyzer.propose_boundaries()

        assert len(boundaries) > 0
        # Each boundary should have an agent_id ending in _agent
        for b in boundaries:
            assert b.agent_id.endswith("_agent")
            assert b.file_count > 0

    def test_ignore_patterns(self, tmp_project):
        """Test ignore patterns are respected."""
        # Create a __pycache__ folder
        pycache = tmp_project / "src" / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_bytes(b"fake bytecode")

        config = AutoAgentConfig(
            min_folder_size_kb=0,
            min_files_per_folder=1,
            ignore_patterns=["__pycache__", "*.pyc"],
        )
        analyzer = CodebaseAnalyzer(str(tmp_project), config)

        stats = analyzer.analyze()

        # __pycache__ should not be in stats
        folder_names = [s.path.name for s in stats]
        assert "__pycache__" not in folder_names

    def test_generate_agent_id(self, tmp_project):
        """Test agent ID generation from path."""
        config = AutoAgentConfig()
        analyzer = CodebaseAnalyzer(str(tmp_project), config)

        api_path = tmp_project / "src" / "api"
        agent_id = analyzer._generate_agent_id(api_path)

        assert agent_id == "src_api_agent"

    def test_max_depth_respected(self, tmp_project):
        """Test max_depth limits folder traversal."""
        # Create deep nested structure
        deep = tmp_project / "a" / "b" / "c" / "d" / "e"
        deep.mkdir(parents=True)
        (deep / "file.py").write_text("# deep")

        config = AutoAgentConfig(
            min_folder_size_kb=0,
            min_files_per_folder=1,
            max_depth=2,  # Only go 2 levels deep
        )
        analyzer = CodebaseAnalyzer(str(tmp_project), config)

        stats = analyzer.analyze()

        # Should not find the deeply nested folder
        folder_names = [s.path.name for s in stats]
        assert "e" not in folder_names


class TestFolderStats:
    """Tests for FolderStats dataclass."""

    def test_is_significant(self, tmp_path):
        """Test is_significant property."""
        significant = FolderStats(
            path=tmp_path,
            total_size_kb=10,
            file_count=5,
            subfolder_count=0,
            languages={"python": 5},
        )
        assert significant.is_significant is True

        empty = FolderStats(
            path=tmp_path,
            total_size_kb=0,
            file_count=0,
            subfolder_count=0,
        )
        assert empty.is_significant is False


class TestPresets:
    """Tests for configuration presets."""

    def test_small_project_preset(self):
        """Test small project preset."""
        config = Presets.small_project()
        assert config.min_folder_size_kb == 100
        assert config.max_depth == 2

    def test_medium_project_preset(self):
        """Test medium project preset."""
        config = Presets.medium_project()
        assert config.min_folder_size_kb == 50
        assert config.max_depth == 3

    def test_large_project_preset(self):
        """Test large project preset."""
        config = Presets.large_project()
        assert config.min_folder_size_kb == 30
        assert config.max_depth == 4

    def test_monorepo_preset(self):
        """Test monorepo preset."""
        config = Presets.monorepo()
        assert config.max_depth == 5
        assert "node_modules" in config.ignore_patterns
