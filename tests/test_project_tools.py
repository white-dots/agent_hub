"""Tests for project-scoped agent tools (enhanced Tier B)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agenthub.agents.project_tools import (
    BLOCKED_COMMAND_PATTERNS,
    ENHANCED_TOOL_DEFINITIONS,
    PendingChange,
    ProjectScope,
    ProjectToolExecutor,
    TechStackDetector,
    create_project_tools,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_dir():
    """Create a temporary project directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()

        # Create a realistic project structure
        (root / "src" / "pricing").mkdir(parents=True)
        (root / "src" / "pricing" / "engine.py").write_text(
            "class PricingEngine:\n"
            "    def calculate_price(self, sku_id: str) -> float:\n"
            '        """Calculate price for a SKU."""\n'
            "        return 100.0\n"
            "\n"
            "    def apply_discount(self, price: float, rate: float) -> float:\n"
            "        return price * (1 - rate)\n"
        )
        (root / "src" / "pricing" / "models.py").write_text(
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class PriceRule:\n"
            "    name: str\n"
            "    discount_rate: float\n"
            "    min_quantity: int = 1\n"
        )
        (root / "src" / "pricing" / "__init__.py").write_text("")

        (root / "src" / "auth").mkdir(parents=True)
        (root / "src" / "auth" / "login.py").write_text(
            "def authenticate(username: str, password: str) -> bool:\n"
            '    """Authenticate a user."""\n'
            "    return True\n"
        )
        (root / "src" / "auth" / "__init__.py").write_text("")

        (root / "src" / "utils").mkdir(parents=True)
        (root / "src" / "utils" / "helpers.py").write_text(
            "def format_currency(amount: float) -> str:\n"
            "    return f'${amount:.2f}'\n"
        )

        # Add some marker files for tech stack detection
        (root / "pyproject.toml").write_text(
            '[project]\nname = "test"\n\n[project.dependencies]\nfastapi = ">=0.100"\npytest = ">=8.0"\n'
        )

        yield root


# ---------------------------------------------------------------------------
# ProjectScope tests
# ---------------------------------------------------------------------------

class TestProjectScope:
    def test_allows_any_path_under_root(self, project_dir):
        scope = ProjectScope(str(project_dir))
        assert scope.is_allowed("src/pricing/engine.py")
        assert scope.is_allowed("src/auth/login.py")
        assert scope.is_allowed("src/utils/helpers.py")
        assert scope.is_allowed("pyproject.toml")

    def test_blocks_path_traversal(self, project_dir):
        scope = ProjectScope(str(project_dir))
        with pytest.raises(ValueError, match="traversal"):
            scope.resolve_path("src/../../../etc/passwd")

    def test_blocks_outside_root(self, project_dir):
        scope = ProjectScope(str(project_dir))
        assert not scope.is_allowed("/etc/passwd")

    def test_resolve_path_returns_absolute(self, project_dir):
        scope = ProjectScope(str(project_dir))
        resolved = scope.resolve_path("src/pricing/engine.py")
        assert resolved.is_absolute()
        assert resolved == project_dir / "src" / "pricing" / "engine.py"


# ---------------------------------------------------------------------------
# ProjectToolExecutor tests
# ---------------------------------------------------------------------------

class TestProjectToolExecutor:
    def test_list_directory(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("list_directory", {"directory": "src"})
        assert "pricing" in result
        assert "auth" in result
        assert "utils" in result

    def test_list_directory_with_depth(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("list_directory", {"directory": ".", "max_depth": 1})
        assert "src/" in result
        assert "pyproject.toml" in result

    def test_list_directory_excludes_git(self, project_dir):
        # Create a .git dir
        (project_dir / ".git").mkdir()
        (project_dir / ".git" / "HEAD").write_text("ref: refs/heads/main")

        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("list_directory", {"directory": "."})
        assert ".git" not in result

    def test_read_file_success(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("read_file", {"path": "src/pricing/engine.py"})
        assert "PricingEngine" in result
        assert "calculate_price" in result

    def test_read_file_tracks_reads(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        executor.execute("read_file", {"path": "src/pricing/engine.py"})
        assert "src/pricing/engine.py" in executor._files_read

    def test_read_file_cross_domain(self, project_dir):
        """Enhanced tools can read ANY file under root (not domain-locked)."""
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        # Can read auth files even though this is a "pricing" agent
        result = executor.execute("read_file", {"path": "src/auth/login.py"})
        assert "authenticate" in result

    def test_read_file_with_offset(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute(
            "read_file", {"path": "src/pricing/engine.py", "offset": 3, "limit": 2}
        )
        assert "lines 3-4" in result

    def test_read_file_not_found(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("read_file", {"path": "nonexistent.py"})
        assert "File not found" in result

    def test_search_files_finds_matches(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("search_files", {"pattern": "calculate_price"})
        assert "engine.py" in result
        assert "calculate_price" in result

    def test_search_files_project_wide(self, project_dir):
        """Search should span the entire project, not just one domain."""
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("search_files", {"pattern": "class"})
        assert "PricingEngine" in result
        assert "PriceRule" in result

    def test_search_files_with_glob(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute(
            "search_files", {"pattern": "def", "file_glob": "*.py"}
        )
        assert ".py" in result

    def test_search_files_in_directory(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute(
            "search_files", {"pattern": "authenticate", "directory": "src/auth"}
        )
        assert "login.py" in result

    def test_search_files_no_matches(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("search_files", {"pattern": "zzz_nonexistent_zzz"})
        assert "No matches" in result

    def test_search_files_invalid_regex(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("search_files", {"pattern": "[invalid"})
        assert "Invalid regex" in result

    def test_unknown_tool(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("bad_tool", {})
        assert "Unknown tool" in result


# ---------------------------------------------------------------------------
# write_file tests
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_write_file_records_change(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("write_file", {
            "path": "src/pricing/engine.py",
            "content": "# Modified\nclass PricingEngine: pass\n",
            "description": "Simplified pricing engine",
        })
        assert "Change recorded" in result
        assert len(executor.get_pending_changes()) == 1

    def test_write_file_does_not_modify_disk(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        original = (project_dir / "src" / "pricing" / "engine.py").read_text()

        executor.execute("write_file", {
            "path": "src/pricing/engine.py",
            "content": "COMPLETELY NEW CONTENT",
            "description": "test",
        })

        # File on disk should be unchanged
        assert (project_dir / "src" / "pricing" / "engine.py").read_text() == original

    def test_write_file_generates_diff(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        # First read the file
        executor.execute("read_file", {"path": "src/pricing/engine.py"})

        result = executor.execute("write_file", {
            "path": "src/pricing/engine.py",
            "content": "class PricingEngine:\n    pass\n",
            "description": "Simplified",
        })
        assert "Diff preview" in result

        changes = executor.get_pending_changes()
        assert len(changes) == 1
        assert changes[0].unified_diff  # Should have a diff

    def test_write_file_auto_reads_if_not_read(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        # Don't read first — write_file should auto-read
        executor.execute("write_file", {
            "path": "src/pricing/engine.py",
            "content": "new content",
            "description": "test",
        })

        # Should have auto-read the file
        assert "src/pricing/engine.py" in executor._files_read

    def test_write_new_file(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("write_file", {
            "path": "src/pricing/new_module.py",
            "content": "# New module\ndef new_func(): pass\n",
            "description": "Add new module",
        })
        assert "Change recorded" in result
        assert "new file" in result

    def test_write_file_path_traversal_blocked(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("write_file", {
            "path": "../../../etc/passwd",
            "content": "hacked",
            "description": "hack",
        })
        assert "Invalid input" in result or "Access denied" in result


# ---------------------------------------------------------------------------
# run_command tests
# ---------------------------------------------------------------------------

class TestRunCommand:
    def test_run_simple_command(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("run_command", {"command": "echo hello"})
        assert "hello" in result

    def test_run_command_with_working_dir(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("run_command", {
            "command": "ls",
            "working_directory": "src/pricing",
        })
        assert "engine.py" in result

    def test_run_command_blocks_dangerous(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("run_command", {"command": "rm -rf /"})
        assert "Blocked" in result

    def test_run_command_blocks_dd(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("run_command", {"command": "dd if=/dev/zero of=/dev/sda"})
        assert "Blocked" in result

    def test_run_command_blocks_shutdown(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("run_command", {"command": "shutdown -h now"})
        assert "Blocked" in result

    def test_run_command_allows_safe_commands(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("run_command", {"command": "python --version"})
        assert "Python" in result or "python" in result.lower()

    def test_run_command_captures_stderr(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("run_command", {
            "command": "python -c 'import sys; sys.stderr.write(\"err\\n\")'"
        })
        assert "STDERR" in result
        assert "err" in result

    def test_run_command_exit_code(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        result = executor.execute("run_command", {"command": "false"})
        assert "Exit code" in result


# ---------------------------------------------------------------------------
# TechStackDetector tests
# ---------------------------------------------------------------------------

class TestTechStackDetector:
    def test_detects_python(self, project_dir):
        result = TechStackDetector.detect(str(project_dir))
        assert "python" in result["languages"]

    def test_detects_fastapi(self, project_dir):
        result = TechStackDetector.detect(str(project_dir))
        assert "fastapi" in result["frameworks"]

    def test_detects_pytest(self, project_dir):
        result = TechStackDetector.detect(str(project_dir))
        assert "pytest" in result["test_frameworks"]

    def test_detects_node_project(self, tmp_path):
        (tmp_path / "package.json").write_text(
            '{"name": "test", "dependencies": {"react": "^18.0", "express": "^4.0"}}'
        )
        result = TechStackDetector.detect(str(tmp_path))
        assert "javascript" in result["languages"]
        assert "react" in result["frameworks"]
        assert "express" in result["frameworks"]

    def test_detects_rust_project(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n')
        result = TechStackDetector.detect(str(tmp_path))
        assert "rust" in result["languages"]
        assert "cargo" in result["build_tools"]

    def test_empty_project(self, tmp_path):
        result = TechStackDetector.detect(str(tmp_path))
        assert result["languages"] == []
        assert result["frameworks"] == []


# ---------------------------------------------------------------------------
# get_summary tests
# ---------------------------------------------------------------------------

class TestExecutorSummary:
    def test_empty_summary(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        summary = executor.get_summary()
        assert "No files explored" in summary

    def test_summary_with_reads(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        executor.execute("read_file", {"path": "src/pricing/engine.py"})
        summary = executor.get_summary()
        assert "Files explored" in summary
        assert "src/pricing/engine.py" in summary

    def test_summary_with_changes(self, project_dir):
        scope = ProjectScope(str(project_dir))
        executor = ProjectToolExecutor(scope)

        executor.execute("write_file", {
            "path": "src/pricing/engine.py",
            "content": "new",
            "description": "test change",
        })
        summary = executor.get_summary()
        assert "Pending changes" in summary
        assert "test change" in summary


# ---------------------------------------------------------------------------
# Tool definitions tests
# ---------------------------------------------------------------------------

def test_enhanced_tool_definitions():
    assert len(ENHANCED_TOOL_DEFINITIONS) == 5
    names = {d["name"] for d in ENHANCED_TOOL_DEFINITIONS}
    assert names == {"list_directory", "read_file", "search_files", "write_file", "run_command"}
    for d in ENHANCED_TOOL_DEFINITIONS:
        assert "description" in d
        assert "input_schema" in d


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------

class TestCreateProjectTools:
    def test_returns_none_without_enhanced_flag(self):
        agent = MagicMock()
        agent.spec.metadata = {"root_path": "/tmp/test"}
        result = create_project_tools(agent)
        assert result is None

    def test_returns_none_without_root_path(self):
        agent = MagicMock()
        agent.root_path = None
        del agent.root_path
        agent.spec.metadata = {"enhanced_tools": True}
        result = create_project_tools(agent)
        assert result is None

    def test_returns_tools_with_valid_config(self, project_dir):
        agent = MagicMock()
        agent.root_path = str(project_dir)
        agent.spec.metadata = {
            "enhanced_tools": True,
            "root_path": str(project_dir),
        }
        result = create_project_tools(agent)
        assert result is not None
        tool_defs, executor = result
        assert len(tool_defs) == 5
        assert isinstance(executor, ProjectToolExecutor)
