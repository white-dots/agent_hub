"""Tests for domain-scoped agent tools."""

import os
import tempfile
from pathlib import Path

import pytest

from agenthub.agents.domain_tools import (
    DomainScope,
    DomainToolExecutor,
    create_domain_tools,
    get_tool_definitions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_dir():
    """Create a temporary project directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()  # resolve symlinks (macOS /var → /private/var)

        # Create a realistic project structure
        (root / "src" / "pricing").mkdir(parents=True)
        (root / "src" / "pricing" / "engine.py").write_text(
            "class PricingEngine:\n"
            "    def calculate_price(self, sku_id: str) -> float:\n"
            "        \"\"\"Calculate price for a SKU.\"\"\"\n"
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
            "    \"\"\"Authenticate a user.\"\"\"\n"
            "    return True\n"
        )
        (root / "src" / "auth" / "__init__.py").write_text("")

        (root / "src" / "utils").mkdir(parents=True)
        (root / "src" / "utils" / "helpers.py").write_text(
            "def format_currency(amount: float) -> str:\n"
            "    return f'${amount:.2f}'\n"
        )

        yield root


# ---------------------------------------------------------------------------
# DomainScope tests
# ---------------------------------------------------------------------------

class TestDomainScope:
    def test_compute_allowed_dirs_from_files(self, project_dir):
        scope = DomainScope(
            str(project_dir),
            ["src/pricing/engine.py", "src/pricing/models.py"],
        )
        # Should derive src/pricing as the allowed dir
        assert len(scope.allowed_dirs) == 1
        assert scope.allowed_dirs[0] == project_dir / "src" / "pricing"

    def test_compute_allowed_dirs_multiple_domains(self, project_dir):
        scope = DomainScope(
            str(project_dir),
            ["src/pricing/engine.py", "src/auth/login.py"],
        )
        assert len(scope.allowed_dirs) == 2
        dirs = {str(d) for d in scope.allowed_dirs}
        assert str(project_dir / "src" / "pricing") in dirs
        assert str(project_dir / "src" / "auth") in dirs

    def test_compute_allowed_dirs_dedup_prefixes(self, project_dir):
        scope = DomainScope(
            str(project_dir),
            [
                "src/pricing/engine.py",
                "src/pricing/models.py",
                "src/pricing/__init__.py",
            ],
        )
        # All in src/pricing → single allowed dir
        assert len(scope.allowed_dirs) == 1

    def test_compute_allowed_dirs_glob_pattern(self, project_dir):
        scope = DomainScope(str(project_dir), ["src/**/*.py"])
        # "src" prefix → allowed_dirs = {root/src}
        assert len(scope.allowed_dirs) == 1
        assert scope.allowed_dirs[0] == project_dir / "src"

    def test_compute_allowed_dirs_root_glob(self, project_dir):
        scope = DomainScope(str(project_dir), ["**/*.py"])
        # Empty prefix after glob strip → root
        assert len(scope.allowed_dirs) == 1
        assert scope.allowed_dirs[0] == project_dir

    def test_is_allowed_within_domain(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        assert scope.is_allowed("src/pricing/engine.py")
        assert scope.is_allowed("src/pricing/models.py")
        assert scope.is_allowed("src/pricing/__init__.py")

    def test_is_allowed_outside_domain(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        assert not scope.is_allowed("src/auth/login.py")
        assert not scope.is_allowed("src/utils/helpers.py")

    def test_path_traversal_blocked(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        with pytest.raises(ValueError, match="traversal"):
            scope.resolve_path("src/pricing/../../etc/passwd")

    def test_outside_root_blocked(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        # Even if it doesn't use .., an absolute path is still blocked
        assert not scope.is_allowed("/etc/passwd")


# ---------------------------------------------------------------------------
# DomainToolExecutor tests
# ---------------------------------------------------------------------------

class TestDomainToolExecutor:
    def test_grep_finds_matches(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute("grep_domain", {"pattern": "calculate_price"})
        assert "engine.py" in result
        assert "calculate_price" in result

    def test_grep_with_file_glob(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py", "src/pricing/models.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute(
            "grep_domain",
            {"pattern": "class", "file_glob": "*.py"},
        )
        assert "PricingEngine" in result
        assert "PriceRule" in result

    def test_grep_no_matches(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute("grep_domain", {"pattern": "nonexistent_xyz"})
        assert "No matches" in result

    def test_grep_respects_domain_scope(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute("grep_domain", {"pattern": "authenticate"})
        # Should NOT find auth/login.py content
        assert "No matches" in result

    def test_read_file_success(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute("read_file", {"path": "src/pricing/engine.py"})
        assert "PricingEngine" in result
        assert "calculate_price" in result
        # Should have line numbers
        assert "1\t" in result or "    1\t" in result

    def test_read_file_outside_domain(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute("read_file", {"path": "src/auth/login.py"})
        assert "Access denied" in result

    def test_read_file_with_offset(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute(
            "read_file", {"path": "src/pricing/engine.py", "offset": 3, "limit": 2}
        )
        # Should start from line 3
        assert "lines 3-4" in result

    def test_list_files(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute("list_files", {"directory": "src/pricing"})
        assert "engine.py" in result
        assert "models.py" in result
        assert "__init__.py" in result

    def test_list_files_outside_domain(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute("list_files", {"directory": "src/auth"})
        assert "Access denied" in result

    def test_list_files_with_pattern(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute(
            "list_files", {"directory": "src/pricing", "pattern": "*.py"}
        )
        assert "engine.py" in result

    def test_unknown_tool(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute("unknown_tool", {})
        assert "Unknown tool" in result

    def test_invalid_regex(self, project_dir):
        scope = DomainScope(
            str(project_dir), ["src/pricing/engine.py"]
        )
        executor = DomainToolExecutor(scope)

        result = executor.execute("grep_domain", {"pattern": "[invalid"})
        assert "Invalid regex" in result


# ---------------------------------------------------------------------------
# Tool definitions test
# ---------------------------------------------------------------------------

def test_get_tool_definitions():
    defs = get_tool_definitions()
    assert len(defs) == 3
    names = {d["name"] for d in defs}
    assert names == {"grep_domain", "read_file", "list_files"}
    for d in defs:
        assert "description" in d
        assert "input_schema" in d


# ---------------------------------------------------------------------------
# Factory function test
# ---------------------------------------------------------------------------

class TestCreateDomainTools:
    def test_returns_none_without_root_path(self):
        """Agent without root_path should return None (fallback to no tools)."""
        from unittest.mock import MagicMock

        agent = MagicMock()
        agent.root_path = None  # No attribute
        del agent.root_path  # Remove so getattr returns None
        agent.project_root = None  # Also remove project_root fallback
        del agent.project_root
        agent.spec.metadata = {}
        agent.spec.context_paths = ["src/pricing/engine.py"]

        result = create_domain_tools(agent)
        assert result is None

    def test_returns_none_without_context_paths(self, project_dir):
        from unittest.mock import MagicMock

        agent = MagicMock()
        agent.root_path = str(project_dir)
        agent.spec.context_paths = []
        agent.spec.metadata = {"root_path": str(project_dir)}

        result = create_domain_tools(agent)
        assert result is None

    def test_returns_tools_with_valid_agent(self, project_dir):
        from unittest.mock import MagicMock

        agent = MagicMock()
        agent.root_path = str(project_dir)
        agent.spec.context_paths = ["src/pricing/engine.py"]
        agent.spec.metadata = {"root_path": str(project_dir)}

        result = create_domain_tools(agent)
        assert result is not None
        tool_defs, executor = result
        assert len(tool_defs) == 3
        assert isinstance(executor, DomainToolExecutor)
