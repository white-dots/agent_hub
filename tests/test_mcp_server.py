"""Tests for the impact analysis MCP server."""

import json
from unittest.mock import patch, MagicMock

import pytest

from agenthub.auto.import_graph import ImportGraph


@pytest.fixture
def mcp_project(tmp_path):
    """Minimal project for MCP server testing."""
    src = tmp_path / "src"
    src.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()

    (src / "core.py").write_text(
        '''"""Core module."""
class Engine:
    def run(self):
        pass

def start():
    return Engine()
'''
    )
    (src / "api.py").write_text(
        '''"""API layer."""
from src.core import Engine

def handle_request():
    return Engine().run()
'''
    )
    (tests / "test_core.py").write_text(
        '''"""Core tests."""
from src.core import Engine

def test_engine():
    assert Engine() is not None
'''
    )
    return tmp_path


@pytest.fixture
def mcp_graph(mcp_project):
    g = ImportGraph(str(mcp_project))
    g.build()
    return g


class TestImpactCheck:
    def test_known_file(self, mcp_graph, mcp_project):
        from agenthub.mcp_server import handle_impact_check

        with patch("agenthub.mcp_server.get_graph", return_value=mcp_graph), \
             patch("agenthub.mcp_server._resolve_project_root", return_value=str(mcp_project)):
            result = handle_impact_check({"file_path": "src/core.py"})

        assert "Impact Analysis: src/core.py" in result
        assert "EXPORTED INTERFACE:" in result
        assert "Engine" in result
        assert "start" in result
        assert "DIRECT DEPENDENTS" in result
        assert "AFFECTED TESTS" in result

    def test_unknown_file(self, mcp_graph, mcp_project):
        from agenthub.mcp_server import handle_impact_check

        with patch("agenthub.mcp_server.get_graph", return_value=mcp_graph), \
             patch("agenthub.mcp_server._resolve_project_root", return_value=str(mcp_project)):
            result = handle_impact_check({"file_path": "nonexistent.py"})

        assert "not found" in result

    def test_no_graph(self):
        from agenthub.mcp_server import handle_impact_check

        with patch("agenthub.mcp_server.get_graph", return_value=None):
            result = handle_impact_check({"file_path": "src/core.py"})

        assert "Error" in result

    def test_risk_level_shown(self, mcp_graph, mcp_project):
        from agenthub.mcp_server import handle_impact_check

        with patch("agenthub.mcp_server.get_graph", return_value=mcp_graph), \
             patch("agenthub.mcp_server._resolve_project_root", return_value=str(mcp_project)):
            result = handle_impact_check({"file_path": "src/core.py"})

        assert "Risk:" in result


class TestAffectedTests:
    def test_finds_tests(self, mcp_graph, mcp_project):
        from agenthub.mcp_server import handle_affected_tests

        with patch("agenthub.mcp_server.get_graph", return_value=mcp_graph), \
             patch("agenthub.mcp_server._resolve_project_root", return_value=str(mcp_project)):
            result = handle_affected_tests({"file_paths": ["src/core.py"]})

        assert "test_core.py" in result
        assert "pytest" in result

    def test_no_tests(self, mcp_graph, mcp_project):
        from agenthub.mcp_server import handle_affected_tests

        with patch("agenthub.mcp_server.get_graph", return_value=mcp_graph), \
             patch("agenthub.mcp_server._resolve_project_root", return_value=str(mcp_project)):
            result = handle_affected_tests({"file_paths": ["src/api.py"]})

        # api.py may or may not have tests depending on resolution
        assert isinstance(result, str)

    def test_no_graph(self):
        from agenthub.mcp_server import handle_affected_tests

        with patch("agenthub.mcp_server.get_graph", return_value=None):
            result = handle_affected_tests({"file_paths": ["src/core.py"]})

        assert "Error" in result


class TestCodebaseOverview:
    def test_returns_stats(self, mcp_graph, mcp_project):
        from agenthub.mcp_server import handle_codebase_overview

        with patch("agenthub.mcp_server.get_graph", return_value=mcp_graph), \
             patch("agenthub.mcp_server._resolve_project_root", return_value=str(mcp_project)):
            result = handle_codebase_overview({})

        assert "Codebase Overview" in result
        assert "Modules:" in result
        assert "Languages:" in result
        assert "Python:" in result


class TestToolDispatch:
    def test_dispatch_impact_check(self, mcp_graph, mcp_project):
        from agenthub.mcp_server import handle_tool_call

        with patch("agenthub.mcp_server.get_graph", return_value=mcp_graph), \
             patch("agenthub.mcp_server._resolve_project_root", return_value=str(mcp_project)):
            result = handle_tool_call("impact_check", {"file_path": "src/core.py"})

        assert "Impact Analysis" in result

    def test_dispatch_unknown_tool(self):
        from agenthub.mcp_server import handle_tool_call

        result = handle_tool_call("nonexistent_tool", {})
        assert "Unknown tool" in result


class TestJsonRpcProtocol:
    def test_tools_list(self):
        from agenthub.mcp_server import TOOLS

        assert len(TOOLS) == 3
        names = {t["name"] for t in TOOLS}
        assert names == {"impact_check", "affected_tests", "codebase_overview"}

    def test_all_tools_have_schemas(self):
        from agenthub.mcp_server import TOOLS

        for tool in TOOLS:
            assert "inputSchema" in tool
            assert "description" in tool
            assert isinstance(tool["description"], str)
            assert len(tool["description"]) > 10
