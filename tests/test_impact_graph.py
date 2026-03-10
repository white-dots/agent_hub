"""Tests for ImportGraph impact analysis methods."""

import pytest

from agenthub.auto.import_graph import ImportGraph


@pytest.fixture
def impact_project(tmp_path):
    """Create a project with a clear dependency chain for impact testing.

    Structure:
        src/
            utils.py          (imported by service.py, helpers.py)
            service.py         (imports utils.py, imported by api.py)
            helpers.py         (imports utils.py)
            api.py             (imports service.py)
            standalone.py      (no imports, no importers)
        tests/
            test_service.py    (imports service.py)
            test_api.py        (imports api.py)
        __tests__/
            helpers.spec.ts    (TS test file)
        src/components/
            Button.tsx         (exports a component)
    """
    src = tmp_path / "src"
    src.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()
    js_tests = tmp_path / "__tests__"
    js_tests.mkdir()
    components = src / "components"
    components.mkdir()

    (src / "utils.py").write_text(
        '''"""Shared utilities."""

__all__ = ["format_name", "parse_date", "MAX_RETRIES"]

MAX_RETRIES = 3

class BaseModel:
    """Base for all models."""
    def validate(self):
        pass
    def save(self):
        pass

def format_name(first: str, last: str) -> str:
    """Format a full name."""
    return f"{first} {last}"

def parse_date(s: str):
    """Parse a date string."""
    return s

def _internal_helper():
    """Not exported."""
    pass
'''
    )

    (src / "service.py").write_text(
        '''"""Business logic."""
from src.utils import format_name, parse_date

class UserService:
    def get_user(self, user_id):
        return {"id": user_id}

    def create_user(self, data):
        name = format_name(data["first"], data["last"])
        return {"name": name}

async def process_batch(items):
    return [format_name(i["f"], i["l"]) for i in items]
'''
    )

    (src / "helpers.py").write_text(
        '''"""Helper functions."""
from src.utils import parse_date

def helper_one():
    return parse_date("2024-01-01")
'''
    )

    (src / "api.py").write_text(
        '''"""API endpoints."""
from src.service import UserService

def get_users():
    svc = UserService()
    return svc.get_user(1)
'''
    )

    (src / "standalone.py").write_text(
        '''"""A standalone module with no deps."""
def solo():
    return 42
'''
    )

    (tests / "test_service.py").write_text(
        '''"""Service tests."""
from src.service import UserService

def test_get_user():
    svc = UserService()
    assert svc.get_user(1) == {"id": 1}
'''
    )

    (tests / "test_api.py").write_text(
        '''"""API tests."""
from src.api import get_users

def test_get_users():
    assert get_users() is not None
'''
    )

    (js_tests / "helpers.spec.ts").write_text(
        '''import { helperOne } from "../src/helpers";
test("helperOne", () => { expect(helperOne()).toBeDefined(); });
'''
    )

    (components / "Button.tsx").write_text(
        '''export interface ButtonProps {
  label: string;
  onClick: () => void;
}

export const BUTTON_SIZES = { sm: 24, md: 32, lg: 48 };

export default function Button({ label, onClick }: ButtonProps) {
  return <button onClick={onClick}>{label}</button>;
}

export class ButtonGroup {
  buttons: ButtonProps[] = [];
}
'''
    )

    return tmp_path


@pytest.fixture
def graph(impact_project):
    """Build an import graph for the impact project."""
    g = ImportGraph(str(impact_project))
    g.build()
    return g


# ------------------------------------------------------------------
# is_test_file
# ------------------------------------------------------------------

class TestIsTestFile:
    def test_python_test_prefix(self, graph):
        assert graph.is_test_file("tests/test_service.py") is True
        assert graph.is_test_file("test_something.py") is True

    def test_python_test_suffix(self, graph):
        assert graph.is_test_file("service_test.py") is True

    def test_python_conftest(self, graph):
        assert graph.is_test_file("conftest.py") is True

    def test_python_tests_directory(self, graph):
        assert graph.is_test_file("tests/anything.py") is True

    def test_python_not_test(self, graph):
        assert graph.is_test_file("src/service.py") is False
        assert graph.is_test_file("src/utils.py") is False

    def test_ts_spec_file(self, graph):
        assert graph.is_test_file("__tests__/helpers.spec.ts") is True
        assert graph.is_test_file("foo.spec.tsx") is True

    def test_ts_test_file(self, graph):
        assert graph.is_test_file("Button.test.tsx") is True
        assert graph.is_test_file("utils.test.ts") is True

    def test_js_test_file(self, graph):
        assert graph.is_test_file("helpers.test.js") is True
        assert graph.is_test_file("app.spec.jsx") is True

    def test_ts_not_test(self, graph):
        assert graph.is_test_file("src/components/Button.tsx") is False


# ------------------------------------------------------------------
# get_transitive_importers
# ------------------------------------------------------------------

class TestGetTransitiveImporters:
    def test_leaf_module_no_importers(self, graph):
        """standalone.py has no importers."""
        result = graph.get_transitive_importers("src/standalone.py")
        assert result == []

    def test_direct_importers_only(self, graph):
        """api.py is only imported by test_api.py (1 hop)."""
        result = graph.get_transitive_importers("src/api.py")
        assert "tests/test_api.py" in result

    def test_transitive_chain(self, graph):
        """utils.py -> service.py -> api.py -> test_api.py should all appear."""
        result = graph.get_transitive_importers("src/utils.py")
        paths = set(result)
        assert "src/service.py" in paths
        assert "src/helpers.py" in paths
        # Transitive: service -> api, service -> test_service
        assert "src/api.py" in paths
        assert "tests/test_service.py" in paths

    def test_max_depth_limits_traversal(self, graph):
        """max_depth=1 should only return direct importers of utils.py."""
        result = graph.get_transitive_importers("src/utils.py", max_depth=1)
        paths = set(result)
        assert "src/service.py" in paths
        assert "src/helpers.py" in paths
        # api.py is 2 hops away, should NOT appear
        assert "src/api.py" not in paths

    def test_unknown_file(self, graph):
        result = graph.get_transitive_importers("nonexistent.py")
        assert result == []

    def test_no_duplicates(self, graph):
        """Result should not contain duplicates even with diamond dependencies."""
        result = graph.get_transitive_importers("src/utils.py")
        assert len(result) == len(set(result))

    def test_does_not_include_self(self, graph):
        result = graph.get_transitive_importers("src/utils.py")
        assert "src/utils.py" not in result


# ------------------------------------------------------------------
# get_affected_tests
# ------------------------------------------------------------------

class TestGetAffectedTests:
    def test_direct_test_dependency(self, graph):
        """service.py is directly imported by test_service.py."""
        result = graph.get_affected_tests(["src/service.py"])
        assert "tests/test_service.py" in result

    def test_transitive_test_dependency(self, graph):
        """utils.py -> service.py -> test_service.py (transitive)."""
        result = graph.get_affected_tests(["src/utils.py"])
        assert "tests/test_service.py" in result

    def test_deeper_transitive(self, graph):
        """utils.py -> service.py -> api.py -> test_api.py."""
        result = graph.get_affected_tests(["src/utils.py"])
        assert "tests/test_api.py" in result

    def test_no_affected_tests(self, graph):
        result = graph.get_affected_tests(["src/standalone.py"])
        assert result == []

    def test_test_file_itself(self, graph):
        """If the changed file IS a test, include it."""
        result = graph.get_affected_tests(["tests/test_service.py"])
        assert "tests/test_service.py" in result

    def test_multiple_files(self, graph):
        """Multiple changed files should union their affected tests."""
        result = graph.get_affected_tests(["src/api.py", "src/helpers.py"])
        assert "tests/test_api.py" in result

    def test_sorted_output(self, graph):
        result = graph.get_affected_tests(["src/utils.py"])
        assert result == sorted(result)


# ------------------------------------------------------------------
# get_exported_interface — Python
# ------------------------------------------------------------------

class TestGetExportedInterfacePython:
    def test_respects_all(self, graph):
        """utils.py has __all__, so only listed names should appear."""
        interface = graph.get_exported_interface("src/utils.py")
        assert interface["language"] == "python"

        func_names = {f["name"] for f in interface["functions"]}
        assert "format_name" in func_names
        assert "parse_date" in func_names
        # _internal_helper is private and not in __all__
        assert "_internal_helper" not in func_names
        # BaseModel is not in __all__
        class_names = {c["name"] for c in interface["classes"]}
        assert "BaseModel" not in class_names

        const_names = {c["name"] for c in interface["constants"]}
        assert "MAX_RETRIES" in const_names

    def test_function_args(self, graph):
        interface = graph.get_exported_interface("src/utils.py")
        format_fn = next(f for f in interface["functions"] if f["name"] == "format_name")
        assert "first" in format_fn["args"]
        assert "last" in format_fn["args"]

    def test_no_all_shows_public(self, graph):
        """service.py has no __all__, so all public names should appear."""
        interface = graph.get_exported_interface("src/service.py")
        class_names = {c["name"] for c in interface["classes"]}
        assert "UserService" in class_names
        func_names = {f["name"] for f in interface["functions"]}
        assert "process_batch" in func_names

    def test_async_detection(self, graph):
        interface = graph.get_exported_interface("src/service.py")
        process_fn = next(f for f in interface["functions"] if f["name"] == "process_batch")
        assert process_fn["is_async"] is True

    def test_class_methods(self, graph):
        interface = graph.get_exported_interface("src/service.py")
        user_svc = next(c for c in interface["classes"] if c["name"] == "UserService")
        assert "get_user" in user_svc["methods"]
        assert "create_user" in user_svc["methods"]

    def test_unknown_file(self, graph):
        interface = graph.get_exported_interface("nonexistent.py")
        assert interface["classes"] == []
        assert interface["functions"] == []


# ------------------------------------------------------------------
# get_exported_interface — TypeScript/JSX
# ------------------------------------------------------------------

class TestGetExportedInterfaceTS:
    def test_tsx_exports(self, graph):
        interface = graph.get_exported_interface("src/components/Button.tsx")
        assert interface["language"] in ("typescript", "javascript")

        func_names = {f["name"] for f in interface["functions"]}
        assert "Button" in func_names

        class_names = {c["name"] for c in interface["classes"]}
        assert "ButtonGroup" in class_names

        const_names = {c["name"] for c in interface["constants"]}
        # BUTTON_SIZES is an exported const
        assert "BUTTON_SIZES" in const_names

        # ButtonProps is an exported interface
        type_names = {c["name"] for c in interface["constants"] if c.get("type_annotation") == "type"}
        assert "ButtonProps" in type_names
