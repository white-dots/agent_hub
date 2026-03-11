"""Shared test fixtures for AgentHub v2."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project structure for testing."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    api_dir = src_dir / "api"
    api_dir.mkdir()
    models_dir = src_dir / "models"
    models_dir.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    (src_dir / "__init__.py").write_text("# src package")
    (api_dir / "__init__.py").write_text("# api package")
    (api_dir / "routes.py").write_text(
        '"""API routes."""\ndef get_users():\n    return []\n'
    )
    (api_dir / "auth.py").write_text(
        '"""Auth module."""\ndef authenticate(token):\n    return True\n'
    )
    (models_dir / "__init__.py").write_text("# models package")
    (models_dir / "user.py").write_text(
        '"""User model."""\nclass User:\n    def __init__(self, name):\n        self.name = name\n'
    )
    (tests_dir / "__init__.py").write_text("# tests package")
    (tests_dir / "test_api.py").write_text(
        '"""API tests."""\ndef test_get_users():\n    pass\n'
    )

    return tmp_path
