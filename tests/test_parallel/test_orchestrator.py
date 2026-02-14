"""Tests for BranchOrchestrator and AgentTeamsAdapter."""

import asyncio
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from agenthub.parallel.models import (
    ImplementationTask,
    ParallelizationPlan,
    RiskLevel,
    SessionSpec,
)
from agenthub.parallel.orchestrator import BranchOrchestrator


# === Fixtures ===


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository."""
    # Initialize git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        capture_output=True,
    )

    # Create initial file and commit
    (tmp_path / "README.md").write_text("# Test Project")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        capture_output=True,
    )

    return tmp_path


@pytest.fixture
def mock_hub():
    """Create a mock AgentHub."""
    hub = MagicMock()
    hub.list_agents.return_value = []
    return hub


@pytest.fixture
def sample_tasks():
    """Create sample implementation tasks."""
    return [
        ImplementationTask(
            task_id="task_1",
            description="Add save button",
            estimated_files=["src/components/Button.tsx"],
            domain_agents=["frontend"],
            complexity="moderate",
        ),
        ImplementationTask(
            task_id="task_2",
            description="Add chart component",
            estimated_files=["src/components/Chart.tsx"],
            domain_agents=["frontend"],
            complexity="moderate",
        ),
    ]


@pytest.fixture
def sample_plan(sample_tasks):
    """Create a sample parallelization plan."""
    return ParallelizationPlan(
        parallel_groups=[["task_1", "task_2"]],
        sequential_order=["task_1", "task_2"],
        overall_risk=RiskLevel.NONE,
        pm_recommendation="parallel",
    )


# === BranchOrchestrator Tests ===


class TestBranchOrchestrator:
    """Tests for BranchOrchestrator."""

    def test_init_requires_git_repo(self, tmp_path, mock_hub):
        """Should raise error if not a git repository."""
        with pytest.raises(ValueError, match="Not a git repository"):
            BranchOrchestrator(str(tmp_path), mock_hub)

    def test_init_with_valid_repo(self, git_repo, mock_hub):
        """Should initialize with valid git repository."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        assert orchestrator._root == git_repo.resolve()
        assert orchestrator._max_parallel == 3
        assert orchestrator._backend == "cli"

    def test_init_custom_params(self, git_repo, mock_hub):
        """Should accept custom parameters."""
        orchestrator = BranchOrchestrator(
            str(git_repo),
            mock_hub,
            max_parallel=5,
            claude_model="claude-opus-4-20250514",
            session_timeout=600,
            execution_backend="agent_teams",
        )

        assert orchestrator._max_parallel == 5
        assert orchestrator._model == "claude-opus-4-20250514"
        assert orchestrator._timeout == 600
        assert orchestrator._backend == "agent_teams"

    def test_capture_base_state(self, git_repo, mock_hub):
        """Should capture current branch and commit."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        orchestrator._capture_base_state()

        assert orchestrator._base_branch in ["main", "master", "HEAD"]
        assert len(orchestrator._base_commit) == 40  # SHA-1 hash length

    def test_ensure_clean_tree_passes(self, git_repo, mock_hub):
        """Should pass with clean working tree."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        # Should not raise
        orchestrator._ensure_clean_tree()

    def test_ensure_clean_tree_fails_with_changes(self, git_repo, mock_hub):
        """Should fail with uncommitted changes to tracked files."""
        # Modify a tracked file (README.md was committed in the fixture)
        (git_repo / "README.md").write_text("modified content")

        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        with pytest.raises(RuntimeError, match="uncommitted changes"):
            orchestrator._ensure_clean_tree()

    def test_ensure_clean_tree_ignores_untracked(self, git_repo, mock_hub):
        """Should pass with untracked files (e.g. cache dirs)."""
        # Create an untracked file (like .agenthub/ cache)
        (git_repo / "untracked.txt").write_text("not tracked")

        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        # Should not raise - untracked files are ignored
        orchestrator._ensure_clean_tree()

    def test_create_worktree(self, git_repo, mock_hub):
        """Should create isolated worktree for a task."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        orchestrator._capture_base_state()

        branch_name, worktree_path = orchestrator._create_worktree("test_task")

        assert branch_name == "parallel/test_task"
        assert worktree_path.exists()
        assert worktree_path.name == "test_task"
        assert ".worktrees/parallel" in str(worktree_path)

        # Should track worktree for cleanup
        assert worktree_path in orchestrator._worktrees

        # Cleanup
        orchestrator._cleanup_worktrees()

    def test_cleanup_worktrees(self, git_repo, mock_hub):
        """Should remove worktrees but keep branches."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        orchestrator._capture_base_state()

        # Create worktree
        branch_name, worktree_path = orchestrator._create_worktree("cleanup_test")
        assert worktree_path.exists()

        # Cleanup
        orchestrator._cleanup_worktrees()

        # Worktree should be removed
        assert not worktree_path.exists()
        assert len(orchestrator._worktrees) == 0

        # Branch should still exist (for merge)
        result = subprocess.run(
            ["git", "branch", "--list", branch_name],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert branch_name in result.stdout

    def test_rollback_deletes_branches(self, git_repo, mock_hub):
        """Rollback should delete parallel branches."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        orchestrator._capture_base_state()

        # Create worktree
        branch_name, worktree_path = orchestrator._create_worktree("rollback_test")
        orchestrator._cleanup_worktrees()  # Cleanup worktree first

        # Verify branch exists
        result = subprocess.run(
            ["git", "branch", "--list", branch_name],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert branch_name in result.stdout

        # Rollback
        orchestrator.rollback()

        # Branch should be deleted
        result = subprocess.run(
            ["git", "branch", "--list", branch_name],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert branch_name not in result.stdout

    def test_parse_boundary_crossings(self, git_repo, mock_hub):
        """Should parse boundary crossing tags from output."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)

        output = """
        Working on task...
        [BOUNDARY_CROSSING: src/utils/helper.ts - need shared utility function]
        Continuing work...
        [BOUNDARY_CROSSING: config/settings.json - need to update config]
        Done.
        """

        crossings = orchestrator._parse_boundary_crossings(output)

        assert len(crossings) == 2
        assert "src/utils/helper.ts - need shared utility function" in crossings
        assert "config/settings.json - need to update config" in crossings

    def test_get_branch_for_task(self, git_repo, mock_hub):
        """Should return correct branch name for task."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)

        assert orchestrator.get_branch_for_task("task_1") == "parallel/task_1"
        assert orchestrator.get_branch_for_task("my-feature") == "parallel/my-feature"

    def test_list_active_worktrees(self, git_repo, mock_hub):
        """Should list active parallel worktrees."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        orchestrator._capture_base_state()

        # Initially no worktrees
        assert orchestrator.list_active_worktrees() == []

        # Create worktrees
        orchestrator._create_worktree("active_1")
        orchestrator._create_worktree("active_2")

        active = orchestrator.list_active_worktrees()
        assert "active_1" in active
        assert "active_2" in active

        # Cleanup
        orchestrator._cleanup_worktrees()

    def test_build_session_spec(self, git_repo, mock_hub, sample_tasks):
        """Should build SessionSpec with scoped prompt."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        orchestrator._capture_base_state()

        branch_name, worktree_path = orchestrator._create_worktree("spec_test")

        spec = orchestrator._build_session_spec(
            sample_tasks[0],
            branch_name,
            worktree_path,
            sample_tasks,
        )

        assert isinstance(spec, SessionSpec)
        assert spec.task == sample_tasks[0]
        assert spec.branch_name == branch_name
        assert spec.worktree_path == str(worktree_path)
        assert "src/components/Button.tsx" in spec.scoped_files
        assert "Add save button" in spec.prompt

        # Cleanup
        orchestrator._cleanup_worktrees()


class TestBranchOrchestratorAsync:
    """Async tests for BranchOrchestrator."""

    def test_spawn_session_cli_async_timeout(self, git_repo, mock_hub):
        """Should handle session timeout."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub, session_timeout=1)
        orchestrator._capture_base_state()

        task = ImplementationTask(
            task_id="timeout_test",
            description="Test timeout",
            estimated_files=[],
        )

        branch_name, worktree_path = orchestrator._create_worktree("timeout_test")
        spec = SessionSpec(
            task=task,
            branch_name=branch_name,
            worktree_path=str(worktree_path),
            prompt="sleep 10",  # Will timeout
            timeout_seconds=1,
        )

        # Mock the subprocess to simulate timeout
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_process.kill = MagicMock()
            mock_exec.return_value = mock_process

            result = asyncio.run(orchestrator._spawn_session_cli_async(spec))

            assert result.success is False
            assert "timed out" in result.error

        orchestrator._cleanup_worktrees()


# === AgentTeamsAdapter Tests ===


class TestAgentTeamsAdapter:
    """Tests for AgentTeamsAdapter."""

    def test_is_available_false_by_default(self, git_repo, mock_hub):
        """Agent Teams should not be available without env var."""
        from agenthub.parallel.teams_adapter import AgentTeamsAdapter

        adapter = AgentTeamsAdapter(str(git_repo), mock_hub)

        with patch.dict(os.environ, {}, clear=True):
            assert adapter.is_available() is False

    def test_is_available_with_env_var(self, git_repo, mock_hub):
        """Agent Teams availability depends on CLI support."""
        from agenthub.parallel.teams_adapter import AgentTeamsAdapter

        adapter = AgentTeamsAdapter(str(git_repo), mock_hub)

        with patch.dict(
            os.environ,
            {"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "true"},
        ):
            # Still False unless CLI actually supports teams
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="usage: claude [options]", returncode=0)
                assert adapter.is_available() is False

                mock_run.return_value = MagicMock(stdout="--teammate flag", returncode=0)
                assert adapter.is_available() is True

    def test_parse_boundary_crossing_from_output(self, git_repo, mock_hub):
        """Should parse boundary crossing from output line."""
        from agenthub.parallel.teams_adapter import AgentTeamsAdapter

        adapter = AgentTeamsAdapter(str(git_repo), mock_hub)

        # Valid crossing
        crossing = adapter._parse_boundary_crossing_from_output(
            "[BOUNDARY_CROSSING: src/config.ts - need config values]",
            "task_1",
        )

        assert crossing is not None
        assert crossing.target_file == "src/config.ts"
        assert crossing.reason == "need config values"
        assert crossing.session_task_id == "task_1"

        # No crossing
        crossing = adapter._parse_boundary_crossing_from_output(
            "Just a regular line of output",
            "task_1",
        )
        assert crossing is None

    def test_enhance_prompt_for_teams(self, git_repo, mock_hub):
        """Should add Agent Teams instructions to prompt."""
        from agenthub.parallel.teams_adapter import AgentTeamsAdapter

        adapter = AgentTeamsAdapter(str(git_repo), mock_hub)
        inbox_path = Path("/tmp/test_inbox.jsonl")

        enhanced = adapter._enhance_prompt_for_teams(
            "Original prompt text",
            inbox_path,
        )

        assert "Agent Teams Mode" in enhanced
        assert "BOUNDARY_CROSSING" in enhanced
        assert str(inbox_path) in enhanced
        assert "Original prompt text" in enhanced

    def test_get_pending_crossings(self, git_repo, mock_hub):
        """Should track pending boundary crossings."""
        from agenthub.parallel.teams_adapter import AgentTeamsAdapter
        from agenthub.parallel.models import BoundaryCrossing

        adapter = AgentTeamsAdapter(str(git_repo), mock_hub)

        # Initially empty
        assert adapter.get_pending_crossings() == []

        # Add pending crossing
        crossing = BoundaryCrossing(
            session_task_id="task_1",
            requesting_agent="task_1",
            target_file="src/utils.ts",
            reason="need utility",
        )
        crossing_id = f"{crossing.session_task_id}:{crossing.target_file}"
        adapter._pending_crossings[crossing_id] = crossing

        pending = adapter.get_pending_crossings()
        assert len(pending) == 1
        assert pending[0] == crossing


# === Integration Tests ===


class TestOrchestratorIntegration:
    """Integration tests for orchestrator with git operations."""

    def test_parallel_worktrees_isolated(self, git_repo, mock_hub):
        """Parallel worktrees should be truly isolated."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        orchestrator._capture_base_state()

        # Create two worktrees
        branch_1, path_1 = orchestrator._create_worktree("isolation_1")
        branch_2, path_2 = orchestrator._create_worktree("isolation_2")

        # Modify file in worktree 1
        (path_1 / "file_1.txt").write_text("content from worktree 1")
        subprocess.run(["git", "add", "."], cwd=path_1, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Change from worktree 1"],
            cwd=path_1,
            capture_output=True,
        )

        # Worktree 2 should not see this change
        assert not (path_2 / "file_1.txt").exists()

        # Modify file in worktree 2
        (path_2 / "file_2.txt").write_text("content from worktree 2")
        subprocess.run(["git", "add", "."], cwd=path_2, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Change from worktree 2"],
            cwd=path_2,
            capture_output=True,
        )

        # Worktree 1 should not see this change
        assert not (path_1 / "file_2.txt").exists()

        # Main repo should not see either change
        assert not (git_repo / "file_1.txt").exists()
        assert not (git_repo / "file_2.txt").exists()

        # But branches should have the changes
        result = subprocess.run(
            ["git", "log", "--oneline", branch_1],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert "worktree 1" in result.stdout

        result = subprocess.run(
            ["git", "log", "--oneline", branch_2],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert "worktree 2" in result.stdout

        # Cleanup
        orchestrator._cleanup_worktrees()

    def test_multiple_worktrees_same_task_recreates(self, git_repo, mock_hub):
        """Creating worktree for same task should recreate it."""
        orchestrator = BranchOrchestrator(str(git_repo), mock_hub)
        orchestrator._capture_base_state()

        # Create first worktree
        branch_1, path_1 = orchestrator._create_worktree("recreate_test")
        (path_1 / "original.txt").write_text("original content")

        # Create worktree again (should recreate)
        branch_2, path_2 = orchestrator._create_worktree("recreate_test")

        assert branch_1 == branch_2
        assert path_1 == path_2
        # Original file should be gone (fresh worktree)
        assert not (path_2 / "original.txt").exists()

        orchestrator._cleanup_worktrees()
