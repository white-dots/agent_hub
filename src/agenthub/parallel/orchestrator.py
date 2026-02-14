from __future__ import annotations
"""Branch orchestration for parallel sessions.

The BranchOrchestrator manages git worktrees and spawns parallel Claude Code
sessions. It supports two execution backends:
- CLI mode: `claude --print` (stable, non-interactive)
- Agent Teams mode: Full Claude Code teammates with inter-agent messaging

IMPORTANT: Uses git worktrees, NOT git checkout.
`git checkout` mutates the working directory globally. If two parallel sessions
both `git checkout` different branches, they race on the same `.git` directory
and corrupt each other. `git worktree add` creates separate working directories
that share the same `.git` object store but have independent checkouts.
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

logger = logging.getLogger(__name__)

from agenthub.parallel.models import (
    ImplementationTask,
    ParallelizationPlan,
    SessionResult,
    SessionSpec,
)
from agenthub.parallel.prompts import build_scoped_prompt
from agenthub.parallel.teams_adapter import AgentTeamsAdapter

if TYPE_CHECKING:
    from agenthub.hub import AgentHub


class BranchOrchestrator:
    """Manages git branches and spawns parallel Claude Code sessions.

    Uses git worktrees for filesystem isolation between parallel sessions.
    Supports two execution backends:
    - "cli": claude --print (one-shot, non-interactive, stable)
    - "agent_teams": Claude Code Agent Teams (full instances, inter-agent
      messaging, real-time boundary crossing negotiation)

    Example:
        >>> orchestrator = BranchOrchestrator(project_root, hub)
        >>> results = orchestrator.execute_plan(plan, tasks)
        >>> for result in results:
        ...     print(f"{result.task_id}: {'success' if result.success else 'failed'}")
    """

    # Directory for worktrees
    WORKTREE_DIR = ".worktrees/parallel"

    def __init__(
        self,
        project_root: str,
        hub: "AgentHub",
        max_parallel: int = 3,
        claude_model: str = "claude-sonnet-4-20250514",
        session_timeout: int = 300,
        execution_backend: Literal["cli", "agent_teams"] = "cli",
    ):
        """Initialize BranchOrchestrator.

        Args:
            project_root: Path to the project root (must be a git repository).
            hub: AgentHub for accessing domain agents.
            max_parallel: Maximum sessions to run in parallel.
            claude_model: Model to use for Claude Code sessions.
            session_timeout: Timeout for each session in seconds.
            execution_backend: "cli" or "agent_teams".
        """
        self._root = Path(project_root).resolve()
        self._hub = hub
        self._max_parallel = max_parallel
        self._model = claude_model
        self._timeout = session_timeout
        self._backend = execution_backend

        self._base_branch: str = ""
        self._base_commit: str = ""
        self._worktrees: list[Path] = []  # Track for cleanup

        # Agent Teams adapter (lazy initialized)
        self._teams_adapter: Optional[AgentTeamsAdapter] = None

        # Verify git repository
        if not (self._root / ".git").exists():
            raise ValueError(f"Not a git repository: {project_root}")

    def execute_plan(
        self,
        plan: ParallelizationPlan,
        tasks: list[ImplementationTask],
    ) -> list[SessionResult]:
        """Execute the parallelization plan.

        Flow:
        1. Capture base state (current branch/commit)
        2. Ensure working tree is clean
        3. For each parallel group:
           a. Create worktrees with branches from base
           b. Spawn sessions in parallel (via selected backend)
           c. Collect results
        4. For sequential tasks (if pm_recommendation is sequential):
           a. Run one at a time in temporary worktree
           b. Merge into base before next
        5. Cleanup worktrees
        6. Return all results

        Args:
            plan: ParallelizationPlan from ConflictRiskAnalyzer.
            tasks: List of ImplementationTask to execute.

        Returns:
            List of SessionResult for each task.
        """
        results: list[SessionResult] = []

        try:
            # Step 1: Capture base state
            self._capture_base_state()

            # Step 2: Ensure clean tree
            self._ensure_clean_tree()

            # Create task lookup
            task_map = {t.task_id: t for t in tasks}

            if plan.pm_recommendation == "parallel":
                # Execute parallel groups
                for group in plan.parallel_groups:
                    group_tasks = [task_map[tid] for tid in group if tid in task_map]
                    group_results = self._execute_parallel_group(group_tasks, tasks)
                    results.extend(group_results)
            else:
                # Execute sequentially
                for task_id in plan.sequential_order:
                    if task_id in task_map:
                        task = task_map[task_id]
                        result = self._execute_single_task(task, tasks)
                        results.append(result)

        finally:
            # Always cleanup worktrees
            self._cleanup_worktrees()

        return results

    # =========================================================================
    # Git State Management
    # =========================================================================

    def _capture_base_state(self) -> None:
        """Record current branch and commit SHA."""
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        self._base_branch = result.stdout.strip() or "HEAD"

        # Get current commit
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        self._base_commit = result.stdout.strip()

    def _ensure_clean_tree(self) -> None:
        """Fail if working tree has uncommitted tracked-file changes.

        Only checks for modified/staged tracked files.  Untracked files
        (e.g. `.agenthub/` cache, `.claude/` settings) are ignored because
        auto-discovery and agent init routinely create them.
        """
        # Check staged changes
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        # Check unstaged changes to tracked files
        unstaged = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )

        dirty_files = (staged.stdout.strip() + "\n" + unstaged.stdout.strip()).strip()
        if dirty_files:
            raise RuntimeError(
                "Working tree has uncommitted changes. "
                "Please commit or stash before running parallel sessions.\n"
                f"Changed files: {dirty_files[:200]}"
            )

    def _create_worktree(self, task_id: str) -> tuple[str, Path]:
        """Create isolated working directory for a parallel session.

        Creates: .worktrees/parallel/{task_id}
        Branch: parallel/{task_id}

        Args:
            task_id: Task identifier for naming.

        Returns:
            Tuple of (branch_name, worktree_path).
        """
        branch_name = f"parallel/{task_id}"
        worktree_path = self._root / self.WORKTREE_DIR / task_id

        # Ensure parent directory exists
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing worktree if present
        if worktree_path.exists():
            subprocess.run(
                ["git", "worktree", "remove", str(worktree_path), "--force"],
                cwd=self._root,
                capture_output=True,
            )

        # Delete branch if it exists
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=self._root,
            capture_output=True,
        )

        # Create new worktree with new branch from current commit
        result = subprocess.run(
            [
                "git", "worktree", "add",
                str(worktree_path),
                "-b", branch_name,
                self._base_commit,
            ],
            cwd=self._root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create worktree: {result.stderr}")

        self._worktrees.append(worktree_path)
        return branch_name, worktree_path

    def _cleanup_worktrees(self) -> None:
        """Remove all parallel worktrees after execution."""
        for worktree_path in self._worktrees:
            try:
                # Get branch name from worktree
                task_id = worktree_path.name
                branch_name = f"parallel/{task_id}"

                # Remove worktree
                subprocess.run(
                    ["git", "worktree", "remove", str(worktree_path), "--force"],
                    cwd=self._root,
                    capture_output=True,
                )

                # Note: We don't delete the branch here - it may be needed for merge
            except Exception as e:
                print(f"Warning: Failed to cleanup worktree {worktree_path}: {e}")

        self._worktrees.clear()

    def rollback(self) -> None:
        """Rollback all changes - cleanup worktrees and delete parallel branches."""
        # Cleanup worktrees
        self._cleanup_worktrees()

        # Also cleanup any orphaned worktrees
        worktree_dir = self._root / self.WORKTREE_DIR
        if worktree_dir.exists():
            shutil.rmtree(worktree_dir, ignore_errors=True)

        # Delete all parallel/* branches
        result = subprocess.run(
            ["git", "branch", "--list", "parallel/*"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )

        for branch in result.stdout.strip().split("\n"):
            branch = branch.strip()
            if branch:
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    cwd=self._root,
                    capture_output=True,
                )

    # =========================================================================
    # Session Execution
    # =========================================================================

    def _execute_parallel_group(
        self,
        group_tasks: list[ImplementationTask],
        all_tasks: list[ImplementationTask],
    ) -> list[SessionResult]:
        """Execute a group of tasks in parallel.

        Args:
            group_tasks: Tasks in this parallel group.
            all_tasks: All tasks (for building scoped prompts).

        Returns:
            List of SessionResult for this group.
        """
        if not group_tasks:
            return []

        # Limit concurrency
        effective_parallel = min(len(group_tasks), self._max_parallel)

        # Build session specs
        specs: list[SessionSpec] = []
        for task in group_tasks:
            branch_name, worktree_path = self._create_worktree(task.task_id)
            spec = self._build_session_spec(
                task, branch_name, worktree_path, all_tasks
            )
            specs.append(spec)

        # Execute based on backend
        if self._backend == "agent_teams" and self._is_agent_teams_available():
            return asyncio.run(self._spawn_parallel_group_agent_teams(specs))
        else:
            return asyncio.run(self._spawn_parallel_group_cli(specs))

    def _execute_single_task(
        self,
        task: ImplementationTask,
        all_tasks: list[ImplementationTask],
    ) -> SessionResult:
        """Execute a single task (for sequential execution).

        Args:
            task: Task to execute.
            all_tasks: All tasks for context.

        Returns:
            SessionResult for this task.
        """
        branch_name, worktree_path = self._create_worktree(task.task_id)
        spec = self._build_session_spec(task, branch_name, worktree_path, all_tasks)

        if self._backend == "agent_teams" and self._is_agent_teams_available():
            results = asyncio.run(self._spawn_parallel_group_agent_teams([spec]))
        else:
            results = asyncio.run(self._spawn_parallel_group_cli([spec]))

        return results[0] if results else SessionResult(
            task_id=task.task_id,
            branch_name=branch_name,
            success=False,
            error="No result returned",
        )

    def _build_session_spec(
        self,
        task: ImplementationTask,
        branch_name: str,
        worktree_path: Path,
        all_tasks: list[ImplementationTask],
    ) -> SessionSpec:
        """Build a SessionSpec for a task.

        Args:
            task: The task to build spec for.
            branch_name: Git branch name.
            worktree_path: Path to the worktree.
            all_tasks: All tasks for determining other files.

        Returns:
            SessionSpec ready for execution.
        """
        # Determine files for this task vs other tasks
        your_files = task.estimated_files + task.estimated_new_files

        other_files: list[str] = []
        for other_task in all_tasks:
            if other_task.task_id != task.task_id:
                other_files.extend(other_task.estimated_files)
                other_files.extend(other_task.estimated_new_files)

        # Remove duplicates
        other_files = list(set(other_files) - set(your_files))

        # Get context from domain agents
        tier_b_context = self._get_tier_b_context(task)
        tier_a_context = self._get_tier_a_context(task)

        # Build scoped prompt
        prompt = build_scoped_prompt(
            task=task,
            your_files=your_files,
            other_files=other_files,
            tier_b_context=tier_b_context,
            tier_a_context=tier_a_context,
        )

        return SessionSpec(
            task=task,
            branch_name=branch_name,
            worktree_path=str(worktree_path),
            scoped_files=your_files,
            scoped_dirs=list(set(str(Path(f).parent) for f in your_files)),
            context_from_agents=task.domain_agents,
            prompt=prompt,
            timeout_seconds=self._timeout,
        )

    def _get_tier_b_context(self, task: ImplementationTask) -> str:
        """Get technical context from Tier B agents for a task."""
        context_parts: list[str] = []

        for agent_id in task.domain_agents:
            try:
                # Get agent from hub
                agents = self._hub.list_agents(tier="B")
                for agent_spec in agents:
                    if agent_spec.agent_id == agent_id:
                        # Get context summary
                        context_parts.append(
                            f"### {agent_spec.name}\n"
                            f"{agent_spec.description}\n"
                            f"Files: {', '.join(agent_spec.context_paths[:5])}"
                        )
                        break
            except Exception:
                pass

        return "\n\n".join(context_parts) if context_parts else ""

    def _get_tier_a_context(self, task: ImplementationTask) -> str:
        """Get business context from Tier A agents for a task."""
        context_parts: list[str] = []

        try:
            agents = self._hub.list_agents(tier="A")
            for agent_spec in agents:
                context_parts.append(
                    f"### {agent_spec.name}\n{agent_spec.description}"
                )
        except Exception:
            pass

        return "\n\n".join(context_parts) if context_parts else ""

    # =========================================================================
    # CLI Backend
    # =========================================================================

    async def _spawn_parallel_group_cli(
        self,
        specs: list[SessionSpec],
    ) -> list[SessionResult]:
        """Spawn multiple CLI sessions concurrently.

        Uses asyncio.gather for concurrent I/O-bound subprocess management.

        Args:
            specs: List of SessionSpec to execute.

        Returns:
            List of SessionResult.
        """
        tasks = [self._spawn_session_cli_async(spec) for spec in specs]
        return await asyncio.gather(*tasks)

    async def _spawn_session_cli_async(self, spec: SessionSpec) -> SessionResult:
        """Spawn a single Claude Code session via CLI (async).

        Args:
            spec: SessionSpec defining the session.

        Returns:
            SessionResult with execution details.
        """
        start_time = time.time()

        try:
            # Build command — use --print for non-interactive mode with
            # --dangerously-skip-permissions so tools (Edit, Write, Bash)
            # can execute without interactive approval in the worktree.
            cmd = [
                "claude",
                "--print", spec.prompt,
                "--dangerously-skip-permissions",
                "--output-format", "json",
                "--model", self._model,
            ]

            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=spec.worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
            )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=spec.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                return SessionResult(
                    task_id=spec.task.task_id,
                    branch_name=spec.branch_name,
                    success=False,
                    error=f"Session timed out after {spec.timeout_seconds}s",
                    time_seconds=time.time() - start_time,
                    execution_backend="cli",
                )

            stdout_text = stdout.decode() if stdout else ""
            stderr_text = stderr.decode() if stderr else ""

            # Check success
            success = process.returncode == 0

            # Auto-commit any changes the session made in the worktree.
            # Claude Code may or may not commit on its own, so we ensure
            # all modifications are captured in a commit for the merge step.
            self._auto_commit_worktree(spec)

            # Get files changed
            files_changed = self._get_files_changed(spec.worktree_path, self._base_branch)
            files_created = [f for f in files_changed if not (self._root / f).exists()]

            # Parse boundary crossings
            boundary_crossings = self._parse_boundary_crossings(stdout_text)

            # Extract token usage from output (if available)
            tokens_used = self._extract_token_usage(stdout_text)

            return SessionResult(
                task_id=spec.task.task_id,
                branch_name=spec.branch_name,
                success=success,
                files_changed=files_changed,
                files_created=files_created,
                stdout=stdout_text,
                tokens_used=tokens_used,
                time_seconds=time.time() - start_time,
                error=stderr_text if not success else None,
                boundary_crossings=boundary_crossings,
                execution_backend="cli",
            )

        except Exception as e:
            return SessionResult(
                task_id=spec.task.task_id,
                branch_name=spec.branch_name,
                success=False,
                error=str(e),
                time_seconds=time.time() - start_time,
                execution_backend="cli",
            )

    def _auto_commit_worktree(self, spec: SessionSpec) -> None:
        """Auto-commit any uncommitted changes in the worktree.

        Claude Code sessions may or may not commit their own changes.
        This ensures all modifications are captured in a commit so the
        merge step has something to work with.

        Args:
            spec: SessionSpec with worktree path and task info.
        """
        wt = spec.worktree_path

        # Check for any uncommitted changes (staged or unstaged)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=wt, capture_output=True, text=True,
        )

        if not status.stdout.strip():
            # No changes to commit — session either already committed
            # or produced no file modifications
            return

        # Stage all changes
        subprocess.run(
            ["git", "add", "-A"],
            cwd=wt, capture_output=True, text=True,
        )

        # Commit with a descriptive message
        task_desc = spec.task.description[:80] if spec.task else "parallel session"
        subprocess.run(
            ["git", "commit", "-m", f"parallel({spec.task.task_id}): {task_desc}",
             "--no-verify"],
            cwd=wt, capture_output=True, text=True,
        )

        logger.info(f"Auto-committed changes in worktree for task {spec.task.task_id}")

    def _get_files_changed(self, worktree_path: str, base_branch: str) -> list[str]:
        """Get files changed in worktree vs base branch.

        Args:
            worktree_path: Path to the worktree.
            base_branch: Base branch to compare against.

        Returns:
            List of changed file paths (relative to repo root).
        """
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{self._base_commit}...HEAD"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return []

        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return files

    def _parse_boundary_crossings(self, stdout: str) -> list[str]:
        """Parse [BOUNDARY_CROSSING: ...] tags from session output.

        Args:
            stdout: Session output text.

        Returns:
            List of boundary crossing descriptions.
        """
        pattern = r"\[BOUNDARY_CROSSING:\s*([^\]]+)\]"
        matches = re.findall(pattern, stdout)
        return matches

    def _extract_token_usage(self, stdout: str) -> int:
        """Extract token usage from Claude output if available.

        Args:
            stdout: Session output text.

        Returns:
            Token count or 0 if not found.
        """
        # Try to parse JSON output
        try:
            import json
            # Look for JSON in output
            json_match = re.search(r"\{.*\"usage\".*\}", stdout, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                usage = data.get("usage", {})
                return usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        except (json.JSONDecodeError, KeyError):
            pass

        return 0

    # =========================================================================
    # Agent Teams Backend
    # =========================================================================

    def _is_agent_teams_available(self) -> bool:
        """Check if Agent Teams is enabled and available.

        Uses the AgentTeamsAdapter to verify availability.
        """
        if self._teams_adapter is None:
            self._teams_adapter = AgentTeamsAdapter(
                str(self._root),
                self._hub,
            )
        return self._teams_adapter.is_available()

    async def _spawn_parallel_group_agent_teams(
        self,
        specs: list[SessionSpec],
    ) -> list[SessionResult]:
        """Spawn teammates and coordinate via Agent Teams.

        Uses AgentTeamsAdapter for inter-agent messaging and
        real-time boundary crossing negotiation.

        Args:
            specs: List of SessionSpec to execute.

        Returns:
            List of SessionResult.
        """
        if self._teams_adapter is None:
            self._teams_adapter = AgentTeamsAdapter(
                str(self._root),
                self._hub,
            )

        # Check availability
        if not self._teams_adapter.is_available():
            print("Agent Teams not available, falling back to CLI")
            return await self._spawn_parallel_group_cli(specs)

        try:
            return await self._teams_adapter.execute_sessions(
                specs,
                timeout_seconds=self._timeout,
            )
        except Exception as e:
            print(f"Agent Teams execution failed: {e}, falling back to CLI")
            return await self._spawn_parallel_group_cli(specs)

    def get_boundary_crossings(self) -> list:
        """Get boundary crossings from Agent Teams adapter.

        Returns:
            List of pending boundary crossings.
        """
        if self._teams_adapter:
            return self._teams_adapter.get_pending_crossings()
        return []

    def get_crossing_resolutions(self) -> list:
        """Get boundary crossing resolutions from Agent Teams adapter.

        Returns:
            List of crossing resolutions.
        """
        if self._teams_adapter:
            return self._teams_adapter.get_crossing_resolutions()
        return []

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_branch_for_task(self, task_id: str) -> Optional[str]:
        """Get the branch name for a task.

        Args:
            task_id: Task identifier.

        Returns:
            Branch name or None if not found.
        """
        return f"parallel/{task_id}"

    def get_worktree_for_task(self, task_id: str) -> Optional[Path]:
        """Get the worktree path for a task.

        Args:
            task_id: Task identifier.

        Returns:
            Worktree path or None if not created.
        """
        worktree_path = self._root / self.WORKTREE_DIR / task_id
        return worktree_path if worktree_path.exists() else None

    def list_active_worktrees(self) -> list[str]:
        """List all active parallel worktrees.

        Returns:
            List of task IDs with active worktrees.
        """
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )

        worktrees = []
        for line in result.stdout.split("\n"):
            if line.startswith("worktree "):
                path = line[9:]
                if self.WORKTREE_DIR in path:
                    task_id = Path(path).name
                    worktrees.append(task_id)

        return worktrees
