from __future__ import annotations
"""Merge coordination for parallel sessions.

The MergeCoordinator handles merging parallel branches back together,
with domain-agent-assisted conflict resolution.

Flow:
1. Attempt git merge of parallel branches into integration branch
2. If textual conflicts: route to owning domain agent for resolution
3. After merge: run tests to detect semantic conflicts
4. If tests fail: escalate to CEO with context
5. Track and report all resolutions
"""

import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from agenthub.parallel.models import (
    ConflictType,
    CrossingResolutionType,
    DomainResolutionProposal,
    MergeConflict,
    MergeResult,
    SessionResult,
)

if TYPE_CHECKING:
    import anthropic

    from agenthub.hub import AgentHub


# Prompt templates for merge resolution
CONFLICT_RESOLUTION_PROMPT = """You are a domain expert resolving a merge conflict.

## Your Domain
Name: {agent_name}
Description: {agent_description}

## Conflict Details
File: {file_path}
Conflict Type: {conflict_type}

### Changes from Branch A ({branch_a}):
```
{diff_a}
```

### Changes from Branch B ({branch_b}):
```
{diff_b}
```

## Task
Propose a resolution that correctly merges both changes while maintaining code correctness.

If you cannot confidently resolve this conflict, say so and explain why.

Respond in JSON format:
```json
{{
    "can_resolve": true/false,
    "proposed_resolution": "The merged code content",
    "reasoning": "Why this resolution is correct",
    "confidence": 0.0-1.0,
    "side_effects": ["Other files that may need updates"],
    "needs_ceo": false
}}
```
"""


class MergeCoordinator:
    """Coordinates merging of parallel session branches.

    Responsibilities:
    - Merging parallel branches into integration branch
    - Detecting and classifying merge conflicts
    - Routing conflicts to owning domain agents
    - Running post-merge tests
    - Escalating unresolvable conflicts to CEO

    Example:
        >>> coordinator = MergeCoordinator(project_root, hub, client)
        >>> result = coordinator.merge_sessions(session_results)
        >>> if result.needs_user_input:
        ...     print(result.escalation_reason)
    """

    # Confidence threshold for auto-accepting resolutions
    AUTO_ACCEPT_CONFIDENCE = 0.7

    # Confidence threshold for escalation to CEO
    ESCALATE_CONFIDENCE = 0.6

    def __init__(
        self,
        project_root: str,
        hub: "AgentHub",
        client: "anthropic.Anthropic",
        run_tests: bool = True,
        test_command: Optional[str] = None,
    ):
        """Initialize MergeCoordinator.

        Args:
            project_root: Path to the project root.
            hub: AgentHub for accessing domain agents.
            client: Anthropic client for LLM calls.
            run_tests: Whether to run tests after merge.
            test_command: Custom test command (auto-detects if None).
        """
        self._root = Path(project_root).resolve()
        self._hub = hub
        self._client = client
        self._run_tests = run_tests
        self._test_command = test_command

        self._integration_branch = ""
        self._conflicts: list[MergeConflict] = []
        self._resolutions: list[DomainResolutionProposal] = []

    def merge_sessions(
        self,
        session_results: list[SessionResult],
        base_branch: str = "main",
    ) -> MergeResult:
        """Merge all parallel session branches.

        Flow:
        1. Create integration branch from base
        2. Merge each session branch in order
        3. For conflicts: attempt domain-agent resolution
        4. After all merges: run tests
        5. Return result with conflict/resolution details

        Args:
            session_results: Results from parallel sessions.
            base_branch: Base branch to merge into.

        Returns:
            MergeResult with merge outcome and details.
        """
        if not session_results:
            return MergeResult(
                success=True,
                merged_branch=base_branch,
                summary="No sessions to merge.",
            )

        # Filter to successful sessions only
        successful_sessions = [s for s in session_results if s.success]
        if not successful_sessions:
            return MergeResult(
                success=False,
                merged_branch="",
                summary="No successful sessions to merge.",
            )

        # Create integration branch
        self._integration_branch = self._create_integration_branch(base_branch)

        # Track all merged files
        all_merged_files: list[str] = []
        all_conflicts: list[MergeConflict] = []
        all_resolutions: list[DomainResolutionProposal] = []

        # Merge each branch
        for session in successful_sessions:
            merge_outcome = self._merge_branch(session.branch_name)

            if merge_outcome["has_conflicts"]:
                # Get conflict details
                conflicts = self._detect_conflicts(
                    session.branch_name,
                    self._integration_branch,
                )
                all_conflicts.extend(conflicts)

                # Try to resolve each conflict
                for conflict in conflicts:
                    resolution = self._resolve_conflict(conflict)
                    if resolution:
                        all_resolutions.append(resolution)

                        if resolution.confidence >= self.AUTO_ACCEPT_CONFIDENCE:
                            # Apply resolution
                            self._apply_resolution(conflict, resolution)
                        else:
                            # Mark as needing CEO input
                            if resolution.confidence < self.ESCALATE_CONFIDENCE:
                                resolution.needs_ceo = True

            all_merged_files.extend(session.files_changed)

        # Check for unresolved conflicts
        unresolved = self._get_unresolved_conflicts(all_conflicts, all_resolutions)

        # Run tests if enabled and no unresolved conflicts
        test_results = None
        semantic_conflicts: list[MergeConflict] = []

        if self._run_tests and not unresolved:
            test_results = self._run_test_suite()

            if not test_results.get("passed", False):
                # Test failures indicate semantic conflicts
                semantic_conflicts = self._detect_semantic_conflicts(
                    test_results,
                    successful_sessions,
                )
                all_conflicts.extend(semantic_conflicts)

        # Determine overall success
        needs_user_input = bool(unresolved) or any(
            r.needs_ceo for r in all_resolutions
        )

        escalation_reason = None
        if needs_user_input:
            escalation_reason = self._build_escalation_reason(
                unresolved,
                [r for r in all_resolutions if r.needs_ceo],
                semantic_conflicts,
            )

        success = not unresolved and not semantic_conflicts

        return MergeResult(
            success=success,
            merged_branch=self._integration_branch,
            conflicts=all_conflicts,
            resolutions=all_resolutions,
            files_merged=list(set(all_merged_files)),
            test_results=test_results,
            needs_user_input=needs_user_input,
            escalation_reason=escalation_reason,
            summary=self._build_summary(
                successful_sessions,
                all_conflicts,
                all_resolutions,
                test_results,
            ),
        )

    # =========================================================================
    # Branch Operations
    # =========================================================================

    def _create_integration_branch(self, base_branch: str) -> str:
        """Create integration branch from base.

        Args:
            base_branch: Base branch to branch from.

        Returns:
            Name of the integration branch.

        Raises:
            RuntimeError: If branch creation fails.
        """
        import uuid

        integration_branch = f"parallel/integration-{uuid.uuid4().hex[:8]}"

        # Create branch from base
        result = subprocess.run(
            ["git", "checkout", "-b", integration_branch, base_branch],
            cwd=self._root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Try falling back to current HEAD if base_branch doesn't exist
            fallback = subprocess.run(
                ["git", "checkout", "-b", integration_branch],
                cwd=self._root,
                capture_output=True,
                text=True,
            )
            if fallback.returncode != 0:
                raise RuntimeError(
                    f"Failed to create integration branch '{integration_branch}' "
                    f"from '{base_branch}': {result.stderr.strip()}. "
                    f"Fallback also failed: {fallback.stderr.strip()}"
                )

        return integration_branch

    def _merge_branch(self, branch_name: str) -> dict:
        """Merge a branch into the integration branch.

        Args:
            branch_name: Branch to merge.

        Returns:
            Dict with merge outcome.
        """
        result = subprocess.run(
            ["git", "merge", "--no-commit", branch_name],
            cwd=self._root,
            capture_output=True,
            text=True,
        )

        has_conflicts = result.returncode != 0 and "CONFLICT" in result.stdout

        if not has_conflicts and result.returncode == 0:
            # Commit the merge
            subprocess.run(
                ["git", "commit", "-m", f"Merge {branch_name}"],
                cwd=self._root,
                capture_output=True,
            )

        return {
            "has_conflicts": has_conflicts,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def _detect_conflicts(
        self,
        branch_a: str,
        branch_b: str,
    ) -> list[MergeConflict]:
        """Detect merge conflicts between branches.

        Args:
            branch_a: First branch.
            branch_b: Second branch.

        Returns:
            List of detected conflicts.
        """
        conflicts: list[MergeConflict] = []

        # Get list of conflicting files
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )

        conflicting_files = [f.strip() for f in result.stdout.split("\n") if f.strip()]

        for file_path in conflicting_files:
            # Get diffs from each side
            diff_a = self._get_diff(file_path, branch_a)
            diff_b = self._get_diff(file_path, branch_b)

            # Find owning agent
            owning_agent = self._find_owning_agent(file_path)

            conflict = MergeConflict(
                file_path=file_path,
                conflict_type=ConflictType.TEXTUAL,
                description=f"Textual merge conflict in {file_path}",
                branch_a=branch_a,
                branch_b=branch_b,
                diff_a=diff_a,
                diff_b=diff_b,
                owning_agent=owning_agent,
            )
            conflicts.append(conflict)

        return conflicts

    def _get_diff(self, file_path: str, branch: str) -> str:
        """Get diff of a file in a branch vs base.

        Args:
            file_path: Path to the file.
            branch: Branch to diff.

        Returns:
            Diff content.
        """
        result = subprocess.run(
            ["git", "show", f"{branch}:{file_path}"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return ""

        return result.stdout[:2000]  # Limit size for prompt

    def _find_owning_agent(self, file_path: str) -> Optional[str]:
        """Find the agent that owns a file.

        Args:
            file_path: Path to the file.

        Returns:
            Agent ID or None.
        """
        try:
            agents = self._hub.list_agents()
            for agent in agents:
                for context_path in agent.context_paths:
                    # Simple matching - could be improved
                    if file_path.startswith(context_path.replace("/**/*.py", "")):
                        return agent.agent_id
                    if context_path.replace("**/*", "") in file_path:
                        return agent.agent_id
        except Exception:
            pass

        return None

    # =========================================================================
    # Conflict Resolution
    # =========================================================================

    def _resolve_conflict(self, conflict: MergeConflict) -> Optional[DomainResolutionProposal]:
        """Attempt to resolve a conflict using domain agent.

        Args:
            conflict: The conflict to resolve.

        Returns:
            Resolution proposal or None if no resolution possible.
        """
        if not conflict.owning_agent:
            # No owning agent - mark for CEO
            return DomainResolutionProposal(
                agent_id="unknown",
                agent_name="Unknown",
                conflict_file=conflict.file_path,
                proposed_resolution="",
                reasoning="No owning agent found for this file",
                confidence=0.0,
                needs_ceo=True,
            )

        # Get agent info
        agent_info = self._get_agent_info(conflict.owning_agent)

        if not agent_info:
            return None

        # Ask agent to resolve
        prompt = CONFLICT_RESOLUTION_PROMPT.format(
            agent_name=agent_info["name"],
            agent_description=agent_info["description"],
            file_path=conflict.file_path,
            conflict_type=conflict.conflict_type.value,
            branch_a=conflict.branch_a,
            branch_b=conflict.branch_b,
            diff_a=conflict.diff_a,
            diff_b=conflict.diff_b,
        )

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            response_text = response.content[0].text
            data = self._parse_json_response(response_text)

            if not data.get("can_resolve", False):
                return DomainResolutionProposal(
                    agent_id=conflict.owning_agent,
                    agent_name=agent_info["name"],
                    conflict_file=conflict.file_path,
                    proposed_resolution="",
                    reasoning=data.get("reasoning", "Agent could not resolve conflict"),
                    confidence=0.0,
                    needs_ceo=True,
                )

            return DomainResolutionProposal(
                agent_id=conflict.owning_agent,
                agent_name=agent_info["name"],
                conflict_file=conflict.file_path,
                proposed_resolution=data.get("proposed_resolution", ""),
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.5)),
                side_effects=data.get("side_effects", []),
                needs_ceo=data.get("needs_ceo", False),
            )

        except Exception as e:
            return DomainResolutionProposal(
                agent_id=conflict.owning_agent,
                agent_name=agent_info["name"],
                conflict_file=conflict.file_path,
                proposed_resolution="",
                reasoning=f"Error during resolution: {e}",
                confidence=0.0,
                needs_ceo=True,
            )

    def _apply_resolution(
        self,
        conflict: MergeConflict,
        resolution: DomainResolutionProposal,
    ) -> bool:
        """Apply a resolution to a conflict.

        Args:
            conflict: The conflict to resolve.
            resolution: The resolution to apply.

        Returns:
            True if successfully applied.
        """
        file_path = self._root / conflict.file_path

        try:
            # Write the resolved content
            file_path.write_text(resolution.proposed_resolution)

            # Stage the file
            subprocess.run(
                ["git", "add", str(file_path)],
                cwd=self._root,
                capture_output=True,
            )

            conflict.auto_resolvable = True
            conflict.suggested_resolution = resolution.proposed_resolution

            return True

        except Exception:
            return False

    def _get_unresolved_conflicts(
        self,
        conflicts: list[MergeConflict],
        resolutions: list[DomainResolutionProposal],
    ) -> list[MergeConflict]:
        """Get conflicts without high-confidence resolutions.

        Args:
            conflicts: All detected conflicts.
            resolutions: All resolution proposals.

        Returns:
            List of unresolved conflicts.
        """
        resolved_files = {
            r.conflict_file
            for r in resolutions
            if r.confidence >= self.AUTO_ACCEPT_CONFIDENCE and not r.needs_ceo
        }

        return [c for c in conflicts if c.file_path not in resolved_files]

    def _get_agent_info(self, agent_id: str) -> Optional[dict]:
        """Get agent info by ID.

        Args:
            agent_id: Agent identifier.

        Returns:
            Dict with agent info or None.
        """
        try:
            agents = self._hub.list_agents()
            for agent in agents:
                if agent.agent_id == agent_id:
                    return {
                        "name": agent.name,
                        "description": agent.description,
                    }
        except Exception:
            pass

        return None

    # =========================================================================
    # Test Execution
    # =========================================================================

    def _run_test_suite(self) -> dict:
        """Run the project's test suite.

        Returns:
            Dict with test results.
        """
        # Auto-detect test command if not specified
        command = self._test_command or self._detect_test_command()

        if not command:
            return {"passed": True, "skipped": True, "reason": "No test command found"}

        try:
            result = subprocess.run(
                command.split(),
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout[:5000],
                "stderr": result.stderr[:5000],
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Test suite timed out"}
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _detect_test_command(self) -> Optional[str]:
        """Auto-detect the test command for the project.

        Returns:
            Test command or None.
        """
        # Check for common test configurations
        if (self._root / "package.json").exists():
            return "npm test"

        if (self._root / "pytest.ini").exists() or (self._root / "pyproject.toml").exists():
            return "pytest"

        if (self._root / "Cargo.toml").exists():
            return "cargo test"

        if (self._root / "go.mod").exists():
            return "go test ./..."

        if (self._root / "Makefile").exists():
            # Check if Makefile has test target
            makefile = (self._root / "Makefile").read_text()
            if "test:" in makefile:
                return "make test"

        return None

    def _detect_semantic_conflicts(
        self,
        test_results: dict,
        sessions: list[SessionResult],
    ) -> list[MergeConflict]:
        """Detect semantic conflicts from test failures.

        Semantic conflicts are when changes from different sessions
        break each other without causing textual merge conflicts.

        Args:
            test_results: Results from test suite.
            sessions: Session results that were merged.

        Returns:
            List of semantic conflicts detected.
        """
        conflicts: list[MergeConflict] = []

        # Parse test output for failure info
        stdout = test_results.get("stdout", "")
        stderr = test_results.get("stderr", "")

        # Look for file references in test failures
        failing_files = self._extract_failing_files(stdout + stderr)

        for file_path in failing_files:
            # Check if multiple sessions touched related files
            touching_sessions = [
                s for s in sessions
                if file_path in s.files_changed
                or any(f.startswith(file_path.rsplit("/", 1)[0]) for f in s.files_changed)
            ]

            if len(touching_sessions) >= 2:
                conflict = MergeConflict(
                    file_path=file_path,
                    conflict_type=ConflictType.SEMANTIC,
                    description=f"Test failure likely caused by conflicting changes to {file_path}",
                    branch_a=touching_sessions[0].branch_name,
                    branch_b=touching_sessions[1].branch_name,
                    owning_agent=self._find_owning_agent(file_path),
                )
                conflicts.append(conflict)

        return conflicts

    def _extract_failing_files(self, output: str) -> list[str]:
        """Extract file paths from test failure output.

        Args:
            output: Test output text.

        Returns:
            List of file paths mentioned in failures.
        """
        # Common patterns for file paths in test output
        patterns = [
            r"(?:File|in|at)\s+[\"']?([^\s\"']+\.(?:py|ts|tsx|js|jsx))[\"']?",
            r"([a-zA-Z_][a-zA-Z0-9_/]*\.(?:py|ts|tsx|js|jsx)):\d+",
        ]

        files = set()
        for pattern in patterns:
            matches = re.findall(pattern, output)
            files.update(matches)

        return list(files)

    # =========================================================================
    # Reporting
    # =========================================================================

    def _build_escalation_reason(
        self,
        unresolved: list[MergeConflict],
        low_confidence: list[DomainResolutionProposal],
        semantic: list[MergeConflict],
    ) -> str:
        """Build explanation for CEO escalation.

        Args:
            unresolved: Unresolved conflicts.
            low_confidence: Low-confidence resolutions needing review.
            semantic: Semantic conflicts from test failures.

        Returns:
            Escalation reason message.
        """
        parts = []

        if unresolved:
            files = ", ".join(c.file_path for c in unresolved[:5])
            parts.append(f"**Unresolved Conflicts ({len(unresolved)}):** {files}")

        if low_confidence:
            files = ", ".join(r.conflict_file for r in low_confidence[:5])
            parts.append(
                f"**Low-Confidence Resolutions ({len(low_confidence)}):** "
                f"Proposed resolutions for {files} need your review"
            )

        if semantic:
            files = ", ".join(c.file_path for c in semantic[:5])
            parts.append(
                f"**Semantic Conflicts ({len(semantic)}):** "
                f"Tests fail after merge, likely due to conflicts in {files}"
            )

        return "\n\n".join(parts)

    def _build_summary(
        self,
        sessions: list[SessionResult],
        conflicts: list[MergeConflict],
        resolutions: list[DomainResolutionProposal],
        test_results: Optional[dict],
    ) -> str:
        """Build human-readable summary of merge.

        Args:
            sessions: Merged session results.
            conflicts: All conflicts.
            resolutions: All resolutions.
            test_results: Test suite results.

        Returns:
            Summary message.
        """
        lines = [
            f"Merged {len(sessions)} parallel sessions.",
        ]

        if conflicts:
            auto_resolved = sum(1 for c in conflicts if c.auto_resolvable)
            lines.append(
                f"Conflicts: {len(conflicts)} total, {auto_resolved} auto-resolved"
            )

        if test_results:
            if test_results.get("skipped"):
                lines.append("Tests: Skipped (no test command)")
            elif test_results.get("passed"):
                lines.append("Tests: Passed")
            else:
                lines.append("Tests: Failed")

        return " ".join(lines)

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response.

        Args:
            text: Response text.

        Returns:
            Parsed dict or empty dict.
        """
        # Try to extract JSON from code blocks
        json_match = re.search(r"```(?:json)?\n?(.*?)```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try parsing whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        return {}

    # =========================================================================
    # Cleanup and Rollback
    # =========================================================================

    def abort_merge(self) -> None:
        """Abort the current merge and reset to clean state."""
        subprocess.run(
            ["git", "merge", "--abort"],
            cwd=self._root,
            capture_output=True,
        )

    def delete_integration_branch(self) -> None:
        """Delete the integration branch."""
        if self._integration_branch:
            # First checkout a different branch
            subprocess.run(
                ["git", "checkout", "-"],
                cwd=self._root,
                capture_output=True,
            )

            # Delete the branch
            subprocess.run(
                ["git", "branch", "-D", self._integration_branch],
                cwd=self._root,
                capture_output=True,
            )

            self._integration_branch = ""

    def finalize_merge(self, target_branch: str = "main") -> bool:
        """Finalize merge by merging integration branch to target.

        Args:
            target_branch: Branch to merge into (usually main).

        Returns:
            True if successful.
        """
        if not self._integration_branch:
            return False

        try:
            # Checkout target
            subprocess.run(
                ["git", "checkout", target_branch],
                cwd=self._root,
                capture_output=True,
                check=True,
            )

            # Merge integration branch
            result = subprocess.run(
                ["git", "merge", "--no-ff", self._integration_branch, "-m",
                 f"Merge parallel sessions from {self._integration_branch}"],
                cwd=self._root,
                capture_output=True,
            )

            if result.returncode == 0:
                # Cleanup integration branch
                self.delete_integration_branch()
                return True

        except Exception:
            pass

        return False
