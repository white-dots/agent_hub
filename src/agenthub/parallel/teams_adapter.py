from __future__ import annotations
"""Adapter for Claude Code Agent Teams integration.

The AgentTeamsAdapter bridges AgentHub's orchestration layer to Claude Code's
Agent Teams feature, enabling real-time inter-agent messaging and boundary
crossing negotiation.

When Agent Teams is unavailable, falls back gracefully to CLI mode.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from agenthub.parallel.models import (
    BoundaryCrossing,
    BoundaryCrossingResolution,
    CrossingResolutionType,
    SessionResult,
    SessionSpec,
)

if TYPE_CHECKING:
    from agenthub.hub import AgentHub


@dataclass
class TeammateHandle:
    """Handle to an active Agent Teams teammate.

    Represents a running Claude Code instance that can receive messages.
    """

    task_id: str
    teammate_id: str
    """The Claude Code teammate ID assigned by Agent Teams."""

    inbox_path: Path
    """Path to the teammate's inbox file for message passing."""

    outbox_path: Path
    """Path to the teammate's outbox file for responses."""

    worktree_path: Path
    """Git worktree this teammate operates in."""

    branch_name: str
    is_active: bool = True


@dataclass
class InboxMessage:
    """Message in a teammate's inbox.

    Used for inter-agent communication in Agent Teams mode.
    """

    from_task: str
    """Task ID of the sending teammate."""

    to_task: str
    """Task ID of the receiving teammate."""

    message_type: str
    """Type: 'boundary_request', 'boundary_response', 'status', 'escalation'."""

    content: str
    """Message content."""

    timestamp: float = field(default_factory=time.time)
    requires_response: bool = False


class AgentTeamsAdapter:
    """Bridges AgentHub to Claude Code Agent Teams.

    Responsibilities:
    - Spawning teammates (full Claude Code instances)
    - Message routing between teammates
    - Real-time boundary crossing detection and negotiation
    - Graceful fallback to CLI when Agent Teams unavailable

    Example:
        >>> adapter = AgentTeamsAdapter(project_root, hub)
        >>> if adapter.is_available():
        ...     results = await adapter.execute_sessions(specs)
        ... else:
        ...     # Fall back to CLI
        ...     pass
    """

    # Environment variable to enable Agent Teams
    ENV_VAR = "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"

    # Inbox polling interval (seconds)
    INBOX_POLL_INTERVAL = 0.5

    # Timeout for boundary crossing negotiation
    BOUNDARY_NEGOTIATION_TIMEOUT = 30

    def __init__(
        self,
        project_root: str,
        hub: "AgentHub",
        on_boundary_crossing: Optional[Callable[[BoundaryCrossing], None]] = None,
    ):
        """Initialize AgentTeamsAdapter.

        Args:
            project_root: Path to the project root.
            hub: AgentHub for accessing domain agents.
            on_boundary_crossing: Optional callback when boundary crossing detected.
        """
        self._root = Path(project_root).resolve()
        self._hub = hub
        self._on_crossing = on_boundary_crossing

        self._teammates: dict[str, TeammateHandle] = {}
        self._inbox_dir = self._root / ".worktrees" / ".agent_teams" / "inboxes"
        self._outbox_dir = self._root / ".worktrees" / ".agent_teams" / "outboxes"

        # For tracking boundary crossings
        self._pending_crossings: dict[str, BoundaryCrossing] = {}
        self._crossing_resolutions: dict[str, BoundaryCrossingResolution] = {}

    def is_available(self) -> bool:
        """Check if Agent Teams feature is available.

        Checks:
        1. Environment variable is set
        2. Claude CLI supports Agent Teams (via --version check)

        Returns:
            True if Agent Teams can be used.
        """
        # Check env var
        if os.environ.get(self.ENV_VAR, "").lower() != "true":
            return False

        # Verify Claude CLI has Agent Teams support
        try:
            import subprocess
            result = subprocess.run(
                ["claude", "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Agent Teams would be indicated in help output
            return "teams" in result.stdout.lower() or "teammate" in result.stdout.lower()
        except Exception:
            return False

    async def execute_sessions(
        self,
        specs: list[SessionSpec],
        timeout_seconds: int = 300,
    ) -> list[SessionResult]:
        """Execute sessions using Agent Teams.

        Flow:
        1. Spawn a teammate for each spec
        2. Monitor inboxes for boundary crossing requests
        3. Route crossing requests to owning agents
        4. Collect results when all teammates complete

        Args:
            specs: SessionSpecs to execute.
            timeout_seconds: Overall timeout for all sessions.

        Returns:
            List of SessionResult.
        """
        if not self.is_available():
            raise RuntimeError("Agent Teams is not available")

        results: list[SessionResult] = []

        try:
            # Setup messaging directories
            self._setup_messaging_dirs()

            # Spawn teammates
            for spec in specs:
                handle = await self._spawn_teammate(spec)
                self._teammates[spec.task.task_id] = handle

            # Monitor and coordinate until all complete
            results = await self._coordinate_until_complete(timeout_seconds)

        finally:
            # Cleanup
            await self._cleanup_teammates()

        return results

    def _setup_messaging_dirs(self) -> None:
        """Create inbox/outbox directories for inter-agent messaging."""
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        self._outbox_dir.mkdir(parents=True, exist_ok=True)

    async def _spawn_teammate(self, spec: SessionSpec) -> TeammateHandle:
        """Spawn a Claude Code teammate for a session.

        Args:
            spec: SessionSpec defining the teammate's scope.

        Returns:
            TeammateHandle for the spawned teammate.
        """
        task_id = spec.task.task_id

        # Create inbox/outbox for this teammate
        inbox_path = self._inbox_dir / f"{task_id}.jsonl"
        outbox_path = self._outbox_dir / f"{task_id}.jsonl"

        # Initialize empty files
        inbox_path.touch()
        outbox_path.touch()

        # Build teammate spawn command
        # Note: This is a placeholder - actual Agent Teams API may differ
        teammate_id = f"teammate_{task_id}"

        # For now, we'll simulate with a regular Claude CLI call
        # that checks for inbox messages periodically
        enhanced_prompt = self._enhance_prompt_for_teams(spec.prompt, inbox_path)

        cmd = [
            "claude",
            "--print", enhanced_prompt,
            "--output-format", "json",
        ]

        # Spawn the teammate process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=spec.worktree_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
                "AGENT_TEAMS_INBOX": str(inbox_path),
                "AGENT_TEAMS_OUTBOX": str(outbox_path),
                "AGENT_TEAMS_TASK_ID": task_id,
            },
        )

        handle = TeammateHandle(
            task_id=task_id,
            teammate_id=teammate_id,
            inbox_path=inbox_path,
            outbox_path=outbox_path,
            worktree_path=Path(spec.worktree_path),
            branch_name=spec.branch_name,
            is_active=True,
        )

        # Store process reference for later
        handle._process = process  # type: ignore

        return handle

    def _enhance_prompt_for_teams(self, prompt: str, inbox_path: Path) -> str:
        """Enhance a prompt with Agent Teams messaging instructions.

        Args:
            prompt: Original session prompt.
            inbox_path: Path to this teammate's inbox.

        Returns:
            Enhanced prompt with messaging instructions.
        """
        teams_instructions = f"""
## Agent Teams Mode

You are running as part of an Agent Teams session. Other teammates are working
on parallel tasks in separate worktrees.

### Boundary Crossing Protocol

If you need to modify or read a file that is OUTSIDE your scoped files, you
MUST signal this by outputting:

```
[BOUNDARY_CROSSING: path/to/file - reason for needing this file]
```

An orchestrator is monitoring for these signals and will coordinate with the
owning teammate.

### Message Inbox

Your inbox is at: {inbox_path}
Check it periodically for messages from other teammates or the orchestrator.

Message format (JSONL):
{{"from_task": "task_id", "to_task": "your_task_id", "message_type": "type", "content": "..."}}

If you receive a boundary_request, evaluate it and respond via stdout:
```
[BOUNDARY_RESPONSE: approve|reject|modify - reason]
```

"""
        return teams_instructions + "\n\n" + prompt

    async def _coordinate_until_complete(
        self,
        timeout_seconds: int,
    ) -> list[SessionResult]:
        """Monitor teammates and coordinate until all complete.

        Handles:
        - Inbox polling for boundary crossings
        - Message routing between teammates
        - Timeout enforcement

        Args:
            timeout_seconds: Overall timeout.

        Returns:
            List of SessionResult.
        """
        start_time = time.time()
        results: list[SessionResult] = []

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                # Force completion with timeout errors
                for task_id, handle in self._teammates.items():
                    if handle.is_active:
                        results.append(SessionResult(
                            task_id=task_id,
                            branch_name=handle.branch_name,
                            success=False,
                            error=f"Session timed out after {timeout_seconds}s",
                            time_seconds=elapsed,
                            execution_backend="agent_teams",
                        ))
                break

            # Check for completed teammates
            all_complete = True
            for task_id, handle in self._teammates.items():
                if handle.is_active:
                    process = getattr(handle, "_process", None)
                    if process and process.returncode is not None:
                        # Process completed
                        result = await self._collect_result(handle, process)
                        results.append(result)
                        handle.is_active = False
                    else:
                        all_complete = False

            if all_complete:
                break

            # Poll inboxes for boundary crossing requests
            await self._poll_and_route_messages()

            # Small sleep to avoid busy loop
            await asyncio.sleep(self.INBOX_POLL_INTERVAL)

        return results

    async def _poll_and_route_messages(self) -> None:
        """Poll outboxes and route messages to appropriate inboxes."""
        for task_id, handle in self._teammates.items():
            if not handle.is_active:
                continue

            # Read outbox
            outbox_path = handle.outbox_path
            if not outbox_path.exists():
                continue

            try:
                with open(outbox_path, "r") as f:
                    lines = f.readlines()

                # Process each message
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        msg_data = json.loads(line)
                        msg = InboxMessage(
                            from_task=msg_data.get("from_task", task_id),
                            to_task=msg_data.get("to_task", ""),
                            message_type=msg_data.get("message_type", ""),
                            content=msg_data.get("content", ""),
                        )
                        await self._route_message(msg)
                    except json.JSONDecodeError:
                        # Check for boundary crossing pattern
                        crossing = self._parse_boundary_crossing_from_output(line, task_id)
                        if crossing:
                            await self._handle_boundary_crossing(crossing)

                # Clear processed messages
                outbox_path.write_text("")

            except Exception as e:
                print(f"Warning: Failed to poll outbox for {task_id}: {e}")

    async def _route_message(self, msg: InboxMessage) -> None:
        """Route a message to the target teammate's inbox.

        Args:
            msg: Message to route.
        """
        target_task = msg.to_task
        if target_task not in self._teammates:
            print(f"Warning: Unknown target task {target_task}")
            return

        target_handle = self._teammates[target_task]
        inbox_path = target_handle.inbox_path

        # Append to inbox
        with open(inbox_path, "a") as f:
            f.write(json.dumps({
                "from_task": msg.from_task,
                "to_task": msg.to_task,
                "message_type": msg.message_type,
                "content": msg.content,
                "timestamp": msg.timestamp,
            }) + "\n")

    def _parse_boundary_crossing_from_output(
        self,
        output_line: str,
        task_id: str,
    ) -> Optional[BoundaryCrossing]:
        """Parse a boundary crossing from session output.

        Args:
            output_line: Line from session output.
            task_id: Task ID of the session.

        Returns:
            BoundaryCrossing if found, None otherwise.
        """
        pattern = r"\[BOUNDARY_CROSSING:\s*([^\-]+)\s*-\s*([^\]]+)\]"
        match = re.search(pattern, output_line)
        if not match:
            return None

        target_file = match.group(1).strip()
        reason = match.group(2).strip()

        # Determine owning agent
        owning_agent = self._find_owning_agent(target_file)

        return BoundaryCrossing(
            session_task_id=task_id,
            requesting_agent=task_id,
            target_file=target_file,
            owning_agent=owning_agent,
            reason=reason,
            detected_at="real_time",
        )

    def _find_owning_agent(self, file_path: str) -> Optional[str]:
        """Find the agent that owns a file.

        Args:
            file_path: Path to the file.

        Returns:
            Agent ID or None if no specific owner.
        """
        try:
            # Use hub to find owning agent
            agents = self._hub.list_agents()
            for agent in agents:
                for context_path in agent.context_paths:
                    if file_path.startswith(context_path) or context_path in file_path:
                        return agent.agent_id
        except Exception:
            pass

        return None

    async def _handle_boundary_crossing(self, crossing: BoundaryCrossing) -> None:
        """Handle a detected boundary crossing.

        Routes to owning agent for approval/rejection.

        Args:
            crossing: The boundary crossing to handle.
        """
        crossing_id = f"{crossing.session_task_id}:{crossing.target_file}"
        self._pending_crossings[crossing_id] = crossing

        # Notify callback if registered
        if self._on_crossing:
            self._on_crossing(crossing)

        # If owning agent is in our teammates, send message
        if crossing.owning_agent:
            # Find teammate with this agent
            for task_id, handle in self._teammates.items():
                # Check if this teammate handles the owning agent
                # (This is a simplified check - real impl would use agent mapping)
                msg = InboxMessage(
                    from_task=crossing.session_task_id,
                    to_task=task_id,
                    message_type="boundary_request",
                    content=json.dumps({
                        "file": crossing.target_file,
                        "reason": crossing.reason,
                        "requesting_task": crossing.session_task_id,
                    }),
                    requires_response=True,
                )
                await self._route_message(msg)
                break

    async def _collect_result(
        self,
        handle: TeammateHandle,
        process,
    ) -> SessionResult:
        """Collect result from a completed teammate.

        Args:
            handle: TeammateHandle of the completed teammate.
            process: The asyncio subprocess.

        Returns:
            SessionResult with execution details.
        """
        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode() if stdout else ""
        stderr_text = stderr.decode() if stderr else ""

        success = process.returncode == 0

        # Parse boundary crossings from output
        boundary_crossings = self._parse_all_boundary_crossings(stdout_text)

        # Get files changed
        files_changed = self._get_files_changed(handle.worktree_path, handle.branch_name)

        return SessionResult(
            task_id=handle.task_id,
            branch_name=handle.branch_name,
            success=success,
            files_changed=files_changed,
            stdout=stdout_text,
            error=stderr_text if not success else None,
            boundary_crossings=boundary_crossings,
            execution_backend="agent_teams",
        )

    def _parse_all_boundary_crossings(self, stdout: str) -> list[str]:
        """Parse all boundary crossings from output.

        Args:
            stdout: Full session output.

        Returns:
            List of boundary crossing descriptions.
        """
        pattern = r"\[BOUNDARY_CROSSING:\s*([^\]]+)\]"
        return re.findall(pattern, stdout)

    def _get_files_changed(self, worktree_path: Path, branch_name: str) -> list[str]:
        """Get files changed in the worktree.

        Args:
            worktree_path: Path to the worktree.
            branch_name: Branch name.

        Returns:
            List of changed file paths.
        """
        import subprocess

        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1..HEAD"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Try against base
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return []
            return [line[3:] for line in result.stdout.strip().split("\n") if line]

        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

    async def _cleanup_teammates(self) -> None:
        """Clean up all teammates and messaging infrastructure."""
        # Kill any still-running processes
        for task_id, handle in self._teammates.items():
            process = getattr(handle, "_process", None)
            if process and process.returncode is None:
                process.kill()
                try:
                    await process.wait()
                except Exception:
                    pass

        # Clean up inbox/outbox files
        for task_id in self._teammates:
            try:
                (self._inbox_dir / f"{task_id}.jsonl").unlink(missing_ok=True)
                (self._outbox_dir / f"{task_id}.jsonl").unlink(missing_ok=True)
            except Exception:
                pass

        self._teammates.clear()

    # =========================================================================
    # Boundary Crossing Resolution
    # =========================================================================

    async def negotiate_boundary_crossing(
        self,
        crossing: BoundaryCrossing,
        timeout_seconds: float = 30,
    ) -> BoundaryCrossingResolution:
        """Negotiate a boundary crossing with the owning agent.

        Sends request to owning agent and waits for response.

        Args:
            crossing: The boundary crossing to negotiate.
            timeout_seconds: Timeout for negotiation.

        Returns:
            BoundaryCrossingResolution with the outcome.
        """
        crossing_id = f"{crossing.session_task_id}:{crossing.target_file}"

        # If already resolved, return cached resolution
        if crossing_id in self._crossing_resolutions:
            return self._crossing_resolutions[crossing_id]

        # If no owning agent, auto-approve with low confidence
        if not crossing.owning_agent:
            resolution = BoundaryCrossingResolution(
                crossing=crossing,
                approved=True,
                resolution_type=CrossingResolutionType.APPROVED_AS_IS,
                reasoning="No specific owning agent found for this file",
                confidence=0.5,
            )
            self._crossing_resolutions[crossing_id] = resolution
            return resolution

        # Send negotiation request
        await self._handle_boundary_crossing(crossing)

        # Wait for response (poll outboxes)
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            # Check for resolution in crossing resolutions
            if crossing_id in self._crossing_resolutions:
                return self._crossing_resolutions[crossing_id]

            await asyncio.sleep(0.5)

        # Timeout - escalate to CEO
        resolution = BoundaryCrossingResolution(
            crossing=crossing,
            approved=False,
            resolution_type=CrossingResolutionType.ESCALATED_TO_CEO,
            reasoning=f"Negotiation timed out after {timeout_seconds}s",
            confidence=0.0,
        )
        self._crossing_resolutions[crossing_id] = resolution
        return resolution

    def get_pending_crossings(self) -> list[BoundaryCrossing]:
        """Get all pending boundary crossings.

        Returns:
            List of unresolved boundary crossings.
        """
        resolved_ids = set(self._crossing_resolutions.keys())
        return [
            c for cid, c in self._pending_crossings.items()
            if cid not in resolved_ids
        ]

    def get_crossing_resolutions(self) -> list[BoundaryCrossingResolution]:
        """Get all boundary crossing resolutions.

        Returns:
            List of all resolutions.
        """
        return list(self._crossing_resolutions.values())
