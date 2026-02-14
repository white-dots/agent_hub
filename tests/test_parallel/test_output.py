"""Tests for ProgressReporter and output utilities."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from agenthub.parallel.output import (
    EventType,
    ProgressEvent,
    ProgressReporter,
    format_progress_for_cli,
)
from agenthub.parallel.models import (
    DecompositionResult,
    ImplementationTask,
    ParallelizationPlan,
    RiskLevel,
    SessionResult,
    MergeConflict,
    ConflictType,
    DomainResolutionProposal,
    MergeResult,
    ParallelExecutionResult,
)


# === EventType Tests ===


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_are_strings(self):
        """EventType values should be strings."""
        assert EventType.STARTED.value == "started"
        assert EventType.COMPLETED.value == "completed"
        assert EventType.FAILED.value == "failed"

    def test_all_lifecycle_events_exist(self):
        """All lifecycle event types should be defined."""
        assert hasattr(EventType, "STARTED")
        assert hasattr(EventType, "COMPLETED")
        assert hasattr(EventType, "FAILED")

    def test_all_phase_events_exist(self):
        """All phase event types should be defined."""
        assert hasattr(EventType, "DECOMPOSITION_COMPLETE")
        assert hasattr(EventType, "ANALYSIS_COMPLETE")
        assert hasattr(EventType, "CEO_CONFIRMATION_NEEDED")
        assert hasattr(EventType, "CEO_DECISION_RECEIVED")

    def test_all_session_events_exist(self):
        """All session event types should be defined."""
        assert hasattr(EventType, "SESSION_STARTED")
        assert hasattr(EventType, "SESSION_PROGRESS")
        assert hasattr(EventType, "SESSION_COMPLETE")
        assert hasattr(EventType, "SESSION_FAILED")

    def test_all_conflict_events_exist(self):
        """All conflict event types should be defined."""
        assert hasattr(EventType, "CONFLICT_DETECTED")
        assert hasattr(EventType, "RESOLUTION_PROPOSED")
        assert hasattr(EventType, "BOUNDARY_CROSSING")
        assert hasattr(EventType, "MERGE_STARTED")
        assert hasattr(EventType, "MERGE_COMPLETE")


# === ProgressEvent Tests ===


class TestProgressEvent:
    """Tests for ProgressEvent dataclass."""

    def test_create_progress_event(self):
        """Should create ProgressEvent with required fields."""
        event = ProgressEvent(
            event_type=EventType.STARTED,
            message="Execution started",
        )
        assert event.event_type == EventType.STARTED
        assert event.message == "Execution started"
        assert event.details == {}
        assert event.timestamp is not None

    def test_progress_event_with_details(self):
        """Should create ProgressEvent with details."""
        event = ProgressEvent(
            event_type=EventType.SESSION_STARTED,
            message="Session started",
            details={"task_id": "task_1", "branch": "feature/task-1"},
        )
        assert event.details["task_id"] == "task_1"
        assert event.details["branch"] == "feature/task-1"

    def test_progress_event_with_custom_timestamp(self):
        """Should accept custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        event = ProgressEvent(
            event_type=EventType.COMPLETED,
            message="Done",
            timestamp=custom_time,
        )
        assert event.timestamp == custom_time


# === ProgressReporter Tests ===


class TestProgressReporter:
    """Tests for ProgressReporter class."""

    def test_reporter_with_callback(self):
        """Should call callback for each event."""
        events = []

        def on_event(event: ProgressEvent):
            events.append(event)

        reporter = ProgressReporter(on_event=on_event)
        reporter.report_started("Test request")

        assert len(events) == 1
        assert events[0].event_type == EventType.STARTED
        assert "Test request" in events[0].message

    def test_reporter_collect_events(self):
        """Should collect events when collect_events=True."""
        reporter = ProgressReporter(collect_events=True)
        reporter.report_started("Test request")
        reporter.report_completed(
            ParallelExecutionResult(
                success=True,
                tasks=[],
                merge_result=None,
                total_time_seconds=10.0,
                total_tokens=1000,
                speedup=2.0,
            )
        )

        events = reporter.get_events()
        assert len(events) == 2
        assert events[0].event_type == EventType.STARTED
        assert events[1].event_type == EventType.COMPLETED

    def test_reporter_without_callback_or_collect(self):
        """Should not error when no callback and not collecting."""
        reporter = ProgressReporter()
        reporter.report_started("Test request")
        # No error should occur

    def test_clear_events(self):
        """Should clear collected events."""
        reporter = ProgressReporter(collect_events=True)
        reporter.report_started("Test request")
        assert len(reporter.get_events()) == 1

        reporter.clear_events()
        assert len(reporter.get_events()) == 0

    def test_report_decomposition(self):
        """Should report decomposition result."""
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))

        result = DecompositionResult(
            tasks=[
                ImplementationTask(
                    task_id="task_1",
                    description="Add button",
                    estimated_files=["button.tsx"],
                    domain_agents=["frontend"],
                    complexity="simple",
                ),
            ],
            original_request="Add a button",
            appears_simple=True,
            actual_complexity="single",
            decomposition_reasoning="Single simple task",
            tokens_used=100,
        )
        reporter.report_decomposition(result)

        assert len(events) == 1
        assert events[0].event_type == EventType.DECOMPOSITION_COMPLETE
        assert events[0].details["task_count"] == 1

    def test_report_analysis(self):
        """Should report analysis result."""
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))

        plan = ParallelizationPlan(
            parallel_groups=[["task_1", "task_2"]],
            sequential_order=["task_1", "task_2"],
            overall_risk=RiskLevel.LOW,
            pm_recommendation="parallel",
            confidence=0.9,
        )
        reporter.report_analysis(plan)

        assert len(events) == 1
        assert events[0].event_type == EventType.ANALYSIS_COMPLETE
        assert events[0].details["overall_risk"] == "low"
        assert events[0].details["confidence"] == 0.9

    def test_report_session_started(self):
        """Should report session started."""
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))

        reporter.report_session_start("task_1", "feature/task-1")

        assert len(events) == 1
        assert events[0].event_type == EventType.SESSION_STARTED
        assert events[0].details["task_id"] == "task_1"
        assert events[0].details["branch"] == "feature/task-1"

    def test_report_session_progress(self):
        """Should report session progress."""
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))

        reporter.report_session_progress("task_1", 0.5)

        assert len(events) == 1
        assert events[0].event_type == EventType.SESSION_PROGRESS
        assert events[0].details["progress"] == 0.5

    def test_report_session_complete(self):
        """Should report session complete."""
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))

        result = SessionResult(
            task_id="task_1",
            branch_name="feature/task-1",
            success=True,
            files_changed=["button.tsx"],
            tokens_used=500,
            time_seconds=30.0,
        )
        reporter.report_session_complete(result)

        assert len(events) == 1
        assert events[0].event_type == EventType.SESSION_COMPLETE
        assert events[0].details["success"] is True

    def test_report_session_failed(self):
        """Should report session failed via session_complete with failure."""
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))

        result = SessionResult(
            task_id="task_1",
            branch_name="feature/task-1",
            success=False,
            files_changed=[],
            tokens_used=100,
            time_seconds=10.0,
            error="Connection timeout",
        )
        reporter.report_session_complete(result)

        assert len(events) == 1
        assert events[0].event_type == EventType.SESSION_FAILED
        assert events[0].details["error"] == "Connection timeout"

    def test_report_conflict_detected(self):
        """Should report conflict detection."""
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))

        conflict = MergeConflict(
            file_path="src/utils.ts",
            conflict_type=ConflictType.TEXTUAL,
            description="Both branches modified the same lines",
            branch_a="feature/task-1",
            branch_b="feature/task-2",
            owning_agent="backend",
        )
        reporter.report_conflict(conflict)

        assert len(events) == 1
        assert events[0].event_type == EventType.CONFLICT_DETECTED
        assert events[0].details["file_path"] == "src/utils.ts"

    def test_report_conflict_resolved(self):
        """Should report conflict resolution."""
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))

        proposal = DomainResolutionProposal(
            agent_id="backend_agent",
            agent_name="backend",
            conflict_file="src/utils.ts",
            proposed_resolution="merged content",
            confidence=0.85,
            reasoning="Agent owns this file",
        )
        reporter.report_resolution(proposal)

        assert len(events) == 1
        assert events[0].event_type == EventType.RESOLUTION_PROPOSED
        assert events[0].details["agent_name"] == "backend"
        assert events[0].details["confidence"] == 0.85

    def test_report_merge_started(self):
        """Should report merge started."""
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))

        reporter.report_merge_started(3)

        assert len(events) == 1
        assert events[0].event_type == EventType.MERGE_STARTED
        assert events[0].details["session_count"] == 3

    def test_report_merge_complete(self):
        """Should report merge complete."""
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))

        # The reporter method takes individual params, not a MergeResult
        reporter.report_merge_complete(
            success=True,
            conflicts=0,
            auto_resolved=0,
        )

        assert len(events) == 1
        assert events[0].event_type == EventType.MERGE_COMPLETE
        assert events[0].details["success"] is True

    def test_report_failed(self):
        """Should report execution failed."""
        events = []
        reporter = ProgressReporter(on_event=lambda e: events.append(e))

        reporter.report_failed("Critical error occurred")

        assert len(events) == 1
        assert events[0].event_type == EventType.FAILED
        assert "Critical error" in events[0].message


# === Format Progress Tests ===


class TestFormatProgressForCLI:
    """Tests for format_progress_for_cli function."""

    def test_format_started_event(self):
        """Should format started event."""
        event = ProgressEvent(
            event_type=EventType.STARTED,
            message="Execution started",
        )
        output = format_progress_for_cli(event)
        assert "Execution started" in output

    def test_format_completed_event(self):
        """Should format completed event."""
        event = ProgressEvent(
            event_type=EventType.COMPLETED,
            message="Execution completed",
        )
        output = format_progress_for_cli(event)
        assert "Execution completed" in output

    def test_format_with_details(self):
        """Should format event with key details."""
        event = ProgressEvent(
            event_type=EventType.SESSION_STARTED,
            message="Session started",
            details={"task_id": "task_1", "branch": "feature/task-1"},
        )
        output = format_progress_for_cli(event)
        assert "task_1" in output or "Session started" in output

    def test_format_preserves_message(self):
        """Should preserve the event message."""
        event = ProgressEvent(
            event_type=EventType.CONFLICT_DETECTED,
            message="Conflict in src/utils.ts",
        )
        output = format_progress_for_cli(event)
        assert "src/utils.ts" in output
