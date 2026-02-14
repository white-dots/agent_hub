"""Tests for DashboardReporter and dashboard integration."""

import json
from unittest.mock import MagicMock, patch, ANY

import pytest

from agenthub.parallel.dashboard import (
    DashboardReporter,
    broadcast_parallel_event,
    create_dashboard_callback,
    DASHBOARD_URL,
)
from agenthub.parallel.output import EventType, ProgressEvent
from agenthub.parallel.models import (
    DecompositionResult,
    ImplementationTask,
    ParallelizationPlan,
    RiskLevel,
    SessionResult,
    MergeResult,
    ParallelExecutionResult,
    MergeConflict,
    ConflictType,
    DomainResolutionProposal,
    BoundaryCrossing,
)


# === broadcast_parallel_event Tests ===


class TestBroadcastParallelEvent:
    """Tests for broadcast_parallel_event function."""

    def test_broadcast_success(self):
        """Should return True when broadcast succeeds."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            result = broadcast_parallel_event(
                "test_event",
                "Test description",
                {"key": "value"},
            )
            assert result is True
            mock_urlopen.assert_called_once()

    def test_broadcast_includes_correct_data(self):
        """Should send correct JSON payload."""
        with patch("urllib.request.Request") as mock_request, patch(
            "urllib.request.urlopen"
        ):
            broadcast_parallel_event(
                "test_event",
                "Test description",
                {"key": "value"},
            )

            # Check the Request was created with correct URL
            call_args = mock_request.call_args
            assert DASHBOARD_URL in call_args[0][0]
            assert "/api/parallel/events" in call_args[0][0]

            # Check the data
            data = json.loads(call_args[1]["data"].decode("utf-8"))
            assert data["event_type"] == "test_event"
            assert data["description"] == "Test description"
            assert data["details"]["key"] == "value"
            assert data["source"] == "parallel_sessions"
            assert "timestamp" in data

    def test_broadcast_failure_returns_false(self):
        """Should return False when broadcast fails."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = ConnectionRefusedError()
            result = broadcast_parallel_event(
                "test_event",
                "Test description",
            )
            assert result is False

    def test_broadcast_timeout_returns_false(self):
        """Should return False on timeout."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError()
            result = broadcast_parallel_event(
                "test_event",
                "Test description",
            )
            assert result is False

    def test_broadcast_custom_url(self):
        """Should use custom dashboard URL when provided."""
        custom_url = "http://custom-dashboard:8080"
        with patch("urllib.request.Request") as mock_request, patch(
            "urllib.request.urlopen"
        ):
            broadcast_parallel_event(
                "test_event",
                "Test description",
                dashboard_url=custom_url,
            )
            call_args = mock_request.call_args
            assert custom_url in call_args[0][0]


# === DashboardReporter Tests ===


class TestDashboardReporter:
    """Tests for DashboardReporter class."""

    @pytest.fixture
    def mock_broadcast(self):
        """Patch broadcast_parallel_event."""
        with patch(
            "agenthub.parallel.dashboard.broadcast_parallel_event"
        ) as mock:
            mock.return_value = True
            yield mock

    def test_reporter_creation(self):
        """Should create reporter with default URL."""
        reporter = DashboardReporter()
        assert reporter._url == DASHBOARD_URL

    def test_reporter_custom_url(self):
        """Should accept custom URL."""
        custom_url = "http://custom:8080"
        reporter = DashboardReporter(dashboard_url=custom_url)
        assert reporter._url == custom_url

    def test_set_execution_id(self, mock_broadcast):
        """Should attach execution ID to events."""
        reporter = DashboardReporter()
        reporter.set_execution_id("exec-123")
        reporter.report_started("Test request")

        call_args = mock_broadcast.call_args
        assert call_args[0][2]["execution_id"] == "exec-123"

    def test_report_started(self, mock_broadcast):
        """Should report execution started."""
        reporter = DashboardReporter()
        reporter.report_started("Add save button and chart")

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "parallel_started"
        assert "Add save button" in args[1]

    def test_report_completed(self, mock_broadcast):
        """Should report execution completed."""
        reporter = DashboardReporter()
        result = ParallelExecutionResult(
            success=True,
            tasks=[],
            merge_result=None,
            total_time_seconds=30.0,
            total_tokens=1500,
            speedup=2.5,
        )
        reporter.report_completed(result)

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "parallel_completed"
        assert args[2]["success"] is True
        assert args[2]["speedup"] == 2.5

    def test_report_failed(self, mock_broadcast):
        """Should report execution failed."""
        reporter = DashboardReporter()
        reporter.report_failed("Critical error")

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "parallel_failed"
        assert "Critical error" in args[1]

    def test_report_decomposition(self, mock_broadcast):
        """Should report task decomposition."""
        reporter = DashboardReporter()
        result = DecompositionResult(
            tasks=[
                ImplementationTask(
                    task_id="task_1",
                    description="Add button",
                    estimated_files=["button.tsx"],
                    domain_agents=["frontend"],
                    complexity="simple",
                ),
                ImplementationTask(
                    task_id="task_2",
                    description="Add chart",
                    estimated_files=["chart.tsx"],
                    domain_agents=["frontend"],
                    complexity="moderate",
                ),
            ],
            original_request="Add button and chart",
            appears_simple=False,
            actual_complexity="multi_independent",
            decomposition_reasoning="Two independent UI tasks",
            tokens_used=200,
        )
        reporter.report_decomposition(result)

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "task_decomposed"
        assert args[2]["task_count"] == 2

    def test_report_analysis(self, mock_broadcast):
        """Should report risk analysis."""
        reporter = DashboardReporter()
        plan = ParallelizationPlan(
            parallel_groups=[["task_1", "task_2"]],
            sequential_order=["task_1", "task_2"],
            overall_risk=RiskLevel.LOW,
            pm_recommendation="parallel",
            confidence=0.9,
        )
        reporter.report_analysis(plan)

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "risk_analyzed"
        assert args[2]["risk"] == "low"

    def test_report_ceo_needed(self, mock_broadcast):
        """Should report CEO confirmation needed."""
        reporter = DashboardReporter()
        plan = ParallelizationPlan(
            parallel_groups=[],
            sequential_order=["task_1"],
            overall_risk=RiskLevel.HIGH,
            pm_recommendation="sequential",
        )
        reporter.report_ceo_needed(plan, "High risk due to overlapping files")

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "ceo_confirmation_needed"
        assert "HIGH" in args[1]

    def test_report_ceo_decision(self, mock_broadcast):
        """Should report CEO decision."""
        reporter = DashboardReporter()
        reporter.report_ceo_decision("proceed_parallel")

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "ceo_decision"
        assert args[2]["decision"] == "proceed_parallel"

    def test_report_session_started(self, mock_broadcast):
        """Should report session started."""
        reporter = DashboardReporter()
        reporter.report_session_started("task_1", "feature/task-1")

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "session_started"
        assert args[2]["task_id"] == "task_1"
        assert args[2]["branch"] == "feature/task-1"

    def test_report_session_progress(self, mock_broadcast):
        """Should report session progress."""
        reporter = DashboardReporter()
        reporter.report_session_progress("task_1", 0.75)

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "session_progress"
        assert "75%" in args[1]

    def test_report_session_complete(self, mock_broadcast):
        """Should report session complete."""
        reporter = DashboardReporter()
        result = SessionResult(
            task_id="task_1",
            branch_name="feature/task-1",
            success=True,
            files_changed=["button.tsx"],
            tokens_used=500,
            time_seconds=30.0,
        )
        reporter.report_session_complete(result)

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "session_completed"
        assert args[2]["success"] is True

    def test_report_merge_started(self, mock_broadcast):
        """Should report merge phase started."""
        reporter = DashboardReporter()
        reporter.report_merge_started(3)

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "merge_started"
        assert args[2]["session_count"] == 3

    def test_report_merge_complete(self, mock_broadcast):
        """Should report merge complete."""
        reporter = DashboardReporter()
        result = MergeResult(
            success=True,
            merged_branch="main",
            conflicts=[],
            resolutions=[],
            files_merged=["utils.ts", "button.tsx"],
            needs_user_input=False,
        )
        reporter.report_merge_complete(result)

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "merge_completed"
        assert args[2]["files_merged"] == 2

    def test_report_boundary_crossing(self, mock_broadcast):
        """Should report boundary crossing."""
        reporter = DashboardReporter()
        crossing = BoundaryCrossing(
            session_task_id="task_1",
            requesting_agent="frontend_agent",
            target_file="src/shared/utils.ts",
            owning_agent="backend",
            reason="File outside session scope",
        )
        reporter.report_boundary_crossing(crossing)

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "boundary_crossing"
        assert args[2]["owning_agent"] == "backend"

    def test_report_conflict(self, mock_broadcast):
        """Should report merge conflict."""
        reporter = DashboardReporter()
        conflict = MergeConflict(
            file_path="src/utils.ts",
            conflict_type=ConflictType.TEXTUAL,
            description="Both branches modified the same lines",
            branch_a="feature/task-1",
            branch_b="feature/task-2",
            owning_agent="backend_agent",
        )
        reporter.report_conflict(conflict)

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "conflict_detected"
        assert args[2]["file_path"] == "src/utils.ts"

    def test_report_resolution(self, mock_broadcast):
        """Should report conflict resolution."""
        reporter = DashboardReporter()
        proposal = DomainResolutionProposal(
            agent_id="backend_agent",
            agent_name="backend",
            conflict_file="src/utils.ts",
            proposed_resolution="merged content",
            confidence=0.9,
            reasoning="Agent owns this file",
        )
        reporter.report_resolution(proposal)

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "conflict_resolved"
        assert args[2]["agent"] == "backend"

    def test_forward_progress_event(self, mock_broadcast):
        """Should forward ProgressEvent to dashboard."""
        reporter = DashboardReporter()
        event = ProgressEvent(
            event_type=EventType.SESSION_STARTED,
            message="Session task_1 started",
            details={"task_id": "task_1"},
        )
        reporter.forward_progress_event(event)

        mock_broadcast.assert_called_once()
        args = mock_broadcast.call_args[0]
        assert args[0] == "session_started"
        assert args[1] == "Session task_1 started"


# === create_dashboard_callback Tests ===


class TestCreateDashboardCallback:
    """Tests for create_dashboard_callback function."""

    def test_creates_callable(self):
        """Should create a callable callback."""
        callback = create_dashboard_callback()
        assert callable(callback)

    def test_callback_forwards_events(self):
        """Should forward events to dashboard."""
        with patch(
            "agenthub.parallel.dashboard.broadcast_parallel_event"
        ) as mock_broadcast:
            callback = create_dashboard_callback()
            event = ProgressEvent(
                event_type=EventType.STARTED,
                message="Test started",
            )
            callback(event)

            mock_broadcast.assert_called_once()

    def test_callback_uses_custom_url(self):
        """Should use custom URL when provided."""
        custom_url = "http://custom:8080"
        with patch(
            "agenthub.parallel.dashboard.broadcast_parallel_event"
        ) as mock_broadcast:
            callback = create_dashboard_callback(custom_url)
            event = ProgressEvent(
                event_type=EventType.STARTED,
                message="Test started",
            )
            callback(event)

            # The DashboardReporter should use the custom URL
            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args
            # The custom URL is passed as the dashboard_url parameter
            assert call_args[0][3] == custom_url or call_args[1].get("dashboard_url") == custom_url
