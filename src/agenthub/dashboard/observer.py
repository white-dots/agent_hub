"""Observer for tracking agent conversations."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import json


class EventType(str, Enum):
    """Types of conversation events."""
    QUERY_RECEIVED = "query_received"
    ROUTING_DECISION = "routing_decision"
    AGENT_STARTED = "agent_started"
    CONTEXT_LOADED = "context_loaded"
    AGENT_RESPONSE = "agent_response"
    ERROR = "error"
    # Claude Code integration events
    CLAUDE_CODE_TASK = "claude_code_task"
    CLAUDE_CODE_SEARCH = "claude_code_search"
    CLAUDE_CODE_READ = "claude_code_read"
    CLAUDE_CODE_EDIT = "claude_code_edit"
    CLAUDE_CODE_RESULT = "claude_code_result"
    # Cache/refresh events
    CONTEXT_REFRESH = "context_refresh"
    GIT_CHANGE_DETECTED = "git_change_detected"
    FILE_CHANGE_DETECTED = "file_change_detected"
    # QC / Concern events (Tier B raise concerns, Tier C synthesis)
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_AGENT_STARTED = "analysis_agent_started"
    ANALYSIS_AGENT_COMPLETE = "analysis_agent_complete"
    CONCERN_RAISED = "concern_raised"
    QC_SYNTHESIS_STARTED = "qc_synthesis_started"
    QC_REPORT_COMPLETE = "qc_report_complete"


@dataclass
class ConversationEvent:
    """A single event in the conversation flow."""

    event_type: EventType
    timestamp: datetime
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "data": self.data,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class ConversationTrace:
    """Complete trace of a conversation."""

    session_id: str
    query: str
    started_at: datetime
    events: list[ConversationEvent] = field(default_factory=list)
    completed_at: Optional[datetime] = None
    final_agent: Optional[str] = None
    final_response: Optional[str] = None
    tokens_used: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "events": [e.to_dict() for e in self.events],
            "final_agent": self.final_agent,
            "final_response": self.final_response[:500] if self.final_response else None,
            "tokens_used": self.tokens_used,
            "duration_ms": int((self.completed_at - self.started_at).total_seconds() * 1000) if self.completed_at else None,
        }


class ConversationObserver:
    """Observer that tracks all agent conversations.

    Attach this to an AgentHub to monitor all queries, routing decisions,
    and agent responses in real-time.

    Example:
        >>> observer = ConversationObserver()
        >>> observer.attach(hub)
        >>>
        >>> # Set up real-time callback
        >>> observer.on_event = lambda e: print(f"Event: {e.event_type}")
        >>>
        >>> hub.run("How does auth work?")
        >>>
        >>> # Get conversation history
        >>> for trace in observer.traces:
        ...     print(trace.query, "->", trace.final_agent)
    """

    def __init__(self, max_traces: int = 100):
        """Initialize observer.

        Args:
            max_traces: Maximum number of traces to keep in memory.
        """
        self.max_traces = max_traces
        self.traces: list[ConversationTrace] = []
        self._current_traces: dict[str, ConversationTrace] = {}
        self._callbacks: list[Callable[[ConversationEvent], None]] = []
        self._hub = None

    def attach(self, hub: "AgentHub") -> None:
        """Attach observer to an AgentHub.

        This wraps the hub's run method to capture all events.
        """
        self._hub = hub
        original_run = hub.run

        def observed_run(query: str, session_id=None, agent_id=None, model=None):
            # Create trace
            from uuid import uuid4
            trace_session = session_id or str(uuid4())

            trace = ConversationTrace(
                session_id=trace_session,
                query=query,
                started_at=datetime.now(),
            )
            self._current_traces[trace_session] = trace

            # Query received event
            self._emit(ConversationEvent(
                event_type=EventType.QUERY_RECEIVED,
                timestamp=datetime.now(),
                session_id=trace_session,
                data={"query": query, "forced_agent": agent_id},
            ))

            # Routing decision
            if not agent_id:
                routed_agent = hub.route(query)
                self._emit(ConversationEvent(
                    event_type=EventType.ROUTING_DECISION,
                    timestamp=datetime.now(),
                    session_id=trace_session,
                    data={
                        "query": query,
                        "selected_agent": routed_agent,
                        "available_agents": [a.agent_id for a in hub.list_agents()],
                    },
                ))
            else:
                routed_agent = agent_id

            # Agent started
            agent = hub.get_agent(routed_agent)
            self._emit(ConversationEvent(
                event_type=EventType.AGENT_STARTED,
                timestamp=datetime.now(),
                session_id=trace_session,
                data={
                    "agent_id": routed_agent,
                    "agent_name": agent.spec.name if agent else "Unknown",
                    "agent_type": agent.spec.metadata.get("module_type", "custom") if agent else "unknown",
                    "tier": "B" if agent and agent.spec.metadata.get("auto_generated") else "A",
                },
            ))

            # Context loaded
            if agent:
                context = agent.get_context()
                self._emit(ConversationEvent(
                    event_type=EventType.CONTEXT_LOADED,
                    timestamp=datetime.now(),
                    session_id=trace_session,
                    data={
                        "agent_id": routed_agent,
                        "context_size": len(context),
                        "context_preview": context[:500] + "..." if len(context) > 500 else context,
                    },
                ))

            try:
                # Execute
                response = original_run(query, session_id, agent_id, model)

                # Response event
                self._emit(ConversationEvent(
                    event_type=EventType.AGENT_RESPONSE,
                    timestamp=datetime.now(),
                    session_id=trace_session,
                    data={
                        "agent_id": response.agent_id,
                        "response_preview": response.content[:500] if response.content else "",
                        "tokens_used": response.tokens_used,
                        "artifacts_count": len(response.artifacts),
                    },
                ))

                # Complete trace
                trace.completed_at = datetime.now()
                trace.final_agent = response.agent_id
                trace.final_response = response.content
                trace.tokens_used = response.tokens_used

                self._store_trace(trace)
                return response

            except Exception as e:
                self._emit(ConversationEvent(
                    event_type=EventType.ERROR,
                    timestamp=datetime.now(),
                    session_id=trace_session,
                    data={"error": str(e)},
                ))
                raise

        hub.run = observed_run

    def on_event(self, callback: Callable[[ConversationEvent], None]) -> None:
        """Register a callback for real-time events.

        Args:
            callback: Function to call for each event.
        """
        self._callbacks.append(callback)

    def _emit(self, event: ConversationEvent) -> None:
        """Emit an event to all callbacks."""
        # Add to current trace
        if event.session_id in self._current_traces:
            self._current_traces[event.session_id].events.append(event)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def _store_trace(self, trace: ConversationTrace) -> None:
        """Store a completed trace."""
        self.traces.append(trace)

        # Remove from current
        if trace.session_id in self._current_traces:
            del self._current_traces[trace.session_id]

        # Trim old traces
        while len(self.traces) > self.max_traces:
            self.traces.pop(0)

    def get_recent_traces(self, limit: int = 20) -> list[dict]:
        """Get recent conversation traces.

        Args:
            limit: Maximum number of traces to return.

        Returns:
            List of trace dicts, newest first.
        """
        return [t.to_dict() for t in reversed(self.traces[-limit:])]

    def get_agent_stats(self) -> dict[str, dict]:
        """Get statistics per agent.

        Returns:
            Dict mapping agent_id to stats.
        """
        stats: dict[str, dict] = {}

        for trace in self.traces:
            if trace.final_agent:
                if trace.final_agent not in stats:
                    stats[trace.final_agent] = {
                        "queries": 0,
                        "total_tokens": 0,
                        "avg_duration_ms": 0,
                        "durations": [],
                    }

                stats[trace.final_agent]["queries"] += 1
                stats[trace.final_agent]["total_tokens"] += trace.tokens_used

                if trace.completed_at and trace.started_at:
                    duration = (trace.completed_at - trace.started_at).total_seconds() * 1000
                    stats[trace.final_agent]["durations"].append(duration)

        # Calculate averages
        for agent_id, data in stats.items():
            if data["durations"]:
                data["avg_duration_ms"] = sum(data["durations"]) / len(data["durations"])
            del data["durations"]

        return stats
