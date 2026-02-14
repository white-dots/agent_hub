"""FastAPI server for AgentHub dashboard."""

import asyncio
import json
from datetime import datetime
from typing import Optional

from agenthub.config import load_env_files
from agenthub.dashboard.observer import ConversationObserver, ConversationEvent, EventType


# Store for connected WebSocket clients
_websocket_clients: list = []
_observer: Optional[ConversationObserver] = None
_hub = None


def create_dashboard_app(hub, observer: Optional[ConversationObserver] = None):
    """Create FastAPI app for the dashboard.

    Args:
        hub: AgentHub instance to monitor.
        observer: Optional pre-configured observer.

    Returns:
        FastAPI app instance.
    """
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("Dashboard requires fastapi. Install with: pip install fastapi uvicorn")

    global _observer, _hub
    _hub = hub

    # Create or use provided observer
    _observer = observer or ConversationObserver()
    _observer.attach(hub)

    # Set up WebSocket broadcast
    def broadcast_event(event: ConversationEvent):
        """Broadcast event to all connected clients."""
        message = event.to_json()
        for client in _websocket_clients[:]:
            try:
                asyncio.create_task(client.send_text(message))
            except Exception:
                _websocket_clients.remove(client)

    _observer.on_event(broadcast_event)

    app = FastAPI(title="AgentHub Dashboard")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the dashboard HTML."""
        return DASHBOARD_HTML

    @app.get("/api/agents")
    async def get_agents():
        """Get all registered agents."""
        agents = []
        for spec in _hub.list_agents():
            # Determine tier: check explicit tier metadata first
            explicit_tier = spec.metadata.get("tier")
            if explicit_tier in ("A", "B", "C"):
                tier = explicit_tier
            elif spec.metadata.get("auto_generated"):
                tier = "B"
            else:
                tier = "A"

            agent_data = {
                "id": spec.agent_id,
                "name": spec.name,
                "description": spec.description,
                "tier": tier,
                "module_type": spec.metadata.get("module_type", "custom"),
                "keywords": spec.context_keywords[:10] if spec.context_keywords else [],
                "role": spec.metadata.get("role", ""),
            }
            # Include R&R if available (Tier B agents)
            if "rnr" in spec.metadata:
                agent_data["rnr"] = spec.metadata["rnr"]
            agents.append(agent_data)
        return {"agents": agents}

    @app.get("/api/debug")
    async def get_debug_info():
        """Debug info about the hub."""
        from pathlib import Path
        info = {
            "has_project_root": hasattr(_hub, '_project_root') and _hub._project_root is not None,
            "has_auto_manager": hasattr(_hub, '_auto_manager') and _hub._auto_manager is not None,
        }
        if info["has_project_root"]:
            info["project_root"] = str(_hub._project_root)
        if info["has_auto_manager"]:
            info["auto_manager_project_root"] = str(getattr(_hub._auto_manager, 'project_root', None))
        return info

    @app.get("/api/agent-status")
    async def get_agent_statuses():
        """Get context awareness status for all agents.

        Returns staleness info showing whether each agent has seen
        recent file changes in their domain.
        """
        statuses = _hub.get_all_agent_context_statuses()
        return {
            "statuses": [
                {
                    "agent_id": s.agent_id,
                    "is_stale": s.is_stale,
                    "status": s.status,
                    "changed_files": s.changed_files[:5],  # Limit to 5 files
                    "changed_file_count": len(s.changed_files),
                    "last_query_time": s.last_query_time.isoformat() if s.last_query_time else None,
                    "last_change_time": s.last_change_time.isoformat() if s.last_change_time else None,
                }
                for s in statuses
            ]
        }

    @app.get("/api/agent-status/{agent_id}")
    async def get_agent_status(agent_id: str):
        """Get context awareness status for a specific agent."""
        status = _hub.get_agent_context_status(agent_id)
        if not status:
            return {"error": "Agent not found or file watching not enabled"}

        return {
            "agent_id": status.agent_id,
            "is_stale": status.is_stale,
            "status": status.status,
            "changed_files": status.changed_files,
            "last_query_time": status.last_query_time.isoformat() if status.last_query_time else None,
            "last_change_time": status.last_change_time.isoformat() if status.last_change_time else None,
        }

    @app.get("/api/agents/{agent_id}/rnr")
    async def get_agent_rnr(agent_id: str):
        """Get the Roles & Responsibilities for a specific agent."""
        agent = _hub.get_agent(agent_id)
        if not agent:
            return {"error": f"Agent '{agent_id}' not found"}

        spec = agent.spec
        rnr = spec.metadata.get("rnr")

        # Determine tier
        explicit_tier = spec.metadata.get("tier")
        if explicit_tier in ("A", "B", "C"):
            tier = explicit_tier
        elif spec.metadata.get("auto_generated"):
            tier = "B"
        else:
            tier = "A"

        if not rnr:
            return {
                "agent_id": agent_id,
                "tier": tier,
                "message": f"This is a Tier {tier} agent. R&R is defined by the agent creator.",
            }

        return {
            "agent_id": agent_id,
            "name": spec.name,
            "tier": "B",
            "role": rnr.get("role"),
            "in_scope": rnr.get("in_scope", []),
            "out_of_scope": rnr.get("out_of_scope", []),
        }

    @app.get("/api/traces")
    async def get_traces(limit: int = 20):
        """Get recent conversation traces."""
        return {"traces": _observer.get_recent_traces(limit)}

    @app.get("/api/stats")
    async def get_stats():
        """Get agent statistics."""
        return {"stats": _observer.get_agent_stats()}

    # =========================================================================
    # DAG Team Execution Endpoints
    # =========================================================================

    @app.get("/api/team-traces")
    async def get_team_traces(limit: int = 20):
        """Get recent team execution traces.

        Returns a list of team execution traces for DAG visualization.
        """
        traces = _hub.get_team_traces(limit=limit)
        return {
            "traces": [
                {
                    "session_id": t["session_id"],
                    "query": t["query"],
                    "timestamp": t["timestamp"],
                    "agents_used": list(t["trace"].get("agent_responses", {}).keys()),
                    "total_tokens": t["trace"].get("total_tokens", 0),
                    "total_time_ms": t["trace"].get("total_time_ms", 0),
                    "layer_count": len(t["trace"].get("execution_layers", [])),
                }
                for t in traces
            ]
        }

    @app.get("/api/team-trace/{session_id}")
    async def get_team_trace(session_id: str):
        """Get a specific team execution trace with full details.

        Returns complete trace data including DAG structure, sub-questions,
        and individual agent responses for visualization.
        """
        trace_data = _hub.get_team_trace(session_id)
        if not trace_data:
            return {"error": f"Trace '{session_id}' not found"}

        trace = trace_data["trace"]

        # Build nodes with status and position info for visualization
        nodes = []
        execution_layers = trace.get("execution_layers", [])
        sub_questions = trace.get("sub_questions", {})
        agent_responses = trace.get("agent_responses", {})
        agent_tokens = trace.get("agent_tokens", {})
        agent_times = trace.get("agent_times", {})

        for layer_idx, layer in enumerate(execution_layers):
            for node_idx, agent_id in enumerate(layer):
                nodes.append({
                    "id": agent_id,
                    "layer": layer_idx,
                    "position": node_idx,
                    "sub_question": sub_questions.get(agent_id, ""),
                    "response": agent_responses.get(agent_id, ""),
                    "tokens": agent_tokens.get(agent_id, 0),
                    "time_ms": agent_times.get(agent_id, 0),
                    "status": "done" if agent_id in agent_responses else "pending",
                })

        # Build edges from dag_structure (dependencies)
        edges = []
        dag_structure = trace.get("dag_structure", {})
        for agent_id, deps in dag_structure.items():
            for dep in deps:
                edges.append({"from": dep, "to": agent_id})

        return {
            "session_id": trace_data["session_id"],
            "query": trace_data["query"],
            "timestamp": trace_data["timestamp"],
            "nodes": nodes,
            "edges": edges,
            "execution_layers": execution_layers,
            "summary": {
                "total_tokens": trace.get("total_tokens", 0),
                "decomposition_tokens": trace.get("decomposition_tokens", 0),
                "synthesis_tokens": trace.get("synthesis_tokens", 0),
                "total_time_ms": trace.get("total_time_ms", 0),
                "parallel_speedup": trace.get("parallel_speedup", 1.0),
            },
        }

    @app.get("/api/routing-rules")
    async def get_routing_rules():
        """Get routing rules showing which keywords map to which agents.

        This makes the automatic routing logic visible and transparent.
        """
        rules = []
        for spec in _hub.list_agents():
            # Determine tier
            explicit_tier = spec.metadata.get("tier")
            if explicit_tier in ("A", "B", "C"):
                tier = explicit_tier
            elif spec.metadata.get("auto_generated"):
                tier = "B"
            else:
                tier = "A"

            # Priority: A=1, B=2, C=3 (meta-agents checked last)
            priority_map = {"A": 1, "B": 2, "C": 3}
            agent_info = {
                "agent_id": spec.agent_id,
                "agent_name": spec.name,
                "tier": tier,
                "keywords": spec.context_keywords or [],
                "description": spec.description,
                "priority": priority_map.get(tier, 2),
            }
            rules.append(agent_info)

        # Sort by priority (Tier A first, then B, then C), then by number of keywords
        rules.sort(key=lambda r: (r["priority"], -len(r["keywords"])))

        return {
            "rules": rules,
            "routing_strategy": "keyword_match",
            "explanation": "Queries are matched against agent keywords. Tier A agents are checked first, then Tier B, then Tier C (meta-agents). First match wins.",
        }

    @app.get("/api/tree")
    async def get_agent_tree():
        """Get the agent tree visualization.

        Returns the ASCII tree showing agent hierarchy and coverage,
        plus structured agent data for card-based visualization.
        """
        from agenthub.auto.tree import build_agent_tree, print_agent_tree
        from pathlib import Path

        # Get project name from the hub's project root
        project_name = "Project"
        if hasattr(_hub, '_project_root') and _hub._project_root:
            project_name = Path(_hub._project_root).name
        elif hasattr(_hub, '_auto_manager') and _hub._auto_manager:
            auto_project_root = getattr(_hub._auto_manager, 'project_root', None)
            if auto_project_root:
                project_name = Path(auto_project_root).name

        tree_text = print_agent_tree(_hub, project_name, use_ascii=True)

        # Also return structured data for interactive tree
        tree_node = build_agent_tree(_hub, project_name, use_ascii=True)

        def node_to_dict(node):
            return {
                "name": node.name,
                "icon": node.icon,
                "description": node.description,
                "is_file": node.is_file,
                "children": [node_to_dict(c) for c in node.children],
            }

        # Build structured agent list for card view
        agents_structured = {
            "project_name": project_name,
            "tier_a": [],
            "tier_b": [],
            "tier_c": [],
            "sub_agents": [],
        }

        for spec in _hub.list_agents():
            tier = spec.metadata.get("tier", "")
            is_sub = spec.metadata.get("is_sub_agent", False)
            parent = spec.metadata.get("parent_agent", "")

            # Count files in context paths
            file_count = 0
            if spec.context_paths:
                for ctx in spec.context_paths:
                    if "**" in ctx:
                        # Glob pattern - estimate based on pattern
                        file_count += 10  # rough estimate
                    else:
                        file_count += 1

            agent_data = {
                "id": spec.agent_id,
                "name": spec.name,
                "description": spec.description or "",
                "keywords": spec.context_keywords[:5] if spec.context_keywords else [],
                "file_count": file_count,
                "context_paths": spec.context_paths[:3] if spec.context_paths else [],
            }

            if is_sub:
                agent_data["parent"] = parent
                agents_structured["sub_agents"].append(agent_data)
            elif tier == "A":
                agents_structured["tier_a"].append(agent_data)
            elif tier == "B":
                agents_structured["tier_b"].append(agent_data)
            elif tier == "C":
                agents_structured["tier_c"].append(agent_data)

        return {
            "ascii": tree_text,
            "tree": node_to_dict(tree_node),
            "agents": agents_structured,
        }

    @app.post("/api/query")
    async def run_query(data: dict):
        """Run a query through the hub."""
        query = data.get("query", "")
        agent_id = data.get("agent_id")

        if not query:
            return {"error": "No query provided"}

        try:
            response = _hub.run(query, agent_id=agent_id)
            return {
                "agent_id": response.agent_id,
                "content": response.content,
                "tokens_used": response.tokens_used,
                "session_id": response.session_id,
            }
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/claude-code/log")
    async def log_claude_code_event(data: dict):
        """Log a Claude Code activity event.

        This endpoint allows Claude Code to send events that appear
        in the dashboard's Live Events panel AND Recent Conversations.

        Expected data:
            event_type: One of task, result, search, read, edit, etc.
            session_id: Optional session identifier
            description: Human-readable description
            details: Optional dict with additional data
                - For 'result' events: agent_id, tokens_used, query, response
        """
        from datetime import datetime
        from uuid import uuid4
        from agenthub.dashboard.observer import ConversationEvent, ConversationTrace, EventType

        event_type_str = data.get("event_type", "claude_code_task")
        session_id = data.get("session_id", str(uuid4())[:8])
        description = data.get("description", "")
        details = data.get("details", {})

        # Map string to EventType
        event_map = {
            "task": EventType.CLAUDE_CODE_TASK,
            "search": EventType.CLAUDE_CODE_SEARCH,
            "read": EventType.CLAUDE_CODE_READ,
            "edit": EventType.CLAUDE_CODE_EDIT,
            "result": EventType.CLAUDE_CODE_RESULT,
            "context_refresh": EventType.CONTEXT_REFRESH,
            "git_change": EventType.GIT_CHANGE_DETECTED,
            "file_change": EventType.FILE_CHANGE_DETECTED,
            # Benchmark lifecycle events — mapped to CLAUDE_CODE_TASK
            # (generic enough to display; the event_type_str is preserved in data)
            "benchmark_started": EventType.CLAUDE_CODE_TASK,
            "benchmark_completed": EventType.CLAUDE_CODE_RESULT,
            "benchmark_repo_started": EventType.CLAUDE_CODE_TASK,
            "benchmark_repo_completed": EventType.CLAUDE_CODE_RESULT,
            "benchmark_task_started": EventType.CLAUDE_CODE_TASK,
            "benchmark_task_completed": EventType.CLAUDE_CODE_RESULT,
        }
        event_type = event_map.get(event_type_str, EventType.CLAUDE_CODE_TASK)

        # Preserve the original event_type_str for the frontend to differentiate
        details["_event_type"] = event_type_str

        # Create and broadcast event
        event = ConversationEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            session_id=session_id,
            data={"description": description, **details},
        )

        # Broadcast to WebSocket clients
        message = event.to_json()
        for client in _websocket_clients[:]:
            try:
                asyncio.create_task(client.send_text(message))
            except Exception:
                _websocket_clients.remove(client)

        # For 'result' events, also create a trace for Recent Conversations
        if event_type_str == "result" and _observer:
            query = details.get("query", "")
            agent_id = details.get("agent_id", "unknown")
            tokens_used = details.get("tokens_used", 0)
            response_content = details.get("response", "")
            team_execution = details.get("team_execution", False)

            # Create a completed trace
            trace = ConversationTrace(
                session_id=session_id,
                query=query[:200] if query else description[:200],
                started_at=datetime.now(),
                completed_at=datetime.now(),
                final_agent="team" if team_execution else agent_id,
                final_response=response_content[:500] if response_content else "",
                tokens_used=tokens_used,
            )
            trace.events.append(event)
            _observer._store_trace(trace)

        return {"status": "ok", "event_type": event_type.value, "session_id": session_id}

    # =========================================================================
    # QC Analysis Endpoints
    # =========================================================================

    @app.get("/api/qc/status")
    async def get_qc_status():
        """Check if QC analysis is enabled and get configuration."""
        return {
            "enabled": _hub.is_qc_enabled if hasattr(_hub, "is_qc_enabled") else False,
            "auto_analyze": getattr(_hub, "_qc_auto_analyze", False),
        }

    @app.post("/api/qc/toggle")
    async def toggle_qc_analysis(data: Optional[dict] = None):
        """Enable or disable QC analysis.

        Expected data:
            enable: bool (optional, toggles if not provided)
            auto_analyze: bool (default True)
        """
        if data is None:
            data = {}

        if not hasattr(_hub, "enable_qc_analysis"):
            return {"error": "QC analysis not supported by this hub"}

        current_enabled = getattr(_hub, "is_qc_enabled", False)

        # Determine target state
        if "enable" in data:
            target_enabled = data["enable"]
        else:
            target_enabled = not current_enabled

        if target_enabled and not current_enabled:
            # Enable QC
            auto_analyze = data.get("auto_analyze", True)

            # Set up concern callback to emit WebSocket events
            def on_concern(concern):
                event = ConversationEvent(
                    event_type=EventType.CONCERN_RAISED,
                    timestamp=datetime.now(),
                    session_id="qc",
                    data={
                        "concern_id": concern.concern_id,
                        "severity": concern.severity.value,
                        "title": concern.title,
                        "description": concern.description[:200] if concern.description else "",
                        "domain": concern.domain,
                    },
                )
                message = event.to_json()
                for client in _websocket_clients[:]:
                    try:
                        asyncio.create_task(client.send_text(message))
                    except Exception:
                        _websocket_clients.remove(client)

            def on_report(report):
                event = ConversationEvent(
                    event_type=EventType.QC_REPORT_COMPLETE,
                    timestamp=datetime.now(),
                    session_id="qc",
                    data={
                        "report_id": report.report_id,
                        "total_concerns": report.total_concerns,
                        "risk_level": report.risk_level,
                        "recommendation": report.recommendation,
                    },
                )
                message = event.to_json()
                for client in _websocket_clients[:]:
                    try:
                        asyncio.create_task(client.send_text(message))
                    except Exception:
                        _websocket_clients.remove(client)

            _hub.enable_qc_analysis(
                auto_analyze=auto_analyze,
                on_concern=on_concern,
                on_report=on_report,
            )

            return {
                "status": "ok",
                "enabled": True,
                "auto_analyze": auto_analyze,
                "message": "QC analysis enabled",
            }

        elif not target_enabled and current_enabled:
            # Disable QC - clear pipeline reference
            _hub._analysis_pipeline = None
            _hub._qc_agent = None

            return {
                "status": "ok",
                "enabled": False,
                "message": "QC analysis disabled",
            }

        else:
            return {
                "status": "ok",
                "enabled": current_enabled,
                "message": "No change",
            }

    @app.get("/api/qc/reports")
    async def get_qc_reports(limit: int = 20):
        """Get recent QC reports.

        Returns a list of recent concern reports from QC analysis.
        """
        if not hasattr(_hub, "get_qc_reports"):
            return {"reports": [], "error": "QC analysis not enabled"}

        try:
            reports = _hub.get_qc_reports(limit=limit)
            return {
                "reports": [
                    {
                        "report_id": r.report_id,
                        "change_set_id": r.change_set_id,
                        "timestamp": r.timestamp.isoformat(),
                        "total_concerns": r.total_concerns,
                        "critical_count": r.critical_count,
                        "high_count": r.high_count,
                        "medium_count": r.medium_count,
                        "low_count": r.low_count,
                        "risk_level": r.risk_level,
                        "recommendation": r.recommendation,
                        "overall_assessment": r.overall_assessment,
                        "agents_consulted": r.agents_consulted,
                    }
                    for r in reports
                ]
            }
        except Exception as e:
            return {"reports": [], "error": str(e)}

    @app.get("/api/qc/reports/{report_id}")
    async def get_qc_report(report_id: str):
        """Get a specific QC report with full details."""
        if not hasattr(_hub, "get_qc_report"):
            return {"error": "QC analysis not enabled"}

        try:
            report = _hub.get_qc_report(report_id)
            if not report:
                return {"error": f"Report '{report_id}' not found"}

            return {
                "report_id": report.report_id,
                "change_set_id": report.change_set_id,
                "timestamp": report.timestamp.isoformat(),
                "total_concerns": report.total_concerns,
                "critical_count": report.critical_count,
                "high_count": report.high_count,
                "medium_count": report.medium_count,
                "low_count": report.low_count,
                "risk_level": report.risk_level,
                "recommendation": report.recommendation,
                "overall_assessment": report.overall_assessment,
                "agents_consulted": report.agents_consulted,
                "total_analysis_time_ms": report.total_analysis_time_ms,
                "total_tokens_used": report.total_tokens_used,
                "concerns": [
                    {
                        "concern_id": c.concern_id,
                        "agent_id": c.agent_id,
                        "domain": c.domain,
                        "category": c.category.value,
                        "severity": c.severity.value,
                        "title": c.title,
                        "description": c.description,
                        "affected_files": c.affected_files,
                        "affected_functions": c.affected_functions,
                        "suggestion": c.suggestion,
                        "confidence": c.confidence,
                    }
                    for c in report.all_concerns
                ],
                "action_items": [
                    {
                        "action_id": a.action_id,
                        "priority": a.priority,
                        "title": a.title,
                        "description": a.description,
                        "related_concerns": a.related_concerns,
                        "estimated_effort": a.estimated_effort,
                    }
                    for a in report.action_items
                ],
            }
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/qc/concerns")
    async def get_recent_concerns(limit: int = 50):
        """Get recent concerns across all QC reports."""
        if not hasattr(_hub, "get_qc_reports"):
            return {"concerns": [], "error": "QC analysis not enabled"}

        try:
            reports = _hub.get_qc_reports(limit=10)
            all_concerns = []

            for report in reports:
                for concern in report.all_concerns:
                    all_concerns.append({
                        "concern_id": concern.concern_id,
                        "report_id": report.report_id,
                        "agent_id": concern.agent_id,
                        "domain": concern.domain,
                        "category": concern.category.value,
                        "severity": concern.severity.value,
                        "title": concern.title,
                        "description": concern.description,
                        "affected_files": concern.affected_files[:3],
                        "suggestion": concern.suggestion,
                        "raised_at": concern.raised_at.isoformat(),
                    })

            # Sort by severity (critical first)
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
            all_concerns.sort(key=lambda c: severity_order.get(c["severity"], 5))

            return {"concerns": all_concerns[:limit]}
        except Exception as e:
            return {"concerns": [], "error": str(e)}

    @app.post("/api/qc/analyze")
    async def trigger_qc_analysis(data: dict):
        """Manually trigger QC analysis on specified files.

        Expected data:
            files: list of file paths to analyze
        """
        if not hasattr(_hub, "analyze_changes"):
            return {"error": "QC analysis not enabled"}

        files = data.get("files", [])
        if not files:
            return {"error": "No files specified"}

        try:
            # Emit analysis started event
            event = ConversationEvent(
                event_type=EventType.ANALYSIS_STARTED,
                timestamp=datetime.now(),
                session_id="manual",
                data={"files": files, "source": "dashboard"},
            )
            message = event.to_json()
            for client in _websocket_clients[:]:
                try:
                    asyncio.create_task(client.send_text(message))
                except Exception:
                    _websocket_clients.remove(client)

            # Run analysis
            report = _hub.analyze_changes(files, source="manual")

            # Emit report complete event
            event = ConversationEvent(
                event_type=EventType.QC_REPORT_COMPLETE,
                timestamp=datetime.now(),
                session_id="manual",
                data={
                    "report_id": report.report_id,
                    "total_concerns": report.total_concerns,
                    "risk_level": report.risk_level,
                    "recommendation": report.recommendation,
                },
            )
            message = event.to_json()
            for client in _websocket_clients[:]:
                try:
                    asyncio.create_task(client.send_text(message))
                except Exception:
                    _websocket_clients.remove(client)

            return {
                "status": "ok",
                "report_id": report.report_id,
                "total_concerns": report.total_concerns,
                "risk_level": report.risk_level,
                "recommendation": report.recommendation,
            }
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/qc/concerns-scan")
    async def scan_tier_b_concerns():
        """Scan all Tier B agents for code concerns.

        Each Tier B agent is queried with: "Do you have any concerns about
        the code in your domain? Report issues with severity levels."

        Returns:
            List of concerns grouped by agent with severity.
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        # Get all Tier B agents
        tier_b_agents = []
        for spec in _hub.list_agents():
            if spec.metadata.get("tier") == "B":
                agent = _hub.get_agent(spec.agent_id)
                if agent:
                    tier_b_agents.append(agent)

        if not tier_b_agents:
            return {"concerns": [], "message": "No Tier B agents found"}

        # Query prompt for concerns
        concerns_prompt = """Analyze the code in your domain and identify any concerns.
For each concern, provide:
1. **Severity**: CRITICAL, HIGH, MEDIUM, or LOW
2. **Title**: Brief description (one line)
3. **File**: The file path if applicable
4. **Details**: Why this is a concern

Focus on:
- Security vulnerabilities
- Performance issues
- Code quality problems
- Missing error handling
- Deprecated patterns
- Potential bugs

Format your response as a list of concerns. If no concerns, say "No concerns found."
"""

        results = []
        errors = []

        # Query each agent
        def query_agent(agent):
            try:
                response = agent.run(concerns_prompt)
                return {
                    "agent_id": agent.spec.agent_id,
                    "agent_name": agent.spec.name,
                    "response": response.content if hasattr(response, "content") else str(response),
                }
            except Exception as e:
                return {
                    "agent_id": agent.spec.agent_id,
                    "agent_name": agent.spec.name,
                    "error": str(e),
                }

        # Run queries in parallel using thread pool
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(query_agent, agent) for agent in tier_b_agents]
            for future in futures:
                result = future.result()
                if "error" in result:
                    errors.append(result)
                else:
                    results.append(result)

        # Parse concerns from responses
        all_concerns = []
        for result in results:
            agent_concerns = _parse_agent_concerns(
                result["response"],
                result["agent_id"],
                result["agent_name"]
            )
            all_concerns.extend(agent_concerns)

        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        all_concerns.sort(key=lambda c: severity_order.get(c.get("severity", "LOW"), 4))

        return {
            "concerns": all_concerns,
            "agents_queried": len(tier_b_agents),
            "agents_responded": len(results),
            "errors": errors,
        }

    def _parse_agent_concerns(response: str, agent_id: str, agent_name: str) -> list:
        """Parse concerns from agent response."""
        concerns = []

        # Check for "no concerns" response
        lower_resp = response.lower()
        if "no concerns" in lower_resp or "no issues" in lower_resp:
            return concerns

        # Parse severity markers
        lines = response.split("\n")
        current_concern = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for severity markers
            upper_line = line.upper()
            severity = None
            for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                if sev in upper_line:
                    severity = sev
                    break

            if severity:
                # Save previous concern
                if current_concern:
                    concerns.append(current_concern)

                # Start new concern
                # Clean up the line - remove markdown, bullets, etc.
                title = line
                for prefix in ["**", "- ", "* ", "• ", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]:
                    title = title.replace(prefix, "")
                for sev in ["CRITICAL:", "HIGH:", "MEDIUM:", "LOW:", "CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                    title = title.replace(sev, "").replace(sev.lower(), "").replace(sev.title(), "")
                title = title.strip(" -:*")

                current_concern = {
                    "severity": severity,
                    "title": title if title else f"Issue in {agent_name}",
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "file": "",
                    "details": "",
                }
            elif current_concern:
                # Add to details
                if "file:" in line.lower() or ".py" in line:
                    # Extract file path
                    current_concern["file"] = line.replace("File:", "").replace("**File**:", "").strip()
                else:
                    current_concern["details"] += line + " "

        # Don't forget last concern
        if current_concern:
            concerns.append(current_concern)

        return concerns

    @app.get("/api/files")
    async def list_project_files(extensions: str = ".py", max_files: int = 500):
        """List project files for QC analysis selection.

        Args:
            extensions: Comma-separated file extensions to include (default: ".py")
            max_files: Maximum number of files to return (default: 500)

        Returns:
            List of files organized by directory.
        """
        from pathlib import Path

        # Get project root from auto manager or first agent's context paths
        project_root = None

        if hasattr(_hub, "_auto_manager") and _hub._auto_manager:
            project_root = getattr(_hub._auto_manager, "_project_root", None)

        if not project_root:
            # Try to infer from agent context paths
            for spec in _hub.list_agents():
                if spec.context_paths:
                    first_path = spec.context_paths[0]
                    # Walk up to find project root
                    path = Path(first_path)
                    if path.is_absolute():
                        project_root = str(path.parent.parent)
                    break

        if not project_root:
            return {"files": [], "error": "Could not determine project root"}

        root = Path(project_root)
        if not root.exists():
            return {"files": [], "error": f"Project root not found: {project_root}"}

        # Parse extensions
        ext_list = [e.strip() for e in extensions.split(",")]
        ext_list = [e if e.startswith(".") else f".{e}" for e in ext_list]

        # Directories to ignore
        ignore_dirs = {
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            ".egg-info", "build", "dist", ".pytest_cache", ".mypy_cache",
        }

        files_by_dir: dict = {}
        file_count = 0

        for ext in ext_list:
            for file_path in root.rglob(f"*{ext}"):
                if file_count >= max_files:
                    break

                # Skip ignored directories
                if any(part in ignore_dirs or part.startswith(".") for part in file_path.parts):
                    continue

                rel_path = str(file_path.relative_to(root))
                dir_path = str(file_path.parent.relative_to(root)) or "."

                if dir_path not in files_by_dir:
                    files_by_dir[dir_path] = []

                files_by_dir[dir_path].append({
                    "path": rel_path,
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                })
                file_count += 1

        # Sort directories and files
        sorted_dirs = sorted(files_by_dir.keys())
        result = []
        for dir_path in sorted_dirs:
            result.append({
                "directory": dir_path,
                "files": sorted(files_by_dir[dir_path], key=lambda f: f["name"]),
            })

        return {
            "project_root": str(root),
            "total_files": file_count,
            "directories": result,
        }

    # =========================================================================
    # Sub-Agent Policy Configuration
    # =========================================================================

    # In-memory storage for sub-agent policy (persisted via config file)
    _sub_agent_policy = {
        "min_files_to_split": 40,
        "min_subdirs_to_split": 2,
        "min_files_per_sub": 8,
        "max_sub_agents": 6,
    }

    @app.get("/api/config/sub-agent-policy")
    async def get_sub_agent_policy():
        """Get current sub-agent policy configuration."""
        return {
            "policy": _sub_agent_policy,
            "description": {
                "min_files_to_split": "Minimum files an agent must have to be considered for subdivision",
                "min_subdirs_to_split": "Minimum distinct subdirectories required for subdivision",
                "min_files_per_sub": "Minimum files each sub-agent must have (prevents over-splitting)",
                "max_sub_agents": "Maximum number of sub-agents to create from a single agent",
            },
        }

    @app.post("/api/config/sub-agent-policy")
    async def update_sub_agent_policy(policy: dict):
        """Update sub-agent policy configuration.

        Args:
            policy: Dict with policy fields to update.

        Returns:
            Updated policy configuration.
        """
        nonlocal _sub_agent_policy

        # Validate and update fields
        valid_fields = {"min_files_to_split", "min_subdirs_to_split", "min_files_per_sub", "max_sub_agents"}
        updates = {}

        for key, value in policy.items():
            if key in valid_fields:
                if not isinstance(value, int) or value < 1:
                    return {"error": f"Invalid value for {key}: must be a positive integer"}
                updates[key] = value

        if not updates:
            return {"error": "No valid fields to update"}

        _sub_agent_policy.update(updates)

        # Try to persist to config file
        try:
            from pathlib import Path
            config_file = Path.home() / ".agenthub" / "sub_agent_policy.json"
            config_file.parent.mkdir(exist_ok=True)
            config_file.write_text(json.dumps(_sub_agent_policy, indent=2))
        except Exception:
            pass  # Non-critical if persistence fails

        return {
            "status": "updated",
            "policy": _sub_agent_policy,
        }

    @app.post("/api/agents/refresh-keywords")
    async def refresh_agent_keywords(agent_id: str = None):
        """Refresh routing keywords for Tier B agents based on current code.

        Keywords are re-extracted from:
        - File/folder names in context_paths
        - Class and function names in the code

        This should be called after significant code changes to ensure
        routing remains accurate.

        Args:
            agent_id: Optional specific agent to refresh. If None, refreshes all.

        Returns:
            Dict with updated keywords per agent.
        """
        try:
            updated = _hub.refresh_agent_keywords(agent_id)

            if not updated:
                return {
                    "status": "no_updates",
                    "message": "No agents were updated. Ensure project_root is set and agents have context_paths.",
                }

            return {
                "status": "updated",
                "agents_updated": len(updated),
                "keywords": updated,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }

    @app.get("/api/agents/keywords")
    async def get_agent_keywords():
        """Get current routing keywords for all agents.

        Returns:
            Dict mapping agent_id to keywords list.
        """
        keywords = {}
        for spec in _hub.list_agents():
            keywords[spec.agent_id] = {
                "name": spec.name,
                "tier": spec.metadata.get("tier", "A" if not spec.metadata.get("auto_generated") else "B"),
                "is_sub_agent": spec.metadata.get("is_sub_agent", False),
                "keywords": spec.context_keywords,
                "context_paths_count": len(spec.context_paths),
            }
        return {"agents": keywords}

    # Load persisted policy on startup
    try:
        from pathlib import Path
        config_file = Path.home() / ".agenthub" / "sub_agent_policy.json"
        if config_file.exists():
            saved = json.loads(config_file.read_text())
            _sub_agent_policy.update(saved)
    except Exception:
        pass  # Use defaults if loading fails

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time event streaming."""
        await websocket.accept()
        _websocket_clients.append(websocket)

        try:
            while True:
                # Keep connection alive, receive any client messages
                data = await websocket.receive_text()
                # Could handle client commands here
        except WebSocketDisconnect:
            _websocket_clients.remove(websocket)

    return app


def run_dashboard(hub, port: int = 3001, host: str = "0.0.0.0"):
    """Run the dashboard server.

    Args:
        hub: AgentHub instance to monitor.
        port: Port to run on (default: 3001).
        host: Host to bind to.

    Note:
        ANTHROPIC_API_KEY is loaded from .env file (in project dir or home dir).
    """
    # Load environment variables from .env files
    loaded = load_env_files(verbose=False)
    if loaded:
        print(f"   Loaded .env from: {', '.join(loaded)}")

    try:
        import uvicorn
    except ImportError:
        raise ImportError("Dashboard requires uvicorn. Install with: pip install uvicorn")

    app = create_dashboard_app(hub)
    print(f"\n🚀 AgentHub Dashboard running at http://localhost:{port}")
    print("   Open in browser to see agent conversations\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentHub Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .event-card { animation: slideIn 0.3s ease-out; }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .tier-a { border-left: 4px solid #3b82f6; }
        .tier-b { border-left: 4px solid #10b981; }
        .tier-c { border-left: 4px solid #8b5cf6; }
        .event-query { background: #fef3c7; }
        .event-routing { background: #dbeafe; }
        .event-agent { background: #d1fae5; }
        .event-response { background: #f3e8ff; }
        .event-error { background: #fee2e2; }
        /* Claude Code events - purple/violet theme */
        .event-claude-code { background: #ede9fe; border-left: 3px solid #8b5cf6; }
        .event-claude-result { background: #c4b5fd; border-left: 3px solid #7c3aed; }
        /* Cache/refresh events - orange/amber theme */
        .event-refresh { background: #fef3c7; border-left: 3px solid #f59e0b; }
        /* Tree styling */
        #tree-ascii {
            line-height: 1.4;
            white-space: pre;
        }
        /* Org Chart Tree Styles */
        .org-tree {
            overflow-x: auto;
            padding: 20px 10px;
        }
        .org-level {
            display: flex;
            justify-content: center;
            gap: 16px;
            margin-bottom: 24px;
            position: relative;
        }
        .org-level::before {
            content: '';
            position: absolute;
            top: -12px;
            left: 50%;
            transform: translateX(-50%);
            height: 12px;
            border-left: 2px solid #4b5563;
        }
        .org-level:first-child::before {
            display: none;
        }
        .org-card {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 12px 16px;
            min-width: 180px;
            max-width: 240px;
            text-align: center;
            position: relative;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        .org-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            border-color: #6366f1;
        }
        .org-card.tier-a {
            border-left: 4px solid #3b82f6;
            background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
        }
        .org-card.tier-b {
            border-left: 4px solid #10b981;
            background: linear-gradient(135deg, #134e4a 0%, #1e293b 100%);
        }
        .org-card.tier-c {
            border-left: 4px solid #8b5cf6;
            background: linear-gradient(135deg, #4c1d95 0%, #1e293b 100%);
        }
        .org-card.project {
            border-left: 4px solid #f59e0b;
            background: linear-gradient(135deg, #78350f 0%, #1e293b 100%);
        }
        .org-card-icon {
            font-size: 24px;
            margin-bottom: 6px;
        }
        .org-card-name {
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 4px;
            color: #f3f4f6;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .org-card-desc {
            font-size: 10px;
            color: #9ca3af;
            line-height: 1.3;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .org-card-badge {
            position: absolute;
            top: -8px;
            right: -8px;
            background: #4b5563;
            color: #e5e7eb;
            font-size: 9px;
            padding: 2px 6px;
            border-radius: 10px;
            font-weight: 500;
        }
        .org-card-badge.tier-a { background: #3b82f6; }
        .org-card-badge.tier-b { background: #10b981; }
        .org-card-badge.tier-c { background: #8b5cf6; }
        .org-card-badge.sub { background: #a855f7; }
        .org-card-files {
            font-size: 9px;
            color: #6b7280;
            margin-top: 6px;
        }
        .org-connector {
            position: relative;
            height: 24px;
        }
        .org-connector::after {
            content: '';
            position: absolute;
            left: 50%;
            top: 0;
            height: 100%;
            border-left: 2px solid #4b5563;
        }
        .org-children {
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            position: relative;
        }
        .org-children::before {
            content: '';
            position: absolute;
            top: -12px;
            left: 10%;
            right: 10%;
            height: 0;
            border-top: 2px solid #4b5563;
        }
        .org-group {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .org-group-label {
            font-size: 11px;
            color: #9ca3af;
            margin-bottom: 8px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .org-group-cards {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            justify-content: center;
        }
        /* Parent agent with sub-agents container */
        .org-parent-with-subs {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .org-sub-connector {
            position: relative;
            height: 20px;
            width: 2px;
            background: #4b5563;
        }
        .org-sub-agents {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: center;
            padding: 8px;
            background: rgba(139, 92, 246, 0.1);
            border-radius: 8px;
            border: 1px dashed #8b5cf6;
            position: relative;
        }
        .org-sub-agents::before {
            content: '';
            position: absolute;
            top: -1px;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-top: 6px solid #8b5cf6;
        }
        .org-card.sub-agent {
            border-left: 4px solid #8b5cf6;
            background: linear-gradient(135deg, #3b2e5a 0%, #1e293b 100%);
            min-width: 140px;
            max-width: 180px;
        }
        /* QC events - red/orange theme for concerns */
        .event-qc-started { background: #fef3c7; border-left: 3px solid #f59e0b; }
        .event-qc-concern { background: #fee2e2; border-left: 3px solid #ef4444; }
        .event-qc-report { background: #d1fae5; border-left: 3px solid #10b981; }
        /* Severity colors */
        .severity-critical { color: #dc2626; font-weight: bold; }
        .severity-high { color: #ea580c; font-weight: bold; }
        .severity-medium { color: #ca8a04; }
        .severity-low { color: #65a30d; }
        .severity-info { color: #6b7280; }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-6">
        <!-- Header -->
        <div class="flex items-center justify-between mb-6">
            <h1 class="text-2xl font-bold">🤖 AgentHub Dashboard</h1>
            <div id="connection-status" class="flex items-center gap-2">
                <span class="w-3 h-3 rounded-full bg-yellow-500"></span>
                <span class="text-sm">Connecting...</span>
            </div>
        </div>

        <!-- Agent Tree View (Collapsible) -->
        <div class="mb-6">
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2 cursor-pointer" onclick="toggleTree()">
                        <h2 class="text-lg font-semibold">🌳 Agent Coverage Tree</h2>
                        <span id="tree-toggle" class="text-gray-400">▼</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <button onclick="toggleTreeView()" class="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded" id="tree-view-toggle">
                            📊 Card View
                        </button>
                        <button onclick="loadTree()" class="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded">
                            🔄 Refresh
                        </button>
                    </div>
                </div>
                <div id="tree-panel" class="mt-4">
                    <!-- Card View (default) -->
                    <div id="tree-cards" class="org-tree">
                        <div class="text-gray-500 text-center py-8">Loading agents...</div>
                    </div>
                    <!-- ASCII View (hidden by default) -->
                    <pre id="tree-ascii" class="text-sm font-mono bg-gray-900 p-4 rounded overflow-x-auto text-gray-300 hidden">Loading...</pre>
                </div>
            </div>
        </div>

        <!-- QC Analysis Panel (Collapsible) -->
        <div class="mb-6">
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2 cursor-pointer" onclick="toggleQCPanel()">
                        <h2 class="text-lg font-semibold">🔍 Code Concerns</h2>
                        <span id="qc-toggle" class="text-gray-400">▼</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <button id="run-qc-btn" onclick="runCodeConcernsAnalysis()" class="text-xs bg-purple-600 hover:bg-purple-700 px-3 py-1 rounded">
                            🔍 Run Analysis
                        </button>
                        <button onclick="loadQCReports()" class="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded">
                            🔄 Refresh
                        </button>
                    </div>
                </div>
                <div id="qc-panel" class="mt-4">
                    <!-- Description -->
                    <div class="bg-gray-900 rounded p-3 mb-4">
                        <p class="text-sm text-gray-400">
                            Ask each Tier B agent: "Do you have any concerns for the latest updates or existing files?"
                        </p>
                        <div id="qc-progress" class="mt-2 hidden">
                            <div class="flex items-center gap-2 text-sm text-blue-400">
                                <span class="animate-spin">⏳</span>
                                <span id="qc-progress-text">Analyzing...</span>
                            </div>
                        </div>
                    </div>

                    <!-- QC Summary -->
                    <div id="qc-summary" class="grid grid-cols-4 gap-4 mb-4">
                        <div class="bg-gray-900 p-3 rounded text-center">
                            <div id="qc-critical-count" class="text-2xl font-bold text-red-500">-</div>
                            <div class="text-xs text-gray-400">Critical</div>
                        </div>
                        <div class="bg-gray-900 p-3 rounded text-center">
                            <div id="qc-high-count" class="text-2xl font-bold text-orange-500">-</div>
                            <div class="text-xs text-gray-400">High</div>
                        </div>
                        <div class="bg-gray-900 p-3 rounded text-center">
                            <div id="qc-medium-count" class="text-2xl font-bold text-yellow-500">-</div>
                            <div class="text-xs text-gray-400">Medium</div>
                        </div>
                        <div class="bg-gray-900 p-3 rounded text-center">
                            <div id="qc-low-count" class="text-2xl font-bold text-green-500">-</div>
                            <div class="text-xs text-gray-400">Low</div>
                        </div>
                    </div>

                    <!-- Concerns Filter -->
                    <div class="flex gap-2 mb-2">
                        <select id="qc-severity-filter" onchange="filterConcerns()" class="text-xs bg-gray-700 rounded px-2 py-1">
                            <option value="all">All Severities</option>
                            <option value="critical">Critical Only</option>
                            <option value="high">High & Above</option>
                            <option value="medium">Medium & Above</option>
                        </select>
                    </div>

                    <!-- Concerns by Agent -->
                    <div class="bg-gray-900 rounded p-3">
                        <h3 class="text-sm font-semibold mb-2 text-gray-300">Concerns by Agent</h3>
                        <div id="qc-concerns-list" class="space-y-2 max-h-[400px] overflow-y-auto">
                            <div class="text-gray-500 text-sm">Click "Run Analysis" to check for concerns</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- DAG Team Execution Panel (Collapsible) -->
        <div class="mb-6">
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2 cursor-pointer" onclick="toggleDAGPanel()">
                        <h2 class="text-lg font-semibold">🔗 DAG Team Executions</h2>
                        <span id="dag-toggle" class="text-gray-400">▶</span>
                        <span id="dag-count-badge" class="text-xs px-2 py-0.5 rounded bg-gray-700 text-gray-400">0 traces</span>
                    </div>
                    <button onclick="loadTeamTraces()" class="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded">
                        🔄 Refresh
                    </button>
                </div>
                <div id="dag-panel" class="mt-4" style="display: none;">
                    <!-- Trace List -->
                    <div class="grid grid-cols-3 gap-4">
                        <div class="col-span-1 bg-gray-900 rounded p-3">
                            <h3 class="text-sm font-semibold mb-2 text-gray-300">Recent Team Queries</h3>
                            <div id="dag-trace-list" class="space-y-2 max-h-[400px] overflow-y-auto">
                                <div class="text-gray-500 text-sm">No team executions yet</div>
                            </div>
                        </div>
                        <!-- DAG Visualization -->
                        <div class="col-span-2 bg-gray-900 rounded p-3">
                            <h3 class="text-sm font-semibold mb-2 text-gray-300">DAG Visualization</h3>
                            <div id="dag-visualization" class="min-h-[300px] flex items-center justify-center">
                                <div class="text-gray-500 text-sm">Select a trace to view DAG</div>
                            </div>
                            <!-- Node Detail -->
                            <div id="dag-node-detail" class="mt-4 hidden">
                                <div class="bg-gray-800 p-3 rounded">
                                    <h4 class="text-sm font-semibold mb-2 text-purple-400" id="dag-node-title">Agent Details</h4>
                                    <div class="space-y-2 text-sm">
                                        <div><span class="text-gray-400">Sub-question:</span> <span id="dag-node-question" class="text-white"></span></div>
                                        <div><span class="text-gray-400">Tokens:</span> <span id="dag-node-tokens" class="text-white"></span></div>
                                        <div><span class="text-gray-400">Time:</span> <span id="dag-node-time" class="text-white"></span></div>
                                        <details class="mt-2">
                                            <summary class="cursor-pointer text-blue-400">View Response</summary>
                                            <pre id="dag-node-response" class="mt-2 text-xs bg-gray-900 p-2 rounded whitespace-pre-wrap max-h-[200px] overflow-y-auto"></pre>
                                        </details>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <!-- Summary Stats -->
                    <div id="dag-summary" class="mt-4 grid grid-cols-5 gap-2 text-center text-xs hidden">
                        <div class="bg-gray-900 p-2 rounded">
                            <div id="dag-total-tokens" class="text-lg font-bold text-blue-400">-</div>
                            <div class="text-gray-500">Total Tokens</div>
                        </div>
                        <div class="bg-gray-900 p-2 rounded">
                            <div id="dag-decomp-tokens" class="text-lg font-bold text-purple-400">-</div>
                            <div class="text-gray-500">Decomposition</div>
                        </div>
                        <div class="bg-gray-900 p-2 rounded">
                            <div id="dag-synth-tokens" class="text-lg font-bold text-green-400">-</div>
                            <div class="text-gray-500">Synthesis</div>
                        </div>
                        <div class="bg-gray-900 p-2 rounded">
                            <div id="dag-total-time" class="text-lg font-bold text-yellow-400">-</div>
                            <div class="text-gray-500">Total Time</div>
                        </div>
                        <div class="bg-gray-900 p-2 rounded">
                            <div id="dag-speedup" class="text-lg font-bold text-orange-400">-</div>
                            <div class="text-gray-500">Speedup</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Settings Panel (Collapsible) -->
        <div class="mb-6">
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2 cursor-pointer" onclick="toggleSettingsPanel()">
                        <h2 class="text-lg font-semibold">⚙️ Sub-Agent Settings</h2>
                        <span id="settings-toggle" class="text-gray-400">▶</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <button onclick="loadSubAgentPolicy()" class="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded">
                            🔄 Refresh
                        </button>
                        <button id="settings-save-btn" onclick="saveSubAgentPolicy()" class="text-xs bg-green-600 hover:bg-green-700 px-2 py-1 rounded">
                            💾 Save
                        </button>
                    </div>
                </div>
                <div id="settings-panel" class="mt-4" style="display: none;">
                    <div class="bg-gray-900 rounded p-4">
                        <p class="text-sm text-gray-400 mb-4">
                            Configure how large Tier B agents are subdivided into focused sub-agents.
                            Sub-agents are created when an agent covers too many files, making it more efficient to have specialized team members.
                        </p>
                        <div class="grid grid-cols-2 gap-4">
                            <!-- Min Files to Split -->
                            <div class="bg-gray-800 p-3 rounded">
                                <label class="text-sm font-medium text-gray-300 block mb-1">Min Files to Split</label>
                                <input type="number" id="setting-min-files-to-split" min="10" max="200" value="40"
                                    class="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm border border-gray-600 focus:border-blue-500 focus:outline-none">
                                <p class="text-xs text-gray-500 mt-1">Agent must have at least this many files to be considered for subdivision</p>
                            </div>
                            <!-- Min Subdirs to Split -->
                            <div class="bg-gray-800 p-3 rounded">
                                <label class="text-sm font-medium text-gray-300 block mb-1">Min Subdirectories</label>
                                <input type="number" id="setting-min-subdirs-to-split" min="2" max="10" value="2"
                                    class="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm border border-gray-600 focus:border-blue-500 focus:outline-none">
                                <p class="text-xs text-gray-500 mt-1">Must span at least this many distinct subdirectories</p>
                            </div>
                            <!-- Min Files Per Sub -->
                            <div class="bg-gray-800 p-3 rounded">
                                <label class="text-sm font-medium text-gray-300 block mb-1">Min Files Per Sub-Agent</label>
                                <input type="number" id="setting-min-files-per-sub" min="3" max="50" value="8"
                                    class="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm border border-gray-600 focus:border-blue-500 focus:outline-none">
                                <p class="text-xs text-gray-500 mt-1">Each sub-agent must have at least this many files (prevents over-splitting)</p>
                            </div>
                            <!-- Max Sub Agents -->
                            <div class="bg-gray-800 p-3 rounded">
                                <label class="text-sm font-medium text-gray-300 block mb-1">Max Sub-Agents</label>
                                <input type="number" id="setting-max-sub-agents" min="2" max="12" value="6"
                                    class="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm border border-gray-600 focus:border-blue-500 focus:outline-none">
                                <p class="text-xs text-gray-500 mt-1">Maximum number of sub-agents to create from a single Tier B agent</p>
                            </div>
                        </div>
                        <div id="settings-status" class="mt-4 text-sm text-center hidden">
                            <span class="text-green-400">✓ Settings saved successfully</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Report Detail Modal -->
        <div id="report-modal" class="fixed inset-0 bg-black bg-opacity-50 hidden z-50 flex items-center justify-center">
            <div class="bg-gray-800 rounded-lg max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden flex flex-col">
                <div class="p-4 border-b border-gray-700 flex items-center justify-between">
                    <h2 class="text-lg font-semibold" id="modal-report-title">QC Report</h2>
                    <button onclick="closeReportModal()" class="text-gray-400 hover:text-white text-2xl">&times;</button>
                </div>
                <div id="modal-report-content" class="p-4 overflow-y-auto flex-1">
                    Loading...
                </div>
                <div class="p-4 border-t border-gray-700 flex justify-end gap-2">
                    <button onclick="exportReport()" class="text-xs bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded">
                        Export JSON
                    </button>
                    <button onclick="closeReportModal()" class="text-xs bg-blue-600 hover:bg-blue-700 px-3 py-1.5 rounded">
                        Close
                    </button>
                </div>
            </div>
        </div>

        <!-- Concern Detail Modal -->
        <div id="concern-modal" class="fixed inset-0 bg-black bg-opacity-50 hidden z-50 flex items-center justify-center">
            <div class="bg-gray-800 rounded-lg max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden flex flex-col">
                <div class="p-4 border-b border-gray-700 flex items-center justify-between">
                    <h2 class="text-lg font-semibold" id="modal-concern-title">Concern Details</h2>
                    <button onclick="closeConcernModal()" class="text-gray-400 hover:text-white text-2xl">&times;</button>
                </div>
                <div id="modal-concern-content" class="p-4 overflow-y-auto flex-1">
                    Loading...
                </div>
            </div>
        </div>

        <div class="grid grid-cols-12 gap-6">
            <!-- Left Panel: Agents -->
            <div class="col-span-3">
                <div class="bg-gray-800 rounded-lg p-4">
                    <h2 class="text-lg font-semibold mb-4">Registered Agents</h2>
                    <div id="agents-list" class="space-y-3">
                        <div class="text-gray-400">Loading...</div>
                    </div>
                </div>

                <!-- Stats -->
                <div class="bg-gray-800 rounded-lg p-4 mt-4">
                    <h2 class="text-lg font-semibold mb-4">Statistics</h2>
                    <div id="stats-panel" class="space-y-2 text-sm">
                        <div class="text-gray-400">No data yet</div>
                    </div>
                </div>

                <!-- Routing Rules -->
                <div class="bg-gray-800 rounded-lg p-4 mt-4">
                    <h2 class="text-lg font-semibold mb-4">Routing Rules</h2>
                    <div id="routing-panel" class="space-y-2 text-sm max-h-[300px] overflow-y-auto">
                        <div class="text-gray-400">Loading...</div>
                    </div>
                </div>
            </div>

            <!-- Middle Panel: Live Events (Claude Code interactions) -->
            <div class="col-span-6">
                <div class="bg-gray-800 rounded-lg p-4 h-[800px] flex flex-col">
                    <h2 class="text-lg font-semibold mb-4">Live Events (Claude Code ↔ Agents)</h2>
                    <div id="events-panel" class="flex-1 overflow-y-auto space-y-3">
                        <div class="text-gray-400 text-center py-8">
                            Waiting for Claude Code queries via MCP...
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Panel: Query & Response -->
            <div class="col-span-3">
                <div class="bg-gray-800 rounded-lg p-4">
                    <h2 class="text-lg font-semibold mb-4">Test Query</h2>
                    <textarea
                        id="query-input"
                        class="w-full bg-gray-700 rounded p-3 text-sm resize-none"
                        rows="3"
                        placeholder="Enter a query to test routing..."
                    ></textarea>
                    <div class="flex gap-2 mt-2">
                        <select id="agent-select" class="bg-gray-700 rounded px-3 py-2 text-sm flex-1">
                            <option value="">Auto-route</option>
                        </select>
                        <button
                            id="send-btn"
                            class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-medium"
                        >
                            Send
                        </button>
                    </div>
                </div>

                <div class="bg-gray-800 rounded-lg p-4 mt-4 h-[400px] flex flex-col">
                    <h2 class="text-lg font-semibold mb-4">Response</h2>
                    <div id="response-panel" class="flex-1 overflow-y-auto">
                        <div class="text-gray-400 text-sm">
                            Send a query to see the response...
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Recent Traces -->
        <div class="bg-gray-800 rounded-lg p-4 mt-6">
            <h2 class="text-lg font-semibold mb-4">Recent Conversations</h2>
            <div id="traces-panel" class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead class="text-left text-gray-400 border-b border-gray-700">
                        <tr>
                            <th class="pb-2">Time</th>
                            <th class="pb-2">Query</th>
                            <th class="pb-2">Agent</th>
                            <th class="pb-2">Tier</th>
                            <th class="pb-2">Tokens</th>
                            <th class="pb-2">Duration</th>
                        </tr>
                    </thead>
                    <tbody id="traces-body" class="divide-y divide-gray-700">
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let agents = [];
        let treeVisible = true;
        let qcVisible = true;

        // Define toggleTree globally before DOM loads
        function toggleTree() {
            const panel = document.getElementById('tree-panel');
            const toggle = document.getElementById('tree-toggle');
            treeVisible = !treeVisible;
            panel.style.display = treeVisible ? 'block' : 'none';
            toggle.textContent = treeVisible ? '▼' : '▶';
        }

        function toggleQCPanel() {
            const panel = document.getElementById('qc-panel');
            const toggle = document.getElementById('qc-toggle');
            qcVisible = !qcVisible;
            panel.style.display = qcVisible ? 'block' : 'none';
            toggle.textContent = qcVisible ? '▼' : '▶';
        }

        async function toggleQCEnabled() {
            const btn = document.getElementById('qc-enable-btn');
            btn.disabled = true;
            btn.textContent = 'Loading...';

            try {
                const res = await fetch('/api/qc/toggle', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const data = await res.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    // Refresh status and reports
                    await loadQCStatus();
                    await loadQCReports();
                }
            } catch (e) {
                console.error('Failed to toggle QC:', e);
                alert('Failed to toggle QC: ' + e.message);
            }

            btn.disabled = false;
        }

        let treeViewMode = 'cards'; // 'cards' or 'ascii'

        function toggleTreeView() {
            const cardsEl = document.getElementById('tree-cards');
            const asciiEl = document.getElementById('tree-ascii');
            const toggleBtn = document.getElementById('tree-view-toggle');

            if (treeViewMode === 'cards') {
                treeViewMode = 'ascii';
                cardsEl.classList.add('hidden');
                asciiEl.classList.remove('hidden');
                toggleBtn.textContent = '🎴 Card View';
            } else {
                treeViewMode = 'cards';
                cardsEl.classList.remove('hidden');
                asciiEl.classList.add('hidden');
                toggleBtn.textContent = '📝 Text View';
            }
        }

        async function loadTree() {
            try {
                const res = await fetch('/api/tree');
                const data = await res.json();

                // Render ASCII view
                const treeAscii = document.getElementById('tree-ascii');
                let html = data.ascii
                    .replace(/\\[P\\]/g, '<span class="text-yellow-400">[P]</span>')
                    .replace(/\\[A\\]/g, '<span class="text-blue-400">[A]</span>')
                    .replace(/\\[B\\]/g, '<span class="text-green-400">[B]</span>')
                    .replace(/\\[C\\]/g, '<span class="text-purple-400">[C]</span>')
                    .replace(/\\* /g, '<span class="text-purple-400">◆ </span>')
                    .replace(/# /g, '<span class="text-gray-500"># </span>')
                    .replace(/\\(api\\)/g, '<span class="text-orange-400">(api)</span>')
                    .replace(/\\(svc\\)/g, '<span class="text-pink-400">(svc)</span>')
                    .replace(/\\(mod\\)/g, '<span class="text-indigo-400">(mod)</span>')
                    .replace(/\\(repo\\)/g, '<span class="text-teal-400">(repo)</span>')
                    .replace(/\\(util\\)/g, '<span class="text-amber-400">(util)</span>')
                    .replace(/\\(cfg\\)/g, '<span class="text-rose-400">(cfg)</span>')
                    .replace(/\\(test\\)/g, '<span class="text-lime-400">(test)</span>')
                    .replace(/\\(code\\)/g, '<span class="text-slate-400">(code)</span>')
                    .replace(/\\(meta\\)/g, '<span class="text-purple-400">(meta)</span>')
                    .replace(/ --- /g, ' <span class="text-gray-500">―</span> ');
                treeAscii.innerHTML = html;

                // Render Card view using structured data
                const treeCards = document.getElementById('tree-cards');
                treeCards.innerHTML = renderTreeCards(data.agents);

            } catch (e) {
                console.error('Failed to load tree:', e);
            }
        }

        function renderTreeCards(agents) {
            if (!agents) return '<div class="text-gray-500 text-center">No agents found</div>';

            const { project_name, tier_a, tier_b, tier_c, sub_agents } = agents;
            const totalAgents = (tier_a?.length || 0) + (tier_b?.length || 0) + (tier_c?.length || 0) + (sub_agents?.length || 0);

            // Group sub-agents by parent
            const subsByParent = {};
            for (const sub of (sub_agents || [])) {
                if (!subsByParent[sub.parent]) subsByParent[sub.parent] = [];
                subsByParent[sub.parent].push(sub);
            }

            // Build HTML
            let html = '';

            // Project Card (top level)
            html += `
                <div class="org-level">
                    <div class="org-card project">
                        <div class="org-card-icon">📦</div>
                        <div class="org-card-name">${escapeHtml(project_name)}</div>
                        <div class="org-card-desc">${totalAgents} agents total</div>
                    </div>
                </div>
            `;

            // Connector
            html += '<div class="org-connector"></div>';

            // Tier A Group
            if (tier_a && tier_a.length > 0) {
                html += `
                    <div class="org-group">
                        <div class="org-group-label">🎯 TIER A - BUSINESS AGENTS (${tier_a.length})</div>
                        <div class="org-group-cards">
                `;
                for (const agent of tier_a) {
                    const shortName = agent.name.replace(/ Agent$/i, '').replace(/ Expert$/i, '');
                    const keywords = agent.keywords?.join(', ') || '';
                    html += `
                        <div class="org-card tier-a" title="${escapeHtml(agent.description)}" onclick="showAgentDetails('${agent.id}')">
                            <span class="org-card-badge tier-a">A</span>
                            <div class="org-card-icon">💼</div>
                            <div class="org-card-name">${escapeHtml(shortName)}</div>
                            <div class="org-card-desc">${escapeHtml((agent.description || '').slice(0, 60))}${(agent.description || '').length > 60 ? '...' : ''}</div>
                            ${keywords ? `<div class="org-card-files">🏷️ ${escapeHtml(keywords.slice(0, 30))}</div>` : ''}
                        </div>
                    `;
                }
                html += '</div></div>';
            }

            // Connector between Tier A and Tier B
            html += '<div class="org-connector"></div>';

            // Tier B Group with sub-agents nested
            if (tier_b && tier_b.length > 0) {
                html += `
                    <div class="org-group">
                        <div class="org-group-label">🔧 TIER B - CODE AGENTS (${tier_b.length})</div>
                        <div class="org-group-cards">
                `;
                for (const agent of tier_b) {
                    const shortName = agent.name.replace(/ Agent$/i, '').replace(/ Expert$/i, '');
                    const subs = subsByParent[agent.id] || [];

                    // Wrap parent + sub-agents in a container if there are sub-agents
                    if (subs.length > 0) {
                        html += `<div class="org-parent-with-subs">`;
                    }

                    html += `
                        <div class="org-card tier-b" title="${escapeHtml(agent.description)}" onclick="showAgentDetails('${agent.id}')">
                            <span class="org-card-badge tier-b">B</span>
                            <div class="org-card-icon">⚙️</div>
                            <div class="org-card-name">${escapeHtml(shortName)}</div>
                            <div class="org-card-desc">${escapeHtml(agent.description.slice(0, 50))}${agent.description.length > 50 ? '...' : ''}</div>
                            ${subs.length > 0 ? `<div class="org-card-files">👥 ${subs.length} sub-agents</div>` : ''}
                        </div>
                    `;

                    // Render sub-agents underneath their parent
                    if (subs.length > 0) {
                        html += `<div class="org-sub-connector"></div>`;
                        html += `<div class="org-sub-agents">`;
                        for (const sub of subs) {
                            const subShortName = sub.name.replace(/ Agent$/i, '').replace(/ Expert$/i, '');
                            html += `
                                <div class="org-card sub-agent" title="${escapeHtml(sub.description)}" onclick="showAgentDetails('${sub.id}')">
                                    <span class="org-card-badge sub">Sub</span>
                                    <div class="org-card-icon">🔹</div>
                                    <div class="org-card-name">${escapeHtml(subShortName)}</div>
                                    <div class="org-card-desc">${escapeHtml((sub.description || '').slice(0, 40))}${(sub.description || '').length > 40 ? '...' : ''}</div>
                                    <div class="org-card-files">↑ ${escapeHtml(shortName)}</div>
                                </div>
                            `;
                        }
                        html += `</div></div>`; // close org-sub-agents and org-parent-with-subs
                    }
                }
                html += '</div></div>';
            }

            // Tier C Group (Meta-Agents like QC)
            if (tier_c && tier_c.length > 0) {
                html += '<div class="org-connector"></div>';
                html += `
                    <div class="org-group">
                        <div class="org-group-label">🔮 TIER C - META AGENTS (${tier_c.length})</div>
                        <div class="org-group-cards">
                `;
                for (const agent of tier_c) {
                    const shortName = agent.name.replace(/ Agent$/i, '').replace(/ Expert$/i, '');
                    html += `
                        <div class="org-card tier-c" title="${escapeHtml(agent.description)}" onclick="showAgentDetails('${agent.id}')">
                            <span class="org-card-badge tier-c">C</span>
                            <div class="org-card-icon">🔮</div>
                            <div class="org-card-name">${escapeHtml(shortName)}</div>
                            <div class="org-card-desc">${escapeHtml((agent.description || '').slice(0, 60))}${(agent.description || '').length > 60 ? '...' : ''}</div>
                        </div>
                    `;
                }
                html += '</div></div>';
            }

            return html;
        }

        function showAgentDetails(agentId) {
            // Could open a modal with agent details, for now just log
            console.log('Agent clicked:', agentId);
            // Populate the query input with a suggestion
            const queryInput = document.getElementById('query-input');
            if (queryInput) {
                queryInput.value = `Tell me about your responsibilities as the ${agentId} agent.`;
                queryInput.focus();
            }
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // =========================================================================
        // Sub-Agent Settings Functions
        // =========================================================================

        let settingsPanelOpen = false;

        function toggleSettingsPanel() {
            settingsPanelOpen = !settingsPanelOpen;
            const panel = document.getElementById('settings-panel');
            const toggle = document.getElementById('settings-toggle');
            panel.style.display = settingsPanelOpen ? 'block' : 'none';
            toggle.textContent = settingsPanelOpen ? '▼' : '▶';
            if (settingsPanelOpen) {
                loadSubAgentPolicy();
            }
        }

        async function loadSubAgentPolicy() {
            try {
                const res = await fetch('/api/config/sub-agent-policy');
                const data = await res.json();

                if (data.policy) {
                    document.getElementById('setting-min-files-to-split').value = data.policy.min_files_to_split;
                    document.getElementById('setting-min-subdirs-to-split').value = data.policy.min_subdirs_to_split;
                    document.getElementById('setting-min-files-per-sub').value = data.policy.min_files_per_sub;
                    document.getElementById('setting-max-sub-agents').value = data.policy.max_sub_agents;
                }
            } catch (e) {
                console.error('Failed to load sub-agent policy:', e);
            }
        }

        async function saveSubAgentPolicy() {
            const policy = {
                min_files_to_split: parseInt(document.getElementById('setting-min-files-to-split').value),
                min_subdirs_to_split: parseInt(document.getElementById('setting-min-subdirs-to-split').value),
                min_files_per_sub: parseInt(document.getElementById('setting-min-files-per-sub').value),
                max_sub_agents: parseInt(document.getElementById('setting-max-sub-agents').value),
            };

            try {
                const res = await fetch('/api/config/sub-agent-policy', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(policy),
                });
                const data = await res.json();

                const statusEl = document.getElementById('settings-status');
                if (data.error) {
                    statusEl.innerHTML = `<span class="text-red-400">✗ Error: ${data.error}</span>`;
                } else {
                    statusEl.innerHTML = '<span class="text-green-400">✓ Settings saved! Restart dashboard to apply changes.</span>';
                }
                statusEl.classList.remove('hidden');

                // Hide status after 3 seconds
                setTimeout(() => {
                    statusEl.classList.add('hidden');
                }, 3000);
            } catch (e) {
                console.error('Failed to save sub-agent policy:', e);
                const statusEl = document.getElementById('settings-status');
                statusEl.innerHTML = `<span class="text-red-400">✗ Failed to save settings</span>`;
                statusEl.classList.remove('hidden');
            }
        }

        async function runCodeConcernsAnalysis() {
            const btn = document.getElementById('run-qc-btn');
            const originalText = btn.innerHTML;
            btn.innerHTML = '⏳ Scanning...';
            btn.disabled = true;

            try {
                const res = await fetch('/api/qc/concerns-scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                });
                const data = await res.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Update counts
                const concerns = data.concerns || [];
                let critical = 0, high = 0, medium = 0, low = 0;
                for (const c of concerns) {
                    if (c.severity === 'CRITICAL') critical++;
                    else if (c.severity === 'HIGH') high++;
                    else if (c.severity === 'MEDIUM') medium++;
                    else low++;
                }

                document.getElementById('qc-critical-count').textContent = critical;
                document.getElementById('qc-high-count').textContent = high;
                document.getElementById('qc-medium-count').textContent = medium;
                document.getElementById('qc-low-count').textContent = low;

                // Update concerns list
                const concernsList = document.getElementById('qc-concerns-list');
                if (concerns.length === 0) {
                    concernsList.innerHTML = '<div class="text-gray-500 text-sm">✓ No concerns found from Tier B agents</div>';
                } else {
                    // Group by agent
                    const byAgent = {};
                    for (const c of concerns) {
                        if (!byAgent[c.agent_name]) byAgent[c.agent_name] = [];
                        byAgent[c.agent_name].push(c);
                    }

                    let html = '';
                    for (const [agentName, agentConcerns] of Object.entries(byAgent)) {
                        html += `<div class="mb-3">
                            <div class="text-sm font-semibold text-blue-400 mb-1">📦 ${agentName}</div>`;
                        for (const c of agentConcerns) {
                            const severityColors = {
                                'CRITICAL': 'bg-red-900 text-red-300',
                                'HIGH': 'bg-orange-900 text-orange-300',
                                'MEDIUM': 'bg-yellow-900 text-yellow-300',
                                'LOW': 'bg-gray-700 text-gray-300'
                            };
                            const color = severityColors[c.severity] || severityColors['LOW'];
                            html += `<div class="bg-gray-750 rounded p-2 mb-1 text-sm">
                                <span class="px-1.5 py-0.5 rounded text-xs ${color}">${c.severity}</span>
                                <span class="ml-2">${c.title}</span>
                                ${c.file ? `<div class="text-gray-500 text-xs mt-1">📄 ${c.file}</div>` : ''}
                                ${c.details ? `<div class="text-gray-400 text-xs mt-1">${c.details}</div>` : ''}
                            </div>`;
                        }
                        html += '</div>';
                    }
                    concernsList.innerHTML = html;
                }

                // Show summary
                const summary = `Scanned ${data.agents_queried} agents, found ${concerns.length} concerns`;
                console.log(summary, data);

            } catch (e) {
                console.error('Failed to run concerns analysis:', e);
                alert('Failed to run analysis: ' + e.message);
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }

        // Initialize after DOM loads
        document.addEventListener('DOMContentLoaded', () => {
            loadAgents();
            loadTraces();
            loadRoutingRules();
            loadTree();
            loadQCStatus();
            loadQCReports();
            loadSubAgentPolicy();
            connectWebSocket();

            document.getElementById('send-btn').addEventListener('click', sendQuery);
            document.getElementById('query-input').addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && e.ctrlKey) sendQuery();
            });
        });

        async function loadQCStatus() {
            try {
                const res = await fetch('/api/qc/status');
                const data = await res.json();
                const badge = document.getElementById('qc-status-badge');
                const btn = document.getElementById('qc-enable-btn');

                if (data.enabled) {
                    badge.textContent = data.auto_analyze ? 'auto' : 'manual';
                    badge.className = 'text-xs px-2 py-0.5 rounded bg-green-900 text-green-300';
                    btn.textContent = 'Disable';
                    btn.className = 'text-xs bg-red-600 hover:bg-red-700 px-2 py-1 rounded';
                } else {
                    badge.textContent = 'disabled';
                    badge.className = 'text-xs px-2 py-0.5 rounded bg-gray-700 text-gray-400';
                    btn.textContent = 'Enable';
                    btn.className = 'text-xs bg-blue-600 hover:bg-blue-700 px-2 py-1 rounded';
                }
            } catch (e) {
                console.error('Failed to load QC status:', e);
            }
        }

        async function loadQCReports() {
            try {
                // Load reports
                const reportsRes = await fetch('/api/qc/reports?limit=10');
                const reportsData = await reportsRes.json();

                // Load concerns
                const concernsRes = await fetch('/api/qc/concerns?limit=50');
                const concernsData = await concernsRes.json();

                // Update summary counts
                let totalCritical = 0, totalHigh = 0, totalMedium = 0, totalLow = 0;
                for (const report of (reportsData.reports || [])) {
                    totalCritical += report.critical_count || 0;
                    totalHigh += report.high_count || 0;
                    totalMedium += report.medium_count || 0;
                    totalLow += report.low_count || 0;
                }

                document.getElementById('qc-critical-count').textContent = totalCritical;
                document.getElementById('qc-high-count').textContent = totalHigh;
                document.getElementById('qc-medium-count').textContent = totalMedium;
                document.getElementById('qc-low-count').textContent = totalLow;

                // Store concerns for filtering/sorting
                allConcernsData = concernsData.concerns || [];

                // Render concerns using the filter function (applies current filters)
                filterConcerns();

                // Update reports list
                const reportsList = document.getElementById('qc-reports-list');
                if (reportsData.reports && reportsData.reports.length > 0) {
                    reportsList.innerHTML = reportsData.reports.map(r => `
                        <div class="p-2 rounded bg-gray-800 text-sm cursor-pointer hover:bg-gray-700" onclick="showReportDetail('${r.report_id}')">
                            <div class="flex items-center justify-between">
                                <span class="font-medium">${r.report_id}</span>
                                <span class="px-2 py-0.5 rounded text-xs ${getRecommendationClass(r.recommendation)}">
                                    ${r.recommendation}
                                </span>
                            </div>
                            <div class="text-gray-400 text-xs mt-1">
                                ${r.total_concerns} concerns · ${new Date(r.timestamp).toLocaleString()}
                            </div>
                            <div class="text-gray-300 text-xs mt-1">${escapeHtml(r.overall_assessment?.slice(0, 100) || '')}...</div>
                        </div>
                    `).join('');
                } else {
                    reportsList.innerHTML = '<div class="text-gray-500 text-sm">No reports yet</div>';
                }
            } catch (e) {
                console.error('Failed to load QC reports:', e);
            }
        }

        function getRecommendationClass(rec) {
            switch (rec) {
                case 'approve': return 'bg-green-900 text-green-300';
                case 'review': return 'bg-yellow-900 text-yellow-300';
                case 'block': return 'bg-red-900 text-red-300';
                default: return 'bg-gray-700 text-gray-300';
            }
        }

        // Store current report data for export
        let currentReportData = null;
        let allConcernsData = [];
        let selectedFiles = new Set();

        async function showReportDetail(reportId) {
            try {
                const res = await fetch(`/api/qc/reports/${reportId}`);
                const data = await res.json();
                if (data.error) {
                    alert('Error loading report: ' + data.error);
                    return;
                }

                currentReportData = data;
                document.getElementById('modal-report-title').textContent = `QC Report: ${data.report_id}`;

                const riskColors = {
                    'critical': 'text-red-500',
                    'high': 'text-orange-500',
                    'medium': 'text-yellow-500',
                    'low': 'text-green-500'
                };

                let html = `
                    <div class="space-y-4">
                        <!-- Summary -->
                        <div class="grid grid-cols-2 gap-4 text-sm">
                            <div class="bg-gray-900 p-3 rounded">
                                <div class="text-gray-400">Risk Level</div>
                                <div class="text-xl font-bold ${riskColors[data.risk_level] || 'text-gray-300'}">${data.risk_level?.toUpperCase() || 'N/A'}</div>
                            </div>
                            <div class="bg-gray-900 p-3 rounded">
                                <div class="text-gray-400">Recommendation</div>
                                <div class="text-xl font-bold ${getRecommendationClass(data.recommendation).replace('bg-', 'text-').replace('-900', '-400')}">${data.recommendation?.toUpperCase() || 'N/A'}</div>
                            </div>
                        </div>

                        <!-- Assessment -->
                        <div class="bg-gray-900 p-3 rounded">
                            <div class="text-gray-400 text-sm mb-1">Overall Assessment</div>
                            <div class="text-gray-200">${escapeHtml(data.overall_assessment || 'No assessment')}</div>
                        </div>

                        <!-- Stats -->
                        <div class="grid grid-cols-4 gap-2 text-center text-xs">
                            <div class="bg-gray-900 p-2 rounded">
                                <div class="text-red-500 font-bold">${data.critical_count || 0}</div>
                                <div class="text-gray-500">Critical</div>
                            </div>
                            <div class="bg-gray-900 p-2 rounded">
                                <div class="text-orange-500 font-bold">${data.high_count || 0}</div>
                                <div class="text-gray-500">High</div>
                            </div>
                            <div class="bg-gray-900 p-2 rounded">
                                <div class="text-yellow-500 font-bold">${data.medium_count || 0}</div>
                                <div class="text-gray-500">Medium</div>
                            </div>
                            <div class="bg-gray-900 p-2 rounded">
                                <div class="text-green-500 font-bold">${data.low_count || 0}</div>
                                <div class="text-gray-500">Low</div>
                            </div>
                        </div>

                        <!-- Concerns -->
                        <div class="bg-gray-900 p-3 rounded">
                            <div class="text-gray-400 text-sm mb-2">Concerns (${data.concerns?.length || 0})</div>
                            <div class="space-y-2 max-h-[300px] overflow-y-auto">
                                ${(data.concerns || []).map(c => `
                                    <div class="p-2 bg-gray-800 rounded cursor-pointer hover:bg-gray-700" onclick="showConcernDetail('${c.concern_id}')">
                                        <div class="flex items-center gap-2">
                                            <span class="severity-${c.severity}">${c.severity?.toUpperCase()}</span>
                                            <span class="text-white font-medium text-sm">${escapeHtml(c.title)}</span>
                                        </div>
                                        <div class="text-gray-400 text-xs mt-1">${escapeHtml(c.domain)} · ${escapeHtml(c.category)}</div>
                                    </div>
                                `).join('') || '<div class="text-gray-500">No concerns</div>'}
                            </div>
                        </div>

                        <!-- Action Items -->
                        ${data.action_items?.length ? `
                        <div class="bg-gray-900 p-3 rounded">
                            <div class="text-gray-400 text-sm mb-2">Action Items</div>
                            <div class="space-y-2">
                                ${data.action_items.map(a => `
                                    <div class="p-2 bg-gray-800 rounded text-sm">
                                        <div class="flex items-center gap-2">
                                            <span class="text-xs px-1.5 py-0.5 rounded bg-blue-900 text-blue-300">P${a.priority}</span>
                                            <span class="text-white">${escapeHtml(a.title)}</span>
                                        </div>
                                        <div class="text-gray-400 text-xs mt-1">${escapeHtml(a.description)}</div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        ` : ''}

                        <!-- Metadata -->
                        <div class="text-xs text-gray-500 flex gap-4">
                            <span>Time: ${data.total_analysis_time_ms}ms</span>
                            <span>Tokens: ${data.total_tokens_used}</span>
                            <span>Agents: ${(data.agents_consulted || []).join(', ') || 'None'}</span>
                        </div>
                    </div>
                `;

                document.getElementById('modal-report-content').innerHTML = html;
                document.getElementById('report-modal').classList.remove('hidden');
            } catch (e) {
                console.error('Failed to load report:', e);
                alert('Failed to load report: ' + e.message);
            }
        }

        function closeReportModal() {
            document.getElementById('report-modal').classList.add('hidden');
        }

        function exportReport() {
            if (!currentReportData) return;
            const blob = new Blob([JSON.stringify(currentReportData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `qc-report-${currentReportData.report_id}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }

        function showConcernDetail(concernId) {
            // Find concern from current report
            const concern = currentReportData?.concerns?.find(c => c.concern_id === concernId);
            if (!concern) return;

            document.getElementById('modal-concern-title').textContent = concern.title;
            document.getElementById('modal-concern-content').innerHTML = `
                <div class="space-y-3">
                    <div class="flex gap-2">
                        <span class="severity-${concern.severity} text-sm">${concern.severity?.toUpperCase()}</span>
                        <span class="text-xs px-2 py-0.5 rounded bg-gray-700 text-gray-300">${concern.category}</span>
                    </div>
                    <div class="bg-gray-900 p-3 rounded">
                        <div class="text-gray-400 text-xs mb-1">Description</div>
                        <div class="text-gray-200 text-sm">${escapeHtml(concern.description)}</div>
                    </div>
                    ${concern.suggestion ? `
                    <div class="bg-gray-900 p-3 rounded">
                        <div class="text-gray-400 text-xs mb-1">Suggestion</div>
                        <div class="text-green-300 text-sm">${escapeHtml(concern.suggestion)}</div>
                    </div>
                    ` : ''}
                    ${concern.affected_files?.length ? `
                    <div class="bg-gray-900 p-3 rounded">
                        <div class="text-gray-400 text-xs mb-1">Affected Files</div>
                        <div class="text-gray-300 text-sm font-mono">${concern.affected_files.join('<br>')}</div>
                    </div>
                    ` : ''}
                    ${concern.affected_functions?.length ? `
                    <div class="bg-gray-900 p-3 rounded">
                        <div class="text-gray-400 text-xs mb-1">Affected Functions</div>
                        <div class="text-gray-300 text-sm font-mono">${concern.affected_functions.join(', ')}</div>
                    </div>
                    ` : ''}
                    <div class="text-xs text-gray-500">
                        <span>Domain: ${concern.domain}</span> ·
                        <span>Agent: ${concern.agent_id}</span> ·
                        <span>Confidence: ${(concern.confidence * 100).toFixed(0)}%</span>
                    </div>
                </div>
            `;
            document.getElementById('concern-modal').classList.remove('hidden');
        }

        function closeConcernModal() {
            document.getElementById('concern-modal').classList.add('hidden');
        }

        // File picker functions
        async function loadProjectFiles() {
            const picker = document.getElementById('file-picker');
            picker.innerHTML = '<div class="text-gray-400">Loading files...</div>';

            try {
                const res = await fetch('/api/files');
                const data = await res.json();

                if (data.error) {
                    picker.innerHTML = `<div class="text-red-400">${escapeHtml(data.error)}</div>`;
                    return;
                }

                if (!data.directories || data.directories.length === 0) {
                    picker.innerHTML = '<div class="text-gray-500">No files found</div>';
                    return;
                }

                let html = '';
                for (const dir of data.directories) {
                    html += `
                        <div class="mb-2">
                            <div class="flex items-center gap-1 text-gray-400 font-medium">
                                <input type="checkbox" class="dir-checkbox" data-dir="${escapeHtml(dir.directory)}" onchange="toggleDirectory(this)">
                                <span>📁 ${escapeHtml(dir.directory)}</span>
                            </div>
                            <div class="ml-4 space-y-0.5">
                                ${dir.files.map(f => `
                                    <label class="flex items-center gap-1 hover:bg-gray-700 rounded px-1 cursor-pointer">
                                        <input type="checkbox" class="file-checkbox" data-path="${escapeHtml(f.path)}" onchange="updateSelectedCount()">
                                        <span class="text-gray-300">${escapeHtml(f.name)}</span>
                                        <span class="text-gray-600 text-xs">(${(f.size / 1024).toFixed(1)}KB)</span>
                                    </label>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }

                picker.innerHTML = html;
                updateSelectedCount();
            } catch (e) {
                picker.innerHTML = `<div class="text-red-400">Error: ${escapeHtml(e.message)}</div>`;
            }
        }

        function toggleDirectory(checkbox) {
            const dir = checkbox.dataset.dir;
            const fileCheckboxes = document.querySelectorAll(`.file-checkbox[data-path^="${dir}/"]`);
            fileCheckboxes.forEach(cb => cb.checked = checkbox.checked);
            updateSelectedCount();
        }

        function updateSelectedCount() {
            const checked = document.querySelectorAll('.file-checkbox:checked');
            selectedFiles = new Set([...checked].map(cb => cb.dataset.path));
            document.getElementById('selected-files-count').textContent = `${selectedFiles.size} files selected`;
            document.getElementById('analyze-btn').disabled = selectedFiles.size === 0;
        }

        async function triggerQCAnalysis() {
            if (selectedFiles.size === 0) {
                alert('Please select files to analyze');
                return;
            }

            const btn = document.getElementById('analyze-btn');
            btn.disabled = true;
            btn.textContent = 'Analyzing...';

            try {
                const res = await fetch('/api/qc/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ files: [...selectedFiles] })
                });
                const data = await res.json();

                if (data.error) {
                    alert('Analysis failed: ' + data.error);
                } else {
                    // Analysis completed - refresh reports
                    await loadQCReports();
                    // Show the new report
                    if (data.report_id) {
                        showReportDetail(data.report_id);
                    }
                }
            } catch (e) {
                alert('Analysis failed: ' + e.message);
            }

            btn.disabled = false;
            btn.textContent = 'Analyze Selected';
            updateSelectedCount();
        }

        // Filter and sort concerns
        function filterConcerns() {
            const filter = document.getElementById('qc-severity-filter').value;
            const sort = document.getElementById('qc-sort').value;

            let concerns = [...allConcernsData];

            // Filter by severity
            if (filter !== 'all') {
                const severityOrder = ['critical', 'high', 'medium', 'low', 'info'];
                const filterIndex = severityOrder.indexOf(filter);
                concerns = concerns.filter(c => severityOrder.indexOf(c.severity) <= filterIndex);
            }

            // Sort
            if (sort === 'severity') {
                const severityOrder = { 'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4 };
                concerns.sort((a, b) => (severityOrder[a.severity] || 5) - (severityOrder[b.severity] || 5));
            } else if (sort === 'time') {
                concerns.sort((a, b) => new Date(b.raised_at) - new Date(a.raised_at));
            } else if (sort === 'category') {
                concerns.sort((a, b) => (a.category || '').localeCompare(b.category || ''));
            }

            renderConcernsList(concerns);
        }

        function renderConcernsList(concerns) {
            const concernsList = document.getElementById('qc-concerns-list');
            if (concerns && concerns.length > 0) {
                concernsList.innerHTML = concerns.map(c => `
                    <div class="p-2 rounded bg-gray-800 text-sm cursor-pointer hover:bg-gray-700 concern-item" onclick="showConcernFromList('${c.concern_id}', '${c.report_id}')">
                        <div class="flex items-center gap-2">
                            <span class="severity-${c.severity}">${c.severity.toUpperCase()}</span>
                            <span class="text-white font-medium">${escapeHtml(c.title)}</span>
                        </div>
                        <div class="text-gray-400 text-xs mt-1">${escapeHtml(c.domain)} · ${escapeHtml(c.category)}</div>
                        <div class="text-gray-300 text-xs mt-1">${escapeHtml(c.description?.slice(0, 150) || '')}${c.description?.length > 150 ? '...' : ''}</div>
                        ${c.affected_files?.length ? `<div class="text-gray-500 text-xs mt-1">Files: ${c.affected_files.join(', ')}</div>` : ''}
                    </div>
                `).join('');
            } else {
                concernsList.innerHTML = '<div class="text-gray-500 text-sm">No concerns match filter</div>';
            }
        }

        async function showConcernFromList(concernId, reportId) {
            // Load the full report to get concern details
            try {
                const res = await fetch(`/api/qc/reports/${reportId}`);
                const data = await res.json();
                if (!data.error) {
                    currentReportData = data;
                    showConcernDetail(concernId);
                }
            } catch (e) {
                console.error('Failed to load concern details:', e);
            }
        }

        // Add real-time concern to the stream
        function addConcernToStream(concernData) {
            // Add to our local data
            allConcernsData.unshift({
                ...concernData,
                raised_at: new Date().toISOString()
            });
            // Keep only recent ones
            if (allConcernsData.length > 50) {
                allConcernsData = allConcernsData.slice(0, 50);
            }
            // Re-render with current filters
            filterConcerns();

            // Flash the concern item
            setTimeout(() => {
                const firstItem = document.querySelector('.concern-item');
                if (firstItem) {
                    firstItem.classList.add('bg-purple-900');
                    setTimeout(() => firstItem.classList.remove('bg-purple-900'), 1000);
                }
            }, 100);
        }

        function connectWebSocket() {
            const statusEl = document.getElementById('connection-status');
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                statusEl.innerHTML = '<span class="w-3 h-3 rounded-full bg-green-500"></span><span class="text-sm">Connected</span>';
            };

            ws.onclose = () => {
                statusEl.innerHTML = '<span class="w-3 h-3 rounded-full bg-red-500"></span><span class="text-sm">Disconnected</span>';
                setTimeout(connectWebSocket, 3000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                addEvent(data);
                loadTraces();
                loadStats();

                // Handle real-time concern streaming
                if (data.event_type === 'concern_raised' && data.data) {
                    addConcernToStream(data.data);
                }

                // Refresh QC panel on QC events
                if (data.event_type && data.event_type.startsWith('analysis_') ||
                    data.event_type === 'concern_raised' ||
                    data.event_type === 'qc_synthesis_started' ||
                    data.event_type === 'qc_report_complete') {
                    loadQCReports();
                }

                // Handle agent stale/fresh events
                if (data.event_type === 'agent_stale' || data.event_type === 'agent_fresh') {
                    loadAgentStatuses();
                }

                // Refresh agent statuses on file changes
                if (data.event_type === 'file_change_detected') {
                    loadAgentStatuses();
                }
            };
        }

        let agentStatuses = {};

        async function loadAgentStatuses() {
            try {
                const res = await fetch('/api/agent-status');
                const data = await res.json();
                agentStatuses = {};
                for (const s of (data.statuses || [])) {
                    agentStatuses[s.agent_id] = s;
                }
                // Re-render agents list with statuses
                renderAgentsList();
            } catch (e) {
                console.error('Failed to load agent statuses:', e);
            }
        }

        function getStatusIndicator(status) {
            // status: "fresh" | "stale" | "never_queried"
            if (status === 'fresh') {
                return '<span class="w-2 h-2 rounded-full bg-green-500 inline-block" title="Up-to-date"></span>';
            } else if (status === 'stale') {
                return '<span class="w-2 h-2 rounded-full bg-yellow-500 inline-block" title="Has pending changes"></span>';
            } else {
                return '<span class="w-2 h-2 rounded-full bg-gray-500 inline-block" title="Never queried"></span>';
            }
        }

        function renderAgentsList() {
            const list = document.getElementById('agents-list');
            const select = document.getElementById('agent-select');

            const tierClass = (tier) => {
                if (tier === 'A') return 'tier-a';
                if (tier === 'B') return 'tier-b';
                if (tier === 'C') return 'tier-c';
                return 'tier-a';
            };
            const tierBgClass = (tier) => {
                if (tier === 'A') return 'bg-blue-900';
                if (tier === 'B') return 'bg-green-900';
                if (tier === 'C') return 'bg-purple-900';
                return 'bg-blue-900';
            };

            list.innerHTML = agents.map(a => {
                const status = agentStatuses[a.id] || { status: 'never_queried' };
                const statusIndicator = getStatusIndicator(status.status);
                const changedCount = status.changed_file_count || 0;
                const changedHint = changedCount > 0 ? ` (${changedCount} changed)` : '';

                return `
                <div class="p-2 rounded ${tierClass(a.tier)} bg-gray-700">
                    <div class="flex items-center gap-2">
                        ${statusIndicator}
                        <span class="font-medium">${a.name}</span>
                    </div>
                    <div class="text-xs text-gray-400">${a.id}${changedHint}</div>
                    <div class="text-xs mt-1">
                        <span class="px-1 py-0.5 rounded ${tierBgClass(a.tier)}">
                            Tier ${a.tier}
                        </span>
                        <span class="text-gray-500 ml-1">${a.module_type}${a.role ? ' · ' + a.role : ''}</span>
                    </div>
                </div>
            `}).join('');

            select.innerHTML = '<option value="">Auto-route</option>' +
                agents.map(a => `<option value="${a.id}">${a.name} (${a.tier})</option>`).join('');
        }

        async function loadAgents() {
            const res = await fetch('/api/agents');
            const data = await res.json();
            agents = data.agents;

            // Also load statuses
            await loadAgentStatuses();
        }

        async function loadTraces() {
            const res = await fetch('/api/traces?limit=10');
            const data = await res.json();

            const tbody = document.getElementById('traces-body');
            tbody.innerHTML = data.traces.map(t => {
                const agent = agents.find(a => a.id === t.final_agent) || {};
                const tierBg = agent.tier === 'A' ? 'bg-blue-900' : agent.tier === 'C' ? 'bg-purple-900' : 'bg-green-900';
                return `
                    <tr class="hover:bg-gray-700">
                        <td class="py-2">${new Date(t.started_at).toLocaleTimeString()}</td>
                        <td class="py-2 max-w-xs truncate">${t.query}</td>
                        <td class="py-2">${t.final_agent || '-'}</td>
                        <td class="py-2">
                            <span class="px-1.5 py-0.5 rounded text-xs ${tierBg}">
                                ${agent.tier || '?'}
                            </span>
                        </td>
                        <td class="py-2">${t.tokens_used || 0}</td>
                        <td class="py-2">${t.duration_ms ? t.duration_ms + 'ms' : '-'}</td>
                    </tr>
                `;
            }).join('');
        }

        async function loadStats() {
            const res = await fetch('/api/stats');
            const data = await res.json();

            const panel = document.getElementById('stats-panel');
            const entries = Object.entries(data.stats);

            if (entries.length === 0) {
                panel.innerHTML = '<div class="text-gray-400">No data yet</div>';
                return;
            }

            panel.innerHTML = entries.map(([id, s]) => `
                <div class="flex justify-between">
                    <span class="text-gray-400">${id}</span>
                    <span>${s.queries} queries, ${s.total_tokens} tokens</span>
                </div>
            `).join('');
        }

        async function loadRoutingRules() {
            const res = await fetch('/api/routing-rules');
            const data = await res.json();

            const panel = document.getElementById('routing-panel');

            if (!data.rules || data.rules.length === 0) {
                panel.innerHTML = '<div class="text-gray-400">No routing rules</div>';
                return;
            }

            let html = `<div class="text-xs text-gray-500 mb-2">${data.explanation}</div>`;

            html += data.rules.map(rule => {
                const tierCls = rule.tier === 'A' ? 'tier-a' : rule.tier === 'C' ? 'tier-c' : 'tier-b';
                const tierBg = rule.tier === 'A' ? 'bg-blue-900' : rule.tier === 'C' ? 'bg-purple-900' : 'bg-green-900';
                return `
                <div class="p-2 rounded bg-gray-700 ${tierCls}">
                    <div class="flex items-center gap-2">
                        <span class="font-medium text-sm">${rule.agent_name}</span>
                        <span class="px-1 py-0.5 rounded text-xs ${tierBg}">
                            Tier ${rule.tier}
                        </span>
                    </div>
                    <div class="text-xs text-gray-400 mt-1">${rule.description || ''}</div>
                    <div class="flex flex-wrap gap-1 mt-2">
                        ${rule.keywords.slice(0, 8).map(kw => `
                            <span class="px-1.5 py-0.5 bg-gray-600 rounded text-xs">${kw}</span>
                        `).join('')}
                        ${rule.keywords.length > 8 ? `<span class="text-xs text-gray-500">+${rule.keywords.length - 8} more</span>` : ''}
                    </div>
                </div>
            `}).join('');

            panel.innerHTML = html;
        }

        // =========================================================================
        // DAG Team Execution Functions
        // =========================================================================

        let dagPanelOpen = false;
        let currentTraceData = null;

        function toggleDAGPanel() {
            dagPanelOpen = !dagPanelOpen;
            const panel = document.getElementById('dag-panel');
            const toggle = document.getElementById('dag-toggle');
            panel.style.display = dagPanelOpen ? 'block' : 'none';
            toggle.textContent = dagPanelOpen ? '▼' : '▶';
            if (dagPanelOpen) {
                loadTeamTraces();
            }
        }

        async function loadTeamTraces() {
            try {
                const res = await fetch('/api/team-traces?limit=20');
                const data = await res.json();

                // Update count badge
                const badge = document.getElementById('dag-count-badge');
                badge.textContent = `${data.traces.length} traces`;

                // Render trace list
                const list = document.getElementById('dag-trace-list');
                if (!data.traces || data.traces.length === 0) {
                    list.innerHTML = '<div class="text-gray-500 text-sm">No team executions yet</div>';
                    return;
                }

                list.innerHTML = data.traces.map(t => `
                    <div class="p-2 rounded bg-gray-800 hover:bg-gray-700 cursor-pointer" onclick="loadTraceDetail('${t.session_id}')">
                        <div class="text-sm truncate text-white">${escapeHtml(t.query.substring(0, 50))}${t.query.length > 50 ? '...' : ''}</div>
                        <div class="flex items-center gap-2 mt-1 text-xs text-gray-400">
                            <span>${t.agents_used.length} agents</span>
                            <span>·</span>
                            <span>${t.layer_count} layers</span>
                            <span>·</span>
                            <span>${t.total_tokens} tokens</span>
                        </div>
                        <div class="text-xs text-gray-500 mt-1">${new Date(t.timestamp).toLocaleTimeString()}</div>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to load team traces:', e);
            }
        }

        async function loadTraceDetail(sessionId) {
            try {
                const res = await fetch(`/api/team-trace/${sessionId}`);
                const data = await res.json();
                if (data.error) {
                    console.error(data.error);
                    return;
                }
                currentTraceData = data;
                renderDAG(data);
                renderDAGSummary(data.summary);
            } catch (e) {
                console.error('Failed to load trace detail:', e);
            }
        }

        function renderDAG(data) {
            const container = document.getElementById('dag-visualization');

            if (!data.nodes || data.nodes.length === 0) {
                container.innerHTML = '<div class="text-gray-500 text-sm">No nodes in DAG</div>';
                return;
            }

            // Calculate dimensions based on layers and nodes
            const layerCount = data.execution_layers.length;
            const maxNodesInLayer = Math.max(...data.execution_layers.map(l => l.length));

            const nodeWidth = 120;
            const nodeHeight = 50;
            const layerGap = 100;
            const nodeGap = 20;

            const svgWidth = Math.max(350, layerCount * (nodeWidth + layerGap));
            const svgHeight = Math.max(200, maxNodesInLayer * (nodeHeight + nodeGap) + 50);

            // Build node positions
            const nodePositions = {};
            data.nodes.forEach(node => {
                const layerX = node.layer * (nodeWidth + layerGap) + 50;
                const nodesInLayer = data.execution_layers[node.layer]?.length || 1;
                const layerHeight = nodesInLayer * (nodeHeight + nodeGap);
                const startY = (svgHeight - layerHeight) / 2;
                const nodeY = startY + node.position * (nodeHeight + nodeGap) + nodeHeight / 2;

                nodePositions[node.id] = { x: layerX + nodeWidth / 2, y: nodeY };
            });

            // Generate SVG
            let svg = `<svg width="${svgWidth}" height="${svgHeight}" class="w-full" viewBox="0 0 ${svgWidth} ${svgHeight}">`;

            // Arrow marker definition
            svg += `
                <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7"
                            refX="10" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#4b5563"/>
                    </marker>
                </defs>
            `;

            // Draw edges first (behind nodes)
            (data.edges || []).forEach(edge => {
                const fromPos = nodePositions[edge.from];
                const toPos = nodePositions[edge.to];
                if (fromPos && toPos) {
                    svg += `
                        <line x1="${fromPos.x + nodeWidth/2}" y1="${fromPos.y}"
                              x2="${toPos.x - nodeWidth/2}" y2="${toPos.y}"
                              stroke="#4b5563" stroke-width="2" marker-end="url(#arrowhead)"/>
                    `;
                }
            });

            // Draw nodes
            data.nodes.forEach(node => {
                const pos = nodePositions[node.id];
                if (!pos) return;

                const statusColor = node.status === 'done' ? '#10b981' : '#6b7280';
                const rx = pos.x - nodeWidth / 2;
                const ry = pos.y - nodeHeight / 2;
                const shortId = node.id.length > 14 ? node.id.substring(0, 12) + '..' : node.id;

                svg += `
                    <g class="cursor-pointer" onclick="showNodeDetail('${node.id}')">
                        <rect x="${rx}" y="${ry}" width="${nodeWidth}" height="${nodeHeight}"
                              rx="6" fill="#1f2937" stroke="${statusColor}" stroke-width="2"/>
                        <text x="${pos.x}" y="${pos.y - 5}" text-anchor="middle" fill="white" font-size="11"
                              font-weight="bold">${shortId}</text>
                        <text x="${pos.x}" y="${pos.y + 10}" text-anchor="middle" fill="#9ca3af" font-size="9">
                            ${node.tokens}t · ${node.time_ms}ms
                        </text>
                    </g>
                `;
            });

            // Layer labels
            data.execution_layers.forEach((layer, idx) => {
                const x = idx * (nodeWidth + layerGap) + 50 + nodeWidth / 2;
                svg += `<text x="${x}" y="20" text-anchor="middle" fill="#6b7280" font-size="10">Layer ${idx + 1}</text>`;
            });

            svg += '</svg>';
            container.innerHTML = svg;
        }

        function showNodeDetail(nodeId) {
            if (!currentTraceData) return;

            const node = currentTraceData.nodes.find(n => n.id === nodeId);
            if (!node) return;

            document.getElementById('dag-node-title').textContent = nodeId;
            document.getElementById('dag-node-question').textContent = node.sub_question || 'N/A';
            document.getElementById('dag-node-tokens').textContent = node.tokens;
            document.getElementById('dag-node-time').textContent = node.time_ms + 'ms';
            document.getElementById('dag-node-response').textContent = node.response || 'No response';

            document.getElementById('dag-node-detail').classList.remove('hidden');
        }

        function renderDAGSummary(summary) {
            if (!summary) return;

            document.getElementById('dag-total-tokens').textContent = summary.total_tokens;
            document.getElementById('dag-decomp-tokens').textContent = summary.decomposition_tokens;
            document.getElementById('dag-synth-tokens').textContent = summary.synthesis_tokens;
            document.getElementById('dag-total-time').textContent = summary.total_time_ms + 'ms';
            document.getElementById('dag-speedup').textContent = summary.parallel_speedup.toFixed(2) + 'x';

            document.getElementById('dag-summary').classList.remove('hidden');
        }

        function addEvent(event) {
            const panel = document.getElementById('events-panel');

            // Remove placeholder
            if (panel.querySelector('.text-gray-400')) {
                panel.innerHTML = '';
            }

            const eventClasses = {
                'query_received': 'event-query',
                'routing_decision': 'event-routing',
                'agent_started': 'event-agent',
                'context_loaded': 'event-agent',
                'agent_response': 'event-response',
                'error': 'event-error',
                // Claude Code events
                'claude_code_task': 'event-claude-code',
                'claude_code_search': 'event-claude-code',
                'claude_code_read': 'event-claude-code',
                'claude_code_edit': 'event-claude-code',
                'claude_code_result': 'event-claude-result',
                // Cache/refresh events
                'context_refresh': 'event-refresh',
                'git_change_detected': 'event-refresh',
                'file_change_detected': 'event-refresh',
                // QC events
                'analysis_started': 'event-qc-started',
                'analysis_agent_started': 'event-qc-started',
                'analysis_agent_complete': 'event-qc-started',
                'concern_raised': 'event-qc-concern',
                'qc_synthesis_started': 'event-qc-started',
                'qc_report_complete': 'event-qc-report',
            };

            // Check if this is a benchmark event (via _event_type detail)
            const benchmarkType = event.data?._event_type || '';
            if (benchmarkType.startsWith('benchmark_')) {
                // Override class for benchmark events
                if (benchmarkType.includes('started')) eventClasses[event.event_type] = 'bg-indigo-100 border-l-3 border-indigo-500';
                if (benchmarkType.includes('completed')) eventClasses[event.event_type] = 'bg-indigo-50 border-l-3 border-indigo-400';
            }

            const eventIcons = {
                'query_received': '📝',
                'routing_decision': '🔀',
                'agent_started': '🤖',
                'context_loaded': '📚',
                'agent_response': '💬',
                'error': '❌',
                // Claude Code events
                'claude_code_task': '🔧',
                'claude_code_search': '🔍',
                'claude_code_read': '📖',
                'claude_code_edit': '✏️',
                'claude_code_result': '✅',
                // Cache/refresh events
                'context_refresh': '🔄',
                'git_change_detected': '📦',
                'file_change_detected': '📁',
                // QC events
                'analysis_started': '🔬',
                'analysis_agent_started': '🔎',
                'analysis_agent_complete': '✓',
                'concern_raised': '⚠️',
                'qc_synthesis_started': '📊',
                'qc_report_complete': '📋',
            };

            // Override icon for benchmark events
            if (benchmarkType === 'benchmark_started') eventIcons[event.event_type] = '🏁';
            if (benchmarkType === 'benchmark_completed') eventIcons[event.event_type] = '🏆';
            if (benchmarkType === 'benchmark_repo_started') eventIcons[event.event_type] = '📂';
            if (benchmarkType === 'benchmark_repo_completed') eventIcons[event.event_type] = '📊';
            if (benchmarkType === 'benchmark_task_started') eventIcons[event.event_type] = '▶️';
            if (benchmarkType === 'benchmark_task_completed') eventIcons[event.event_type] = '✅';

            const div = document.createElement('div');
            div.className = `event-card p-2 rounded text-sm ${eventClasses[event.event_type] || 'bg-gray-700'}`;
            div.innerHTML = `
                <div class="flex items-center gap-2 text-gray-900">
                    <span>${eventIcons[event.event_type] || '•'}</span>
                    <span class="font-medium">${event.event_type}</span>
                    <span class="text-xs text-gray-600 ml-auto">${new Date(event.timestamp).toLocaleTimeString()}</span>
                </div>
                <div class="mt-1 text-xs text-gray-700">
                    ${formatEventData(event)}
                </div>
            `;

            panel.insertBefore(div, panel.firstChild);

            // Keep only last 50 events
            while (panel.children.length > 50) {
                panel.removeChild(panel.lastChild);
            }
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function formatEventData(event) {
            const d = event.data;
            const repoTag = d.repo ? `<span class="inline-block bg-indigo-200 text-indigo-800 text-xs px-1.5 py-0.5 rounded font-medium mr-1">${escapeHtml(d.repo)}</span>` : '';

            // Handle benchmark lifecycle events first
            const benchmarkType = d._event_type || '';
            if (benchmarkType.startsWith('benchmark_')) {
                switch (benchmarkType) {
                    case 'benchmark_started':
                        return `<strong>Benchmark Started</strong><br>Repos: ${(d.repos || []).join(', ')}<br>Conditions: ${(d.conditions || []).join(', ')}<br>Run ID: ${d.run_id || '?'}`;
                    case 'benchmark_completed':
                        return `<strong>Benchmark Complete</strong><br>Passed: ${d.passed || 0}/${d.total || 0}<br>Run ID: ${d.run_id || '?'}`;
                    case 'benchmark_repo_started':
                        return `${repoTag}<strong>Repo Testing Started</strong> — Condition ${d.condition || '?'}`;
                    case 'benchmark_repo_completed':
                        return `${repoTag}<strong>Repo Testing Done</strong> — ${d.passed || 0}/${d.total || 0} passed (Condition ${d.condition || '?'})`;
                    case 'benchmark_task_started':
                        return `${repoTag}<strong>${escapeHtml(d.task_id || '?')}</strong> (${d.task_type || '?'}, Condition ${d.condition || '?'})`;
                    case 'benchmark_task_completed':
                        const status = d.success ? (d.test_passed !== undefined ? (d.test_passed ? '✅ PASS' : '❌ FAIL') : '✅ OK') : '❌ ERROR';
                        return `${repoTag}<strong>${escapeHtml(d.task_id || '?')}</strong> ${status} (${d.time_seconds || 0}s)`;
                    default:
                        return `${repoTag}${d.description || JSON.stringify(d).slice(0, 100)}`;
                }
            }

            switch (event.event_type) {
                case 'query_received':
                    return `"${d.query}"${d.forced_agent ? ` → forced to ${d.forced_agent}` : ''}`;
                case 'routing_decision':
                    return `Selected: <strong>${d.selected_agent}</strong> from ${d.available_agents?.length || 0} agents`;
                case 'agent_started':
                    return `${d.agent_name} (Tier ${d.tier}, ${d.agent_type})`;
                case 'context_loaded':
                    return `${(d.context_size / 1024).toFixed(1)}KB context loaded`;
                case 'agent_response':
                    return `${d.tokens_used} tokens, ${d.artifacts_count} artifacts`;
                case 'error':
                    return `Error: ${d.error}`;
                // Claude Code events
                case 'claude_code_task':
                    const queryText = d.query || d.description || '';
                    return `${repoTag}<div class="mb-1"><strong>Query:</strong></div>
                        <div class="bg-gray-800 p-2 rounded text-white whitespace-pre-wrap max-h-40 overflow-y-auto">${escapeHtml(queryText)}</div>`;
                case 'claude_code_search':
                    return `${repoTag}<strong>Search:</strong> ${d.description}${d.pattern ? ` (${d.pattern})` : ''}`;
                case 'claude_code_read':
                    return `${repoTag}<strong>Read:</strong> ${d.description}${d.file ? ` - ${d.file}` : ''}`;
                case 'claude_code_edit':
                    return `${repoTag}<strong>Edit:</strong> ${d.description}${d.file ? ` - ${d.file}` : ''}`;
                case 'claude_code_result':
                    const resp = d.response || d.description || '';
                    const query = d.query || '';
                    return `${repoTag}<div class="mb-1"><strong>Agent:</strong> ${d.agent_id || 'unknown'} | <strong>Tokens:</strong> ${d.tokens_used || 0}</div>
                        ${query ? `<details class="mb-2"><summary class="cursor-pointer text-blue-600">View Query</summary>
                        <div class="bg-gray-800 p-2 rounded text-white whitespace-pre-wrap max-h-32 overflow-y-auto mt-1">${escapeHtml(query)}</div></details>` : ''}
                        <details open><summary class="cursor-pointer text-green-600 font-medium">Response</summary>
                        <div class="bg-gray-800 p-2 rounded text-white whitespace-pre-wrap max-h-96 overflow-y-auto mt-1">${escapeHtml(resp)}</div></details>`;
                // Cache/refresh events
                case 'context_refresh':
                    const agents = d.agents || [];
                    return `<strong>Context Refreshed</strong><br>Agents: ${agents.length > 0 ? agents.join(', ') : 'all'}`;
                case 'git_change_detected':
                    return `<strong>Git Change Detected</strong><br>New commit: ${d.new_commit?.slice(0, 8) || 'unknown'}<br>Previous: ${d.old_commit?.slice(0, 8) || 'unknown'}`;
                case 'file_change_detected':
                    const files = d.files || [];
                    return `<strong>File Changes Detected</strong><br>Files: ${files.slice(0, 5).join(', ')}${files.length > 5 ? ` +${files.length - 5} more` : ''}`;
                // QC events
                case 'analysis_started':
                    const analysisFiles = d.files || [];
                    return `<strong>QC Analysis Started</strong><br>Files: ${analysisFiles.slice(0, 5).join(', ')}${analysisFiles.length > 5 ? ` +${analysisFiles.length - 5} more` : ''}`;
                case 'analysis_agent_started':
                    return `<strong>Agent Analyzing:</strong> ${d.agent_id || 'unknown'}<br>Domain: ${d.domain || 'unknown'}`;
                case 'analysis_agent_complete':
                    return `<strong>Agent Complete:</strong> ${d.agent_id || 'unknown'}<br>Concerns: ${d.concerns_count || 0} | ${d.analysis_time_ms || 0}ms`;
                case 'concern_raised':
                    return `<span class="severity-${d.severity || 'medium'}">${(d.severity || 'MEDIUM').toUpperCase()}</span> <strong>${escapeHtml(d.title || 'Concern')}</strong><br>${escapeHtml(d.description?.slice(0, 100) || '')}`;
                case 'qc_synthesis_started':
                    return `<strong>QC Synthesis Started</strong><br>Total concerns: ${d.total_concerns || 0}`;
                case 'qc_report_complete':
                    return `<strong>QC Report Complete</strong><br>Report: ${d.report_id || 'unknown'}<br>Risk: <span class="severity-${d.risk_level || 'low'}">${(d.risk_level || 'low').toUpperCase()}</span> | Recommendation: ${d.recommendation || 'unknown'}`;
                default:
                    return JSON.stringify(d).slice(0, 100);
            }
        }

        async function sendQuery() {
            const input = document.getElementById('query-input');
            const select = document.getElementById('agent-select');
            const responsePanel = document.getElementById('response-panel');
            const btn = document.getElementById('send-btn');

            const query = input.value.trim();
            if (!query) return;

            btn.disabled = true;
            btn.textContent = 'Sending...';
            responsePanel.innerHTML = '<div class="text-gray-400">Processing...</div>';

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        agent_id: select.value || null,
                    }),
                });
                const data = await res.json();

                if (data.error) {
                    responsePanel.innerHTML = `<div class="text-red-400">Error: ${data.error}</div>`;
                } else {
                    const agent = agents.find(a => a.id === data.agent_id) || {};
                    const tierBg = agent.tier === 'A' ? 'bg-blue-900' : agent.tier === 'C' ? 'bg-purple-900' : 'bg-green-900';
                    responsePanel.innerHTML = `
                        <div class="mb-2 flex items-center gap-2">
                            <span class="px-2 py-1 rounded text-xs ${tierBg}">
                                ${data.agent_id}
                            </span>
                            <span class="text-xs text-gray-400">${data.tokens_used} tokens</span>
                        </div>
                        <div class="whitespace-pre-wrap text-sm">${data.content}</div>
                    `;
                }
            } catch (e) {
                responsePanel.innerHTML = `<div class="text-red-400">Error: ${e.message}</div>`;
            }

            btn.disabled = false;
            btn.textContent = 'Send';
        }
    </script>
</body>
</html>
"""
