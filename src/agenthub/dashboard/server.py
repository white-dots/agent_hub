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
            agent_data = {
                "id": spec.agent_id,
                "name": spec.name,
                "description": spec.description,
                "tier": "B" if spec.metadata.get("auto_generated") else "A",
                "module_type": spec.metadata.get("module_type", "custom"),
                "keywords": spec.context_keywords[:10],
            }
            # Include R&R if available (Tier B agents)
            if "rnr" in spec.metadata:
                agent_data["rnr"] = spec.metadata["rnr"]
            agents.append(agent_data)
        return {"agents": agents}

    @app.get("/api/agents/{agent_id}/rnr")
    async def get_agent_rnr(agent_id: str):
        """Get the Roles & Responsibilities for a specific agent."""
        agent = _hub.get_agent(agent_id)
        if not agent:
            return {"error": f"Agent '{agent_id}' not found"}

        spec = agent.spec
        rnr = spec.metadata.get("rnr")

        if not rnr:
            return {
                "agent_id": agent_id,
                "tier": "A" if not spec.metadata.get("auto_generated") else "B",
                "message": "This is a Tier A agent. R&R is defined by the agent creator.",
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

    @app.get("/api/routing-rules")
    async def get_routing_rules():
        """Get routing rules showing which keywords map to which agents.

        This makes the automatic routing logic visible and transparent.
        """
        rules = []
        for spec in _hub.list_agents():
            agent_info = {
                "agent_id": spec.agent_id,
                "agent_name": spec.name,
                "tier": "B" if spec.metadata.get("auto_generated") else "A",
                "keywords": spec.context_keywords,
                "description": spec.description,
                "priority": 1 if not spec.metadata.get("auto_generated") else 2,  # Tier A first
            }
            rules.append(agent_info)

        # Sort by priority (Tier A first), then by number of keywords
        rules.sort(key=lambda r: (r["priority"], -len(r["keywords"])))

        return {
            "rules": rules,
            "routing_strategy": "keyword_match",
            "explanation": "Queries are matched against agent keywords. Tier A agents are checked first, then Tier B. First match wins.",
        }

    @app.get("/api/tree")
    async def get_agent_tree():
        """Get the agent tree visualization.

        Returns the ASCII tree showing agent hierarchy and coverage.
        """
        from agenthub.auto.tree import build_agent_tree, print_agent_tree

        # Try to get project name from first agent's context paths
        project_name = "Project"
        for spec in _hub.list_agents():
            if spec.context_paths:
                from pathlib import Path
                # Get the root directory from context paths
                first_path = spec.context_paths[0]
                parts = Path(first_path).parts
                if parts:
                    project_name = parts[0]
                    break

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

        return {
            "ascii": tree_text,
            "tree": node_to_dict(tree_node),
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
        in the dashboard's Live Events panel.

        Expected data:
            event_type: One of claude_code_task, claude_code_search,
                       claude_code_read, claude_code_edit, claude_code_result
            session_id: Optional session identifier
            description: Human-readable description
            details: Optional dict with additional data
        """
        from datetime import datetime
        from uuid import uuid4
        from agenthub.dashboard.observer import ConversationEvent, EventType

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
        }
        event_type = event_map.get(event_type_str, EventType.CLAUDE_CODE_TASK)

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
                    <button onclick="loadTree()" class="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded">
                        🔄 Refresh
                    </button>
                </div>
                <div id="tree-panel" class="mt-4">
                    <pre id="tree-ascii" class="text-sm font-mono bg-gray-900 p-4 rounded overflow-x-auto text-gray-300">Loading...</pre>
                </div>
            </div>
        </div>

        <!-- QC Analysis Panel (Collapsible) -->
        <div class="mb-6">
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2 cursor-pointer" onclick="toggleQCPanel()">
                        <h2 class="text-lg font-semibold">🔍 QC Analysis</h2>
                        <span id="qc-toggle" class="text-gray-400">▼</span>
                        <span id="qc-status-badge" class="text-xs px-2 py-0.5 rounded bg-gray-700 text-gray-400">checking...</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <button id="qc-enable-btn" onclick="toggleQCEnabled()" class="text-xs bg-blue-600 hover:bg-blue-700 px-2 py-1 rounded">
                            Enable
                        </button>
                        <button onclick="loadQCReports()" class="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded">
                            🔄 Refresh
                        </button>
                    </div>
                </div>
                <div id="qc-panel" class="mt-4">
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

                    <!-- Recent Concerns -->
                    <div class="bg-gray-900 rounded p-3">
                        <h3 class="text-sm font-semibold mb-2 text-gray-300">Recent Concerns</h3>
                        <div id="qc-concerns-list" class="space-y-2 max-h-[300px] overflow-y-auto">
                            <div class="text-gray-500 text-sm">No concerns yet</div>
                        </div>
                    </div>

                    <!-- Recent Reports -->
                    <div class="bg-gray-900 rounded p-3 mt-4">
                        <h3 class="text-sm font-semibold mb-2 text-gray-300">Recent Reports</h3>
                        <div id="qc-reports-list" class="space-y-2 max-h-[200px] overflow-y-auto">
                            <div class="text-gray-500 text-sm">No reports yet</div>
                        </div>
                    </div>
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

        async function loadTree() {
            try {
                const res = await fetch('/api/tree');
                const data = await res.json();
                const treeAscii = document.getElementById('tree-ascii');

                // Add color highlighting to the tree
                let html = data.ascii
                    // Project icon
                    .replace(/\\[P\\]/g, '<span class="text-yellow-400">[P]</span>')
                    // Tier A icon
                    .replace(/\\[A\\]/g, '<span class="text-blue-400">[A]</span>')
                    // Tier B icon
                    .replace(/\\[B\\]/g, '<span class="text-green-400">[B]</span>')
                    // Agent icon
                    .replace(/\\* /g, '<span class="text-purple-400">◆ </span>')
                    // Keywords icon
                    .replace(/# /g, '<span class="text-gray-500"># </span>')
                    // Module type icons
                    .replace(/\\(api\\)/g, '<span class="text-orange-400">(api)</span>')
                    .replace(/\\(svc\\)/g, '<span class="text-pink-400">(svc)</span>')
                    .replace(/\\(mod\\)/g, '<span class="text-indigo-400">(mod)</span>')
                    .replace(/\\(repo\\)/g, '<span class="text-teal-400">(repo)</span>')
                    .replace(/\\(util\\)/g, '<span class="text-amber-400">(util)</span>')
                    .replace(/\\(cfg\\)/g, '<span class="text-rose-400">(cfg)</span>')
                    .replace(/\\(test\\)/g, '<span class="text-lime-400">(test)</span>')
                    .replace(/\\(code\\)/g, '<span class="text-slate-400">(code)</span>')
                    // Description separator
                    .replace(/ --- /g, ' <span class="text-gray-500">―</span> ');

                treeAscii.innerHTML = html;
            } catch (e) {
                console.error('Failed to load tree:', e);
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
                const concernsRes = await fetch('/api/qc/concerns?limit=20');
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

                // Update concerns list
                const concernsList = document.getElementById('qc-concerns-list');
                if (concernsData.concerns && concernsData.concerns.length > 0) {
                    concernsList.innerHTML = concernsData.concerns.map(c => `
                        <div class="p-2 rounded bg-gray-800 text-sm">
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
                    concernsList.innerHTML = '<div class="text-gray-500 text-sm">No concerns yet</div>';
                }

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

        async function showReportDetail(reportId) {
            try {
                const res = await fetch(`/api/qc/reports/${reportId}`);
                const data = await res.json();
                if (data.error) {
                    alert('Error loading report: ' + data.error);
                    return;
                }
                // For now, show in alert. Could open a modal instead.
                alert(`Report: ${data.report_id}\\n\\nAssessment: ${data.overall_assessment}\\n\\nRisk: ${data.risk_level}\\nRecommendation: ${data.recommendation}\\n\\nConcerns: ${data.total_concerns}\\nTokens: ${data.total_tokens_used}`);
            } catch (e) {
                console.error('Failed to load report:', e);
            }
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
                // Refresh QC panel on QC events
                if (data.event_type && data.event_type.startsWith('analysis_') ||
                    data.event_type === 'concern_raised' ||
                    data.event_type === 'qc_synthesis_started' ||
                    data.event_type === 'qc_report_complete') {
                    loadQCReports();
                }
            };
        }

        async function loadAgents() {
            const res = await fetch('/api/agents');
            const data = await res.json();
            agents = data.agents;

            const list = document.getElementById('agents-list');
            const select = document.getElementById('agent-select');

            list.innerHTML = agents.map(a => `
                <div class="p-2 rounded ${a.tier === 'A' ? 'tier-a' : 'tier-b'} bg-gray-700">
                    <div class="font-medium">${a.name}</div>
                    <div class="text-xs text-gray-400">${a.id}</div>
                    <div class="text-xs mt-1">
                        <span class="px-1 py-0.5 rounded ${a.tier === 'A' ? 'bg-blue-900' : 'bg-green-900'}">
                            Tier ${a.tier}
                        </span>
                        <span class="text-gray-500 ml-1">${a.module_type}</span>
                    </div>
                </div>
            `).join('');

            select.innerHTML = '<option value="">Auto-route</option>' +
                agents.map(a => `<option value="${a.id}">${a.name} (${a.tier})</option>`).join('');
        }

        async function loadTraces() {
            const res = await fetch('/api/traces?limit=10');
            const data = await res.json();

            const tbody = document.getElementById('traces-body');
            tbody.innerHTML = data.traces.map(t => {
                const agent = agents.find(a => a.id === t.final_agent) || {};
                return `
                    <tr class="hover:bg-gray-700">
                        <td class="py-2">${new Date(t.started_at).toLocaleTimeString()}</td>
                        <td class="py-2 max-w-xs truncate">${t.query}</td>
                        <td class="py-2">${t.final_agent || '-'}</td>
                        <td class="py-2">
                            <span class="px-1.5 py-0.5 rounded text-xs ${agent.tier === 'A' ? 'bg-blue-900' : 'bg-green-900'}">
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

            html += data.rules.map(rule => `
                <div class="p-2 rounded bg-gray-700 ${rule.tier === 'A' ? 'tier-a' : 'tier-b'}">
                    <div class="flex items-center gap-2">
                        <span class="font-medium text-sm">${rule.agent_name}</span>
                        <span class="px-1 py-0.5 rounded text-xs ${rule.tier === 'A' ? 'bg-blue-900' : 'bg-green-900'}">
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
            `).join('');

            panel.innerHTML = html;
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
                    return `<div class="mb-1"><strong>Query:</strong></div>
                        <div class="bg-gray-800 p-2 rounded text-white whitespace-pre-wrap max-h-40 overflow-y-auto">${escapeHtml(queryText)}</div>`;
                case 'claude_code_search':
                    return `<strong>Search:</strong> ${d.description}${d.pattern ? ` (${d.pattern})` : ''}`;
                case 'claude_code_read':
                    return `<strong>Read:</strong> ${d.description}${d.file ? ` - ${d.file}` : ''}`;
                case 'claude_code_edit':
                    return `<strong>Edit:</strong> ${d.description}${d.file ? ` - ${d.file}` : ''}`;
                case 'claude_code_result':
                    const resp = d.response || d.description || '';
                    const query = d.query || '';
                    return `<div class="mb-1"><strong>Agent:</strong> ${d.agent_id || 'unknown'} | <strong>Tokens:</strong> ${d.tokens_used || 0}</div>
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
                    responsePanel.innerHTML = `
                        <div class="mb-2 flex items-center gap-2">
                            <span class="px-2 py-1 rounded text-xs ${agent.tier === 'A' ? 'bg-blue-900' : 'bg-green-900'}">
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
