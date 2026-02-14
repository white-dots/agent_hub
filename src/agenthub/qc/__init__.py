from __future__ import annotations
"""QC (Quality Control) module for code change analysis.

This module provides:
- Tier B agent analysis capabilities (raise concerns)
- Tier C QC Agent for concern synthesis
- Pipeline for orchestrating change analysis

Example:
    >>> from agenthub import AgentHub
    >>> from agenthub.qc import ChangeAnalysisPipeline
    >>>
    >>> hub = AgentHub(client)
    >>> # ... register agents ...
    >>>
    >>> # Enable QC analysis
    >>> hub.enable_qc_analysis(
    ...     auto_analyze=True,
    ...     on_report=lambda r: print(f"QC: {r.recommendation}")
    ... )
    >>>
    >>> # Or manually analyze changes
    >>> report = hub.analyze_changes(["src/api/routes.py"])
    >>> print(f"Risk: {report.risk_level}")
    >>> for concern in report.all_concerns:
    ...     print(f"  [{concern.severity}] {concern.title}")
"""

from agenthub.qc.models import (
    ActionItem,
    AgentAnalysisResult,
    ChangeSet,
    Concern,
    ConcernCategory,
    ConcernReport,
    ConcernSeverity,
    FileChange,
)
from agenthub.qc.pipeline import ChangeAnalysisPipeline
from agenthub.qc.qc_agent import QCAgent

__all__ = [
    # Models
    "FileChange",
    "ChangeSet",
    "Concern",
    "ConcernSeverity",
    "ConcernCategory",
    "AgentAnalysisResult",
    "ActionItem",
    "ConcernReport",
    # Pipeline
    "ChangeAnalysisPipeline",
    # QC Agent
    "QCAgent",
]
