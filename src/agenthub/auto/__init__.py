"""Auto-analysis for existing codebases.

Provides static import graph analysis for impact checking.
"""

from agenthub.auto.import_graph import ImportGraph, ImportEdge, ModuleNode

__all__ = [
    "ImportGraph",
    "ImportEdge",
    "ModuleNode",
]
