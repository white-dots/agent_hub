from __future__ import annotations
"""Conflict risk analysis for parallel sessions.

The ConflictRiskAnalyzer determines whether tasks can safely run in parallel
by analyzing file overlaps, import dependencies, and consulting domain agents.

Key principle: Agents can only UPGRADE risk, never downgrade from static analysis.
"""

import json
import re
from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING, Optional

from agenthub.parallel.models import (
    AgentConflictAssessment,
    FileOverlap,
    ImplementationTask,
    OverlapType,
    ParallelizationPlan,
    RiskLevel,
)

if TYPE_CHECKING:
    from agenthub.agents.base import BaseAgent
    from agenthub.auto.import_graph import ImportGraph
    from agenthub.auto.sub_agent_manager import SubAgentManager
    from agenthub.hub import AgentHub


# Prompt for agent conflict assessment
AGENT_CONFLICT_PROMPT = """You are analyzing potential conflicts between two parallel tasks.

Your domain: {agent_name}
Your files: {agent_paths}

Task A ({task_a_id}): {task_a_description}
Expected files: {task_a_files}

Task B ({task_b_id}): {task_b_description}
Expected files: {task_b_files}

Static analysis found: {static_findings}

As a domain expert, do you see any conflicts that static analysis might miss?
Consider:
- Database operations that could conflict
- Shared state or caches
- API contracts that both tasks might change
- Business logic dependencies
- Race conditions if run in parallel

Respond in JSON:
```json
{{
    "has_concern": true/false,
    "concern_description": "Detailed explanation if concerned",
    "severity": "none|low|medium|high",
    "affected_files": ["files where conflict might occur"]
}}
```

IMPORTANT: Only raise concerns for issues that static analysis CANNOT detect.
If the only concern is file overlap, that's already handled - focus on semantic conflicts.
"""


class ConflictRiskAnalyzer:
    """Analyzes conflict risk between tasks.

    Four-phase analysis:
    A. Static file overlap (fast, free)
    B. Import dependency overlap (uses ImportGraph)
    C. Tier B agent consultation (semantic)
    D. Tier A business review (business impact)

    Agents can only UPGRADE risk, never downgrade from static analysis.
    This is intentional: false negatives (missing a conflict) are far worse
    than false positives (unnecessary sequential execution).

    Example:
        >>> analyzer = ConflictRiskAnalyzer(import_graph, hub)
        >>> plan = analyzer.analyze(tasks, consult_agents=True)
        >>> print(f"Risk level: {plan.overall_risk}")
        >>> print(f"Recommendation: {plan.pm_recommendation}")
    """

    # Hub modules with high in-degree are risky to modify in parallel
    HUB_THRESHOLD = 5

    # Depth for transitive import analysis
    IMPORT_DEPTH = 2

    def __init__(
        self,
        import_graph: "ImportGraph",
        hub: "AgentHub",
        sub_agent_manager: Optional["SubAgentManager"] = None,
    ):
        """Initialize ConflictRiskAnalyzer.

        Args:
            import_graph: ImportGraph for dependency analysis.
            hub: AgentHub for agent consultation.
            sub_agent_manager: Optional SubAgentManager for precise agent routing.
        """
        self._graph = import_graph
        self._hub = hub
        self._sub_manager = sub_agent_manager

    def analyze(
        self,
        tasks: list[ImplementationTask],
        consult_agents: bool = True,
    ) -> ParallelizationPlan:
        """Perform full conflict analysis pipeline.

        Args:
            tasks: List of tasks to analyze.
            consult_agents: Whether to consult domain agents (slower but more thorough).

        Returns:
            ParallelizationPlan with risk assessment and groupings.
        """
        if len(tasks) <= 1:
            # Single task - no conflict possible
            return ParallelizationPlan(
                parallel_groups=[[tasks[0].task_id]] if tasks else [],
                sequential_order=[tasks[0].task_id] if tasks else [],
                overall_risk=RiskLevel.NONE,
                confidence=1.0,
                reasoning="Single task - no parallelization needed.",
                pm_recommendation="parallel",
            )

        # Phase A: Static file overlap
        direct_overlaps = self._check_direct_overlap(tasks)

        # Phase B: Import dependency analysis
        import_overlaps = self._check_import_overlap(tasks)
        hub_overlaps = self._check_hub_overlap(tasks)
        model_overlaps = self._check_model_overlap(tasks)

        all_static_overlaps = direct_overlaps + import_overlaps + hub_overlaps + model_overlaps

        # Calculate static risk
        static_risk = self._calculate_static_risk(all_static_overlaps)

        # Phase C & D: Agent consultation (if enabled)
        tier_b_assessments: list[AgentConflictAssessment] = []
        tier_a_assessments: list[AgentConflictAssessment] = []

        if consult_agents:
            tier_b_assessments = self._consult_tier_b_agents(tasks, all_static_overlaps)
            tier_a_assessments = self._consult_tier_a_agents(
                tasks, static_risk, tier_b_assessments
            )

        # Build the plan
        return self._build_plan(
            tasks,
            all_static_overlaps,
            tier_b_assessments,
            tier_a_assessments,
        )

    # =========================================================================
    # Phase A: Static File Overlap
    # =========================================================================

    def _check_direct_overlap(
        self, tasks: list[ImplementationTask]
    ) -> list[FileOverlap]:
        """Check if any two tasks modify the same file.

        Direct overlap is HIGH risk - the same file cannot be safely
        modified by two parallel sessions.
        """
        overlaps: list[FileOverlap] = []

        # Build file -> tasks mapping
        file_to_tasks: dict[str, list[str]] = defaultdict(list)

        for task in tasks:
            for file_path in task.estimated_files:
                file_to_tasks[file_path].append(task.task_id)
            for file_path in task.estimated_new_files:
                file_to_tasks[file_path].append(task.task_id)

        # Find overlaps
        for file_path, task_ids in file_to_tasks.items():
            if len(task_ids) > 1:
                overlaps.append(
                    FileOverlap(
                        file_path=file_path,
                        tasks_touching=task_ids,
                        overlap_type=OverlapType.DIRECT,
                        risk_level=RiskLevel.HIGH,
                    )
                )

        return overlaps

    # =========================================================================
    # Phase B: Import Dependency Analysis
    # =========================================================================

    def _check_import_overlap(
        self, tasks: list[ImplementationTask]
    ) -> list[FileOverlap]:
        """Check for transitive import overlaps (depth=2).

        If Task A modifies file X, and Task B modifies file Y,
        and X imports Y (directly or transitively), there's a conflict risk.
        """
        overlaps: list[FileOverlap] = []

        if not self._graph or not self._graph._built:
            return overlaps

        for task_a, task_b in combinations(tasks, 2):
            # Get all imports for each task's files
            imports_a = self._get_transitive_imports(
                task_a.estimated_files, self.IMPORT_DEPTH
            )
            imports_b = self._get_transitive_imports(
                task_b.estimated_files, self.IMPORT_DEPTH
            )

            # Check for import overlap
            shared_imports = imports_a & imports_b

            # Also check if one task's files import the other's
            files_a = set(task_a.estimated_files)
            files_b = set(task_b.estimated_files)

            # A imports B
            a_imports_b = imports_a & files_b
            # B imports A
            b_imports_a = imports_b & files_a

            conflict_files = shared_imports | a_imports_b | b_imports_a

            for file_path in conflict_files:
                overlaps.append(
                    FileOverlap(
                        file_path=file_path,
                        tasks_touching=[task_a.task_id, task_b.task_id],
                        overlap_type=OverlapType.SHARED_IMPORT,
                        risk_level=RiskLevel.MEDIUM,
                    )
                )

        return overlaps

    def _get_transitive_imports(
        self, files: list[str], depth: int
    ) -> set[str]:
        """Get all imports from a set of files up to a given depth."""
        all_imports: set[str] = set()
        current_level: set[str] = set(files)

        for _ in range(depth):
            next_level: set[str] = set()
            for file_path in current_level:
                if file_path in self._graph.nodes:
                    node = self._graph.nodes[file_path]
                    # Get files this imports
                    for imported in node.imports:
                        if imported in self._graph.nodes:
                            next_level.add(imported)
                    # Get files that import this
                    for importer in node.imported_by:
                        next_level.add(importer)

            all_imports.update(next_level)
            current_level = next_level

        return all_imports

    def _check_hub_overlap(
        self, tasks: list[ImplementationTask]
    ) -> list[FileOverlap]:
        """Check if tasks share hub modules (high in-degree).

        Hub modules are heavily imported - modifying them in parallel
        is risky because many other modules depend on them.
        """
        overlaps: list[FileOverlap] = []

        if not self._graph or not self._graph._built:
            return overlaps

        # Find hub modules
        hub_modules: set[str] = set()
        for node_path, node in self._graph.nodes.items():
            if len(node.imported_by) >= self.HUB_THRESHOLD:
                hub_modules.add(node_path)

        # Check if multiple tasks touch hub modules
        for hub_module in hub_modules:
            touching_tasks: list[str] = []

            for task in tasks:
                task_files = set(task.estimated_files + task.estimated_new_files)
                # Direct touch
                if hub_module in task_files:
                    touching_tasks.append(task.task_id)
                    continue

                # Touch via import
                task_imports = self._get_transitive_imports(task.estimated_files, 1)
                if hub_module in task_imports:
                    touching_tasks.append(task.task_id)

            if len(touching_tasks) > 1:
                overlaps.append(
                    FileOverlap(
                        file_path=hub_module,
                        tasks_touching=touching_tasks,
                        overlap_type=OverlapType.SHARED_TYPE,
                        risk_level=RiskLevel.MEDIUM,
                    )
                )

        return overlaps

    def _check_model_overlap(
        self, tasks: list[ImplementationTask]
    ) -> list[FileOverlap]:
        """Check for shared database models, types, schemas.

        Files with 'model', 'schema', 'types' in the path are often
        shared definitions that are risky to modify in parallel.
        """
        overlaps: list[FileOverlap] = []

        # Keywords that suggest shared definitions
        shared_keywords = ["model", "schema", "types", "interface", "dto", "entity"]

        # Collect all files from all tasks
        task_files: dict[str, list[str]] = {}
        for task in tasks:
            task_files[task.task_id] = task.estimated_files + task.estimated_new_files

        # Find model/schema files touched by multiple tasks
        for file_path in set().union(*[set(f) for f in task_files.values()]):
            file_lower = file_path.lower()
            is_shared = any(kw in file_lower for kw in shared_keywords)

            if is_shared:
                touching_tasks = [
                    task_id
                    for task_id, files in task_files.items()
                    if file_path in files
                ]

                if len(touching_tasks) > 1:
                    overlaps.append(
                        FileOverlap(
                            file_path=file_path,
                            tasks_touching=touching_tasks,
                            overlap_type=OverlapType.SHARED_CONFIG,
                            risk_level=RiskLevel.MEDIUM,
                        )
                    )

        return overlaps

    def _calculate_static_risk(self, overlaps: list[FileOverlap]) -> RiskLevel:
        """Calculate overall risk from static analysis."""
        if not overlaps:
            return RiskLevel.NONE

        # Get max risk from overlaps
        risk_order = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
        max_risk = RiskLevel.NONE

        for overlap in overlaps:
            if risk_order.index(overlap.risk_level) > risk_order.index(max_risk):
                max_risk = overlap.risk_level

        return max_risk

    # =========================================================================
    # Phase C: Tier B Agent Consultation
    # =========================================================================

    def _consult_tier_b_agents(
        self,
        tasks: list[ImplementationTask],
        static_overlaps: list[FileOverlap],
    ) -> list[AgentConflictAssessment]:
        """Ask domain agents about conflicts.

        Routes to sub-agents when available for more precise assessment.
        Uses Haiku for speed.

        Agents can only UPGRADE risk, never downgrade static findings.
        """
        assessments: list[AgentConflictAssessment] = []

        # Get all Tier B agents
        tier_b_agents = self._hub.list_agents(tier="B")

        # For each pair of tasks, consult relevant agents
        for task_a, task_b in combinations(tasks, 2):
            # Find agents that might be relevant to these tasks
            relevant_agents = self._find_relevant_agents(
                task_a, task_b, tier_b_agents
            )

            for agent_spec in relevant_agents:
                try:
                    assessment = self._assess_conflict_with_agent(
                        task_a, task_b, agent_spec, static_overlaps
                    )
                    assessments.append(assessment)
                except Exception as e:
                    print(f"Warning: Agent consultation failed for {agent_spec.agent_id}: {e}")

        return assessments

    def _find_relevant_agents(
        self, task_a: ImplementationTask, task_b: ImplementationTask, agents
    ) -> list:
        """Find agents relevant to a pair of tasks."""
        relevant = []

        all_files = set(
            task_a.estimated_files
            + task_a.estimated_new_files
            + task_b.estimated_files
            + task_b.estimated_new_files
        )

        for agent_spec in agents:
            # Check if agent's domain overlaps with task files
            for ctx_path in agent_spec.context_paths:
                for file_path in all_files:
                    if self._path_matches(file_path, ctx_path):
                        relevant.append(agent_spec)
                        break
                else:
                    continue
                break

        return relevant

    def _path_matches(self, file_path: str, pattern: str) -> bool:
        """Check if a file path matches a context pattern."""
        import fnmatch

        file_path = file_path.replace("\\", "/")
        pattern = pattern.replace("\\", "/")

        if fnmatch.fnmatch(file_path, pattern):
            return True
        if pattern.endswith("/**") or pattern.endswith("/*"):
            dir_pattern = pattern.rstrip("/*")
            if file_path.startswith(dir_pattern):
                return True
        if file_path.startswith(pattern.rstrip("/")):
            return True

        return False

    def _assess_conflict_with_agent(
        self,
        task_a: ImplementationTask,
        task_b: ImplementationTask,
        agent_spec,
        static_overlaps: list[FileOverlap],
    ) -> AgentConflictAssessment:
        """Ask a specific agent to assess conflict between two tasks."""
        # Format static findings for context
        relevant_overlaps = [
            o for o in static_overlaps
            if task_a.task_id in o.tasks_touching and task_b.task_id in o.tasks_touching
        ]

        static_findings = "No static overlaps found."
        if relevant_overlaps:
            static_findings = "\n".join(
                f"- {o.file_path}: {o.overlap_type.value} overlap"
                for o in relevant_overlaps
            )

        prompt = AGENT_CONFLICT_PROMPT.format(
            agent_name=agent_spec.name,
            agent_paths=", ".join(agent_spec.context_paths[:5]),
            task_a_id=task_a.task_id,
            task_a_description=task_a.description,
            task_a_files=", ".join(task_a.estimated_files[:5]),
            task_b_id=task_b.task_id,
            task_b_description=task_b.description,
            task_b_files=", ".join(task_b.estimated_files[:5]),
            static_findings=static_findings,
        )

        response = self._hub.client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        response_text = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        data = self._parse_json_response(response_text)

        severity_map = {
            "none": RiskLevel.NONE,
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
        }

        return AgentConflictAssessment(
            agent_id=agent_spec.agent_id,
            agent_name=agent_spec.name,
            task_pair=(task_a.task_id, task_b.task_id),
            has_concern=data.get("has_concern", False),
            concern_description=data.get("concern_description", ""),
            severity=severity_map.get(data.get("severity", "none"), RiskLevel.NONE),
            affected_files=data.get("affected_files", []),
            tokens_used=tokens_used,
        )

    # =========================================================================
    # Phase D: Tier A Business Review
    # =========================================================================

    def _consult_tier_a_agents(
        self,
        tasks: list[ImplementationTask],
        static_risk: RiskLevel,
        tier_b_assessments: list[AgentConflictAssessment],
    ) -> list[AgentConflictAssessment]:
        """Ask Tier A agents for business impact assessment.

        Tier A agents can upgrade risk even if static analysis shows NONE.
        They understand business-level dependencies that code analysis misses.
        """
        assessments: list[AgentConflictAssessment] = []

        # Get Tier A agents
        tier_a_agents = self._hub.list_agents(tier="A")

        if not tier_a_agents:
            return assessments

        # Only consult Tier A if there are potential issues
        # or if the request spans multiple business domains
        should_consult = (
            static_risk != RiskLevel.NONE
            or any(a.has_concern for a in tier_b_assessments)
            or len(tasks) >= 3
        )

        if not should_consult:
            return assessments

        # Consult each Tier A agent
        for agent_spec in tier_a_agents:
            for task_a, task_b in combinations(tasks, 2):
                try:
                    assessment = self._assess_business_conflict(
                        task_a, task_b, agent_spec, static_risk, tier_b_assessments
                    )
                    if assessment.has_concern:
                        assessments.append(assessment)
                except Exception as e:
                    print(f"Warning: Tier A consultation failed for {agent_spec.agent_id}: {e}")

        return assessments

    def _assess_business_conflict(
        self,
        task_a: ImplementationTask,
        task_b: ImplementationTask,
        agent_spec,
        static_risk: RiskLevel,
        tier_b_assessments: list[AgentConflictAssessment],
    ) -> AgentConflictAssessment:
        """Ask Tier A agent about business-level conflicts."""
        # Summarize Tier B concerns
        relevant_b_concerns = [
            a for a in tier_b_assessments
            if task_a.task_id in a.task_pair and task_b.task_id in a.task_pair
        ]

        tier_b_summary = "No technical concerns raised."
        if relevant_b_concerns:
            tier_b_summary = "\n".join(
                f"- {a.agent_name}: {a.concern_description}"
                for a in relevant_b_concerns
                if a.has_concern
            )

        prompt = f"""You are a business analyst reviewing two parallel tasks for conflicts.

Task A: {task_a.description}
Task B: {task_b.description}

Static analysis risk: {static_risk.value}
Technical concerns: {tier_b_summary}

From a business perspective:
1. Do these tasks affect the same business workflow?
2. Could running them in parallel cause data consistency issues?
3. Are there customer-facing implications?
4. Would a partial failure (one succeeds, one fails) cause problems?

Respond in JSON:
```json
{{
    "has_concern": true/false,
    "concern_description": "Business-level concern if any",
    "severity": "none|low|medium|high"
}}
```"""

        response = self._hub.client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        data = self._parse_json_response(response.content[0].text)
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        severity_map = {
            "none": RiskLevel.NONE,
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
        }

        return AgentConflictAssessment(
            agent_id=agent_spec.agent_id,
            agent_name=agent_spec.name,
            task_pair=(task_a.task_id, task_b.task_id),
            has_concern=data.get("has_concern", False),
            concern_description=data.get("concern_description", ""),
            severity=severity_map.get(data.get("severity", "none"), RiskLevel.NONE),
            affected_files=[],
            tokens_used=tokens_used,
        )

    # =========================================================================
    # Phase E: Plan Building
    # =========================================================================

    def _build_plan(
        self,
        tasks: list[ImplementationTask],
        overlaps: list[FileOverlap],
        tier_b: list[AgentConflictAssessment],
        tier_a: list[AgentConflictAssessment],
    ) -> ParallelizationPlan:
        """Build the parallelization plan.

        Risk aggregation: Take max(static_risk, tier_b_risk, tier_a_risk)
        Agents can only UPGRADE risk, never downgrade.
        """
        # Calculate overall risk
        static_risk = self._calculate_static_risk(overlaps)
        agent_risk = RiskLevel.NONE

        all_assessments = tier_b + tier_a
        for assessment in all_assessments:
            if assessment.has_concern:
                risk_order = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
                if risk_order.index(assessment.severity) > risk_order.index(agent_risk):
                    agent_risk = assessment.severity

        # Overall risk is max of static and agent assessments
        risk_order = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
        overall_risk = max(static_risk, agent_risk, key=lambda r: risk_order.index(r))

        # Build parallel groups and sequential order
        parallel_groups, sequential_order = self._compute_groupings(
            tasks, overlaps, all_assessments
        )

        # Determine recommendation
        if overall_risk == RiskLevel.HIGH:
            pm_recommendation = "sequential"
            reasoning = self._build_sequential_reasoning(overlaps, all_assessments)
        else:
            pm_recommendation = "parallel"
            reasoning = self._build_parallel_reasoning(overlaps, all_assessments)

        # Calculate estimated speedup
        total_sequential = sum(t.estimated_tokens for t in tasks) / 15000 * 60  # ~60s per 15k tokens
        if pm_recommendation == "parallel" and parallel_groups:
            parallel_time = max(
                sum(
                    next(t.estimated_tokens for t in tasks if t.task_id == tid)
                    for tid in group
                )
                for group in parallel_groups
            ) / 15000 * 60
            speedup = total_sequential / parallel_time if parallel_time > 0 else 1.0
        else:
            speedup = 1.0

        # Calculate confidence
        confidence = self._calculate_confidence(overlaps, all_assessments)

        return ParallelizationPlan(
            parallel_groups=parallel_groups,
            sequential_order=sequential_order,
            file_overlaps=overlaps,
            agent_assessments=all_assessments,
            overall_risk=overall_risk,
            confidence=confidence,
            reasoning=reasoning,
            pm_recommendation=pm_recommendation,
            estimated_speedup=round(speedup, 2),
            estimated_total_tokens=sum(t.estimated_tokens for t in tasks),
        )

    def _compute_groupings(
        self,
        tasks: list[ImplementationTask],
        overlaps: list[FileOverlap],
        assessments: list[AgentConflictAssessment],
    ) -> tuple[list[list[str]], list[str]]:
        """Compute parallel groups and sequential order.

        Returns (parallel_groups, sequential_order).
        """
        task_ids = [t.task_id for t in tasks]

        # Build conflict graph
        conflicts: dict[str, set[str]] = {tid: set() for tid in task_ids}

        # Add conflicts from overlaps
        for overlap in overlaps:
            if overlap.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
                for i, tid1 in enumerate(overlap.tasks_touching):
                    for tid2 in overlap.tasks_touching[i + 1:]:
                        conflicts[tid1].add(tid2)
                        conflicts[tid2].add(tid1)

        # Add conflicts from agent assessments
        for assessment in assessments:
            if assessment.has_concern and assessment.severity in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
                tid1, tid2 = assessment.task_pair
                conflicts[tid1].add(tid2)
                conflicts[tid2].add(tid1)

        # Add explicit dependencies
        for task in tasks:
            for dep_id in task.depends_on:
                conflicts[task.task_id].add(dep_id)
                conflicts[dep_id].add(task.task_id)

        # Compute parallel groups using greedy coloring
        parallel_groups = self._greedy_parallel_groups(task_ids, conflicts)

        # Compute sequential order respecting dependencies
        sequential_order = self._topological_sort(tasks)

        return parallel_groups, sequential_order

    def _greedy_parallel_groups(
        self, task_ids: list[str], conflicts: dict[str, set[str]]
    ) -> list[list[str]]:
        """Greedily assign tasks to parallel groups (graph coloring)."""
        groups: list[list[str]] = []
        assigned: set[str] = set()

        for tid in task_ids:
            if tid in assigned:
                continue

            # Find a group where this task has no conflicts
            placed = False
            for group in groups:
                if not any(other in conflicts[tid] for other in group):
                    group.append(tid)
                    assigned.add(tid)
                    placed = True
                    break

            if not placed:
                groups.append([tid])
                assigned.add(tid)

        return groups

    def _topological_sort(self, tasks: list[ImplementationTask]) -> list[str]:
        """Sort tasks respecting dependencies (for sequential order)."""
        # Build dependency graph
        deps: dict[str, set[str]] = {t.task_id: set(t.depends_on) for t in tasks}
        result: list[str] = []
        remaining = set(t.task_id for t in tasks)

        while remaining:
            # Find tasks with no remaining dependencies
            ready = [tid for tid in remaining if not (deps[tid] & remaining)]

            if not ready:
                # Cycle detected - just add remaining in any order
                result.extend(remaining)
                break

            # Add ready tasks in original order
            for tid in [t.task_id for t in tasks if t.task_id in ready]:
                result.append(tid)
                remaining.remove(tid)

        return result

    def _build_sequential_reasoning(
        self, overlaps: list[FileOverlap], assessments: list[AgentConflictAssessment]
    ) -> str:
        """Build reasoning for sequential recommendation."""
        parts = ["Risk is HIGH. Recommending sequential execution."]

        high_overlaps = [o for o in overlaps if o.risk_level == RiskLevel.HIGH]
        if high_overlaps:
            files = list(set(o.file_path for o in high_overlaps))[:3]
            parts.append(f"Direct file conflicts: {', '.join(files)}")

        high_concerns = [a for a in assessments if a.has_concern and a.severity == RiskLevel.HIGH]
        if high_concerns:
            parts.append("Agent concerns:")
            for concern in high_concerns[:3]:
                parts.append(f"  - {concern.agent_name}: {concern.concern_description[:100]}")

        return "\n".join(parts)

    def _build_parallel_reasoning(
        self, overlaps: list[FileOverlap], assessments: list[AgentConflictAssessment]
    ) -> str:
        """Build reasoning for parallel recommendation."""
        if not overlaps and not any(a.has_concern for a in assessments):
            return "No conflicts detected. Safe to run in parallel."

        parts = ["Parallel execution recommended with monitoring."]

        if overlaps:
            parts.append(f"Found {len(overlaps)} potential overlaps (not blocking).")

        concerns = [a for a in assessments if a.has_concern]
        if concerns:
            parts.append(f"{len(concerns)} agent concerns raised (severity below HIGH).")

        return "\n".join(parts)

    def _calculate_confidence(
        self, overlaps: list[FileOverlap], assessments: list[AgentConflictAssessment]
    ) -> float:
        """Calculate PM's confidence in the plan.

        Higher confidence when:
        - Few overlaps
        - Agents agree
        - Clear dependency structure
        """
        confidence = 0.95

        # Reduce for overlaps
        if overlaps:
            confidence -= min(len(overlaps) * 0.05, 0.2)

        # Reduce for agent concerns
        concerns = sum(1 for a in assessments if a.has_concern)
        confidence -= min(concerns * 0.05, 0.2)

        # Reduce if agents disagree
        concern_severities = [a.severity for a in assessments if a.has_concern]
        if len(set(concern_severities)) > 1:
            confidence -= 0.1

        return max(confidence, 0.3)

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response."""
        json_match = re.search(r"```(?:json)?\n?(.*?)```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        return {}

    # =========================================================================
    # Helper: Get owning agent for a file
    # =========================================================================

    def _get_owning_agent(self, file_path: str) -> Optional["BaseAgent"]:
        """Get the most specific agent owning a file.

        Sub-agent > team lead > Tier B.
        """
        if self._sub_manager:
            agent = self._sub_manager.get_most_specific_agent(file_path)
            if agent:
                return agent

        # Fall back to checking all agents
        for agent_spec in self._hub.list_agents(tier="B"):
            for ctx_path in agent_spec.context_paths:
                if self._path_matches(file_path, ctx_path):
                    # Get actual agent instance
                    if hasattr(self._hub, "_auto_manager") and self._hub._auto_manager:
                        return self._hub._auto_manager._auto_agents.get(agent_spec.agent_id)

        return None
