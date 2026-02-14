from __future__ import annotations
"""Prompt templates for parallel sessions.

This module contains the prompt templates used by the BranchOrchestrator
to create scoped prompts for Claude Code sessions.
"""

# Main template for scoped sessions
SCOPED_SESSION_PROMPT = """You are working on a specific task within a larger codebase.
Other teams are working on other tasks simultaneously on separate branches.

TASK: {task_description}

YOUR TEAM'S FILES (you should modify these):
{your_files}

OTHER TEAMS' FILES (do NOT modify these):
{other_files}

RELEVANT TECHNICAL CONTEXT (from your Tier B team):
{tier_b_context}

BUSINESS CONTEXT (from Tier A agents):
{tier_a_context}

IMPORTANT CONSTRAINTS:
- Focus ONLY on the task described above
- Only modify files within YOUR TEAM'S FILES listed above
- Do NOT modify any file listed under OTHER TEAMS' FILES
- If you discover you NEED to modify a file outside your scope:
  1. STOP modifying that file
  2. Note it clearly with the tag: [BOUNDARY_CROSSING: {{file_path}} - {{reason}}]
  3. Continue implementing what you can without that modification
  4. The orchestrator will coordinate with the other team
- Run relevant tests when done (tests may fail if you noted boundary crossings)

Please implement this task and run tests to verify.
"""

# Template for summarizing technical context from domain agents
TIER_B_CONTEXT_TEMPLATE = """### {agent_name} Domain Knowledge

{agent_context}
"""

# Template for summarizing business context from Tier A agents
TIER_A_CONTEXT_TEMPLATE = """### Business Context: {agent_name}

{business_context}
"""

# Template for CEO confirmation prompt (MEDIUM/HIGH risk)
CEO_CONFIRMATION_PROMPT = """I've analyzed your request and broken it into {task_count} tasks.

## Task Breakdown
{task_list}

## Risk Assessment: {risk_level}

{risk_explanation}

## Agent Concerns
{agent_concerns}

## PM Recommendation: {recommendation}

{recommendation_details}

## Options
1. {option_1}
2. {option_2}
3. Modify task breakdown

What would you like to do?
"""

# Template for presenting parallel plan to CEO
PARALLEL_PLAN_TEMPLATE = """**Proceed with parallel execution** (recommended for {risk_level} risk)

Parallel groups:
{parallel_groups}

Estimated speedup: {speedup}x
Estimated total time: {estimated_time}

Monitoring will be enabled for boundary crossings.
"""

# Template for presenting sequential plan to CEO
SEQUENTIAL_PLAN_TEMPLATE = """**Run sequentially** (recommended for {risk_level} risk)

Execution order:
{sequential_order}

Why sequential:
{reasoning}

Estimated total time: {estimated_time}

You can override and run in parallel if you accept the conflict risk.
"""

# Template for boundary crossing message (in session output)
BOUNDARY_CROSSING_TAG = "[BOUNDARY_CROSSING: {file_path} - {reason}]"

# Template for progress updates
PROGRESS_UPDATE_TEMPLATE = """## Parallel Execution Progress

{task_status}

### Active Sessions
{active_sessions}

### Completed
{completed_count}/{total_count} tasks

### Boundary Crossings
{boundary_crossings}
"""

# Template for merge conflict escalation to CEO
MERGE_CONFLICT_ESCALATION = """## Merge Conflict Requires Your Decision

**File:** {file_path}

**Conflict Type:** {conflict_type}

### Changes from {branch_a}
```
{diff_a}
```

### Changes from {branch_b}
```
{diff_b}
```

### Agent Proposals

{agent_proposals}

### My Recommendation
{recommendation}

**Options:**
1. Accept proposal from {recommended_agent}
2. Accept proposal from {other_agent}
3. Provide custom resolution
4. Abort and rollback
"""

# Template for final execution summary
EXECUTION_SUMMARY_TEMPLATE = """## Parallel Execution Complete

### Results
- **Status:** {status}
- **Total Time:** {total_time}
- **Sequential Estimate:** {sequential_time}
- **Speedup:** {speedup}x
- **Total Tokens:** {total_tokens}

### Tasks Completed
{task_summary}

### Files Changed
{files_changed}

### Merge Summary
{merge_summary}

### Test Results
{test_results}
"""


def format_your_files(files: list[str]) -> str:
    """Format the list of files the session can modify."""
    if not files:
        return "No specific files assigned - use your best judgment."

    return "\n".join(f"- {f}" for f in files)


def format_other_files(files: list[str]) -> str:
    """Format the list of files other teams are modifying."""
    if not files:
        return "No other teams' files to avoid."

    return "\n".join(f"- {f}" for f in files)


def format_task_list(tasks: list, with_dependencies: bool = True) -> str:
    """Format a list of tasks for display."""
    lines = []

    for i, task in enumerate(tasks, 1):
        deps_str = ""
        if with_dependencies and hasattr(task, "depends_on") and task.depends_on:
            deps_str = f" (depends on: {', '.join(task.depends_on)})"

        lines.append(f"{i}. **{task.task_id}**: {task.description}{deps_str}")

        if hasattr(task, "estimated_files") and task.estimated_files:
            lines.append(f"   Files: {', '.join(task.estimated_files[:5])}")

    return "\n".join(lines)


def format_parallel_groups(groups: list[list[str]]) -> str:
    """Format parallel groups for display."""
    if not groups:
        return "No parallel groups."

    lines = []
    for i, group in enumerate(groups, 1):
        lines.append(f"Group {i}: {', '.join(group)}")

    return "\n".join(lines)


def format_sequential_order(order: list[str]) -> str:
    """Format sequential order for display."""
    if not order:
        return "No sequential order defined."

    return "\n".join(f"{i}. {task_id}" for i, task_id in enumerate(order, 1))


def format_agent_concerns(assessments: list) -> str:
    """Format agent concerns for display."""
    concerns = [a for a in assessments if a.has_concern]

    if not concerns:
        return "No concerns raised by domain agents."

    lines = []
    for concern in concerns:
        lines.append(
            f"- **{concern.agent_name}** ({concern.severity.value}): "
            f"{concern.concern_description}"
        )

    return "\n".join(lines)


def format_agent_proposals(proposals: list) -> str:
    """Format agent resolution proposals for display."""
    if not proposals:
        return "No agent proposals available."

    lines = []
    for proposal in proposals:
        lines.append(f"### {proposal.agent_name} (confidence: {proposal.confidence:.0%})")
        lines.append(f"**Reasoning:** {proposal.reasoning}")
        lines.append("**Proposed resolution:**")
        lines.append(f"```\n{proposal.proposed_resolution}\n```")
        if proposal.side_effects:
            lines.append(f"**Side effects:** {', '.join(proposal.side_effects)}")
        lines.append("")

    return "\n".join(lines)


def build_scoped_prompt(
    task,
    your_files: list[str],
    other_files: list[str],
    tier_b_context: str = "",
    tier_a_context: str = "",
) -> str:
    """Build a complete scoped prompt for a Claude Code session.

    Args:
        task: The ImplementationTask for this session.
        your_files: Files this session can modify.
        other_files: Files other sessions are modifying (avoid these).
        tier_b_context: Technical context from domain agents.
        tier_a_context: Business context from Tier A agents.

    Returns:
        Complete prompt string for the session.
    """
    return SCOPED_SESSION_PROMPT.format(
        task_description=task.description,
        your_files=format_your_files(your_files),
        other_files=format_other_files(other_files),
        tier_b_context=tier_b_context or "No additional technical context.",
        tier_a_context=tier_a_context or "No additional business context.",
    )


def build_ceo_confirmation_prompt(
    decomposition,
    plan,
) -> str:
    """Build the CEO confirmation prompt for MEDIUM/HIGH risk.

    Args:
        decomposition: DecompositionResult with task breakdown.
        plan: ParallelizationPlan with risk assessment.

    Returns:
        CEO confirmation prompt string.
    """
    # Format risk explanation
    if plan.overall_risk.value == "high":
        risk_explanation = (
            "Tasks have direct file conflicts or hard dependencies. "
            "Running in parallel would likely cause merge conflicts."
        )
        option_1 = "Run sequentially as recommended"
        option_2 = "Override: run in parallel anyway (high conflict risk)"
        recommendation = "Sequential"
        recommendation_details = SEQUENTIAL_PLAN_TEMPLATE.format(
            risk_level=plan.overall_risk.value.upper(),
            sequential_order=format_sequential_order(plan.sequential_order),
            reasoning=plan.reasoning,
            estimated_time="~" + str(int(plan.estimated_total_tokens / 15000 * 1.5)) + " min",
        )
    else:
        risk_explanation = (
            "Some potential overlaps detected, but agents believe "
            "parallel execution is safe with monitoring."
        )
        option_1 = "Proceed with parallel execution (recommended)"
        option_2 = "Run sequentially instead (safer but slower)"
        recommendation = "Parallel with caution"
        recommendation_details = PARALLEL_PLAN_TEMPLATE.format(
            risk_level=plan.overall_risk.value.upper(),
            parallel_groups=format_parallel_groups(plan.parallel_groups),
            speedup=plan.estimated_speedup,
            estimated_time="~" + str(int(plan.estimated_total_tokens / 15000 / plan.estimated_speedup)) + " min",
        )

    return CEO_CONFIRMATION_PROMPT.format(
        task_count=len(decomposition.tasks),
        task_list=format_task_list(decomposition.tasks),
        risk_level=plan.overall_risk.value.upper(),
        risk_explanation=risk_explanation,
        agent_concerns=format_agent_concerns(plan.agent_assessments),
        recommendation=recommendation,
        recommendation_details=recommendation_details,
        option_1=option_1,
        option_2=option_2,
    )
