from __future__ import annotations
"""Task decomposition for parallel sessions.

The TaskDecomposer is the "Project Manager" that breaks multi-part requests
into discrete implementation tasks, using a two-pass approach:
1. Domain Survey: Ask each agent if the request touches their domain
2. Structured Decomposition: LLM decomposes with full domain context
"""

import json
import re
import uuid
from typing import TYPE_CHECKING, Optional

from agenthub.parallel.models import (
    DecompositionResult,
    DomainClaim,
    ImplementationTask,
)

if TYPE_CHECKING:
    import anthropic

    from agenthub.auto.import_graph import ImportGraph
    from agenthub.hub import AgentHub


# Prompt templates
DOMAIN_SURVEY_PROMPT = """You are evaluating whether a user request touches your domain of expertise.

Your domain: {agent_name}
Your description: {agent_description}
Your files: {agent_paths}

User request: "{request}"

Analyze whether this request requires changes to or knowledge of code in your domain.

Respond in JSON format:
```json
{{
    "is_involved": true/false,
    "confidence": 0.0-1.0,
    "description": "What you would need to do (if involved)",
    "estimated_files": ["file1.py", "file2.py"]
}}
```

Be conservative - only claim involvement if you're reasonably sure the request touches your domain.
"""

DECOMPOSITION_PROMPT = """You are a Project Manager breaking down a user request into discrete implementation tasks.

## User Request
"{request}"

## Domain Survey Results
The following agents have claimed this request touches their domain:

{domain_claims}

## Codebase Structure
Available domains: {available_domains}

## Task

Break this request into discrete implementation tasks that can potentially be worked on in parallel.

Guidelines:
1. Each task should be completable by a single Claude Code session
2. Identify dependencies between tasks (which must complete before others can start)
3. Estimate which files each task will modify
4. Tasks that touch different domains can often run in parallel
5. Tasks that modify the same files CANNOT run in parallel safely

Respond in JSON format:
```json
{{
    "tasks": [
        {{
            "task_id": "task_1",
            "description": "Clear description of what to implement",
            "estimated_files": ["path/to/file1.py"],
            "estimated_new_files": ["path/to/new_file.py"],
            "domain_agents": ["agent_id_1"],
            "complexity": "trivial|moderate|complex",
            "depends_on": []
        }}
    ],
    "actual_complexity": "single|multi_independent|multi_dependent|multi_mixed",
    "reasoning": "Explanation of how you broke this down"
}}
```
"""

CEO_BRIEFING_TEMPLATE = """Heads up: This looks like a simple request, but it actually touches multiple domains.

**Your request:** "{request}"

**What I found:**
{complexity_summary}

**Involved teams:**
{involved_teams}

Would you like me to proceed with this breakdown, or would you prefer a different approach?
"""


class TaskDecomposer:
    """Breaks user requests into implementation tasks.

    The Project Manager - uses a two-pass decomposition approach:
    1. Domain Survey: Fast check with each agent (uses Haiku for speed)
    2. Structured Decomposition: Full LLM pass with domain context

    Example:
        >>> decomposer = TaskDecomposer(client, hub, import_graph)
        >>> result = decomposer.decompose("Add a save button and chart component")
        >>> print(f"Decomposed into {len(result.tasks)} tasks")
        >>> for task in result.tasks:
        ...     print(f"  - {task.description}")
    """

    # Survey response threshold - claims below this confidence are filtered
    SURVEY_CONFIDENCE_THRESHOLD = 0.3

    def __init__(
        self,
        client: "anthropic.Anthropic",
        hub: "AgentHub",
        import_graph: Optional["ImportGraph"] = None,
    ):
        """Initialize TaskDecomposer.

        Args:
            client: Anthropic client for LLM calls.
            hub: AgentHub for accessing domain agents.
            import_graph: Optional ImportGraph for dependency analysis.
        """
        self._client = client
        self._hub = hub
        self._graph = import_graph

    def decompose(self, request: str) -> DecompositionResult:
        """Decompose a user request into implementation tasks.

        Flow:
        1. Survey all domain agents (Tier A + Tier B)
        2. Collect domain claims
        3. Build structured prompt with claims
        4. LLM generates task breakdown
        5. Generate CEO briefing if hidden complexity detected

        Args:
            request: The user's multi-part request.

        Returns:
            DecompositionResult with tasks and metadata.
        """
        total_tokens = 0

        # Phase 1: Domain survey
        claims = self._survey_domains(request)
        survey_tokens = sum(c.confidence * 50 for c in claims)  # Approximate
        total_tokens += int(survey_tokens)

        # Quick exit: if only one domain involved, likely single task
        involved_claims = [c for c in claims if c.is_involved]

        if len(involved_claims) <= 1:
            # Single domain - create single task
            task = self._create_single_task(request, involved_claims)
            return DecompositionResult(
                tasks=[task],
                original_request=request,
                appears_simple=True,
                actual_complexity="single",
                decomposition_reasoning="Single domain involved, no decomposition needed.",
                ceo_briefing=None,
                tokens_used=total_tokens,
            )

        # Phase 2: Structured decomposition
        decomposition_result, decomposition_tokens = self._structured_decomposition(
            request, claims
        )
        total_tokens += decomposition_tokens

        # Check for hidden complexity
        appears_simple = self._appears_simple(request)
        ceo_briefing = None

        if appears_simple and decomposition_result["actual_complexity"] != "single":
            ceo_briefing = self._generate_ceo_briefing(
                request, decomposition_result, involved_claims
            )

        # Build tasks from decomposition
        tasks = self._build_tasks(decomposition_result, claims)

        return DecompositionResult(
            tasks=tasks,
            original_request=request,
            appears_simple=appears_simple,
            actual_complexity=decomposition_result.get("actual_complexity", "multi_mixed"),
            decomposition_reasoning=decomposition_result.get("reasoning", ""),
            ceo_briefing=ceo_briefing,
            tokens_used=total_tokens,
        )

    def _survey_domains(self, request: str) -> list[DomainClaim]:
        """Survey all domain agents to see which are involved.

        Uses Haiku for speed. Returns claims from agents with confidence >= threshold.

        Optimizations:
        - Skip LLM survey for agents with zero keyword overlap (saves 40-60% tokens)
        - Run LLM surveys in parallel (saves ~80% wall-clock time)

        Args:
            request: The user request to analyze.

        Returns:
            List of DomainClaim from all agents (including non-involved).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        claims: list[DomainClaim] = []

        # Get all agents (Tier A business + Tier B code)
        all_agents = self._hub.list_agents()
        request_lower = request.lower()
        request_words = set(request_lower.split())

        # Separate agents that need LLM survey from those that don't
        agents_to_survey = []
        for agent_spec in all_agents:
            if not self._has_keyword_overlap(request_lower, request_words, agent_spec):
                claims.append(DomainClaim(
                    agent_id=agent_spec.agent_id,
                    agent_name=agent_spec.name,
                    is_involved=False,
                    description="No keyword overlap with request",
                    estimated_files=[],
                    confidence=0.0,
                ))
            else:
                agents_to_survey.append(agent_spec)

        # Run LLM surveys in parallel (IO-bound, safe to parallelize)
        if agents_to_survey:
            with ThreadPoolExecutor(max_workers=min(len(agents_to_survey), 6)) as pool:
                futures = {
                    pool.submit(self._survey_single_agent, request, spec): spec
                    for spec in agents_to_survey
                }
                for future in as_completed(futures):
                    spec = futures[future]
                    try:
                        claim = future.result(timeout=30)
                        claims.append(claim)
                    except Exception as e:
                        print(f"Warning: Survey failed for {spec.agent_id}: {e}")

        return claims

    def _has_keyword_overlap(self, request_lower: str, request_words: set, agent_spec) -> bool:
        """Check if agent has any keyword overlap with request.

        This is a fast heuristic to skip LLM surveys for clearly irrelevant agents.

        Args:
            request_lower: Lowercased request string.
            request_words: Set of words in the request.
            agent_spec: Agent specification to check.

        Returns:
            True if there's potential overlap, False if clearly no match.
        """
        # Check agent keywords
        for keyword in agent_spec.context_keywords:
            kw_lower = keyword.lower()
            # Check if keyword is in request (substring match)
            if kw_lower in request_lower:
                return True
            # Check word-level match
            if kw_lower in request_words:
                return True

        # Check agent name/description for relevance
        agent_name_lower = agent_spec.name.lower()
        agent_desc_lower = agent_spec.description.lower()

        # Extract meaningful words from request (skip common words)
        common_words = {"the", "a", "an", "is", "are", "to", "for", "and", "or", "in", "on", "with", "how", "what", "why", "can", "should", "would", "could", "please", "help", "me", "i", "we", "you"}
        meaningful_words = request_words - common_words

        for word in meaningful_words:
            if len(word) >= 3:  # Skip very short words
                if word in agent_name_lower or word in agent_desc_lower:
                    return True

        # Check if request mentions file paths that match agent's context
        for path in agent_spec.context_paths[:5]:  # Check first 5 paths
            path_parts = path.lower().replace("/", " ").replace("\\", " ").split()
            for part in path_parts:
                if len(part) >= 3 and part in request_lower:
                    return True

        return False

    def _survey_single_agent(self, request: str, agent_spec) -> DomainClaim:
        """Survey a single agent for domain involvement.

        Args:
            request: The user request.
            agent_spec: AgentSpec to survey.

        Returns:
            DomainClaim with the agent's assessment.
        """
        prompt = DOMAIN_SURVEY_PROMPT.format(
            agent_name=agent_spec.name,
            agent_description=agent_spec.description,
            agent_paths=", ".join(agent_spec.context_paths[:10]),  # Limit for prompt
            request=request,
        )

        # Use Haiku for fast survey
        response = self._client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # Parse response
        response_text = response.content[0].text
        data = self._parse_json_response(response_text)

        return DomainClaim(
            agent_id=agent_spec.agent_id,
            agent_name=agent_spec.name,
            is_involved=data.get("is_involved", False),
            description=data.get("description", ""),
            estimated_files=data.get("estimated_files", []),
            confidence=float(data.get("confidence", 0.0)),
        )

    def _structured_decomposition(
        self, request: str, claims: list[DomainClaim]
    ) -> tuple[dict, int]:
        """Perform structured decomposition using full LLM.

        Args:
            request: The user request.
            claims: Domain claims from survey.

        Returns:
            Tuple of (decomposition dict, tokens used).
        """
        # Format domain claims for prompt
        involved_claims = [c for c in claims if c.is_involved]
        claims_text = "\n".join(
            f"- **{c.agent_name}** (confidence: {c.confidence:.1f}): {c.description}"
            for c in involved_claims
        )

        # Get available domains
        all_agents = self._hub.list_agents()
        domains_text = ", ".join(a.name for a in all_agents)

        prompt = DECOMPOSITION_PROMPT.format(
            request=request,
            domain_claims=claims_text or "No agents claimed involvement.",
            available_domains=domains_text,
        )

        response = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        response_text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens

        data = self._parse_json_response(response_text)

        return data, tokens

    def _build_tasks(
        self, decomposition: dict, claims: list[DomainClaim]
    ) -> list[ImplementationTask]:
        """Build ImplementationTask objects from decomposition.

        Args:
            decomposition: Parsed decomposition result.
            claims: Domain claims for cross-referencing.

        Returns:
            List of ImplementationTask objects.
        """
        tasks: list[ImplementationTask] = []

        for task_data in decomposition.get("tasks", []):
            task_id = task_data.get("task_id", f"task_{uuid.uuid4().hex[:8]}")

            # Map domain agents
            domain_agent_ids = task_data.get("domain_agents", [])
            if not domain_agent_ids:
                # Try to infer from estimated files
                domain_agent_ids = self._infer_agents_from_files(
                    task_data.get("estimated_files", []), claims
                )

            # Estimate tokens based on complexity
            complexity = task_data.get("complexity", "moderate")
            token_estimates = {"trivial": 5000, "moderate": 15000, "complex": 30000}
            estimated_tokens = token_estimates.get(complexity, 15000)

            task = ImplementationTask(
                task_id=task_id,
                description=task_data.get("description", ""),
                estimated_files=task_data.get("estimated_files", []),
                estimated_new_files=task_data.get("estimated_new_files", []),
                domain_agents=domain_agent_ids,
                complexity=complexity,
                estimated_tokens=estimated_tokens,
                depends_on=task_data.get("depends_on", []),
            )
            tasks.append(task)

        return tasks

    def _create_single_task(
        self, request: str, claims: list[DomainClaim]
    ) -> ImplementationTask:
        """Create a single task when no decomposition needed.

        Args:
            request: The user request.
            claims: Involved domain claims.

        Returns:
            Single ImplementationTask.
        """
        domain_agents = [c.agent_id for c in claims if c.is_involved]
        estimated_files = []
        for claim in claims:
            if claim.is_involved:
                estimated_files.extend(claim.estimated_files)

        return ImplementationTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            description=request,
            estimated_files=list(set(estimated_files)),
            estimated_new_files=[],
            domain_agents=domain_agents,
            complexity="moderate",
            estimated_tokens=15000,
            depends_on=[],
        )

    def _infer_agents_from_files(
        self, files: list[str], claims: list[DomainClaim]
    ) -> list[str]:
        """Infer which agents own a set of files.

        Args:
            files: List of file paths.
            claims: Domain claims to check against.

        Returns:
            List of agent IDs that likely own these files.
        """
        agent_ids: set[str] = set()

        for claim in claims:
            if not claim.is_involved:
                continue
            # Check if any estimated files overlap
            for claim_file in claim.estimated_files:
                for file in files:
                    if file in claim_file or claim_file in file:
                        agent_ids.add(claim.agent_id)
                        break

        return list(agent_ids)

    def _appears_simple(self, request: str) -> bool:
        """Check if a request appears simple on the surface.

        Uses heuristics to detect requests that look simple but may be complex.

        Args:
            request: The user request.

        Returns:
            True if the request appears simple.
        """
        # Simple heuristics
        request_lower = request.lower()

        # Multiple explicit items suggest complexity
        if " and " in request_lower or ", " in request_lower:
            return False

        # Numbered lists suggest complexity
        if re.search(r"\d\.", request):
            return False

        # Short requests are more likely to appear simple
        word_count = len(request.split())
        if word_count > 30:
            return False

        return True

    def _generate_ceo_briefing(
        self,
        request: str,
        decomposition: dict,
        claims: list[DomainClaim],
    ) -> str:
        """Generate a briefing when hidden complexity is detected.

        Only called when appears_simple=True but actual_complexity != "single".

        Args:
            request: The original request.
            decomposition: The decomposition result.
            claims: Involved domain claims.

        Returns:
            CEO briefing message.
        """
        task_count = len(decomposition.get("tasks", []))
        complexity_summary = f"This request requires {task_count} separate tasks."

        if decomposition.get("actual_complexity") == "multi_dependent":
            complexity_summary += " Some tasks depend on others and must run sequentially."
        elif decomposition.get("actual_complexity") == "multi_independent":
            complexity_summary += " These tasks are independent and can run in parallel."

        # Format involved teams
        involved_teams = "\n".join(
            f"- **{c.agent_name}**: {c.description}"
            for c in claims
            if c.is_involved
        )

        return CEO_BRIEFING_TEMPLATE.format(
            request=request,
            complexity_summary=complexity_summary,
            involved_teams=involved_teams or "No specific teams identified.",
        )

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response.

        Args:
            text: Response text that may contain JSON.

        Returns:
            Parsed dict, or empty dict if parsing fails.
        """
        # Try to extract JSON from code blocks
        json_match = re.search(r"```(?:json)?\n?(.*?)```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try parsing the whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        return {}
