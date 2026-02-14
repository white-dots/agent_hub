from __future__ import annotations
"""Pre-computed routing index for fast query-time agent selection.

This module builds an inverted index during setup time, allowing O(keywords)
lookup instead of O(agents × keywords) at query time. The index is cached
on disk with git-aware invalidation.
"""

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agenthub.models import AgentSpec


@dataclass
class AgentRoutingMetadata:
    """Routing metadata for a single agent."""

    agent_id: str
    keywords: list[str]
    keyword_weights: dict[str, float]  # keyword -> weight (1.0 = normal, 2.0 = high priority)
    domain: Optional[str]  # frontend, backend, database, test, etc.
    tier: str  # "A" or "B"
    description: str
    is_sub_agent: bool = False

    # Per-agent routing settings (from AgentSpec.routing)
    exclusions: list[str] = field(default_factory=list)
    fallback_agent_id: Optional[str] = None
    priority: int = 0
    min_confidence: float = 0.0
    prefer_exact_match: bool = False


@dataclass
class RoutingIndex:
    """Pre-computed routing index for fast agent lookup.

    The inverted index maps keywords to lists of (agent_id, weight) tuples,
    enabling fast scoring at query time.
    """

    # Inverted index: keyword -> list of (agent_id, weight)
    keyword_to_agents: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    # Agent metadata for additional filtering
    agent_metadata: dict[str, AgentRoutingMetadata] = field(default_factory=dict)

    # Domain mappings for quick domain-based filtering
    domain_to_agents: dict[str, list[str]] = field(default_factory=dict)

    # IDF factors for keywords (pre-computed at build time).
    # keyword -> idf_score. Higher = more discriminating.
    keyword_idf: dict[str, float] = field(default_factory=dict)

    # Index metadata
    generated_at: str = ""
    git_commit: Optional[str] = None
    project_hash: str = ""
    agent_count: int = 0

    def get_agents_for_keyword(self, keyword: str) -> list[tuple[str, float]]:
        """Get agents that match a keyword with their weights.

        Args:
            keyword: The keyword to look up.

        Returns:
            List of (agent_id, weight) tuples.
        """
        return self.keyword_to_agents.get(keyword.lower(), [])

    def get_agents_for_domain(self, domain: str) -> list[str]:
        """Get all agents for a specific domain.

        Args:
            domain: Domain name (frontend, backend, database, test).

        Returns:
            List of agent IDs.
        """
        return self.domain_to_agents.get(domain.lower(), [])

    def get_agent_domain(self, agent_id: str) -> Optional[str]:
        """Get the domain of a specific agent.

        Args:
            agent_id: The agent ID.

        Returns:
            Domain name or None.
        """
        metadata = self.agent_metadata.get(agent_id)
        return metadata.domain if metadata else None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "keyword_to_agents": self.keyword_to_agents,
            "agent_metadata": {
                aid: {
                    "agent_id": m.agent_id,
                    "keywords": m.keywords,
                    "keyword_weights": m.keyword_weights,
                    "domain": m.domain,
                    "tier": m.tier,
                    "description": m.description,
                    "is_sub_agent": m.is_sub_agent,
                    "exclusions": m.exclusions,
                    "fallback_agent_id": m.fallback_agent_id,
                    "priority": m.priority,
                    "min_confidence": m.min_confidence,
                    "prefer_exact_match": m.prefer_exact_match,
                }
                for aid, m in self.agent_metadata.items()
            },
            "domain_to_agents": self.domain_to_agents,
            "keyword_idf": self.keyword_idf,
            "generated_at": self.generated_at,
            "git_commit": self.git_commit,
            "project_hash": self.project_hash,
            "agent_count": self.agent_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingIndex":
        """Create from JSON dict."""
        index = cls(
            keyword_to_agents={
                k: [(a, w) for a, w in v] for k, v in data.get("keyword_to_agents", {}).items()
            },
            domain_to_agents=data.get("domain_to_agents", {}),
            keyword_idf=data.get("keyword_idf", {}),
            generated_at=data.get("generated_at", ""),
            git_commit=data.get("git_commit"),
            project_hash=data.get("project_hash", ""),
            agent_count=data.get("agent_count", 0),
        )

        # Restore agent metadata
        for aid, m in data.get("agent_metadata", {}).items():
            index.agent_metadata[aid] = AgentRoutingMetadata(
                agent_id=m["agent_id"],
                keywords=m["keywords"],
                keyword_weights=m["keyword_weights"],
                domain=m.get("domain"),
                tier=m["tier"],
                description=m["description"],
                is_sub_agent=m.get("is_sub_agent", False),
                exclusions=m.get("exclusions", []),
                fallback_agent_id=m.get("fallback_agent_id"),
                priority=m.get("priority", 0),
                min_confidence=m.get("min_confidence", 0.0),
                prefer_exact_match=m.get("prefer_exact_match", False),
            )

        return index


class RoutingIndexBuilder:
    """Builds routing index from discovered agents.

    This class analyzes agent specs to create an optimized routing index
    that can be used for fast query-time agent selection.

    Example:
        >>> builder = RoutingIndexBuilder(project_root="/path/to/project")
        >>> index = builder.build(agents)
        >>> builder.save(index)
    """

    # Domain detection keywords
    DOMAIN_INDICATORS = {
        "frontend": [
            "react", "component", "tsx", "jsx", "css", "style", "ui",
            "button", "form", "modal", "hook", "useState", "useEffect",
            "redux", "styled", "tailwind", "html", "dom", "view",
        ],
        "backend": [
            "api", "endpoint", "route", "controller", "service", "handler",
            "request", "response", "middleware", "fastapi", "flask", "django",
            "express", "server", "rest", "graphql",
        ],
        "database": [
            "model", "schema", "migration", "query", "sql", "database", "db",
            "orm", "entity", "repository", "postgres", "mysql", "sqlite",
            "mongodb", "redis", "table", "column",
        ],
        "test": [
            "test", "spec", "mock", "fixture", "jest", "pytest", "unittest",
            "coverage", "assertion", "expect", "describe", "it",
        ],
    }

    # High-priority keywords that should get extra weight
    HIGH_PRIORITY_SUFFIXES = ["expert", "specialist", "manager", "handler"]

    def __init__(self, project_root: str):
        """Initialize the builder.

        Args:
            project_root: Path to the project root.
        """
        self.project_root = Path(project_root).resolve()
        self._cache_dir = Path.home() / ".agenthub" / "projects"

    def build(self, agents: list["AgentSpec"]) -> RoutingIndex:
        """Build routing index from agent specs.

        Args:
            agents: List of AgentSpec objects.

        Returns:
            Populated RoutingIndex.
        """
        index = RoutingIndex(
            generated_at=datetime.now().isoformat(),
            git_commit=self._get_git_commit(),
            project_hash=self._get_project_hash(),
            agent_count=len(agents),
        )

        for agent in agents:
            # Build agent metadata
            metadata = self._build_agent_metadata(agent)
            index.agent_metadata[agent.agent_id] = metadata

            # Build inverted keyword index
            for keyword in metadata.keywords:
                keyword_lower = keyword.lower()
                weight = metadata.keyword_weights.get(keyword_lower, 1.0)

                if keyword_lower not in index.keyword_to_agents:
                    index.keyword_to_agents[keyword_lower] = []

                index.keyword_to_agents[keyword_lower].append(
                    (agent.agent_id, weight)
                )

            # Build domain mapping
            if metadata.domain:
                if metadata.domain not in index.domain_to_agents:
                    index.domain_to_agents[metadata.domain] = []
                index.domain_to_agents[metadata.domain].append(agent.agent_id)

        # Pre-compute IDF for each keyword.
        # IDF = log(N / df) + 1.0 where N = total agents, df = agents with this keyword.
        num_agents = max(len(agents), 1)
        for keyword, agent_weights in index.keyword_to_agents.items():
            df = len(agent_weights)
            index.keyword_idf[keyword] = math.log(num_agents / df) + 1.0

        return index

    def _build_agent_metadata(self, agent: "AgentSpec") -> AgentRoutingMetadata:
        """Build routing metadata for a single agent.

        Respects agent-declared routing settings when present. Falls back to
        heuristic inference only when the agent's RoutingConfig fields are empty.

        Args:
            agent: AgentSpec object.

        Returns:
            AgentRoutingMetadata with extracted routing info.
        """
        routing = getattr(agent, "routing", None)

        # Normalize keywords
        keywords = [k.lower().strip() for k in agent.context_keywords]

        # ── Keyword weights ──
        # If agent declares its own weights, use those; otherwise infer.
        if routing and routing.keyword_weights:
            keyword_weights = {k.lower(): w for k, w in routing.keyword_weights.items()}
            # Ensure all context_keywords have at least a default weight
            for kw in keywords:
                if kw not in keyword_weights:
                    keyword_weights[kw] = 1.0
        else:
            keyword_weights = {}
            for kw in keywords:
                weight = 1.0
                if any(suffix in kw for suffix in self.HIGH_PRIORITY_SUFFIXES):
                    weight = 1.5
                if kw in self.DOMAIN_INDICATORS:
                    weight = 2.0
                keyword_weights[kw] = weight

        # ── Domain ──
        # If agent declares domains, use the first one; otherwise infer.
        if routing and routing.domains:
            domain = routing.domains[0].lower()
        else:
            domain = self._detect_domain(agent, keywords)

        # Determine tier
        tier = "B" if agent.metadata.get("auto_generated") else "A"

        # ── Per-agent routing settings ──
        exclusions = [e.lower() for e in routing.exclusions] if routing else []
        fallback_agent_id = routing.fallback_agent_id if routing else None
        priority = routing.priority if routing else 0
        min_confidence = routing.min_confidence if routing else 0.0
        prefer_exact_match = routing.prefer_exact_match if routing else False

        return AgentRoutingMetadata(
            agent_id=agent.agent_id,
            keywords=keywords,
            keyword_weights=keyword_weights,
            domain=domain,
            tier=tier,
            description=agent.description or "",
            is_sub_agent=agent.metadata.get("is_sub_agent", False),
            exclusions=exclusions,
            fallback_agent_id=fallback_agent_id,
            priority=priority,
            min_confidence=min_confidence,
            prefer_exact_match=prefer_exact_match,
        )

    def _detect_domain(self, agent: "AgentSpec", keywords: list[str]) -> Optional[str]:
        """Detect the domain of an agent based on keywords and name.

        Args:
            agent: AgentSpec object.
            keywords: Normalized keywords.

        Returns:
            Domain name or None.
        """
        agent_name_lower = agent.name.lower()
        agent_id_lower = agent.agent_id.lower()

        # Check agent name/ID for domain
        for domain, indicators in self.DOMAIN_INDICATORS.items():
            if domain in agent_name_lower or domain in agent_id_lower:
                return domain

        # Check keyword overlap
        keyword_set = set(keywords)
        best_domain = None
        best_overlap = 0

        for domain, indicators in self.DOMAIN_INDICATORS.items():
            indicator_set = set(indicators)
            overlap = len(keyword_set & indicator_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_domain = domain

        # Require at least 2 keyword matches
        return best_domain if best_overlap >= 2 else None

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return None

    def _get_project_hash(self) -> str:
        """Get hash of project path for cache key."""
        return hashlib.md5(str(self.project_root).encode()).hexdigest()[:12]

    def _get_cache_path(self) -> Path:
        """Get path to cached index file."""
        project_hash = self._get_project_hash()
        cache_dir = self._cache_dir / project_hash
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "routing_index.json"

    def save(self, index: RoutingIndex) -> Path:
        """Save index to disk cache.

        Args:
            index: RoutingIndex to save.

        Returns:
            Path to saved file.
        """
        cache_path = self._get_cache_path()
        with open(cache_path, "w") as f:
            json.dump(index.to_dict(), f, indent=2)
        return cache_path

    def load(self) -> Optional[RoutingIndex]:
        """Load index from disk cache if valid.

        Returns:
            RoutingIndex if cache is valid, None otherwise.
        """
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)

            index = RoutingIndex.from_dict(data)

            # Validate cache - check git commit
            current_commit = self._get_git_commit()
            if current_commit and index.git_commit:
                if current_commit != index.git_commit:
                    # Git changed - invalidate cache
                    return None

            return index

        except Exception:
            return None

    def invalidate_cache(self) -> None:
        """Invalidate the cached index."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            cache_path.unlink()


class IndexedKeywordRouter:
    """Keyword router that uses pre-computed routing index.

    This router is faster than KeywordRouter for large agent counts
    because it uses an inverted index for O(keywords) lookup.

    Supports per-agent routing settings: priority tiebreakers, exclusion
    penalties, confidence thresholds, exact-match preferences, and
    fallback chains.

    Example:
        >>> index = builder.build(agents)
        >>> router = IndexedKeywordRouter(index)
        >>> agent_id = router.route("how does the API handle auth?")
    """

    # Domain exclusion - query domain vs agent domain penalty
    DOMAIN_MISMATCH_PENALTY = 5
    # Penalty when query matches an agent's exclusion list
    EXCLUSION_PENALTY = 5
    # Maximum fallback chain depth to prevent infinite loops
    MAX_FALLBACK_DEPTH = 3

    def __init__(self, index: RoutingIndex, case_sensitive: bool = False):
        """Initialize with routing index.

        Args:
            index: Pre-computed RoutingIndex.
            case_sensitive: Whether matching is case-sensitive.
        """
        self.index = index
        self.case_sensitive = case_sensitive

    def route(self, query: str) -> Optional[str]:
        """Route query to best matching agent, with fallback chain support.

        If the best agent's confidence threshold isn't met, follows the
        fallback chain (up to MAX_FALLBACK_DEPTH hops).

        Args:
            query: The user's query.

        Returns:
            Agent ID of best match, or None.
        """
        scores = self.get_all_scores(query)
        if not scores:
            return None

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Find best agent, compute max possible score for normalization
        max_possible = self._estimate_max_score(query)

        for agent_id, score in ranked:
            if score <= 0:
                break

            metadata = self.index.agent_metadata.get(agent_id)
            if not metadata:
                return agent_id

            # Check confidence threshold
            if metadata.min_confidence > 0.0 and max_possible > 0:
                normalized = score / max_possible
                if normalized < metadata.min_confidence:
                    # Try fallback chain
                    fallback = self._follow_fallback_chain(
                        metadata.fallback_agent_id, scores, max_possible
                    )
                    if fallback:
                        return fallback
                    # No valid fallback — try next ranked agent
                    continue

            return agent_id

        return None

    def _follow_fallback_chain(
        self, start_id: Optional[str], scores: dict[str, float], max_possible: float
    ) -> Optional[str]:
        """Follow fallback chain until an agent accepts or chain exhausts.

        Args:
            start_id: First fallback agent ID.
            scores: Pre-computed scores for all agents.
            max_possible: Max possible score for normalization.

        Returns:
            Agent ID that accepts, or None.
        """
        visited: set[str] = set()
        current = start_id

        for _ in range(self.MAX_FALLBACK_DEPTH):
            if not current or current in visited:
                break
            visited.add(current)

            metadata = self.index.agent_metadata.get(current)
            if not metadata:
                break

            # Check if this fallback agent accepts
            agent_score = scores.get(current, 0.0)
            if metadata.min_confidence <= 0.0:
                return current  # Agent accepts anything
            if max_possible > 0 and (agent_score / max_possible) >= metadata.min_confidence:
                return current

            current = metadata.fallback_agent_id

        return None

    def _estimate_max_score(self, query: str) -> float:
        """Estimate max possible score for a query (for normalization).

        Uses total keyword weight that could match as the ceiling.
        """
        query_text = query if self.case_sensitive else query.lower()
        total = 0.0
        for keyword, agent_weights in self.index.keyword_to_agents.items():
            if keyword in query_text:
                # Max weight any agent could get for this keyword
                if agent_weights:
                    total += max(w for _, w in agent_weights) + 0.5  # include exact-match bonus
        return max(total, 1.0)

    def get_all_scores(self, query: str) -> dict[str, float]:
        """Get scores for all agents.

        Uses whole-word matching as the primary signal and substring matching
        as a weak fallback (0.3x weight). Applies pre-computed IDF weighting
        so keywords shared across many agents contribute less.

        Incorporates per-agent routing settings: priority tiebreakers,
        exclusion penalties, and exact-match preferences.

        Args:
            query: The user's query.

        Returns:
            Dict mapping agent_id to score.
        """
        query_text = query if self.case_sensitive else query.lower()
        query_words = set(query_text.split())

        # Initialize scores
        scores: dict[str, float] = {
            aid: 0.0 for aid in self.index.agent_metadata
        }

        # Score using inverted index - O(query_keywords)
        for keyword, agent_weights in self.index.keyword_to_agents.items():
            is_whole_word = keyword in query_words
            is_substring = (not is_whole_word) and (keyword in query_text)

            if not is_whole_word and not is_substring:
                continue

            idf_factor = self.index.keyword_idf.get(keyword, 1.0)

            for agent_id, weight in agent_weights:
                if is_whole_word:
                    # PRIMARY: whole-word match (full weight + exact bonus)
                    metadata = self.index.agent_metadata.get(agent_id)
                    bonus = 1.5 if (metadata and metadata.prefer_exact_match) else 0.5
                    scores[agent_id] += (weight + bonus) * idf_factor
                else:
                    # SECONDARY: substring-only match (0.3x weight)
                    scores[agent_id] += weight * 0.3 * idf_factor

        # Apply domain mismatch penalty
        query_domains = self._detect_query_domains(query_text)
        if query_domains:
            for agent_id in scores:
                agent_domain = self.index.get_agent_domain(agent_id)
                if agent_domain and agent_domain not in query_domains:
                    scores[agent_id] = max(0, scores[agent_id] - self.DOMAIN_MISMATCH_PENALTY)

        # Apply per-agent exclusion penalty
        for agent_id, metadata in self.index.agent_metadata.items():
            if metadata.exclusions:
                for excl in metadata.exclusions:
                    if excl in query_text:
                        scores[agent_id] = max(0, scores[agent_id] - self.EXCLUSION_PENALTY)
                        break  # One penalty per agent is enough

        # Apply priority tiebreaker (small bonus to not overwhelm keyword scores)
        for agent_id, metadata in self.index.agent_metadata.items():
            if metadata.priority != 0:
                scores[agent_id] += metadata.priority * 0.1

        return scores

    def _detect_query_domains(self, query: str) -> set[str]:
        """Detect which domain(s) a query is about."""
        domains = set()
        for domain, agents in self.index.domain_to_agents.items():
            # Check if query contains domain-specific keywords
            domain_keywords = set()
            for agent_id in agents:
                metadata = self.index.agent_metadata.get(agent_id)
                if metadata:
                    domain_keywords.update(metadata.keywords)

            matching = sum(1 for kw in domain_keywords if kw in query)
            if matching >= 2:
                domains.add(domain)

        return domains

    def get_candidates(self, query: str, min_score: float = 1.0) -> list[str]:
        """Get candidate agents with score above threshold.

        Args:
            query: The user's query.
            min_score: Minimum score to be considered a candidate.

        Returns:
            List of agent IDs sorted by score (highest first).
        """
        scores = self.get_all_scores(query)
        candidates = [
            (aid, score) for aid, score in scores.items() if score >= min_score
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in candidates]
