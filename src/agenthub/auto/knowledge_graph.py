"""Auto-generated codebase knowledge graph.

Builds a semantic relationship graph from ImportGraph and DomainAnalysis
outputs at discover_all_agents() time.  No LLM calls — pure static analysis.

Node types:
  - DomainNode: discovered code domains (from DomainAnalysis)
  - EntityNode: classes/models shared across domains
  - ConceptNode: function-name-derived concepts

Edge types:
  - IMPORTS: direct import dependency between domains
  - SHARES_ENTITY: two domains define/use the same class
  - CALLS_INTO: cross-domain function calls
  - DOMAIN_OVERLAP: keyword intersection between domains
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenthub.auto.domain_analysis import Domain
    from agenthub.auto.import_graph import ImportGraph


# ── Node types ──────────────────────────────────────────────

@dataclass
class DomainNode:
    domain_id: str
    name: str
    agent_id: str
    modules: list[str]
    keywords: list[str]


@dataclass
class EntityNode:
    name: str          # Class name, e.g. "Product"
    domains: list[str] # domain_ids that define/use this entity
    modules: list[str] # file paths where it appears


@dataclass
class ConceptNode:
    name: str          # e.g. "discount", "authentication"
    domains: list[str]
    source: str = "function"  # "function" | "keyword"


# ── Edge types ──────────────────────────────────────────────

@dataclass
class KGEdge:
    source: str     # domain_id or entity name
    target: str     # domain_id or entity name
    edge_type: str  # IMPORTS | SHARES_ENTITY | CALLS_INTO | DOMAIN_OVERLAP
    weight: float = 1.0
    detail: str = ""


# ── Knowledge Graph ─────────────────────────────────────────

class CodebaseKnowledgeGraph:
    """Lightweight semantic graph built from static analysis."""

    def __init__(
        self,
        import_graph: "ImportGraph",
        domains: list["Domain"],
    ) -> None:
        self._graph = import_graph
        self._domains = domains

        self.domain_nodes: dict[str, DomainNode] = {}
        self.entity_nodes: dict[str, EntityNode] = {}
        self.concept_nodes: dict[str, ConceptNode] = {}
        self.edges: list[KGEdge] = []

        # Lookup helpers
        self._module_to_domain: dict[str, str] = {}
        self._domain_edges: dict[str, list[KGEdge]] = defaultdict(list)
        self._built = False

    # ── Build ────────────────────────────────────────────

    def build(self) -> None:
        """Populate nodes and edges from static analysis."""
        self._build_domain_nodes()
        self._build_module_index()
        self._build_entity_nodes()
        self._build_concept_nodes()
        self._build_import_edges()
        self._build_shared_entity_edges()
        self._build_domain_overlap_edges()
        self._index_edges()
        self._built = True

    def _build_domain_nodes(self) -> None:
        for d in self._domains:
            self.domain_nodes[d.agent_id] = DomainNode(
                domain_id=d.agent_id,
                name=d.name,
                agent_id=d.agent_id,
                modules=list(d.modules),
                keywords=list(d.keywords),
            )

    def _build_module_index(self) -> None:
        """Map each module path → owning domain_id."""
        for d in self._domains:
            for mod in d.modules:
                self._module_to_domain[mod] = d.agent_id

    def _build_entity_nodes(self) -> None:
        """Extract classes from ImportGraph nodes and map to domains."""
        class_to_modules: dict[str, list[str]] = defaultdict(list)

        for path, node in self._graph.nodes.items():
            for cls in node.classes:
                # Skip very short or private class names
                if len(cls) < 3 or cls.startswith("_"):
                    continue
                class_to_modules[cls].append(path)

        for cls, modules in class_to_modules.items():
            domains = list({
                self._module_to_domain[m]
                for m in modules
                if m in self._module_to_domain
            })
            if domains:
                self.entity_nodes[cls] = EntityNode(
                    name=cls, domains=domains, modules=modules,
                )

    def _build_concept_nodes(self) -> None:
        """Extract concepts from function names, grouped by domain."""
        concept_domains: dict[str, set[str]] = defaultdict(set)

        for path, node in self._graph.nodes.items():
            domain_id = self._module_to_domain.get(path)
            if not domain_id:
                continue
            for func in node.functions:
                # Split camelCase/snake_case into words
                words = re.findall(r"[a-z]+", func.lower())
                for w in words:
                    if len(w) >= 4 and w not in {
                        "self", "init", "main", "test", "true", "false",
                        "none", "list", "dict", "args", "kwargs", "async",
                        "return", "class", "import", "from", "with",
                    }:
                        concept_domains[w].add(domain_id)

        for concept, domains in concept_domains.items():
            if len(domains) >= 1:
                self.concept_nodes[concept] = ConceptNode(
                    name=concept,
                    domains=sorted(domains),
                    source="function",
                )

    def _build_import_edges(self) -> None:
        """Find cross-domain import edges (IMPORTS + CALLS_INTO)."""
        for edge in self._graph.edges:
            src_domain = self._module_to_domain.get(edge.source)
            tgt_domain = self._module_to_domain.get(edge.target)

            if src_domain and tgt_domain and src_domain != tgt_domain:
                # Cross-domain import
                edge_type = "CALLS_INTO" if edge.imported_names else "IMPORTS"
                detail = ", ".join(edge.imported_names[:5]) if edge.imported_names else ""
                self.edges.append(KGEdge(
                    source=src_domain,
                    target=tgt_domain,
                    edge_type=edge_type,
                    weight=1.0 + len(edge.imported_names) * 0.2,
                    detail=detail,
                ))

    def _build_shared_entity_edges(self) -> None:
        """Find SHARES_ENTITY edges: entities appearing in 2+ domains."""
        for entity in self.entity_nodes.values():
            if len(entity.domains) >= 2:
                # Create edge between each pair of domains sharing this entity
                for i, d1 in enumerate(entity.domains):
                    for d2 in entity.domains[i + 1:]:
                        self.edges.append(KGEdge(
                            source=d1,
                            target=d2,
                            edge_type="SHARES_ENTITY",
                            weight=2.0,  # Shared entities are strong signals
                            detail=entity.name,
                        ))

    def _build_domain_overlap_edges(self) -> None:
        """Find DOMAIN_OVERLAP edges: keyword intersection between domains."""
        domain_list = list(self.domain_nodes.values())
        for i, d1 in enumerate(domain_list):
            kw1 = set(d1.keywords)
            for d2 in domain_list[i + 1:]:
                kw2 = set(d2.keywords)
                overlap = kw1 & kw2
                # Only count meaningful overlaps (ignore very common words)
                overlap -= {"app", "src", "api", "get", "set", "run", "init", "main"}
                if len(overlap) >= 2:
                    self.edges.append(KGEdge(
                        source=d1.domain_id,
                        target=d2.domain_id,
                        edge_type="DOMAIN_OVERLAP",
                        weight=len(overlap) * 0.5,
                        detail=", ".join(sorted(overlap)[:5]),
                    ))

    def _index_edges(self) -> None:
        """Build domain → edges lookup."""
        self._domain_edges.clear()
        for edge in self.edges:
            self._domain_edges[edge.source].append(edge)
            self._domain_edges[edge.target].append(edge)

    # ── Query API ────────────────────────────────────────

    def get_related_domains(
        self, domain_id: str,
    ) -> list[tuple[str, str, float]]:
        """Get domains related to the given domain.

        Returns:
            List of (related_domain_id, edge_type, weight) sorted by weight desc.
        """
        results: dict[str, tuple[str, float]] = {}
        for edge in self._domain_edges.get(domain_id, []):
            other = edge.target if edge.source == domain_id else edge.source
            if other in results:
                # Accumulate weights for same pair
                prev_type, prev_weight = results[other]
                results[other] = (
                    f"{prev_type}+{edge.edge_type}",
                    prev_weight + edge.weight,
                )
            else:
                results[other] = (edge.edge_type, edge.weight)

        return sorted(
            [(did, etype, w) for did, (etype, w) in results.items()],
            key=lambda x: x[2],
            reverse=True,
        )

    def get_shared_entities(
        self, domain_a: str, domain_b: str,
    ) -> list[str]:
        """Get entity names shared between two domains."""
        return [
            e.name for e in self.entity_nodes.values()
            if domain_a in e.domains and domain_b in e.domains
        ]

    def query_relevant_domains(
        self, query: str,
    ) -> list[tuple[str, float]]:
        """Find domains relevant to a query using keywords + entities.

        Returns:
            List of (domain_id, score) sorted by score desc.
        """
        query_lower = query.lower()
        query_words = set(re.findall(r"[a-z]+", query_lower))
        scores: dict[str, float] = defaultdict(float)

        # Score by domain keywords
        for domain in self.domain_nodes.values():
            kw_set = set(k.lower() for k in domain.keywords)
            overlap = query_words & kw_set
            if overlap:
                scores[domain.domain_id] += len(overlap) * 1.0

            # Substring matching for partial words
            for qw in query_words:
                for kw in kw_set:
                    if len(qw) >= 4 and len(kw) >= 4 and qw != kw:
                        if qw in kw or kw in qw:
                            scores[domain.domain_id] += 0.5

        # Score by entity name mentions
        for entity in self.entity_nodes.values():
            if entity.name.lower() in query_lower:
                for did in entity.domains:
                    scores[did] += 2.0

        # Score by concept mentions
        for concept in self.concept_nodes.values():
            if concept.name in query_words:
                for did in concept.domains:
                    scores[did] += 0.5

        return sorted(
            [(did, s) for did, s in scores.items() if s > 0],
            key=lambda x: x[1],
            reverse=True,
        )

    def to_routing_hints(self) -> dict[str, list[str]]:
        """Generate routing hints: agent_id → [related agent_ids].

        Used by KeywordRouter to boost related agents when one scores high.
        """
        hints: dict[str, list[str]] = {}
        for domain_id in self.domain_nodes:
            related = self.get_related_domains(domain_id)
            if related:
                hints[domain_id] = [r[0] for r in related[:5]]
        return hints

    # ── Summary ──────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            "domains": len(self.domain_nodes),
            "entities": len(self.entity_nodes),
            "shared_entities": sum(
                1 for e in self.entity_nodes.values() if len(e.domains) >= 2
            ),
            "concepts": len(self.concept_nodes),
            "edges": len(self.edges),
            "edge_types": dict(
                __import__("collections").Counter(e.edge_type for e in self.edges)
            ),
        }

    def get_summary(self) -> str:
        """Human-readable summary of the knowledge graph."""
        stats = self.get_stats()
        lines = [
            f"Knowledge Graph: {stats['domains']} domains, "
            f"{stats['entities']} entities ({stats['shared_entities']} shared), "
            f"{stats['edges']} edges",
        ]

        if stats["edge_types"]:
            parts = [f"{k}={v}" for k, v in stats["edge_types"].items()]
            lines.append(f"  Edges: {', '.join(parts)}")

        # Show top cross-domain relationships
        for domain_id in list(self.domain_nodes)[:5]:
            related = self.get_related_domains(domain_id)
            if related:
                top = related[0]
                lines.append(
                    f"  {domain_id} ↔ {top[0]} ({top[1]}, w={top[2]:.1f})"
                )

        return "\n".join(lines)
