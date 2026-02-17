"""Tests for CodebaseKnowledgeGraph auto-generation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agenthub.auto.knowledge_graph import (
    CodebaseKnowledgeGraph,
    ConceptNode,
    DomainNode,
    EntityNode,
    KGEdge,
)


def _make_domain(agent_id: str, name: str, modules: list[str], keywords: list[str]):
    """Create a mock Domain object."""
    d = MagicMock()
    d.agent_id = agent_id
    d.name = name
    d.modules = modules
    d.keywords = keywords
    return d


def _make_module_node(path, classes=None, functions=None, imports=None, imported_by=None):
    """Create a mock ModuleNode."""
    node = MagicMock()
    node.classes = classes or []
    node.functions = functions or []
    node.imports = imports or []
    node.imported_by = imported_by or []
    return node


def _make_import_edge(source, target, imported_names=None):
    """Create a mock ImportEdge."""
    edge = MagicMock()
    edge.source = source
    edge.target = target
    edge.imported_names = imported_names or []
    return edge


def _make_graph(nodes: dict, edges: list):
    """Create a mock ImportGraph."""
    graph = MagicMock()
    graph.nodes = nodes
    graph.edges = edges
    return graph


class TestCodebaseKnowledgeGraph:
    """Tests for the KG builder."""

    def _build_simple_kg(self):
        """Build a simple KG with 2 domains sharing an entity."""
        nodes = {
            "services/pricing.py": _make_module_node(
                "services/pricing.py",
                classes=["Product", "PricingService"],
                functions=["get_best_price", "calculate_discount"],
            ),
            "services/analytics.py": _make_module_node(
                "services/analytics.py",
                classes=["Product", "AnalyticsReport"],
                functions=["generate_report", "get_sales_trend"],
            ),
            "api/pricing_api.py": _make_module_node(
                "api/pricing_api.py",
                classes=["PricingRouter"],
                functions=["update_price"],
                imports=["services/pricing.py"],
            ),
        }
        edges = [
            _make_import_edge("api/pricing_api.py", "services/pricing.py", ["PricingService"]),
        ]
        graph = _make_graph(nodes, edges)

        domains = [
            _make_domain(
                "pricing_agent", "Pricing",
                ["services/pricing.py", "api/pricing_api.py"],
                ["pricing", "price", "discount", "sales", "revenue"],
            ),
            _make_domain(
                "analytics_agent", "Analytics",
                ["services/analytics.py"],
                ["analytics", "report", "sales", "trend", "revenue"],
            ),
        ]

        kg = CodebaseKnowledgeGraph(graph, domains)
        kg.build()
        return kg

    def test_build_creates_domain_nodes(self):
        kg = self._build_simple_kg()
        assert "pricing_agent" in kg.domain_nodes
        assert "analytics_agent" in kg.domain_nodes
        assert kg.domain_nodes["pricing_agent"].name == "Pricing"

    def test_build_creates_entity_nodes(self):
        kg = self._build_simple_kg()
        assert "Product" in kg.entity_nodes
        assert "PricingService" in kg.entity_nodes
        assert "AnalyticsReport" in kg.entity_nodes

    def test_shared_entity_detected(self):
        kg = self._build_simple_kg()
        product = kg.entity_nodes["Product"]
        assert len(product.domains) == 2
        assert "pricing_agent" in product.domains
        assert "analytics_agent" in product.domains

    def test_shared_entity_edges(self):
        kg = self._build_simple_kg()
        shared_edges = [e for e in kg.edges if e.edge_type == "SHARES_ENTITY"]
        assert len(shared_edges) >= 1
        # Should have edge between pricing and analytics via Product
        entity_details = [e.detail for e in shared_edges]
        assert "Product" in entity_details

    def test_domain_overlap_edges(self):
        """Domains sharing 'sales' keyword should have DOMAIN_OVERLAP edge."""
        kg = self._build_simple_kg()
        overlap_edges = [e for e in kg.edges if e.edge_type == "DOMAIN_OVERLAP"]
        # pricing and analytics both have "sales" keyword
        assert len(overlap_edges) >= 1

    def test_get_related_domains(self):
        kg = self._build_simple_kg()
        related = kg.get_related_domains("pricing_agent")
        assert len(related) >= 1
        related_ids = [r[0] for r in related]
        assert "analytics_agent" in related_ids

    def test_get_shared_entities(self):
        kg = self._build_simple_kg()
        shared = kg.get_shared_entities("pricing_agent", "analytics_agent")
        assert "Product" in shared

    def test_query_relevant_domains(self):
        kg = self._build_simple_kg()
        results = kg.query_relevant_domains("what is the product pricing strategy?")
        assert len(results) >= 1
        # pricing_agent should score highest
        assert results[0][0] == "pricing_agent"

    def test_to_routing_hints(self):
        kg = self._build_simple_kg()
        hints = kg.to_routing_hints()
        assert "pricing_agent" in hints
        assert "analytics_agent" in hints["pricing_agent"]

    def test_concept_nodes_extracted(self):
        kg = self._build_simple_kg()
        # "calculate" and "discount" should be extracted from function names
        concept_names = set(kg.concept_nodes.keys())
        assert "discount" in concept_names or "calculate" in concept_names

    def test_get_stats(self):
        kg = self._build_simple_kg()
        stats = kg.get_stats()
        assert stats["domains"] == 2
        assert stats["entities"] >= 3
        assert stats["edges"] >= 1

    def test_get_summary_returns_string(self):
        kg = self._build_simple_kg()
        summary = kg.get_summary()
        assert isinstance(summary, str)
        assert "Knowledge Graph" in summary

    def test_empty_graph(self):
        """KG with no domains should build without errors."""
        graph = _make_graph({}, [])
        kg = CodebaseKnowledgeGraph(graph, [])
        kg.build()
        assert kg.get_stats()["domains"] == 0
        assert kg.get_stats()["edges"] == 0

    def test_single_domain_no_overlap(self):
        """Single domain should have no cross-domain edges."""
        nodes = {
            "services/auth.py": _make_module_node(
                "services/auth.py", classes=["AuthService"], functions=["login"],
            ),
        }
        graph = _make_graph(nodes, [])
        domains = [
            _make_domain("auth_agent", "Auth", ["services/auth.py"], ["auth", "login"]),
        ]
        kg = CodebaseKnowledgeGraph(graph, domains)
        kg.build()
        assert kg.get_stats()["domains"] == 1
        shared = [e for e in kg.edges if e.edge_type == "SHARES_ENTITY"]
        assert len(shared) == 0

    def test_cross_domain_import_edge(self):
        """Import from one domain to another should create CALLS_INTO edge."""
        nodes = {
            "services/pricing.py": _make_module_node(
                "services/pricing.py", classes=["PricingService"],
            ),
            "api/ads.py": _make_module_node(
                "api/ads.py", classes=["AdsRouter"],
                imports=["services/pricing.py"],
            ),
        }
        edges = [
            _make_import_edge("api/ads.py", "services/pricing.py", ["PricingService"]),
        ]
        graph = _make_graph(nodes, edges)
        domains = [
            _make_domain("pricing_agent", "Pricing", ["services/pricing.py"], ["pricing"]),
            _make_domain("ads_agent", "Ads", ["api/ads.py"], ["ads"]),
        ]
        kg = CodebaseKnowledgeGraph(graph, domains)
        kg.build()

        call_edges = [e for e in kg.edges if e.edge_type == "CALLS_INTO"]
        assert len(call_edges) >= 1
        assert call_edges[0].source == "ads_agent"
        assert call_edges[0].target == "pricing_agent"
