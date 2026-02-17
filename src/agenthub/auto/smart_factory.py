from __future__ import annotations
"""Smart agent factory that creates agents based on semantic code analysis.

This factory uses CodebaseDiscovery to understand the codebase and create
agents with meaningful descriptions, keywords, and context.

Supports two modes:
- Dynamic domains (default): Uses import graph analysis to discover natural
  module clusters without hardcoded patterns. Works with any project type.
- Legacy mode: Uses hardcoded patterns like "api", "service", "model".
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agenthub.agents.base import BaseAgent, heuristic_scope_check
from agenthub.auto.config import AutoAgentConfig
from agenthub.auto.discovery import CodebaseDiscovery, ModuleInfo, ProjectProfile
from agenthub.auto.domain_analysis import Domain, DomainAnalysis
from agenthub.auto.dynamic_rnr import DynamicRnRGenerator
from agenthub.auto.import_graph import ImportGraph
from agenthub.context import ContextBuilder
from agenthub.models import AgentSpec, RoutingConfig

if TYPE_CHECKING:
    import anthropic


class SmartCodeAgent(BaseAgent):
    """Auto-generated agent with semantic understanding of its module."""

    # Language hints for code blocks
    LANG_MAP = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "jsx",
        ".json": "json",
        ".sql": "sql",
        ".css": "css",
        ".scss": "scss",
        ".html": "html",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
    }

    def __init__(
        self,
        spec: AgentSpec,
        client: "anthropic.Anthropic",
        root_path: str,
        module_paths: list[str],
    ):
        super().__init__(spec, client)
        self.root_path = root_path
        self.module_paths = module_paths
        self.context_builder = ContextBuilder(root_path)

    def build_context(self) -> str:
        """Build context from the agent's assigned modules."""
        parts = []

        # Read each module file
        for module_path in self.module_paths:
            full_path = Path(self.root_path) / module_path
            if full_path.exists():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    # Determine language for syntax highlighting
                    suffix = full_path.suffix.lower()
                    lang = self.LANG_MAP.get(suffix, "")
                    parts.append(f"### {module_path}\n```{lang}\n{content}\n```")
                except Exception:
                    pass

        # Limit total size
        result = "\n\n".join(parts)
        if len(result) > self.spec.max_context_size:
            result = result[: self.spec.max_context_size] + "\n... [truncated]"

        return result

    def _quick_scope_check(self, query: str) -> dict:
        """Quick check if query is in this agent's scope without LLM call.

        Uses the shared heuristic check (keyword overlap + exclusion list)
        to catch obviously out-of-scope queries before any LLM call.

        Args:
            query: User's query to check.

        Returns:
            Dict with 'in_scope' bool, 'message' str, and optional 'suggested_agent'.
        """
        return heuristic_scope_check(self.spec, query)


class SmartAgentFactory:
    """Creates intelligent agents based on semantic code analysis.

    Unlike the basic AutoAgentFactory, this factory:
    - Uses AST parsing to understand code structure
    - Groups modules by type (API, service, model, etc.)
    - Generates meaningful keywords from code content
    - Creates rich system prompts based on module purpose
    - Generates R&R (Roles & Responsibilities) for each agent

    Example:
        >>> factory = SmartAgentFactory(client, "./my-project")
        >>> agents = factory.create_agents()
        >>> for agent in agents:
        ...     print(f"{agent.spec.agent_id}: {agent.spec.description}")
    """

    # R&R definitions by module type
    ROLES_AND_RESPONSIBILITIES = {
        "api": {
            "role": "API & Endpoints Expert",
            "in_scope": [
                "HTTP endpoint definitions and routing",
                "Request/response handling and validation",
                "API authentication and authorization",
                "REST/GraphQL interface design",
                "API error handling and status codes",
                "Request middleware and interceptors",
            ],
            "out_of_scope": [
                "Business logic implementation (ask service_agent)",
                "Database queries and ORM (ask model_agent or repository_agent)",
                "Configuration and environment setup (ask config_agent)",
                "Utility functions and helpers (ask util_agent)",
            ],
        },
        "service": {
            "role": "Business Logic & Services Expert",
            "in_scope": [
                "Business rules and domain logic",
                "Service layer implementations",
                "Data processing and transformations",
                "Integration between components",
                "Transaction management",
                "Workflow orchestration",
            ],
            "out_of_scope": [
                "API endpoint definitions (ask api_agent)",
                "Database schema and models (ask model_agent)",
                "Raw SQL queries (ask repository_agent)",
                "Configuration files (ask config_agent)",
            ],
        },
        "model": {
            "role": "Data Models & Schema Expert",
            "in_scope": [
                "Database schema definitions",
                "ORM model classes (SQLAlchemy, Django ORM, etc.)",
                "Field types and validations",
                "Model relationships (foreign keys, joins)",
                "Data migrations",
                "Pydantic/dataclass definitions",
            ],
            "out_of_scope": [
                "Business logic using these models (ask service_agent)",
                "API serialization (ask api_agent)",
                "Complex queries (ask repository_agent)",
                "How data flows through the app (ask service_agent)",
            ],
        },
        "repository": {
            "role": "Data Access & Query Expert",
            "in_scope": [
                "Database queries and CRUD operations",
                "Query optimization",
                "Data access patterns",
                "Repository pattern implementations",
                "Raw SQL and query builders",
            ],
            "out_of_scope": [
                "Schema definitions (ask model_agent)",
                "Business rules (ask service_agent)",
                "API layer (ask api_agent)",
            ],
        },
        "util": {
            "role": "Utilities & Helpers Expert",
            "in_scope": [
                "Helper functions and utilities",
                "Common shared code",
                "String/date/number formatting",
                "Validation helpers",
                "Logging utilities",
            ],
            "out_of_scope": [
                "Domain-specific business logic (ask service_agent)",
                "API handling (ask api_agent)",
                "Data models (ask model_agent)",
            ],
        },
        "config": {
            "role": "Configuration & Settings Expert",
            "in_scope": [
                "Environment variables and .env files",
                "Configuration classes and settings",
                "Feature flags",
                "Application initialization",
                "Dependency injection setup",
            ],
            "out_of_scope": [
                "Business logic (ask service_agent)",
                "API routes (ask api_agent)",
                "Database models (ask model_agent)",
            ],
        },
        "test": {
            "role": "Testing Expert",
            "in_scope": [
                "Test files and test cases",
                "Test fixtures and mocks",
                "Testing patterns and strategies",
                "Test configuration",
            ],
            "out_of_scope": [
                "Production code implementation (ask the relevant agent)",
                "How features actually work (ask the relevant agent)",
            ],
        },
        "default": {
            "role": "Code Module Expert",
            "in_scope": [
                "The specific files assigned to this agent",
                "Code structure and patterns used",
                "Functions and classes in these files",
            ],
            "out_of_scope": [
                "Code in other modules (ask the relevant agent)",
            ],
        },
    }

    # System prompt templates by module type (with R&R)
    PROMPT_TEMPLATES = {
        "api": """# ROLE & RESPONSIBILITIES

**You are the {role} for this project.**

## YOUR RESPONSIBILITIES (IN SCOPE)
{in_scope}

## NOT YOUR RESPONSIBILITIES (OUT OF SCOPE)
{out_of_scope}

## IMPORTANT: SCOPE CHECK
Before answering any question:
1. Check if it falls within your responsibilities above
2. If NOT in scope, respond with:
   "This question is outside my area of expertise. I handle API endpoints and HTTP routing.
   For [topic], please ask the [suggested_agent] instead."
3. If IN scope, provide a detailed, accurate answer

---

You know these API endpoints intimately:
{modules}

You can:
- Explain how endpoints work
- Show request/response formats
- Describe authentication requirements
- Suggest improvements

Reference specific routes and handlers in your answers.""",

        "service": """# ROLE & RESPONSIBILITIES

**You are the {role} for this project.**

## YOUR RESPONSIBILITIES (IN SCOPE)
{in_scope}

## NOT YOUR RESPONSIBILITIES (OUT OF SCOPE)
{out_of_scope}

## IMPORTANT: SCOPE CHECK
Before answering any question:
1. Check if it falls within your responsibilities above
2. If NOT in scope, respond with:
   "This question is outside my area of expertise. I handle business logic and services.
   For [topic], please ask the [suggested_agent] instead."
3. If IN scope, provide a detailed, accurate answer

---

You know these services intimately:
{modules}

You can:
- Explain business rules and logic
- Trace data flow through services
- Identify dependencies
- Suggest optimizations

Reference specific functions and classes in your answers.""",

        "model": """# ROLE & RESPONSIBILITIES

**You are the {role} for this project.**

## YOUR RESPONSIBILITIES (IN SCOPE)
{in_scope}

## NOT YOUR RESPONSIBILITIES (OUT OF SCOPE)
{out_of_scope}

## IMPORTANT: SCOPE CHECK
Before answering any question:
1. Check if it falls within your responsibilities above
2. If NOT in scope, respond with:
   "This question is outside my area of expertise. I handle data models and schemas.
   For [topic], please ask the [suggested_agent] instead."
3. If IN scope, provide a detailed, accurate answer

---

You know these data models intimately:
{modules}

You can:
- Explain table structures and relationships
- Describe field validations
- Show how models are used
- Suggest schema improvements

Reference specific models and fields in your answers.""",

        "repository": """# ROLE & RESPONSIBILITIES

**You are the {role} for this project.**

## YOUR RESPONSIBILITIES (IN SCOPE)
{in_scope}

## NOT YOUR RESPONSIBILITIES (OUT OF SCOPE)
{out_of_scope}

## IMPORTANT: SCOPE CHECK
Before answering any question:
1. Check if it falls within your responsibilities above
2. If NOT in scope, respond with:
   "This question is outside my area of expertise. I handle data access and queries.
   For [topic], please ask the [suggested_agent] instead."
3. If IN scope, provide a detailed, accurate answer

---

You know these repositories/data access patterns intimately:
{modules}

You can:
- Explain query patterns and CRUD operations
- Show how data is fetched and stored
- Describe query optimization techniques
- Identify N+1 queries or performance issues

Reference specific queries and methods in your answers.""",

        "util": """# ROLE & RESPONSIBILITIES

**You are the {role} for this project.**

## YOUR RESPONSIBILITIES (IN SCOPE)
{in_scope}

## NOT YOUR RESPONSIBILITIES (OUT OF SCOPE)
{out_of_scope}

## IMPORTANT: SCOPE CHECK
Before answering any question:
1. Check if it falls within your responsibilities above
2. If NOT in scope, respond with:
   "This question is outside my area of expertise. I handle utility functions and helpers.
   For [topic], please ask the [suggested_agent] instead."
3. If IN scope, provide a detailed, accurate answer

---

You know these utilities intimately:
{modules}

You can:
- Explain helper functions and their usage
- Show common patterns and utilities
- Describe formatting and validation helpers

Reference specific utility functions in your answers.""",

        "config": """# ROLE & RESPONSIBILITIES

**You are the {role} for this project.**

## YOUR RESPONSIBILITIES (IN SCOPE)
{in_scope}

## NOT YOUR RESPONSIBILITIES (OUT OF SCOPE)
{out_of_scope}

## IMPORTANT: SCOPE CHECK
Before answering any question:
1. Check if it falls within your responsibilities above
2. If NOT in scope, respond with:
   "This question is outside my area of expertise. I handle configuration and settings.
   For [topic], please ask the [suggested_agent] instead."
3. If IN scope, provide a detailed, accurate answer

---

You know these configuration files intimately:
{modules}

You can:
- Explain configuration options and settings
- Show environment variable usage
- Describe application initialization
- Help with dependency setup

Reference specific config values in your answers.""",

        "test": """# ROLE & RESPONSIBILITIES

**You are the {role} for this project.**

## YOUR RESPONSIBILITIES (IN SCOPE)
{in_scope}

## NOT YOUR RESPONSIBILITIES (OUT OF SCOPE)
{out_of_scope}

## IMPORTANT: SCOPE CHECK
Before answering any question:
1. Check if it falls within your responsibilities above
2. If NOT in scope, respond with:
   "This question is outside my area of expertise. I handle test files and testing patterns.
   For [topic], please ask the [suggested_agent] instead."
3. If IN scope, provide a detailed, accurate answer

---

You know these test files intimately:
{modules}

You can:
- Explain test cases and their purpose
- Show testing patterns and fixtures
- Describe mock setup and test data
- Help write new tests

Reference specific test cases in your answers.""",

        "default": """# ROLE & RESPONSIBILITIES

**You are the {role} for this project.**

## YOUR RESPONSIBILITIES (IN SCOPE)
{in_scope}

## NOT YOUR RESPONSIBILITIES (OUT OF SCOPE)
{out_of_scope}

## IMPORTANT: SCOPE CHECK
Before answering any question:
1. Check if it falls within your responsibilities above
2. If NOT in scope, respond with:
   "This question is outside my area of expertise.
   For [topic], please ask the [suggested_agent] instead."
3. If IN scope, provide a detailed, accurate answer

---

You know these files intimately:
{modules}

You can:
- Explain how the code works
- Show usage patterns
- Identify potential issues
- Suggest improvements

Reference specific code in your answers.""",
    }

    def __init__(
        self,
        client: "anthropic.Anthropic",
        project_root: str,
        config: AutoAgentConfig | None = None,
        use_dynamic_domains: bool = True,
        tier_a_agents: list[AgentSpec] | None = None,
    ):
        """Initialize the smart agent factory.

        Args:
            client: Anthropic API client.
            project_root: Path to the project root.
            config: Optional configuration for auto-agent generation.
            use_dynamic_domains: If True (default), uses import graph analysis
                to discover natural module clusters. If False, uses hardcoded
                patterns like "api", "service", "model".
            tier_a_agents: Optional list of Tier A agent specs to provide
                business context for dynamic domain detection.
        """
        self.client = client
        self.project_root = Path(project_root).resolve()
        self.config = config or AutoAgentConfig()
        self.use_dynamic_domains = use_dynamic_domains
        self.tier_a_agents = tier_a_agents or []
        self.discovery = CodebaseDiscovery(
            str(self.project_root),
            ignore_patterns=self.config.ignore_patterns,
        )
        # Store import graph for later access (e.g., by enable_teams())
        self._import_graph: ImportGraph | None = None

    def create_agents(self) -> list[BaseAgent]:
        """Analyze codebase and create specialized agents.

        Uses dynamic domain detection if use_dynamic_domains is True,
        otherwise falls back to legacy pattern-based detection.

        Returns:
            List of configured agent instances.
        """
        if self.use_dynamic_domains:
            return self._create_agents_dynamic()
        return self._create_agents_legacy()

    def _create_agents_dynamic(self) -> list[BaseAgent]:
        """Create agents using dynamic domain detection.

        Uses import graph analysis to discover natural module clusters
        without relying on hardcoded patterns.
        """
        # Build import graph
        import_graph = ImportGraph(
            str(self.project_root),
            ignore_patterns=self.config.ignore_patterns,
        )
        import_graph.build()

        # Store for later access (e.g., by enable_teams())
        self._import_graph = import_graph

        # Analyze domains
        domain_analysis = DomainAnalysis(
            import_graph=import_graph,
            tier_a_agents=self.tier_a_agents,
            min_cluster_size=2,
            max_domains=self.config.max_agents,
        )
        domains = domain_analysis.analyze()

        # Store domains for knowledge graph building
        self._domains = domains

        # Generate R&R for each domain
        rnr_generator = DynamicRnRGenerator(import_graph, domains)

        # Create agents
        agents = []
        for domain in domains:
            rnr = rnr_generator.generate(domain)
            agent = self._create_agent_for_domain(domain, rnr)
            if agent:
                agents.append(agent)

        return agents

    def _create_agent_for_domain(
        self,
        domain: Domain,
        rnr: "DynamicRnRGenerator",
    ) -> BaseAgent | None:
        """Create an agent from a dynamically discovered domain."""
        from agenthub.auto.dynamic_rnr import DynamicRnR

        if not isinstance(rnr, DynamicRnR):
            return None

        # Generate system prompt
        rnr_generator = DynamicRnRGenerator(
            ImportGraph(str(self.project_root)), [domain]
        )
        system_prompt = rnr_generator.generate_system_prompt(domain, rnr)

        # Calculate context size based on domain size
        # Use domain's total size as a baseline, with some headroom
        domain_size_kb = domain.total_size_bytes / 1024
        # Context should be at least 1.5x the source size to allow for formatting
        # But capped at configured max
        min_context_kb = max(50, domain_size_kb * 1.5)
        context_size_kb = min(
            max(min_context_kb, self.config.max_agent_context_kb),
            self.config.max_agent_context_kb * 2,  # Allow 2x for large domains
        )

        # Build per-agent routing config from R&R analysis
        routing = self._build_routing_config_from_rnr(domain, rnr)

        spec = AgentSpec(
            agent_id=domain.agent_id,
            name=domain.name + " Expert",
            description=domain.description,
            routing=routing,
            context_keywords=domain.keywords,
            context_paths=domain.modules,
            max_context_size=int(context_size_kb * 1024),
            system_prompt=system_prompt,
            metadata={
                "auto_generated": True,
                "tier": "B",
                "root_path": str(self.project_root),
                "domain_name": domain.name,
                "module_count": len(domain.modules),
                "central_module": domain.central_module,
                "related_tier_a": domain.related_tier_a,
                "generated_at": datetime.now().isoformat(),
                "detection_mode": "dynamic",
                "primary_language": domain.primary_language,
                "total_size_bytes": domain.total_size_bytes,
                "total_size_kb": round(domain_size_kb, 2),
                "context_size_kb": round(context_size_kb, 2),
                "rnr": {
                    "role": rnr.role,
                    "in_scope": rnr.in_scope,
                    "out_of_scope": rnr.out_of_scope,
                    "redirect_hints": rnr.redirect_hints,
                },
            },
        )

        return SmartCodeAgent(
            spec=spec,
            client=self.client,
            root_path=str(self.project_root),
            module_paths=domain.modules,
        )

    def _build_routing_config_from_rnr(
        self,
        domain: Domain,
        rnr: "DynamicRnR",
    ) -> RoutingConfig:
        """Build per-agent routing config from R&R analysis.

        Maps semantic analysis into structured routing preferences:
        - in_scope → domains (what this agent handles)
        - out_of_scope → exclusions (what to reject)
        - redirect_hints → fallback_agent_id
        - top domain keywords → keyword_weights (highest-signal terms)

        Args:
            domain: The discovered domain.
            rnr: Roles & Responsibilities data from DynamicRnRGenerator.

        Returns:
            Populated RoutingConfig.
        """
        from agenthub.auto.dynamic_rnr import DynamicRnR

        if not isinstance(rnr, DynamicRnR):
            return RoutingConfig()

        # Domains: role + top in_scope items
        domains = []
        if rnr.role:
            domains.append(rnr.role.lower().split()[0])  # First word of role
        for scope in rnr.in_scope[:3]:
            term = scope.lower().strip()
            if term and term not in domains:
                domains.append(term)

        # Exclusions: out_of_scope items
        exclusions = [s.lower().strip() for s in rnr.out_of_scope[:5] if s.strip()]

        # Keyword weights: top-5 domain keywords get 2.0, rest get 1.0
        keyword_weights = {}
        for i, kw in enumerate(domain.keywords):
            keyword_weights[kw.lower()] = 2.0 if i < 5 else 1.0

        # Fallback: try to resolve redirect hints to an agent ID
        fallback_agent_id = None
        if rnr.redirect_hints:
            # redirect_hints are typically like "Ask the Backend Expert" or
            # "Route to api_agent". Extract likely agent_id or name.
            for hint in rnr.redirect_hints:
                hint_lower = hint.lower().replace(" ", "_")
                # Simple heuristic: if it looks like an agent_id, use it
                if "_" in hint_lower and len(hint_lower) < 40:
                    fallback_agent_id = hint_lower
                    break

        return RoutingConfig(
            keyword_weights=keyword_weights,
            domains=domains,
            exclusions=exclusions,
            priority=0,
            min_confidence=0.3,  # Auto-generated agents use moderate threshold
            fallback_agent_id=fallback_agent_id,
            prefer_exact_match=False,
        )

    def _create_agents_legacy(self) -> list[BaseAgent]:
        """Create agents using legacy pattern-based detection.

        Uses hardcoded patterns like "api", "service", "model" to
        classify modules.
        """
        # Analyze the codebase
        profile = self.discovery.analyze()

        # Group modules by type
        agents = []

        # Create agents by module type
        type_groups = self._group_modules_by_type(profile.modules)

        for module_type, modules in type_groups.items():
            if not modules:
                continue

            agent = self._create_agent_for_type(module_type, modules, profile)
            if agent:
                agents.append(agent)

        return agents

    def _group_modules_by_type(
        self, modules: list[ModuleInfo]
    ) -> dict[str, list[ModuleInfo]]:
        """Group modules by their type."""
        groups: dict[str, list[ModuleInfo]] = {}
        for module in modules:
            groups.setdefault(module.module_type, []).append(module)
        return groups

    def _create_agent_for_type(
        self,
        module_type: str,
        modules: list[ModuleInfo],
        profile: ProjectProfile,
    ) -> BaseAgent | None:
        """Create an agent for a specific module type."""
        if not modules:
            return None

        # Collect all keywords
        all_keywords = set()
        for m in modules:
            all_keywords.update(m.keywords)

        # Add type-specific keywords
        type_keywords = {
            "api": ["api", "endpoint", "route", "request", "response", "http"],
            "service": ["service", "logic", "process", "handle", "execute"],
            "model": ["model", "schema", "table", "field", "database", "entity"],
            "repository": ["repository", "query", "find", "save", "delete"],
            "util": ["util", "helper", "common", "shared"],
            "config": ["config", "setting", "environment", "option"],
        }
        all_keywords.update(type_keywords.get(module_type, []))

        # Generate agent ID
        agent_id = f"{module_type}_agent"

        # Generate name
        name_map = {
            "api": "API Expert",
            "service": "Business Logic Expert",
            "model": "Data Model Expert",
            "repository": "Data Access Expert",
            "util": "Utilities Expert",
            "config": "Configuration Expert",
            "test": "Test Expert",
        }
        name = name_map.get(module_type, f"{module_type.title()} Expert")

        # Generate description
        module_names = [m.name for m in modules[:5]]
        if len(modules) > 5:
            module_names.append(f"+{len(modules) - 5} more")
        description = f"Expert on {module_type} layer: {', '.join(module_names)}"

        # Get R&R definitions
        rnr = self.ROLES_AND_RESPONSIBILITIES.get(
            module_type,
            self.ROLES_AND_RESPONSIBILITIES["default"]
        )

        # Format in-scope as bullet points
        in_scope_formatted = "\n".join(f"- {item}" for item in rnr["in_scope"])

        # Format out-of-scope as bullet points
        out_of_scope_formatted = "\n".join(f"- {item}" for item in rnr["out_of_scope"])

        # Generate system prompt with R&R
        modules_list = "\n".join(
            f"- {m.path}: {m.description}" for m in modules[:10]
        )
        template = self.PROMPT_TEMPLATES.get(module_type, self.PROMPT_TEMPLATES["default"])
        system_prompt = template.format(
            role=rnr["role"],
            in_scope=in_scope_formatted,
            out_of_scope=out_of_scope_formatted,
            modules=modules_list
        )

        # Collect module paths
        module_paths = [str(m.path) for m in modules]

        spec = AgentSpec(
            agent_id=agent_id,
            name=name,
            description=description,
            context_keywords=list(all_keywords)[:30],
            context_paths=module_paths,
            max_context_size=self.config.max_agent_context_kb * 1024,
            system_prompt=system_prompt,
            metadata={
                "auto_generated": True,
                "tier": "B",
                "root_path": str(self.project_root),
                "module_type": module_type,
                "module_count": len(modules),
                "generated_at": datetime.now().isoformat(),
                "detection_mode": "legacy",
                "rnr": {
                    "role": rnr["role"],
                    "in_scope": rnr["in_scope"],
                    "out_of_scope": rnr["out_of_scope"],
                },
            },
        )

        return SmartCodeAgent(
            spec=spec,
            client=self.client,
            root_path=str(self.project_root),
            module_paths=module_paths,
        )

    def get_project_summary(self) -> str:
        """Get a summary of the analyzed project."""
        profile = self.discovery.analyze()

        lines = [
            f"Framework: {profile.framework}",
            f"Type: {profile.project_type}",
            f"Total modules: {len(profile.modules)}",
            "",
        ]

        # Count by type
        type_counts: dict[str, int] = {}
        for m in profile.modules:
            type_counts[m.module_type] = type_counts.get(m.module_type, 0) + 1

        lines.append("Modules by type:")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {t}: {count}")

        if profile.database_models:
            lines.append(f"\nDatabase models: {', '.join(profile.database_models[:10])}")

        if profile.api_endpoints:
            lines.append(f"API endpoints: {len(profile.api_endpoints)}")

        return "\n".join(lines)

    def get_dynamic_summary(self) -> str:
        """Get a summary of dynamic domain analysis.

        Returns:
            Summary string showing discovered domains.
        """
        if not self.use_dynamic_domains:
            return "Dynamic domains not enabled. Set use_dynamic_domains=True."

        # Build import graph
        import_graph = ImportGraph(
            str(self.project_root),
            ignore_patterns=self.config.ignore_patterns,
        )
        import_graph.build()

        # Get graph stats
        stats = import_graph.get_stats()

        # Analyze domains
        domain_analysis = DomainAnalysis(
            import_graph=import_graph,
            tier_a_agents=self.tier_a_agents,
            min_cluster_size=2,
            max_domains=self.config.max_agents,
        )

        lines = [
            "=== Dynamic Domain Analysis ===",
            "",
            f"Total modules: {stats['total_modules']}",
            f"Import edges: {stats['total_edges']}",
            f"Natural clusters: {stats['num_clusters']}",
            f"Hub modules: {stats['hub_modules']}",
            f"Leaf modules: {stats['leaf_modules']}",
            "",
        ]

        lines.append(domain_analysis.get_summary())

        return "\n".join(lines)
