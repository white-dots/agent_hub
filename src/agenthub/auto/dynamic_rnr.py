"""Dynamic R&R (Roles & Responsibilities) generation.

This module generates R&R for Tier B agents dynamically based on
actual code analysis, without relying on hardcoded patterns.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from agenthub.auto.domain_analysis import Domain
from agenthub.auto.import_graph import ImportGraph


@dataclass
class DynamicRnR:
    """Dynamically generated Roles & Responsibilities."""

    role: str  # Role title (e.g., "Authentication & User Access Expert")
    in_scope: list[str]  # What this agent handles
    out_of_scope: list[str]  # What to redirect elsewhere
    redirect_hints: dict[str, str] = field(default_factory=dict)  # topic -> agent_id


class DynamicRnRGenerator:
    """Generates R&R dynamically from import graph analysis.

    Instead of hardcoded rules like "api handles HTTP endpoints",
    this class analyzes actual code patterns to determine what
    each domain is responsible for.

    Example:
        >>> graph = ImportGraph("./project")
        >>> graph.build()
        >>> analysis = DomainAnalysis(graph, [])
        >>> domains = analysis.analyze()
        >>> generator = DynamicRnRGenerator(graph, domains)
        >>> rnr = generator.generate(domains[0])
        >>> print(rnr.role)
    """

    # Common patterns that help identify responsibilities
    # These are hints, not rigid classifications
    RESPONSIBILITY_HINTS = {
        # Function/class name patterns -> responsibility description
        r"(get|fetch|load|read)_": "Data retrieval and loading",
        r"(create|add|insert|new)_": "Creating new records/entities",
        r"(update|modify|edit|change)_": "Updating existing data",
        r"(delete|remove|drop)_": "Deletion and cleanup",
        r"(validate|check|verify)_": "Validation and verification",
        r"(parse|decode|deserialize)_": "Parsing and data transformation",
        r"(encode|serialize|format)_": "Encoding and formatting",
        r"(send|post|publish|emit)_": "Sending data/events",
        r"(receive|subscribe|listen)_": "Receiving data/events",
        r"(auth|login|logout|session)": "Authentication and sessions",
        r"(permission|access|role|acl)": "Authorization and permissions",
        r"(cache|memo|store)": "Caching and storage",
        r"(log|track|metric|trace)": "Logging and metrics",
        r"(config|setting|option|env)": "Configuration management",
        r"(test|mock|stub|fixture)": "Testing utilities",
        r"(route|endpoint|handler|view)": "Request handling",
        r"(model|schema|entity|table)": "Data models and schemas",
        r"(service|manager|controller)": "Business logic orchestration",
        r"(util|helper|common)": "Utility functions",
        r"(error|exception|fault)": "Error handling",
        r"(connect|disconnect|pool)": "Connection management",
        r"(queue|worker|job|task)": "Background job processing",
        r"(notify|alert|email|sms)": "Notifications",
        r"(import|export|migrate)": "Data import/export",
        r"(search|query|filter|find)": "Search and filtering",
        r"(sort|order|rank)": "Sorting and ordering",
        r"(page|paginate|limit|offset)": "Pagination",
        r"(transform|convert|map)": "Data transformation",
    }

    def __init__(self, import_graph: ImportGraph, domains: list[Domain]):
        """Initialize the R&R generator.

        Args:
            import_graph: Built ImportGraph for analyzing connections.
            domains: List of discovered domains.
        """
        self.graph = import_graph
        self.domains = domains

        # Build domain lookup by agent_id
        self.domain_by_id = {d.agent_id: d for d in domains}
        self.domain_by_module: dict[str, Domain] = {}
        for domain in domains:
            for module in domain.modules:
                self.domain_by_module[module] = domain

    def generate(self, domain: Domain) -> DynamicRnR:
        """Generate R&R for a domain.

        Args:
            domain: Domain to generate R&R for.

        Returns:
            DynamicRnR with role, in_scope, out_of_scope, and redirect hints.
        """
        # Generate role title
        role = self._generate_role_title(domain)

        # Infer responsibilities from code
        in_scope = self._infer_in_scope(domain)

        # Find out-of-scope based on what we import but don't own
        out_of_scope, redirects = self._infer_out_of_scope(domain)

        return DynamicRnR(
            role=role,
            in_scope=in_scope,
            out_of_scope=out_of_scope,
            redirect_hints=redirects,
        )

    def _generate_role_title(self, domain: Domain) -> str:
        """Generate a role title for the domain."""
        # Use domain name as base
        name = domain.name

        # Add "Expert" suffix
        if not name.lower().endswith("expert"):
            name = f"{name} Expert"

        return name

    def _infer_in_scope(self, domain: Domain) -> list[str]:
        """Infer what this domain is responsible for.

        Analyzes function names, class names, and file structure
        to determine responsibilities.
        """
        responsibilities: dict[str, int] = {}

        for module_path in domain.modules:
            if module_path not in self.graph.nodes:
                continue

            node = self.graph.nodes[module_path]

            # Analyze function names
            for func in node.functions:
                func_lower = func.lower()
                for pattern, description in self.RESPONSIBILITY_HINTS.items():
                    if re.search(pattern, func_lower):
                        responsibilities[description] = (
                            responsibilities.get(description, 0) + 1
                        )

            # Analyze class names
            for cls in node.classes:
                cls_lower = cls.lower()
                for pattern, description in self.RESPONSIBILITY_HINTS.items():
                    if re.search(pattern, cls_lower):
                        responsibilities[description] = (
                            responsibilities.get(description, 0) + 1
                        )

            # Analyze file/folder names
            path = Path(module_path)
            for part in [path.stem] + list(path.parts[:-1]):
                part_lower = part.lower()
                for pattern, description in self.RESPONSIBILITY_HINTS.items():
                    if re.search(pattern, part_lower):
                        responsibilities[description] = (
                            responsibilities.get(description, 0) + 1
                        )

        # Sort by frequency and convert to list
        sorted_resp = sorted(responsibilities.items(), key=lambda x: x[1], reverse=True)

        in_scope = []

        # Add top responsibilities
        for desc, count in sorted_resp[:6]:
            in_scope.append(desc)

        # Add module-specific items
        file_desc = self._describe_modules(domain.modules[:5])
        for desc in file_desc:
            if desc not in in_scope:
                in_scope.append(desc)

        # Add keyword-based responsibilities
        keyword_resp = self._keywords_to_responsibilities(domain.keywords)
        for resp in keyword_resp[:3]:
            if resp not in in_scope:
                in_scope.append(resp)

        # Ensure we have at least some responsibilities
        if not in_scope:
            in_scope = [
                f"Code in {domain.folder_prefix or 'this module group'}",
                "Functions and classes defined in these files",
                "Understanding how these modules work together",
            ]

        return in_scope[:8]  # Limit to 8 items

    def _describe_modules(self, modules: list[str]) -> list[str]:
        """Generate descriptions based on module names."""
        descriptions = []

        for module_path in modules:
            stem = Path(module_path).stem
            if stem == "__init__":
                continue

            # Convert snake_case to readable description
            words = stem.replace("_", " ").replace("-", " ")
            desc = f"The {words} functionality"
            descriptions.append(desc)

        return descriptions

    def _keywords_to_responsibilities(self, keywords: list[str]) -> list[str]:
        """Convert domain keywords to responsibility descriptions."""
        responsibilities = []

        for keyword in keywords:
            # Skip common/meaningless keywords
            if keyword.lower() in {
                "init",
                "main",
                "base",
                "core",
                "utils",
                "helpers",
                "common",
            }:
                continue

            # Check if keyword matches any pattern
            for pattern, description in self.RESPONSIBILITY_HINTS.items():
                if re.search(pattern, keyword.lower()):
                    if description not in responsibilities:
                        responsibilities.append(description)
                    break
            else:
                # No pattern match, create generic responsibility
                resp = f"Handling {keyword}-related operations"
                if resp not in responsibilities:
                    responsibilities.append(resp)

        return responsibilities

    def _infer_out_of_scope(self, domain: Domain) -> tuple[list[str], dict[str, str]]:
        """Infer what this domain should NOT handle.

        Looks at what other domains the modules import but don't own,
        and suggests redirecting those questions.
        """
        out_of_scope = []
        redirects: dict[str, str] = {}

        # Find all modules this domain imports
        imported_modules: set[str] = set()
        for module_path in domain.modules:
            if module_path in self.graph.nodes:
                node = self.graph.nodes[module_path]
                imported_modules.update(node.imports)

        # Find which domains own those modules
        external_domains: dict[str, set[str]] = {}  # domain_id -> imported_modules

        for imported_module in imported_modules:
            # Skip if it's in our domain
            if imported_module in domain.modules:
                continue

            # Find which domain owns this module
            if imported_module in self.domain_by_module:
                owner_domain = self.domain_by_module[imported_module]
                if owner_domain.agent_id != domain.agent_id:
                    if owner_domain.agent_id not in external_domains:
                        external_domains[owner_domain.agent_id] = set()
                    external_domains[owner_domain.agent_id].add(imported_module)

        # Generate out-of-scope items based on external dependencies
        for other_agent_id, modules in external_domains.items():
            other_domain = self.domain_by_id.get(other_agent_id)
            if not other_domain:
                continue

            # Create out-of-scope description
            topic = other_domain.name.lower()
            desc = f"{other_domain.name} implementation details"
            out_of_scope.append(desc)

            # Add redirect hint
            redirects[topic] = other_agent_id

        # Add generic out-of-scope items if we don't have enough
        if len(out_of_scope) < 2:
            generic_out = [
                "Code in modules not listed in my scope",
                "Implementation details of external dependencies",
            ]
            for item in generic_out:
                if item not in out_of_scope:
                    out_of_scope.append(item)

        return out_of_scope[:6], redirects

    def generate_system_prompt(self, domain: Domain, rnr: DynamicRnR) -> str:
        """Generate a complete system prompt with R&R.

        Args:
            domain: Domain for the agent.
            rnr: Generated R&R.

        Returns:
            System prompt string.
        """
        # Format in-scope as bullet points
        in_scope_str = "\n".join(f"- {item}" for item in rnr.in_scope)

        # Format out-of-scope with redirect hints
        out_scope_items = []
        for item in rnr.out_of_scope:
            # Check if we have a redirect for this
            redirect_agent = None
            for topic, agent_id in rnr.redirect_hints.items():
                if topic.lower() in item.lower():
                    redirect_agent = agent_id
                    break

            if redirect_agent:
                out_scope_items.append(f"- {item} (ask {redirect_agent})")
            else:
                out_scope_items.append(f"- {item}")

        out_of_scope_str = "\n".join(out_scope_items)

        # Format modules list
        modules_str = "\n".join(f"- {m}" for m in domain.modules[:15])
        if len(domain.modules) > 15:
            modules_str += f"\n- ... and {len(domain.modules) - 15} more files"

        prompt = f"""# ROLE & RESPONSIBILITIES

**You are the {rnr.role} for this project.**

## YOUR RESPONSIBILITIES (IN SCOPE)
{in_scope_str}

## NOT YOUR RESPONSIBILITIES (OUT OF SCOPE)
{out_of_scope_str}

## IMPORTANT: SCOPE CHECK
Before answering any question:
1. Check if it falls within your responsibilities above
2. If NOT in scope, respond with:
   "This question is outside my area of expertise. I handle {domain.name.lower()}.
   For [topic], please ask the [suggested_agent] instead."
3. If IN scope, provide a detailed, accurate answer

---

## YOUR MODULE FILES
{modules_str}

## KEYWORDS
{', '.join(domain.keywords[:10])}

---

You can:
- Explain how the code in your modules works
- Answer questions about functions and classes in your scope
- Show usage patterns and examples
- Suggest improvements within your area

Reference specific code from your modules in your answers."""

        return prompt

    def generate_all(self) -> dict[str, DynamicRnR]:
        """Generate R&R for all domains.

        Returns:
            Dict mapping agent_id to DynamicRnR.
        """
        result = {}
        for domain in self.domains:
            result[domain.agent_id] = self.generate(domain)
        return result
