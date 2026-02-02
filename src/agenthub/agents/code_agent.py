"""Code agent for codebase navigation and explanation."""

from typing import TYPE_CHECKING

from agenthub.agents.base import BaseAgent
from agenthub.context import ContextBuilder
from agenthub.models import AgentSpec

if TYPE_CHECKING:
    import anthropic


class CodeAgent(BaseAgent):
    """Agent specialized in codebase navigation and explanation.

    This agent knows the codebase structure and can:
    - Explain how code works
    - Suggest improvements
    - Write new code matching project style
    - Debug issues
    - Review code

    Example:
        >>> agent = CodeAgent(client, project_root="./my-project")
        >>> hub.register(agent)
        >>> response = hub.run("How does the authentication work?")
    """

    def __init__(
        self,
        client: "anthropic.Anthropic",
        project_root: str = ".",
        agent_id: str = "code_agent",
        include_patterns: list[str] | None = None,
        max_context_size: int = 40000,
    ):
        """Initialize CodeAgent.

        Args:
            client: Anthropic client for API calls.
            project_root: Root directory of the project.
            agent_id: Unique identifier for this agent.
            include_patterns: Glob patterns for files to include in context.
            max_context_size: Maximum characters for context.
        """
        spec = AgentSpec(
            agent_id=agent_id,
            name="Code Expert",
            description="Knows the codebase structure, patterns, and style",
            context_keywords=[
                "code",
                "function",
                "class",
                "implement",
                "refactor",
                "bug",
                "error",
                "fix",
                "write",
                "create",
                "method",
                "module",
            ],
            context_paths=include_patterns or ["src/**/*.py", "**/*.py"],
            max_context_size=max_context_size,
            system_prompt="""You are a code expert for this project.
You know the codebase intimately and can:
- Explain how code works
- Suggest improvements
- Write new code matching project style
- Debug issues
- Review code

Always reference specific files and line numbers when relevant.
Be concise but thorough in your explanations.""",
        )
        super().__init__(spec, client)
        self.project_root = project_root
        self.include_patterns = include_patterns or ["src/**/*.py", "**/*.py"]
        self.max_context_size = max_context_size
        self.context_builder = ContextBuilder(project_root)

    def build_context(self) -> str:
        """Build context from project files."""
        parts: list[str] = []

        # Directory structure
        parts.append("## Project Structure\n```")
        parts.append(self.context_builder.read_directory_structure())
        parts.append("```")

        # Key source files
        parts.append("\n## Source Code")
        parts.append(
            self.context_builder.read_files(
                patterns=self.include_patterns,
                max_size=self.max_context_size,
            )
        )

        return "\n".join(parts)


class DBAgent(BaseAgent):
    """Agent specialized in database operations.

    This agent knows the database schema and can:
    - Write efficient SQL queries
    - Explain table relationships
    - Optimize query performance
    - Design schema changes
    """

    def __init__(
        self,
        client: "anthropic.Anthropic",
        connection_string: str | None = None,
        schema_file: str | None = None,
        agent_id: str = "db_agent",
    ):
        """Initialize DBAgent.

        Args:
            client: Anthropic client for API calls.
            connection_string: Database connection string for schema extraction.
            schema_file: Path to a file containing schema documentation.
            agent_id: Unique identifier for this agent.
        """
        spec = AgentSpec(
            agent_id=agent_id,
            name="Database Expert",
            description="Knows database schemas and can write queries",
            context_keywords=[
                "database",
                "query",
                "sql",
                "table",
                "schema",
                "select",
                "insert",
                "update",
                "join",
                "postgresql",
                "mysql",
                "index",
            ],
            system_prompt="""You are a database expert.
You know the database schema intimately and can:
- Write efficient SQL queries
- Explain table relationships
- Optimize query performance
- Design schema changes

Always consider indexes and query plans.
Suggest improvements when you see potential issues.""",
        )
        super().__init__(spec, client)
        self.connection_string = connection_string
        self.schema_file = schema_file

    def build_context(self) -> str:
        """Build context from database schema."""
        parts: list[str] = []

        # Try to load from schema file first
        if self.schema_file:
            try:
                from pathlib import Path

                schema_path = Path(self.schema_file)
                if schema_path.exists():
                    parts.append("## Database Schema")
                    parts.append(schema_path.read_text(encoding="utf-8"))
            except Exception as e:
                parts.append(f"# Error reading schema file: {e}")

        # Try to extract from database
        elif self.connection_string:
            from agenthub.context import SQLContext

            sql_context = SQLContext(self.connection_string)
            parts.append("## Database Schema")
            parts.append(sql_context.build())

        else:
            parts.append("# No database schema configured")

        return "\n\n".join(parts)


class APIAgent(BaseAgent):
    """Agent specialized in API design and integration.

    This agent knows the API structure and can:
    - Design RESTful endpoints
    - Handle authentication flows
    - Debug API issues
    - Write API client code
    """

    def __init__(
        self,
        client: "anthropic.Anthropic",
        project_root: str = ".",
        agent_id: str = "api_agent",
        api_patterns: list[str] | None = None,
    ):
        """Initialize APIAgent.

        Args:
            client: Anthropic client for API calls.
            project_root: Root directory of the project.
            agent_id: Unique identifier for this agent.
            api_patterns: Glob patterns for API-related files.
        """
        spec = AgentSpec(
            agent_id=agent_id,
            name="API Expert",
            description="Knows API endpoints, authentication, and integration patterns",
            context_keywords=[
                "api",
                "endpoint",
                "rest",
                "request",
                "response",
                "authentication",
                "oauth",
                "http",
                "webhook",
                "route",
            ],
            context_paths=api_patterns or ["api/**/*.py", "routes/**/*.py"],
            system_prompt="""You are an API expert.
You know the API structure and can:
- Design RESTful endpoints
- Handle authentication flows
- Debug API issues
- Write API client code

Always consider error handling and rate limiting.
Follow REST best practices.""",
        )
        super().__init__(spec, client)
        self.project_root = project_root
        self.api_patterns = api_patterns or [
            "api/**/*.py",
            "routes/**/*.py",
            "**/endpoints.py",
            "**/views.py",
        ]
        self.context_builder = ContextBuilder(project_root)

    def build_context(self) -> str:
        """Build context from API files."""
        parts: list[str] = []

        # API routes
        parts.append("## API Endpoints")
        parts.append(
            self.context_builder.read_files(
                patterns=self.api_patterns,
                max_size=30000,
            )
        )

        # OpenAPI spec if exists
        try:
            from pathlib import Path

            for spec_name in ["openapi.yaml", "openapi.json", "swagger.yaml", "swagger.json"]:
                openapi_path = Path(self.project_root) / spec_name
                if openapi_path.exists():
                    parts.append("\n## OpenAPI Specification")
                    content = openapi_path.read_text(encoding="utf-8")
                    parts.append(content[:10000])  # Limit size
                    break
        except Exception:
            pass

        return "\n".join(parts)
