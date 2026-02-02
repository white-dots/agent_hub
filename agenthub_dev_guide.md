# AgentHub: A Lightweight Agent Orchestration Library

> **Author:** John  
> **Status:** Development Spec v0.1  
> **Type:** Open Source Python Library (PyPI)  
> **Goal:** Solve coding agent context limits through specialized agent orchestration

---

## 0. Library Vision

### What This Is

**AgentHub** is a Python library that lets developers build and orchestrate specialized AI agents that maintain focused context. Instead of one monolithic agent processing everything, users compose lightweight agents that know their domain.

### Target Users

1. **Developers building AI-powered tools** - Who need agents with domain expertise
2. **Teams with large codebases** - Who hit context limits with coding assistants
3. **Data engineers & scientists** - Who want agents that know their schemas/pipelines

### Design Philosophy

```
Simple things should be simple. Complex things should be possible.
```

- **5-minute quickstart** - Install, create agent, run query
- **No framework lock-in** - Works with or without LangChain/LangGraph
- **Batteries included, but removable** - Sensible defaults, everything configurable
- **Type-safe** - Full Pydantic models, IDE autocomplete works

### Public API Surface (What Users Import)

```python
from agenthub import AgentHub, Agent, AgentSpec
from agenthub.context import ContextBuilder, FileContext, SQLContext
from agenthub.routing import KeywordRouter, LLMRouter
from agenthub.session import Session, InMemoryStore, FileStore

# That's it. Everything else is internal.
```

### Comparable Libraries

| Library | Focus | AgentHub Differentiator |
|---------|-------|------------------------|
| LangChain | General LLM orchestration | Simpler, context-focused |
| CrewAI | Multi-agent collaboration | Lighter weight, no roles/tasks abstraction |
| AutoGen | Conversational agents | Single-purpose agents, not chat |
| Semantic Kernel | Enterprise AI | Python-native, minimal dependencies |

---

## 1. Problem Statement

When developing with coding agents (like Claude Opus), the ~200k context window gets consumed by the entire codebase. This leads to:

- **Context overflow** on large projects
- **Slow responses** due to processing irrelevant code
- **Higher costs** from unnecessary token usage
- **Degraded quality** as the model loses focus

### Solution

Build specialized agents that **already know where to look**. Instead of one agent processing everything, route queries to domain experts with focused, cached context.

```
┌─────────────────────────────────────────────────────────┐
│                      User Query                         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                     AgentHub                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Intent Classifier (rule-based → LLM upgrade)     │  │
│  └───────────────────────┬───────────────────────────┘  │
└──────────────────────────┼──────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Code Agent  │   │  DB Agent   │   │  API Agent  │
│             │   │             │   │             │
│ - knows src/│   │ - knows     │   │ - knows     │
│ - patterns  │   │   schemas   │   │   endpoints │
│ - style     │   │ - queries   │   │ - auth      │
└─────────────┘   └─────────────┘   └─────────────┘
```

---

## 2. Architecture Overview

### 2.1 Core Components

| Component | Responsibility | File |
|-----------|---------------|------|
| **AgentHub** | Routes queries, manages sessions | `hub.py` |
| **AgentSpec** | Defines agent capabilities & requirements | `models.py` |
| **BaseAgent** | Abstract agent interface | `agents/base.py` |
| **Session** | Conversation state & history | `session.py` |
| **ContextStore** | Cached context per agent | `context.py` |

### 2.2 Design Principles

1. **No frameworks initially** - Pure Python, add LangGraph/etc later if needed
2. **Pydantic everywhere** - Type safety from day one
3. **Agents are functions** - Start simple, upgrade to classes when needed
4. **Context is king** - Each agent maintains focused, relevant context

---

## 3. Data Models

### 3.1 Core Models (`models.py`)

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal, Any
from datetime import datetime
from enum import Enum

class AgentCapability(str, Enum):
    """What an agent can do"""
    CODE_READ = "code_read"
    CODE_WRITE = "code_write"
    DB_QUERY = "db_query"
    DB_WRITE = "db_write"
    API_CALL = "api_call"
    FILE_SYSTEM = "file_system"
    WEB_SEARCH = "web_search"

class AgentSpec(BaseModel):
    """Registration spec for an agent"""
    agent_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="What this agent does")
    capabilities: list[AgentCapability] = Field(default_factory=list)
    
    # Context management
    context_paths: list[str] = Field(
        default_factory=list,
        description="File/folder paths this agent knows about"
    )
    context_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that should route to this agent"
    )
    
    # Resource hints
    estimated_tokens: int = Field(default=2000, description="Typical token usage")
    max_context_size: int = Field(default=50000, description="Max context this agent uses")
    
    # Behavior
    system_prompt: str = Field(default="", description="Agent's system prompt")
    temperature: float = Field(default=0.7, ge=0, le=1)


class Message(BaseModel):
    """A single message in conversation"""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """Conversation session with an agent"""
    session_id: str
    agent_id: str
    messages: list[Message] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Token tracking
    total_tokens_used: int = 0
    

class AgentResponse(BaseModel):
    """Standardized agent response"""
    content: str
    agent_id: str
    session_id: str
    tokens_used: int = 0
    artifacts: list["Artifact"] = Field(default_factory=list)
    needs_followup: bool = False
    suggested_agent: Optional[str] = None  # For agent handoff


class Artifact(BaseModel):
    """Structured output from agent"""
    artifact_type: Literal["code", "file", "sql", "json", "markdown"]
    content: str
    filename: Optional[str] = None
    language: Optional[str] = None
    description: str = ""
```

### 3.2 Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings
from typing import Optional

class AgentHubConfig(BaseSettings):
    """Configuration via environment variables"""
    
    # API Keys
    anthropic_api_key: str
    openai_api_key: Optional[str] = None  # For embeddings if needed
    
    # Defaults
    default_model: str = "claude-sonnet-4-20250514"
    opus_model: str = "claude-opus-4-0-20250115"
    
    # Limits
    max_tokens_per_session: int = 100000
    max_context_per_agent: int = 50000
    
    # Storage
    session_storage_path: str = "./sessions"
    context_cache_path: str = "./context_cache"
    
    class Config:
        env_file = ".env"
        env_prefix = "AGENTHUB_"
```

---

## 4. Core Implementation

### 4.1 Base Agent (`agents/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Optional
import anthropic
from models import AgentSpec, Message, AgentResponse, Session

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, spec: AgentSpec, client: anthropic.Anthropic):
        self.spec = spec
        self.client = client
        self._context_cache: Optional[str] = None
    
    @abstractmethod
    def build_context(self) -> str:
        """Build the agent's specialized context.
        Override this to load relevant files, schemas, etc.
        """
        pass
    
    def get_context(self, force_refresh: bool = False) -> str:
        """Get cached context or rebuild"""
        if self._context_cache is None or force_refresh:
            self._context_cache = self.build_context()
        return self._context_cache
    
    def run(
        self, 
        query: str, 
        session: Session,
        model: Optional[str] = None
    ) -> AgentResponse:
        """Execute the agent on a query"""
        
        # Build messages
        messages = self._build_messages(query, session)
        
        # Call API
        response = self.client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            max_tokens=4096,
            system=self._build_system_prompt(),
            messages=messages,
            temperature=self.spec.temperature
        )
        
        # Parse response
        content = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        
        return AgentResponse(
            content=content,
            agent_id=self.spec.agent_id,
            session_id=session.session_id,
            tokens_used=tokens_used,
            artifacts=self._extract_artifacts(content)
        )
    
    def _build_system_prompt(self) -> str:
        """Combine spec prompt with context"""
        context = self.get_context()
        return f"""{self.spec.system_prompt}

## Your Specialized Context

{context}

## Instructions

- Focus only on your domain of expertise
- If a query is outside your scope, say so clearly
- Reference the context above when answering
- Be concise but thorough
"""
    
    def _build_messages(self, query: str, session: Session) -> list[dict]:
        """Build message list from session history + new query"""
        messages = []
        
        # Add history (limit to avoid context overflow)
        for msg in session.messages[-10:]:  # Last 10 messages
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Add new query
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages
    
    def _extract_artifacts(self, content: str) -> list:
        """Extract code blocks and other artifacts from response.
        Override for custom artifact extraction.
        """
        # Basic implementation - extract code blocks
        import re
        artifacts = []
        
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for lang, code in matches:
            from models import Artifact
            artifacts.append(Artifact(
                artifact_type="code",
                content=code.strip(),
                language=lang or "text"
            ))
        
        return artifacts
```

### 4.2 AgentHub (`hub.py`)

```python
from typing import Optional, Callable
import uuid
from models import AgentSpec, Session, Message, AgentResponse
from agents.base import BaseAgent
import anthropic

class AgentHub:
    """Central orchestrator for agent routing and management"""
    
    def __init__(self, client: Optional[anthropic.Anthropic] = None):
        self.client = client or anthropic.Anthropic()
        self._agents: dict[str, BaseAgent] = {}
        self._sessions: dict[str, Session] = {}
        self._router: Optional[Callable] = None
    
    # ==================== Agent Registration ====================
    
    def register(self, agent: BaseAgent) -> None:
        """Register an agent with the hub"""
        self._agents[agent.spec.agent_id] = agent
        print(f"✓ Registered agent: {agent.spec.agent_id}")
    
    def unregister(self, agent_id: str) -> None:
        """Remove an agent from the hub"""
        if agent_id in self._agents:
            del self._agents[agent_id]
    
    def list_agents(self) -> list[AgentSpec]:
        """List all registered agent specs"""
        return [a.spec for a in self._agents.values()]
    
    # ==================== Session Management ====================
    
    def create_session(self, agent_id: Optional[str] = None) -> Session:
        """Create a new conversation session"""
        session = Session(
            session_id=str(uuid.uuid4()),
            agent_id=agent_id or "router"
        )
        self._sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve an existing session"""
        return self._sessions.get(session_id)
    
    # ==================== Routing ====================
    
    def set_router(self, router_fn: Callable[[str, list[AgentSpec]], str]) -> None:
        """Set custom routing function.
        
        Args:
            router_fn: Function that takes (query, agent_specs) -> agent_id
        """
        self._router = router_fn
    
    def _default_router(self, query: str) -> str:
        """Simple keyword-based routing"""
        query_lower = query.lower()
        
        for agent_id, agent in self._agents.items():
            # Check keywords
            for keyword in agent.spec.context_keywords:
                if keyword.lower() in query_lower:
                    return agent_id
        
        # Default to first agent or raise
        if self._agents:
            return list(self._agents.keys())[0]
        raise ValueError("No agents registered")
    
    def route(self, query: str) -> str:
        """Determine which agent should handle a query"""
        if self._router:
            return self._router(query, self.list_agents())
        return self._default_router(query)
    
    # ==================== Execution ====================
    
    def run(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        model: Optional[str] = None
    ) -> AgentResponse:
        """Execute a query through the hub.
        
        Args:
            query: User's query
            session_id: Existing session ID (creates new if None)
            agent_id: Force specific agent (auto-routes if None)
            model: Override model for this call
        
        Returns:
            AgentResponse with content and metadata
        """
        # Get or create session
        if session_id:
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
        else:
            session = self.create_session()
        
        # Route to agent
        target_agent_id = agent_id or self.route(query)
        
        if target_agent_id not in self._agents:
            raise ValueError(f"Agent {target_agent_id} not registered")
        
        agent = self._agents[target_agent_id]
        
        # Update session
        session.agent_id = target_agent_id
        session.messages.append(Message(role="user", content=query))
        
        # Execute
        response = agent.run(query, session, model=model)
        
        # Store response in session
        session.messages.append(Message(
            role="assistant", 
            content=response.content,
            metadata={"tokens": response.tokens_used}
        ))
        session.total_tokens_used += response.tokens_used
        
        return response
    
    # ==================== Utilities ====================
    
    def refresh_all_contexts(self) -> None:
        """Force all agents to rebuild their context caches"""
        for agent in self._agents.values():
            agent.get_context(force_refresh=True)
        print(f"✓ Refreshed context for {len(self._agents)} agents")
```

### 4.3 Context Builder (`context.py`)

```python
from pathlib import Path
from typing import Optional
import os

class ContextBuilder:
    """Utility for building agent context from files"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
    
    def read_files(
        self, 
        patterns: list[str],
        max_size: int = 50000
    ) -> str:
        """Read files matching patterns into context string.
        
        Args:
            patterns: Glob patterns like ["src/**/*.py", "*.md"]
            max_size: Maximum total characters
        
        Returns:
            Formatted context string
        """
        content_parts = []
        total_size = 0
        
        for pattern in patterns:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    try:
                        text = file_path.read_text(encoding='utf-8')
                        
                        # Check size limit
                        if total_size + len(text) > max_size:
                            remaining = max_size - total_size
                            if remaining > 1000:  # Worth including partial
                                text = text[:remaining] + "\n... [truncated]"
                            else:
                                continue
                        
                        relative_path = file_path.relative_to(self.base_path)
                        content_parts.append(f"### {relative_path}\n```\n{text}\n```")
                        total_size += len(text)
                        
                    except Exception as e:
                        print(f"Warning: Could not read {file_path}: {e}")
        
        return "\n\n".join(content_parts)
    
    def read_directory_structure(
        self, 
        path: str = ".",
        max_depth: int = 3,
        ignore: list[str] = None
    ) -> str:
        """Generate directory tree for context.
        
        Args:
            path: Starting directory
            max_depth: How deep to traverse
            ignore: Patterns to ignore
        
        Returns:
            Tree-formatted string
        """
        ignore = ignore or [
            "__pycache__", ".git", "node_modules", 
            ".venv", "venv", ".pytest_cache", "*.pyc"
        ]
        
        def should_ignore(name: str) -> bool:
            import fnmatch
            return any(fnmatch.fnmatch(name, pattern) for pattern in ignore)
        
        def build_tree(dir_path: Path, prefix: str = "", depth: int = 0) -> list[str]:
            if depth > max_depth:
                return []
            
            lines = []
            try:
                entries = sorted(dir_path.iterdir(), key=lambda e: (e.is_file(), e.name))
                entries = [e for e in entries if not should_ignore(e.name)]
                
                for i, entry in enumerate(entries):
                    is_last = i == len(entries) - 1
                    connector = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{connector}{entry.name}")
                    
                    if entry.is_dir():
                        extension = "    " if is_last else "│   "
                        lines.extend(build_tree(
                            entry, prefix + extension, depth + 1
                        ))
            except PermissionError:
                pass
            
            return lines
        
        tree_lines = build_tree(self.base_path / path)
        return "\n".join(tree_lines)
    
    def read_sql_schemas(self, connection_string: str, tables: list[str] = None) -> str:
        """Extract database schema for context.
        
        Note: Requires psycopg2 or sqlalchemy
        """
        # Implementation depends on your DB setup
        # This is a placeholder for PostgreSQL
        try:
            import psycopg2
            
            conn = psycopg2.connect(connection_string)
            cursor = conn.cursor()
            
            # Get table schemas
            if tables:
                table_filter = f"AND table_name IN ({','.join(['%s']*len(tables))})"
                params = tables
            else:
                table_filter = ""
                params = []
            
            cursor.execute(f"""
                SELECT table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' {table_filter}
                ORDER BY table_name, ordinal_position
            """, params)
            
            schemas = {}
            for table, column, dtype, nullable in cursor.fetchall():
                if table not in schemas:
                    schemas[table] = []
                schemas[table].append(f"  {column}: {dtype} {'NULL' if nullable == 'YES' else 'NOT NULL'}")
            
            result = []
            for table, columns in schemas.items():
                result.append(f"Table: {table}\n" + "\n".join(columns))
            
            conn.close()
            return "\n\n".join(result)
            
        except ImportError:
            return "# Database schema extraction requires psycopg2"
        except Exception as e:
            return f"# Error extracting schema: {e}"
```

---

## 5. Example Agents

### 5.1 Code Agent (`agents/code_agent.py`)

```python
from agents.base import BaseAgent
from models import AgentSpec
from context import ContextBuilder

class CodeAgent(BaseAgent):
    """Agent specialized in your codebase"""
    
    def __init__(self, client, project_root: str = "."):
        spec = AgentSpec(
            agent_id="code_agent",
            name="Code Expert",
            description="Knows the codebase structure, patterns, and style",
            context_keywords=[
                "code", "function", "class", "implement", "refactor",
                "bug", "error", "fix", "write", "create"
            ],
            context_paths=[
                "src/**/*.py",
                "tests/**/*.py"
            ],
            system_prompt="""You are a code expert for this project.
You know the codebase intimately and can:
- Explain how code works
- Suggest improvements
- Write new code matching project style
- Debug issues
- Review code

Always reference specific files and line numbers when relevant."""
        )
        super().__init__(spec, client)
        self.project_root = project_root
        self.context_builder = ContextBuilder(project_root)
    
    def build_context(self) -> str:
        """Build context from project files"""
        parts = []
        
        # Directory structure
        parts.append("## Project Structure\n```")
        parts.append(self.context_builder.read_directory_structure())
        parts.append("```")
        
        # Key source files
        parts.append("\n## Source Code")
        parts.append(self.context_builder.read_files(
            patterns=["src/**/*.py", "*.py"],
            max_size=40000
        ))
        
        return "\n".join(parts)


class DBAgent(BaseAgent):
    """Agent specialized in database operations"""
    
    def __init__(self, client, connection_string: str):
        spec = AgentSpec(
            agent_id="db_agent",
            name="Database Expert",
            description="Knows database schemas and can write queries",
            context_keywords=[
                "database", "query", "sql", "table", "schema",
                "select", "insert", "update", "join", "postgresql"
            ],
            system_prompt="""You are a database expert.
You know the database schema intimately and can:
- Write efficient SQL queries
- Explain table relationships
- Optimize query performance
- Design schema changes

Always consider indexes and query plans."""
        )
        super().__init__(spec, client)
        self.connection_string = connection_string
        self.context_builder = ContextBuilder()
    
    def build_context(self) -> str:
        """Build context from database schema"""
        schema = self.context_builder.read_sql_schemas(
            self.connection_string
        )
        return f"## Database Schema\n\n{schema}"


class APIAgent(BaseAgent):
    """Agent specialized in API design and integration"""
    
    def __init__(self, client, project_root: str = "."):
        spec = AgentSpec(
            agent_id="api_agent",
            name="API Expert",
            description="Knows API endpoints, authentication, and integration patterns",
            context_keywords=[
                "api", "endpoint", "rest", "request", "response",
                "authentication", "oauth", "http", "webhook"
            ],
            context_paths=[
                "api/**/*.py",
                "routes/**/*.py"
            ],
            system_prompt="""You are an API expert.
You know the API structure and can:
- Design RESTful endpoints
- Handle authentication flows
- Debug API issues
- Write API client code

Always consider error handling and rate limiting."""
        )
        super().__init__(spec, client)
        self.project_root = project_root
        self.context_builder = ContextBuilder(project_root)
    
    def build_context(self) -> str:
        """Build context from API files"""
        parts = []
        
        # API routes
        parts.append("## API Endpoints")
        parts.append(self.context_builder.read_files(
            patterns=["api/**/*.py", "routes/**/*.py", "**/endpoints.py"],
            max_size=30000
        ))
        
        # OpenAPI spec if exists
        try:
            openapi_path = self.context_builder.base_path / "openapi.yaml"
            if openapi_path.exists():
                parts.append("\n## OpenAPI Specification")
                parts.append(openapi_path.read_text()[:10000])
        except:
            pass
        
        return "\n".join(parts)
```

### 5.2 Smartstore Agent (For Your Project)

```python
class SmartstoreAgent(BaseAgent):
    """Agent specialized in Naver Smartstore operations"""
    
    def __init__(self, client, project_root: str = "."):
        spec = AgentSpec(
            agent_id="smartstore_agent",
            name="Smartstore Expert",
            description="Knows Naver Smartstore API, ad data, and e-commerce patterns",
            context_keywords=[
                "smartstore", "naver", "ad", "광고", "스마트스토어",
                "product", "상품", "order", "주문", "campaign"
            ],
            system_prompt="""You are an expert in Naver Smartstore development.
You know:
- Smartstore API structure and authentication
- Ad campaign management
- Product and order data pipelines
- Korean e-commerce patterns

Always reference the specific API endpoints and data formats."""
        )
        super().__init__(spec, client)
        self.project_root = project_root
        self.context_builder = ContextBuilder(project_root)
    
    def build_context(self) -> str:
        """Build context from smartstore project"""
        parts = []
        
        # Project docs
        parts.append("## Project Documentation")
        parts.append(self.context_builder.read_files(
            patterns=["*.md", "docs/**/*.md"],
            max_size=10000
        ))
        
        # API integration code
        parts.append("\n## Smartstore Integration Code")
        parts.append(self.context_builder.read_files(
            patterns=[
                "src/**/smartstore*.py",
                "src/**/naver*.py",
                "src/**/ad*.py"
            ],
            max_size=30000
        ))
        
        return "\n".join(parts)
```

---

## 6. LLM-Based Router (Upgrade Path)

When keyword routing isn't enough, upgrade to LLM-based routing:

```python
def create_llm_router(client: anthropic.Anthropic):
    """Create an LLM-based router function"""
    
    def llm_router(query: str, agents: list[AgentSpec]) -> str:
        # Build agent descriptions
        agent_info = "\n".join([
            f"- {a.agent_id}: {a.description}"
            for a in agents
        ])
        
        response = client.messages.create(
            model="claude-haiku-3-5-20241022",  # Fast & cheap for routing
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"""Given this query, which agent should handle it?

Query: {query}

Available agents:
{agent_info}

Respond with ONLY the agent_id, nothing else."""
            }]
        )
        
        agent_id = response.content[0].text.strip()
        
        # Validate
        valid_ids = [a.agent_id for a in agents]
        if agent_id not in valid_ids:
            return valid_ids[0]  # Fallback
        
        return agent_id
    
    return llm_router


# Usage:
# hub.set_router(create_llm_router(client))
```

---

## 7. Usage Examples

### 7.1 Basic Setup

```python
import anthropic
from hub import AgentHub
from agents.code_agent import CodeAgent, DBAgent

# Initialize
client = anthropic.Anthropic()
hub = AgentHub(client)

# Register agents
hub.register(CodeAgent(client, project_root="./my_project"))
hub.register(DBAgent(client, connection_string="postgresql://..."))

# Run queries
response = hub.run("How does the user authentication work?")
print(response.content)

# Continue conversation
response = hub.run(
    "Can you add rate limiting to that?",
    session_id=response.session_id
)
```

### 7.2 With Session Management

```python
# Create a dedicated session
session = hub.create_session(agent_id="code_agent")

# Multiple turns
r1 = hub.run("Show me the main entry point", session_id=session.session_id)
r2 = hub.run("What patterns does it use?", session_id=session.session_id)
r3 = hub.run("How can I add logging?", session_id=session.session_id)

# Check token usage
print(f"Total tokens: {session.total_tokens_used}")
```

### 7.3 Force Specific Agent

```python
# Override routing
response = hub.run(
    "What's the database schema for users?",
    agent_id="db_agent"  # Force DB agent even if query mentions code
)
```

---

## 8. Project Structure

```
agenthub/
├── pyproject.toml
├── .env.example
├── README.md
│
├── src/
│   └── agenthub/
│       ├── __init__.py
│       ├── hub.py           # AgentHub class
│       ├── models.py        # Pydantic models
│       ├── config.py        # Configuration
│       ├── context.py       # Context builders
│       │
│       └── agents/
│           ├── __init__.py
│           ├── base.py      # BaseAgent
│           ├── code_agent.py
│           ├── db_agent.py
│           └── api_agent.py
│
├── tests/
│   ├── __init__.py
│   ├── test_hub.py
│   ├── test_agents.py
│   └── test_routing.py
│
└── examples/
    ├── basic_usage.py
    └── smartstore_setup.py
```

---

## 9. Development Phases

### Phase 0: Foundation (Week 1) ✅ Start Here

**Goal:** Get something working end-to-end

- [ ] Create project structure
- [ ] Implement `models.py` with Pydantic models
- [ ] Implement `BaseAgent` with simple context
- [ ] Implement `AgentHub` with keyword routing
- [ ] Create ONE working agent (CodeAgent)
- [ ] Test with real queries

**Success Criteria:** Can route a query to an agent and get a response

### Phase 1: Context Management (Week 2)

**Goal:** Make agents actually useful

- [ ] Implement `ContextBuilder`
- [ ] Add file reading with size limits
- [ ] Add context caching
- [ ] Add context refresh mechanism
- [ ] Test with your smartstore_ai project

**Success Criteria:** Agent responses reference actual code

### Phase 2: Session & State (Week 3)

**Goal:** Enable multi-turn conversations

- [ ] Implement session persistence (start with JSON files)
- [ ] Add message history management
- [ ] Add token tracking
- [ ] Implement session limits

**Success Criteria:** Can continue conversations across runs

### Phase 3: Advanced Routing (Week 4)

**Goal:** Smart agent selection

- [ ] Implement LLM-based router
- [ ] Add confidence scoring
- [ ] Handle ambiguous queries
- [ ] Add agent handoff suggestions

**Success Criteria:** Queries reliably go to right agent

### Phase 4: Production Ready (Week 5+)

**Goal:** Ready for real use

- [ ] Add proper error handling
- [ ] Add retry logic
- [ ] Add observability (logging, tracing)
- [ ] Add cost tracking
- [ ] Write tests
- [ ] Create CLI interface

---

## 10. Configuration Files

### 10.1 `pyproject.toml`

```toml
[project]
name = "agenthub"
version = "0.1.0"
description = "Lightweight agent orchestration for context-efficient LLM applications"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
db = ["psycopg2-binary>=2.9"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### 10.2 `.env.example`

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional
AGENTHUB_DEFAULT_MODEL=claude-sonnet-4-20250514
AGENTHUB_MAX_TOKENS_PER_SESSION=100000
AGENTHUB_SESSION_STORAGE_PATH=./sessions
AGENTHUB_CONTEXT_CACHE_PATH=./context_cache

# For DB Agent
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
```

---

## 11. Quick Start Commands

```bash
# Create project
mkdir agenthub && cd agenthub
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install anthropic pydantic pydantic-settings python-dotenv

# Create structure
mkdir -p src/agenthub/agents tests examples

# Start coding!
code .  # Open VS Code
```

---

## 12. Testing Strategy

```python
# tests/test_hub.py
import pytest
from agenthub.hub import AgentHub
from agenthub.models import AgentSpec

class MockAgent:
    def __init__(self, agent_id: str):
        self.spec = AgentSpec(
            agent_id=agent_id,
            name="Mock",
            description="Test agent",
            context_keywords=["test", "mock"]
        )
    
    def run(self, query, session, model=None):
        from agenthub.models import AgentResponse
        return AgentResponse(
            content=f"Mock response for: {query}",
            agent_id=self.spec.agent_id,
            session_id=session.session_id
        )
    
    def get_context(self, force_refresh=False):
        return "Mock context"

def test_agent_registration():
    hub = AgentHub(client=None)
    hub.register(MockAgent("test_agent"))
    assert "test_agent" in [a.agent_id for a in hub.list_agents()]

def test_keyword_routing():
    hub = AgentHub(client=None)
    hub.register(MockAgent("test_agent"))
    
    agent_id = hub.route("this is a test query")
    assert agent_id == "test_agent"

def test_session_creation():
    hub = AgentHub(client=None)
    session = hub.create_session()
    assert session.session_id is not None
```

---

## 13. Library API Design

### 13.1 Public vs Internal

```
agenthub/
├── __init__.py          # Public API exports
├── _internal/           # Private implementation (underscore = don't import)
│   ├── routing.py
│   └── context_utils.py
├── hub.py               # Public: AgentHub
├── agent.py             # Public: Agent, AgentSpec
├── context.py           # Public: ContextBuilder
├── session.py           # Public: Session, stores
└── routing.py           # Public: Router classes
```

### 13.2 `__init__.py` - The Public Contract

```python
"""
AgentHub: Lightweight agent orchestration for context-efficient LLM applications.

Basic usage:
    >>> from agenthub import AgentHub, Agent
    >>> hub = AgentHub()
    >>> hub.register(my_agent)
    >>> response = hub.run("How does authentication work?")
"""

from agenthub.hub import AgentHub
from agenthub.agent import Agent, AgentSpec, AgentResponse
from agenthub.context import ContextBuilder
from agenthub.session import Session
from agenthub.routing import KeywordRouter, LLMRouter

__version__ = "0.1.0"
__all__ = [
    "AgentHub",
    "Agent", 
    "AgentSpec",
    "AgentResponse",
    "ContextBuilder",
    "Session",
    "KeywordRouter",
    "LLMRouter",
]
```

### 13.3 Extension Points (How Users Customize)

Users should be able to extend without modifying library code:

```python
# 1. Custom Agents - Inherit from Agent
class MyCustomAgent(Agent):
    def build_context(self) -> str:
        return "My specialized context"

# 2. Custom Routers - Implement RouterProtocol
class MyRouter:
    def route(self, query: str, agents: list[AgentSpec]) -> str:
        return "agent_id"

hub.set_router(MyRouter())

# 3. Custom Session Stores - Implement SessionStore protocol
class RedisStore:
    def get(self, session_id: str) -> Session | None: ...
    def save(self, session: Session) -> None: ...
    def delete(self, session_id: str) -> None: ...

hub = AgentHub(session_store=RedisStore())

# 4. Custom LLM Clients - Any client with messages.create()
hub = AgentHub(client=my_custom_client)
```

### 13.4 Protocols (Duck Typing for Flexibility)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMClient(Protocol):
    """Any object with this method works as a client"""
    def messages(self) -> "MessagesAPI": ...

@runtime_checkable  
class Router(Protocol):
    """Any object with this method works as a router"""
    def route(self, query: str, agents: list[AgentSpec]) -> str: ...

@runtime_checkable
class SessionStore(Protocol):
    """Any object with these methods works as a store"""
    def get(self, session_id: str) -> Session | None: ...
    def save(self, session: Session) -> None: ...
```

---

## 14. Documentation Strategy

### 14.1 Documentation Tiers

| Tier | Content | Location |
|------|---------|----------|
| **README** | Install, 5-min quickstart, badges | `README.md` |
| **Docstrings** | Every public class/method | In code |
| **Examples** | Common use cases | `examples/` |
| **API Reference** | Auto-generated from docstrings | `docs/api/` |
| **Guides** | Tutorials, concepts | `docs/guides/` |

### 14.2 README Template

```markdown
# 🤖 AgentHub

[![PyPI version](https://badge.fury.io/py/agenthub.svg)](https://pypi.org/project/agenthub/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Lightweight agent orchestration for context-efficient LLM applications.**

Stop hitting context limits. Build specialized agents that know their domain.

## Installation

```bash
pip install agenthub
```

## Quick Start: Auto-Agents for Existing Codebases

Point AgentHub at your existing project and get intelligent code agents instantly:

```python
from agenthub import AgentHub
from agenthub.auto import enable_auto_agents

# Point at your existing codebase
hub = AgentHub()
agents = enable_auto_agents(hub, "./my-django-project")

print(f"Created {len(agents)} agents automatically")
# Created 8 agents automatically

# Now query your codebase
response = hub.run("How does user authentication work?")
print(response.content)

response = hub.run("What does the OrderSerializer do?")
print(response.content)
```

## Two-Tier Agent System

| Tier | Created By | Purpose |
|------|-----------|---------|
| **A: Business** | You (manually) | Domain expertise, APIs, business logic |
| **B: Code** | AgentHub (auto) | Codebase navigation, file explanations |

```python
from agenthub import AgentHub, Agent, AgentSpec
from agenthub.auto import enable_auto_agents

hub = AgentHub()

# Tier A: Your custom business agent
class PricingAgent(Agent):
    spec = AgentSpec(
        agent_id="pricing",
        name="Pricing Expert",
        description="Knows pricing strategy and margins",
        keywords=["price", "margin", "discount"]
    )
    
    def build_context(self) -> str:
        return "... your pricing documentation ..."

hub.register(PricingAgent())

# Tier B: Auto-generated code agents
enable_auto_agents(hub, "./my-project")

# Both work together
hub.run("What's our margin on premium products?")  # → PricingAgent
hub.run("Show me the pricing calculation code")    # → Auto code agent
```

## Why AgentHub?

| Problem | Solution |
|---------|----------|
| Context overflow on large codebases | Auto-generated agents with focused context |
| Manual agent setup is tedious | Point at folder, agents created automatically |
| High token costs | Only load relevant code per query |

## Documentation

- [Getting Started](docs/getting-started.md)
- [Auto-Agents Guide](docs/auto-agents.md)
- [Creating Custom Agents](docs/creating-agents.md)
- [API Reference](docs/api/)

## License

MIT
```

### 14.3 Docstring Standard (Google Style)

```python
class AgentHub:
    """Central orchestrator for agent routing and management.
    
    AgentHub manages a collection of specialized agents and routes
    queries to the appropriate agent based on content analysis.
    
    Args:
        client: LLM client instance. Defaults to Anthropic client.
        session_store: Storage backend for sessions. Defaults to in-memory.
        router: Query routing strategy. Defaults to keyword matching.
    
    Example:
        >>> hub = AgentHub()
        >>> hub.register(MyAgent())
        >>> response = hub.run("How does X work?")
        >>> print(response.content)
    
    Attributes:
        agents: Dict of registered agents by ID.
        sessions: Active session storage.
    """
```

---

## 15. Publishing to PyPI

### 15.1 Updated `pyproject.toml`

```toml
[project]
name = "agenthub"
version = "0.1.0"
description = "Lightweight agent orchestration for context-efficient LLM applications"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "John", email = "your@email.com"}
]
keywords = ["ai", "agents", "llm", "orchestration", "anthropic", "claude"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0"]
db = ["psycopg2-binary>=2.9"]
all = ["agenthub[openai,db]"]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "ruff>=0.5",
    "mypy>=1.0",
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.24",
]

[project.urls]
Homepage = "https://github.com/yourusername/agenthub"
Documentation = "https://agenthub.readthedocs.io"
Repository = "https://github.com/yourusername/agenthub"
Issues = "https://github.com/yourusername/agenthub/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agenthub"]

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "W", "D"]

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=agenthub --cov-report=term-missing"
```

### 15.2 Publishing Workflow

```bash
# 1. Build
pip install build
python -m build

# 2. Test on TestPyPI first
pip install twine
twine upload --repository testpypi dist/*

# 3. Test install
pip install --index-url https://test.pypi.org/simple/ agenthub

# 4. Publish to real PyPI
twine upload dist/*
```

### 15.3 GitHub Actions CI/CD

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*
```

---

## 16. Versioning Strategy

### Semantic Versioning

```
MAJOR.MINOR.PATCH

0.1.0 - Initial release (alpha)
0.2.0 - Add new feature (still alpha, breaking changes OK)
0.9.0 - Feature complete, seeking feedback
1.0.0 - Stable API, breaking changes = major bump
1.1.0 - New features, backwards compatible
1.1.1 - Bug fixes only
```

### What's Public API (Don't Break After 1.0)

- Class names: `AgentHub`, `Agent`, `AgentSpec`
- Method signatures: `hub.run()`, `hub.register()`
- Model fields: `AgentSpec.agent_id`, `response.content`

### What's Internal (Can Change Anytime)

- Anything in `_internal/`
- Private methods (`_build_messages`)
- Implementation details

---

## 17. Updated Project Structure (Library-Ready)

```
agenthub/
├── .github/
│   └── workflows/
│       ├── ci.yml           # Tests on PR
│       └── publish.yml      # PyPI on release
│
├── src/
│   └── agenthub/
│       ├── __init__.py      # Public API
│       ├── py.typed         # PEP 561 marker
│       ├── hub.py
│       ├── agent.py
│       ├── context.py
│       ├── session.py
│       ├── routing.py
│       │
│       ├── auto/            # Auto-agent submodule
│       │   ├── __init__.py  # from agenthub.auto import ...
│       │   ├── analyzer.py  # CodebaseAnalyzer
│       │   ├── factory.py   # AutoAgentFactory  
│       │   ├── manager.py   # AutoAgentManager
│       │   ├── config.py    # AutoAgentConfig, Presets
│       │   └── watcher.py   # File watcher (optional)
│       │
│       └── _internal/       # Private implementation
│           └── utils.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared fixtures
│   ├── test_hub.py
│   ├── test_agent.py
│   ├── test_routing.py
│   └── test_auto/           # Auto-agent tests
│       ├── test_analyzer.py
│       └── test_manager.py
│
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── creating-agents.md
│   ├── auto-agents.md       # Auto-agent documentation
│   └── api/
│       └── reference.md
│
├── examples/
│   ├── 01_basic_usage.py
│   ├── 02_custom_agents.py
│   ├── 03_auto_agents.py    # Point at existing codebase
│   └── 04_full_setup.py     # Tier A + Tier B together
│
├── .gitignore
├── .env.example
├── LICENSE                   # MIT
├── README.md
├── pyproject.toml
└── CHANGELOG.md
```

### 17.1 Auto Submodule Public API

```python
# src/agenthub/auto/__init__.py
"""
Auto-agent generation for existing codebases.

Usage:
    >>> from agenthub.auto import enable_auto_agents, AutoAgentConfig
    >>> hub = AgentHub()
    >>> enable_auto_agents(hub, "./my-project")
"""

from agenthub.auto.config import AutoAgentConfig, Presets
from agenthub.auto.manager import AutoAgentManager
from agenthub.auto.analyzer import CodebaseAnalyzer

# Convenience function
def enable_auto_agents(
    hub: "AgentHub",
    project_root: str,
    config: AutoAgentConfig = None
) -> list[str]:
    """Enable auto-agents for an existing codebase.
    
    Args:
        hub: AgentHub instance to register agents with
        project_root: Path to existing codebase
        config: Optional configuration (uses defaults if None)
    
    Returns:
        List of auto-generated agent IDs
        
    Example:
        >>> from agenthub import AgentHub
        >>> from agenthub.auto import enable_auto_agents
        >>> 
        >>> hub = AgentHub()
        >>> agents = enable_auto_agents(hub, "./my-django-project")
        >>> print(f"Created {len(agents)} agents")
    """
    manager = AutoAgentManager(hub, project_root, config)
    return manager.scan_and_register()

__all__ = [
    "enable_auto_agents",
    "AutoAgentConfig", 
    "Presets",
    "AutoAgentManager",
    "CodebaseAnalyzer",
]
```

---

## 19. Notes & Gotchas

### Common Pitfalls

1. **Context overflow** - Always set max_size limits on context builders
2. **Token counting** - Use `anthropic` token counting, don't estimate
3. **Session leaks** - Clean up old sessions periodically
4. **API rate limits** - Add retry logic with exponential backoff

### Performance Tips

1. **Cache aggressively** - Context rarely changes, cache it
2. **Use Haiku for routing** - Fast and cheap
3. **Batch context reads** - Don't read files one by one
4. **Lazy load contexts** - Only build when first needed

### Library-Specific Considerations

1. **Don't import heavy dependencies at top level** - Lazy import optional deps
2. **Keep core dependencies minimal** - Only `anthropic` and `pydantic` required
3. **Test on multiple Python versions** - 3.11, 3.12, 3.13
4. **Type hints are documentation** - Users rely on autocomplete

### Your Specific Considerations

Since you're working on smartstore_ai:
- Keep Naver API docs in a dedicated agent's context
- Consider a separate agent for Korean language queries
- Your Airflow experience maps well to agent orchestration patterns

---

## 20. Development Phases (Updated for Library)

### Phase 0: Foundation (Week 1) ✅ Start Here

**Goal:** Get core hub working end-to-end

- [ ] Create project structure with `src/` layout
- [ ] Implement `models.py` with Pydantic models
- [ ] Implement `Agent` base class with simple context
- [ ] Implement `AgentHub` with keyword routing
- [ ] Create ONE working agent (CodeAgent)
- [ ] Test with real queries

**Success Criteria:** Can route a query to an agent and get a response

### Phase 1: Auto-Agents Core (Week 2) ⭐ Key Differentiator

**Goal:** Auto-generate agents from existing codebases

- [ ] Implement `CodebaseAnalyzer` with size/count thresholds
- [ ] Implement `AutoAgentFactory` 
- [ ] Implement `AutoAgentManager`
- [ ] Create `enable_auto_agents()` convenience function
- [ ] Test on your smartstore_ai project

**Success Criteria:** Point at a folder, agents created automatically

### Phase 2: Clean API (Week 3)

**Goal:** Design the public interface

- [ ] Define `__init__.py` exports for core and auto
- [ ] Add Protocols for extensibility
- [ ] Write docstrings for all public classes
- [ ] Create numbered examples (01_basic, 02_custom, 03_auto)

**Success Criteria:** A new user can understand the API from examples

### Phase 3: Testing & Docs (Week 4)

**Goal:** Library quality

- [ ] Write unit tests (>80% coverage)
- [ ] Add integration tests for auto-agents
- [ ] Set up GitHub Actions CI
- [ ] Write README and getting-started guide
- [ ] Write auto-agents guide

**Success Criteria:** Tests pass, docs render, CI green

### Phase 4: Alpha Release (Week 5)

**Goal:** Get it on PyPI

- [ ] Review API one more time
- [ ] Add CHANGELOG.md
- [ ] Publish to TestPyPI
- [ ] Test install in fresh environment
- [ ] Publish to PyPI as 0.1.0

**Success Criteria:** `pip install agenthub` works, auto-agents work on any codebase

### Phase 5: Iterate (Ongoing)

- [ ] Gather feedback from users
- [ ] Add LLM-based router
- [ ] Add file watcher for live refresh
- [ ] Add configuration presets (small/medium/large/monorepo)
- [ ] Consider async support
- [ ] Work toward 1.0 stable

---

## 21. Next Steps After Reading This

### For Library Development:

1. **Create GitHub repo** - `agenthub` (check name availability on PyPI first!)
2. **Set up project structure** with `src/` layout (10 min)
3. **Copy core code** - models, hub, base agent (10 min)
4. **Write `__init__.py`** with clean exports (5 min)
5. **Create one example** that works (15 min)
6. **Push to GitHub** with MIT license

### Quick Check - Is Name Available?

```bash
# Check PyPI
pip index versions agenthub  # Should return "not found"

# Check GitHub
# Visit github.com/yourusername/agenthub
```

### Alternative Names (if taken):

- `agenthub-py`
- `agent-orchestrator`
- `contexthub`
- `llm-agents`

---

*Last updated: 2026-02-01*
