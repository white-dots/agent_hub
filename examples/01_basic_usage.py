"""Basic AgentHub usage example.

This example shows how to:
1. Create an AgentHub instance
2. Register a custom agent
3. Run queries through the hub
"""

import anthropic

from agenthub import AgentHub, AgentSpec, BaseAgent, ContextBuilder


# Define a custom agent
class MyCodeAgent(BaseAgent):
    """A simple code agent that knows about a project."""

    def __init__(self, client: anthropic.Anthropic, project_root: str = "."):
        spec = AgentSpec(
            agent_id="my_code_agent",
            name="My Code Expert",
            description="Knows about the codebase structure and can help with code questions",
            context_keywords=["code", "function", "class", "file", "module", "implement"],
            system_prompt="""You are a helpful code assistant.
You know the project structure and can answer questions about the code.
Be concise and reference specific files when relevant.""",
        )
        super().__init__(spec, client)
        self.project_root = project_root
        self.context_builder = ContextBuilder(project_root)

    def build_context(self) -> str:
        """Build context from project files."""
        parts = []

        # Add directory structure
        parts.append("## Project Structure\n```")
        parts.append(self.context_builder.read_directory_structure(max_depth=3))
        parts.append("```")

        # Add source files (limit to 30KB)
        parts.append("\n## Source Files")
        parts.append(
            self.context_builder.read_files(
                patterns=["**/*.py"],
                max_size=30000,
            )
        )

        return "\n".join(parts)


def main():
    # Initialize Anthropic client
    client = anthropic.Anthropic()

    # Create the hub
    hub = AgentHub(client=client)

    # Register our agent
    agent = MyCodeAgent(client, project_root=".")
    hub.register(agent)

    print(f"Registered agents: {[a.agent_id for a in hub.list_agents()]}")

    # Run a query
    print("\nAsking: 'What files are in this project?'")
    response = hub.run("What files are in this project?")
    print(f"\nResponse:\n{response.content}")
    print(f"\nTokens used: {response.tokens_used}")

    # Continue the conversation
    print("\n" + "=" * 50)
    print("Asking follow-up: 'Can you explain the main structure?'")
    response = hub.run(
        "Can you explain the main structure?",
        session_id=response.session_id,  # Continue same session
    )
    print(f"\nResponse:\n{response.content}")


if __name__ == "__main__":
    main()
