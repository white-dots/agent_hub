"""Auto-agents example.

This example shows how to use the auto-agent feature to automatically
create agents for an existing codebase.
"""

import anthropic

from agenthub import AgentHub
from agenthub.auto import (
    AutoAgentConfig,
    Presets,
    enable_auto_agents,
    print_coverage_map,
)


def main():
    # Initialize
    client = anthropic.Anthropic()
    hub = AgentHub(client=client)

    # Point at your existing codebase
    # For this example, we'll analyze the agenthub source code itself
    project_root = "./src"

    print("=" * 60)
    print("Auto-Agent Generation Demo")
    print("=" * 60)

    # Option 1: Use default configuration
    print("\n1. Using default configuration...")
    agents = enable_auto_agents(hub, project_root)
    print(f"   Created {len(agents)} agents automatically")

    # List the created agents
    print("\n   Auto-generated agents:")
    for agent in hub.list_agents(tier="B"):
        print(f"   - {agent.agent_id}: {agent.description}")

    # Option 2: Use a preset (uncomment to try)
    # print("\n2. Using large project preset...")
    # hub2 = AgentHub(client=client)
    # agents2 = enable_auto_agents(
    #     hub2,
    #     project_root,
    #     config=Presets.large_project()
    # )

    # Option 3: Custom configuration (uncomment to try)
    # print("\n3. Using custom configuration...")
    # hub3 = AgentHub(client=client)
    # custom_config = AutoAgentConfig(
    #     min_folder_size_kb=20,
    #     min_files_per_folder=2,
    #     max_depth=4,
    #     ignore_patterns=["__pycache__", ".git", "tests"],
    # )
    # agents3 = enable_auto_agents(hub3, project_root, config=custom_config)

    # Show coverage report
    print("\n" + "-" * 60)
    print_coverage_map(hub, project_root)

    # Now query the codebase
    print("\n" + "=" * 60)
    print("Querying the codebase...")
    print("=" * 60)

    if agents:
        queries = [
            "What modules are in this project?",
            "How does the routing work?",
            "What is the ContextBuilder class?",
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            try:
                response = hub.run(query)
                print(f"Agent: {response.agent_id}")
                print(f"Response: {response.content[:300]}...")
            except Exception as e:
                print(f"Error: {e}")

    # Demonstrate refresh after code changes
    print("\n" + "=" * 60)
    print("Refreshing agents (e.g., after code changes)...")
    added, removed = hub.refresh_auto_agents()
    print(f"Added: {len(added)}, Removed: {len(removed)}")


def demo_two_tier_system():
    """Demonstrate using both Tier A and Tier B agents together."""
    from agenthub import AgentSpec, BaseAgent

    client = anthropic.Anthropic()
    hub = AgentHub(client=client)

    # Tier A: Create a custom business agent
    class BusinessAgent(BaseAgent):
        def __init__(self, client):
            spec = AgentSpec(
                agent_id="business_agent",
                name="Business Expert",
                description="Knows about business rules and requirements",
                context_keywords=["business", "requirement", "feature", "user story"],
            )
            super().__init__(spec, client)

        def build_context(self) -> str:
            return "## Business Context\nThis is a sample business context."

    hub.register(BusinessAgent(client))
    print("Tier A agent registered: business_agent")

    # Tier B: Auto-generate code agents
    agents = enable_auto_agents(hub, "./src")
    print(f"Tier B agents registered: {len(agents)} agents")

    # List all agents by tier
    print("\nAgents by tier:")
    print(f"  Tier A (Business): {len(hub.list_agents(tier='A'))}")
    print(f"  Tier B (Code): {len(hub.list_agents(tier='B'))}")

    # Routing works across both tiers
    # Business queries go to Tier A, code queries go to Tier B


if __name__ == "__main__":
    main()
    # Uncomment to see two-tier demo:
    # demo_two_tier_system()
