"""Integration test: verify tool-use loop fires for an agent with domain tools.

Bypasses routing — sends a question directly to an agent known to have
context_paths (i.e. tools available) and checks the response metadata.
"""

import os
import sys
import time

# Ensure the project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/Documents/Smartstore/.env"), override=True)

from agenthub.auto import discover_all_agents

SMARTSTORE_ROOT = os.path.expanduser("~/Documents/Smartstore")


def main():
    # --- Setup ---
    print("Setting up hub with discover_all_agents...")
    hub, summary = discover_all_agents(SMARTSTORE_ROOT)
    print(summary)

    # Show available agents and their tool status
    print("\n=== Agent inventory ===")
    agents_with_tools = []
    for agent_id, agent in hub._agents.items():
        has_paths = len(agent.spec.context_paths) > 0
        has_root = getattr(agent, "root_path", None) or agent.spec.metadata.get("root_path")
        has_tools = has_paths and has_root is not None
        n_paths = len(agent.spec.context_paths)
        if has_tools:
            agents_with_tools.append((agent_id, n_paths))
        print(f"  {agent_id}: tools={has_tools}, paths={n_paths}")

    if not agents_with_tools:
        print("\nERROR: No agents have tools enabled!")
        return

    # Pick the agent with the most context_paths (best chance of answering)
    target_agent_id = max(agents_with_tools, key=lambda x: x[1])[0]
    print(f"\n=== Testing agent: {target_agent_id} (paths={dict(agents_with_tools)[target_agent_id]}) ===")

    target_agent = hub._agents[target_agent_id]

    # Show a few context_paths so we can craft a good question
    print(f"\nSample context_paths:")
    for p in target_agent.spec.context_paths[:10]:
        print(f"  {p}")
    print(f"  ... ({len(target_agent.spec.context_paths)} total)")

    # Verify tools are actually created
    tools_result = target_agent._get_domain_tools()
    if tools_result is None:
        print("\nERROR: _get_domain_tools() returned None!")
        return
    tool_defs, executor = tools_result
    print(f"\nTools created: {[t['name'] for t in tool_defs]}")
    print(f"Allowed dirs: {[str(d) for d in executor.scope.allowed_dirs]}")

    # --- Run a query that requires tool use ---
    # Ask about something that's likely NOT in the pre-loaded context
    # (pre-loaded context is capped at 20K when tools active)
    from agenthub.models import Session
    session = Session(session_id="test-tools", agent_id=target_agent_id)

    # Query must contain at least one of the agent's keywords to pass
    # heuristic scope check. get_backend_agent keywords: get, backend, app, product, tenant
    query = (
        "How does the backend handle product creation? "
        "Search through the app files and show me the relevant API endpoints and models."
    )

    print(f"\nQuery: {query}")
    print("Running agent (with tools)...")
    start = time.time()

    response = target_agent.run(query, session)

    elapsed = time.time() - start
    print(f"\n=== Response ===")
    print(f"Agent: {response.agent_id}")
    print(f"Tokens: {response.tokens_used}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Metadata: {response.metadata}")
    print(f"Used tools: {response.metadata.get('used_tools', False)}")
    print(f"\nAnswer length: {len(response.content)} chars")
    print(f"Answer (first 2000 chars):")
    print(repr(response.content[:200]))
    print("---")
    print(response.content.strip()[:2000])

    if response.metadata.get("scope_rejected"):
        print("\nWARNING: Agent rejected query at scope check stage. Tools were NOT exercised.")
    elif response.metadata.get("used_tools"):
        print("\nSUCCESS: Tool-use path was taken!")
    else:
        print("\nINFO: Single-shot path was taken (no tools).")


if __name__ == "__main__":
    main()
