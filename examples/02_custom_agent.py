"""Custom agent example.

This example shows how to create specialized agents for different domains
and use them together in a hub.
"""

import anthropic

from agenthub import AgentHub, AgentSpec, BaseAgent


class PricingAgent(BaseAgent):
    """Agent specialized in pricing and margin analysis.

    This is an example of a Tier A (business/domain) agent that has
    specialized knowledge about pricing strategies.
    """

    def __init__(self, client: anthropic.Anthropic):
        spec = AgentSpec(
            agent_id="pricing_agent",
            name="Pricing Expert",
            description="Expert on pricing strategies, margins, and discount policies",
            context_keywords=[
                "price",
                "pricing",
                "margin",
                "discount",
                "cost",
                "revenue",
                "profit",
            ],
            system_prompt="""You are a pricing strategy expert.
You understand:
- Margin calculations and optimization
- Discount strategies and their impact
- Competitive pricing analysis
- Revenue optimization

When answering:
- Be specific about calculations
- Consider both short-term and long-term impacts
- Suggest data-driven approaches""",
        )
        super().__init__(spec, client)

    def build_context(self) -> str:
        """Build context with pricing knowledge."""
        return """## Pricing Guidelines

### Margin Targets
- Standard products: 25-35% gross margin
- Premium products: 40-50% gross margin
- Volume discounts: Max 15% off standard price

### Discount Policies
- First-time customer: Up to 10% off
- Bulk orders (>100 units): 5-15% tiered discount
- Seasonal sales: Up to 20% off select items

### Competitive Analysis
- Monitor competitor pricing weekly
- Match price within 5% for commodity items
- Maintain premium pricing for differentiated products

### Key Metrics
- Customer Acquisition Cost (CAC): ~$50
- Customer Lifetime Value (CLV): ~$500
- Target CLV/CAC ratio: 10:1
"""


class InventoryAgent(BaseAgent):
    """Agent specialized in inventory management."""

    def __init__(self, client: anthropic.Anthropic):
        spec = AgentSpec(
            agent_id="inventory_agent",
            name="Inventory Expert",
            description="Expert on inventory management, stock levels, and supply chain",
            context_keywords=[
                "inventory",
                "stock",
                "supply",
                "warehouse",
                "order",
                "restock",
                "SKU",
            ],
            system_prompt="""You are an inventory management expert.
You understand:
- Stock level optimization
- Reorder points and safety stock
- Supply chain logistics
- Warehouse operations

When answering:
- Consider lead times and demand variability
- Balance carrying costs vs stockout risks
- Suggest inventory optimization strategies""",
        )
        super().__init__(spec, client)

    def build_context(self) -> str:
        """Build context with inventory knowledge."""
        return """## Inventory Management Guidelines

### Stock Levels
- Safety stock: 2 weeks of average demand
- Reorder point: Safety stock + lead time demand
- Maximum stock: 6 weeks of demand

### Key SKUs
| Category | Lead Time | Min Stock | Reorder Point |
|----------|-----------|-----------|---------------|
| Electronics | 14 days | 100 units | 200 units |
| Apparel | 21 days | 150 units | 300 units |
| Home goods | 7 days | 50 units | 100 units |

### Warehouse Zones
- Zone A: Fast movers (80% of picks)
- Zone B: Medium movers
- Zone C: Slow movers

### KPIs
- Inventory turnover target: 8x/year
- Stockout rate: <2%
- Order accuracy: >99.5%
"""


def main():
    # Initialize
    client = anthropic.Anthropic()
    hub = AgentHub(client=client)

    # Register multiple specialized agents
    hub.register(PricingAgent(client))
    hub.register(InventoryAgent(client))

    print("Registered agents:")
    for agent in hub.list_agents():
        print(f"  - {agent.agent_id}: {agent.description}")

    # Queries automatically route to the right agent
    print("\n" + "=" * 60)

    # This should route to pricing_agent
    query1 = "What's our target margin for premium products?"
    print(f"\nQuery: {query1}")
    response1 = hub.run(query1)
    print(f"Routed to: {response1.agent_id}")
    print(f"Response: {response1.content[:200]}...")

    print("\n" + "-" * 60)

    # This should route to inventory_agent
    query2 = "What's the reorder point for electronics?"
    print(f"\nQuery: {query2}")
    response2 = hub.run(query2)
    print(f"Routed to: {response2.agent_id}")
    print(f"Response: {response2.content[:200]}...")

    print("\n" + "-" * 60)

    # Force a specific agent regardless of routing
    query3 = "How do stock levels affect our pricing?"
    print(f"\nQuery: {query3}")
    print("(Forcing pricing_agent)")
    response3 = hub.run(query3, agent_id="pricing_agent")
    print(f"Routed to: {response3.agent_id}")
    print(f"Response: {response3.content[:200]}...")


if __name__ == "__main__":
    main()
