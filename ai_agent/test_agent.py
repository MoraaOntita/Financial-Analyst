import asyncio
from ai_agent.agent import agent_query

async def main():
    user_inputs = [
        "Show me the average selling price by fuel type",
        "What are current used car market trends in India 2026",
        "List some cars in the dataset"
    ]

    for query in user_inputs:
        print(f"\nUser Query: {query}")
        result = await agent_query(query)
        print("Agent Result:", result)

if __name__ == "__main__":
    asyncio.run(main())