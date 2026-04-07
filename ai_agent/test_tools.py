import asyncio
from fastmcp import Client
from prettytable import PrettyTable

async def main():
    async with Client("http://127.0.0.1:8000/mcp") as client:

        # -------------------------
        # 1️⃣ Query Tool
        # -------------------------
        sql = 'SELECT "Car_Name", "Selling_Price" FROM "Car_dataset" LIMIT 5;'
        query_result = await client.call_tool("query_used_cars_data", {"sql_query": sql})
        query_data = query_result.data

        rows = query_data.get("rows", [])
        if rows:
            table = PrettyTable()
            table.field_names = rows[0].keys()
            for row in rows:
                table.add_row(list(row.values()))

            print("\nQuery Tool Result:")
            print(table)
        else:
            print("No rows returned.")

        # -------------------------
        # 2️⃣ Aggregation Tool
        # -------------------------
        agg_result = await client.call_tool(
            "aggregate_used_car_data",
            {
                "metric": "avg",
                "column": "Selling_Price",
                "group_by": "Fuel_Type",
                "filters": {"Transmission": "Manual"}
            }
        )
        agg_data = agg_result.data

        results = agg_data.get("results", [])
        if results:
            table = PrettyTable()
            table.field_names = results[0].keys()
            for row in results:
                table.add_row(list(row.values()))

            print("\nAggregation Tool Result:")
            print(table)
        else:
            print("No aggregation results.")

        # -------------------------
        # 3️⃣ Web Search Tool
        # -------------------------
        search_result = await client.call_tool(
            "search_web",
            {"query": "current used car market trends India 2026"}
        )
        search_data = search_result.data

        print("\nWeb Search Tool Result:")
        for idx, item in enumerate(search_data.get("results", []), 1):
            print(f"{idx}. {item}")

if __name__ == "__main__":
    asyncio.run(main())