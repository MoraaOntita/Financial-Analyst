# ai_agent/mcp_server/server.py
from fastmcp import FastMCP
from ai_agent.tool_handlers.query import query_used_cars_data, QueryUsedCarsParams
from ai_agent.tool_handlers.aggregation import aggregate_used_car_data, AggregateParams
from ai_agent.tool_handlers.web_search import search_web

mcp = FastMCP(name="used-car-analyst-agent")

# -------------------------------
# Tools
# -------------------------------

@mcp.tool(name="query_used_cars_data")
def query_used_cars_data_tool(params: QueryUsedCarsParams):
    # Unpack params to call the function
    return query_used_cars_data(
        columns=params.columns,
        distinct=params.distinct,
        filters=params.filters,
        limit=params.limit
    )

@mcp.tool(name="aggregate_used_car_data")
def aggregate_used_car_data_tool(params: AggregateParams):
    return aggregate_used_car_data(
        metric=params.metric,
        column=params.column,
        group_by=params.group_by,
        filters=params.filters
    )

@mcp.tool(name="search_web")
def search_web_tool(query: str):
    return search_web(query)

# -------------------------------
# Run server over HTTP
# -------------------------------
if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)