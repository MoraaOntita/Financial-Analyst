from ai_agent.services.supabase_client import supabase

from pydantic import BaseModel
from typing import Optional, Dict

class AggregateParams(BaseModel):
    metric: str
    column: str
    group_by: Optional[str] = None
    filters: Optional[Dict] = {}

def aggregate_used_car_data(metric: str, column: str, group_by: str = None, filters: dict = None):
    """
    Performs aggregation using Supabase + Python
    """

    # Step 1: Fetch data
    query = supabase.table("Car_dataset").select("*")

    # Apply filters if provided
    if filters:
        for key, value in filters.items():
            query = query.eq(key, value)

    response = query.execute()
    data = response.data

    if not data:
        return {"results": []}

    # Step 2: Aggregation
    results = {}

    for row in data:
        group_key = row[group_by] if group_by else "all"

        if group_key not in results:
            results[group_key] = []

        value = row.get(column)
        if value is not None:
            results[group_key].append(value)

    # Step 3: Compute metric
    output = []

    for group, values in results.items():
        if not values:
            continue

        if metric == "avg":
            agg_value = sum(values) / len(values)
        elif metric == "min":
            agg_value = min(values)
        elif metric == "max":
            agg_value = max(values)
        elif metric == "count":
            agg_value = len(values)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        row = {column: round(agg_value, 2) if isinstance(agg_value, float) else agg_value}

        if group_by:
            row[group_by] = group

        output.append(row)

    return {"results": output}