from ai_agent.services.supabase_client import supabase
from pydantic import BaseModel
from typing import Optional, Dict, Union

class AggregateParams(BaseModel):
    metric: str
    column: str
    group_by: Optional[str] = None
    filters: Optional[Dict[str, Union[str, Dict[str, Union[int, float]]]]] = {}

def aggregate_used_car_data(metric: str, column: str, group_by: str = None, filters: dict = None):
    """
    Performs aggregation using Supabase + Python
    Supports advanced filters: lt, gt, lte, gte, eq, neq
    """

    filters = filters or {}

    allowed_columns = {
        "car_name","year","selling_price","present_price",
        "kms_driven","fuel_type","seller_type",
        "transmission","owner"
    }

    # Validate column
    if column not in allowed_columns:
        raise ValueError(f"Invalid column: {column}")

    if group_by and group_by not in allowed_columns:
        raise ValueError(f"Invalid group_by column: {group_by}")

    # Step 1: Build query (select only needed columns)
    select_cols = [column]
    if group_by:
        select_cols.append(group_by)

    query = supabase.table("Car_dataset").select(",".join(select_cols))

    # Step 2: Apply filters (UPDATED - case insensitive)
    for key, value in filters.items():
        if key not in allowed_columns:
            raise ValueError(f"Invalid filter column: {key}")

        if isinstance(value, dict):
            for op, v in value.items():
                if op == "lt":
                    query = query.lt(key, v)
                elif op == "gt":
                    query = query.gt(key, v)
                elif op == "lte":
                    query = query.lte(key, v)
                elif op == "gte":
                    query = query.gte(key, v)
                elif op == "eq":
                    if isinstance(v, str):
                        query = query.ilike(key, v.strip())
                    else:
                        query = query.eq(key, v)

                elif op == "neq":
                    if isinstance(v, str):
                        query = query.not_.ilike(key, v.strip())
                    else:
                        query = query.neq(key, v)

                else:
                    raise ValueError(f"Unsupported operator: {op}")

        else:
            if isinstance(value, str):
                query = query.ilike(key, value.strip())
            else:
                query = query.eq(key, value)

                
    # Execute query
    response = query.execute()
    data = response.data

    if not data:
        return {"results": []}

    # Step 3: Aggregate
    results = {}

    for row in data:
        group_key = row[group_by] if group_by else "all"

        if group_key not in results:
            results[group_key] = []

        value = row.get(column)
        if isinstance(value, (int, float)):
            results[group_key].append(value)

    # Step 4: Compute metric
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

        row = {
            column: round(agg_value, 2) if isinstance(agg_value, float) else agg_value
        }

        if group_by:
            row[group_by] = group

        output.append(row)

    return {"results": output}