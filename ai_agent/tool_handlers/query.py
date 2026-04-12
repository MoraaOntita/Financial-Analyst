# ai_agent/tool_handlers/query.py
from ai_agent.services.supabase_client import supabase
from pydantic import BaseModel
from typing import List, Dict, Optional, Union

class QueryUsedCarsParams(BaseModel):
    columns: List[str]
    distinct: Optional[bool] = False
    filters: Optional[Dict[str, Union[str, Dict[str, Union[int, float]]]]] = {}
    limit: Optional[int] = 50

def query_used_cars_data(columns, distinct=False, filters=None, limit=50):
    """
    Flexible query for Car_dataset table.
    - columns: list of columns to select
    - distinct: whether to select distinct rows
    - filters: dict of filters {column_name: value or {"lt": val, "gt": val}}
    - limit: number of rows to return
    """
    filters = filters or {}

    # Validate table columns
    allowed_columns = {"car_name","year","selling_price","present_price","kms_driven",
                       "fuel_type","seller_type","transmission","owner"}
    for col in columns:
        if col not in allowed_columns:
            raise ValueError(f"Invalid column: {col}")

    # Build Supabase query
    table = supabase.table("Car_dataset")

    # Start select
    select_cols = ",".join(columns)
    if distinct:
        select_cols = f"distinct {select_cols}"
    query_builder = table.select(select_cols)

    # Apply filters
    for col, val in filters.items():
        if isinstance(val, dict):
            for op, v in val.items():
                if op == "lt":
                    query_builder = query_builder.lt(col, v)
                elif op == "gt":
                    query_builder = query_builder.gt(col, v)
                elif op == "lte":
                    query_builder = query_builder.lte(col, v)
                elif op == "gte":
                    query_builder = query_builder.gte(col, v)
                elif op == "eq":
                    # case-insensitive match for strings
                    if isinstance(v, str):
                        query_builder = query_builder.ilike(col, v.strip())
                    else:
                        query_builder = query_builder.eq(col, v)
                elif op == "neq":
                    if isinstance(v, str):
                        query_builder = query_builder.not_.ilike(col, v.strip())
                    else:
                        query_builder = query_builder.neq(col, v)
        else:
            # case-insensitive match for strings
            if isinstance(val, str):
                query_builder = query_builder.ilike(col, val.strip())
            else:
                query_builder = query_builder.eq(col, val)

    # Apply limit
    query_builder = query_builder.limit(limit)

    # Execute
    response = query_builder.execute()
    data, error = response if isinstance(response, tuple) else (response.data, None)
    if error:
        raise Exception(f"Database query failed: {error}")

    return {"rows": data, "row_count": len(data) if data else 0}