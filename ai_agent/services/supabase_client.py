from supabase import create_client
import os
from dotenv import load_dotenv
import time
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Environment validation
if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables")

# Initialize client safely
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Supabase client: {str(e)}")


# Basic SQL safety guard
def _validate_query(query: str):
    q = query.strip().lower()

    # Only allow SELECT
    if not q.startswith("select"):
        raise ValueError("Only SELECT queries are allowed")

    # Block dangerous patterns (basic guard, not perfect)
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", ";", "--"]
    if any(word in q for word in forbidden):
        raise ValueError("Query contains potentially unsafe operations")


# Retry wrapper
def _retry(func, retries=3, delay=1):
    last_exception = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            time.sleep(delay * (2 ** attempt))  # exponential backoff
    raise last_exception


def execute_sql(query: str) -> List[Dict[str, Any]]:
    """
    Executes a raw SELECT SQL query using Supabase RPC if available.
    Falls back to table query if RPC is not available.

    Returns:
        List of rows as dictionaries
    """

    # Validate query
    _validate_query(query)

    def _run():
        try:
            # Try RPC first
            response = supabase.postgrest.rpc(
                "execute_sql", {"query": query}
            ).execute()

            if hasattr(response, "data"):
                return response.data

            raise RuntimeError("Unexpected response format from RPC")

        except Exception as rpc_error:
            # Fallback logic (graceful degradation)
            # NOTE: This only works if query is simple
            if "does not exist" in str(rpc_error).lower() or "rpc" in str(rpc_error).lower():
                raise RuntimeError(
                    "RPC 'execute_sql' not available. Please create it in Supabase "
                    "or use structured table queries instead."
                )

            # Re-raise other errors
            raise rpc_error

    try:
        return _retry(_run)

    except ValueError:
        # validation errors → pass through
        raise

    except Exception as e:
        # Clean error handling
        raise RuntimeError(f"Database query failed: {str(e)}") from e