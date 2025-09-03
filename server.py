import os, json
import httpx
from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_http_headers
import asyncio
from typing import Any, Dict, List, Optional

# Point this at your platform
# PLATFORM_API_BASE = os.environ.get("PLATFORM_API_BASE", "https://api.mydomain.com")


mcp = FastMCP(name="Personize", instructions="""
Public remote MCP server for tools like research_company_with_tavily, memorizer and Recall.
Supply Personize API key as a header: 
- X-Platform-API-Key: <key>  (preferred) 
or 
- Authorization: Bearer <key>
""")

def _extract_api_key() -> str | None:
    # Works for HTTP transport: read current request headers
    # (FastMCP provides helpers for this)
    # https://gofastmcp.com/servers/context  (HTTP Headers & Access Tokens)
    headers = get_http_headers()
    if not headers:
        return None
    h = headers.get("x-platform-api-key") or headers.get("X-Platform-API-Key")
    if h:
        return h.strip()
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None

# @mcp.tool
# async def memorize(key: str, value: str, ttl_seconds: int | None = None, ctx: Context | None = None) -> dict:
#     """Store a value under a key in our platform."""
#     api_key = _extract_api_key()
#     if not api_key:
#         raise ValueError("Missing API key. Pass X-Platform-API-Key or Authorization: Bearer <key>.")
#     async with httpx.AsyncClient(timeout=15) as client:
#         resp = await client.post(
#             f"{PLATFORM_API_BASE}/v1/memory",
#             json={"key": key, "value": value, "ttl_seconds": ttl_seconds},
#             headers={"Authorization": f"Bearer {api_key}"},
#         )
#         resp.raise_for_status()
#         return resp.json()

# @mcp.tool
# async def recall(key: str, ctx: Context | None = None) -> dict:
#     """Retrieve a value previously stored under a key in our platform."""
#     api_key = _extract_api_key()
#     if not api_key:
#         raise ValueError("Missing API key. Pass X-Platform-API-Key or Authorization: Bearer <key>.")
#     async with httpx.AsyncClient(timeout=15) as client:
#         resp = await client.get(
#             f"{PLATFORM_API_BASE}/v1/memory/{key}",
#             headers={"Authorization": f"Bearer {api_key}"},
#         )
#         resp.raise_for_status()
#         return resp.json()

@mcp.tool(description="Query a company's site via Tavily: Products, Services, Team, News, Key customers, FAQ, Pricing")
async def research_company_with_tavily(company_name: str, website_url: str, ctx: Context | None = None) -> dict:
    """
    Calls Tavily API to search within the given website for:
    Products, Services, Team, News, Key customers, FAQ, Pricing
    Returns Tavily's JSON response.
    """
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    TAVILY_SEARCH_URL = "https://api.tavily.com/search"
    if not TAVILY_API_KEY:
        raise ValueError("Missing TAVILY_API_KEY environment variable on the server.")

    # Build the query exactly as requested
    query = (
        f"Products, Services, Team, News, Key customers, FAQ, Pricing of {company_name} "
        f"- site:{website_url}"
    )

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        # Interpret “first 5 page” as top 5 results from Tavily
        "max_results": 5,
        # Use deeper crawling/quality for company research
        "search_depth": "advanced",
        # Keep the search restricted to the provided site as an extra guard
        "include_domains": [website_url],
        # Helpful structured answer + sources in the response
        "include_answer": True,
        # Keep payload small unless you specifically want full page text
        "include_raw_content": False,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(TAVILY_SEARCH_URL, json=payload)
        # Raise for non-2xx so the model surfaces a clear error message
        resp.raise_for_status()
        data = resp.json()

    # Optional: log a small, non-sensitive breadcrumb
    if ctx:
        await ctx.info(f"Tavily search for '{company_name}' on {website_url} returned {len(data.get('results', []))} results.")

    return data



# ---- Pinecone search tool ----

@mcp.tool(description="Search Pinecone text records within a user's namespace.")
async def pinecone_search(UserId: str, query: str, top_k: int | None = None, ctx: Context | None = None) -> dict:
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    index_host = os.environ.get("PINECONE_INDEX_HOST")  # optional

    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY in environment.")
    if not index_name:
        raise ValueError("Missing PINECONE_INDEX_NAME in environment.")
    if not UserId or not UserId.strip():
        raise ValueError("UserId is required.")
    if not query or not query.strip():
        raise ValueError("query is required.")
    k = int(top_k or 5)

    try:
        from pinecone import Pinecone
    except Exception as e:
        raise RuntimeError("Pinecone SDK not available. Ensure 'pinecone' is installed and redeploy.") from e

    fields_wanted = ['text', 'Tag', 'Type', 'RecordId', 'UserId', 'TimeStamp']
    query_payload = {"inputs": {"text": query}, "top_k": k}

    def _search_sync() -> Any:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name, host=index_host) if index_host else pc.Index(index_name)
        return index.search(namespace=UserId.strip(), query=query_payload, fields=fields_wanted)

    try:
        result_obj = await asyncio.to_thread(_search_sync)
    except Exception as e:
        return {"success": False, "error": "pinecone_search_failed", "message": str(e)}

    # ---------- Coerce to plain JSON safely ----------
    # Prefer pydantic v2 json; fall back to dict() / best-effort
    plain: Dict[str, Any] | None = None
    try:
        if hasattr(result_obj, "model_dump_json"):
            plain = json.loads(result_obj.model_dump_json())         # pydantic v2 safe JSON
        elif hasattr(result_obj, "model_dump"):
            plain = result_obj.model_dump(mode="json")               # pydantic v2 dict
        elif hasattr(result_obj, "to_dict"):
            plain = result_obj.to_dict()                             # some SDKs
        elif hasattr(result_obj, "dict"):
            plain = result_obj.dict()                                # pydantic v1
        elif isinstance(result_obj, dict):
            plain = result_obj
    except Exception:
        plain = None

    # If still not plain (to avoid circular refs), build a minimal safe payload
    if plain is None:
        plain = {}

    # ---------- Build safe, pruned payload ----------
    # Expect shape like plain['result']['hits'][i]['fields']...
    result_section = (plain.get("result", {}) if isinstance(plain, dict) else {}) or {}
    hits = result_section.get("hits", []) if isinstance(result_section, dict) else []

    matches: List[str] = []
    safe_hits: List[Dict[str, Any]] = []
    try:
        for i, h in enumerate(hits):
            # h is expected to be a dict
            fields = h.get("fields", {}) if isinstance(h, dict) else {}
            text_val = fields.get("text")
            if isinstance(text_val, str):
                matches.append(text_val)

            # prune to JSON-safe subset (score + requested fields only)
            safe_fields = {k: v for k, v in fields.items() if k in fields_wanted}
            safe_hits.append({
                "rank": i + 1,
                "score": h.get("score"),
                "fields": safe_fields
            })
    except Exception:
        # if parsing fails, leave matches/safe_hits as-is
        pass

    # Minimal raw that’s guaranteed JSON-safe (no circular refs)
    safe_raw = {
        "count": len(safe_hits),
        "hits": safe_hits
    }

    if ctx:
        await ctx.info(f"Pinecone: {len(matches)} matches (namespace={UserId.strip()}, top_k={k}).")

    return {"success": True, "matches": matches, "raw": safe_raw}

# if __name__ == "__main__":
#     # For local testing only (remote hosting uses Cloud)
#     mcp.run(transport="http", host="127.0.0.1", port=8000)
